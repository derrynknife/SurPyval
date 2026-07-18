"""Event marks (competing-risks recurrent events).

An item can experience events of several mutually exclusive types over time.
``handle_xicn`` now carries a per-row mark ``e``; the nonparametric
``CauseSpecificMCF`` routes through it (gaining validation, sorting and
truncation), and the parametric ``CauseSpecificNHPP`` fits a separate intensity
model per cause. For a marked Poisson process the cause-specific processes are
independent thinned Poisson processes, so each cause's intensity is the NHPP
fit to that cause's events over the full observation window -- these tests
check that mark plumbing, the routing, and independent parameter recovery.
"""

import numpy as np
import pytest

from surpyval import handle_xicn
from surpyval.recurrent import CrowAMSAA
from surpyval.recurrent.competing_risks import (
    CauseSpecificMCF,
    CauseSpecificNHPP,
)

# --- marks in handle_xicn ---------------------------------------------------


def test_handle_xicn_sorts_marks_with_events():
    x = np.array([3, 1, 2, 5])
    i = np.array([1, 1, 1, 1])
    c = np.array([0, 0, 0, 1])
    e = np.array(["B", "A", "A", None], dtype=object)
    data = handle_xicn(x, i, c, e=e)
    # marks travel with their event through the (i, x) sort
    assert list(data.x) == [1.0, 2.0, 3.0, 5.0]
    assert list(data.e) == ["A", "A", "B", None]
    assert data.event_types == ["A", "B"]


def test_handle_xicn_normalises_missing_marks_to_none():
    # NaN / pandas-NA style missing marks become the single None sentinel, so
    # they are not mistaken for a distinct cause.
    x = np.array([1.0, 2.0, 3.0])
    e = np.array(["A", np.nan, "A"], dtype=object)
    data = handle_xicn(x, np.array([1, 1, 1]), np.array([0, 0, 1]), e=e)
    assert data.event_types == ["A"]
    assert data.e[1] is None


def test_handle_xicn_marks_length_checked():
    with pytest.raises(ValueError, match="x and e must have the same length"):
        handle_xicn(np.array([1.0, 2.0]), e=np.array(["A"]))


def test_marks_rejected_with_windows():
    with pytest.raises(ValueError, match="does not support event-type"):
        handle_xicn(
            np.array([3.0]),
            np.array([1]),
            e=np.array(["A"]),
            windows={1: [(0, 10)]},
        )


# --- CauseSpecificMCF routed through handle_xicn ----------------------------


def test_cause_specific_mcf_counts_by_cause():
    x = [3, 1, 5, 2, 4, 6]
    i = [1, 1, 1, 2, 2, 2]
    c = [0, 0, 1, 0, 0, 1]
    e = ["A", "B", None, "A", "A", None]
    model = CauseSpecificMCF.fit(x, i, c, e=e)
    assert model.event_types == ["A", "B"]
    # three A events over two items -> MCF reaches 1.5; one B event -> 0.5
    assert np.isclose(model.mcf(6, "A")[0], 1.5)
    assert np.isclose(model.mcf(6, "B")[0], 0.5)


def test_cause_specific_mcf_requires_marks():
    with pytest.raises(ValueError, match="required for a cause-specific"):
        CauseSpecificMCF.fit([1, 2, 3], [1, 1, 1], [0, 0, 1])


def test_cause_specific_mcf_validates_via_handler():
    # Routing through handle_xicn means malformed input is now caught (here a
    # right-censoring row that is not an item's last row).
    with pytest.raises(ValueError, match="right censored"):
        CauseSpecificMCF.fit(
            [1, 2, 3],
            [1, 1, 1],
            c=[1, 0, 0],
            e=["A", "A", None],
        )


def test_cause_specific_mcf_from_df():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "t": [3, 1, 5, 2, 4, 6],
            "item": [1, 1, 1, 2, 2, 2],
            "cens": [0, 0, 1, 0, 0, 1],
            "mark": ["A", "B", np.nan, "A", "A", np.nan],
        }
    )
    model = CauseSpecificMCF.fit_from_df(
        df, "t", "mark", i_col="item", c_col="cens"
    )
    assert model.event_types == ["A", "B"]
    assert np.isclose(model.mcf(6, "A")[0], 1.5)
    assert model.df is df


# --- CauseSpecificNHPP: independent per-cause intensity recovery -------------


def _simulate_two_cause_marks(paramsA, paramsB, T, items, seed):
    truthA = CrowAMSAA.from_params(paramsA)
    truthB = CrowAMSAA.from_params(paramsB)
    dA = truthA.time_terminated_simulation_data(T, items=items, seed=seed)
    dB = truthB.time_terminated_simulation_data(T, items=items, seed=seed + 1)
    xs, ii, cc, ee = [], [], [], []
    for item in range(items):
        for xv in dA.x[(dA.i == item) & (dA.c == 0)]:
            xs.append(xv)
            ii.append(item)
            cc.append(0)
            ee.append("A")
        for xv in dB.x[(dB.i == item) & (dB.c == 0)]:
            xs.append(xv)
            ii.append(item)
            cc.append(0)
            ee.append("B")
        xs.append(T)
        ii.append(item)
        cc.append(1)
        ee.append(None)
    return (
        np.array(xs),
        np.array(ii),
        np.array(cc),
        np.array(ee, dtype=object),
    )


def test_cause_specific_nhpp_recovers_independent_intensities():
    x, i, c, e = _simulate_two_cause_marks(
        [6.0, 1.3], [10.0, 0.8], T=25.0, items=500, seed=1
    )
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    assert model.event_types == ["A", "B"]
    assert np.allclose(model.models["A"].params, [6.0, 1.3], rtol=0.15)
    assert np.allclose(model.models["B"].params, [10.0, 0.8], rtol=0.15)


def test_cause_specific_nhpp_total_cif_is_sum_of_causes():
    x, i, c, e = _simulate_two_cause_marks(
        [6.0, 1.3], [10.0, 0.8], T=25.0, items=200, seed=3
    )
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    grid = np.array([5.0, 12.0, 25.0])
    total = model.total_cif(grid)
    parts = model.cif(grid, "A") + model.cif(grid, "B")
    assert np.allclose(total, parts)


def test_cause_specific_nhpp_per_cause_models_carry_inference():
    x, i, c, e = _simulate_two_cause_marks(
        [6.0, 1.3], [10.0, 0.8], T=25.0, items=100, seed=5
    )
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    # each per-cause model is a full ParametricRecurrenceModel
    a = model.models["A"]
    assert np.isfinite(a.aic)
    assert a.iif(10.0) > 0
    assert a.cif(25.0) > a.cif(5.0)


def test_cause_specific_nhpp_requires_marks():
    with pytest.raises(ValueError, match="required for a cause-specific"):
        CauseSpecificNHPP.fit([1, 2, 3], [1, 1, 1], [0, 0, 1])


def test_cause_specific_nhpp_rejects_interval_censoring():
    data = handle_xicn(
        np.array([[1.0, 2.0], [3.0, 3.0]]),
        np.array([1, 1]),
        c=np.array([2, 1]),
        n=np.array([1, 1]),
        e=np.array(["A", None], dtype=object),
    )
    with pytest.raises(ValueError, match="exact"):
        CauseSpecificNHPP.fit_from_recurrent_data(data)


def test_cause_specific_nhpp_item_with_no_events_of_a_cause():
    # Item 2 never fails from cause B, but its window still contributes to B's
    # compensator (so B's intensity is not overestimated). The fit must run and
    # give a smaller B intensity than a data set where both items fail from B.
    x = [2.0, 5.0, 8.0, 3.0, 6.0, 9.0]
    i = [1, 1, 1, 2, 2, 2]
    c = [0, 0, 1, 0, 0, 1]
    e = ["A", "B", None, "A", "A", None]  # only item 1 has a B event
    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    assert set(model.event_types) == {"A", "B"}
    # B seen once over two full windows -> positive but small intensity
    cif_b = float(np.asarray(model.cif(8.0, "B")))
    cif_a = float(np.asarray(model.cif(8.0, "A")))
    assert cif_b > 0
    assert cif_b < cif_a
