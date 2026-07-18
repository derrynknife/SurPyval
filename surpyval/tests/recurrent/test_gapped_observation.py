"""Gapped (multi-window) observation for recurrent events.

An item may be observed only over several disjoint time windows, with
unobserved gaps in between (events may occur during a gap but are never
recorded). For an NHPP the event counts over disjoint windows are independent,
so a gapped item's likelihood factorises over its windows; ``handle_xicn``
expands each window into a synthetic single-window sub-item (its events plus a
right-censored close at the window end, entering at the window start). These
tests check that expansion, that it reproduces the exact gapped likelihood and
recovers parameters, that the nonparametric MCF at-risk set respects the gaps,
that the virtual-age / renewal models reject gapped data, and that malformed
window specifications are rejected.
"""

import numpy as np
import pytest

from surpyval import Weibull, handle_xicn
from surpyval.recurrent import (
    ARA,
    ARI,
    CrowAMSAA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
    NonParametricCounting,
)

# --- expansion into synthetic single-window sub-items -----------------------


def test_windows_expand_to_synthetic_items_with_closes():
    x = np.array([3, 7, 10, 25, 33, 38])
    i = np.array([1, 1, 1, 1, 1, 1])
    data = handle_xicn(x, i, windows={1: [(0, 10), (20, 40)]})

    # two windows -> two synthetic items, each ending in a c=1 close row
    assert set(np.unique(data.i)) == {1, 2}
    assert data.window_map == {1: (1, (0.0, 10.0)), 2: (1, (20.0, 40.0))}
    # window 1: events 3, 7, 10 then a close at 10; entry (tl) 0
    w1 = data.i == 1
    assert list(data.x[w1]) == [3.0, 7.0, 10.0, 10.0]
    assert list(data.c[w1]) == [0.0, 0.0, 0.0, 1.0]
    assert np.all(data.tl[w1] == 0.0)
    # window 2: events 25, 33, 38 then a close at 40; entry (tl) 20
    w2 = data.i == 2
    assert list(data.x[w2]) == [25.0, 33.0, 38.0, 40.0]
    assert list(data.c[w2]) == [0.0, 0.0, 0.0, 1.0]
    assert np.all(data.tl[w2] == 20.0)


def test_empty_window_still_contributes_a_close():
    # A window with no events is still an observation period: it becomes a
    # lone right-censored close row spanning the window.
    data = handle_xicn(
        np.array([3.0, 7.0]),
        np.array([1, 1]),
        windows={1: [(0, 10), (20, 40)]},
    )
    w2 = data.i == 2
    assert list(data.x[w2]) == [40.0]
    assert list(data.c[w2]) == [1.0]
    assert np.all(data.tl[w2] == 20.0)


def test_window_map_defaults_to_none_for_ordinary_data():
    data = handle_xicn(np.array([1.0, 2.0, 3.0]), np.array([1, 1, 1]))
    assert data.window_map is None
    assert data.observation_windows is None


# --- the gapped likelihood matches the hand-built synthetic representation --


def test_gapped_likelihood_matches_manual_window_items():
    x = np.array([3, 7, 10, 25, 33, 38])
    i = np.array([1, 1, 1, 1, 1, 1])
    gapped = handle_xicn(x, i, windows={1: [(0, 10), (20, 40)]})

    # The same thing spelled out by hand as two left-truncated single-window
    # items, each closed by an explicit right-censoring row.
    manual = handle_xicn(
        np.array([3, 7, 10, 25, 33, 38, 40]),
        np.array([1, 1, 1, 2, 2, 2, 2]),
        c=np.array([0, 0, 0, 0, 0, 0, 1]),
        tl=np.array([0, 0, 0, 20, 20, 20, 20]),
    )
    f_gapped = CrowAMSAA.create_negll_func(gapped)
    f_manual = CrowAMSAA.create_negll_func(manual)
    for p in ([1.4, 0.9], [4.0, 1.2], [2.0, 1.0]):
        assert np.isclose(f_gapped(np.array(p)), f_manual(np.array(p)))


# --- parameter recovery from a gapped NHPP simulation -----------------------


def test_nhpp_recovers_parameters_from_gapped_data():
    truth = CrowAMSAA.from_params([5.0, 1.4])
    full = truth.time_terminated_simulation_data(30.0, items=400, seed=7)

    # Impose a common unobserved gap (12, 20): each item is observed on
    # [0, 12] and [20, 30], and events inside the gap are dropped.
    gap, T = (12.0, 20.0), 30.0
    xs, ii, windows = [], [], {}
    for item in np.unique(full.i):
        m = (full.i == item) & (full.c == 0)
        ev = full.x[m]
        ev = ev[(ev <= gap[0]) | (ev >= gap[1])]
        xs.append(ev)
        ii.append(np.full(ev.size, item))
        windows[item] = [(0.0, gap[0]), (gap[1], T)]
    x = np.concatenate(xs)
    i = np.concatenate(ii)

    model = CrowAMSAA.fit(x, i, windows=windows)
    assert np.allclose(model.params, [5.0, 1.4], rtol=0.15)


# --- nonparametric MCF at-risk set respects the gaps ------------------------


def test_mcf_at_risk_set_excludes_item_during_its_gap():
    # Item A observed continuously on [0, 50]; item B observed on [0, 10] and
    # [30, 50] with a gap. At a time inside B's gap only A is at risk.
    x = np.array([5.0, 20.0, 40.0, 5.0, 35.0])
    i = np.array(["A", "A", "A", "B", "B"])
    windows = {"A": [(0.0, 50.0)], "B": [(0.0, 10.0), (30.0, 50.0)]}
    mcf = NonParametricCounting.fit(x, i, windows=windows)

    r_at = dict(zip(mcf.x, mcf.r))
    # x = 5 is in a window of both A and B -> risk set 2
    assert r_at[5.0] == 2
    # x = 20 falls in B's gap -> only A at risk
    assert r_at[20.0] == 1
    # x = 35 is back in B's second window and still in A's window -> 2
    assert r_at[35.0] == 2


# --- the virtual-age / renewal models reject gapped observation -------------


@pytest.mark.parametrize(
    "fit_call",
    [
        lambda d: GeneralizedRenewal.fit_from_recurrent_data(
            d, dist=Weibull, kijima="i"
        ),
        lambda d: GeneralizedOneRenewal.fit_from_recurrent_data(
            d, dist=Weibull
        ),
        lambda d: ARA.fit_from_recurrent_data(d, dist=Weibull, m=1),
        lambda d: ARI.fit_from_recurrent_data(d, dist=CrowAMSAA, m=1),
    ],
)
def test_renewal_models_reject_gapped_data(fit_call):
    # A single window starting at 0 still carries a window_map, so the guard
    # fires regardless of the left-truncation check.
    data = handle_xicn(
        np.array([3.0, 7.0, 10.0]), np.array([1, 1, 1]), windows={1: [(0, 15)]}
    )
    with pytest.raises(ValueError, match="gapped"):
        fit_call(data)


# --- malformed window specifications are rejected ---------------------------


def test_windows_require_all_rows_observed():
    with pytest.raises(ValueError, match="observed event"):
        handle_xicn(
            np.array([3.0, 7.0]),
            np.array([1, 1]),
            c=np.array([0, 1]),
            windows={1: [(0, 10)]},
        )


def test_windows_must_cover_every_item():
    with pytest.raises(ValueError, match="every item"):
        handle_xicn(
            np.array([3.0, 7.0]),
            np.array([1, 2]),
            windows={1: [(0, 10)]},
        )


def test_windows_reject_overlap():
    with pytest.raises(ValueError, match="overlapping"):
        handle_xicn(
            np.array([3.0]), np.array([1]), windows={1: [(0, 10), (5, 20)]}
        )


def test_windows_reject_reversed_or_empty():
    with pytest.raises(ValueError, match="reversed"):
        handle_xicn(np.array([3.0]), np.array([1]), windows={1: [(10, 5)]})
    with pytest.raises(ValueError, match="reversed"):
        handle_xicn(np.array([3.0]), np.array([1]), windows={1: [(5, 5)]})


def test_windows_reject_non_finite():
    with pytest.raises(ValueError, match="non-finite"):
        handle_xicn(np.array([3.0]), np.array([1]), windows={1: [(0, np.inf)]})


def test_events_outside_all_windows_rejected():
    with pytest.raises(ValueError, match="outside all"):
        handle_xicn(
            np.array([3.0, 15.0]),
            np.array([1, 1]),
            windows={1: [(0, 10), (20, 30)]},
        )


def test_windows_mutually_exclusive_with_truncation():
    with pytest.raises(ValueError, match="must not also"):
        handle_xicn(
            np.array([3.0]), np.array([1]), tl=0.0, windows={1: [(0, 10)]}
        )


def test_windows_reject_covariates():
    with pytest.raises(ValueError, match="does not support covariates"):
        handle_xicn(
            np.array([3.0]),
            np.array([1]),
            Z=np.array([[1.0]]),
            windows={1: [(0, 10)]},
        )


# --- HPP also accepts gapped observation ------------------------------------


def test_hpp_accepts_windows():
    from surpyval.recurrent import HPP

    # Homogeneous Poisson process of rate 2 observed on [0, 4] and [6, 10]
    # (an unobserved gap on (4, 6)). Simulate events directly in each window.
    rate = 2.0
    wins = [(0.0, 4.0), (6.0, 10.0)]
    rng = np.random.default_rng(3)
    xs, ii, windows = [], [], {}
    for item in range(300):
        for a, b in wins:
            k = rng.poisson(rate * (b - a))
            ev = np.sort(rng.uniform(a, b, size=k))
            xs.append(ev)
            ii.append(np.full(ev.size, item))
        windows[item] = wins
    model = HPP.fit(np.concatenate(xs), np.concatenate(ii), windows=windows)
    assert np.isclose(model.params[0], rate, rtol=0.1)
