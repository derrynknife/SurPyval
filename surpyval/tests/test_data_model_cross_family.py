"""Data shaped for one model family, handed to another: each refuses in
words, naming what to use."""

import numpy as np
import pytest

from surpyval import SurpyvalData, WeibullJointPH
from surpyval.degradation import DegradationAnalysis
from surpyval.multistate import (
    MultiState,
    PanelMultiState,
    PanelMultiStateRegression,
)
from surpyval.recurrent import CrowAMSAA, ProportionalIntensityNHPP

TM = [("healthy", "ill"), ("healthy", "dead"), ("ill", "dead")]
READINGS = {u: {0.0: 1.0, 1.0: 2.0, 2.0: 3.0} for u in (1, 2, 3)}


def _stays(**kw):
    """Stays in states: units 1 and 2 have two rows each."""
    return SurpyvalData(
        [2, 6, 3, 9, 4],
        e=["ill", "dead", "ill", None, "dead"],
        s=["healthy", "ill", "healthy", "ill", "healthy"],
        tl=[0, 2, 0, 3, 0],
        i=[1, 1, 2, 2, 3],
        **kw,
    )


def _repeated(**kw):
    """Repeated events without states: unit 1 has three rows."""
    return SurpyvalData([1, 2, 3, 5], c=[0, 0, 0, 1], i=[1, 1, 1, 2], **kw)


# -- multi-state data ------------------------------------------------------


def test_states_refused_by_recurrent():
    with pytest.raises(ValueError, match="one state only") as info:
        CrowAMSAA.fit(_stays())
    assert "multi-state models" in str(info.value)


def test_states_refused_by_degradation():
    with pytest.raises(ValueError, match="one state only") as info:
        DegradationAnalysis.fit(_stays(m=READINGS), threshold=5.0)
    assert "multi-state models" in str(info.value)


def test_states_refused_by_joint():
    with pytest.raises(ValueError, match="one state only") as info:
        WeibullJointPH.fit(_stays(m=READINGS))
    assert "multi-state models" in str(info.value)


# -- several rows per unit, no states --------------------------------------


def test_several_rows_refused_by_multistate():
    data = SurpyvalData(
        [1, 2, 3, 5], c=[0, 0, 0, 1], e=["f", "f", "f", None], i=[1, 1, 1, 2]
    )
    with pytest.raises(ValueError, match="several rows") as info:
        MultiState.fit(data)
    assert "SurpyvalData(x, ..., s=s, e=e, i=i)" in str(info.value)


def test_several_rows_refused_by_degradation():
    data = _repeated(m={1: {0.0: 1.0, 1.0: 2.0}, 2: {0.0: 1.0, 2.0: 2.0}})
    with pytest.raises(ValueError, match="one row per unit") as info:
        DegradationAnalysis.fit(data, threshold=5.0)
    assert "recurrent models" in str(info.value)


def test_several_rows_refused_by_joint():
    data = _repeated(
        Z=np.ones((4, 1)),
        m={1: {0.0: 1.0, 1.0: 2.0}, 2: {0.0: 1.0, 2.0: 2.0}},
    )
    with pytest.raises(ValueError, match="one row per unit") as info:
        WeibullJointPH.fit(data)
    assert "recurrent models" in str(info.value)


# -- measurements -----------------------------------------------------------


def test_measurements_refused_by_multistate():
    with pytest.raises(ValueError, match="takes no markers") as info:
        MultiState.fit(_stays(m=READINGS), transitions=TM)
    assert "without" in str(info.value)


@pytest.mark.parametrize(
    "m",
    [
        {1: {0.0: 1.0}},
        {"vibration": {1: {0.0: 1.0}}, "temperature": {2: {0.0: 2.0}}},
    ],
    ids=["one", "several"],
)
def test_measurements_refused_by_recurrent(m):
    with pytest.raises(ValueError, match="takes no markers") as info:
        CrowAMSAA.fit(_repeated(m=m))
    assert "without" in str(info.value)


# -- the panel models take arrays of visits --------------------------------


@pytest.mark.parametrize("model", [PanelMultiState, PanelMultiStateRegression])
def test_panel_models_refuse_surpyval_data(model):
    data = SurpyvalData(
        [1, 2, 3, 1, 2],
        s=["ok", "ok", "worn", "ok", "worn"],
        e=[None, "worn", "failed", "worn", None],
        i=[1, 1, 1, 2, 2],
        Z=np.ones((5, 1)),
    )
    with pytest.raises(TypeError, match="arrays of visits") as info:
        model.fit(data, transitions=[("ok", "worn"), ("worn", "failed")])
    assert "MultiState.fit(data)" in str(info.value)


# -- ranges given with x, in a recurrent fit --------------------------------


def test_recurrent_range_rows_fit_as_the_arrays():
    nan = np.nan
    data = SurpyvalData(
        x=[1, 2, nan, 5],
        xl=[nan, nan, 2.5, nan],
        xr=[nan, nan, 3.5, nan],
        c=[0, 0, 2, 1],
        i=["A", "A", "A", "B"],
    )
    expected = CrowAMSAA.fit(
        np.array([[1, 1], [2, 2], [2.5, 3.5], [5, 5]]),
        i=["A", "A", "A", "B"],
        c=[0, 0, 2, 1],
    )
    np.testing.assert_allclose(CrowAMSAA.fit(data).params, expected.params)


def test_recurrent_range_rows_with_changing_covariates_refused():
    nan = np.nan
    data = SurpyvalData(
        x=[1, 2, nan, 2, 3, 6],
        xl=[nan, nan, 2.5, nan, nan, nan],
        xr=[nan, nan, 3.5, nan, nan, nan],
        c=[0, 1, 2, 0, 0, 1],
        i=["A", "A", "A", "B", "B", "B"],
        Z=[[0.0], [1.0], [1.0], [0.0], [0.0], [1.0]],
    )
    with pytest.raises(ValueError, match=r"ranges \(xl, xr\)") as info:
        ProportionalIntensityNHPP.fit(data)
    assert "unit 'A'" in str(info.value)
