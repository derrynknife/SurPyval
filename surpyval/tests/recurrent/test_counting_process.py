import numpy as np
import pytest

from surpyval.recurrent import (
    HPP,
    CountingProcess,
    CoxLewis,
    CrowAMSAA,
    Duane,
    ProportionalIntensityNHPP,
)
from surpyval.recurrent.parametric import CountingProcess as ParametricCP
from surpyval.utils.recurrent_utils import handle_xicn


def test_countingprocess_exported_consistently():
    # The same class should be reachable from both export points.
    assert CountingProcess is ParametricCP


@pytest.mark.parametrize("dist", [Duane, CrowAMSAA, CoxLewis])
def test_nhpp_intensity_models_are_counting_processes(dist):
    assert isinstance(dist, CountingProcess)


def test_hpp_is_a_counting_process():
    assert isinstance(HPP, CountingProcess)


def test_counting_process_cannot_be_instantiated():
    # It is abstract: the intensity functions must be supplied by subclasses.
    with pytest.raises(TypeError):
        CountingProcess()


def _toy_recurrent_data():
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    Z = np.array([[0.0], [1.0], [0.0], [1.0], [0.0]])
    return handle_xicn(x, Z=Z, as_recurrent_data=True)


@pytest.mark.parametrize("bad_dist", [object(), "Duane", Duane.__class__, 42])
def test_proportional_intensity_rejects_non_counting_process(bad_dist):
    data = _toy_recurrent_data()
    with pytest.raises(TypeError):
        ProportionalIntensityNHPP.fit_from_recurrent_data(data, dist=bad_dist)


def test_proportional_intensity_accepts_counting_process():
    # A genuine counting process must pass the type guard. The guard runs
    # before any optimisation, so it is enough to confirm that no TypeError
    # about ``dist`` is raised (any downstream numerical error is unrelated).
    data = _toy_recurrent_data()
    try:
        ProportionalIntensityNHPP.fit_from_recurrent_data(data, dist=Duane)
    except TypeError as e:  # pragma: no cover - guard must not trigger
        pytest.fail("valid CountingProcess was rejected: {}".format(e))
    except Exception:
        pass
