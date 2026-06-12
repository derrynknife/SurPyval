"""
Regression tests for specific bug fixes in the univariate parametric module.
"""

import numpy as np
import pytest

from surpyval import (
    Bernoulli,
    Beta,
    ExactEventTime,
    Exponential,
    LogLogistic,
    LogNormal,
    Normal,
    Parametric,
    Weibull,
)


def test_mse_with_offset():
    # MSE used to pass the offset parameter, gamma, straight into the
    # distribution's ff resulting in a TypeError.
    np.random.seed(1)
    x = Weibull.random(100, 10, 3) + 10

    model = Weibull.fit(x, how="MSE", offset=True)

    assert model.params.size == 2
    assert 5 < model.gamma < 15
    assert np.isfinite(model.sf(x.mean()))


def test_exact_event_time_Hf():
    # Hf used to drop the T parameter and fail with a TypeError.
    model = ExactEventTime.from_params(10)

    Hf = model.Hf([5, 15])
    assert Hf[0] == 0
    assert np.isinf(Hf[1])


def test_loglogistic_offset_initial_guess_length():
    # The offset initial guess returned one parameter too many, which the
    # parameter transforms silently truncated.
    init = LogLogistic._parameter_initialiser(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]), offset=True
    )
    assert len(init) == LogLogistic.k + 1


def test_lognormal_offset_initial_guess_length():
    # The offset initial guess ignored the offset flag and returned one
    # parameter too few, so offset fits crashed before optimising.
    init = LogNormal._parameter_initialiser(
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]), offset=True
    )
    assert len(init) == LogNormal.k + 1


def test_mps_right_truncation_clamped_to_support():
    # Right-truncation values beyond the distribution's support were not
    # clamped (the check was made against the left truncation values).
    np.random.seed(5)
    x = Beta.random(100, 2, 3)

    model = Beta.fit(x, how="MPS", tr=2.0)
    clamped = Beta.fit(x, how="MPS", tr=1.0)

    assert np.allclose(model.params, clamped.params)


def test_mom_truncation_error_message():
    # The error message referred to maximum product spacing.
    with pytest.raises(ValueError, match="Method of moments"):
        Weibull.fit(np.array([1.0, 2.0, 3.0]), how="MOM", tl=0.5)


def test_zi_support_error_message():
    # Raised an empty ValueError.
    with pytest.raises(ValueError, match="zero-inflated"):
        Normal.fit(np.array([1.0, 2.0, 3.0]), zi=True)


def test_mpp_left_censoring_heuristic_error_message():
    # Raised an empty ValueError.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([-1, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="Turnbull"):
        Weibull.fit(x, c, how="MPP", heuristic="Nelson-Aalen")


def test_exponential_mpp():
    # Exponential's custom mpp returned a bare tuple instead of a results
    # dict, so fitting with how="MPP" raised an AttributeError.
    np.random.seed(1)
    x = Exponential.random(100, 0.2)

    model = Exponential.fit(x, how="MPP")
    assert np.isclose(model.params[0], 0.2, rtol=0.3)

    model = Exponential.fit(x + 3, how="MPP", offset=True)
    assert np.isclose(model.gamma, 3, atol=1)


def test_bernoulli_dict_round_trip():
    # Bernoulli_ did not subclass ParametricFitter, so from_dict rejected
    # its own to_dict output as an unknown distribution.
    model = Bernoulli.from_params(0.3)
    recovered = Parametric.from_dict(model.to_dict())
    assert np.allclose(recovered.params, model.params)


def test_exact_event_time_dict_round_trip():
    model = ExactEventTime.from_params(10)
    recovered = Parametric.from_dict(model.to_dict())
    assert np.allclose(recovered.params, model.params)
