"""
Regression tests for the distribution-level defect batch (#257).
"""

import numpy as np
import pytest

from surpyval import (
    Bernoulli,
    Beta,
    ExactEventTime,
    Exponential,
    ExpoWeibull,
    Gamma,
    GumbelLEV,
    LogNormal,
    Logistic,
)


def test_lognormal_fits_negative_mu():
    # mu (mean of log-data) is any real; a (0, None) bound crashed every
    # fit with geometric mean < 1.
    np.random.seed(0)
    m = LogNormal.fit(LogNormal.random(500, -0.5, 0.5))
    assert m.params[0] == pytest.approx(-0.5, abs=0.1)


def test_bernoulli_fit_works_for_general_inputs():
    assert Bernoulli.fit([0, 1, 1, 0, 1]).params[0] == pytest.approx(0.6)
    assert Bernoulli.fit([1, 1]).params[0] == 1.0
    assert Bernoulli.fit([0, 1], n=[3, 1]).params[0] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        Bernoulli.fit([0, 2, 1])
    assert Bernoulli.from_params(0.3).params[0] == pytest.approx(0.3)


def test_exponential_offset_mpp_rr_x_inversion():
    np.random.seed(0)
    x = Exponential.random(2000, 0.5) + 10
    m = Exponential.fit(x, how="MPP", rr="x", offset=True)
    assert m.params[0] == pytest.approx(0.5, rel=0.3)
    assert m.gamma == pytest.approx(10.0, abs=0.5)


def test_gamma_censored_mpp_rr_x_does_not_crash():
    np.random.seed(1)
    x = Gamma.random(300, 3, 2)
    c = (x > 2.5).astype(int)
    m = Gamma.fit(np.minimum(x, 2.5), c=c, how="MPP", rr="x")
    assert np.all(np.isfinite(m.params))


@pytest.mark.parametrize("rr", ["x", "y"])
def test_gamma_offset_mpp_recovers_offset(rr):
    np.random.seed(2)
    x = Gamma.random(300, 3, 2) + 10
    m = Gamma.fit(x, how="MPP", rr=rr, offset=True)
    assert m.gamma == pytest.approx(10.0, abs=1.5)
    assert m.params[0] == pytest.approx(3.0, rel=0.5)


def test_expo_weibull_tail_is_stable():
    # 1 - (1 - e^-t)^mu underflowed to exactly 0 once e^-t < 1e-16.
    s = float(ExpoWeibull.sf(8, 3, 4, 1.2))
    assert 0.0 < s < 1e-18
    assert np.isfinite(float(ExpoWeibull.Hf(8, 3, 4, 1.2)))
    assert np.isfinite(float(ExpoWeibull.log_sf(8, 3, 4, 1.2)))
    # Moderate-x values agree with the naive form.
    naive = 1 - (1 - np.exp(-((2 / 3) ** 4))) ** 1.2
    assert float(ExpoWeibull.sf(2, 3, 4, 1.2)) == pytest.approx(
        naive, abs=1e-12
    )


def test_probability_plot_transform_roundtrips():
    F = np.array([0.05, 0.3, 0.7, 0.95])
    assert np.allclose(
        Exponential.mpp_inv_y_transform(Exponential.mpp_y_transform(F)), F
    )
    assert np.allclose(
        GumbelLEV.mpp_inv_y_transform(GumbelLEV.mpp_y_transform(F)), F
    )
    assert np.allclose(
        Beta.mpp_inv_y_transform(Beta.mpp_y_transform(F, 2.0, 3.0), 2.0, 3.0),
        F,
    )


def test_logistic_log_functions_deep_tail():
    assert np.isfinite(Logistic.log_sf(np.array([-800000.0]), 0.0, 1.0)).all()
    assert np.isfinite(Logistic.log_ff(np.array([800000.0]), 0.0, 1.0)).all()


def test_exact_event_time_informative_error():
    with pytest.raises(ValueError, match="right-censored"):
        ExactEventTime.fit(np.array([1.0, 2.0, 3.0]), c=np.array([1, 1, 1]))
