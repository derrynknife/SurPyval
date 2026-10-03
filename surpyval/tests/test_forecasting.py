"""Forecasting failures from the current state of a fleet (#581).

``surpyval.forecast(model, age, horizon, Z=, n=, limit=)``: the expected
failures of units in service at their ages, each from the model's
conditional survival, with a Poisson-binomial prediction interval.
"""

import pickle
import warnings

import numpy as np
import pytest
from scipy.stats import binom

import surpyval as sp
from surpyval.forecasting import Forecast


def _convolved_cdf(p, n):
    pmf = np.array([1.0])
    for pi, ni in zip(p, n):
        pmf = np.convolve(pmf, binom.pmf(np.arange(ni + 1), ni, pi))
    return np.cumsum(pmf)


def test_581_warranty_cohorts_by_hand():
    # The issue's warranty forecast, written by hand for every model:
    # cohorts of survivors x (F(a + h) - F(a + h - 1)) / (1 - F(a)),
    # capped at the end of the warranty.
    model = sp.Weibull.from_params([60.0, 1.5])
    age = np.array([3.0, 2.0, 1.0, 11.5, 13.0])
    n = np.array([950, 1000, 1000, 800, 700])
    result = sp.forecast(model, age, horizon=[1, 2, 3], n=n, limit=12)
    by_hand = []
    for h in (1, 2, 3):
        end = np.minimum(age + h, 12.0)
        p = np.where(
            end > age, (model.ff(end) - model.ff(age)) / model.sf(age), 0.0
        )
        by_hand.append(np.sum(n * p))
    np.testing.assert_allclose(result.expected, by_hand, rtol=1e-12)
    np.testing.assert_allclose(
        np.cumsum(result.period_expected), result.expected, rtol=1e-12
    )
    # Past its warranty a cohort contributes nothing; at 11.5 only half
    # a month counts.
    assert np.all(result.probability[4] == 0)
    assert np.all(result.probability[3] == result.probability[3, 0])
    np.testing.assert_allclose(result.unit_expected.sum(0), result.expected)


def test_581_interval_is_the_exact_poisson_binomial():
    model = sp.Weibull.from_params([20.0, 2.5])
    age = np.array([1.0, 5.0, 12.0, 20.0, 30.0])
    n = np.array([3, 1, 4, 2, 1])
    result = sp.forecast(model, age, horizon=[2.0, 6.0], n=n, alpha_ci=0.2)
    for k in range(2):
        p = result.probability[:, k]
        cdf = _convolved_cdf(p, n)
        assert result.lower[k] == np.searchsorted(cdf, 0.1)
        assert result.upper[k] == np.searchsorted(cdf, 0.9)
        np.testing.assert_allclose(
            result.variance[k], np.sum(n * p * (1 - p)), rtol=1e-12
        )
    q = result.probability[:, 1] - result.probability[:, 0]
    cdf = _convolved_cdf(q, n)
    assert result.period_lower[1] == np.searchsorted(cdf, 0.1)
    assert result.period_upper[1] == np.searchsorted(cdf, 0.9)


def test_581_regression_fleet_uses_each_units_covariates():
    # The wind-farm forecast: P(removal within a year | survived to its
    # age, its site and load), summed over the fleet; it was
    # 1 - sf(a + 1, Z) / sf(a, Z) by hand.
    rng = np.random.default_rng(0)
    Z = np.column_stack([rng.binomial(1, 0.5, 120), rng.normal(0, 0.3, 120)])
    x = 10 * rng.weibull(2.5, 120) * np.exp(Z @ np.array([0.36, 2.0]))
    model = sp.WeibullAFT.fit(x, Z)
    age = rng.uniform(0, 12, 40)
    Zf = Z[:40]
    result = sp.forecast(model, age, horizon=1.0, Z=Zf)
    p = 1 - model.sf(age + 1, Zf) / model.sf(age, Zf)
    np.testing.assert_allclose(result.probability[:, 0], p, rtol=1e-10)
    np.testing.assert_allclose(result.expected, p.sum(), rtol=1e-10)
    # One covariate row for every unit.
    one = sp.forecast(model, age, horizon=1.0, Z=Zf[0])
    p0 = 1 - model.sf(age + 1, Zf[0]) / model.sf(age, Zf[0])
    np.testing.assert_allclose(one.probability[:, 0], p0, rtol=1e-10)
    cox = sp.CoxPH.fit(x, Z)
    p_cox = 1 - cox.sf(age + 1, Zf) / cox.sf(age, Zf)
    np.testing.assert_allclose(
        sp.forecast(cox, age, 1.0, Z=Zf).probability[:, 0],
        p_cox,
        rtol=1e-10,
    )


def test_581_models_without_cs_use_the_survival_ratio():
    x = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    km = sp.KaplanMeier.fit(x)
    result = sp.forecast(km, [2.5, 4.5], horizon=2.0)
    np.testing.assert_allclose(
        result.probability[:, 0], 1 - km.sf([4.5, 6.5]) / km.sf([2.5, 4.5])
    )


def test_581_units_the_model_says_cannot_survive_warn():
    model = sp.Uniform.from_params([0.0, 10.0])
    with pytest.warns(RuntimeWarning, match="survival of 0") as caught:
        result = sp.forecast(model, [2.0, 12.0], horizon=1.0)
    assert caught[0].filename == __file__
    assert np.isnan(result.expected[0])
    assert result.probability[0, 0] == pytest.approx(0.125)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"horizon": [2.0, 1.0]}, "increasing"),
        ({"horizon": 0.0}, "positive"),
        ({"age": [1.0, np.nan]}, "age must be finite"),
        ({"n": [1, 2.5]}, "whole numbers"),
        ({"n": [1, -1]}, "whole numbers"),
        ({"Z": [[1.0]]}, "takes no covariates"),
        ({"alpha_ci": 1.5}, "alpha_ci"),
    ],
)
def test_581_refuses_bad_input(kwargs, match):
    model = sp.Weibull.from_params([10.0, 2.0])
    args = {"age": [1.0, 2.0], "horizon": 1.0} | kwargs
    with pytest.raises(ValueError, match=match):
        sp.forecast(model, **args)


def test_581_regression_needs_covariates():
    rng = np.random.default_rng(1)
    Z = rng.binomial(1, 0.5, 50)
    model = sp.WeibullPH.fit(10 * rng.weibull(2, 50), Z)
    with pytest.raises(ValueError, match="give the covariates"):
        sp.forecast(model, [1.0], 1.0)


def test_581_large_fleet_is_fast_and_normal_like():
    # 20,000 units at distinct ages: the exact distribution would take
    # some 20 s per horizon; the refined normal approximation is used.
    rng = np.random.default_rng(2)
    model = sp.Weibull.from_params([60.0, 1.5])
    age = rng.uniform(0, 100, 20000)
    n = rng.integers(1, 50, 20000)
    result = sp.forecast(model, age, horizon=[6.0, 12.0], n=n)
    sd = np.sqrt(result.variance)
    assert np.all(np.abs(result.upper - (result.expected + 1.96 * sd)) < 3)
    assert np.all(np.abs(result.lower - (result.expected - 1.96 * sd)) < 3)


def test_581_forecast_pickles_and_prints():
    model = sp.Weibull.from_params([60.0, 1.5])
    result = sp.forecast(model, [3.0, 2.0], horizon=[1, 2], n=[10, 20])
    assert isinstance(result, Forecast)
    again = pickle.loads(pickle.dumps(result))
    np.testing.assert_array_equal(again.expected, result.expected)
    text = repr(result)
    assert text.startswith("Forecast of failures: 30 units; 95%")
    assert len(text.splitlines()) == 4
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.forecast(model, [3.0], horizon=1.0)
