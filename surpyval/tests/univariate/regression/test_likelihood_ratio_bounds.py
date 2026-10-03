"""Likelihood-ratio bounds of the parametric regression models (#583).

``cb(method="lr")`` and ``param_cb(method="lr")`` of a
``ParametricRegressionModel`` are the univariate models' likelihood-ratio
bounds over all the regression model's parameters: a bound is the extreme
of the function at the covariates asked for over the region
``{theta : 2[nll(theta) - nll_hat] <= chi2_1}``, which is where the
function's profile deviance reaches the critical value. The tests check
that against a profile computed here, independently of the searches, on
the accelerated life test of #583; the coverage at its use condition is
in ``calibration/test_coverage_regression.py``.
"""

import warnings

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.stats import chi2

import surpyval as sp
from surpyval import (
    AcceleratedLife,
    Weibull,
    WeibullAFT,
    WeibullPH,
    life_models,
)

# The #583 design: Arrhenius in temperature, a power law in voltage, a
# Weibull shape of 2.2; 85/105/125 C by 450/500 V, 12 units a cell, 3000 h.
K_BOLTZMANN = 8.617e-5
A_TRUE = 0.7 / K_BOLTZMANN
USE = np.array([[45.0 + 273.15, 400.0]])
FIVE_YEARS = 5 * 8760.0


def _alt_data(seed=583):
    rng = np.random.default_rng(seed)
    T, V = np.meshgrid(
        np.array([85.0, 105.0, 125.0]) + 273.15, [450.0, 500.0], indexing="ij"
    )
    Z = np.repeat(np.column_stack([T.ravel(), V.ravel()]), 12, axis=0)
    life = 118.0 * np.exp(A_TRUE / Z[:, 0]) * Z[:, 1] ** -3.0
    t = life * rng.weibull(2.2, len(Z))
    return np.minimum(t, 3000.0), (t > 3000.0).astype(int), Z


@pytest.fixture(scope="module")
def alt():
    x, c, Z = _alt_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AcceleratedLife(Weibull, life_models.PowerExponential).fit(
            x, Z=Z, c=c
        )
    return model


def _sf_profile_deviance(model, x, z, s):
    """The profile deviance of ``sf(x, z) = s``: the least negative
    log-likelihood of the Weibull power-exponential model with that
    survival, over (beta, a, n), with ``c`` the value that gives it."""
    T, V = z
    w = -np.log(s)

    def nll(v):
        beta, a, n = np.exp(v[0]), v[1], v[2]
        # sf = exp(-(x / L)^beta): L = x / w^(1 / beta), L = c e^(a/T) V^n
        log_c = np.log(x) - np.log(w) / beta - a / T - n * np.log(V)
        theta = [1.0, beta, np.exp(log_c), a, n]
        return float(model.model.neg_ll(model.data, *theta))

    p = np.asarray(model.params, dtype=float)
    v = np.array([np.log(p[1]), p[3], p[4]])
    # Nelder-Mead from the estimate, restarted until it stops moving
    best = np.inf
    for _ in range(5):
        res = minimize(nll, v, method="Nelder-Mead", options=_NM)
        if not res.fun < best - 1e-10:
            break
        best, v = res.fun, res.x
    return 2.0 * (best - model._neg_ll)


_NM = {"xatol": 1e-9, "fatol": 1e-11, "maxiter": 20000, "maxfev": 20000}


def test_583_cb_lr_bound_is_where_the_profile_deviance_is_chi2(alt):
    lo, hi = alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, method="lr")
    crit = chi2.ppf(0.9, 1)
    assert lo < alt.sf(FIVE_YEARS, USE) < hi
    for bound in (lo, hi):
        dev = _sf_profile_deviance(alt, FIVE_YEARS, USE[0], bound)
        assert dev == pytest.approx(crit, abs=2e-4)


def test_583_param_cb_lr_bound_is_where_the_profile_deviance_is_chi2(alt):
    lo, hi = alt.param_cb("n", alpha_ci=0.1, method="lr")
    crit = chi2.ppf(0.9, 1)
    x, c, Z = _alt_data()
    fitter = AcceleratedLife(Weibull, life_models.PowerExponential)
    for bound in (lo, hi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            profile = fitter.fit(x, Z=Z, c=c, fixed={"n": bound})
        dev = 2.0 * (profile._neg_ll - alt._neg_ll)
        assert dev == pytest.approx(crit, abs=1e-4)
    assert lo < alt.params[4] < hi


def test_583_the_aft_form_of_the_model_gives_the_same_lr_bounds(alt):
    # WeibullAFT on [1/T, log V] is the same model (the issue found the
    # same estimate and Wald bands): the likelihood region, and so its
    # bounds, do not depend on the parameterisation. (The calibration
    # study fits it in this form, twenty times as fast.)
    x, c, Z = _alt_data()

    def terms(Z):
        return np.column_stack([1.0 / Z[:, 0], np.log(Z[:, 1])])

    aft = WeibullAFT.fit(x, Z=terms(Z), c=c)
    np.testing.assert_allclose(
        aft.cb(FIVE_YEARS, terms(USE), alpha_ci=0.1, method="lr"),
        alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, method="lr"),
        rtol=1e-6,
    )
    # n = -beta_1
    np.testing.assert_allclose(
        -aft.param_cb("beta_1", alpha_ci=0.1, method="lr")[::-1],
        alt.param_cb("n", alpha_ci=0.1, method="lr"),
        rtol=1e-5,
    )


def test_583_lr_is_an_option_and_wald_stays_the_default(alt):
    wald = alt.cb(FIVE_YEARS, USE, alpha_ci=0.1)
    assert np.array_equal(
        wald, alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, method="wald")
    )
    lr = alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, method="LR")
    assert not np.allclose(wald, lr)
    # The aliases of the univariate method
    for name in ("likelihood", "likelihood-ratio", "profile"):
        assert np.array_equal(
            lr, alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, method=name)
        )
    with pytest.raises(ValueError, match="method"):
        alt.cb(FIVE_YEARS, USE, method="bootstrap")
    with pytest.raises(ValueError, match="method"):
        alt.param_cb("n", method="bootstrap")


def test_583_the_sf_ff_and_Hf_bounds_are_one_bound(alt):
    sf = alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, method="lr")
    ff = alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, on="ff", method="lr")
    Hf = alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, on="Hf", method="lr")
    np.testing.assert_allclose(ff, 1.0 - sf[::-1], rtol=1e-12)
    np.testing.assert_allclose(Hf, -np.log(sf[::-1]), rtol=1e-10)


def test_583_one_sided_bounds_are_the_two_sided_ends_at_twice_alpha(alt):
    two = alt.cb(FIVE_YEARS, USE, alpha_ci=0.2, method="lr")
    lower = alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, bound="lower", method="lr")
    upper = alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, bound="upper", method="lr")
    assert lower == two[0] and upper == two[1]
    two_n = alt.param_cb("n", alpha_ci=0.2, method="lr")
    lower_n = alt.param_cb("n", alpha_ci=0.1, bound="lower", method="lr")
    assert lower_n.shape == (1,) and lower_n[0] == two_n[0]


def test_583_shapes_rows_and_below_the_support(alt):
    x = np.array([-1.0, 1000.0, FIVE_YEARS])
    rows = np.array([[358.15, 450.0], [378.15, 475.0], USE[0]])
    band = alt.cb(x, rows, alpha_ci=0.1, method="lr")
    assert band.shape == (3, 2)
    # Below the support nothing has happened yet: the bound is sf = 1.
    assert np.array_equal(band[0], [1.0, 1.0])
    # Each row's bound is the bound at that row alone.
    alone = alt.cb(x[2], rows[2], alpha_ci=0.1, method="lr")
    np.testing.assert_allclose(band[2], alone, rtol=1e-12)
    est = alt.sf(x[1:], rows[1:])
    assert np.all((band[1:, 0] < est) & (est < band[1:, 1]))
    # One row for every time
    assert alt.cb(x[1:], USE, method="lr", bound="lower").shape == (2,)


def test_583_param_cb_lr_respects_the_parameter_space(alt):
    # The life model's constant c is positive: its Wald interval is not
    # (a standard error ten times the estimate), the profile one is.
    lo, hi = alt.param_cb("c", alpha_ci=0.1, method="lr")
    assert 0.0 < lo < alt.params[2] < hi
    assert alt.param_cb("c", alpha_ci=0.1)[0] < 0.0


@pytest.mark.parametrize("fitter", [WeibullPH, WeibullAFT])
def test_583_lr_bounds_of_the_log_linear_families(fitter):
    # A centred fit (#463): the bounds are searched in the centred
    # parameters, the parameters' intervals in the reported ones.
    rng = np.random.default_rng(1)
    Z = rng.uniform(5.0, 7.0, size=(60, 1))
    x = 10.0 * np.exp(-0.5 * (Z[:, 0] - 6.0)) * rng.weibull(1.5, 60)
    model = fitter.fit(x, Z=Z)
    assert model._fit_centring is not None
    crit = chi2.ppf(0.95, 1)
    lo, hi = model.param_cb("beta_0", method="lr")
    for bound in (lo, hi):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            profile = fitter.fit(x, Z=Z, fixed={"beta_0": bound})
        dev = 2.0 * (profile._neg_ll - model._neg_ll)
        assert dev == pytest.approx(crit, abs=1e-4)
    band = model.cb([5.0, 10.0], [[6.0], [9.0]], method="lr")
    est = model.sf([5.0, 10.0], [[6.0], [9.0]])
    assert np.all((band[:, 0] < est) & (est < band[:, 1]))
    rates = model.cb([5.0, 10.0], [6.0], on="hf", method="lr")
    assert np.all(rates > 0) and np.all(rates[:, 0] < rates[:, 1])


def test_583_lr_bounds_of_a_fixed_parameter_and_a_restored_model(alt):
    x, c, Z = _alt_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = WeibullPH.fit(x, Z=Z / 100.0, c=c, fixed={"beta": 2.0})
    assert np.array_equal(
        model.param_cb("beta", method="lr"), np.array([2.0, 2.0])
    )
    lo, hi = model.param_cb("beta_0", method="lr")
    assert lo < model.params[2] < hi
    restored = type(model).from_dict(model.to_dict())
    with pytest.raises(ValueError, match="data"):
        restored.cb(100.0, Z[0] / 100.0, method="lr")
    with pytest.raises(ValueError, match="data"):
        restored.param_cb("beta_0", method="lr")
    # The Wald bounds still come from the stored covariance.
    assert np.all(np.isfinite(restored.cb(100.0, Z[0] / 100.0)))


def test_583_a_bound_on_the_additive_hazards_model():
    rng = np.random.default_rng(3)
    Z = rng.uniform(0, 1, size=(80, 1))
    x = rng.weibull(1.5, 80) * 10.0 / (1.0 + Z[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullAH.fit(x, Z=Z)
    band = model.cb([2.0, 8.0], [0.5], method="lr")
    est = model.sf([2.0, 8.0], [0.5])
    assert np.all((band[:, 0] < est) & (est < band[:, 1]))
