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


# -- quantile_cb -------------------------------------------------------------
def test_583_quantile_cb_wald_is_the_delta_method_on_the_log_quantile(alt):
    # log t_p = log c + a / T + n log V + log(-log(1 - p)) / beta
    T, V = USE[0]

    def log_tp(theta):
        beta, c, a, n = theta[1], theta[2], theta[3], theta[4]
        life = np.log(c) + a / T + n * np.log(V)
        return life + np.log(-np.log(0.9)) / beta

    theta = np.asarray(alt.params, dtype=float)
    cov = alt.covariance()
    grad = np.empty(len(theta))
    for j in range(len(theta)):
        h = 1e-6 * max(1.0, abs(theta[j]))
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        grad[j] = (log_tp(up) - log_tp(down)) / (2 * h)
    se = np.sqrt(grad @ cov @ grad)
    z = 1.6448536269514722
    expected = np.exp(log_tp(theta) + np.array([-z, z]) * se)
    got = alt.quantile_cb(0.1, USE, alpha_ci=0.1)
    np.testing.assert_allclose(got, expected, rtol=1e-5)
    assert got[0] < alt.qf(0.1, USE) < got[1]


def test_583_quantile_cb_lr_is_the_sf_band_inverted(alt):
    # t_p <= T exactly when F(T) >= p: the likelihood-ratio bounds on the
    # B10 life are the times at which the likelihood-ratio band on sf is
    # 0.9, and they come from the same region.
    lo, hi = alt.quantile_cb(0.1, USE, alpha_ci=0.1, method="lr")
    assert lo < alt.qf(0.1, USE) < hi
    assert alt.cb(lo, USE, alpha_ci=0.1, method="lr")[0] == pytest.approx(
        0.9, abs=1e-6
    )
    assert alt.cb(hi, USE, alpha_ci=0.1, method="lr")[1] == pytest.approx(
        0.9, abs=1e-6
    )


def test_583_quantile_cb_shapes_options_and_sides(alt):
    rows = np.array([[358.15, 450.0], USE[0]])
    assert alt.quantile_cb([0.1, 0.5], rows).shape == (2, 2)
    assert alt.quantile_cb([0.1, 0.5], USE, bound="lower").shape == (2,)
    assert np.ndim(alt.quantile_cb(0.1, USE, bound="upper")) == 0
    two = alt.quantile_cb(0.5, USE, alpha_ci=0.2, method="lr")
    upper = alt.quantile_cb(0.5, USE, alpha_ci=0.1, bound="upper", method="lr")
    assert upper == two[1]
    with pytest.raises(ValueError, match="'p' must be in"):
        alt.quantile_cb(1.0, USE)
    with pytest.raises(ValueError, match="method"):
        alt.quantile_cb(0.1, USE, method="bootstrap")


@pytest.mark.parametrize("fitter", [WeibullPH, sp.LogNormalAFT])
def test_583_quantile_cb_of_a_centred_fit(fitter):
    rng = np.random.default_rng(1)
    Z = rng.uniform(5.0, 7.0, size=(60, 1))
    x = 10.0 * np.exp(-0.5 * (Z[:, 0] - 6.0)) * rng.weibull(1.5, 60)
    model = fitter.fit(x, Z=Z)
    q = model.qf([0.1, 0.5], [[6.0], [9.0]])
    for method in ("wald", "lr"):
        b = model.quantile_cb([0.1, 0.5], [[6.0], [9.0]], method=method)
        assert np.all((b[:, 0] < q) & (q < b[:, 1]))


def test_583_a_model_with_searches_kept_pickles(alt):
    # The searches are kept on the model (#573: fitted models pickle).
    import pickle

    alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, method="lr")
    restored = pickle.loads(pickle.dumps(alt))
    np.testing.assert_array_equal(
        restored.cb(FIVE_YEARS, USE, alpha_ci=0.1, method="lr"),
        alt.cb(FIVE_YEARS, USE, alpha_ci=0.1, method="lr"),
    )


# -- cb_tvc (#617) -----------------------------------------------------------
@pytest.fixture(scope="module")
def ph():
    rng = np.random.default_rng(0)
    Z = rng.uniform(0, 1, (200, 1))
    x = 100 * rng.weibull(2, 200) * np.exp(-0.5 * Z[:, 0])
    return WeibullPH.fit(x, Z)


def _schedule_profile_deviance(model, t, s):
    """The profile deviance of the survival ``s`` at ``t`` along the step
    schedule Z = 0 on [0, 30), 1 after, for a Weibull PH model: its
    cumulative hazard there is ``alpha^-beta [30^beta + e^b0 (t^beta -
    30^beta)]``, so ``alpha`` is the value that gives ``-log s`` for each
    ``(beta, b0)``, over which the likelihood is maximised."""
    w = -np.log(s)

    def nll(v):
        beta, b0 = np.exp(v[0]), v[1]
        total = 30.0**beta + np.exp(b0) * (t**beta - 30.0**beta)
        alpha = (total / w) ** (1.0 / beta)
        return float(model.model.neg_ll(model.data, alpha, beta, b0))

    p = np.asarray(model.params, dtype=float)
    v = np.array([np.log(p[1]), p[2]])
    best = np.inf
    for _ in range(5):
        res = minimize(nll, v, method="Nelder-Mead", options=_NM)
        if not res.fun < best - 1e-10:
            break
        best, v = res.fun, res.x
    return 2.0 * (best - model._neg_ll)


def test_617_cb_tvc_lr_bound_is_where_the_profile_deviance_is_chi2(ph):
    schedule = np.array([[0.0], [1.0]])
    lo, hi = ph.cb_tvc(60.0, schedule, xl=[0.0, 30.0], method="lr")
    crit = chi2.ppf(0.95, 1)
    assert lo < ph.sf_tvc(60.0, schedule, xl=[0.0, 30.0]) < hi
    for bound in (lo, hi):
        dev = _schedule_profile_deviance(ph, 60.0, bound)
        assert dev == pytest.approx(crit, abs=2e-4)


def test_617_cb_tvc_lr_along_a_constant_path_is_cb_lr(ph):
    flat = sp.CovariatePath.from_points([0], [0.5])
    t = np.array([40.0, 80.0])
    np.testing.assert_allclose(
        ph.cb_tvc(t, flat, method="lr"),
        ph.cb(t, [0.5], method="lr"),
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        ph.cb_tvc(t, flat, on="Hf", bound="upper", method="lr"),
        ph.cb(t, [0.5], on="Hf", bound="upper", method="lr"),
        rtol=1e-10,
    )


def test_617_cb_tvc_lr_is_an_option_with_the_shapes_of_wald(ph):
    ramp = sp.CovariatePath.from_points([0, 50], [0.0, 1.0])
    t = np.array([0.0, 40.0, 80.0])
    wald = ph.cb_tvc(t, ramp)
    assert np.array_equal(wald, ph.cb_tvc(t, ramp, method="wald"))
    lr = ph.cb_tvc(t, ramp, method="likelihood")
    assert lr.shape == wald.shape == (3, 2)
    # Nothing has happened at 0: the bound is the estimate, as Wald's.
    assert np.array_equal(lr[0], [1.0, 1.0])
    est = ph.sf_tvc(t[1:], ramp)
    assert np.all((lr[1:, 0] < est) & (est < lr[1:, 1]))
    # Close to Wald's on this well-determined fit, and not the same.
    np.testing.assert_allclose(lr, wald, rtol=0.02)
    assert not np.allclose(lr, wald, rtol=1e-6)
    # The sf, ff and Hf bounds are one bound.
    ff = ph.cb_tvc(t[1:], ramp, on="ff", method="lr")
    np.testing.assert_allclose(ff, 1.0 - lr[1:, ::-1], rtol=1e-12)
    # Conditional on survival to 30: certain to there.
    given = ph.cb_tvc([20.0, 60.0], ramp, given=30.0, method="lr")
    assert np.array_equal(given[0], [1.0, 1.0])
    est = ph.sf_tvc(60.0, ramp, given=30.0)
    assert given[1, 0] < est < given[1, 1]
    assert ph.cb_tvc(80.0, ramp, bound="lower", method="lr").shape == ()
    with pytest.raises(ValueError, match="method"):
        ph.cb_tvc(t, ramp, method="bootstrap")
