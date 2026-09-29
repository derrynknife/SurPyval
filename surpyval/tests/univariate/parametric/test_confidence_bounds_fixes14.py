"""Regression tests for the confidence-bound fixes of #411, #413, #414,
#415 and #418 (principles 18 and 22)."""

import warnings

import numpy as np
import pytest
from scipy.special import ndtri

import surpyval as surv
from surpyval.tests.conformance.registry import CASE_BY_NAME
from surpyval.utils.linalg import delta_method_se


def _fresh(name):
    # A fresh fit of a conformance fixture, not the shared cached one.
    case = CASE_BY_NAME[name]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return case.fit(case.data())


def _quiet(fn, *args, **kwargs):
    # Fails on any warning, deliberate or raw.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fn(*args, **kwargs)


# -- #413: rate bounds where the rate is 0 ----------------------------------
def test_rate_bounds_are_zero_below_an_offset():
    # hf and df are 0 below gamma; their log-scale bounds were log(0),
    # [nan, nan]. Both bounds are now 0.
    np.random.seed(3)
    x = surv.Weibull.random(40, 10, 2) + 5
    model = surv.Weibull.fit(x, offset=True)
    assert model.gamma > 4.0
    for on in ("hf", "df"):
        cb = _quiet(model.cb, [1.0, 4.0], on=on)
        np.testing.assert_array_equal(cb, np.zeros((2, 2)))
        for side in ("lower", "upper"):
            np.testing.assert_array_equal(
                _quiet(model.cb, [1.0, 4.0], on=on, bound=side), [0.0, 0.0]
            )
    # And inside the support they are still a proper interval.
    t = model.gamma + 5.0
    lo, hi = model.cb(t, on="hf")
    assert 0 < lo < model.hf(t) < hi


def test_discrete_hazard_bounds_are_zero_where_there_is_no_mass():
    # The Geometric has no mass at k = 0, so hf(0) = 0; the bounds were
    # [nan, nan].
    np.random.seed(4)
    model = surv.Geometric.fit(surv.Geometric.random(60, 0.3))
    assert model.hf(0) == 0.0
    np.testing.assert_array_equal(_quiet(model.cb, 0, on="hf"), [0.0, 0.0])


# -- #414: the discrete hazard's bound is about the model's hazard ----------
@pytest.mark.parametrize(
    "dist, params",
    [
        (surv.Poisson, (4.0,)),
        (surv.Geometric, (0.25,)),
        (surv.DiscreteWeibull, (0.9, 1.5)),
    ],
)
def test_discrete_hazard_bounds_contain_the_hazard(dist, params):
    # hf(k) = df(k) / sf(k - 1); the bound was built on df(k) / sf(k):
    # Poisson(4) at k = 6: hf 0.371 but the bound was about 0.588.
    np.random.seed(5)
    model = dist.fit(dist.random(200, *params))
    k = np.array([2.0, 4.0, 6.0])
    hf = model.hf(k)
    np.testing.assert_allclose(hf, model.df(k) / model.sf(k - 1), rtol=1e-12)
    cb = model.cb(k, on="hf")
    assert np.all(cb[:, 0] <= hf) and np.all(hf <= cb[:, 1])
    # A discrete hazard is a probability: the bounds stay in [0, 1].
    assert np.all((cb >= 0) & (cb <= 1))
    # The interval closes onto the hazard as alpha_ci -> 1.
    np.testing.assert_allclose(
        model.cb(k, on="hf", alpha_ci=1 - 1e-6), np.c_[hf, hf], rtol=1e-5
    )


def test_discrete_lfp_hazard_is_conditioned_on_the_step_before():
    # Parametric.hf of a limited-failure (or zero-inflated) discrete model
    # was df(k) / sf(k), not the discrete hazard df(k) / sf(k - 1) that
    # the distributions and the other models use.
    model = surv.Poisson.from_params([3.0], p=0.8)
    k = np.array([0.0, 2.0, 5.0])
    np.testing.assert_allclose(
        model.hf(k), model.df(k) / model.sf(k - 1), rtol=1e-12
    )


# -- #415: Royston-Parmar one-sided bounds ----------------------------------
@pytest.fixture(scope="module")
def royston_parmar():
    np.random.seed(6)
    x = surv.Weibull.random(150, 10, 1.6)
    return surv.RoystonParmar.fit(x, df=2)


@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_royston_parmar_one_sided_is_the_matching_end(royston_parmar, on):
    # bound='lower' on ff and Hf returned the upper end (ff(10) = 0.436:
    # two-sided at 0.1 [0.291, 0.615], lower at 0.05 0.615).
    model = royston_parmar
    x = np.array([4.0, 10.0, 20.0])
    two = model.cb(x, on=on, alpha_ci=0.1)
    est = getattr(model, on)(x)
    assert np.all(two[:, 0] < est) and np.all(est < two[:, 1])
    for k, side in enumerate(("lower", "upper")):
        np.testing.assert_allclose(
            model.cb(x, on=on, alpha_ci=0.05, bound=side),
            two[:, k],
            rtol=1e-12,
        )


def test_royston_parmar_refuses_an_unknown_bound(royston_parmar):
    # bound='both' was taken as 'upper'.
    with pytest.raises(ValueError, match="bound"):
        royston_parmar.cb(10.0, bound="both")


# -- #418: the regression survival bound on the Hf scale --------------------
@pytest.mark.parametrize("name", ["GumbelPH", "GumbelAFT", "NormalPH"])
def test_regression_Hf_bounds_have_no_ceiling(name):
    # sf was clipped at 1e-15 on the logit scale, so the Hf bounds stopped
    # at -log(1e-15) = 34.54: GumbelPH's Hf(22, [1, -0.2]) = 110.6 had the
    # bounds [34.54, 34.54].
    model = _fresh(name)
    Z = np.array([1.0, -0.2])
    H = model.Hf(22.0, Z)
    assert H > 34.6
    lo, hi = model.cb(22.0, Z, on="Hf")
    assert lo <= H <= hi and hi > 34.6
    # The interval closes onto the estimate as alpha_ci -> 1.
    np.testing.assert_allclose(
        model.cb(22.0, Z, on="Hf", alpha_ci=1 - 1e-6), [H, H], rtol=1e-5
    )
    # One-sided bounds are the matching ends of the two-sided one.
    two = model.cb(22.0, Z, on="Hf", alpha_ci=0.1)
    for k, side in enumerate(("lower", "upper")):
        np.testing.assert_allclose(
            model.cb(22.0, Z, on="Hf", alpha_ci=0.05, bound=side),
            two[k],
            rtol=1e-12,
        )


def test_regression_sf_bounds_are_the_logit_wald_bounds():
    # The survival bound is still the logit-scale Wald bound, now formed
    # from the cumulative hazard: at a moderate sf it is unchanged.
    np.random.seed(7)
    Z = np.random.binomial(1, 0.5, 120).reshape(-1, 1).astype(float)
    x = surv.Weibull.random(120, 10, 2) * np.exp(-0.5 * Z[:, 0])
    model = surv.WeibullPH.fit(x, Z)
    t, z = np.array([3.0, 8.0, 15.0]), np.array([[1.0]])
    params, cov = np.asarray(model.params), model.covariance()

    def logit(p):
        s = model.model.sf(t, z, *p)
        return np.log(s / (1 - s))

    se = delta_method_se(logit, params, cov)
    q = ndtri(0.975)
    L = logit(params)[:, None] + np.array([-q, q]) * se[:, None]
    np.testing.assert_allclose(model.cb(t, z), 1 / (1 + np.exp(-L)), rtol=1e-6)


# -- #411: a Wald bound that does not exist warns and is nan ----------------
def test_param_cb_with_a_negative_variance_warns():
    # GeneralizedRenewal's fixture fit puts q at 2.7e-16, where the inverse
    # Hessian's diagonal is negative: param_cb was [nan, nan] with only
    # numpy's raw "invalid value encountered in sqrt".
    model = _fresh("GeneralizedRenewal")
    for name in ("alpha", "q"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with np.errstate(all="raise"):
                cb = model.param_cb(name)
        assert np.all(np.isnan(cb)) and cb.shape == (2,)
        assert len(caught) == 1, [str(w.message) for w in caught]
        message = str(caught[0].message)
        assert f"'{name}'" in message and "undefined" in message
        assert caught[0].filename == __file__


def test_param_cb_at_the_edge_of_an_interval_support_warns():
    # ARI's fixture fit puts rho at exactly 1.0, the upper end of (0, 1):
    # param_cb('rho') raised ZeroDivisionError (the logit of 1).
    model = _fresh("ARI")
    with pytest.warns(RuntimeWarning, match="'rho' is undefined"):
        cb = model.param_cb("rho", bound="lower")
    assert cb.shape == (1,) and np.isnan(cb[0])


def test_parametric_param_cb_with_a_negative_variance_warns():
    np.random.seed(8)
    model = surv.Weibull.fit(surv.Weibull.random(30, 10, 3))
    model.hess_inv = np.array([[-1.0, 0.0], [0.0, 0.1]])
    with pytest.warns(RuntimeWarning, match="'alpha' is undefined") as rec:
        cb = model.param_cb("alpha")
    assert np.all(np.isnan(cb)) and len(rec) == 1
    assert rec[0].filename == __file__
    # The other parameter's bound is unaffected, and silent.
    assert np.all(np.isfinite(_quiet(model.param_cb, "beta")))


def test_function_cb_with_a_negative_variance_warns():
    # Uniform's covariance on its fixture is not positive definite, so the
    # delta-method variance of df = 1 / (b - a) is negative: the df
    # bounds were a silent [nan, nan] everywhere inside the support.
    model = _fresh("Uniform")
    with pytest.warns(RuntimeWarning, match=r"df at x = \[5.0\]") as rec:
        cb = model.cb([1.0, 5.0], on="df")
    assert len(rec) == 1 and rec[0].filename == __file__
    np.testing.assert_array_equal(cb[0], [0.0, 0.0])  # below a: rate 0
    assert np.all(np.isnan(cb[1]))


# -- #421: likelihood-ratio bands ------------------------------------------
def test_lr_band_does_not_stall_on_the_far_side_of_the_estimate():
    # A density at x peaks in the scale, and the warm-started search for
    # the lower df bound stopped on that peak: Rayleigh's 99% df band at
    # 14.6 was [0.0502, 0.0504], above the estimate 0.0359.
    model = _fresh("Rayleigh")
    x = np.array([3.2, 8.0, 14.6])
    df = model.df(x)
    for alpha in (0.01, 0.05, 0.2):
        cb = _quiet(model.cb, x, on="df", alpha_ci=alpha, method="lr")
        assert np.all(cb[:, 0] <= df) and np.all(df <= cb[:, 1]), cb


def test_lr_band_of_one_parameter_is_the_extreme_over_its_interval():
    # With one free parameter the likelihood region is the profile
    # interval, and the band is the extreme of the function over it. The
    # search found one end or the other: Geometric's df(5) lower bound
    # was 0.0740 in a sweep over [2, 5, 8] and 0.0652 queried alone.
    model = _fresh("Geometric")
    x = np.array([2.0, 5.0, 8.0])
    for alpha in (0.05, 0.2):
        band = model.cb(x, on="df", alpha_ci=alpha, method="lr")
        lo, hi = model.param_cb("p", alpha_ci=alpha, method="lr")
        grid = np.linspace(lo, hi, 2001)
        want = np.array(
            [
                [f.min(), f.max()]
                for f in (surv.Geometric.df(k, grid) for k in x)
            ]
        )
        np.testing.assert_allclose(band, want, rtol=1e-5)
        for k in range(x.size):
            np.testing.assert_allclose(
                model.cb(x[k], on="df", alpha_ci=alpha, method="lr"),
                band[k],
                rtol=1e-8,
            )
