"""
Tests for the confidence bounds of univariate parametric models.
"""

import numpy as np
import pytest
from autograd import jacobian
from scipy.special import ndtri as z

import surpyval as surv


@pytest.fixture(scope="module")
def gumbel_model():
    # Gumbel has no distribution-specific R_cb, so it exercises the
    # general confidence bound solution in Parametric.cb.
    np.random.seed(1)
    x = surv.Gumbel.random(100, 10, 2)
    return surv.Gumbel.fit(x)


@pytest.fixture(scope="module")
def weibull_model():
    # Weibull has its own R_cb and so exercises the specialised path.
    np.random.seed(2)
    x = surv.Weibull.random(100, 10, 2)
    return surv.Weibull.fit(x)


def delta_method_var(model, func):
    jac = np.atleast_2d(jacobian(func)(np.array(model.params)))
    return np.einsum("ij,jk,ik->i", jac, model.hess_inv, jac)


def test_sf_cb_uses_full_quadratic_form(gumbel_model):
    # The delta method variance must be J @ Sigma @ J.T, which counts
    # the off-diagonal covariance terms twice.
    model = gumbel_model
    t = np.array([7.0, 10.0, 13.0])

    def sf_func(params):
        return model.dist.sf(t - model.gamma, *params)

    var_R = delta_method_var(model, sf_func)
    R_hat = model.sf(t)
    diff = -z(0.05 / 2) * np.sqrt(var_R) / (R_hat * (1 - R_hat))
    lower = R_hat / (R_hat + (1 - R_hat) * np.exp(diff))
    upper = R_hat / (R_hat + (1 - R_hat) * np.exp(-diff))

    cb = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    assert np.allclose(cb[:, 0], lower)
    assert np.allclose(cb[:, 1], upper)


@pytest.mark.parametrize("on", ["sf", "ff", "Hf", "hf", "df"])
@pytest.mark.parametrize("fixture", ["gumbel_model", "weibull_model"])
def test_two_sided_bounds_bracket_point_estimate(on, fixture, request):
    model = request.getfixturevalue(fixture)
    t = np.linspace(6, 14, 9)
    cb = model.cb(t, on=on, bound="two-sided", alpha_ci=0.05)
    point = getattr(model, on)(t)
    assert np.all(cb[:, 0] < point)
    assert np.all(point < cb[:, 1])


@pytest.mark.parametrize("on", ["sf", "ff", "Hf", "hf", "df"])
@pytest.mark.parametrize("fixture", ["gumbel_model", "weibull_model"])
def test_one_sided_bounds_match_two_sided(on, fixture, request):
    # A one-sided bound at alpha is the corresponding side of the
    # two-sided bound at 2 * alpha.
    model = request.getfixturevalue(fixture)
    t = np.linspace(6, 14, 9)
    two_sided = model.cb(t, on=on, bound="two-sided", alpha_ci=0.1)
    lower = model.cb(t, on=on, bound="lower", alpha_ci=0.05)
    upper = model.cb(t, on=on, bound="upper", alpha_ci=0.05)
    assert np.allclose(two_sided[:, 0], lower)
    assert np.allclose(two_sided[:, 1], upper)


def test_hf_df_cb_match_delta_method(gumbel_model):
    # hf and df bounds come from the delta method applied directly to
    # each function, with a log transform to keep them positive.
    model = gumbel_model
    t = np.array([7.0, 10.0, 13.0])

    def df_func(params):
        return model.dist.df(t - model.gamma, *params)

    def hf_func(params):
        return model.dist.df(t - model.gamma, *params) / model.dist.sf(
            t - model.gamma, *params
        )

    for on, func in [("df", df_func), ("hf", hf_func)]:
        g_hat = func(np.array(model.params))
        var_g = delta_method_var(model, func)
        diff = -z(0.05 / 2) * np.sqrt(var_g) / g_hat
        expected = np.vstack([g_hat * np.exp(-diff), g_hat * np.exp(diff)]).T
        cb = model.cb(t, on=on, bound="two-sided", alpha_ci=0.05)
        assert np.allclose(cb, expected)


def test_lfp_cb_centred_on_full_sf():
    # For an LFP model the bounds must be centred on the full survival
    # function, 1 - p + p * R(t), and respect its 1 - p asymptote.
    np.random.seed(3)
    n = 500
    x = surv.Weibull.random(n, 10, 2)
    c = np.zeros(n)
    # Half the population never fails: censor it beyond all failures
    never = np.random.uniform(size=n) > 0.5
    x[never] = x.max() + 1
    c[never] = 1

    model = surv.Weibull.fit(x, c=c, lfp=True)
    assert model.p < 1

    t = np.linspace(5, 50, 20)
    cb = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    sf = model.sf(t)
    assert np.all(cb[:, 0] < sf)
    assert np.all(sf < cb[:, 1])
    # The bounds inherit the LFP scaling, so cannot fall below 1 - p
    assert np.all(cb[:, 0] >= 1 - model.p)
