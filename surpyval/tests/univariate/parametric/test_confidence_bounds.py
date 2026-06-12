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


@pytest.fixture(scope="module")
def lfp_model():
    np.random.seed(3)
    n = 500
    x = surv.Weibull.random(n, 10, 2)
    c = np.zeros(n)
    # Half the population never fails: censor it beyond all failures
    never = np.random.uniform(size=n) > 0.5
    x[never] = x.max() + 1
    c[never] = 1
    return surv.Weibull.fit(x, c=c, lfp=True)


def test_lfp_cb_centred_on_full_sf(lfp_model):
    # For an LFP model the bounds must be centred on the full survival
    # function, 1 - p + p * R(t).
    model = lfp_model
    assert model.p < 1

    t = np.linspace(5, 50, 20)
    cb = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    sf = model.sf(t)
    assert np.all(cb[:, 0] < sf)
    assert np.all(sf < cb[:, 1])
    assert np.all(cb > 0)
    assert np.all(cb < 1)
    # p is estimated, not known, so at large t the lower bound falls
    # below the fitted 1 - p asymptote
    assert cb[-1, 0] < 1 - model.p


def test_cov_matrix_extends_hess_inv(lfp_model):
    # The full covariance covers (alpha, beta, p); its parameter block
    # is exactly hess_inv, and the estimated p has variance.
    model = lfp_model
    assert model.cov_matrix.shape == (3, 3)
    assert np.allclose(model.cov_matrix[:2, :2], model.hess_inv)
    assert model.cov_matrix[2, 2] > 0


def test_lfp_sf_cb_includes_p_variance(lfp_model):
    # The sf bounds must come from the delta method over the extended
    # vector (alpha, beta, p) applied to the full survival function.
    model = lfp_model
    t = np.array([5.0, 15.0, 40.0])

    def sf_func(phi):
        *params, p = phi
        return 1 - p + p * model.dist.sf(t - model.gamma, *params)

    phi_hat = np.array([*model.params, model.p])
    jac = np.atleast_2d(jacobian(sf_func)(phi_hat))
    var_R = np.einsum("ij,jk,ik->i", jac, model.cov_matrix, jac)
    R_hat = model.sf(t)
    diff = -z(0.05 / 2) * np.sqrt(var_R) / (R_hat * (1 - R_hat))
    lower = R_hat / (R_hat + (1 - R_hat) * np.exp(diff))
    upper = R_hat / (R_hat + (1 - R_hat) * np.exp(-diff))

    cb = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    assert np.allclose(cb[:, 0], lower)
    assert np.allclose(cb[:, 1], upper)

    # Treating p as fixed must give strictly narrower bounds
    var_fixed = np.einsum(
        "ij,jk,ik->i", jac[:, :2], model.hess_inv, jac[:, :2]
    )
    assert np.all(var_R > var_fixed)


def test_param_cb_p(lfp_model):
    model = lfp_model
    lower, upper = model.param_cb("p", alpha_ci=0.05)
    assert 0 < lower < model.p < upper < 1
    assert np.allclose(
        model.param_cb("p", alpha_ci=0.025, bound="lower"), lower
    )
    assert np.allclose(
        model.param_cb("p", alpha_ci=0.025, bound="upper"), upper
    )


def test_zi_lfp_cb():
    # Bounds for a model with both zero-inflation and a limited failure
    # population are computed over (alpha, beta, p, f0) and stay valid.
    np.random.seed(4)
    n = 1000
    x = surv.Weibull.random(n, 10, 2)
    c = np.concatenate((np.zeros(n), np.zeros(100), np.ones(100)))
    x = np.concatenate((x, np.zeros(100), x.max() * np.ones(100) + 1))

    model = surv.Weibull.fit(x, c=c, zi=True, lfp=True)
    assert model.cov_matrix.shape == (4, 4)
    assert model.cov_matrix[2, 2] > 0
    assert model.cov_matrix[3, 3] > 0

    t = np.linspace(1, 40, 10)
    sf = model.sf(t)
    for on in ["sf", "ff", "Hf", "hf", "df"]:
        cb = model.cb(t, on=on, bound="two-sided", alpha_ci=0.05)
        point = getattr(model, on)(t)
        assert np.all(cb[:, 0] < point)
        assert np.all(point < cb[:, 1])

    cb = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    assert np.all(cb > 0)
    assert np.all(cb < 1)
    assert np.all(cb[:, 0] < sf) and np.all(sf < cb[:, 1])

    f0_lower, f0_upper = model.param_cb("f0", alpha_ci=0.05)
    assert 0 < f0_lower < model.f0 < f0_upper < 1


def test_cb_round_trips_through_serialization(lfp_model):
    model = lfp_model
    t = np.linspace(5, 50, 10)
    expected = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    restored = surv.Parametric.from_dict(model.to_dict())
    assert np.allclose(
        restored.cb(t, on="sf", bound="two-sided", alpha_ci=0.05), expected
    )
