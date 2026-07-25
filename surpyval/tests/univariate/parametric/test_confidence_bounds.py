"""
Tests for the confidence bounds of univariate parametric models.
"""

import numpy as np
import pytest
from autograd import hessian, jacobian
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


def test_fixed_param_has_no_variance():
    np.random.seed(5)
    x = surv.Weibull.random(500, 10, 2)
    model = surv.Weibull.fit(x, fixed={"beta": 2.0})

    # A fixed parameter is known, not estimated: zero variance and a
    # degenerate confidence interval
    assert np.all(model.hess_inv[1, :] == 0)
    assert np.all(model.hess_inv[:, 1] == 0)
    assert np.allclose(model.param_cb("beta"), [2.0, 2.0])

    # The free parameter's variance is conditional on the fixed value:
    # it matches a Hessian computed over alpha alone
    def nll(a):
        return model.dist._neg_ll_func(model.surv_data, a[0], 2.0, 0, 0, 1)

    h = hessian(nll)(np.array([model.params[0]]))
    assert np.allclose(model.hess_inv[0, 0], 1 / h[0, 0])

    t = np.linspace(6, 14, 9)
    cb = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    sf = model.sf(t)
    assert np.all(cb[:, 0] < sf) and np.all(sf < cb[:, 1])


def test_fixed_p_lfp():
    np.random.seed(6)
    n = 500
    x = surv.Weibull.random(n, 10, 2)
    c = np.zeros(n)
    never = np.random.uniform(size=n) > 0.5
    x[never] = x.max() + 1
    c[never] = 1

    model = surv.Weibull.fit(x, c=c, lfp=True, fixed={"p": 0.5})
    assert np.isclose(model.p, 0.5)
    assert np.all(model.cov_matrix[2, :] == 0)
    assert np.all(model.cov_matrix[:, 2] == 0)
    assert np.allclose(model.param_cb("p"), [model.p, model.p])

    t = np.linspace(5, 50, 10)
    cb = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    sf = model.sf(t)
    assert np.all(cb[:, 0] < sf) and np.all(sf < cb[:, 1])


def test_fixed_f0_zi():
    np.random.seed(7)
    x = surv.Weibull.random(500, 10, 2)
    x = np.concatenate([x, np.zeros(50)])

    model = surv.Weibull.fit(x, zi=True, fixed={"f0": 0.1})
    assert np.isclose(model.f0, 0.1)
    assert np.all(model.cov_matrix[2, :] == 0)
    assert np.all(model.cov_matrix[:, 2] == 0)
    assert np.allclose(model.param_cb("f0"), [model.f0, model.f0])


@pytest.mark.parametrize("dist_name", ["Weibull", "Gamma"])
def test_offset_model_cb(dist_name):
    # Weibull exercises the closed-form R_cb with an offset; Gamma the
    # generic delta-method path. gamma carries no variance.
    np.random.seed(8)
    dist = getattr(surv, dist_name)
    params = {"Weibull": (10, 2), "Gamma": (3, 2)}[dist_name]
    x = dist.random(500, *params) + 10
    model = dist.fit(x, offset=True)
    assert model.gamma > 5

    t = np.linspace(np.quantile(x, 0.1), np.quantile(x, 0.9), 8)
    for on in ["sf", "ff", "Hf", "hf", "df"]:
        cb = model.cb(t, on=on, bound="two-sided", alpha_ci=0.05)
        point = getattr(model, on)(t)
        assert np.all(cb[:, 0] < point)
        assert np.all(point < cb[:, 1])


def test_cb_round_trips_through_serialization(lfp_model):
    model = lfp_model
    t = np.linspace(5, 50, 10)
    expected = model.cb(t, on="sf", bound="two-sided", alpha_ci=0.05)
    restored = surv.Parametric.from_dict(model.to_dict())
    assert np.allclose(
        restored.cb(t, on="sf", bound="two-sided", alpha_ci=0.05), expected
    )


# ---------------------------------------------------------------------------
# Likelihood-ratio (profile) parameter bounds: param_cb(..., method="lr")
# ---------------------------------------------------------------------------


def _deviance_at(model, name, value):
    """2 * (profile neg-ll at ``value`` - neg-ll at the MLE)."""
    idx = model.dist.param_map[name]
    nll_hat = float(
        model.dist._neg_ll_func(
            model.surv_data, *model.params, model.gamma, model.f0, model.p
        )
    )
    return 2.0 * (model._profile_neg_ll(idx, value) - nll_hat)


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_lr_bound_deviance_hits_chi2_critical(weibull_model, name):
    # The defining property: at each LR bound the profile deviance equals the
    # chi-squared(1) critical value for the level.
    lo, hi = weibull_model.param_cb(name, method="lr", alpha_ci=0.05)
    crit = z(0.975) ** 2
    assert _deviance_at(weibull_model, name, lo) == pytest.approx(
        crit, abs=1e-4
    )
    assert _deviance_at(weibull_model, name, hi) == pytest.approx(
        crit, abs=1e-4
    )


@pytest.mark.parametrize("name", ["alpha", "beta"])
def test_lr_brackets_point_estimate(weibull_model, name):
    lo, hi = weibull_model.param_cb(name, method="lr")
    hat = weibull_model.params[weibull_model.dist.param_map[name]]
    assert lo < hat < hi


def test_lr_agrees_with_wald_in_large_samples():
    # With plenty of data the profile likelihood is close to quadratic, so the
    # LR interval and the Wald interval nearly coincide.
    np.random.seed(11)
    x = surv.Weibull.random(3000, 10.0, 2.0)
    m = surv.Weibull.fit(x)
    for name in ("alpha", "beta"):
        wald = m.param_cb(name, method="wald")
        lr = m.param_cb(name, method="lr")
        assert np.allclose(wald, lr, rtol=2e-2)


def test_lr_one_sided_inside_two_sided(weibull_model):
    lo2, hi2 = weibull_model.param_cb("beta", method="lr", bound="two-sided")
    upper = weibull_model.param_cb("beta", method="lr", bound="upper")[0]
    lower = weibull_model.param_cb("beta", method="lr", bound="lower")[0]
    # One-sided bounds use the smaller critical value, so they sit strictly
    # inside the corresponding two-sided bound.
    assert upper < hi2
    assert lower > lo2


def test_lr_single_parameter_distribution():
    np.random.seed(7)
    x = surv.Exponential.random(80, 0.1)
    m = surv.Exponential.fit(x)
    lo, hi = m.param_cb("failure_rate", method="lr")
    hat = m.params[0]
    assert lo < hat < hi
    crit = z(0.975) ** 2
    assert _deviance_at(m, "failure_rate", hi) == pytest.approx(crit, abs=1e-4)


def test_lr_handles_right_censoring():
    np.random.seed(9)
    x = surv.Weibull.random(120, 20.0, 1.5)
    c = (x > 25).astype(int)
    x = np.minimum(x, 25.0)
    m = surv.Weibull.fit(x=x, c=c)
    lo, hi = m.param_cb("beta", method="lr")
    assert lo < m.params[1] < hi


def test_lr_rejects_deserialised_model(weibull_model):
    # LR bounds need the data; a rehydrated model does not carry it.
    restored = surv.Parametric.from_dict(weibull_model.to_dict())
    with pytest.raises(ValueError, match="original data"):
        restored.param_cb("beta", method="lr")
    # Wald still works from the stored covariance.
    restored.param_cb("beta", method="wald")


def test_lr_rejects_offset_model():
    np.random.seed(13)
    x = surv.Weibull.random(200, 10.0, 2.0) + 5.0
    m = surv.Weibull.fit(x, offset=True)
    with pytest.raises(NotImplementedError):
        m.param_cb("alpha", method="lr")


def test_param_cb_rejects_unknown_method(weibull_model):
    with pytest.raises(ValueError, match="Unknown confidence-bound method"):
        weibull_model.param_cb("beta", method="bootstrap")
