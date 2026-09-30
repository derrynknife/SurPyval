"""Confidence bounds on B-lives and the mean of a parametric model (#494).

``quantile_cb(p)`` and ``mean_cb()``, named as the nonparametric models'
are, with ``bound=`` and ``method=`` as ``cb`` has them. The Wald bound is
the delta method on the log of the quantile (mean) above the support's
start; the likelihood-ratio bound is the extreme of it over the
parameters' likelihood region.
"""

import numpy as np
import pytest
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

import surpyval as sp

# 20 units, the last five suspended at 70
X = np.array(
    [17.9, 28.9, 33.0, 41.5, 42.1, 45.6, 48.4, 51.8, 51.9, 54.1, 55.6]
    + [61.2, 67.8, 68.6, 68.9, 70, 70, 70, 70, 70]
)
C = np.array([0] * 15 + [1] * 5)

# R 4.x, survival::survreg(Surv(x, d) ~ 1, dist=...), then
# predict(type="uquantile", p=c(0.1, 0.5), se.fit=TRUE) and
# exp(fit -/+ qnorm(0.975) * se.fit)
R_SURVREG = {
    "Weibull": [[22.86730250, 44.99812722], [48.61939354, 67.15344941]],
    "LogNormal": [[23.97286350, 41.03024463], [45.10237789, 67.75192394]],
}


@pytest.mark.parametrize("name", sorted(R_SURVREG))
def test_wald_quantile_bounds_match_r_survreg(name):
    model = getattr(sp, name).fit(X, C)
    np.testing.assert_allclose(
        model.quantile_cb([0.1, 0.5]), R_SURVREG[name], rtol=1e-6
    )


def test_the_weibull_b10_bound_is_the_closed_form():
    # log t_p = log alpha + log(-log(1 - p)) / beta
    model = sp.Weibull.fit(X, C)
    alpha, beta = model.params
    w = np.log(-np.log(0.9))
    grad = np.array([1 / alpha, -w / beta**2])
    se = np.sqrt(grad @ model.hess_inv @ grad)
    t10 = alpha * np.exp(w / beta)
    lower = t10 * np.exp(-1.6448536 * se)
    assert model.quantile_cb(0.1, bound="lower") == pytest.approx(lower)


def test_the_lr_b10_bound_is_where_the_profile_deviance_is_chi2():
    model = sp.Weibull.fit(X, C)
    lo, hi = model.quantile_cb(0.1, method="lr")
    w = -np.log(0.9)

    def deviance(t10):
        # the Weibull with B10 = t10, the shape re-optimised
        def nll(log_beta):
            beta = np.exp(log_beta)
            fixed = {"alpha": t10 / w ** (1 / beta), "beta": beta}
            return sp.Weibull.fit(X, C, fixed=fixed)._neg_ll

        res = minimize_scalar(
            nll, bounds=(-3, 3), method="bounded", options={"xatol": 1e-10}
        )
        return 2 * (res.fun - model._neg_ll)

    crit = chi2.ppf(0.95, 1)
    assert deviance(lo) == pytest.approx(crit, abs=1e-4)
    assert deviance(hi) == pytest.approx(crit, abs=1e-4)
    assert lo < model.qf(0.1) < hi


def test_shapes_and_sides():
    model = sp.Weibull.fit(X, C)
    assert model.quantile_cb(0.1).shape == (2,)
    assert model.quantile_cb([0.1, 0.5]).shape == (2, 2)
    assert np.ndim(model.quantile_cb(0.1, bound="lower")) == 0
    two = model.quantile_cb(0.1, alpha_ci=0.1)
    assert model.quantile_cb(0.1, bound="lower") == pytest.approx(two[0])
    assert model.quantile_cb(0.1, bound="upper") == pytest.approx(two[1])
    mean = model.mean_cb(alpha_ci=0.1)
    assert mean.shape == (2,)
    assert mean[0] < model.mean() < mean[1]
    assert model.mean_cb(bound="lower") == pytest.approx(mean[0])


def test_mean_wald_bound_is_the_delta_method_on_the_log_mean():
    model = sp.Exponential.fit(X, C)
    # mean = 1 / rate: se(log mean) = se(rate) / rate
    rate = model.params[0]
    se = np.sqrt(model.hess_inv[0, 0]) / rate
    want = np.exp(np.log(1 / rate) + np.array([-1, 1]) * 1.959964 * se)
    np.testing.assert_allclose(model.mean_cb(), want, rtol=1e-5)


def test_offset_quantile_bounds_are_above_the_offset():
    model = sp.Weibull.fit(X + 100, C, offset=True, how="MLE")
    lo, hi = model.quantile_cb(0.01)
    assert model.gamma < lo < model.qf(0.01) < hi


def test_an_lfp_quantile_past_p_and_mean_are_infinite():
    rng = np.random.default_rng(0)
    x = np.r_[sp.Weibull.random(20, 10, 2, random_state=rng), [30] * 30]
    c = np.r_[np.zeros(20), np.ones(30)]
    model = sp.Weibull.fit(x, c, lfp=True)
    assert model.p < 0.6
    assert np.all(np.isinf(model.quantile_cb(0.9)))
    assert np.all(np.isinf(model.mean_cb()))
    lo, hi = model.quantile_cb(0.1)
    assert lo < model.qf(0.1) < hi


def test_a_discrete_quantile_inverts_the_band_on_ff():
    model = sp.Geometric.fit(sp.Geometric.random(40, 0.2, random_state=1))
    lo, hi = model.quantile_cb(0.5)
    band = model.cb(np.arange(0, 40), on="ff")
    assert lo == np.argmax(band[:, 1] >= 0.5)
    assert hi == np.argmax(band[:, 0] >= 0.5)
    assert lo <= model.qf(0.5) <= hi


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"p": 0.0}, "'p' must be in"),
        ({"p": 1.0}, "'p' must be in"),
        ({"p": 0.1, "bound": "both"}, "bound must be"),
        ({"p": 0.1, "method": "boot"}, "Unknown confidence-bound"),
        ({"p": 0.1, "alpha_ci": 1.2}, "'alpha_ci'"),
    ],
)
def test_invalid_arguments_raise(kwargs, match):
    model = sp.Weibull.fit(X, C)
    with pytest.raises(ValueError, match=match):
        model.quantile_cb(**kwargs)


def test_only_mle_has_bounds():
    model = sp.Weibull.fit(X, C, how="MPS")
    with pytest.raises(ValueError, match="Only MLE"):
        model.quantile_cb(0.1)
    with pytest.raises(ValueError, match="Only MLE"):
        model.mean_cb()


# Coverage: calibration/test_coverage_parametric.py (nightly) checks the
# Wald quantile_cb(0.1) and mean_cb at n = 100 (the LR ones are too slow).
