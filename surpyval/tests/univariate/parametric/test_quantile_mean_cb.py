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
    assert model.lfp_p < 0.6
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


# #591: the discrete bound's search asks for the band at blocks of counts,
# in one vectorised call each, rather than once per count.


def _bisection_as_before(values, start, level):
    # The search ``_quantile_cb_discrete`` had, one count at a time
    asked = []

    def at(k):
        asked.append(k)
        return values(np.array([k]))[0] >= level

    if at(start):
        return start, asked
    lo, step = start, 1.0
    while not at(start + step):
        lo = start + step
        step *= 2
        if step > 2.0**40:
            return np.inf, asked
    hi = start + step
    while hi - lo > 1:
        mid = np.floor((lo + hi) / 2)
        if at(mid):
            hi = mid
        else:
            lo = mid
    return hi, asked


def test_591_the_block_search_is_the_bisection():
    from surpyval.univariate.parametric.parametric import _first_reaching

    rng = np.random.default_rng(591)
    for _ in range(300):
        start = float(rng.integers(0, 2))
        # A rising step, sometimes with a dip (a Wald band can turn back
        # in a tail), sometimes never reaching the level
        edge = 10.0 ** rng.uniform(0, 13)
        dip = rng.uniform(0, edge) if rng.random() < 0.3 else -1.0

        def values(ks, edge=edge, dip=dip):
            ks = np.asarray(ks, dtype=float)
            out = np.where(ks >= edge, 0.9, 0.1)
            return np.where(np.abs(ks - dip) < 0.25 * edge, 0.1, out)

        want, asked_before = _bisection_as_before(values, start, 0.5)
        for block in (1, 3, 7, 1023):
            asked = []

            def recorded(ks, values=values):
                asked.extend(np.asarray(ks).tolist())
                assert len(ks) <= block
                return values(ks)

            got = _first_reaching(recorded, start, 0.5, block)
            assert got == want
            if block == 1:
                assert asked == asked_before


def test_591_discrete_quantile_cb_evaluates_the_band_in_blocks(monkeypatch):
    from surpyval.tests.conformance.registry import CASE_BY_NAME
    from surpyval.univariate.parametric.parametric import Parametric

    case = CASE_BY_NAME["BetaGeometric"]
    model = case.fit(case.data())
    calls = []
    real = Parametric._cb_sf_bound

    def counted(self, x, *args, **kwargs):
        calls.append(np.size(x))
        return real(self, x, *args, **kwargs)

    monkeypatch.setattr(Parametric, "_cb_sf_bound", counted)
    got = model.quantile_cb([0.1, 0.5, 0.9])
    # The bounds of the search one count at a time (v0.22)
    np.testing.assert_array_equal(got, [[1, 1], [1, 3], [6, np.inf]])
    # Six searches (two ends of three bounds), each the doubling's 42
    # counts in one call and the bisection's next ten levels in another:
    # 8 calls. One count at a time it was 60, 42 of them for the upper
    # bound at 0.9, whose band never reaches it.
    assert len(calls) <= 12
    assert max(calls) == 42


def test_591_a_custom_distribution_is_differentiated_a_point_at_a_time(
    monkeypatch,
):
    # The one-pass gradient assumes functions that take the parameters
    # point by point; a user's CustomDistribution may not, so a Discretize
    # of one keeps the gradient a point at a time.
    from surpyval.univariate.parametric.parametric import Parametric

    def weibull_hf(x, lam, beta):
        return (beta / lam) * (x / lam) ** (beta - 1)

    custom = sp.CustomDistribution(
        "q591_weibull",
        weibull_hf,
        ["lam", "beta"],
        ((0, None), (0, None)),
        (0, np.inf),
    )
    rng = np.random.default_rng(3)
    k = np.ceil(rng.weibull(1.5, 80) * 6.0)
    model = sp.Discretize(custom).fit(k)
    seen = []
    real = Parametric._cb_delta_var

    def spy(self, func, ctx, n_points=None):
        seen.append(n_points)
        return real(self, func, ctx, n_points)

    monkeypatch.setattr(Parametric, "_cb_delta_var", spy)
    got = model.quantile_cb([0.2, 0.8])
    assert seen and all(n is None for n in seen)
    assert np.all(got[:, 0] <= got[:, 1])


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"p": 0.0}, "'p' must be in"),
        ({"p": 1.0}, "'p' must be in"),
        ({"p": 0.1, "bound": "both"}, "'bound' must be one of"),
        ({"p": 0.1, "method": "boot"}, "'method' must be one of"),
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
