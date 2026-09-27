"""Parameter bounds, default starting points, and restored models."""

import json

import numpy as np
import pytest
from autograd import numpy as anp

import surpyval as surv
from surpyval.univariate.parametric.fitters import bounds_convert


def test_a_finite_two_sided_bound_is_enforced():
    # Only (0, 1) had a bounded transform; any other finite pair fell
    # through to the identity and the bound was silently ignored.
    transform, inverse, *_ = bounds_convert(
        np.array([3.0]), ((2.0, 5.0),), None, {"k": 0}
    )
    for u in (-50.0, -1.0, 0.0, 1.0, 50.0):
        v = inverse(np.array([u]))[0]
        assert 2.0 <= v <= 5.0
    assert inverse(transform(np.array([3.7])))[0] == pytest.approx(3.7)
    # (0, 1) keeps its own, unchanged map
    t01, i01, *_ = bounds_convert(np.array([0.3]), ((0, 1),), None, {"p": 0})
    assert t01(np.array([0.3]))[0] == pytest.approx(10 * np.arctanh(-0.4))


def test_a_custom_distribution_respects_a_two_sided_bound():
    # an exponential whose rate is bounded to (0.5, 1): data with rate 0.1
    # would pull it far below 0.5 if the bound were ignored
    def Hf(x, *params):
        return params[0] * x

    bounded = surv.CustomDistribution(
        "BoundedExponential", Hf, ["rate"], ((0.5, 1.0),), (0, anp.inf)
    )
    rng = np.random.default_rng(0)
    model = bounded.fit(rng.exponential(10.0, 200))
    assert 0.5 <= model.params[0] <= 1.0
    assert model.params[0] == pytest.approx(0.5, abs=0.01)


def _gompertz_makeham():
    def Hf(x, *params):
        return params[0] * x + (params[1] / params[2]) * (
            anp.exp(params[2] * x) - 1
        )

    return surv.CustomDistribution(
        "GompertzMakeham",
        Hf,
        ["lambda", "alpha", "beta"],
        ((0, None), (0, None), (0, None)),
        (0, anp.inf),
    )


def test_custom_distribution_default_start_finds_the_data_scale():
    # Started at (1, 1, 1) the likelihood of human lifetimes (~70 years)
    # was ~1e18 and flat to machine precision, and the fit "succeeded"
    # there after one step. The grid of starting magnitudes finds the
    # optimum that a hand-tuned init reaches.
    rng = np.random.default_rng(1)
    lam, alpha, beta = 0.68e-3, 28.7e-6, 102.3e-3
    # inverse-transform sampling by bisection on H(x) = -log U
    u = rng.uniform(size=2000)
    target = -np.log(u)
    lo, hi = np.zeros_like(u), np.full_like(u, 200.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        H = lam * mid + alpha / beta * (np.exp(beta * mid) - 1)
        lo, hi = np.where(H < target, mid, lo), np.where(H < target, hi, mid)
    x = (lo + hi) / 2
    GM = _gompertz_makeham()
    default = GM.fit(x)
    tuned = GM.fit(x, init=[1e-3, 1e-4, 0.1])
    assert default.neg_ll() == pytest.approx(tuned.neg_ll(), rel=1e-6)
    assert default.params == pytest.approx(tuned.params, rel=1e-3)


def test_lfp_default_start_reaches_the_better_optimum():
    # Meeker's limited-failure-population data: the old default start led
    # to p = 0.116 (neg_ll 302.9) while the optimum is p = 0.0067 (293.0)
    f = [0.1, 0.1, 0.15, 0.6, 0.8, 0.8, 1.2, 2.5, 3.0, 4.0, 4.0, 6.0]
    f += [10.0, 10.0, 12.5, 20.0, 20.0, 43.0, 43.0, 48.0, 48.0, 54.0]
    f += [74.0, 84.0, 94.0, 168.0, 263.0, 593.0]
    x, c, n, _ = surv.fs_to_xcnt(f, [1370.0] * 4128)
    model = surv.Weibull.fit(x, c, n, lfp=True)
    assert model.p == pytest.approx(0.0067, abs=0.0005)
    assert model.neg_ll() == pytest.approx(293.03, abs=0.01)


def test_a_restored_model_keeps_its_likelihood():
    rng = np.random.default_rng(2)
    model = surv.Weibull.fit(rng.weibull(2.0, 50) * 10)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.neg_ll() == model.neg_ll()
    assert restored.aic() == model.aic()
    # the dict stores the criteria's sample size too
    assert restored.bic() == model.bic()
    with pytest.raises(ValueError, match="with_data=True"):
        restored.plot()
    with_data = surv.from_dict(
        json.loads(json.dumps(model.to_dict(with_data=True)))
    )
    assert with_data.bic() == pytest.approx(model.bic())
    # a model built from parameters still has no likelihood
    with pytest.raises(ValueError, match="fit with data"):
        surv.Weibull.from_params([10.0, 2.0]).neg_ll()


def test_fit_best_says_why_when_no_candidate_has_a_finite_aic_c():
    # three failures and many survivors: d = 3 <= k + 1 for every
    # two-parameter candidate, so AIC_c is undefined for all of them
    x = [1.0, 2.0, 3.0] + [10.0] * 20
    c = [0, 0, 0] + [1] * 20
    with pytest.raises(ValueError, match="metric='aic'"):
        surv.fit_best(x, c=c, metric="aic_c", include=["Weibull"])
    assert surv.fit_best(x, c=c, metric="aic", include=["Weibull"]) is not None
