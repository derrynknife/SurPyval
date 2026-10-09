"""Parameter bounds, default starting points, and restored models."""

import json
import warnings

import numpy as np
import pytest
from autograd import numpy as anp

import surpyval as surv
from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.univariate.parametric.parametric_fitter import (
    OutsideSupportError,
)


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
    assert model.lfp_p == pytest.approx(0.0067, abs=0.0005)
    assert model.neg_ll() == pytest.approx(293.03, abs=0.01)


def test_a_restored_model_keeps_its_likelihood():
    rng = np.random.default_rng(2)
    model = surv.Weibull.fit(rng.weibull(2.0, 50) * 10)
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.neg_ll() == model.neg_ll()
    assert restored.aic() == model.aic()
    # the dict stores the criteria's sample size too
    assert restored.bic() == model.bic()
    # the plotting points need the data; plot() itself draws the curve
    # alone for a model without data (#485)
    with pytest.raises(ValueError, match="with_data=True"):
        restored.get_plot_data()
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


# ---------------------------------------------------------------------------
# Data outside the support; ``from_params`` validation; a
# restored model's plot data.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"zi": True}, {"lfp": True}, {"how": "MPS"}, {"how": "MSE"}],
)
@pytest.mark.parametrize("dist", [W, surv.Exponential, surv.Gamma])
def test_611_right_censoring_below_the_support_is_refused(dist, kwargs):
    # As the regressions on the distribution refuse it (#565), with the
    # same error: no unit can be censored before the support starts. The
    # univariate fit took it as a suspension carrying no information.
    np.random.seed(2)
    x = np.append(W.random(100, 10, 2), -1.0)
    c = np.append(np.zeros(100), 1)
    with pytest.raises(OutsideSupportError) as err:
        dist.fit(x, c, **kwargs)
    assert "a unit cannot be censored before 0" in str(err.value)


def test_611_univariate_and_regression_refuse_alike():
    # One wording for both (principle 21).
    x, c = [-1.0, 2, 3, 4, 5, 6], [1, 0, 0, 0, 0, 0]
    with pytest.raises(OutsideSupportError) as uni:
        W.fit(x, c)
    with pytest.raises(OutsideSupportError) as reg:
        surv.WeibullPH.fit(x, [[0], [1], [0], [1], [0], [1]], c=c)
    assert str(uni.value) == str(reg.value)


def test_611_right_censoring_at_the_support_start_is_kept():
    # A suspension at 0 carries no information (R = 1) and is accepted
    # as before, with the fit of the data without it.
    np.random.seed(2)
    x = W.random(100, 10, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        model = W.fit(np.append(x, 0.0), np.append(np.zeros(100), 1))
    assert model.params == pytest.approx(W.fit(x).params, rel=1e-6)


def test_611_offset_fit_takes_a_negative_censored_time():
    # The offset moves the support to (gamma, inf), so a negative time is
    # not refused; a right-censored one does not cap gamma either (S = 1
    # below it, #633), which stays below the first failure.
    np.random.seed(2)
    x = np.append(W.random(100, 10, 2), -1.0)
    c = np.append(np.zeros(100), 1)
    model = W.fit(x, c, offset=True)
    assert model.gamma < x[:100].min()
    assert np.isfinite(model.neg_ll())


def test_611_mixture_and_dataframe_fits_refuse_it_too():
    import pandas as pd

    x = np.append(W.random(60, 10, 2, random_state=1), -1.0)
    c = np.append(np.zeros(60), 1)
    with pytest.raises(OutsideSupportError):
        surv.MixtureModel.fit(x, c, dist=W, m=2)
    df = pd.DataFrame({"x": x, "c": c})
    with pytest.raises(OutsideSupportError):
        W.fit_from_df(df, x_col="x", c_col="c")


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda: W.from_params([10, 2], lfp_p=1.5), "must be in"),
        (lambda: W.from_params([10, 2], f0=-0.1), "must be in"),
        (lambda: W.from_params([10, 2], lfp_p=0.3, f0=0.4), "less than lfp_p"),
        (lambda: surv.Normal.from_params([1, 2], f0=0.1), "starting at 0"),
        (lambda: surv.Beta4.from_params([2, 3, 5, 1]), "a < b"),
        (lambda: surv.Uniform.from_params([4, 1]), "a < b"),
    ],
)
def test_from_params_validates(call, match):
    with pytest.raises(ValueError, match=match):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: surv.Beta.fit([0.2, 0.3, 0.5, 1.5], [0, 0, 0, 1]),
        lambda: surv.Poisson.fit([-2, 1, 2, 3]),
    ],
)
def test_out_of_support_data_is_refused(call):
    with pytest.raises(ValueError, match="outside the support"):
        call()


def test_support_message_is_one_clear_line():
    with pytest.raises(ValueError) as err:
        W.fit([0.0, 1, 2, 3])
    message = str(err.value)
    assert message.startswith("Some of your data")
    assert "(0, inf)" in message and "[0, inf]" not in message


def test_restored_model_get_plot_data_needs_the_data():
    np.random.seed(1)
    model = W.fit(W.random(30, 10, 3))
    with pytest.raises(ValueError, match="needs the data"):
        surv.from_dict(model.to_dict()).get_plot_data()
