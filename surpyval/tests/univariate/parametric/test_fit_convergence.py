"""Fits that failed silently or depended on the data's units (#392, #393,
#412, #427, #428).

Principles 12 and 13: a fit returns the optimum of its stated estimator,
and never fails silently; principle 6: the units do not matter.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import lognorm

import surpyval as sp
import surpyval as surv
from surpyval.tests._helpers import no_warnings
from surpyval.tests.conformance.registry import (
    ah_data,
    reg_data,
    stress_data,
    uni_data,
)
from surpyval.univariate.parametric.fitters import fallback_minimize

UNVERIFIED = "did not reach a verified maximum"
# The Beta4 case can also end on the edge where its likelihood is infinite,
# which has its own warning (#385); since every fit searches in units of its
# start (#366) it does on the data below.
UNBOUNDED = UNVERIFIED + "|No finite maximum"


# ---------------------------------------------------------------------------
# #427: a start far from the maximum
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dist, far",
    [
        # alpha x1e6: the ladder took Newton-CG's "success" at alpha 1.03e7,
        # beta 0.099 (ll -78.2 against -37.9)
        (sp.Weibull, [1.0315e7, 2.318]),
        # mu x1e6: BFGS stopped with "precision loss" at the maximum, and
        # TNC then "succeeded" from the start at mu, sigma 9.68, 9.62
        (sp.Normal, [9.08e6, 4.15]),
        # r x1e6: every rung stops in the Poisson limit (r -> inf), 1.2
        # below the maximum, at a point with a zero gradient
        (sp.NegativeBinomial, [4.13e6, 0.557]),
    ],
)
def test_a_far_start_reaches_the_maximum(dist, far):
    data = uni_data()
    if dist.discrete:
        data["x"] = np.round(data["x"])
    ref = no_warnings(dist.fit, **data)
    got = no_warnings(dist.fit, **data, init=far)
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-4)
    np.testing.assert_allclose(got.neg_ll(), ref.neg_ll(), rtol=1e-8)


def test_a_verified_first_rung_stops_the_ladder():
    # The check costs one gradient and one Hessian; a fit BFGS solves is
    # not sent down the rest of the ladder.
    model = no_warnings(sp.Weibull.fit, **uni_data())
    assert model.optimizer == "BFGS"


def test_precision_loss_at_the_maximum_is_accepted():
    # From mu x1e6 BFGS ends with "precision loss" at the maximum; that
    # answer is verified and kept, so the fit neither warns nor moves on
    # to a worse rung.
    data = uni_data()
    ref = sp.Normal.fit(**data)
    model = no_warnings(
        sp.Normal.fit, **data, init=[ref.params[0] * 1e6, 4.15]
    )
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-6)


def test_an_unverified_answer_warns():
    # The four-parameter beta's likelihood is unbounded (alpha < 1 with
    # a at the smallest value), so no search ends at a maximum here; the
    # fit used to return an edge point in silence.
    x = np.array([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8])
    with pytest.warns(UserWarning, match=UNBOUNDED):
        sp.Beta4.fit(x)


def test_the_uniform_is_not_warned_about_its_edges():
    # Its maximum is on the data's extremes by construction.
    x = np.linspace(2.0, 8.0, 50)
    model = no_warnings(sp.Uniform.fit, x, fixed={"a": 1.5})
    assert model.params[1] == pytest.approx(8.0, rel=1e-4)


# ---------------------------------------------------------------------------
# #412 and #393: truncation in the upper tail, and the units of truncated fits
# ---------------------------------------------------------------------------
def test_the_truncation_term_is_exact_where_F_rounds_to_one():
    x, tl = np.array([2.0, 3.0]), np.ones(2)
    data = sp.SurpyvalData(
        x=x, c=np.zeros(2, int), n=np.ones(2, int), tl=tl, tr=np.inf
    )
    exact = lognorm(0.6, scale=np.exp(-5.0))
    expected = np.sum(exact.logpdf(x) - exact.logsf(tl))  # -23.73
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ll = sp.LogNormal._log_likelihood(data, -5.0, 0.6, 0.0, 0.0, 1.0)
    # It was +inf, with a divide-by-zero warning
    np.testing.assert_allclose(float(ll), expected, rtol=1e-10)


def test_an_interval_in_the_upper_tail_is_exact():
    # (5, 7] under Weibull(0.02, 1.5): F is 1 at both ends in floating
    # point, and the log-likelihood was -inf. Exactly,
    # log(S(5) - S(7)) = log S(5) + log(1 - S(7) / S(5)).
    data = sp.SurpyvalData(xl=[5.0], xr=[7.0])
    ll = sp.Weibull._log_likelihood(data, 0.02, 1.5, 0.0, 0.0, 1.0)
    log_s5, log_s7 = -((5 / 0.02) ** 1.5), -((7 / 0.02) ** 1.5)
    expected = log_s5 + np.log1p(-np.exp(log_s7 - log_s5))
    np.testing.assert_allclose(float(ll), expected, rtol=1e-12)


_TRUNCATED = dict(
    x=np.array([10.5, 6.5, 2.5, 11.0]),
    c=np.array([1, 0, 1, -1]),
    tr=np.array([13.0, np.inf, np.inf, np.inf]),
)


@pytest.mark.parametrize("dist", [sp.Normal, sp.Gumbel, sp.Weibull])
@pytest.mark.parametrize("k", [7.3, 1e-3, 1e3])
def test_a_truncated_fit_is_unit_free(dist, k):
    # Normal: 8.536, 2.461 on the data (not the maximum: -4.034 against
    # -4.026) but 48.87, 23.04 on it x 7.3; Gumbel and Weibull failed with
    # "MLE Failed" at some scales. The NaN truncation term (#412) made
    # every gradient rung fail, leaving Powell to stop where it could.
    ref = no_warnings(dist.fit, **_TRUNCATED)
    got = no_warnings(
        dist.fit,
        x=_TRUNCATED["x"] * k,
        c=_TRUNCATED["c"],
        tr=_TRUNCATED["tr"] * k,
    )
    scaled = np.array(ref.params, dtype=float)
    scaled[0] *= k
    if dist is not sp.Weibull:
        scaled[1] *= k
    np.testing.assert_allclose(got.params, scaled, rtol=1e-4)
    # The log-likelihood moves by log k per exact observation only
    np.testing.assert_allclose(
        got.neg_ll(), ref.neg_ll() + np.log(k), rtol=1e-6
    )


def test_the_truncated_normal_fit_is_the_maximum():
    model = no_warnings(sp.Normal.fit, **_TRUNCATED)
    np.testing.assert_allclose(model.params, [8.7445, 2.4586], atol=1e-4)
    np.testing.assert_allclose(-model.neg_ll(), -4.0263, atol=1e-4)


# ---------------------------------------------------------------------------
# #392: data with no maximum
# ---------------------------------------------------------------------------
_NO_MAXIMUM = [
    # A spike at 0.5 explains a failure there and one before 1
    dict(x=[0.5, 1.0], c=[0, -1]),
    # (1, 3] and (2, 4] share (2, 3]
    dict(xl=[1.0, 2.0], xr=[3.0, 4.0]),
    # An exact value inside an interval, with a failure after 1
    dict(x=[[1.0, 3.0], [2.0, 2.0], [1.0, 1.0]], c=[2, 0, 1]),
]


@pytest.mark.parametrize("data", _NO_MAXIMUM)
@pytest.mark.parametrize(
    "dist", [sp.Weibull, sp.Normal, sp.Gamma, sp.LogNormal, sp.Gumbel]
)
def test_data_with_no_maximum_are_refused(dist, data):
    with pytest.raises(ValueError, match="no maximum"):
        dist.fit(**data)


def test_an_offset_makes_a_one_parameter_family_spike():
    with pytest.raises(ValueError, match="no maximum"):
        sp.Exponential.fit(x=[0.5, 1.0, 0.5], c=[0, -1, 0], offset=True)


@pytest.mark.parametrize("data", _NO_MAXIMUM[:2])
def test_a_one_parameter_family_has_a_maximum_there(data):
    # The Exponential cannot concentrate: its fit stands
    model = no_warnings(sp.Exponential.fit, **data)
    assert np.isfinite(model.params).all()


def test_a_fixed_parameter_restores_the_maximum():
    model = no_warnings(
        sp.Weibull.fit, x=[0.5, 1.0], c=[0, -1], fixed={"beta": 2}
    )
    assert np.isfinite(model.params).all()


def test_data_that_disagree_are_fitted():
    model = no_warnings(sp.Weibull.fit, xl=[1.0, 3.0], xr=[2.0, 4.0])
    assert np.isfinite(model.params).all()


# ---------------------------------------------------------------------------
# #428: accelerated life from a far start
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("life_model", ["InversePower", "Linear"])
def test_an_accelerated_life_fit_from_a_far_start(life_model):
    # InversePower from its first life parameter x1e6 ended 14.7 below
    # the maximum; Linear 0.13 below. No warning either way.
    data = stress_data(1)
    fitter = sp.AcceleratedLife(
        sp.Weibull, getattr(sp.life_models, life_model)
    )
    ref = no_warnings(fitter.fit, **data)
    far = np.array(ref.params, dtype=float)
    far[2] *= 1e6
    got = no_warnings(fitter.fit, **data, init=far)
    np.testing.assert_allclose(got._neg_ll, ref._neg_ll, rtol=1e-6)
    np.testing.assert_allclose(got.params, ref.params, rtol=1e-3)


# ---------------------------------------------------------------------------
# Additive hazards: TNC's result was never checked
# ---------------------------------------------------------------------------
def test_an_additive_hazards_fit_with_no_event_level_says_so():
    # A covariate that is 1 on exactly the censored rows: that group has
    # no events. LogNormalAH returned a coefficient of -6.2e8 (sf(2) inf)
    # in silence, then said the likelihood had no finite maximum (#392).
    # Kept inside its support (#828) the coefficient stops where that
    # group's H reaches 0, and the fit says it ended on that boundary.
    data = reg_data()
    data["Z"] = np.array(data["Z"], dtype=float)
    data["Z"][:, 0] = np.asarray(data["c"]) == 1
    with pytest.warns(UserWarning, match="ended on the boundary"):
        model = sp.LogNormalAH.fit(**data)
    assert np.isfinite(model.params).all()


def test_an_additive_hazards_fit_at_its_maximum_is_silent():
    # (on its own fixture, inside its support: see ah_data)
    model = no_warnings(sp.WeibullAH.fit, **ah_data())
    assert np.isfinite(model.params).all()


# ---------------------------------------------------------------------------
# ``fallback_minimize`` ends on Nelder-Mead.
# ---------------------------------------------------------------------------


def test_fallback_minimize_last_rung_is_nelder_mead():
    # A jacobian that is wrong everywhere makes BFGS fail and a zero
    # hessian skips Newton-CG, so the last rung must finish the job. The
    # objective has a kink at its minimum, where finite-difference BFGS
    # (the old last rung) loses precision and fails too.
    def fun(v):
        return abs(v[0] - 3.0) + abs(v[1] + 1.0)

    def bad_jac(v):
        return np.array([1.0, 1.0])

    def zero_hess(v):
        return np.zeros((2, 2))

    res = fallback_minimize(fun, np.array([0.0, 0.0]), (), bad_jac, zero_hess)
    assert res.success
    # Nelder-Mead reports no gradient; either BFGS would
    assert "jac" not in res
    np.testing.assert_allclose(res.x, [3.0, -1.0], atol=1e-3)


# ---------------------------------------------------------------------------
# MSE keeps the better optimum across its fallbacks.
# ---------------------------------------------------------------------------


def test_mse_keeps_the_better_optimum_across_its_fallbacks():
    np.random.seed(3)
    x = surv.Normal.random(200, 5.0, 2.0)
    cens = float(np.quantile(x, 0.85))
    c = (x > cens).astype(int)
    xc = np.where(x > cens, cens, x)
    base = surv.Normal.fit(xc, c, how="MSE").params
    small = surv.Normal.fit(xc * 1e-3, c, how="MSE").params
    assert small * 1e3 == pytest.approx(base, rel=1e-5)
