"""Turnbull estimator: truncation correctness, convergence controls and
memory-efficient EM (the interval-censoring results against R's icenReg
live in test_np.py)."""

import numpy as np
import pytest

import surpyval
import surpyval as sp
import surpyval as surv
from surpyval import KaplanMeier, Turnbull
from surpyval.tests._helpers import (
    TURNBULL_MIXED_CENSORING,
    fit_turnbull_quietly,
    no_warnings,
    quietly,
)
from surpyval.univariate import nonparametric as nonp
from surpyval.univariate.nonparametric import plotting_positions


def _left_truncated_sample(n=800, seed=5):
    np.random.seed(seed)
    x = surpyval.Weibull.random(n, 10, 2)
    tl = surpyval.Uniform.random(n, 0, 8)
    keep = x > tl
    return x[keep], tl[keep]


def test_turnbull_left_truncation_matches_kaplan_meier():
    # For exactly observed, left-truncated data the Turnbull NPMLE is the
    # Kaplan-Meier estimator with delayed entry, which surpyval's KM
    # already handles through the risk set. Before the ghost step was
    # fixed, Turnbull silently ignored truncation entirely and returned
    # the (biased) untruncated estimate here.
    x, tl = _left_truncated_sample()
    km = surpyval.KaplanMeier.fit(x, tl=tl)
    tb = surpyval.Turnbull.fit(
        x, tl=tl, turnbull_estimator="Kaplan-Meier", max_iter=20_000
    )
    grid = np.array([4.0, 6.0, 8.0, 10.0, 12.0])
    assert np.allclose(tb.sf(grid), km.sf(grid), atol=5e-3)

    # And the biased no-truncation estimate is measurably different, so
    # the assertion above genuinely discriminates.
    untruncated = surpyval.KaplanMeier.fit(x)
    assert np.max(np.abs(untruncated.sf(grid) - km.sf(grid))) > 0.02


def test_turnbull_right_truncation_recovers_cdf_shape():
    # Right-truncated data identifies F up to a scale, so compare the
    # fitted CDF's shape against the true Weibull CDF conditioned on the
    # truncation horizon.
    np.random.seed(7)
    x = surpyval.Weibull.random(4000, 10, 2)
    keep = x < 14.0
    tb = surpyval.Turnbull.fit(
        x[keep], tr=14.0, turnbull_estimator="Kaplan-Meier"
    )
    grid = np.array([4.0, 6.0, 8.0, 10.0, 12.0])
    F_true = 1 - np.exp(-((grid / 10.0) ** 2))
    ratio_hat = tb.ff(grid) / tb.ff(12.0)
    ratio_true = F_true / F_true[-1]
    assert np.allclose(ratio_hat, ratio_true, atol=0.05)


def test_turnbull_interval_censored_with_left_truncation():
    # Interval censoring and truncation together must run and produce a
    # monotone survival curve over positive masses.
    np.random.seed(11)
    x = surpyval.Weibull.random(500, 10, 2)
    tl = surpyval.Uniform.random(500, 0, 6)
    keep = np.floor(x) > tl
    xl = np.floor(x[keep])
    xr = xl + 1.0
    model = surpyval.Turnbull.fit(xl=xl, xr=xr, tl=tl[keep])
    R = model.R[np.isfinite(model.R)]
    assert np.all(np.diff(R) <= 1e-12)
    assert R[0] <= 1.0 + 1e-12 and R[-1] >= -1e-12


def test_turnbull_max_iter_warns_and_reports():
    x, tl = _left_truncated_sample()
    with pytest.warns(UserWarning, match="did not converge"):
        model = surpyval.Turnbull.fit(x, tl=tl, max_iter=5)
    assert model.converged is False
    assert model.iters == 5


def test_turnbull_tol_controls_iterations():
    # A looser tolerance must converge in no more iterations than a tight
    # one, without warning on well-behaved data.
    x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10]])
    loose = surpyval.Turnbull.fit(x, tol=1e-4)
    tight = surpyval.Turnbull.fit(x, tol=1e-12, max_iter=10_000)
    assert loose.converged and tight.converged
    assert loose.iters <= tight.iters
    # Both agree to well within the loose tolerance's practical effect.
    assert np.allclose(loose.R, tight.R, atol=1e-3, equal_nan=True)


def test_turnbull_equals_kaplan_meier_with_right_censoring():
    # Classical identity: on exactly observed + right-censored data the
    # Turnbull NPMLE *is* the Kaplan-Meier estimator. This exercises the
    # zero-width Turnbull intervals (every exact time is a duplicated
    # bound carrying a point mass) against surpyval's R-validated KM.
    np.random.seed(3)
    x = np.round(surpyval.Weibull.random(300, 10, 2), 2)
    c = (np.random.uniform(size=300) < 0.3).astype(int)
    km = surpyval.KaplanMeier.fit(x, c=c)
    tb = surpyval.Turnbull.fit(x, c=c, turnbull_estimator="Kaplan-Meier")
    grid = np.quantile(x, [0.1, 0.3, 0.5, 0.7, 0.9])
    assert np.allclose(tb.sf(grid), km.sf(grid), atol=1e-12)


def test_turnbull_all_censoring_types_together():
    # Exact (0), right (1), left (-1) and interval (2) censoring in one
    # dataset: the fit must run, keep the duplicated zero-width bounds
    # for every exact time, and return a monotone survival curve.
    xl = [2.0, 4.0, 5.0, 3.0, 1.0, 4.0, 2.0, 6.0, 7.0, 2.5]
    xr = [2.0, 4.0, 5.0, 3.0, 1.0, 6.0, 5.0, 6.0, 7.0, 2.5]
    c = np.array([0, 0, 1, -1, 0, 2, 2, 1, 0, 0])
    n = np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1])
    model = surpyval.Turnbull.fit(xl=xl, xr=xr, c=c, n=n)
    for v in (1.0, 2.0, 2.5, 4.0, 7.0):
        assert (model.bounds == v).sum() == 2
    R = model.R[np.isfinite(model.R)]
    assert np.all(np.diff(R) <= 1e-12)


def test_turnbull_against_lifelines_npmle_reference():
    # Overlapping intervals whose endpoints are all distinct, so the
    # NPMLE does not depend on open/closed interval conventions (which
    # differ between packages: surpyval uses [l, r), icenReg (l, r],
    # lifelines [l, r]). Reference survival values computed with
    # lifelines 0.30.3 ``KaplanMeierFitter.fit_interval_censoring``
    # (its residual ~1e-4 unconverged mass is inside the tolerance).
    left = [0.5, 1.2, 2.1, 1.8, 3.4, 4.2, 5.1, 4.8, 6.3, 7.2, 2.7, 8.4]
    right = [2.4, 3.1, 4.5, 6.1, 5.6, 6.8, 7.9, 9.2, 8.8, 9.9, 10.4, 11.3]
    model = surpyval.Turnbull.fit(
        xl=left,
        xr=right,
        turnbull_estimator="Kaplan-Meier",
        tol=1e-12,
        max_iter=100_000,
    )
    probe = [2.5, 3.2, 4.6, 5.7, 6.9, 8.0]
    reference = [0.714780, 0.714780, 0.714622, 0.326064, 0.326064, 0.325946]
    assert np.allclose(model.sf(probe), reference, atol=1e-3)


def test_turnbull_truncated_confidence_bounds_match_kaplan_meier():
    # The estimation ladder's ghost events make the truncated *estimate*
    # correct, but they are not observations: a variance computed from
    # the ghost-inflated risk set was up to ~33% too narrow. The variance
    # now comes from an observed-information ladder that reduces to the
    # Kaplan-Meier delayed-entry risk set for exactly observed
    # left-truncated data, so the bounds must match KM's.
    x, tl = _left_truncated_sample()
    km = surpyval.KaplanMeier.fit(x, tl=tl)
    tb = surpyval.Turnbull.fit(
        x, tl=tl, turnbull_estimator="Kaplan-Meier", max_iter=20_000
    )
    grid = np.array([6.0, 8.0, 10.0, 12.0])
    assert np.allclose(tb.R_cb(grid), km.R_cb(grid), atol=2e-3)

    # Untruncated fits keep using the estimation ladder for the variance
    # (no separate attributes appear).
    plain = surpyval.Turnbull.fit(x)
    assert not hasattr(plain, "var_r") and not hasattr(plain, "var_d")


def test_turnbull_estimator_options_on_fractional_ladder():
    # The Kaplan-Meier / Nelson-Aalen / Fleming-Harrington choice is
    # applied to the EM's expected-count ladder inside every iteration,
    # so the three options must produce distinct, correctly ordered
    # survival curves (NA >= FH >= KM), each fitted on fractional
    # (partial) event counts.
    xl = [2.0, 4.0, 5.0, 3.0, 1.0, 4.0, 2.0, 6.0, 7.0, 2.5, 3.6, 5.5]
    xr = [2.0, 4.0, 5.0, 3.0, 1.0, 6.0, 5.0, 6.0, 7.0, 2.5, 8.0, 9.0]
    c = np.array([0, 0, 1, -1, 0, 2, 2, 1, 0, 0, 2, 2])
    n = np.array([1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 3, 1])
    models = {
        est: surpyval.Turnbull.fit(
            xl=xl, xr=xr, c=c, n=n, turnbull_estimator=est
        )
        for est in ("Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington")
    }
    # The EM distributes events across intervals, so the death ladder
    # must contain genuinely fractional counts.
    d = models["Kaplan-Meier"].d
    fractional = d[(d > 1e-9) & (np.abs(d - np.round(d)) > 1e-6)]
    assert fractional.size > 0

    grid = [2.6, 4.1, 6.5]
    km = models["Kaplan-Meier"].sf(grid)
    na = models["Nelson-Aalen"].sf(grid)
    fh = models["Fleming-Harrington"].sf(grid)
    assert not np.allclose(km, na)
    assert np.all(na >= km - 1e-12)
    assert np.all((fh >= km - 1e-12) & (fh <= na + 1e-12))
    # Each model carries the variance for its own estimator.
    for model in models.values():
        assert np.isfinite(model.greenwood).any()


def test_turnbull_docstring_example_unchanged():
    # The rewrite must reproduce the long-standing example output.
    x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10]])
    model = surpyval.Turnbull.fit(x)
    expected = [
        1.0,
        1.0,
        0.63472351,
        0.29479882,
        0.2631432,
        0.2631432,
        0.2631432,
        0.09680497,
    ]
    assert np.allclose(model.R, expected, atol=1e-6)


# -- #203: EM correctness under truncation ------------------------------------


def _degenerate_case():
    # The reproduction from issue #203: a small, heavily truncated, mixed-
    # censoring sample on which the truncated EM used to migrate all mass
    # below the observation windows and return an all-zero survival curve
    # silently.
    x = [1, 2, [3, 6], 7, 8, 9, [5, 9], [4, 10], [7, 10], 11, 12]
    c = [1, 1, 2, 0, 0, 0, 2, 2, 2, -1, 0]
    n = [1, 2, 1, 3, 2, 2, 1, 1, 2, 1, 1]
    tl = [0, 0, 0, 0, 0, 2, 3, 3, 1, 1, 5]
    return x, c, n, tl


def test_turnbull_previously_degenerate_case_now_identifiable():
    # The #203 reproduction collapsed because left-censored and
    # entry-spanning interval supports extended below the observation
    # windows. With each support intersected with its own truncation
    # window (#273) the data are identifiable and the EM converges to a
    # healthy estimate instead of the all-zero fixed point.
    x, c, n, tl = _degenerate_case()
    model = surpyval.Turnbull.fit(x=x, c=c, n=n, tl=tl, max_iter=200000)
    assert model.degenerate is False
    assert model.converged is True
    R = np.asarray(model.R)
    assert R[-1] < 0.1  # survival still falls essentially to zero
    assert np.nanmax(R) == pytest.approx(1.0)
    assert np.all(np.diff(R) <= 1e-12)  # monotone non-increasing


def test_turnbull_non_convergence_still_warns():
    # The same case stopped early must warn rather than return silently.
    x, c, n, tl = _degenerate_case()
    with pytest.warns(UserWarning, match="did not converge"):
        surpyval.Turnbull.fit(x=x, c=c, n=n, tl=tl, max_iter=100)


def test_turnbull_healthy_truncated_fit_is_not_flagged():
    # A well-sized left-truncated sample is a genuine, identifiable fit: it
    # must not trip the degenerate detector.
    x, tl = _left_truncated_sample(n=600, seed=7)
    model = surpyval.Turnbull.fit(
        x, tl=tl, turnbull_estimator="Kaplan-Meier", max_iter=5000
    )
    assert model.degenerate is False


def test_turnbull_truncated_estimators_converge():
    # With the Kaplan-Meier self-consistency M-step, the hazard-form
    # estimators (which used to iterate on a biased update and never reach
    # ``tol`` under truncation) now converge on a healthy truncated sample.
    rng = np.random.default_rng(5)
    x = rng.weibull(1.5, 250) * 10
    tl = rng.uniform(0, 3, 250)
    keep = x > tl
    x, tl = x[keep], tl[keep]
    for est in ("Kaplan-Meier", "Fleming-Harrington", "Nelson-Aalen"):
        model = surpyval.Turnbull.fit(
            x, tl=tl, turnbull_estimator=est, max_iter=5000
        )
        assert model.converged is True
        assert model.degenerate is False


def test_turnbull_left_truncation_recovers_survival():
    # All three inner estimators recover S(median) of the true Weibull to
    # well within 0.04 on a left-truncated sample.
    rng = np.random.default_rng(11)
    x = rng.weibull(1.5, 400) * 10
    tl = rng.uniform(0, 3, 400)
    keep = x > tl
    x, tl = x[keep], tl[keep]
    median = 10 * (np.log(2)) ** (1 / 1.5)
    for est in ("Kaplan-Meier", "Fleming-Harrington", "Nelson-Aalen"):
        model = surpyval.Turnbull.fit(
            x, tl=tl, turnbull_estimator=est, max_iter=5000
        )
        s_med = float(np.atleast_1d(model.sf(median))[0])
        assert abs(s_med - 0.5) < 0.04


def test_turnbull_untruncated_default_is_unchanged():
    # The #203 fix is scoped to truncated fits; the documented untruncated
    # Fleming-Harrington example must be byte-for-byte unchanged.
    x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10]])
    model = surpyval.Turnbull.fit(x)
    expected = [
        1.0,
        1.0,
        0.63472351,
        0.29479882,
        0.2631432,
        0.2631432,
        0.2631432,
        0.09680497,
    ]
    assert np.allclose(model.R, expected, atol=1e-6)
    assert model.degenerate is False


def test_untruncated_censored_variance_matches_greenwood():
    # The estimation ladder redistributes right-censored mass as fractional
    # expected events, silently understating the variance (#260). On data
    # where Turnbull reduces exactly to KM, the confidence intervals must
    # now match Greenwood's exactly.
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    c = np.array([0, 0, 1, 0, 1])
    km_model = surpyval.KaplanMeier.fit(x, c=c)
    tb_model = surpyval.Turnbull.fit(x, c=c, turnbull_estimator="Kaplan-Meier")
    km_g = dict(zip(km_model.x, km_model.greenwood))
    tb_g = {}
    for xx, g in zip(tb_model.x, tb_model.greenwood):
        tb_g[xx] = g
    for xx, g in km_g.items():
        assert g == pytest.approx(tb_g[xx], abs=1e-12)
    assert np.allclose(
        km_model.cb([2.5, 4.0]), tb_model.cb([2.5, 4.0]), atol=1e-9
    )


def test_km_entry_tie_uses_strict_entry_convention():
    # (entry, exit] risk intervals (R survival / lifelines): a subject
    # entering exactly at an event time is not at risk for that event.
    # KM previously counted it (sf(2) = 0.8333) and disagreed with the
    # Turnbull NPMLE on identical data (#260).
    x = np.array([2.0, 3.0, 3.0, 4.0, 5.0, 6.0])
    tl = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    km_model = surpyval.KaplanMeier.fit(x, tl=tl)
    tb_model = surpyval.Turnbull.fit(
        x, tl=tl, turnbull_estimator="Kaplan-Meier"
    )
    assert float(np.ravel(km_model.sf(2.0))[0]) == pytest.approx(0.75)
    assert float(np.ravel(tb_model.sf(2.0))[0]) == pytest.approx(
        0.75, abs=1e-6
    )


def test_value_at_own_truncation_time_rejected():
    # A value at exactly its own left-truncation time has a zero-length
    # observation window under (entry, exit]; it previously slipped
    # through and silently distorted the Turnbull estimate (#260).
    with pytest.raises(ValueError, match="strictly less"):
        surpyval.Turnbull.fit(
            np.array([2.0, 3.0, 4.0]), tl=np.array([2.0, 1.0, 1.0])
        )
    with pytest.raises(ValueError, match="strictly less"):
        surpyval.KaplanMeier.fit(
            np.array([2.0, 3.0, 4.0]), tl=np.array([2.0, 1.0, 1.0])
        )


def test_max_iter_zero_raises():
    with pytest.raises(ValueError, match="max_iter"):
        surpyval.Turnbull.fit(
            np.array([1.0, 2.0, 3.0]), c=np.array([0, 1, 0]), max_iter=0
        )


def test_only_the_km_option_is_the_npmle():
    # The three turnbull_estimator options are not three ways of computing
    # one number: the EM recovers the same r and d, and they then differ in
    # how those become a survival curve. Only Kaplan-Meier is the
    # non-parametric MLE; Nelson-Aalen and Fleming-Harrington are exp(-H)
    # constructions that are not maximising anything.
    #
    # This pins the figures quoted in the ``turnbull_estimator`` docstring,
    # which exist because comparing a default (FH) Turnbull fit against
    # KaplanMeier and reading the gap as a defect is an easy mistake -- it
    # is the mistake #260 was originally filed on.
    from scipy.optimize import minimize

    x = np.array([2.0, 3.0, 3.0, 4.0, 5.0, 6.0])
    tl = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    support = np.array([2.0, 3.0, 4.0, 5.0, 6.0])

    def neg_ll(u):
        # masses on the support, softmax-parameterised so they stay simplex
        p = np.exp(u - u.max())
        p = p / p.sum()
        total = 0.0
        for xi, ti in zip(x, tl):
            mass = p[support == xi].sum()
            at_risk = p[support > ti].sum()  # (entry, exit]
            if mass <= 0 or at_risk <= 0:
                return 1e6
            total += np.log(mass) - np.log(at_risk)
        return -total

    best = None
    for seed in range(6):
        res = minimize(
            neg_ll,
            np.random.default_rng(seed).normal(size=support.size),
            method="Nelder-Mead",
            options={"maxiter": 40000, "fatol": 1e-14, "xatol": 1e-12},
        )
        if best is None or res.fun < best.fun:
            best = res

    p = np.exp(best.x - best.x.max())
    p = p / p.sum()
    npmle_sf2 = p[support > 2.0].sum()
    assert npmle_sf2 == pytest.approx(0.75, abs=1e-5)

    got = {
        est: float(
            np.ravel(
                surpyval.Turnbull.fit(x, tl=tl, turnbull_estimator=est).sf(2.0)
            )[0]
        )
        for est in ("Kaplan-Meier", "Fleming-Harrington", "Nelson-Aalen")
    }

    # KM reaches the NPMLE; the other two are elsewhere, and are ordered.
    assert got["Kaplan-Meier"] == pytest.approx(npmle_sf2, abs=1e-5)
    assert got["Fleming-Harrington"] == pytest.approx(0.7652, abs=1e-3)
    assert got["Nelson-Aalen"] == pytest.approx(0.7788, abs=1e-3)
    assert (
        got["Kaplan-Meier"] < got["Fleming-Harrington"] < got["Nelson-Aalen"]
    )


# ---------------------------------------------------------------------------
# #391: Turnbull with every row right truncated.
# ---------------------------------------------------------------------------


# -- #391: Turnbull with every row right truncated ----------------------------
# The Kaplan-Meier option reaches 0 where all the mass is placed; the
# default Fleming-Harrington one never reaches 0, and must agree with the
# same data without the right truncation.


def _km_turnbull(**kw):
    return quietly(sp.Turnbull.fit, turnbull_estimator="Kaplan-Meier", **kw)


def test_turnbull_failure_at_its_right_truncation_time():
    # One failure at 1, observable up to 1: all the mass is at 1, so
    # sf(1) is 0. Turnbull gave 1 (ladder x = [1], R = [1]).
    model = _km_turnbull(x=[1.0], c=[0], tr=[1.0])
    assert model.sf([1.0])[0] == 0.0
    fh = quietly(sp.Turnbull.fit, x=[1.0], c=[0], tr=[1.0])
    untruncated = quietly(sp.Turnbull.fit, x=[1.0], c=[0])
    np.testing.assert_allclose(fh.sf([0.5, 1.0]), untruncated.sf([0.5, 1.0]))


def test_turnbull_two_failures_right_truncated_at_the_last():
    # Failures at 1 and 2, both observable up to 2: sf(2) is 0. Turnbull
    # gave 0.5; with the second row untruncated (tr = inf) it gave 0.
    x, c = [1.0, 2.0], [0, 0]
    model = _km_turnbull(x=x, c=c, tr=[2.0, 2.0])
    np.testing.assert_allclose(model.sf([1.0, 2.0]), [0.5, 0.0])
    fh = quietly(sp.Turnbull.fit, x=x, c=c, tr=[2.0, 2.0])
    one = quietly(sp.Turnbull.fit, x=x, c=c, tr=[2.0, np.inf])
    np.testing.assert_allclose(fh.sf([1.0, 2.0]), one.sf([1.0, 2.0]))


def test_turnbull_right_censored_inside_its_window():
    # Censored at 1 and observable up to 2: the event is in (1, 2], so
    # sf(2) is 0. Turnbull gave 1.
    model = _km_turnbull(x=[1.0], c=[1], tr=[2.0])
    np.testing.assert_allclose(model.sf([1.0, 2.0]), [1.0, 0.0])


def test_turnbull_left_censored_at_its_right_truncation_time():
    # Left censored at 1, observable up to 1: the ladder was empty and sf
    # raised IndexError (index -1 of an empty R).
    model = _km_turnbull(x=[1.0], c=[-1], tr=[1.0])
    np.testing.assert_allclose(model.sf([0.5, 1.0]), [1.0, 0.0])


def test_turnbull_right_truncated_matches_the_kaplan_meier():
    # Exact data, every row observable up to a time past the last failure
    # (so the truncation excludes nothing): the Turnbull-KM is the
    # Kaplan-Meier, to its last failure.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    model = _km_turnbull(x=x, c=np.zeros(4, int), tr=np.full(4, 5.0))
    km = sp.KaplanMeier.fit(x)
    np.testing.assert_allclose(model.sf(x), km.sf(x), atol=1e-8)


# ---------------------------------------------------------------------------
# The Turnbull variance ladder is aligned with the survival
# estimate, so ``cb()`` does not step a piece early on
# interval-censored data; where ``turnbull_estimator`` acts,
# and an unknown one is refused.
# ---------------------------------------------------------------------------


class TestVarianceAlignment:
    def test_interval_censored_bounds_where_estimate_is_one(self):
        # Before the fix, cb(1) was [0, 1] while sf(1) was 1: the variance
        # at 1 already held the expected failures in (1, 2].
        model = fit_turnbull_quietly(
            xl=[0, 1, 2, 3],
            xr=[2, 3, 4, 5],
            turnbull_estimator="Kaplan-Meier",
        )
        assert model.sf(1) == 1.0
        np.testing.assert_allclose(model.cb(1), [1.0, 1.0])

    def test_mixed_example_bounds_where_estimate_is_one(self):
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING, turnbull_estimator="Kaplan-Meier"
        )
        assert model.sf(5.5) == 1.0
        np.testing.assert_allclose(model.cb(5.5), [1.0, 1.0])

    @pytest.mark.parametrize(
        "estimator", ["Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington"]
    )
    def test_reported_ladder_generates_the_estimate(self, estimator):
        # The r and d reported with the model are the ones behind R (and,
        # without truncation, behind the variance) at the same x.
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING, turnbull_estimator=estimator
        )
        np.testing.assert_allclose(
            nonp.FIT_FUNCS[estimator](model.r, model.d), model.R, atol=1e-12
        )
        np.testing.assert_allclose(
            nonp.VAR_FUNCS[estimator](model.r, model.d),
            model.greenwood,
            equal_nan=True,
        )

    def test_variance_zero_exactly_where_estimate_is_one(self):
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING, turnbull_estimator="Kaplan-Meier"
        )
        at_one = model.R == 1.0
        assert at_one.any() and (~at_one).any()
        assert (model.greenwood[at_one] == 0).all()
        assert (
            model.greenwood[~at_one & np.isfinite(model.greenwood)] > 0
        ).all()

    def test_truncated_interval_censored_bounds(self):
        # Same slicing on the truncated (observed-count) variance ladder.
        model = fit_turnbull_quietly(
            xl=[0, 1, 2, 3, 1],
            xr=[2, 3, 4, 5, 4],
            tl=[-1, -1, 0.5, 0.5, 0],
            turnbull_estimator="Kaplan-Meier",
            max_iter=20_000,
        )
        # The EM leaves the estimate 1 up to round-off before 2.
        at_one = model.R > 1 - 1e-12
        assert at_one.sum() == 3
        np.testing.assert_allclose(model.cb(model.x[at_one]), 1.0)
        assert (model.greenwood[at_one] == 0).all()
        # And the bounds bracket the estimate everywhere they are defined.
        cb = model.cb(model.x)
        assert (cb[:, 0] <= model.R + 1e-12).all()
        assert (cb[:, 1] >= model.R - 1e-12).all()

    def test_mass_only_on_innermost_intervals(self):
        # Without truncation the NPMLE puts no mass outside Turnbull's
        # innermost intervals; restricting the EM to them removes the
        # slowly-decaying residual mass (d ~ 2e-8 on (4, 5] here) that made
        # the estimate 1 - 1e-9 and its log(-log) bounds [0, 1].
        model = fit_turnbull_quietly(
            **TURNBULL_MIXED_CENSORING, turnbull_estimator="Kaplan-Meier"
        )
        assert model.converged
        np.testing.assert_array_equal(model.d[model.x <= 5], 0.0)
        np.testing.assert_array_equal(model.R[model.x <= 5], 1.0)

    def test_right_censored_still_matches_kaplan_meier(self):
        x = [1, 2, 2, 3, 4, 5, 5, 6, 7, 9]
        c = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1]
        tb = fit_turnbull_quietly(x=x, c=c, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, c=c)
        grid = np.linspace(1, 9, 33)
        np.testing.assert_allclose(tb.sf(grid), km.sf(grid), atol=1e-10)
        np.testing.assert_allclose(tb.cb(grid), km.cb(grid), atol=1e-10)

    def test_left_truncated_still_matches_kaplan_meier(self):
        x = [2, 3, 3, 4, 5, 6, 7, 8]
        tl = [0, 0, 1, 1, 2, 2, 3, 0]
        tb = fit_turnbull_quietly(
            x=x, tl=tl, turnbull_estimator="Kaplan-Meier"
        )
        km = KaplanMeier.fit(x=x, tl=tl)
        grid = np.linspace(2, 8, 25)
        np.testing.assert_allclose(tb.sf(grid), km.sf(grid), atol=1e-9)
        np.testing.assert_allclose(tb.cb(grid), km.cb(grid), atol=1e-9)


class TestTurnbullEstimatorOption:
    def test_docstring_says_truncation_iterates_with_kaplan_meier(self):
        doc = surv.KaplanMeier.fit.__doc__
        assert (
            "With truncation the EM always iterates with the Kaplan-Meier"
            in doc
        )

    def test_truncated_ladder_does_not_depend_on_option(self):
        # What the docstring now says: under truncation the EM itself is
        # the same whatever the option.
        kw = dict(x=[2, 3, 3, 4, 5, 6], tl=[0, 0, 1, 1, 2, 2])
        km = fit_turnbull_quietly(**kw, turnbull_estimator="Kaplan-Meier")
        na = fit_turnbull_quietly(**kw, turnbull_estimator="Nelson-Aalen")
        np.testing.assert_allclose(km.r, na.r)
        np.testing.assert_allclose(km.d, na.d)
        assert km.sf(2) == pytest.approx(0.75)
        assert na.sf(2) == pytest.approx(0.779, abs=5e-4)

    @pytest.mark.parametrize("call", ["fit", "function", "plotting"])
    def test_unknown_estimator_raises(self, call):
        with pytest.raises(ValueError, match="turnbull_estimator.*'foo'"):
            if call == "fit":
                Turnbull.fit(xl=[0, 1], xr=[2, 3], turnbull_estimator="foo")
            elif call == "function":
                nonp.turnbull(
                    np.array([1.0, 2.0]),
                    np.array([0, 0]),
                    np.array([1, 1]),
                    np.array([[-np.inf, np.inf]] * 2),
                    estimator="foo",
                )
            else:
                plotting_positions(
                    [1.0, 2.0, 3.0],
                    heuristic="Turnbull",
                    turnbull_estimator="foo",
                )

    def test_error_lists_the_options(self):
        with pytest.raises(ValueError) as info:
            Turnbull.fit([1.0, 2.0], turnbull_estimator="Kaplan Meier")
        for option in ("Fleming-Harrington", "Nelson-Aalen", "Kaplan-Meier"):
            assert option in str(info.value)


# ---------------------------------------------------------------------------
# Linear interpolation, quiet truncated fits and the
# Nelson-Aalen EM.
# ---------------------------------------------------------------------------


def test_turnbull_linear_interp_matches_kaplan_meier():
    x, c = [1, 2, 3, 4, 5, 6], [0, 1, 0, 0, 1, 0]
    tb = sp.Turnbull.fit(x, c=c, turnbull_estimator="Kaplan-Meier")
    km = sp.KaplanMeier.fit(x, c=c)
    grid = [1.5, 2.5, 3.5, 4.5, 5.5]
    np.testing.assert_allclose(
        tb.sf(grid, interp="linear"), km.sf(grid, interp="linear"), atol=1e-8
    )


def test_turnbull_healthy_truncated_fits_do_not_warn():
    # Right-truncated exact data (the Lynden-Bell estimator) and
    # left-truncated exact data (the Kaplan-Meier) are identifiable.
    rt = no_warnings(
        sp.Turnbull.fit,
        [7, 3, 5, 2, 7, 6],
        tr=[11, 3, 7, 5, 7, 6],
        turnbull_estimator="Kaplan-Meier",
    )
    assert rt.exploitable_mass == 0.0
    x, tl = [8, 1, 4, 7, 8, 2], [5, 0, 1, 0, 7, 1]
    lt = no_warnings(
        sp.Turnbull.fit, x, tl=tl, turnbull_estimator="Kaplan-Meier"
    )
    assert lt.exploitable_mass == 0.0
    np.testing.assert_allclose(
        lt.sf([1, 2, 4, 7]),
        sp.KaplanMeier.fit(x, tl=tl).sf([1, 2, 4, 7]),
        atol=1e-8,
    )


def test_turnbull_nelson_aalen_em_converges_on_complete_data():
    # The last risk count flipped between 9e-16 and 0 every iteration.
    model = no_warnings(
        sp.Turnbull.fit,
        [1, 2, 3, 4],
        n=[2, 1, 2, 1],
        turnbull_estimator="Nelson-Aalen",
    )
    assert model.converged
    assert model.iters < 50


# ---------------------------------------------------------------------------
# #272: interval/left-censored observations follow the (l, r]
# convention; #273: under truncation the variance ladder uses
# observed counts; the support-window intersection; degenerate
# intervals reduce to exact times.
# ---------------------------------------------------------------------------


class TestIntervalEndpointConvention:
    def test_right_endpoint_tie_matches_npmle(self):
        # 272: exact {2} x1, {3} x5, interval (1, 3] x5. The (l, r] NPMLE
        # puts mass 1/6 at 2 and 5/6 at 3 (lifelines NPMLE agrees), so
        # S(2.5) = 5/6. The old support excluded the atom at 3 and gave
        # S(2.5) = 5/11.
        tb = Turnbull.fit(
            x=[2] * 1 + [3] * 5 + [[1, 3]] * 5,
            c=[0] * 6 + [2] * 5,
            turnbull_estimator="Kaplan-Meier",
        )
        sf25 = float(np.ravel(tb.sf(2.5))[0])
        assert sf25 == pytest.approx(5 / 6, abs=1e-6)

    def test_left_censored_at_event_time(self):
        # 272: left-censored at 3 means "failed at or before 3": with
        # exact {3} x5 + left-censored-at-3 x5 all mass sits at 3.
        tb = Turnbull.fit(
            x=[3.0] * 5 + [3.0] * 5,
            c=[0] * 5 + [-1] * 5,
            turnbull_estimator="Kaplan-Meier",
        )
        R = np.asarray(tb.R)
        # Survival is 1 before 3 and 0 at/after 3.
        assert R[-1] == pytest.approx(0.0, abs=1e-9)

    def test_generic_intervals_unchanged(self):
        # Non-coinciding endpoints were already correct; sanity-pin one.
        x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10]])
        model = Turnbull.fit(x)
        R = np.asarray(model.R)
        assert np.all(np.isfinite(R))
        assert np.all(np.diff(R) <= 1e-12)


class TestTruncatedVariance:
    def test_variance_matches_delayed_entry_km(self):
        # 273: exact + right-censored + left-truncated reduces to
        # delayed-entry KM — including the Greenwood ladder, which used
        # to produce a -1.5e15 increment at the last event.
        x = [2, 3, 4, 5, 6, 7, 8]
        c = [0, 0, 1, 0, 0, 0, 0]
        tl = [0, 1, 1, 2, 3, 0, 5]
        tb = Turnbull.fit(x=x, c=c, tl=tl, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, c=c, tl=tl)

        tb_x = np.asarray(tb.x)
        tb_gw = np.asarray(tb.greenwood)
        km_x = np.asarray(km.x)
        km_gw = np.asarray(km.greenwood)
        # Compare at each KM event time (TB's ladder has doubled bounds;
        # take the last TB position at or before each KM time).
        for xv, gv in zip(km_x, km_gw):
            idx = np.searchsorted(tb_x, xv, side="right") - 1
            if np.isnan(gv):
                assert np.isnan(tb_gw[idx])
            else:
                assert tb_gw[idx] == pytest.approx(gv, abs=1e-10)
        # No negative variance anywhere on the ladder.
        finite = tb_gw[np.isfinite(tb_gw)]
        assert np.all(finite >= 0)

    def test_survival_still_matches_delayed_entry_km(self):
        x = [2, 3, 4, 5, 6, 7, 8]
        c = [0, 0, 1, 0, 0, 0, 0]
        tl = [0, 1, 1, 2, 3, 0, 5]
        tb = Turnbull.fit(x=x, c=c, tl=tl, turnbull_estimator="Kaplan-Meier")
        km = KaplanMeier.fit(x=x, c=c, tl=tl)
        t_eval = [2.5, 3.5, 5.5, 6.5, 7.5]
        np.testing.assert_allclose(
            np.ravel(tb.sf(t_eval)), np.ravel(km.sf(t_eval)), atol=1e-9
        )


class TestSupportWindowIntersection:
    def test_left_censored_with_entry_converges(self):
        # 273: valid left-censored + delayed-entry data used to hit the
        # degenerate all-zero fixed point because the (-inf, x] support
        # was not intersected with the (tl, inf) window.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            tb = Turnbull.fit(
                x=[3, 4, 5, 6, 7],
                c=[0, -1, 0, 0, 0],
                tl=[1, 2, 2, 3, 1],
                turnbull_estimator="Kaplan-Meier",
            )
        R = np.asarray(tb.R)
        assert np.all(np.isfinite(R))
        assert np.nanmax(R) == pytest.approx(1.0)


class TestDegenerateIntervalReducibility:
    def test_degenerate_intervals_match_1d_form(self):
        # 273: the same exact + right-censored data in degenerate-interval
        # form must give identical survival and Greenwood variance to the
        # 1-D form (both on the KM-reducible observed-count ladder).
        x1 = [1, 2, 3, 4, 5, 6]
        c1 = [0, 0, 1, 0, 1, 0]
        x2 = [[v, v] if cc == 0 else [v, np.inf] for v, cc in zip(x1, c1)]
        t1 = Turnbull.fit(x=x1, c=c1, turnbull_estimator="Kaplan-Meier")
        t2 = Turnbull.fit(x=x2, turnbull_estimator="Kaplan-Meier")
        np.testing.assert_allclose(
            np.asarray(t1.greenwood),
            np.asarray(t2.greenwood),
            atol=1e-12,
        )
        km = KaplanMeier.fit(x=x1, c=c1)
        t_eval = [1.5, 2.5, 3.5, 4.5, 5.5]
        np.testing.assert_allclose(
            np.ravel(t2.sf(t_eval)), np.ravel(km.sf(t_eval)), atol=1e-8
        )
