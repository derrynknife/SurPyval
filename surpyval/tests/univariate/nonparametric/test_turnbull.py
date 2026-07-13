"""Turnbull estimator: truncation correctness, convergence controls and
memory-efficient EM (the interval-censoring results against R's icenReg
live in test_np.py)."""

import numpy as np
import pytest

import surpyval


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
