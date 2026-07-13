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
