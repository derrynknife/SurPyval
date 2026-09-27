"""The Turnbull pieces each observation may fail in, and may be seen in (#368).

Piece ``j`` of a Turnbull fit is ``(bounds[j], bounds[j+1]]``. Two index
searches were one piece off:

- a right-censored observation at ``x`` (``T > x``) could not fail in the
  piece ``(x, next bound]`` just after its censoring time;
- a right-truncated window ``(tl, tr]`` took in the piece ``(tr, next
  bound]`` just after its truncation time.

Both only matter when some other observation can fail in that piece, so
exact and right-censored data were unaffected, but with interval censoring
or right truncation the EM converged to a curve that was not the
maximum-likelihood estimate. The checks below compare the fitted
log-likelihood with one written directly from the data's (l, r] and
(tl, tr] conventions, independently of the piece indexing.
"""

import importlib
import warnings

import numpy as np
import pytest

from surpyval import KaplanMeier, Turnbull
from surpyval.utils import xcnt_handler

# The package re-exports the function under the module's name.
turnbull_module = importlib.import_module(
    "surpyval.univariate.nonparametric.turnbull"
)


def _fit(**kwargs):
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        model = Turnbull.fit(
            turnbull_estimator="Kaplan-Meier", max_iter=20000, **kwargs
        )
    return model, [str(w.message) for w in record]


# -- An independent likelihood ------------------------------------------------


def _candidate_sets(x, c, t):
    """Each row's event set and window on points that represent every
    distribution: each distinct finite value, the midpoint between
    neighbours, and a point below and above them all."""
    xl, xr = x[:, 0], x[:, 1]
    tl, tr = t[:, 0], t[:, 1]
    vals = np.unique(np.concatenate([xl, xr, tl, tr]))
    vals = vals[np.isfinite(vals)]
    pts = np.sort(
        np.concatenate(
            [
                vals,
                (vals[:-1] + vals[1:]) / 2,
                [vals.min() - 1, vals.max() + 1],
            ]
        )
    )
    events = np.select(
        [c[:, None] == 0, c[:, None] == 1, c[:, None] == -1],
        [pts == xl[:, None], pts > xl[:, None], pts <= xr[:, None]],
        (pts > xl[:, None]) & (pts <= xr[:, None]),
    )
    window = (pts > tl[:, None]) & (pts <= tr[:, None])
    return pts, (events & window).astype(float), window.astype(float)


def _loglik(p, events, window, n):
    return float(np.sum(n * (np.log(events @ p) - np.log(window @ p))))


def _independent_max(events, window, n, iters=5000):
    """Self-consistency EM on the candidate points from three starts."""
    rng = np.random.default_rng(0)
    K = events.shape[1]
    best = -np.inf
    for start in range(3):
        p = np.ones(K) / K if start == 0 else rng.dirichlet(np.ones(K))
        for _ in range(iters):
            seen = p * ((n / (events @ p)) @ events)
            ghosts = p * ((n / (window @ p)) @ (1 - window))
            p = (seen + ghosts) / (seen + ghosts).sum()
        best = max(best, _loglik(p, events, window, n))
    return best


def _fitted_loglik(model, pts, events, window, n):
    """The fitted curve's log-likelihood, each piece's mass placed on the
    candidate point inside it."""
    bounds = model.bounds
    M = bounds.size
    R = np.concatenate([model.R, model.R_lower[-1:]])
    mass = -np.diff(np.concatenate([[1.0], R, [0.0]]))
    p = np.zeros(pts.size)
    for j in range(M):
        a = bounds[j]
        b = bounds[j + 1] if j + 1 < M else np.inf
        if np.isposinf(b) or np.isposinf(a):
            point = pts.max()
        elif np.isneginf(a):
            point = pts.min()
        elif a == b:
            point = a
        else:
            point = (a + b) / 2
        p[pts == point] += max(mass[j], 0.0)
    return _loglik(p / p.sum(), events, window, n)


def _gap(**data):
    x, c, n, t = xcnt_handler(**data)
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = np.column_stack([x, x])
    pts, events, window = _candidate_sets(x, c, t)
    model, _ = _fit(**data)
    best = _independent_max(events, window, n)
    return model, best - _fitted_loglik(model, pts, events, window, n)


# -- The issue's example ------------------------------------------------------


def test_right_censored_support_includes_the_piece_after_it():
    # One failure in (1, 2] and one unit censored at 1.5: all the mass in
    # (1.5, 2] gives S(1.5) = 1 and S(1) - S(2) = 1, a likelihood of 1.
    # The censored unit's support used to start at 2, so the fit was
    # S(1.5) = 0.75, S(2) = 0.5: a likelihood of 0.375.
    model, _ = _fit(x=[[1, 2], [1.5, 1.5]], c=[2, 1])
    s1, s15, s2 = model.sf([1.0, 1.5, 2.0])
    assert (s1 - s2) * s15 == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(model.R, [1.0, 1.0, 0.0], atol=1e-9)


def test_right_censored_at_an_exact_failure_time():
    # Censored at 1.5 where another unit failed: the censored unit's event
    # is still after 1.5, so it may be in (1.5, 2] too. The likelihood
    # p(1.5) * S(1.5) * (S(1) - S(2)) is at most 1/4, with half the mass
    # at 1.5 and half in (1.5, 2].
    model, _ = _fit(x=[[1, 2], [1.5, 1.5], [1.5, 1.5]], c=[2, 1, 0])
    np.testing.assert_allclose(model.x, [1.0, 1.5, 1.5, 2.0])
    np.testing.assert_allclose(model.R, [1.0, 1.0, 0.5, 0.0], atol=1e-9)


def test_right_truncation_window_ends_at_the_truncation_time():
    # One failure at 1 observable only up to 2, one at 1 untruncated and
    # one in (1.5, 3]. The likelihood p(1) / F(2) * p(1) * P(1.5, 3] is
    # 1/4 with half the mass at 1 and half in (2, 3]: mass in (2, 3] costs
    # the truncated row nothing, as it could not have been seen there.
    # The window used to include (2, 3], giving 2/3 at 1 and 1/6 in each
    # of (1.5, 2] and (2, 3], a likelihood of 0.18.
    model, messages = _fit(
        x=[[1, 1], [1.5, 3], [1, 1]], c=[0, 2, 0], tr=[2, np.inf, np.inf]
    )
    assert messages == []
    assert model.npmle == "exists"
    np.testing.assert_allclose(model.x, [1.0, 1.0, 1.5, 2.0, 3.0])
    np.testing.assert_allclose(model.R, [1.0, 0.5, 0.5, 0.5, 0.0], atol=1e-9)


# -- The maximum likelihood, checked independently ----------------------------


@pytest.mark.parametrize(
    "data",
    [
        # No truncation: right censored at 3 inside the interval (3, 5].
        dict(
            x=[[3, 3], [6, 6], [3, 3], [2, 2], [3, 5]],
            c=[1, -1, 1, -1, 2],
        ),
        # Left truncation.
        dict(x=[1, 4, 1], c=[1, -1, 1], tl=[-1, -np.inf, -np.inf]),
        # Right truncation: censored at 0 and seen only up to 3, failed
        # by 3 and seen only up to 5. All the mass in (0, 3] gives 1.
        dict(x=[0, 3], c=[1, -1], tr=[3, 5]),
        # Both.
        dict(x=[7, 5], c=[-1, 1], tl=[4, 3], tr=[8, 8]),
    ],
    ids=["untruncated", "left_truncated", "right_truncated", "both"],
)
def test_fit_reaches_the_independent_maximum(data):
    # Before the fix these fits were 1.10, 1.39, 1.39 and 0.81 below the
    # maximum log-likelihood.
    model, gap = _gap(**data)
    assert model.npmle == "exists"
    assert model.converged
    assert gap < 1e-6


def test_support_inside_a_right_truncated_window():
    # A failure at 1 and a unit censored at 1 seen only up to 4: its
    # event is in (1, 4]. Its support used to start at the piece after 4,
    # outside its own window, so the fit had likelihood zero. The maximum
    # of p(1) * P(1, 4] / F(4) is 1/4, half the mass at 1 and half in
    # (1, 4].
    model, gap = _gap(x=[1, 1], c=[0, 1], tl=[-1, -1], tr=[np.inf, 4])
    assert model.npmle == "exists"
    np.testing.assert_allclose(model.sf([1, 4]), [0.5, 0.0], atol=1e-9)
    assert gap < 1e-6


# -- Verdicts that change with the corrected windows (#327) -------------------


def test_right_truncated_intervals_npmle_does_not_exist():
    # (3, 4] seen only up to 6 and (0, 1] seen only up to 3. With mass a
    # in (0, 1] and b in (3, 4] the likelihood is b / (a + b) * a / a,
    # which tends to 1 as a -> 0 but is 0/0 at a = 0: no maximum. The
    # over-wide windows made the second row pay for mass in (3, 4], so
    # the verdict was "exists" and the fit settled at a = b.
    model, messages = _fit(x=[[3, 4], [0, 1]], c=[2, 2], tr=[6, 3])
    assert model.npmle == "does not exist"
    assert any("does not exist" in m for m in messages)


def test_right_truncated_exact_and_left_censored_is_not_unique():
    # A failure at 1 seen only up to 2 and a failure by 7: any split of
    # the mass between 1 and (2, 7] has likelihood 1.
    model, messages = _fit(x=[1, 7], c=[0, -1], tr=[2, np.inf])
    assert model.npmle == "not unique"
    assert any("not unique" in m for m in messages)


def test_double_truncation_reproducer_no_longer_raises():
    # The verdict's time lookup indexed one past the end of ``bounds`` on
    # these data (an IndexError) when the windows were one piece too wide.
    model, messages = _fit(
        x=[[0, 1], [6, 6], [0, 0], [3, 6]],
        c=[2, 1, 0, 2],
        tl=[-1, -np.inf, -2, 2],
        tr=[2, 11, 3, 8],
    )
    assert model.npmle == "does not exist"
    assert any("does not exist" in m for m in messages)


# -- Delayed entry ------------------------------------------------------------


def test_delayed_entry_gap_keeps_the_kaplan_meier():
    # Censored at 1 before the second unit enters at 5: the censored
    # unit's support now includes (1, 5], where nothing fixes the mass
    # ("not unique"). The Kaplan-Meier with delayed entry is the maximiser
    # with none there, and the fit returns it.
    data = dict(x=[1, 6], c=[1, 0], tl=[0, 5])
    model, messages = _fit(**data)
    assert model.npmle == "not unique"
    assert any("not unique" in m for m in messages)
    t = np.array([1.0, 3.0, 5.0, 5.5, 6.0])
    np.testing.assert_allclose(
        model.sf(t), KaplanMeier.fit(**data).sf(t), atol=1e-9
    )


def test_right_censored_rows_leave_the_variance_risk_set_at_censoring():
    # Under truncation the variance ladder counts a right-censored unit at
    # risk up to its censoring time. It was also counted in the piece
    # after it, (3.5, 4] here, reported at 4. At each reported time b the
    # risk set is the units entered before b and still under observation
    # at b.
    x = np.array([2, 3, 3.5, 4, 5, 6.0])
    c = np.array([0, 0, 1, 0, 1, 0])
    tl = np.array([0, 0, 1, 1, 2, 2.0])
    out = turnbull_module.turnbull(
        *xcnt_handler(x=x, c=c, tl=tl), "Kaplan-Meier"
    )
    expected = [np.sum((tl < b) & (x >= b)) for b in out["x"]]
    np.testing.assert_array_equal(out["var_r"], expected)
