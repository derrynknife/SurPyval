"""The Turnbull-score split under truncation (issue #188, stage 2).

``kind="non-parametric"`` used to raise on right-truncated data and on
truncated data with left or interval censoring. A truncated row's score
is now the score of its truncation-conditioned likelihood: the event's
score less the score of its window. These check:

- the scores are the derivative of each row's conditional log-likelihood
  along the proportional hazards path ``S**theta``, at every censoring
  type with left, right and two-sided truncation;
- on left-truncated right-censored data they are the delayed-entry
  martingale residuals, their sum over a child is the delayed-entry
  log-rank numerator ``O - E``, and the split chooses as the risk-set
  log-rank split does;
- truncation that excludes nothing changes nothing;
- the window term is what keeps the test valid when the truncation
  depends on a covariate with no effect: without it the scores split on
  that covariate;
- trees on left-truncated interval-censored and on right-truncated data
  split on a planted effect, with Turnbull leaves that predict and
  round-trip, and conditional inference stays a single leaf on null data;
- every case of the full data model builds a non-parametric tree, and
  the forest runs end to end.
"""

import contextlib
import io
import json
import warnings

import numpy as np
import pytest

from surpyval.beta.ml.forest import RandomSurvivalForest, SurvivalTree
from surpyval.beta.ml.forest.conditional_inference import (
    max_statistic,
    nonparametric_scores,
)
from surpyval.beta.ml.forest.log_rank_split import (
    at_risk_on_grid,
    deaths_on_grid,
    log_rank_split,
)
from surpyval.beta.ml.forest.node import TerminalNode
from surpyval.beta.ml.forest.turnbull_score_split import (
    _interval_scores,
    log_rank_scores,
    pooled_cumulative_hazard,
    turnbull_score_split,
)
from surpyval.tests._helpers import tree_leaves
from surpyval.univariate.nonparametric.nonparametric import NonParametric
from surpyval.utils.surpyval_data import SurpyvalData


def _H_at(data):
    # The step function the scores evaluate H with: just after q.
    times, H = pooled_cumulative_hazard(data)

    def H_at(q):
        idx = np.searchsorted(times, q, side="right") - 1
        out = np.where(idx >= 0, H[np.maximum(idx, 0)], 0.0)
        return np.where(np.isposinf(q), np.inf, out)

    return H_at


def _left_truncated_inspections(seed, n=200, effect=1.0, entry_shift=0.0):
    """Units enter at a random time (``entry_shift`` later if feature 1 is
    above 0.5) and are only seen if they had not failed by then; they are
    then inspected every 2 time units up to 30, so a failure is interval
    censored, left censored at the first inspection after entry (failed
    between entry and it), or right censored at 30. Feature 0 above 0.5
    multiplies life by ``effect``."""
    rng = np.random.default_rng(seed)
    grid = np.arange(0.0, 32.0, 2.0)
    x, c, tl, Z = [], [], [], []
    while len(x) < n:
        z = rng.uniform(0, 1, 3)
        entry = rng.uniform(0, 3) + entry_shift * (z[1] > 0.5)
        T = 10 * rng.weibull(1.5) * (effect if z[0] > 0.5 else 1.0)
        if T <= entry:
            continue
        if T > 30:
            x.append(30.0)
            c.append(1)
        else:
            j = np.searchsorted(grid, T)
            if grid[j - 1] <= entry:
                x.append(grid[j])
                c.append(-1)
            else:
                x.append([grid[j - 1], grid[j]])
                c.append(2)
        tl.append(entry)
        Z.append(z)
    return dict(x=x, c=np.array(c), tl=np.array(tl), Z=np.array(Z))


def _right_truncated(seed, n=200, effect=1.0, truncation_shift=0.0):
    """Failures recorded exactly, but only for units that failed before a
    unit's truncation time (retrospective sampling), which is
    ``truncation_shift`` earlier if feature 1 is above 0.5. Feature 0
    above 0.5 multiplies life by ``effect``."""
    rng = np.random.default_rng(seed)
    x, tr, Z = [], [], []
    while len(x) < n:
        z = rng.uniform(0, 1, 3)
        bound = rng.uniform(6, 30) - truncation_shift * (z[1] > 0.5)
        T = 10 * rng.weibull(1.5) * (effect if z[0] > 0.5 else 1.0)
        if T > bound:
            continue
        x.append(T)
        tr.append(bound)
        Z.append(z)
    return dict(x=np.array(x), tr=np.array(tr), Z=np.array(Z))


def _data(d):
    d = dict(d)
    d.pop("Z")
    return SurpyvalData(**d, group_and_sort=False)


# -- the scores ------------------------------------------------------------


def _mixed_data():
    # Every censoring type, with left truncation on some rows, right
    # truncation on others and both on two.
    x = [1.5, 4.0, [2.0, 3.5], 0.8, 6.0, [1.0, 2.5], 3.0, 5.5, 2.2, 7.0]
    c = [0, 1, 2, -1, 0, 2, -1, 1, 0, 0]
    t = np.array(
        [
            [-np.inf, np.inf],
            [1.0, np.inf],
            [0.5, np.inf],
            [-np.inf, 9.0],
            [2.0, 10.0],
            [-np.inf, np.inf],
            [1.0, 8.0],
            [0.2, np.inf],
            [-np.inf, np.inf],
            [1.0, 12.0],
        ]
    )
    return SurpyvalData(x, c, t=t, group_and_sort=False)


def test_scores_are_the_conditional_likelihood_gradient():
    # Along S**theta (hazard theta h), a row's conditional log-likelihood
    # is log[S(a)**theta - S(b)**theta] - log[S(tl)**theta - S(tr)**theta]
    # for an event in (a, b] (its interval within its window), and
    # log(theta) + theta log S(x) less the window term for an event
    # observed at x; the score is its derivative at theta = 1.
    data = _mixed_data()
    H_at = _H_at(data)
    x = np.asarray(data.x, dtype=float)
    c = np.asarray(data.c)
    tl, tr = data.t[:, 0], data.t[:, 1]
    a = np.maximum(np.where(c == -1, -np.inf, x[:, 0]), tl)
    b = np.minimum(np.where(c == 1, np.inf, x[:, 1]), tr)

    def S(q, theta):
        return np.exp(-theta * H_at(q))

    def log_likelihood(theta):
        with np.errstate(divide="ignore"):
            event = np.where(
                c == 0,
                np.log(theta) + np.log(S(x[:, 0], theta)),
                np.log(S(a, theta) - S(b, theta)),
            )
        return event - np.log(S(tl, theta) - S(tr, theta))

    step = 1e-6
    numeric = (log_likelihood(1 + step) - log_likelihood(1 - step)) / (
        2 * step
    )
    np.testing.assert_allclose(log_rank_scores(data), numeric, atol=1e-6)


def test_untruncated_rows_keep_their_scores_by_hand():
    # g(-inf, inf) = 0: the window term of an untruncated row vanishes.
    data = _mixed_data()
    H_at = _H_at(data)
    untruncated = ~(np.isfinite(data.t).any(axis=1))
    x = np.asarray(data.x, dtype=float)[untruncated]
    c = np.asarray(data.c)[untruncated]
    lo = np.where(c == -1, -np.inf, x[:, 0])
    hi = np.where(c == 1, np.inf, x[:, 1])
    np.testing.assert_allclose(
        log_rank_scores(data)[untruncated],
        _interval_scores(H_at, lo, hi, c == 0),
    )


def _delayed_entry(seed, n=150, effect=1.0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(size=(n, 3))
    entry = rng.uniform(0, 3, n)
    T = entry + 8 * rng.weibull(1.5, n) * np.where(Z[:, 0] > 0.5, effect, 1)
    C = entry + rng.uniform(2, 20, n)
    data = SurpyvalData(
        np.minimum(T, C), (C < T).astype(int), tl=entry, group_and_sort=False
    )
    return data, Z


@pytest.mark.parametrize("seed", range(3))
def test_left_truncated_right_censored_scores_are_martingale_residuals(seed):
    # The Turnbull estimate fitted with the truncation has the
    # delayed-entry Nelson-Aalen hazard, and the window term is
    # log S(entry) = -H(entry): the scores are
    # delta - [H(x) - H(entry)], whose sum over a child is O - E.
    data, Z = _delayed_entry(seed)
    scores = log_rank_scores(data)
    np.testing.assert_allclose(
        scores, nonparametric_scores(data)[:, 0], atol=1e-6
    )
    grid, Y, d = data.to_xrd()
    for v in [0.3, 0.5, 0.8]:
        left = Z[:, 1] <= v
        o_minus_e = np.sum(
            deaths_on_grid(data[left], grid)
            - at_risk_on_grid(data[left], grid) * d / Y
        )
        assert scores[left].sum() == pytest.approx(o_minus_e, abs=1e-5)


def test_split_agrees_with_the_risk_set_log_rank_with_delayed_entry():
    # Same numerators, and variances that agree asymptotically.
    same = 0
    for seed in range(10):
        data, Z = _delayed_entry(seed, effect=0.5)
        a = log_rank_split(data, Z, 5, 2, [0, 1, 2])
        b = turnbull_score_split(data, Z, 5, 2, [0, 1, 2])
        assert a[0] == b[0] == 0
        same += bool(np.isclose(a[1], b[1]))
        assert abs(a[1] - b[1]) < 0.2
    assert same >= 6, same


def test_truncation_that_excludes_nothing_changes_nothing():
    d = _left_truncated_inspections(0, n=120)
    # Interval- and right-censored rows only: a left-censored row may
    # have failed at any time before its inspection, so a nonparametric
    # fit can put probability before any entry time, and no entry is
    # vacuous for it.
    keep = d["c"] != -1
    x = [xi for xi, k in zip(d["x"], keep) if k]
    c = d["c"][keep]
    plain = SurpyvalData(x, c, group_and_sort=False)
    # Entry before time 0 on half the rows: S(entry) = 1, a zero window
    # term, so the scores are exactly the untruncated ones.
    tl = np.where(np.arange(c.size) % 2 == 0, -1.0, -np.inf)
    vacuous = SurpyvalData(x, c, tl=tl, group_and_sort=False)
    np.testing.assert_allclose(
        log_rank_scores(vacuous), log_rank_scores(plain), atol=1e-8
    )
    # A common right truncation time past every observation shifts every
    # score by the same amount, so the split is the same.
    rt = _right_truncated(1, n=120, effect=0.5)
    exact = SurpyvalData(rt["x"], group_and_sort=False)
    common = SurpyvalData(rt["x"], tr=np.full(120, 1e3), group_and_sort=False)
    shift = log_rank_scores(common) - log_rank_scores(exact)
    np.testing.assert_allclose(shift, shift[0], atol=1e-8)
    assert turnbull_score_split(
        common, rt["Z"], 5, 2, [0, 1, 2]
    ) == turnbull_score_split(exact, rt["Z"], 5, 2, [0, 1, 2])


def _p_value(scores, data, z):
    failures = (np.asarray(data.c) != 1).astype(float)
    return max_statistic(scores[:, None], data.n, z, failures, 5, 2)[1]


def _event_scores_only(data):
    # The scores as if the rows were untruncated: what the split would
    # use without the window term.
    x = np.asarray(data.x, dtype=float)
    c = np.asarray(data.c)
    x = x if x.ndim == 2 else np.column_stack([x, x])
    lo = np.where(c == -1, -np.inf, x[:, 0])
    hi = np.where(c == 1, np.inf, x[:, 1])
    return _interval_scores(_H_at(data), lo, hi, c == 0)


@pytest.mark.parametrize(
    "maker, shift",
    [
        (_left_truncated_inspections, {"entry_shift": 4.0}),
        (_right_truncated, {"truncation_shift": 16.0}),
    ],
    ids=["delayed-entry", "right-truncation"],
)
def test_truncation_that_depends_on_a_null_covariate(maker, shift):
    # Feature 1 changes only the truncation, not the failure time. The
    # scores of the conditional likelihood have mean zero given the
    # window, so feature 1 is rarely significant; the event scores alone
    # are associated with it and find it far more often. (Rates at 0.05:
    # 0.005 against 0.39 with delayed entry, over 200 data sets of 200
    # rows; 0.03 against 0.49 with right truncation, over 100 of 150.)
    with_window, event_only = 0, 0
    reps = 40
    for seed in range(reps):
        d = maker(seed, n=150, **shift)
        data = _data(d)
        z = d["Z"][:, 1]
        with_window += _p_value(log_rank_scores(data), data, z) < 0.05
        event_only += _p_value(_event_scores_only(data), data, z) < 0.05
    assert with_window <= 5, (with_window, event_only)
    assert event_only >= with_window + 4, (with_window, event_only)


# -- trees and forests -----------------------------------------------------


@pytest.mark.parametrize(
    "maker", [_left_truncated_inspections, _right_truncated]
)
def test_tree_finds_a_planted_effect_with_turnbull_leaves(maker):
    d = maker(0, effect=0.5)
    tree = SurvivalTree.fit(
        **d,
        kind="non-parametric",
        max_depth=1,
        n_features_split="all",
        random_state=0,
    )
    root = tree._root
    assert root.split_feature_index == 0
    assert abs(root.split_feature_value - 0.5) < 0.1
    leaves = tree_leaves(root)
    with warnings.catch_warnings():
        # A leaf's Turnbull EM may report slow convergence.
        warnings.simplefilter("ignore", UserWarning)
        assert all(
            isinstance(leaf.model, NonParametric)
            and leaf.model.model == "Turnbull"
            for leaf in leaves
        )
    rows = np.array([[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    s = tree.sf([4.0, 8.0], rows, grid=True)
    assert np.all(s[1] < s[0])
    restored = SurvivalTree.from_dict(json.loads(json.dumps(tree.to_dict())))
    np.testing.assert_allclose(restored.sf([4.0, 8.0], rows, grid=True), s)


@pytest.mark.parametrize(
    "maker, shift",
    [
        (_left_truncated_inspections, {"entry_shift": 4.0}),
        (_right_truncated, {"truncation_shift": 4.0}),
    ],
    ids=["delayed-entry", "right-truncation"],
)
def test_ctree_mostly_stays_a_leaf_without_an_effect(maker, shift):
    # (Over 50 data sets: 48 single leaves with delayed entry, 45 with
    # right truncation; with an effect of 0.5 every tree splits.)
    leaves = 0
    for seed in range(12):
        tree = SurvivalTree.fit(
            **maker(seed, **shift),
            kind="non-parametric",
            n_features_split="all",
            selection="ctree",
        )
        leaves += isinstance(tree._root, TerminalNode)
    assert leaves >= 10, leaves
    tree = SurvivalTree.fit(
        **maker(0, effect=0.5, **shift),
        kind="non-parametric",
        n_features_split="all",
        selection="ctree",
        max_depth=1,
    )
    assert tree._root.split_feature_index == 0
    assert tree._root.p_value < 1e-3


def test_every_data_case_builds_a_non_parametric_tree():
    rng = np.random.default_rng(0)
    n = 80
    Z = rng.uniform(size=(n, 2))
    T = 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] > 0.5, 0.5, 1)
    lo, hi = np.floor(T), np.floor(T) + 1
    entry = np.minimum(0.3 * lo, 2.0)
    cases = {
        "left truncated interval": dict(xl=lo, xr=hi, tl=entry),
        "right truncated interval": dict(xl=lo, xr=hi, tr=hi + 3.0),
        "two-sided, every censoring": dict(
            x=np.where(np.arange(n) % 4 == 0, hi, T),
            c=np.where(np.arange(n) % 4 == 0, -1, 0),
            tl=entry * (np.arange(n) % 3 == 0) - (np.arange(n) % 3 != 0),
            tr=hi + 5.0,
        ),
    }
    for name, kw in cases.items():
        with warnings.catch_warnings():
            # A Turnbull leaf may warn that its NPMLE is not unique.
            warnings.simplefilter("ignore", UserWarning)
            tree = SurvivalTree.fit(
                Z=Z,
                kind="non-parametric",
                n_features_split="all",
                max_depth=2,
                random_state=0,
                **kw,
            )
            s = tree.sf([2.0, 6.0], Z[:5], grid=True)
        assert np.isfinite(s).all() and (np.diff(s, axis=1) <= 0).all(), name


def test_forest_on_right_truncated_data():
    d = _right_truncated(3, n=150, effect=0.5)
    oob = {}
    for depth in (0, 2):
        log = io.StringIO()
        with contextlib.redirect_stderr(log), warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            forest = RandomSurvivalForest.fit(
                **d,
                n_trees=10,
                max_depth=depth,
                kind="non-parametric",
                n_features_split="all",
                random_state=0,
            )
            oob[depth] = forest.oob_log_likelihood()
    assert np.isfinite(oob[0]) and np.isfinite(oob[2])
    assert oob[2] > oob[0], oob
    s = forest.sf(5.0, [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    assert s[1] < s[0]
