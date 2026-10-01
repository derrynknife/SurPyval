"""The Turnbull-score split of the non-parametric tree (issue #188).

``kind="non-parametric"`` splits a node with left- or interval-censored
rows by the standardised sum of the left child's log-rank scores under
the node's pooled Turnbull estimate. These check:

- the score formula by hand at every censoring type, on data whose
  NPMLE is known in closed form;
- the reduction on right-censored data: the scores are the classic
  log-rank (Savage) scores ``delta - H_NA``, their sum over a child is
  the log-rank numerator ``O - E`` at every candidate, and the split
  chooses as the log-rank split does;
- the permutation variance by hand;
- a tree finds a planted split on interval-censored data, with Turnbull
  leaves;
- the forest end to end, and its out-of-bag log-likelihood (#186) beats
  the no-split forest when there is an effect;
- truncated interval-censored data are split too (stage 2; see
  test_turnbull_score_truncation.py).
"""

import contextlib
import io
import warnings

import numpy as np
import pytest

from surpyval import NelsonAalen
from surpyval.beta.ml.forest import RandomSurvivalForest, SurvivalTree
from surpyval.beta.ml.forest.log_rank_split import (
    at_risk_on_grid,
    deaths_on_grid,
    log_rank_split,
)
from surpyval.beta.ml.forest.node import TerminalNode
from surpyval.beta.ml.forest.turnbull_score_split import (
    log_rank_scores,
    turnbull_score,
    turnbull_score_split,
)
from surpyval.univariate.nonparametric.nonparametric import NonParametric
from surpyval.utils.surpyval_data import SurpyvalData


def _leaves(node):
    if isinstance(node, TerminalNode):
        return [node]
    return _leaves(node.left_child) + _leaves(node.right_child)


def _right_censored(n=80, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, 3))
    T = 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] > 0.5, 0.6, 1.0)
    C = rng.uniform(3, 25, n)
    data = SurpyvalData(
        np.minimum(T, C), (C < T).astype(int), group_and_sort=False
    )
    return data, Z


def _inspection_data(n=200, seed=0, factor=0.4, binary=False):
    """Units inspected every 2 time units up to 20: a failure is known
    only to lie between two inspections (interval censored), before the
    first (left censored) or after the last (right censored). Feature 0
    above 0.5 multiplies life by ``factor``; the others are noise."""
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 1, (n, 3))
    if binary:
        Z[:, 0] = rng.integers(0, 2, n)
    T = 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] > 0.5, factor, 1.0)
    grid = np.arange(0.0, 22.0, 2.0)
    k = np.minimum(np.searchsorted(grid, T), grid.size - 1)
    c = np.where(k == 1, -1, np.where(T > 20, 1, 2))
    x = [
        grid[j] if ci == -1 else (20.0 if ci == 1 else [grid[j - 1], grid[j]])
        for j, ci in zip(k, c)
    ]
    return x, Z, c


# -- the scores ----------------------------------------------------------


def test_scores_by_hand_at_every_censoring_type():
    # Left censored at 1, interval (2, 3], observed at 4, right censored
    # at 5: disjoint supports, so the NPMLE puts 1/4 on each and its
    # Nelson-Aalen hazard steps by 1/4, 1/3 and 1/2.
    data = SurpyvalData([1, [2, 3], 4, 5], [-1, 2, 0, 1], group_and_sort=False)
    H1, H3, H4 = 1 / 4, 1 / 4 + 1 / 3, 1 / 4 + 1 / 3 + 1 / 2
    S1, S3 = np.exp(-H1), np.exp(-H3)
    expected = [
        # left: L -> 0 limit, -S(R) log S(R) / (1 - S(R))
        H1 * S1 / (1 - S1),
        # interval: [S(L) log S(L) - S(R) log S(R)] / [S(L) - S(R)]
        (-H1 * S1 + H3 * S3) / (S1 - S3),
        # observed: 1 + log S(t)
        1 - H4,
        # right: log S(L)
        -H4,
    ]
    np.testing.assert_allclose(log_rank_scores(data), expected, rtol=1e-8)


def test_scores_are_the_savage_scores_on_right_censored_data():
    # With ties: the NPMLE is the Kaplan-Meier, whose risk sets and
    # events are the observed ones, so H is the Nelson-Aalen exactly.
    rng = np.random.default_rng(0)
    x = np.round(10 * rng.weibull(1.5, 60), 0) + 1
    c = (rng.random(60) < 0.3).astype(int)
    data = SurpyvalData(x, c, group_and_sort=False)
    H = NelsonAalen.fit(x, c).Hf(x)
    np.testing.assert_allclose(log_rank_scores(data), (c == 0) - H, atol=1e-8)


@pytest.mark.parametrize("seed", range(3))
def test_score_sum_is_the_log_rank_numerator(seed):
    data, Z = _right_censored(seed=seed)
    scores = log_rank_scores(data)
    grid, Y, d = data.to_xrd()
    for v in np.quantile(Z[:, 1], [0.2, 0.5, 0.8]):
        left = Z[:, 1] <= v
        observed_minus_expected = np.sum(
            deaths_on_grid(data[left], grid)
            - at_risk_on_grid(data[left], grid) * d / Y
        )
        # the scores sum to zero, so no centring is needed
        assert scores.sum() == pytest.approx(0, abs=1e-6)
        assert scores[left].sum() == pytest.approx(
            observed_minus_expected, abs=1e-6
        )


def test_permutation_variance_by_hand():
    # Four interval-censored rows, one feature; the statistic at a cut is
    # |sum of the left child's scores - n_L c_bar| over the square root of
    # n_L n_R / (n (n - 1)) sum (c - c_bar)^2.
    data = SurpyvalData(
        [[0, 2], [1, 3], [2, 5], [4, 6]], [2, 2, 2, 2], group_and_sort=False
    )
    Z = np.array([[1.0], [2.0], [3.0], [4.0]])
    c = log_rank_scores(data)
    c_bar = c.mean()
    for v, n_left in [(1.0, 1), (2.0, 2), (3.0, 3)]:
        variance = n_left * (4 - n_left) / 12 * np.sum((c - c_bar) ** 2)
        expected = abs(c[:n_left].sum() - n_left * c_bar) / np.sqrt(variance)
        assert turnbull_score(0, v, data, Z) == pytest.approx(expected)
    # With two rows per child required, the only admissible cut is at 2.
    assert turnbull_score_split(data, Z, 2, 1, [0]) == (0, 2.0)


def test_split_maximises_the_statistic():
    x, Z, c = _inspection_data(n=60, seed=3)
    data = SurpyvalData(x, c, group_and_sort=False)
    u, v = turnbull_score_split(data, Z, 5, 2, [0, 1, 2])
    best = turnbull_score(u, v, data, Z)
    for f in range(3):
        for value in np.unique(Z[:, f]):
            left = Z[:, f] <= value
            if min(left.sum(), (~left).sum()) < 5:
                continue
            if min((c[left] != 1).sum(), (c[~left] != 1).sum()) < 2:
                continue
            assert turnbull_score(f, value, data, Z) <= best + 1e-9


def test_count_weights_equal_repeated_rows():
    x, Z, c = _inspection_data(n=60, seed=4)
    n = np.where(np.arange(60) % 3 == 0, 2, 1)
    weighted = SurpyvalData(x, c, n, group_and_sort=False)
    rows = np.repeat(np.arange(60), n)
    repeated = SurpyvalData(
        [x[i] for i in rows], c[rows], group_and_sort=False
    )
    np.testing.assert_allclose(
        log_rank_scores(weighted)[rows], log_rank_scores(repeated), atol=1e-6
    )


# -- the split on right-censored data: the log-rank's choices ------------


def test_split_agrees_with_the_log_rank_split_on_right_censored_data():
    # The numerators agree exactly; the permutation variance and the
    # log-rank's hypergeometric one agree asymptotically, so the choices
    # agree on most samples and are close on the rest.
    same = 0
    for seed in range(12):
        data, Z = _right_censored(n=150, seed=seed)
        a = log_rank_split(data, Z, 5, 2, [0, 1, 2])
        b = turnbull_score_split(data, Z, 5, 2, [0, 1, 2])
        assert b[0] == a[0] == 0
        same += bool(np.isclose(a[1], b[1]))
        assert abs(a[1] - b[1]) < 0.25
    assert same >= 8


def test_no_admissible_split_returns_the_sentinel():
    data, Z = _right_censored(n=8)
    assert turnbull_score_split(data, Z, 5, 2, [0, 1, 2]) == (
        -1,
        -float("inf"),
    )


# -- the tree and forest on interval-censored data -----------------------


def test_tree_finds_a_planted_binary_split():
    x, Z, c = _inspection_data(binary=True)
    assert set(c) == {-1, 1, 2}
    np.random.seed(0)
    tree = SurvivalTree.fit(
        x=x,
        Z=Z,
        c=c,
        kind="non-parametric",
        max_depth=1,
        n_features_split="all",
    )
    assert tree._root.split_feature_index == 0
    assert tree._root.split_feature_value == 0.0
    assert all(
        isinstance(leaf.model, NonParametric)
        and leaf.model.model == "Turnbull"
        for leaf in _leaves(tree._root)
    )
    s = tree.sf(6.0, [[0.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
    assert s[1] < s[0]


def test_tree_finds_a_planted_threshold():
    x, Z, c = _inspection_data(n=300, seed=1)
    np.random.seed(0)
    tree = SurvivalTree.fit(
        x=x,
        Z=Z,
        c=c,
        kind="non-parametric",
        max_depth=1,
        n_features_split="all",
    )
    assert tree._root.split_feature_index == 0
    assert abs(tree._root.split_feature_value - 0.5) < 0.1


def test_forest_end_to_end_and_out_of_bag_gain():
    x, Z, c = _inspection_data(n=200, seed=2)
    scores = {}
    for depth in (0, 2):
        np.random.seed(0)
        with contextlib.redirect_stderr(io.StringIO()):
            forest = RandomSurvivalForest.fit(
                x=x,
                Z=Z,
                c=c,
                n_trees=20,
                max_depth=depth,
                kind="non-parametric",
                n_features_split="all",
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            scores[depth] = forest.oob_log_likelihood()
    assert np.isfinite(scores[0]) and np.isfinite(scores[2])
    assert scores[2] > scores[0] + 0.03, scores
    # predictions: the effect is recovered
    s = forest.sf(6.0, [[0.2, 0.5, 0.5], [0.8, 0.5, 0.5]])
    assert s[1] < s[0]
    importance = forest.feature_importances(random_state=0)
    assert importance["Z0"] > 0.03
    assert importance["Z0"] > np.abs(importance.iloc[1:]).max()


@pytest.mark.parametrize(
    "truncation",
    [{"tl": 0.5}, {"tr": 25.0}, {"tl": 0.5, "tr": 25.0}],
    ids=["left", "right", "both"],
)
def test_truncated_interval_data_is_split(truncation):
    # Raised until stage 2 (see test_turnbull_score_truncation.py).
    x, Z, c = _inspection_data(n=200, binary=True)
    tree = SurvivalTree.fit(
        x=x,
        Z=Z,
        c=c,
        kind="non-parametric",
        max_depth=1,
        n_features_split="all",
        random_state=0,
        **truncation,
    )
    assert tree._root.split_feature_index == 0


def test_left_truncated_right_censored_data_keeps_the_log_rank():
    data, Z = _right_censored(n=60)
    np.random.seed(0)
    tree = SurvivalTree.fit(
        x=data.x,
        Z=Z,
        c=data.c,
        tl=0.1 * data.x,
        kind="non-parametric",
        max_depth=1,
    )
    assert all(
        leaf.model.model == "Nelson-Aalen" for leaf in _leaves(tree._root)
    )
