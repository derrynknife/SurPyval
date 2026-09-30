"""Conditional-inference selection, ``selection="ctree"`` (issue #188).

A ctree node tests each feature for association with the scores of the
tree's split statistic by its maximally selected statistic, chooses the
feature with the smallest p-value, and splits only if that p-value,
Bonferroni-adjusted, is below ``alpha_split``. These check:

- the p-value of the maximally selected statistic: exact against closed
  forms for two cuts, and against a Monte Carlo of the Gaussian chain for
  many; its size under the null, on real scores, for every kind;
- the scores: the parametric ones are the gradient of each row's
  log-likelihood at every censoring type and truncation; the
  non-parametric ones are the log-rank scores, and with delayed entry
  their sum over a child is the log-rank numerator O - E;
- the statistic is invariant to the row order, to counts against
  repeated rows, and to an increasing transformation of a feature;
- selection bias: an informative feature with two values beats noise
  features with many values far more often than under greedy search;
- on null data ctree mostly stops at the root, where greedy always
  splits; and it works on right-, left- and interval-censored data;
- depth and leaf-size limits still apply; ``selection="greedy"`` is
  today's tree; the options round-trip through serialisation; invalid
  options raise.
"""

import contextlib
import io
import json

import numpy as np
import pytest
from scipy import stats
from scipy.integrate import quad
from scipy.special import owens_t

from surpyval.beta.ml.forest import RandomSurvivalForest, SurvivalTree
from surpyval.beta.ml.forest.conditional_inference import (
    _MAX_TEST_CUTS,
    _pooled_mle,
    ctree_select,
    max_chain_sf,
    max_statistic,
    max_statistic_p_value,
    node_scores,
    nonparametric_scores,
    parametric_scores,
)
from surpyval.beta.ml.forest.log_rank_split import (
    at_risk_on_grid,
    deaths_on_grid,
)
from surpyval.beta.ml.forest.node import IntermediateNode, TerminalNode
from surpyval.beta.ml.forest.turnbull_score_split import (
    log_rank_scores,
    turnbull_score,
)
from surpyval.utils.surpyval_data import SurpyvalData


def _splits(node):
    if isinstance(node, TerminalNode):
        return []
    return (
        [(int(node.split_feature_index), float(node.split_feature_value))]
        + _splits(node.left_child)
        + _splits(node.right_child)
    )


def _depth(node):
    if isinstance(node, TerminalNode):
        return 0
    return 1 + max(_depth(node.left_child), _depth(node.right_child))


def _leaves(node):
    if isinstance(node, TerminalNode):
        return [node]
    return _leaves(node.left_child) + _leaves(node.right_child)


def _survival_data(seed, n=150, effect=1.0, censoring="right"):
    """Feature 0 takes two values and multiplies life by ``effect`` when
    it is 1; features 1-3 are continuous noise. Right censoring, or
    inspections every two time units (interval censored, left censored
    before the first, right censored after the last), or left censoring
    only (a failure before 3 is found at 3)."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.integers(0, 2, n), rng.uniform(size=(n, 3))])
    T = 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] == 1, effect, 1.0)
    if censoring == "right":
        C = rng.uniform(3, 25, n)
        return dict(x=np.minimum(T, C), c=(C < T).astype(int), Z=Z)
    if censoring == "left":
        return dict(x=np.where(T < 3, 3.0, T), c=np.where(T < 3, -1, 0), Z=Z)
    grid = np.arange(0.0, 26.0, 2.0)
    k = np.minimum(np.searchsorted(grid, T), grid.size - 1)
    c = np.where(k == 1, -1, np.where(T > 24, 1, 2))
    x = [
        grid[j] if cj == -1 else 24.0 if cj == 1 else [grid[j - 1], grid[j]]
        for j, cj in zip(k, c)
    ]
    return dict(x=x, c=c, Z=Z)


def _root_feature(**kwargs):
    tree = SurvivalTree.fit(max_depth=1, n_features_split="all", **kwargs)
    root = tree._root
    if isinstance(root, TerminalNode):
        return -1
    return int(root.split_feature_index)


# -- the p-value of the maximally selected statistic ----------------------


def _two_cuts_exact(b, rho, q):
    # P(max(Q1, Q2) >= b) = P(Q1 >= b) + P(Q1 < b <= Q2): for one score
    # by Owen's T (P(Z1 >= h, Z2 >= h) = Phi(-h) - 2 T(h, a)), for two by
    # adaptive quadrature over the non-central chi law of Q2 given Q1.
    tail = stats.chi2.sf(b, q)
    if q == 1:
        h = np.sqrt(b)
        a = np.sqrt((1 - rho) / (1 + rho))
        both = 4 * stats.norm.sf(h) - 4 * (owens_t(h, a) + owens_t(h, 1 / a))
        return 2 * tail - both
    s2 = 1 - rho**2

    def crossing(w):
        return stats.chi.pdf(w, q) * stats.ncx2.sf(
            b / s2, q, rho**2 * w**2 / s2
        )

    return tail + quad(crossing, 0, np.sqrt(b), epsabs=1e-14, limit=200)[0]


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize("b", [1.0, 4.0, 9.0, 25.0])
@pytest.mark.parametrize("rho", [0.1, 0.5, 0.9, 0.99, 0.999])
def test_two_cuts_are_exact(q, b, rho):
    assert max_chain_sf(b, np.array([rho]), q) == pytest.approx(
        _two_cuts_exact(b, rho, q), rel=1e-9
    )


@pytest.mark.parametrize("q", [1, 2])
def test_one_cut_is_the_chi_square_p_value(q):
    assert max_statistic_p_value(5.0, np.array([40.0]), 100.0, q) == (
        pytest.approx(stats.chi2.sf(5.0, q))
    )


@pytest.mark.parametrize("q", [1, 2])
@pytest.mark.parametrize(
    "cuts",
    [
        np.arange(5, 196),  # every cut of a continuous feature
        np.round(np.linspace(5, 195, 64)).astype(int),
        np.arange(20, 181, 20),
        np.array([60, 130]),
    ],
    ids=["191 cuts", "64 cuts", "9 cuts", "2 cuts"],
)
def test_p_value_matches_the_gaussian_chain(q, cuts):
    # The standardised statistics of a random walk's partial sums (its
    # scores centred) at the cuts: the maximum's tail probability at its
    # 90%, 95% and 99% points, against the p-value.
    rng = np.random.default_rng(0)
    N, R = 200, 20000
    e = rng.standard_normal((R, N, q))
    S = np.cumsum(e - e.mean(1, keepdims=True), axis=1)[:, cuts - 1]
    Q = (S**2).sum(-1) / (cuts * (N - cuts) / N)
    largest = Q.max(1)
    for level in [0.1, 0.05, 0.01]:
        b = np.quantile(largest, 1 - level)
        p = max_statistic_p_value(b, cuts, N, q)
        # Monte Carlo error: about 3 standard errors
        assert abs(p - level) < 3 * np.sqrt(level * (1 - level) / R), (b, p)


@pytest.mark.parametrize(
    "kind, censoring, card",
    [
        ("non-parametric", "right", "continuous"),
        ("non-parametric", "right", "three values"),
        ("non-parametric", "interval", "continuous"),
        ("exponential", "right", "continuous"),
        ("exponential", "interval", "continuous"),
        ("weibull", "right", "continuous"),
    ],
)
def test_p_value_has_its_size_under_the_null(kind, censoring, card):
    # A feature drawn independently of the data: the p-value of its
    # maximally selected statistic should be uniform. 20 data sets (10 for
    # the slower two-score Weibull), each scored once, times 20 features.
    p_values = []
    for seed in range(10 if kind == "weibull" else 20):
        d = _survival_data(seed, censoring=censoring)
        data = SurpyvalData(d["x"], d["c"], group_and_sort=False)
        scores = node_scores(data, kind)
        failures = (np.asarray(data.c) != 1).astype(float)
        rng = np.random.default_rng(1000 + seed)
        for _ in range(20):
            z = (
                rng.uniform(size=len(failures))
                if card == "continuous"
                else rng.integers(0, 3, len(failures)).astype(float)
            )
            p_values.append(
                max_statistic(scores, data.n, z, failures, 5, 2)[1]
            )
    p_values = np.array(p_values)
    for level in [0.01, 0.05, 0.1, 0.5]:
        rate = np.mean(p_values <= level)
        sd = np.sqrt(level * (1 - level) / p_values.size)
        assert abs(rate - level) < 3.5 * sd + 0.005, (level, rate)
    assert stats.kstest(p_values, "uniform").pvalue > 1e-3
    # Rejection rates at 0.05 in these six cases: 0.033 to 0.055, and
    # 0.075 for the Weibull (200 p-values; 0.061 over 1000).


# -- the scores -------------------------------------------------------------


def _mixed_data():
    # Every censoring type, with left truncation on some rows and right
    # truncation on others.
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
            [-np.inf, 8.0],
            [0.2, np.inf],
            [-np.inf, np.inf],
            [1.0, 12.0],
        ]
    )
    return SurpyvalData(x, c, t=t, group_and_sort=False)


def _row_log_likelihood(theta, data, kind):
    # Each row's log-likelihood, from the closed-form S and f.
    if kind == "weibull":
        alpha, beta = np.exp(theta)
    else:
        alpha, beta = np.exp(-theta[0]), 1.0

    def S(q):
        q = np.clip(q, 0, None)
        return np.exp(-((q / alpha) ** beta))

    def f(q):
        return beta / alpha * (q / alpha) ** (beta - 1) * S(q)

    x = np.asarray(data.x, dtype=float)
    lo, hi, c = x[:, 0], x[:, -1], data.c
    with np.errstate(divide="ignore"):
        return _choose(c, lo, hi, S, f) - np.log(
            S(data.t[:, 0]) - S(data.t[:, 1])
        )


def _choose(c, lo, hi, S, f):
    return np.where(
        c == 0,
        np.log(f(lo)),
        np.where(
            c == 1,
            np.log(S(lo)),
            np.where(c == -1, np.log1p(-S(hi)), np.log(S(lo) - S(hi))),
        ),
    )


@pytest.mark.parametrize("kind", ["exponential", "weibull"])
def test_parametric_scores_are_the_log_likelihood_gradient(kind):
    data = _mixed_data()
    theta = _pooled_mle(data, kind)
    scores = parametric_scores(data, kind)
    assert scores.shape == (10, theta.size)
    step = 1e-6
    for j in range(theta.size):
        e = np.zeros(theta.size)
        e[j] = step
        numeric = (
            _row_log_likelihood(theta + e, data, kind)
            - _row_log_likelihood(theta - e, data, kind)
        ) / (2 * step)
        np.testing.assert_allclose(scores[:, j], numeric, rtol=1e-6, atol=1e-8)
    # At the pooled maximum the scores sum to zero.
    np.testing.assert_allclose(scores.sum(0), 0, atol=1e-4)


def test_exponential_scores_on_right_censored_data():
    # delta - lambda x: the parametric counterpart of the log-rank score
    rng = np.random.default_rng(0)
    x = rng.exponential(5, 50)
    c = (rng.random(50) < 0.3).astype(int)
    data = SurpyvalData(x, c, group_and_sort=False)
    lam = (c == 0).sum() / x.sum()
    np.testing.assert_allclose(
        parametric_scores(data, "exponential")[:, 0],
        (c == 0) - lam * x,
        rtol=1e-5,
        atol=1e-6,
    )


def test_nonparametric_scores_are_the_log_rank_scores():
    d = _survival_data(0, censoring="interval")
    data = SurpyvalData(d["x"], d["c"], group_and_sort=False)
    np.testing.assert_array_equal(
        nonparametric_scores(data)[:, 0], log_rank_scores(data)
    )


def test_delayed_entry_scores_sum_to_the_log_rank_numerator():
    # With left truncation the score is delta - [H(x) - H(entry)], H the
    # Nelson-Aalen estimate of the delayed-entry risk sets, and its sum
    # over a child is O - E of the log-rank test with those risk sets.
    rng = np.random.default_rng(2)
    n = 120
    entry = rng.uniform(0, 3, n)
    T = entry + 8 * rng.weibull(1.5, n)
    C = entry + rng.uniform(2, 20, n)
    Z = rng.uniform(size=(n, 1))
    data = SurpyvalData(
        np.minimum(T, C),
        (C < T).astype(int),
        tl=entry,
        group_and_sort=False,
    )
    scores = nonparametric_scores(data)[:, 0]
    grid, Y, d = data.to_xrd()
    assert scores.sum() == pytest.approx(0, abs=1e-8)
    for v in [0.3, 0.5, 0.8]:
        left = Z[:, 0] <= v
        o_minus_e = np.sum(
            deaths_on_grid(data[left], grid)
            - at_risk_on_grid(data[left], grid) * d / Y
        )
        assert scores[left].sum() == pytest.approx(o_minus_e, abs=1e-8)


# -- the statistic ----------------------------------------------------------


def test_statistic_is_the_square_of_the_turnbull_score_split_statistic():
    d = _survival_data(1, n=60, censoring="interval")
    data = SurpyvalData(d["x"], d["c"], group_and_sort=False)
    Z = d["Z"]
    scores = nonparametric_scores(data)
    failures = (np.asarray(data.c) != 1).astype(float)
    b, _, n_cuts = max_statistic(scores, data.n, Z[:, 1], failures, 1, 0)
    assert n_cuts == np.unique(Z[:, 1]).size - 1 <= _MAX_TEST_CUTS
    best = max(turnbull_score(1, v, data, Z) for v in np.unique(Z[:, 1])[:-1])
    assert b == pytest.approx(best**2, rel=1e-10)


def _statistic(data, z, kind="non-parametric"):
    scores = node_scores(data, kind)
    failures = (np.asarray(data.c) != 1).astype(float)
    return max_statistic(scores, data.n, z, failures, 5, 2)


@pytest.mark.parametrize("kind", ["non-parametric", "exponential", "weibull"])
def test_statistic_ignores_the_row_order(kind):
    d = _survival_data(3, effect=0.7)
    data = SurpyvalData(d["x"], d["c"], group_and_sort=False)
    order = np.random.default_rng(0).permutation(len(d["c"]))
    shuffled = SurpyvalData(d["x"][order], d["c"][order], group_and_sort=False)
    for j in range(4):
        a = _statistic(data, d["Z"][:, j], kind)
        b = _statistic(shuffled, d["Z"][order, j], kind)
        np.testing.assert_allclose(a, b, rtol=1e-6)


def test_statistic_counts_equal_repetition():
    d = _survival_data(4, n=60, effect=0.6)
    n = np.random.default_rng(1).integers(1, 4, 60)
    counted = SurpyvalData(d["x"], d["c"], n=n, group_and_sort=False)
    rows = np.repeat(np.arange(60), n)
    repeated = SurpyvalData(d["x"][rows], d["c"][rows], group_and_sort=False)
    for j in range(4):
        z = d["Z"][:, j]
        # The same cuts and the same statistic; the leaf-size limits
        # count rows, so compare with limits that do not bind.
        scores_c = node_scores(counted, "non-parametric")
        scores_r = node_scores(repeated, "non-parametric")
        fc = (counted.c != 1).astype(float)
        fr = (repeated.c != 1).astype(float)
        a = max_statistic(scores_c, counted.n, z, fc, 1, 0)
        b = max_statistic(scores_r, repeated.n, z[rows], fr, 1, 0)
        np.testing.assert_allclose(a, b, rtol=1e-8)


def test_statistic_ignores_an_increasing_transformation():
    d = _survival_data(5, effect=0.7)
    data = SurpyvalData(d["x"], d["c"], group_and_sort=False)
    z = d["Z"][:, 1]
    np.testing.assert_allclose(
        _statistic(data, z), _statistic(data, np.exp(10 * z)), rtol=1e-12
    )


# -- selection --------------------------------------------------------------

N_REPLICATES = 60


def test_ctree_removes_the_preference_for_many_valued_features():
    # A weak effect of a two-valued feature against three continuous
    # noise features, each offering about 140 cuts to its one. The root's
    # feature over 60 seeded data sets, with alpha_split=1 so that ctree
    # always makes its choice (only its choice is compared here): greedy
    # picks the informative feature in 53% of them (a noise feature in
    # 47%), ctree in 88% (a noise feature in 10%).
    picks = {"greedy": [], "ctree": []}
    for seed in range(N_REPLICATES):
        d = _survival_data(seed, effect=0.7)
        for selection in picks:
            picks[selection].append(
                _root_feature(
                    **d,
                    kind="non-parametric",
                    selection=selection,
                    alpha_split=1.0,
                )
            )
    greedy = np.mean(np.array(picks["greedy"]) == 0)
    ctree = np.mean(np.array(picks["ctree"]) == 0)
    assert greedy < 0.7, greedy
    assert ctree > 0.8, ctree
    assert ctree > greedy + 0.2, (greedy, ctree)
    assert np.mean(np.array(picks["ctree"]) > 0) < 0.15


@pytest.mark.parametrize("kind", ["non-parametric", "exponential"])
def test_ctree_mostly_stops_at_the_root_on_null_data(kind):
    # No effect: greedy splits every time; ctree's root, with four
    # features Bonferroni-adjusted at alpha_split = 0.05, splits in at
    # most about 5% of data sets (here 1 of 60 for the non-parametric
    # kind and 0 of 60 for the exponential; over 400 data sets, 3.8% and
    # 3.5%, Bonferroni being slightly conservative).
    greedy, ctree = [], []
    for seed in range(N_REPLICATES):
        d = _survival_data(100 + seed)
        greedy.append(_root_feature(**d, kind=kind, selection="greedy") != -1)
        ctree.append(_root_feature(**d, kind=kind, selection="ctree") != -1)
    assert all(greedy)
    assert np.mean(ctree) <= 0.1, np.mean(ctree)


@pytest.mark.parametrize("censoring", ["right", "left", "interval"])
@pytest.mark.parametrize("kind", ["non-parametric", "exponential", "weibull"])
def test_ctree_works_on_every_censoring_type(kind, censoring):
    # A strong effect of feature 0 is found (and cut between its two
    # values); the same data without the effect is left unsplit.
    d = _survival_data(7, n=200, effect=0.4, censoring=censoring)
    tree = SurvivalTree.fit(
        **d, kind=kind, n_features_split="all", selection="ctree"
    )
    root = tree._root
    assert isinstance(root, IntermediateNode)
    assert int(root.split_feature_index) == 0
    assert 0 <= root.split_feature_value < 1
    assert root.p_value < 1e-3
    null = _survival_data(8, n=200, censoring=censoring)
    tree = SurvivalTree.fit(
        **null, kind=kind, n_features_split="all", selection="ctree"
    )
    assert isinstance(tree._root, TerminalNode)


def test_ctree_on_left_truncated_data():
    rng = np.random.default_rng(3)
    n = 200
    entry = rng.uniform(0, 3, n)
    Z = np.column_stack([rng.integers(0, 2, n), rng.uniform(size=n)])
    T = entry + 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] == 1, 0.4, 1)
    C = entry + rng.uniform(2, 25, n)
    fit = dict(
        x=np.minimum(T, C),
        c=(C < T).astype(int),
        tl=entry,
        n_features_split="all",
        selection="ctree",
        max_depth=1,
    )
    for kind in ["non-parametric", "exponential"]:
        root = SurvivalTree.fit(Z=Z, kind=kind, **fit)._root
        assert int(root.split_feature_index) == 0
        # Only noise: no split
        root = SurvivalTree.fit(Z=Z[:, 1:], kind=kind, **fit)._root
        assert isinstance(root, TerminalNode)


def test_the_other_stopping_rules_still_apply():
    d = _survival_data(9, n=300, effect=0.3)
    fit = dict(kind="non-parametric", n_features_split="all")
    tree = SurvivalTree.fit(**d, **fit, selection="ctree", max_depth=1)
    assert _depth(tree._root) == 1
    tree = SurvivalTree.fit(**d, **fit, selection="ctree", min_leaf_samples=40)
    assert min(len(leaf.data) for leaf in _leaves(tree._root)) >= 40
    # A stricter alpha_split never gives a bigger tree.
    sizes = [
        len(
            _leaves(
                SurvivalTree.fit(
                    **d, **fit, alpha_split=a, selection="ctree"
                )._root
            )
        )
        for a in [0.5, 0.05, 1e-6, 1e-30]
    ]
    assert sizes == sorted(sizes, reverse=True) and sizes[-1] == 1, sizes


def test_bonferroni_counts_the_features_tested():
    d = _survival_data(10, effect=0.7)
    data = SurpyvalData(d["x"], d["c"], group_and_sort=False)
    one = ctree_select(data, d["Z"], "non-parametric", 5, 2, [0])
    four = ctree_select(data, d["Z"], "non-parametric", 5, 2, range(4))
    assert one[0] == four[0] == 0
    assert four[1] == pytest.approx(min(1.0, 4 * one[1]))


@pytest.mark.filterwarnings("ignore:.*in the sample of every tree")
def test_ctree_forest():
    d = _survival_data(11, effect=0.5)
    with contextlib.redirect_stderr(io.StringIO()):
        forest = RandomSurvivalForest.fit(
            **d,
            kind="non-parametric",
            n_trees=10,
            selection="ctree",
            random_state=0,
        )
    assert forest.selection == "ctree"
    for tree in forest.trees:
        assert tree.selection == "ctree"
    # Every split any tree made is on the informative feature or was
    # significant; the noise features are rarely used.
    features = [f for tree in forest.trees for f, _ in _splits(tree._root)]
    assert features.count(0) > len(features) / 2
    assert forest.oob_log_likelihood() > -np.inf


# -- greedy is unchanged; serialisation; validation -------------------------

# Splits of seeded greedy trees (np.random.seed(4), max_depth=2,
# n_features_split=2) on the data of ``_greedy_data``, from the code
# before ``selection`` existed (793cad6).
GREEDY_SPLITS = {
    ("right", "non-parametric"): [
        (0, 1.0),
        (2, 0.5125974559554912),
        (2, 0.39297044395520897),
    ],
    ("right", "exponential"): [(0, 1.0), (0, 0.0), (1, 0.4045937288478155)],
    ("right", "weibull"): [
        (0, 1.0),
        (2, 0.7468603856498379),
        (1, 0.3742438334784708),
    ],
    ("interval", "non-parametric"): [
        (0, 1.0),
        (2, 0.25375519476203323),
        (2, 0.39297044395520897),
    ],
    ("interval", "exponential"): [
        (0, 1.0),
        (2, 0.19663923977212372),
        (2, 0.39297044395520897),
    ],
    ("interval", "weibull"): [
        (0, 1.0),
        (2, 0.19663923977212372),
        (1, 0.6827989078603502),
    ],
}

# and the tree's sf at x = [2, 6] for Z = [0, 0.2, 0.7] and [2, 0.8, 0.1]
GREEDY_SF = {
    ("right", "non-parametric"): [
        [0.9394130628134758, 0.6918214299103509],
        [0.8668778997501816, 0.600781597182098],
    ],
    ("right", "weibull"): [
        [0.914864449994829, 0.6482164543192853],
        [0.886889236779209, 0.39846524921481696],
    ],
    ("interval", "exponential"): [
        [0.8300143680933468, 0.5718166950525637],
        [0.7395999997403062, 0.40456723470983613],
    ],
}


def _greedy_data(censoring, n=60, seed=3):
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.integers(0, 3, n), rng.uniform(size=(n, 2))])
    T = 10 * rng.weibull(1.5, n) * np.where(Z[:, 0] == 2, 0.5, 1.0)
    if censoring == "right":
        C = rng.uniform(3, 25, n)
        return dict(x=np.minimum(T, C), c=(C < T).astype(int), Z=Z)
    lo = np.floor(T)
    return dict(x=np.column_stack([lo, lo + 1]), c=np.full(n, 2), Z=Z)


@pytest.mark.parametrize("censoring, kind", sorted(GREEDY_SPLITS))
@pytest.mark.parametrize("explicit", [False, True])
def test_greedy_is_todays_tree(censoring, kind, explicit):
    options = {"selection": "greedy"} if explicit else {}
    np.random.seed(4)
    tree = SurvivalTree.fit(
        **_greedy_data(censoring),
        kind=kind,
        max_depth=2,
        n_features_split=2,
        **options,
    )
    assert _splits(tree._root) == GREEDY_SPLITS[censoring, kind]
    if (censoring, kind) in GREEDY_SF:
        sf = tree.sf([2.0, 6.0], [[0.0, 0.2, 0.7], [2.0, 0.8, 0.1]])
        np.testing.assert_allclose(sf, GREEDY_SF[censoring, kind], rtol=1e-9)
    assert all(
        getattr(node, "p_value", None) is None
        for node in [tree._root, tree._root.left_child]
    )


def test_ctree_options_round_trip():
    d = _survival_data(12, effect=0.4)
    tree = SurvivalTree.fit(
        **d,
        kind="exponential",
        n_features_split="all",
        selection="ctree",
        alpha_split=0.01,
    )
    blob = json.loads(json.dumps(tree.to_dict()))
    assert blob["selection"] == "ctree" and blob["alpha_split"] == 0.01
    restored = SurvivalTree.from_dict(blob)
    assert restored.selection == "ctree" and restored.alpha_split == 0.01
    assert restored._root.p_value == tree._root.p_value < 0.01
    x, Z = [1.0, 5.0], d["Z"][:6]
    np.testing.assert_array_equal(restored.sf(x, Z), tree.sf(x, Z))

    with contextlib.redirect_stderr(io.StringIO()):
        forest = RandomSurvivalForest.fit(
            **d,
            kind="exponential",
            n_trees=3,
            selection="ctree",
            alpha_split=0.2,
            random_state=1,
        )
    blob = json.loads(json.dumps(forest.to_dict()))
    restored = RandomSurvivalForest.from_dict(blob)
    assert restored.selection == "ctree" and restored.alpha_split == 0.2
    np.testing.assert_array_equal(restored.sf(x, Z), forest.sf(x, Z))


def test_a_tree_saved_before_selection_reads_as_greedy():
    np.random.seed(0)
    tree = SurvivalTree.fit(**_survival_data(13), kind="exponential")
    blob = tree.to_dict()
    del blob["selection"], blob["alpha_split"]
    restored = SurvivalTree.from_dict(blob)
    assert restored.selection == "greedy"


@pytest.mark.parametrize(
    "options, match",
    [
        ({"selection": "cforest"}, "selection"),
        ({"selection": "ctree", "alpha_split": 0.0}, "alpha_split"),
        ({"selection": "ctree", "alpha_split": 1.5}, "alpha_split"),
        ({"selection": "ctree", "alpha_split": "0.05"}, "alpha_split"),
    ],
)
def test_invalid_options_raise(options, match):
    d = _survival_data(14, n=30)
    with pytest.raises(ValueError, match=match):
        SurvivalTree.fit(**d, kind="exponential", **options)
    with pytest.raises(ValueError, match=match):
        RandomSurvivalForest.fit(**d, kind="exponential", n_trees=2, **options)
