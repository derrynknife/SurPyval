"""Behavioural and structural tests for the survival tree and forest.

These pin the properties a user relies on but that the split/leaf
correctness tests do not touch:

- prediction coherence: ``ff = 1 - sf``, ``Hf = -log(sf)``, ``sf``
  monotone non-increasing within ``[0, 1]``, non-negative ``df``/``hf``;
- structural guarantees: ``max_depth``, ``min_leaf_samples`` and
  ``min_leaf_failures`` are honoured by every leaf;
- determinism: the same numpy seed builds the same tree;
- degenerate inputs: all-censored data, constant covariates, tiny
  samples, tied event times and count weights (``n``) all build valid
  trees rather than crashing;
- forest ensemble maths: the forest ``sf`` is exactly the tree average,
  the ``"Hf"`` ensemble method averages cumulative hazards, prediction
  shapes for single/multiple covariate vectors, mortality ordering and
  a concordance sanity check;
- a depth-2 tree recovers a two-feature interaction.
"""

import numpy as np
import pytest

from surpyval import Weibull
from surpyval.ml.forest import RandomSurvivalForest, SurvivalTree
from surpyval.ml.forest.node import IntermediateNode, TerminalNode
from surpyval.univariate.parametric import NeverOccurs

X_GRID = np.linspace(0.25, 25.0, 12)
Z_FAST = np.array([1.0, 0.0])
Z_SLOW = np.array([0.0, 0.0])


def _signal_data(n=100, seed=5, censoring=0.2):
    """Feature 0 is a strong binary signal (scale 3 vs 12), feature 1
    is noise."""
    rng = np.random.default_rng(seed)
    Z = np.column_stack(
        [rng.integers(0, 2, n).astype(float), rng.normal(0, 1, n)]
    )
    scale = np.where(Z[:, 0] > 0.5, 3.0, 12.0)
    x = scale * rng.weibull(1.5, n)
    c = (rng.random(n) < censoring).astype(int)
    return x, Z, c


def _leaves(node):
    if isinstance(node, TerminalNode):
        return [node]
    return _leaves(node.left_child) + _leaves(node.right_child)


def _tree_depth(node):
    if isinstance(node, TerminalNode):
        return 0
    return 1 + max(_tree_depth(node.left_child), _tree_depth(node.right_child))


def _structure(node):
    """Nested tuple of (feature, value) splits, for equality checks."""
    if isinstance(node, TerminalNode):
        return ("leaf", len(node.data.x))
    return (
        node.split_feature_index,
        node.split_feature_value,
        _structure(node.left_child),
        _structure(node.right_child),
    )


@pytest.fixture(scope="module")
def signal_tree():
    x, Z, c = _signal_data()
    np.random.seed(21)
    return SurvivalTree.fit(
        x=x,
        Z=Z,
        c=c,
        n_features_split="all",
        min_leaf_samples=15,
        min_leaf_failures=8,
        max_depth=2,
    )


@pytest.fixture(scope="module")
def signal_forest():
    x, Z, c = _signal_data(n=90, seed=8)
    np.random.seed(3)
    forest = RandomSurvivalForest.fit(
        x=x,
        Z=Z,
        c=c,
        n_trees=4,
        min_leaf_samples=20,
        n_features_split="all",
    )
    return forest, x, Z, c


# -- prediction coherence -----------------------------------------------------


@pytest.mark.parametrize("Z_vec", [Z_FAST, Z_SLOW], ids=["fast", "slow"])
def test_tree_ff_complements_sf(signal_tree, Z_vec):
    sf = signal_tree.sf(X_GRID, Z_vec)
    ff = signal_tree.ff(X_GRID, Z_vec)
    np.testing.assert_allclose(sf + ff, 1.0, atol=1e-9)


@pytest.mark.parametrize("Z_vec", [Z_FAST, Z_SLOW], ids=["fast", "slow"])
def test_tree_Hf_is_negative_log_sf(signal_tree, Z_vec):
    sf = signal_tree.sf(X_GRID, Z_vec)
    Hf = signal_tree.Hf(X_GRID, Z_vec)
    np.testing.assert_allclose(Hf, -np.log(sf), rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("Z_vec", [Z_FAST, Z_SLOW], ids=["fast", "slow"])
def test_tree_sf_monotone_and_bounded(signal_tree, Z_vec):
    sf = signal_tree.sf(X_GRID, Z_vec)
    assert np.all(sf >= -1e-12) and np.all(sf <= 1 + 1e-12)
    assert np.all(np.diff(sf) <= 1e-12)  # non-increasing
    assert np.all(signal_tree.df(X_GRID, Z_vec) >= 0)
    assert np.all(signal_tree.hf(X_GRID, Z_vec) >= 0)


# -- structural guarantees ----------------------------------------------------


def test_max_depth_is_honoured(signal_tree):
    assert isinstance(signal_tree._root, IntermediateNode)
    assert _tree_depth(signal_tree._root) <= 2


def test_min_leaf_samples_is_honoured(signal_tree):
    for leaf in _leaves(signal_tree._root):
        assert len(leaf.data.x) >= 15


def test_min_leaf_failures_is_honoured(signal_tree):
    for leaf in _leaves(signal_tree._root):
        event_weight = leaf.data.n[leaf.data.c != 1].sum()
        assert event_weight >= 8


def test_max_depth_zero_is_the_pooled_fit():
    x, Z, c = _signal_data(n=60, seed=9)
    np.random.seed(0)
    tree = SurvivalTree.fit(x=x, Z=Z, c=c, max_depth=0)
    assert isinstance(tree._root, TerminalNode)
    pooled = Weibull.fit(x=x, c=c)
    np.testing.assert_allclose(tree._root.model.params, pooled.params)


def test_tree_structure_is_deterministic_under_seed():
    x, Z, c = _signal_data()
    np.random.seed(42)
    tree_1 = SurvivalTree.fit(x=x, Z=Z, c=c, n_features_split=1, max_depth=2)
    np.random.seed(42)
    tree_2 = SurvivalTree.fit(x=x, Z=Z, c=c, n_features_split=1, max_depth=2)
    assert _structure(tree_1._root) == _structure(tree_2._root)


# -- degenerate inputs --------------------------------------------------------


def test_all_right_censored_is_a_never_occurs_leaf():
    rng = np.random.default_rng(2)
    n = 30
    x = np.linspace(1.0, 10.0, n)
    Z = rng.normal(0.0, 1.0, (n, 2))
    np.random.seed(0)
    tree = SurvivalTree.fit(x=x, Z=Z, c=np.ones(n), n_features_split="all")
    assert isinstance(tree._root, TerminalNode)
    assert tree._root.model is NeverOccurs
    assert np.all(tree.sf(X_GRID, Z[0]) == 1.0)


def test_constant_covariates_give_a_single_leaf():
    x, _, c = _signal_data(n=40, seed=3)
    Z = np.ones((40, 2))
    np.random.seed(0)
    tree = SurvivalTree.fit(x=x, Z=Z, c=c, n_features_split="all")
    assert isinstance(tree._root, TerminalNode)


def test_tiny_dataset_gives_a_single_valid_leaf():
    # 4 samples cannot be split under min_leaf_samples=5 (the default)
    rng = np.random.default_rng(4)
    x = np.array([3.0, 4.0, 5.0, 6.0])
    Z = rng.normal(0.0, 1.0, (4, 2))
    np.random.seed(0)
    tree = SurvivalTree.fit(x=x, Z=Z, c=np.zeros(4), n_features_split="all")
    assert isinstance(tree._root, TerminalNode)
    sf = tree.sf(X_GRID, Z[0])
    assert np.all(np.isfinite(sf)) and np.all((sf >= 0) & (sf <= 1))


def test_tied_event_times_build_a_valid_tree():
    # All events at the same time: the Weibull MLE degenerates, and the
    # leaf must fall back rather than crash the tree.
    rng = np.random.default_rng(6)
    n = 30
    x = np.full(n, 5.0)
    c = (rng.random(n) < 0.3).astype(int)
    Z = rng.normal(0.0, 1.0, (n, 2))
    np.random.seed(0)
    tree = SurvivalTree.fit(x=x, Z=Z, c=c, n_features_split="all")
    sf = np.asarray(tree.sf(X_GRID, Z[0]), dtype=float)
    assert np.all(np.isfinite(sf)) and np.all((sf >= 0) & (sf <= 1))


def test_count_weights_match_expanded_data():
    x = np.array([2.0, 3.0, 5.0, 7.0, 11.0])
    n = np.array([3, 1, 4, 2, 2])
    c = np.array([0, 1, 0, 0, 1])
    Z = np.zeros((5, 1))
    np.random.seed(0)
    tree_weighted = SurvivalTree.fit(x=x, Z=Z, c=c, n=n, max_depth=0)
    np.random.seed(0)
    tree_expanded = SurvivalTree.fit(
        x=np.repeat(x, n),
        Z=np.zeros((int(n.sum()), 1)),
        c=np.repeat(c, n),
        max_depth=0,
    )
    np.testing.assert_allclose(
        tree_weighted._root.model.params,
        tree_expanded._root.model.params,
        rtol=1e-6,
    )


# -- leaf model families ------------------------------------------------------


def test_weibull_kind_leaves_are_weibull(signal_tree):
    for leaf in _leaves(signal_tree._root):
        assert leaf.model.dist.name == "Weibull"


def test_exponential_kind_leaves_are_exponential():
    x, Z, c = _signal_data()
    np.random.seed(21)
    tree = SurvivalTree.fit(
        x=x,
        Z=Z,
        c=c,
        n_features_split="all",
        min_leaf_samples=15,
        max_depth=2,
        kind="exponential",
    )
    for leaf in _leaves(tree._root):
        assert leaf.model.dist.name == "Exponential"


# -- forest ensemble maths ----------------------------------------------------


def test_forest_sf_is_the_mean_of_tree_sfs(signal_forest):
    forest, *_ = signal_forest
    expected = np.mean(
        [tree.sf(X_GRID, Z_FAST) for tree in forest.trees], axis=0
    )
    np.testing.assert_allclose(forest.sf(X_GRID, Z_FAST), expected)


def test_forest_Hf_ensemble_method_averages_hazard(signal_forest):
    forest, *_ = signal_forest
    mean_Hf = np.mean(
        [tree.Hf(X_GRID, Z_FAST) for tree in forest.trees], axis=0
    )
    np.testing.assert_allclose(
        forest.sf(X_GRID, Z_FAST, ensemble_method="Hf"), np.exp(-mean_Hf)
    )
    np.testing.assert_allclose(forest.Hf(X_GRID, Z_FAST), mean_Hf)


def test_forest_prediction_shapes(signal_forest):
    forest, *_ = signal_forest
    single = forest.sf(X_GRID, Z_FAST)
    assert single.shape == (X_GRID.size,)

    stacked = forest.sf(X_GRID, np.array([Z_FAST, Z_SLOW]))
    assert stacked.shape == (2, X_GRID.size)
    np.testing.assert_allclose(stacked[0], single)
    np.testing.assert_allclose(stacked[1], forest.sf(X_GRID, Z_SLOW))


def test_forest_mortality_orders_the_groups(signal_forest):
    forest, *_ = signal_forest
    m_fast = float(forest.mortality(X_GRID, Z_FAST)[0])
    m_slow = float(forest.mortality(X_GRID, Z_SLOW)[0])
    assert m_fast > m_slow


def test_forest_concordance_beats_chance(signal_forest):
    forest, x, Z, c = signal_forest
    assert forest.score(x, Z, c) > 0.7


def test_forest_accepts_1d_covariates():
    rng = np.random.default_rng(12)
    n = 60
    z = rng.integers(0, 2, n).astype(float)
    x = np.where(z > 0.5, 3.0, 12.0) * rng.weibull(1.5, n)
    np.random.seed(5)
    forest = RandomSurvivalForest.fit(
        x=x, Z=z, c=np.zeros(n), n_trees=2, min_leaf_samples=15
    )
    assert forest.Z.shape == (n, 1)
    s_fast = float(np.atleast_1d(forest.sf(4.0, np.array([1.0])))[0])
    s_slow = float(np.atleast_1d(forest.sf(4.0, np.array([0.0])))[0])
    assert s_fast < s_slow


def test_missing_Z_raises():
    x = [1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="Z"):
        SurvivalTree.fit(x=x)
    with pytest.raises(ValueError, match="Z"):
        RandomSurvivalForest.fit(x=x)


@pytest.mark.parametrize("kind", ["exponential", "non-parametric"])
def test_forest_kind_passthrough(kind):
    x, Z, c = _signal_data(n=60, seed=10)
    np.random.seed(2)
    forest = RandomSurvivalForest.fit(
        x=x,
        Z=Z,
        c=c,
        n_trees=2,
        min_leaf_samples=15,
        n_features_split="all",
        kind=kind,
    )
    assert forest.kind == kind
    assert all(tree.kind == kind for tree in forest.trees)
    s_fast = float(np.atleast_1d(forest.sf(4.0, Z_FAST))[0])
    s_slow = float(np.atleast_1d(forest.sf(4.0, Z_SLOW))[0])
    assert s_fast < s_slow


# -- interactions -------------------------------------------------------------


def test_depth_two_tree_recovers_two_feature_interaction():
    # Scale 12 / 2**(Z0 + Z1): both features carry signal, so a depth-2
    # tree must order the four corners of the covariate square.
    rng = np.random.default_rng(14)
    n = 160
    Z = np.column_stack(
        [
            rng.integers(0, 2, n).astype(float),
            rng.integers(0, 2, n).astype(float),
        ]
    )
    scale = 12.0 / 2.0 ** (Z[:, 0] + Z[:, 1])
    x = scale * rng.weibull(1.7, n)
    np.random.seed(17)
    tree = SurvivalTree.fit(
        x=x,
        Z=Z,
        c=np.zeros(n),
        n_features_split="all",
        min_leaf_samples=15,
        max_depth=2,
    )
    t = 4.0
    s_00 = float(tree.sf(t, np.array([0.0, 0.0]))[0])
    s_01 = float(tree.sf(t, np.array([0.0, 1.0]))[0])
    s_10 = float(tree.sf(t, np.array([1.0, 0.0]))[0])
    s_11 = float(tree.sf(t, np.array([1.0, 1.0]))[0])
    assert s_11 < min(s_01, s_10)
    assert max(s_01, s_10) < s_00
