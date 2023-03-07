import numpy as np
import pytest

from surpyval import Weibull
from surpyval.regression.forest.tree import Tree


def test_tree_no_split():
    """
    Literally just a Tree with max_depth=1 (i.e. one TerminalNode), check
    model.
    """
    x = [10, 12, 8, 9, 11, 12, 13, 9]
    Z = [0] * 8
    c = [0] * len(x)

    exp_weibull = Weibull.fit(x=x, c=c)

    tree = Tree(x=x, Z=Z, c=c, max_depth=0)

    actual_weibull = tree._root.model

    print(exp_weibull, actual_weibull)

    assert pytest.approx(exp_weibull.params) == actual_weibull.params


def test_tree_one_split_one_feature():
    """
    Basically test_log_rank_split_one_binary_feature, checking the leaf samples
    and their Weibull parameters.
    """
    # Samples
    x = [10, 12, 8, 9, 11, 12, 13, 9] + [50, 60, 40, 45, 55, 60, 65, 45]
    Z = [0] * 8 + [1] * 8
    c = [0] * len(x)

    # Expected
    exp_left_weibull = Weibull.fit(x=x[:8], c=c[:8])
    exp_right_weibull = Weibull.fit(x=x[8:], c=c[8:])

    # Actual
    tree = Tree(x=x, Z=Z, c=c, max_depth=1)

    # Assert Weibull models
    left_weibull = tree._root.left_child.model
    right_weibull = tree._root.right_child.model
    assert pytest.approx(exp_left_weibull.params) == left_weibull.params
    assert pytest.approx(exp_right_weibull.params) == right_weibull.params


def test_tree_one_split_two_features():
    """
    Have two features, one completely irrelevant, the other far more
    predictive.
    """
    # Set random seed
    rng = np.random.default_rng(seed=0)

    # Samples
    x_left = rng.uniform(5, 10, 50)
    x_right = rng.uniform(100, 105, 50)
    x = np.concatenate((x_left, x_right))

    # Covariants
    z_0_irrelevant = rng.uniform(0, 1, 100)
    z_1_relevant = np.concatenate(
        (rng.uniform(0, 1, 50), rng.uniform(100, 101, 50))
    )
    Z = np.array([z_0_irrelevant, z_1_relevant]).transpose()
    c = [0] * len(x)

    # Actual
    tree = Tree(x=x, Z=Z, c=c, max_depth=1, n_features_split="all")

    # Assert feature 1 was selected
    assert tree._root.split_feature_index == 1


def test_tree_one_split_two_features_n_features_split():
    """
    Now test n_features_split.
    """
    # Set random seed
    rng = np.random.default_rng(seed=0)

    # Samples
    x_left = rng.uniform(5, 10, 50)
    x_right = rng.uniform(100, 105, 50)
    x = np.concatenate((x_left, x_right))

    # Covariants
    z_0_irrelevant = rng.uniform(0, 1, 100)
    z_1_relevant = np.concatenate(
        (rng.uniform(0, 1, 50), rng.uniform(100, 101, 50))
    )
    Z = np.array([z_0_irrelevant, z_1_relevant]).transpose()
    c = [0] * len(x)

    # Different n_features_split
    tree_one_feature = Tree(x=x, Z=Z, c=c, max_depth=1, n_features_split=1)
    assert len(tree_one_feature._root.feature_indices_in) == 1

    tree_two_features = Tree(x=x, Z=Z, c=c, max_depth=1, n_features_split=2)
    assert len(tree_two_features._root.feature_indices_in) == 2

    tree_one_feature_float = Tree(
        x=x, Z=Z, c=c, max_depth=1, n_features_split=0.5
    )
    assert len(tree_one_feature_float._root.feature_indices_in) == 1

    tree_one_feature_sqrt = Tree(
        x=x, Z=Z, c=c, max_depth=1, n_features_split="sqrt"
    )
    assert len(tree_one_feature_sqrt._root.feature_indices_in) == 1

    tree_one_feature_log2 = Tree(
        x=x, Z=Z, c=c, max_depth=1, n_features_split="log2"
    )
    assert len(tree_one_feature_log2._root.feature_indices_in) == 1

    tree_two_features_all = Tree(
        x=x, Z=Z, c=c, max_depth=1, n_features_split="all"
    )
    assert len(tree_two_features_all._root.feature_indices_in) == 2
