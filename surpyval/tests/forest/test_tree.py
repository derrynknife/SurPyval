import numpy as np
import pytest

from surpyval import Weibull
from surpyval.regression.forest.tree import Tree

"""
NOTES

Things to test:
- One split - feature split
- One split - Weibull fit + prediction
- One split - all functions
- Bootstrapping - one split

Convention:
Z = Covariant Matrix
x = Event-time vector
c = Censor vector


"""


def test_tree_one_split_one_feature_no_bootstrapping():
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

    # Assert samples in leaf nodes are as expected
    assert x[:8] == sorted(tree._root.left_node.x)
    assert x[8:] == sorted(tree._root.right_node.x)

    # Assert Weibull models
    left_weibull = tree._root.left_node.model
    right_weibull = tree._root.right_node.model
    assert pytest.approx(exp_left_weibull.params) == left_weibull.params
    assert pytest.approx(exp_right_weibull.params) == right_weibull.params


def test_tree_one_split_one_feature_with_bootstrapping():
    """
    Basically the previous test, but with bootstrapping, and checking
    still the correct samples are in the correct leaves, and that the
    Weibull parameters are approximately as expected.
    """
    # Samples
    x = [10, 12, 8, 9, 11, 12, 13, 9] + [50, 60, 40, 45, 55, 60, 65, 45]
    Z = [0] * 8 + [1] * 8
    c = [0] * len(x)

    # Expected
    exp_left_weibull = Weibull.fit(x=x[:8], c=c[:8])
    exp_right_weibull = Weibull.fit(x=x[8:], c=c[8:])

    # Actual
    tree = Tree(x=x, Z=Z, c=c)

    # May as well check it only undergoes one split
    assert tree.depth == 1

    # Assert samples in leaf nodes are as expected
    assert set(x[:8]).issuperset(tree._root.left_node.x)
    assert set(x[8:]).issuperset(tree._root.right_node.x)

    # Assert Weibull models
    left_weibull = tree._root.left_node.model
    right_weibull = tree._root.right_node.model
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
    x = np.concatenate(x_left, x_right)
    z_0_irrelevant = rng.uniform(0, 1, 100)
    z_1_relevant = np.concatenate(
        rng.uniform(0, 1, 50), rng.uniform(100, 101, 50)
    )
    Z = np.array([z_0_irrelevant, z_1_relevant]).transpose()
    c = [0] * len(x)

    # Actual
    tree = Tree(x=x, Z=Z, c=c, max_depth=1)

    # Assert feature 1 was selected
    assert tree._root.split_feature_index == 1
