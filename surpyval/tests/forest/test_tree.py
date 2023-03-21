import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn.preprocessing import OrdinalEncoder

# For scikit-survival test
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import OneHotEncoder
from sksurv.tree.tree import SurvivalTree as sksurv_SurvivalTree

from surpyval import Weibull
from surpyval.regression.forest.node import TerminalNode
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


def assert_trees_equal(surv_tree: Tree, sksurv_tree: sksurv_SurvivalTree):
    # Get the scikit-survival underlying Tree object (actually from
    # scikit-learn)
    sklearn_tree = sksurv_tree.tree_

    # Get the current node
    surv_curr_node = surv_tree._root
    sksurv_curr_node: int = 0

    def dfs_assert_trees_equal(
        surv_curr_node,
        sksurv_curr_node: int,
    ):
        # If surv_curr_node is a TerminalNode, assert sksurv_curr_node is also
        # a leaf node
        if isinstance(surv_curr_node, TerminalNode):
            assert (
                sklearn_tree.children_left[sksurv_curr_node]
                == sklearn_tree.children_right[sksurv_curr_node]
                == -1
            )
            return

        # Else, it's an intermediate node, which needs its feature-value split
        # checked
        assert (
            surv_curr_node.split_feature_index
            == sklearn_tree.feature[sksurv_curr_node]
        )

        assert (
            pytest.approx(surv_curr_node.split_feature_value)
            == sklearn_tree.threshold[sksurv_curr_node]
        )

        # And continue the DFS
        dfs_assert_trees_equal(
            surv_curr_node.left_child,
            sklearn_tree.children_left[sksurv_curr_node],
        )
        dfs_assert_trees_equal(
            surv_curr_node.right_child,
            sklearn_tree.children_right[sksurv_curr_node],
        )

    # Begin DFS
    dfs_assert_trees_equal(surv_curr_node, sksurv_curr_node)


def test_tree_reference_split_one_split_one_feature():
    # Samples
    x = [10, 12, 8, 9, 11, 12, 13, 9] + [50, 60, 40, 45, 55, 60, 65, 45]
    Z = [0] * 8 + [1] * 8
    c = [0] * len(x)

    # Surpyval
    surv_tree = Tree(x=x, Z=Z, c=c, max_depth=1, n_features_split="all")

    # Scikit-survival
    X = np.array(Z, ndmin=2).transpose()
    y = np.array(
        list(zip([True] * len(x), x)),
        dtype=[("Status", bool), ("Survival", "<f8")],
    )
    sksurv_tree = sksurv_SurvivalTree(max_depth=1, max_features=None)
    sksurv_tree.fit(X=X, y=y)

    assert_trees_equal(surv_tree, sksurv_tree)


def test_tree_reference_split_one_split_two_features():
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

    # Surpyval
    tree = Tree(x=x, Z=Z, c=c, max_depth=1, n_features_split="all")

    # Scikit-survival
    X = Z
    y = np.array(
        list(zip([True] * len(x), x)),
        dtype=[("Status", bool), ("Survival", "<f8")],
    )
    sksurv_tree = sksurv_SurvivalTree(max_depth=1, max_features=None)
    sksurv_tree.fit(X=X, y=y)

    assert_trees_equal(tree, sksurv_tree)


def test_tree_reference_splits_gbsg2():
    # Prep data input
    X, y = load_gbsg2()

    grade_str = X.loc[:, "tgrade"].astype(object).values[:, np.newaxis]
    grade_num = OrdinalEncoder(categories=[["I", "II", "III"]]).fit_transform(
        grade_str
    )

    X_no_grade = X.drop("tgrade", axis=1)
    Xt = OneHotEncoder().fit_transform(X_no_grade)
    Xt.loc[:, "tgrade"] = grade_num

    # Fit a sksurv SurvivalTree for min samples per leaf of 15, and no
    # randomisation of features for splitting
    sksurv_tree = sksurv_SurvivalTree(
        min_samples_split=2,
        min_samples_leaf=15,
        max_features=None,
        max_depth=2,
    )
    sksurv_tree.fit(Xt, y)

    # Prep and fit a surpyval Tree
    def sksurv_Xy_to_surv_xZc(X: pd.DataFrame, y: NDArray):
        return y["time"], X.to_numpy(), np.logical_not(y["cens"]).astype(int)

    x, Z, c = sksurv_Xy_to_surv_xZc(Xt, y)

    surv_tree = Tree(
        x=x,
        Z=Z,
        c=c,
        n_features_split="all",
        min_leaf_failures=15,
        max_depth=2,
    )

    assert_trees_equal(surv_tree, sksurv_tree)
