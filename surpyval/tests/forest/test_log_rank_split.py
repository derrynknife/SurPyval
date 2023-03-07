import numpy as np

from surpyval.regression.forest.log_rank_split import log_rank_split


def test_log_rank_split_one_binary_feature():
    """Simplest case. One feature and two samples."""
    x = np.array(
        [10, 12, 8, 9, 11, 12, 13, 9, 10, 10]
        + [50, 60, 40, 45, 55, 60, 65, 45, 50, 50]
    )
    Z = np.array([0] * 10 + [1] * 10)
    c = np.array([0] * len(x))

    lrs = log_rank_split(x, Z, c, min_leaf_samples=6, feature_indices_in=[0])

    # Assert feature 0 (the only feature) is returned
    assert lrs[0] == 0

    # Assert feature 0 value 0 (left children have Z_0 <= 0)
    assert lrs[1] == 0


def test_log_rank_split_one_feature_four_samples():
    x = np.array([10, 11] + [24, 25])
    Z = np.array([0, 0.2] + [15.1, 15])
    c = np.array([0] * len(x))

    lrs = log_rank_split(x, Z, c, min_leaf_samples=1, feature_indices_in=[0])

    assert lrs[0] == 0
    assert lrs[1] == 0


def test_log_rank_split_two_features_two_samples():
    """
    Idea is to have two features, one with basically no predictive ability
    and one with plenty of predictive ability.
    """
    x = np.array([15, 75])
    Z = np.array([[0.3, 1], [0.3, 3]])  # Feature 1 should be selected
    c = np.array([0] * len(x))

    lrs = log_rank_split(
        x, Z, c, min_leaf_samples=1, feature_indices_in=[0, 1]
    )

    assert lrs[0] == 1
    assert lrs[1] == 1
