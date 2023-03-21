from typing import Iterable

import numpy as np
from numpy.typing import NDArray
from sksurv.compare import compare_survival

from surpyval.utils.surv_sksurv_transformations import (  # sksurv's log-rank
    surv_xZc_to_sksurv_Xy,
)


def log_rank_split(
    x: NDArray,
    Z: NDArray,
    c: NDArray,
    min_leaf_failures: int,
    feature_indices_in: Iterable[int],
    assert_reference: bool = False,
) -> tuple[int, float]:
    r"""
    Returns the best feature index and value according to the Log-Rank split
    criterion.

    That is, it returns

    .. math::

        (u^*, v^*) = {\arg \max}_{u \in feature_indices_in,
        v \in Z_u}\left( |L(u, v)|
        \right )

    i.e. the feature index :math:`u^*` and value :math:`v^*` which maximises
    the :math:`|L(u, v)|` where

    .. math::

        L(u, v) =
        \frac {\sum_{j=0}^m d_{j,L} - Y_{j,L} \frac{d_j}{Y_j}}
        {\sqrt{\sum_{j=0}^m \frac{Y_{j,L}}{Y_j}(1 - \frac{Y_{j,L}}{Y_j})
        (\frac{Y_j-d_j}{Y_j-1})d_j}}

    where:
    - :math:`x_0<...<x_m` the unique time samples in :math:`x`
    - :math:`d_j,L \& d_j,R` = the number of deaths exactly at time :math:`x_j`
      for the left and right child nodes
    - :math:`Y_{j,L} \& Y_{j,R}` = the number of at risk samples at at time
      :math:`x_j`, that is those that are still alive or have a death exactly
      at :math:`x_j`, for the left and right child nodes

    Remembering, the return split is for the left childs feature
    :math:`u^* \leq v^*`, and right child :math:`u^* > v^*`.

    Note: It wraps scikit-survival's compare_survival() function. In the future
    this should be a native implementation.


    Parameters
    ----------
    x : NDArray
        Survival times
    Z : NDArray
        Covariant matrix
    c : NDArray
        Censor vector

    Returns
    -------
    tuple[int, float]
        The feature index and value of the maximal Log-Rank split, these will
        be (-1, -Inf) if insufficient samples were provided to satisfy the
        min_leaf_failures constraint.
    """
    # Transform to scikit-survival form
    if Z.ndim == 1:
        Z = np.reshape(Z, (1, -1)).transpose()
    X, y = surv_xZc_to_sksurv_Xy(x=x, Z=Z, c=c)

    # Best values
    best_feature_index = -1
    best_feature_value = float("-inf")
    best_log_rank = float("-inf")

    # Inner function used in ensuring the min_leaf_failures constraint is
    # respected
    def breaks_min_leaf_failures_constraint():
        left_child_samples = len(
            np.where(np.logical_and(Z[:, u] <= v, c == 0))[0]
        )
        right_child_samples = len(
            np.where(np.logical_and(Z[:, u] > v, c == 0))[0]
        )
        if (
            left_child_samples < min_leaf_failures
            or right_child_samples < min_leaf_failures
        ):
            return True
        return False

    # Loop over features
    for u in range(X.shape[1]):
        possible_feature_values = np.unique(X[:, u])

        # If there's <2 unique values to consider, move on to the next feature
        if len(possible_feature_values) < 2:
            continue

        # Else, go over each possible feature value
        for i, v in enumerate(possible_feature_values):
            if breaks_min_leaf_failures_constraint():
                continue

            split = (v + possible_feature_values[i + 1]) * 0.5

            groups = (X[:, u] <= split).astype(int)
            log_rank_u_v, _ = compare_survival(y, groups)
            if log_rank_u_v > best_log_rank:
                best_feature_index = u
                best_feature_value = split
                best_log_rank = log_rank_u_v

    return best_feature_index, best_feature_value
