from math import sqrt
from typing import Iterable

import numpy as np
import pytest
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
    # Sort x, Z, and c, in x, making sure Z is 2d (required for future calcs)
    sort_idxs = np.argsort(x)
    x = x[sort_idxs]
    c = c[sort_idxs]
    if Z.ndim == 1:
        Z = np.reshape(Z[sort_idxs], (1, -1)).transpose()
    else:
        Z[sort_idxs]

    # Calculate the d vector, where d[j] is the number of distinct deaths
    # at t[j]
    death_indices = np.where(c == 0)[0]  # Uncensored => deaths
    death_xs = x[death_indices]  # Death times

    # The 't' and 'd' vectors, that is t[j] is the j-th dinstinct (uncensored)
    # death time, and d[j] is the number of distinct deaths at that time
    t, d = np.unique(death_xs, return_counts=True)
    m = len(t)  # How many unique times in the samples there are

    # Now the Y vector needs to be calculated
    # Y[j] = the number of people still 'at risk' at time t[j], that is the
    # sum of samples who's survival times are greater than t[j] (irrelevant
    # of censorship)
    Y = np.array([len(x[x >= t_j]) for t_j in t])

    # Now let's find the best (u, v) pair
    max_log_rank_magnitude = float("-inf")
    best_u = -1  # Placeholder value
    best_v = -float("inf")  # Placeholder value

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

    for u in feature_indices_in:
        for v in np.unique(Z[:, u])[:-1]:
            # Discard the (u, v) pair if it means a leaf will
            # have < min_leaf_failures samples
            if breaks_min_leaf_failures_constraint():
                continue

            abs_log_rank = abs(log_rank(u, v, x, Z, c, t, d, Y, m))

            if assert_reference:
                reference_abs_log_rank, _ = compare_survival(
                    surv_xZc_to_sksurv_Xy(x, Z, c)[1],
                    (Z[:, u] <= v).astype(int),
                )
                try:
                    assert (
                        pytest.approx(abs_log_rank) == reference_abs_log_rank
                    )
                except AssertionError:
                    raise AssertionError(
                        f"abs_log_rank={abs_log_rank:.3f} != "
                        f"reference_abs_log_rank={reference_abs_log_rank:.3f}"
                    )

            if abs_log_rank > max_log_rank_magnitude:
                max_log_rank_magnitude = abs_log_rank
                best_u = u
                best_v = v

    return best_u, best_v


def log_rank(
    u: int,
    v: float,
    x: NDArray,
    Z: NDArray,
    c: NDArray,
    t: NDArray,
    d: NDArray,
    Y: NDArray,
    m: int,
) -> float:
    """Returns L(u, v)."""

    # Define the d_L and Y_L vectors
    d_L = np.zeros(m)
    Y_L = np.zeros(m)

    # Get sample-indices (i) of those that would end up in the left child
    left_child_indices = np.where(Z[:, u] <= v)[0]
    left_child_x = x[left_child_indices]
    left_child_c = c[left_child_indices]

    for j in range(m):
        # Number of uncensored deaths at t[j]
        d_L[j] = sum(left_child_x[left_child_c == 0] == t[j])

        # Number 'at risk', that is those still alive + have an event (death
        # or censor) at t[j]
        Y_L[j] = sum(left_child_x >= t[j])

    # Perform the j-sums
    numerator = 0
    denominator_inside_sqrt = 0  # Must sqrt() after j loop

    for j in range(m):
        if not Y[j] >= 2:
            # Denominator contribution would be undefined
            # (Y[j] - 1 <= 0 -> bad!)
            continue
        numerator += d_L[j] - Y_L[j] * d[j] / Y[j]
        denominator_inside_sqrt += (
            Y_L[j]
            / Y[j]
            * (1 - Y_L[j] / Y[j])
            * (Y[j] - d[j])
            / (Y[j] - 1)
            * d[j]
        )

    L_u_v_return = numerator / sqrt(denominator_inside_sqrt)

    return L_u_v_return
