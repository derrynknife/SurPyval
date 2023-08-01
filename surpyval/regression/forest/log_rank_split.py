from math import sqrt
from typing import Iterable

import numpy as np
from numpy.typing import NDArray

from surpyval.utils.surpyval_data import SurpyvalData


def numpy_fill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, out=idx)
    return arr[idx]


def log_rank_split(
    data: SurpyvalData,
    Z: NDArray,
    min_leaf_samples: int,
    min_leaf_failures: int,
    feature_indices_in: Iterable[int],
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

    # Now let's find the best (u, v) pair
    max_log_rank_magnitude = float("-inf")
    best_u = -1  # Placeholder value
    best_v = -float("inf")  # Placeholder value

    for u in feature_indices_in:
        Z_u = Z[:, u]
        for v in np.unique(Z_u):
            # Discard the (u, v) pair if it means a leaf will
            # have < min_leaf_failures samples
            mask = Z_u <= v
            if Z_u[mask].size < min_leaf_samples:
                continue
            elif Z_u[~mask].size < min_leaf_samples:
                continue
            elif (data.c[mask] != 1).sum() <= min_leaf_failures:
                continue
            elif (data.c[~mask] != 1).sum() <= min_leaf_failures:
                continue

            abs_log_rank = log_rank(u, v, data, Z)

            if abs_log_rank > max_log_rank_magnitude:
                max_log_rank_magnitude = abs_log_rank
                best_u = u
                best_v = v

    return best_u, best_v


def log_rank(
    u: int,
    v: float,
    data: SurpyvalData,
    Z: NDArray,
) -> float:
    """Returns L(u, v)."""

    # Get sample-indices (i) of those that would end up in the left child
    left_child_indices = np.where(Z[:, u] <= v)[0]
    data_left_child = data[left_child_indices]
    # left_child_x = data.x[left_child_indices]
    # left_child_c = data.c[left_child_indices]

    # left_child_x, idx = np.unique(left_child_x, return_inverse=True)
    # d_L = np.bincount(idx, weights=1 - left_child_c)
    # do_L = np.bincount(idx, weights=left_child_c)
    x_left, Y_L, d_L = data_left_child.to_xrd()
    all_x, Y, d = data.to_xrd()

    # expand d_L to match all_x
    # idx = np.searchsorted(x_left, all_x, side="right") - 1
    expanded_d_L = np.zeros_like(all_x)
    expanded_Y_L = np.empty_like(all_x)
    expanded_Y_L.fill(np.nan)
    # expanded_do_L = np.zeros_like(all_x)
    # x_l_indices = np.in1d(all_x, left_child_x).nonzero()[0]
    x_l_indices = np.in1d(all_x, data_left_child.x).nonzero()[0]
    expanded_d_L[x_l_indices] = d_L
    expanded_Y_L[x_l_indices] = Y_L
    # expanded_do_L[x_l_indices] = do_L

    (index,) = np.where(~np.isnan(expanded_Y_L))
    first_non_nan = index[0]
    last_non_nan = index[-1]
    expanded_Y_L[first_non_nan:last_non_nan] = numpy_fill(
        expanded_Y_L[first_non_nan:last_non_nan]
    )
    if first_non_nan != 0:
        expanded_Y_L[:first_non_nan] = expanded_Y_L[first_non_nan]
        # expanded_Y_L[:first_non_nan] = 0

    if last_non_nan != expanded_Y_L.size - 1:
        # expanded_Y_L[last_non_nan+1:] = 0
        expanded_Y_L[last_non_nan + 1 :] = (
            expanded_Y_L[last_non_nan] - expanded_d_L[last_non_nan]
        )

    Y_L = expanded_Y_L
    d_L = expanded_d_L

    # Y_L = expanded_Y_L[first_non_nan:last_non_nan+1:]
    # d_L = expanded_d_L[first_non_nan:last_non_nan+1:]
    # Y = Y[first_non_nan:last_non_nan+1:]
    # d = d[first_non_nan:last_non_nan+1:]

    # do_L = expanded_do_L

    # raise ValueError("STOP")
    # Find the risk set from the expanded d_L and do_L
    # These are the event and censored counts at each x
    # Y_L = (d_L.sum() + do_L.sum()) - d_L.cumsum()
    # + d_L - do_L.cumsum() + do_L

    # Filter to where Y > 1
    mask = Y > 1
    Y_L = Y_L[mask]
    Y = Y[mask]
    d_L = d_L[mask]
    d = d[mask]

    numerator = np.sum(d_L - Y_L * (d / Y))
    denominator_inside_sqrt = np.sum(
        (Y_L / Y) * (1.0 - Y_L / Y) * (Y - d) / (Y - 1) * d
    )

    if denominator_inside_sqrt == 0:
        return -float("inf")

    try:
        v = np.abs(numerator / sqrt(denominator_inside_sqrt))
        return v
    except ZeroDivisionError:
        raise ValueError("Numerator or denominator is NaN")
