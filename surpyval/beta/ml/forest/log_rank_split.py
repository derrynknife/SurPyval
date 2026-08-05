from collections.abc import Iterable
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from surpyval.utils.surpyval_data import SurpyvalData


def _weight_at_or_after(values, weights, grid):
    """Total weight of ``values >= t``, for every ``t`` in ``grid``.

    ``grid`` is assumed sorted, as ``to_xrd`` returns it.
    """
    order = np.argsort(values, kind="stable")
    ordered = np.asarray(values, dtype=float)[order]
    w = np.asarray(weights, dtype=float)[order]
    # suffix[i] is the weight from position i to the end; the trailing
    # zero covers a grid time past every value.
    suffix = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
    return suffix[np.searchsorted(ordered, grid, side="left")]


def at_risk_on_grid(data: SurpyvalData, grid: NDArray) -> NDArray:
    r"""At-risk count of ``data`` at each time in ``grid``.

    An observation is at risk at :math:`t` when it has entered and not
    yet left -- :math:`t_l < t \leq x` -- which is the ``(entry, exit]``
    convention ``xcnt_to_xrd`` uses, so that a subject entering exactly
    at an event time is not at risk for it.

    Counted directly, rather than by carrying a risk ladder forward from
    the times where this subset happens to have observations. Forward
    filling is what #287 got wrong: it carried :math:`Y(t_j)` to later
    grid times without removing the deaths and censorings *at*
    :math:`t_j`, and it extended the final value past the last
    observation having subtracted only the deaths, so a subset ending in
    a censored observation kept a phantom at risk for ever. Both
    inflated the count, and the split statistic built on it was wrong by
    factors of several.

    Since ``tl <= x`` always holds, the observations with ``tl >= t`` are
    a subset of those with ``x >= t``, so the count is the difference of
    two suffix sums rather than a scan over the grid.
    """
    x = np.asarray(data.x, dtype=float)
    n = np.asarray(data.n, dtype=float)
    tl = np.asarray(data.t[:, 0], dtype=float)
    return _weight_at_or_after(x, n, grid) - _weight_at_or_after(tl, n, grid)


def deaths_on_grid(data: SurpyvalData, grid: NDArray) -> NDArray:
    """Observed-death count of ``data`` at each time in ``grid``.

    Every ``x`` in ``data`` is a time in ``grid`` -- the grid comes from
    the pooled data this is a subset of -- so each death lands exactly.
    """
    x = np.asarray(data.x, dtype=float)
    n = np.asarray(data.n, dtype=float)
    observed = np.asarray(data.c) == 0
    return np.bincount(
        np.searchsorted(grid, x[observed], side="left"),
        weights=n[observed],
        minlength=grid.size,
    )[: grid.size]


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
    data : SurpyvalData
        Survival data (x, c, n, t)
    Z : NDArray
        Covariant matrix, of shape (n_samples, n_features)
    min_leaf_samples : int
        Minimum number of samples each child must have
    min_leaf_failures : int
        Minimum number of failures each child must have
    feature_indices_in : Iterable[int]
        Indices of the features to consider for the split

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
            elif (data.c[mask] != 1).sum() < min_leaf_failures:
                continue
            elif (data.c[~mask] != 1).sum() < min_leaf_failures:
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

    # The statistic is a sum over the *pooled* event times, so the left
    # child's risk set and deaths are needed at each of them -- including
    # the times where the left child itself has no observation. Both are
    # counted directly on that grid; see ``at_risk_on_grid``.
    all_x, Y, d = data.to_xrd()
    Y_L = at_risk_on_grid(data_left_child, all_x)
    d_L = deaths_on_grid(data_left_child, all_x)

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

    if denominator_inside_sqrt <= 0:
        return -float("inf")

    try:
        v = np.abs(numerator / sqrt(denominator_inside_sqrt))
        return v
    except ZeroDivisionError:
        raise ValueError("Numerator or denominator is NaN")
