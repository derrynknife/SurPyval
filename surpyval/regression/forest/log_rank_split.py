import numpy as np
from numpy.typing import NDArray


def log_rank_split(x: NDArray, Z: NDArray, c: NDArray) -> tuple[int, float]:
    r"""
    Returns the best feature index and value according to the Log-Rank split
    criterion.

    That is, it returns

    .. math::

        (u^*, v^*) = {\arg \max}_{0 \leq u < k, v \in Z_u}\left( |L(u, v)|
        \right )

    where :math:`k` = the number of features in Z (`Z.shape()[1]`),
    i.e. the feature index :math:`u^*` and value :math:`v^*` which maximises
    the :math:`|L(u, v)|` where

    .. math::

        L(u, v) =
        \frac {\sum_{j=1}^m d_{j,L} - Y_{j,L} \frac{d_j}{Y_j}}
        {\sqrt{\sum_{j=1}^m \frac{Y_{j,L}}{Y_j}(1 - \frac{Y_{j,L}}{Y_j})
        (\frac{Y_j-d_j}{Y_j-1})d_j}}

    where:
    - :math:`x_1<...<x_m` the unique time samples in :math:`x`
    - :math:`d_j,L \& d_j,R` = the number of deaths exactly at time :math:`x_j`
      for the left and right child nodes
    - :math:`Y_{j,L} \& Y_{j,R}` = the number of at risk samples at at time
      :math:`x_j`, that is those that are still alive or have a death exactly
      at :math:`x_j`, for the left and right child nodes


    Parameters
    ----------
    x : NDArray
        _description_
    Z : NDArray
        _description_
    c : NDArray
        _description_

    Returns
    -------
    tuple[int, float]
        The feature index and value of the maximal Log-Rank split
    """

    # Sort x, Z, and c, in x
    sort_idxs = np.argsort(x)
    x = x[sort_idxs]
    Z = Z[sort_idxs]
    c = c[sort_idxs]

    # Log-rank calculation needs the unique elements of x
    death_indices = np.where(c == 0)[0]
    death_xs = x[death_indices]
    deaths = np.zeros_like(x_uniq)

    # Maximise log-rank magnitude
    k = Z.shape()[1]

    max_log_rank_magnitude = float("-inf")
    best_u = None
    best_v = None

    for u in range(k):
        for v in Z[:, u]:
            if abs(log_rank(u, v, events, c)) > max_log_rank_magnitude:
                best_u = u
                best_v = v

    return best_u, best_v


def log_rank(u: int, v: float, events: dict[float, int], c: NDArray) -> float:
    return np.sum(events[j])
