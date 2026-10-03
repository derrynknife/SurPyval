from collections.abc import Iterable
from math import sqrt

import numpy as np
from numpy.typing import NDArray

from surpyval.beta.ml.forest.deviance_split import (
    needs_full_likelihood_split,
)
from surpyval.utils.data_formats import _entered_before
from surpyval.utils.surpyval_data import SurpyvalData


def _weight_at_or_after(
    values: NDArray, weights: NDArray, grid: NDArray
) -> NDArray:
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
        Minimum ``n``-weighted number of failures (rows that are not right
        censored) each child must have, counted as the deviance split
        counts them
    feature_indices_in : Iterable[int]
        Indices of the features to consider for the split

    Returns
    -------
    tuple[int, float]
        The feature index and value of the maximal Log-Rank split, these will
        be (-1, -Inf) if insufficient samples were provided to satisfy the
        min_leaf_failures constraint.

    Raises
    ------
    ValueError
        If ``data`` has left or interval censoring or right truncation,
        which have no risk sets: such data is split by
        :func:`~surpyval.beta.ml.forest.turnbull_score_split.turnbull_score_split`
        or :func:`~surpyval.beta.ml.forest.deviance_split.deviance_split`.
    """
    # The tree routes such data to another split; called directly, the
    # risk sets below would come from a Turnbull fit whose grid is not
    # the raw times, and the statistic would be silently wrong.
    if needs_full_likelihood_split(data):
        raise ValueError(
            "log_rank_split needs observed and right-censored data "
            "(optionally left truncated): left or interval censoring and "
            "right truncation have no risk sets. Use turnbull_score_split "
            "or deviance_split for such data."
        )
    # The failures each child keeps, n-weighted as the deviance split
    # counts them (#193).
    event_weight = data.n * (data.c != 1)

    # Each feature is sorted once and every threshold scored from the
    # cumulative at-risk and death counts of the rows below it (#190);
    # see ``_log_rank_scan``.
    scan = _LogRankScan(data)
    max_log_rank_magnitude = float("-inf")
    best_u = -1  # Placeholder value
    best_v = -float("inf")  # Placeholder value
    n_rows = len(data)
    total_events = event_weight.sum()

    for u in feature_indices_in:
        Z_u = Z[:, u]
        values = np.unique(Z_u)
        order = np.argsort(Z_u, kind="stable")
        # The left child of value v is every row with Z_u <= v: the first
        # n_left rows in the feature's order.
        n_left = np.searchsorted(Z_u[order], values, side="right")
        events_left = np.concatenate([[0], np.cumsum(event_weight[order])])[
            n_left
        ]
        # Discard the (u, v) pairs that leave a child with too few
        # samples or failures
        ok = (
            (n_left >= min_leaf_samples)
            & (n_rows - n_left >= min_leaf_samples)
            & (events_left >= min_leaf_failures)
            & (total_events - events_left >= min_leaf_failures)
        )
        if not ok.any():
            continue
        statistic = np.full(values.size, -np.inf)
        statistic[ok] = scan.statistics(order, n_left[ok])
        k = int(np.argmax(statistic))
        if statistic[k] > max_log_rank_magnitude:
            max_log_rank_magnitude = float(statistic[k])
            best_u = u
            best_v = values[k]

    return best_u, best_v


def _risk_sets(data: SurpyvalData) -> tuple[NDArray, NDArray, NDArray]:
    """``data.to_xrd()`` for observed and right-censored data (optionally
    left truncated) -- the same arithmetic as
    :func:`~surpyval.utils.xcnt_to_xrd`, so the same numbers -- without
    validating the node's rows again: they were validated when the tree
    was fitted, and the validation cost more than the split search."""
    x = np.asarray(data.x, dtype=float)
    c = np.asarray(data.c)
    n = np.asarray(data.n)
    grid, idx = np.unique(x, return_inverse=True)
    d = np.bincount(idx, weights=n * (1 - c))
    do = np.bincount(idx, weights=n * c)
    e = _entered_before(np.asarray(data.t[:, 0], dtype=float), grid, n)
    r = e + d - d.cumsum() + do - do.cumsum()
    return grid, r.astype(int), d.astype(int)


# The most elements a block of the at-risk scan holds at once.
_SCAN_BLOCK = 1 << 18


class _LogRankScan:
    """The log-rank statistic of every left child that is a prefix of the
    rows in some order, from cumulative counts.

    The statistic is a sum over the pooled event times (the node's
    ``to_xrd`` grid) of the left child's deaths ``d_L`` and at-risk count
    ``Y_L``. A row is at risk on the grid times in ``(t_l, x]`` -- a
    contiguous run of grid indices ``[lo, hi)`` -- so adding it to the
    left child adds its count at ``lo`` and removes it at ``hi`` of a
    difference vector. Cumulating the difference vectors over the rows in
    a feature's order, and each over the grid, gives ``Y_L`` for every
    prefix at once; ``d_L`` likewise. This is the at-risk count of
    :func:`at_risk_on_grid` and the death count of :func:`deaths_on_grid`
    (with integer counts, exactly the same numbers), and the statistic is
    then evaluated by :func:`log_rank`'s own expressions, so each
    threshold's statistic is the one :func:`log_rank` returns for it, bit
    for bit, at the cost of one pass over the rows per feature rather
    than a new subset per threshold.
    """

    def __init__(self, data: SurpyvalData) -> None:
        grid, Y, d = _risk_sets(data)
        grid = np.asarray(grid, dtype=float)
        x = np.asarray(data.x, dtype=float)
        tl = np.asarray(data.t[:, 0], dtype=float)
        self.n = np.asarray(data.n, dtype=float)
        self.m = grid.size
        # Grid indices [lo, hi) where each row is at risk: tl < t <= x.
        self.lo = np.searchsorted(grid, tl, side="right")
        self.hi = np.searchsorted(grid, x, side="right")
        # Each observed death lands on its own grid time.
        observed = np.asarray(data.c) == 0
        self.death = np.where(
            observed, np.searchsorted(grid, x, side="left"), self.m
        )
        # The statistic sums over the times with more than one at risk.
        self.keep = np.asarray(Y) > 1
        self.Y = np.asarray(Y, dtype=float)[self.keep]
        self.d = np.asarray(d, dtype=float)[self.keep]

    def statistics(self, order: NDArray, n_left: NDArray) -> NDArray:
        """``|L|`` for each left child made of the first ``n_left`` rows
        (increasing) of ``order``; ``-inf`` where it is undefined."""
        Y_L, d_L = self._left_counts(order, n_left)
        Y, d = self.Y, self.d
        # log_rank's expressions, row by row
        numerator = np.sum(d_L - Y_L * (d / Y), axis=1)
        denominator_inside_sqrt = np.sum(
            (Y_L / Y) * (1.0 - Y_L / Y) * (Y - d) / (Y - 1) * d, axis=1
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.abs(numerator / np.sqrt(denominator_inside_sqrt))
        # A zero (or undefined) variance gives no statistic
        return np.where(denominator_inside_sqrt > 0, out, -np.inf)

    def _left_counts(
        self, order: NDArray, n_left: NDArray
    ) -> tuple[NDArray, NDArray]:
        m = self.m
        Y_L = np.empty((n_left.size, int(self.keep.sum())))
        d_L = np.empty_like(Y_L)
        n = self.n[order]
        lo, hi, death = self.lo[order], self.hi[order], self.death[order]
        # Cumulated over the rows in blocks, carrying the running totals,
        # so memory stays bounded however many rows there are.
        block = max(1, _SCAN_BLOCK // (m + 1))
        carry_risk = np.zeros(m + 1)
        carry_death = np.zeros(m + 1)
        done = 0
        for start in range(0, int(n_left.max()), block):
            stop = min(start + block, int(n_left.max()))
            rows = np.arange(stop - start)
            risk = np.zeros((rows.size, m + 1))
            # One entry per row, so plain fancy assignment suffices
            risk[rows, lo[start:stop]] = n[start:stop]
            risk[rows, hi[start:stop]] -= n[start:stop]
            dead = np.zeros((rows.size, m + 1))
            dead[rows, death[start:stop]] = n[start:stop]
            risk = carry_risk + np.cumsum(risk, axis=0)
            dead = carry_death + np.cumsum(dead, axis=0)
            carry_risk, carry_death = risk[-1], dead[-1]
            # The prefixes that end in this block
            here = (n_left > start) & (n_left <= stop)
            if here.any():
                last = n_left[here] - 1 - start
                Y_L[here] = np.cumsum(risk[last], axis=1)[:, :m][:, self.keep]
                d_L[here] = dead[last][:, :m][:, self.keep]
                done += int(here.sum())
        assert done == n_left.size
        return Y_L, d_L


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
