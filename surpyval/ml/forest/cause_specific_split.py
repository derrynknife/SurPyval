r"""Splits of a survival tree that tell the causes of failure apart.

A tree grown with the cause of each failure (``SurvivalTree.fit(...,
e=...)``) chooses its cuts by one of two rules:

- ``"cause-specific"`` (the default): for each cause :math:`k`, the
  log-rank statistic :math:`L_k` of the cut with cause :math:`k`'s failures
  as the events and every other row (another cause's failure, or a
  censoring) censored -- the cause-specific log-rank test. The cut chosen
  maximises :math:`\sum_k L_k^2`. Under no effect of the cut the
  cause-specific statistics are asymptotically independent (the causes'
  counting-process martingales are orthogonal), so the sum is the
  :math:`\chi^2_K` statistic of the joint test that the cut changes no
  cause's hazard. Squaring keeps a cut that raises one cause's hazard and
  lowers another's from cancelling, as a sum of the signed statistics
  would. With one cause it is :math:`L^2`, so the cut is the one
  :func:`~surpyval.ml.forest.log_rank_split.log_rank_split` chooses.
  Nominating a cause (``event=``) uses that cause's statistic alone.
- ``"gray"``: Gray's (1988) two-sample statistic for the cumulative
  incidence of one nominated cause (``event=``), as
  :func:`~surpyval.univariate.competing_risks.gray_test` computes it.
  This targets the incidence itself rather than the hazard, but a cut
  costs a whole Gray test (its variance sums over the event times for
  each candidate cut), so it is far slower than the cause-specific rule
  on a large node, and it refuses delayed entry, as ``gray_test`` does.

Both take observed and right-censored rows, with delayed entry for the
cause-specific rule; the tree refuses left- and interval-censored rows
and right truncation with causes.
"""

from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from surpyval.ml.forest.log_rank_split import _LogRankScan
from surpyval.univariate.competing_risks.nonparametric.gray_test import (
    _score_and_variance,
)
from surpyval.utils.surpyval_data import SurpyvalData

#: The split rules a tree grown with causes takes.
SPLITS = ("cause-specific", "gray")


def cause_codes(e: Any, causes: list) -> NDArray:
    """Each row's cause as its position in ``causes``; -1 for a row with
    no cause (a censoring) or a cause not listed."""
    index = {label: k for k, label in enumerate(causes)}
    return np.fromiter(
        (-1 if v is None else index.get(v, -1) for v in e),
        dtype=np.intp,
        count=len(e),
    )


def _cause_data(data: SurpyvalData, failed: NDArray) -> SurpyvalData:
    """``data`` with the rows ``failed`` observed and every other row right
    censored (the cause-specific view of one cause)."""
    c = np.where(failed, 0, 1)
    return SurpyvalData(
        np.asarray(data.x), c, np.asarray(data.n), data.t, handle=False
    )


def _allowed(
    order: NDArray,
    n_left: NDArray,
    n_rows: int,
    min_leaf_samples: int,
    min_leaf_failures: int,
    failures: NDArray,
    per_cause: list[NDArray],
) -> NDArray:
    """The cuts that leave each child at least ``min_leaf_samples`` rows,
    ``min_leaf_failures`` failures in all and, for each cause in
    ``per_cause`` (its failure weights), ``min_leaf_failures`` of it."""

    def left_sum(weight: NDArray) -> NDArray:
        return np.concatenate([[0], np.cumsum(weight[order])])[n_left]

    total = failures.sum()
    left = left_sum(failures)
    ok = (
        (n_left >= min_leaf_samples)
        & (n_rows - n_left >= min_leaf_samples)
        & (left >= min_leaf_failures)
        & (total - left >= min_leaf_failures)
    )
    for weight in per_cause:
        left_k = left_sum(weight)
        ok &= (left_k >= min_leaf_failures) & (
            weight.sum() - left_k >= min_leaf_failures
        )
    return ok


def _per_cause_minimum(
    data: SurpyvalData,
    codes: NDArray,
    n_causes: int,
    min_leaf_failures: int,
    per_cause_min: bool,
) -> list[NDArray]:
    """The failure weights of the causes whose count each child must keep:
    with ``per_cause_min`` (a parametric leaf fits each cause on its own),
    every cause the node has enough of to give both children
    ``min_leaf_failures``. A rarer cause may fall wholly in one child; the
    other then has none of it, and gives it zero incidence."""
    if not per_cause_min:
        return []
    n = np.asarray(data.n, dtype=float)
    out = []
    for k in range(n_causes):
        weight = n * (codes == k)
        if weight.sum() >= 2 * min_leaf_failures:
            out.append(weight)
    return out


def cause_specific_split(
    data: SurpyvalData,
    Z: NDArray,
    min_leaf_samples: int,
    min_leaf_failures: int,
    feature_indices_in: Iterable[int],
    causes: list,
    event: Any = None,
    per_cause_min: bool = False,
) -> tuple[int, float]:
    r"""The feature and value of the best cut by the cause-specific
    log-rank rule (see the module docstring): the largest
    :math:`\sum_k L_k^2` over the causes (cause ``event``'s alone when one
    is nominated).

    A cause whose statistic is undefined at a cut (no failures of it in
    the node, or a zero variance) adds nothing there; a cut with no
    defined statistic is not taken. ``min_leaf_failures`` counts every
    failure, and with ``per_cause_min`` also the failures of each cause
    the node has at least twice as many of.

    Returns ``(-1, -inf)`` when no cut is allowed.
    """
    codes = cause_codes(data.e, causes)
    n = np.asarray(data.n, dtype=float)
    failures = n * (codes >= 0)
    wanted = range(len(causes)) if event is None else [causes.index(event)]
    scans = []
    for k in wanted:
        failed = codes == k
        if failed.any():
            scans.append(_LogRankScan(_cause_data(data, failed)))
    per_cause = _per_cause_minimum(
        data, codes, len(causes), min_leaf_failures, per_cause_min
    )
    best = float("-inf")
    best_u, best_v = -1, -float("inf")
    n_rows = len(data)
    if not scans:
        return best_u, best_v
    for u in feature_indices_in:
        Z_u = Z[:, u]
        values = np.unique(Z_u)
        order = np.argsort(Z_u, kind="stable")
        n_left = np.searchsorted(Z_u[order], values, side="right")
        ok = _allowed(
            order,
            n_left,
            n_rows,
            min_leaf_samples,
            min_leaf_failures,
            failures,
            per_cause,
        )
        if not ok.any():
            continue
        if len(scans) == 1:
            # One cause: |L|, the plain log-rank rule's own statistic
            combined = scans[0].statistics(order, n_left[ok])
        else:
            total = np.zeros(int(ok.sum()))
            defined = np.zeros(total.size, dtype=bool)
            for scan in scans:
                stat = scan.statistics(order, n_left[ok])
                finite = np.isfinite(stat)
                total += np.where(finite, stat, 0.0) ** 2
                defined |= finite
            combined = np.where(defined, total, -np.inf)
        statistic = np.full(values.size, -np.inf)
        statistic[ok] = combined
        k = int(np.argmax(statistic))
        if statistic[k] > best:
            best = float(statistic[k])
            best_u, best_v = u, values[k]
    return best_u, best_v


def gray_split(
    data: SurpyvalData,
    Z: NDArray,
    min_leaf_samples: int,
    min_leaf_failures: int,
    feature_indices_in: Iterable[int],
    causes: list,
    event: Any,
    per_cause_min: bool = False,
) -> tuple[int, float]:
    r"""The feature and value of the cut with the largest Gray two-sample
    statistic :math:`U^2 / V` for cause ``event``'s cumulative incidence,
    left child against right (``rho = 0``, the standard test).

    The statistic is that of
    :func:`~surpyval.univariate.competing_risks.gray_test` with the two
    children as the groups. Each allowed cut is scored by its own Gray
    computation over the node's event times, so the cost is the number of
    cuts times the number of event times. The children's counts are kept
    for every cut of a feature at once, a (distinct values x event times)
    array.

    Returns ``(-1, -inf)`` when no cut is allowed or no cut's variance is
    positive.
    """
    codes = cause_codes(data.e, causes)
    target = causes.index(event)
    n = np.asarray(data.n, dtype=float)
    x = np.asarray(data.x, dtype=float)
    failures = n * (codes >= 0)
    per_cause = _per_cause_minimum(
        data, codes, len(causes), min_leaf_failures, per_cause_min
    )
    best = float("-inf")
    best_u, best_v = -1, -float("inf")
    if not (codes == target).any():
        return best_u, best_v
    times, t_idx = np.unique(x, return_inverse=True)
    T = times.size
    cause = (codes == target).astype(float)
    competing = ((codes >= 0) & (codes != target)).astype(float)
    n_rows = len(data)

    def by_time(weight: NDArray) -> NDArray:
        out = np.zeros(T)
        np.add.at(out, t_idx, weight)
        return out

    all_counts = by_time(n)
    all_d1 = by_time(n * cause)
    all_d2 = by_time(n * competing)

    for u in feature_indices_in:
        Z_u = Z[:, u]
        values = np.unique(Z_u)
        order = np.argsort(Z_u, kind="stable")
        n_left = np.searchsorted(Z_u[order], values, side="right")
        ok = _allowed(
            order,
            n_left,
            n_rows,
            min_leaf_samples,
            min_leaf_failures,
            failures,
            per_cause,
        )
        if not ok.any():
            continue
        # Each row's value bin, so the left child of value j is bins <= j
        bins = np.searchsorted(values, Z_u)
        counts = np.zeros((values.size, T))
        d1 = np.zeros((values.size, T))
        d2 = np.zeros((values.size, T))
        np.add.at(counts, (bins, t_idx), n)
        np.add.at(d1, (bins, t_idx), n * cause)
        np.add.at(d2, (bins, t_idx), n * competing)
        counts, d1, d2 = (np.cumsum(a, axis=0) for a in (counts, d1, d2))
        statistic = np.full(values.size, -np.inf)
        for j in np.flatnonzero(ok):
            group_counts = np.vstack([counts[j], all_counts - counts[j]])
            # Everyone with x >= t is at risk at t
            Y = group_counts[:, ::-1].cumsum(axis=1)[:, ::-1]
            D1 = np.vstack([d1[j], all_d1 - d1[j]])
            D2 = np.vstack([d2[j], all_d2 - d2[j]])
            with np.errstate(divide="ignore", invalid="ignore"):
                U, V = _score_and_variance(Y, D1, D2, 0.0)
            if V[0, 0] > 0 and np.isfinite(V[0, 0]):
                statistic[j] = U[0] ** 2 / V[0, 0]
        k = int(np.argmax(statistic))
        if statistic[k] > best:
            best = float(statistic[k])
            best_u, best_v = u, values[k]
    return best_u, best_v


class CauseSplit:
    """What a tree grown with causes passes down to every node: the causes
    (in the tree's order), the split rule, the nominated cause (``None``
    for every cause) and whether each child must keep
    ``min_leaf_failures`` of each cause (a parametric leaf)."""

    def __init__(
        self, causes: list, split: str, event: Any, per_cause_min: bool
    ) -> None:
        self.causes = causes
        self.split = split
        self.event = event
        self.per_cause_min = per_cause_min

    def best_cut(
        self,
        data: SurpyvalData,
        Z: NDArray,
        min_leaf_samples: int,
        min_leaf_failures: int,
        feature_indices_in: Iterable[int],
    ) -> tuple[int, float]:
        """The node's cut by the tree's rule; ``(-1, -inf)`` for none."""
        rule = gray_split if self.split == "gray" else cause_specific_split
        return rule(
            data,
            Z,
            min_leaf_samples,
            min_leaf_failures,
            feature_indices_in,
            self.causes,
            self.event,
            self.per_cause_min,
        )
