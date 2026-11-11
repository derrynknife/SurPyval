r"""The Aalen-Johansen product integral and its variances.

With :math:`dA(t_j)` the ``(K, K)`` matrix of the transition hazard
increments at the event time :math:`t_j` (off the diagonal, the
transitions' increments; on it, minus the sum of its row), the
transition probabilities are the product integral

.. math::
    P(s, t) = \prod_{s < t_j \le t} \left(I + dA(t_j)\right).

Two variance recursions are here: mstate's Greenwood-type and Aalen-type
ones (de Wreede, Fiocco and Putter, 2010), which take the covariance of
the increments, and the infinitesimal jackknife (the derivative of the
estimate in each unit's weight, R ``survfit``'s ``std.err``).

Vectorising a ``(K, K)`` matrix stacks its columns (as R's ``matrix``),
so element ``(a, b)`` is at ``a + K b``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from surpyval.multistate._data import SpellData


@dataclass
class EventTable:
    """The numbers at risk and of transitions at each event time."""

    t: npt.NDArray  # (m,) distinct event times
    r: npt.NDArray  # (K, m) number at risk in each state
    d: npt.NDArray  # (K, K, m) transitions from row-state to column-state
    first: npt.NDArray  # (N,) each row's first event index at risk
    last: npt.NDArray  # (N,) each row's last event index at risk


def event_table(data: SpellData) -> EventTable:
    """The risk sets of the event times: a spell is at risk in its state
    at ``t`` when ``tl < t <= x``."""
    k = data.tm.n_states
    events = data.h >= 0
    t = np.unique(data.x[events])
    m = t.size
    first = np.searchsorted(t, data.tl, side="right")
    last = np.searchsorted(t, data.x, side="right") - 1
    diff = np.zeros((k, m + 1))
    ok = first <= last
    np.add.at(diff, (data.g[ok], first[ok]), data.n[ok])
    np.add.at(diff, (data.g[ok], last[ok] + 1), -data.n[ok])
    r = np.cumsum(diff, axis=1)[:, :m]
    d = np.zeros((k, k, m))
    np.add.at(
        d, (data.g[events], data.h[events], last[events]), data.n[events]
    )
    return EventTable(t=t, r=r, d=d, first=first, last=last)


def increments(table: EventTable) -> npt.NDArray:
    """``(m, K, K)``: the Nelson-Aalen increments :math:`d / r` off the
    diagonal and minus their row sums on it."""
    with np.errstate(divide="ignore", invalid="ignore"):
        dA = np.where(
            table.d > 0, table.d / table.r[:, None, :], 0.0
        ).transpose(2, 0, 1)
    return with_diagonal(dA)


def with_diagonal(dA: npt.NDArray) -> npt.NDArray:
    """``dA`` with each diagonal element minus the sum of the rest of
    its row."""
    dA = np.array(dA, dtype=float)
    k = dA.shape[-1]
    idx = np.arange(k)
    dA[..., idx, idx] = 0.0
    dA[..., idx, idx] = -dA.sum(axis=-1)
    return dA


def start_index(t: npt.NDArray, s: float) -> int:
    """The first event index after ``s`` (events at ``s`` itself are
    before the start: P(s, s) is the identity)."""
    return int(np.searchsorted(t, s, side="right"))


def product_integral(dA: npt.NDArray, j0: int) -> npt.NDArray:
    """``(m, K, K)``: :math:`P(s, t_j)` for each event index ``j``, the
    identity for ``j < j0``."""
    m, k, _ = dA.shape
    out = np.empty((m, k, k))
    P = np.eye(k)
    out[:j0] = P
    for j in range(j0, m):
        P = P @ (np.eye(k) + dA[j])
        out[j] = P
    return out


def greenwood_increment_cov(table: EventTable) -> npt.NDArray:
    """``(m, K^2, K^2)``: the Greenwood-type covariance of the increments
    (mstate's ``msfit(vartype = "greenwood")``): within one 'from' state
    :math:`d_{gh}(r_g - d_{gh}) / r_g^3` on a transition and
    :math:`-d_{gh} d_{gk} / r_g^3` between two, nothing across states."""
    k, _, m = table.d.shape
    out = np.zeros((m, k, k, k, k))
    with np.errstate(divide="ignore", invalid="ignore"):
        r3 = np.where(table.r > 0, table.r**3, np.inf)  # (k, m)
    for g in range(k):
        d = table.d[g]  # (k, m)
        if not np.any(d):
            continue
        cov = -d[:, None, :] * d[None, :, :] / r3[g]
        idx = np.arange(k)
        cov[idx, idx, :] = d * (table.r[g] - d) / r3[g]
        out[:, g, :, g, :] = cov.transpose(2, 0, 1)
    return _fill_diagonals(out)


def aalen_increment_cov(table: EventTable) -> npt.NDArray:
    """``(m, K^2, K^2)``: the Aalen-type covariance of the increments
    (mstate's ``msfit`` default): :math:`d_{gh} / r_g^2` on each
    transition and nothing between two."""
    k, _, m = table.d.shape
    out = np.zeros((m, k, k, k, k))
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.where(table.d > 0, table.d / table.r[:, None, :] ** 2, 0.0)
    for g in range(k):
        for h in range(k):
            if g != h:
                out[:, g, h, g, h] = v[g, h]
    return _fill_diagonals(out)


def _fill_diagonals(cov: npt.NDArray) -> npt.NDArray:
    """Complete ``(m, K, K, K, K)`` covariances of the off-diagonal
    increments with those of the diagonal ones (minus the row sums), and
    return them as ``(m, K^2, K^2)`` in the column-stacked order."""
    m, k = cov.shape[0], cov.shape[1]
    idx = np.arange(k)
    # Diagonal (g, g) against an off-diagonal (g2, h2): minus the sum over
    # the rest of row g.
    off = cov.copy()
    off[:, idx, idx, :, :] = 0.0
    off[:, :, :, idx, idx] = 0.0
    row = -off.sum(axis=2)  # (m, g, g2, h2): cov(dA_gg, dA_g2h2)
    for g in range(k):
        cov[:, g, g, :, :] = row[:, g]
        cov[:, :, :, g, g] = row[:, g]
    for g in range(k):
        for g2 in range(k):
            mask = np.ones(k, bool)
            mask[g2] = False
            cov[:, g, g, g2, g2] = -cov[:, g, g, g2, mask].sum(axis=-1)
    # (g, h) -> g + K h: transpose to (h, g) before the C-order reshape.
    return cov.transpose(0, 2, 1, 4, 3).reshape(m, k * k, k * k)


def propagate_variance(
    dA: npt.NDArray, cov_dA: npt.NDArray, j0: int, method: str
) -> npt.NDArray:
    """``(m, K^2, K^2)``: the covariance of :math:`\\mathrm{vec}\\,P(s,
    t_j)` by mstate's ``probtrans`` forward recursion, zero for
    ``j < j0``. ``method`` is ``"greenwood"`` (the second term at
    :math:`P(s, t_j-)`) or ``"aalen"`` (at :math:`P(s, t_j)`)."""
    m, k, _ = dA.shape
    eye = np.eye(k)
    P = np.eye(k)
    V = np.zeros((k * k, k * k))
    out = np.zeros((m, k * k, k * k))
    for j in range(j0, m):
        step = eye + dA[j]
        left = np.kron(step.T, eye)
        P_new = P @ step
        Pv = P if method == "greenwood" else P_new
        right = np.kron(eye, Pv)
        V = left @ V @ left.T + right @ cov_dA[j] @ right.T
        P = P_new
        out[j] = V
    return out


def robust_variance(
    data: SpellData,
    table: EventTable,
    dA: npt.NDArray,
    j0: int,
    p0: npt.NDArray,
    p0_influence: npt.NDArray | None = None,
) -> npt.NDArray:
    """``(m, A, K)``: the infinitesimal-jackknife variance of each row of
    :math:`p_0 P(s, t_j)`, for the ``(A, K)`` starting distributions
    ``p0``.

    The influence of a unit on a row is carried forward through the
    product, :math:`U(t_j) = U(t_{j-1})(I + dA_j) + p(t_{j-1})
    \\partial dA_j`, where a spell at risk in state :math:`g` moves
    :math:`dA_j[g]` by :math:`(dN - dA_j[g]) / r_g` per unit (``dN`` is
    :math:`e_h - e_g` for its own transition to :math:`h` at :math:`t_j`).
    The variance is :math:`\\sum_c n_c U_c^2` over the units.
    ``p0_influence`` (``(n_units, A, K)``) is the units' influence on
    ``p0`` itself, for an estimated starting distribution.
    """
    m, k, _ = dA.shape
    A = p0.shape[0]
    U = (
        np.zeros((data.n_clusters, A, k))
        if p0_influence is None
        else np.array(p0_influence, dtype=float)
    )
    # The count of each unit (equal on its rows).
    n_unit = np.zeros(data.n_clusters)
    n_unit[data.cluster] = data.n
    out = np.zeros((m, A, k))
    p = np.array(p0, dtype=float)
    out[:j0] = np.einsum("c,cak->ak", n_unit, U**2)
    # The rows at risk at each event: sorted by their first index.
    first, last = table.first, table.last
    order = np.argsort(first, kind="stable")
    first_sorted = first[order]
    active = np.zeros(0, dtype=np.intp)
    pos = int(np.searchsorted(first_sorted, j0, side="left"))
    # Rows that came under risk before j0 and are still at risk.
    before = order[:pos]
    active = before[last[before] >= j0]
    eye = np.eye(k)
    for j in range(j0, m):
        nxt = int(np.searchsorted(first_sorted, j, side="right"))
        if nxt > pos:
            active = np.concatenate([active, order[pos:nxt]])
            pos = nxt
        active = active[last[active] >= j]
        step = eye + dA[j]
        moving = np.flatnonzero(np.any(dA[j] != 0, axis=1))
        U_new = U @ step if moving.size else U
        if moving.size:
            rows = active[np.isin(data.g[active], moving)]
            g = data.g[rows]
            r = table.r[g, j]
            dN = np.zeros((rows.size, k))
            ends = (data.h[rows] >= 0) & (last[rows] == j)
            dN[ends, data.h[rows][ends]] += 1.0
            dN[ends, g[ends]] -= 1.0
            delta = (dN - dA[j][g]) / r[:, None]  # (rows, k)
            contrib = p[:, g].T[:, :, None] * delta[:, None, :]
            np.add.at(U_new, data.cluster[rows], contrib)
        U = U_new
        p = p @ step
        out[j] = np.einsum("c,cak->ak", n_unit, U**2)
    return out


def at_risk_distribution(
    data: SpellData, s: float, first_event: float | None = None
) -> tuple[npt.NDArray, npt.NDArray]:
    """The states of the units at risk just after ``s`` (``tl <= s <
    x``), as a distribution, and each unit's influence on it. With
    ``first_event``, those at risk at that time instead (``tl <
    first_event <= x``): R ``survfit``'s default, for a start at the
    earliest entry, where few may have entered yet."""
    k = data.tm.n_states
    if first_event is None:
        here = (data.tl <= s) & (data.x > s)
    else:
        here = (data.tl < first_event) & (data.x >= first_event)
    total = data.n[here].sum()
    p0 = np.zeros(k)
    if total == 0:
        raise ValueError(
            f"No unit is under observation at s={s:g}, so the states at "
            "s are unknown: give 'from_state', or an 's' at which units "
            "are at risk."
        )
    np.add.at(p0, data.g[here], data.n[here])
    p0 = p0 / total
    infl = np.zeros((data.n_clusters, 1, k))
    rows = np.flatnonzero(here)
    one_hot = np.zeros((rows.size, k))
    one_hot[np.arange(rows.size), data.g[rows]] = 1.0
    np.add.at(infl, data.cluster[rows], ((one_hot - p0) / total)[:, None])
    return p0, infl
