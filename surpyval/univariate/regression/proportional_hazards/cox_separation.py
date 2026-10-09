"""Whether a Cox partial likelihood has no finite maximum, and along
which combination of the coefficients (#728).

The partial likelihood is concave, so it has no finite maximum exactly
when it keeps increasing along some direction ``d`` of the coefficients
that is not flat (a flat one is aliased, see ``cox_ph._cox_aliased``).
Along ``beta + s d`` each event time's term tends, as ``s`` grows, to
the slope ``sum_D d'z - d_t max_R d'z`` (Efron or Breslow; ``D`` the
deaths, ``d_t`` their number, ``R`` the risk set), which is 0 when the
deaths all share the largest value of ``d'z`` in their risk set and
negative otherwise. The exact and Kalbfleisch-Prentice likelihoods ask
less: that each death's ``d'z`` be at least every survivor's, not that
the deaths agree. A direction meeting that at every event time, and
strictly somewhere (a survivor at risk below the deaths), is a monotone
likelihood: the complete separation of the events from the survivors
by the combination ``d'Z`` of the covariates.

The test of each coefficient's information alone (``cox_ph.
_diverged_columns``) sees such a run-off only when ``d`` is a single
column. Along a combination the fit went on to report a "verified"
maximum (#728: ``beta = (61.9, -247.6)`` with standard errors 0.6 and
0.5), and the verdict depended on where the search happened to stop,
and so on the order of the rows. :func:`runoff_direction` decides it
from the data alone.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import linprog

#: The tie methods whose deaths at a time must share the largest ``d'z``
#: (the others ask only that each be at least the survivors').
_TIED_EQUAL = ("efron", "breslow")

#: The most cutting planes added in one round, and the most rounds.
_CUTS_PER_ROUND = 2000
_MAX_ROUNDS = 100


class _RiskSets:
    """The event times of every stratum laid end to end on one axis, and
    where each row sits on it.

    ``lo`` and ``hi`` are, per survivor row ``surv``, the first and last
    positions at which it is at risk as a survivor, and ``dead`` the
    death rows in time order, ``dead_at`` their positions. A row is at
    risk at the event times in ``(tl, x]`` (``cox_at_risk_mask``), and a
    death is a survivor before its own time.
    """

    def __init__(
        self,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
        strata: "npt.NDArray | None",
    ) -> None:
        rows = len(x)
        lo = np.zeros(rows, dtype=int)
        hi = np.full(rows, -1, dtype=int)
        own = np.full(rows, -1, dtype=int)
        labels = np.zeros(rows, dtype=int) if strata is None else strata
        offset = 0
        for s in np.unique(labels):
            idx = np.flatnonzero((labels == s) & (n > 0))
            died = c[idx] == 0
            times = np.unique(x[idx][died])
            first = np.searchsorted(times, tl[idx], side="right")
            last = np.searchsorted(times, x[idx], side="right") - 1
            lo[idx] = offset + first
            hi[idx] = offset + last - died
            own[idx[died]] = offset + last[died]
            offset += times.size
        self.T = offset
        self.surv = np.flatnonzero(lo <= hi)
        self.lo, self.hi = lo[self.surv], hi[self.surv]
        dead = np.flatnonzero(own >= 0)
        order = np.argsort(own[dead], kind="stable")
        self.dead = dead[order]
        self.dead_at = own[self.dead]
        # Where each time's run of deaths starts and ends
        self.first = np.searchsorted(self.dead_at, np.arange(self.T))
        self.last = np.r_[self.first[1:], self.dead.size] - 1

    def deaths_range(self, v: npt.NDArray) -> tuple[npt.NDArray, ...]:
        """Per event time, the least and the largest ``v`` of its deaths,
        and the rows that have them."""
        key = self.dead[np.lexsort((v[self.dead], self.dead_at))]
        low, high = key[self.first], key[self.last]
        return v[low], low, v[high], high


def _range_argmin(
    values: npt.NDArray, lo: npt.NDArray, hi: npt.NDArray
) -> npt.NDArray:
    """For each range ``lo..hi`` (inclusive, ``lo <= hi``), the position
    of the least of ``values`` in it, by a sparse table."""
    table = [np.arange(values.size)]
    length = 1
    while 2 * length <= values.size:
        prev = table[-1]
        a, b = prev[:-length], prev[length:]
        table.append(np.where(values[b] < values[a], b, a))
        length *= 2
    k = np.floor(np.log2(hi - lo + 1)).astype(int)
    out = np.empty(lo.size, dtype=int)
    for level in np.unique(k):
        at = k == level
        row = table[level]
        a = row[lo[at]]
        b = row[hi[at] - (1 << int(level)) + 1]
        out[at] = np.where(values[b] < values[a], b, a)
    return out


def runoff_direction(
    x: npt.NDArray,
    Z: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    strata: "npt.NDArray | None",
    tie_method: str,
) -> "tuple[npt.NDArray, bool] | None":
    """A direction of the coefficients along which the partial likelihood
    of the rows ``x, Z, c, n, tl`` (stratum labels ``strata``, or
    ``None``) keeps increasing without bound, scaled so that its largest
    component is 1 in size, and whether each of its columns runs off
    alone; ``None`` where there is no such direction, so that the maximum
    is finite. ``Z`` should hold only the identified columns (none
    aliased). Columns that run off alone are preferred to a combination:
    the direction is then +-1 at each of them.

    The combination is found by a linear programme in the direction ``d``
    alone (each covariate scaled to unit spread, ``|d_i| <= 1``): every
    survivor at risk at an event time at or below its deaths, the deaths
    level with each other for Efron and Breslow, and the total gap of the
    survivors below the deaths maximised. Those are as many constraints
    as pairs of a unit and an event time it is at risk at, so they are
    added as cutting planes, the ones the last direction broke, each
    round's check costing ``O(n log T)``.
    """
    x = np.asarray(x, dtype=float)
    c = np.asarray(c)
    n = np.asarray(n, dtype=float)
    tl = np.asarray(tl, dtype=float)
    Z = np.asarray(Z, dtype=float).reshape(len(x), -1)
    p = Z.shape[1]
    if p == 0:
        return None
    Zc = Z - (n @ Z) / n.sum()
    scale = np.sqrt((n @ Zc**2) / n.sum())
    scale[~(scale > 0)] = 1.0
    Zs = Zc / scale
    rs = _RiskSets(x, c, n, tl, strata)
    if rs.T == 0 or rs.surv.size == 0:
        return None
    tied_equal = tie_method in _TIED_EQUAL

    def cuts(d: npt.NDArray) -> tuple[list, bool]:
        """The constraints ``d`` breaks, as pairs (row above, row below),
        the worst first, and whether some survivor is clearly below its
        deaths."""
        v = Zs @ d
        tol = 1e-7 * max(1.0, float(np.abs(v).max()))
        v_low, low, v_high, high = rs.deaths_range(v)
        where = _range_argmin(v_low, rs.lo, rs.hi)
        gap = v_low[where] - v[rs.surv]
        broken = np.flatnonzero(gap < -tol)
        pairs = [
            (int(rs.surv[j]), int(low[where[j]]), -gap[j])
            for j in broken.tolist()
        ]
        if tied_equal:
            spread = v_high - v_low
            pairs += [
                (int(high[t]), int(low[t]), spread[t])
                for t in np.flatnonzero(spread > tol).tolist()
            ]
        pairs.sort(key=lambda pair: -pair[2])
        # The widest gap of a survivor below a death, at the time with the
        # highest death of those it is at risk at
        top = v_high[_range_argmin(-v_high, rs.lo, rs.hi)] - v[rs.surv]
        return [pair[:2] for pair in pairs], bool(top.max() > 1e3 * tol)

    # The columns that run off alone, as the test of each coefficient's
    # information reports them, where there are some; together they are
    # a direction too (the directions of increase are a convex cone).
    alone = np.zeros(p)
    for i in range(p):
        for sign in (1.0, -1.0):
            d = np.zeros(p)
            d[i] = sign
            broken, strict = cuts(d)
            if strict and not broken:
                alone[i] = sign
                break
    if np.any(alone):
        return alone, True

    # The total gap, the sum over the event times, their survivors j and
    # their deaths k of n_j n_k (d'z_k - d'z_j): 0 at d = 0.
    n_s, n_d = n[rs.surv], n[rs.dead]
    D_t = np.bincount(rs.dead_at, weights=n_d, minlength=rs.T)
    N_t = np.zeros(rs.T + 1)
    np.add.at(N_t, rs.lo, n_s)
    np.add.at(N_t, rs.hi + 1, -n_s)
    N_t = np.cumsum(N_t)[: rs.T]
    # The deaths over each survivor's range of times
    D_cum = np.r_[0.0, np.cumsum(D_t)]
    reach = n_s * (D_cum[rs.hi + 1] - D_cum[rs.lo])
    cost = reach @ Zs[rs.surv] - (N_t[rs.dead_at] * n_d) @ Zs[rs.dead]
    planes = np.zeros((0, p))
    seen: set = set()
    for _ in range(_MAX_ROUNDS):
        res = linprog(
            cost,
            A_ub=planes if planes.size else None,
            b_ub=np.zeros(len(planes)) if planes.size else None,
            bounds=[(-1.0, 1.0)] * p,
            method="highs",
        )
        if res.status != 0:
            return None
        d = np.asarray(res.x, dtype=float)
        d[np.abs(d) < 1e-9] = 0.0
        if not np.any(d):
            return None
        broken, strict = cuts(d)
        if not broken:
            if not strict:
                return None
            direction = d / scale
            return direction / np.abs(direction).max(), False
        new = [pair for pair in broken if pair not in seen]
        if not new:
            # The solver's tolerance keeps breaking the planes it has
            return None
        new = new[:_CUTS_PER_ROUND]
        seen.update(new)
        above, below = np.array(new).T
        planes = np.vstack([planes, Zs[above] - Zs[below]])
    return None


#: How far a unit's linear predictor may be from the average unit's
#: before the fit asks the data whether its likelihood runs off (#728)
FAR = 20.0

#: The largest Newton decrement, ``sqrt(score' info^-1 score)``, at an
#: answer of Newton-Raphson's that the fit takes without asking the data
#: whether the likelihood runs off (#746).
#:
#: Why that, with an information not collapsed
#: (:func:`information_collapsed`), rules a run-off out. Along a run-off
#: direction ``d`` every event time's term rises: its deaths have the
#: largest ``d'z`` of its risk set (each death at least the survivors',
#: for the exact and KP methods; risk sets weighted, for Fine-Gray), so
#: its slope is the deaths' ``d'z`` less the risk set's weighted mean, at
#: least 0. Its curvature is the weighted variance of ``d'z`` (of the
#: subsets' sums for KP), and as every value lies within ``G`` below the
#: deaths', the variance is at most ``G`` times the slope (``d G`` for d
#: tied deaths): along ``d`` the information is at most ``G`` times the
#: slope ``f'``. By Cauchy-Schwarz the decrement ``lam`` has ``lam^2 >=
#: f'^2 / (d' info d) >= f' / G``, so the information along ``d`` is at
#: most ``G^2 lam^2``. At the start (every weight alike) the risk set
#: where a survivor lies ``G`` below a death has a variance of at least
#: ``G^2 / (2 N)``, ``N`` the units (counted by ``n``, over the least
#: ``n``). An information not collapsed keeps 1e-4 of that, so ``1e-4 /
#: (2 N) <= lam^2 <= 1e-12``: a run-off can pass only with ``N`` above
#: 5e7 (and a risk set as lopsided as that bound). Every other answer
#: (the root-finder's, BFGS's) is suspect: BFGS stopped where the score
#: was below the verification's tolerance in the units of a covariate
#: spanning 1e-7, with every unit within 2 of the average and the
#: information at 0.29 of the start's, and the fit reported a verified
#: maximum of a likelihood with none. A search far out, where the
#: information underflows, is suspect too.
DECREMENT = 1e-6


def newton_converged(res: Any) -> bool:
    """Whether Newton-Raphson's answer ``res`` has a decrement of at most
    ``DECREMENT``: it converged, to its own tolerance at most that."""
    score, hess = np.atleast_1d(res.jac), np.atleast_2d(res.hess)
    try:
        lam2 = float(score @ np.linalg.solve(hess, score))
    except np.linalg.LinAlgError:
        return False
    return bool(abs(lam2) <= DECREMENT**2)


def information_collapsed(
    info: npt.NDArray, info_at_start: npt.NDArray
) -> bool:
    """Whether the information has fallen, in some direction of the
    coefficients, below 1e-4 of what it was at the start (the least
    eigenvalue of ``info`` relative to ``info_at_start``): the sign of a
    run-off along a combination of them, which a maximum's information,
    however strong the effects, does not show. The data then say whether
    it is one (:func:`runoff_direction`)."""
    try:
        L = np.linalg.cholesky(np.atleast_2d(info_at_start))
        half = np.linalg.solve(L, np.atleast_2d(info))
        relative = np.linalg.solve(L, half.T)
        least = np.linalg.eigvalsh(0.5 * (relative + relative.T))[0]
    except np.linalg.LinAlgError:
        return True
    return not least >= 1e-4
