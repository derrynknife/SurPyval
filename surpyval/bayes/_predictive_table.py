"""A posterior predictive tabulated once, to a stated accuracy.

A grid posterior has thousands of points (tens of thousands with two or
three free parameters), and each value of its predictive distribution is
a sum over all of them: a fraction of a millisecond per point per value,
so a calculation that asks for tens of thousands of values (an optimal
replacement interval, a failure-finding search, an integral) took
minutes.

Here the predictive cumulative hazard ``H(t)`` is worked out exactly (the
full sum) at a set of times and read between them by a cubic in ``log H``
against ``log(t - edge)`` (``t`` itself for a distribution on the whole
line), with the exact slope at each node (``t h(t) / H(t)``, from the
exact density), so the cubic is a Hermite one: its error falls with the
fourth power of the spacing. The nodes are placed adaptively. Each
interval is checked against the full sum at two points, its middle and
0.15 of its width (between them they see both the cubic's value and its
slope, the hazard, near their largest errors), on ``sf``, on ``ff``
(relative to itself, so the lower tail keeps its digits) and on the
hazard (to ten times the tolerance: it is the cubic's slope), and split
there until it is within the tolerance: every interval read from the
table has been checked itself. An interval that cannot be brought within
the tolerance (a kink, a stretch where ``H`` is flat to a rounding), or
whose cubic is not monotone, is left to the full sum. Outside the
tabulated range (the far tails) the full sum is used, so no value is
extrapolated.

Below the predictive support's lower edge (the smallest offset, or the
distribution's own edge) the values are known exactly: nothing has failed,
or only the zero-inflation mass.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.bayes.prior import quantity_at

#: The relative error each interval is checked to, on ``sf`` and ``ff``
#: (and ``hf``, see ``_HAZARD``). The points checked see at least 64% of
#: the largest error (see ``_CHECK_AT``), so the table's error is inside
#: 1e-6 relative.
RTOL = 3e-7
#: The hazard (and density) is checked to this multiple of ``RTOL``: it
#: is the cubic's slope, an order less accurate than its value, and
#: holding it to ``RTOL`` too took about three times the nodes. So
#: ``hf`` and ``df`` are within 1e-5 relative.
_HAZARD = 10.0
#: A posterior with more points than this is tabulated; one with fewer is
#: summed exactly (it is cheap, and a small set of draws is often a test
#: or a one-point check that wants the exact mixture).
TABULATE_ABOVE = 2000
#: The probability that sets the tabulated range: from where the
#: posterior's points (all but a thousandth of their weight) have a
#: failure probability of 1e-10, to where they have a survival
#: probability of 1e-10. Outside it the full sum is used.
_TAIL = 1e-10
_MASS = 1e-3
#: With a limited failure population the upper end is where the points
#: that fail have a survival probability of 1e-6: past it ``H`` is flat to
#: a few digits short of a rounding, which is no place to read a hazard
#: from a table (the full sum is used there).
_TAIL_LFP = 1e-6
#: With zero inflation the lower end is where the points have a failure
#: probability (past the mass at 0) of 1e-6: below it ``H`` differs from
#: the mass's own by too few digits to read a hazard from.
_TAIL_ZI = 1e-6
#: The starting nodes, the most rounds of splitting, and the budget: the
#: most evaluations of the full sum, as values of the points' functions
#: (so a posterior of fewer points may take more nodes), and the fewest.
_START = 17
_ROUNDS = 40
_BUDGET = 5e7
_FEWEST = 300
#: Where in an interval it is checked: at two points. A cubic Hermite's
#: error is, to leading orders, ``s^2 (1 - s)^2 (a + b (s - 1/2))``: its
#: even part's value is largest at the middle, where its slope (the
#: hazard) is zero, and the slope is largest at 0.21 and 0.79; the odd
#: part's slope is largest at the middle and zero at 0.28 and 0.72. The
#: middle and 0.15 see each part's value and slope at 64% to 100% of its
#: largest.
_CHECK_AT = 0.5
_CHECK_AT2 = 0.15


def _hermite(
    v: npt.NDArray,
    V: npt.NDArray,
    Y: npt.NDArray,
    D: npt.NDArray,
    k: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """The cubic Hermite interpolant of ``(V, Y)`` with slopes ``D`` on
    interval ``k``, and its derivative, at ``v``."""
    v0, v1 = V[k], V[k + 1]
    y0, y1 = Y[k], Y[k + 1]
    w = v1 - v0
    d0, d1 = D[k] * w, D[k + 1] * w
    s = (v - v0) / w
    s2 = s * s
    s3 = s2 * s
    y = (
        (2 * s3 - 3 * s2 + 1) * y0
        + (s3 - 2 * s2 + s) * d0
        + (3 * s2 - 2 * s3) * y1
        + (s3 - s2) * d1
    )
    dy = (
        (6 * s2 - 6 * s) * (y0 - y1)
        + (3 * s2 - 4 * s + 1) * d0
        + (3 * s2 - 2 * s) * d1
    ) / w
    return y, dy


class PredictiveTable:
    """The predictive cumulative hazard of a Bayesian fit, tabulated.

    ``exact(t)`` must return the predictive ``(F, S, f, H)`` at the 1-D
    ``t`` by the full sum (``H`` with its digits in both tails).
    """

    def __init__(self, model: Any, exact: Any, rtol: float = RTOL):
        self.rtol = float(rtol)
        th = model._theta()
        N = len(model.posterior.weights)
        dist = model.dist
        lo_edge = float(dist.support[0])
        gamma = np.asarray(th.get("gamma", np.zeros(N)), dtype=float)
        self.zi = bool(model.zi)
        self.log_time = bool(np.isfinite(lo_edge))
        self.edge = (
            lo_edge + float(np.min(gamma)) if self.log_time else -np.inf
        )
        # Below the edge: nothing has failed, or (zero inflation, from 0)
        # only the mass at 0, whose share is the same at every such time.
        self.below = (0.0, 1.0)
        if self.zi and self.log_time and self.edge > 0:
            F0, S0, _, _ = exact(np.array([0.0]))
            self.below = (float(F0[0]), float(S0[0]))
        # The range: the points' own quantiles of the base distribution
        # (past their offsets), all but _MASS of the weight inside.
        w = np.asarray(model.posterior.weights, dtype=float)
        base = {n: th[n] for n in dist.parameter_names}

        def where(p: float, level: float) -> float:
            with np.errstate(all="ignore"):
                q = gamma + np.asarray(
                    quantity_at(dist, ("q", p), base), dtype=float
                )
            ok = np.isfinite(q)
            if not np.any(ok):
                return float("nan")
            q, ww = q[ok], w[ok]
            o = np.argsort(q, kind="stable")
            c = np.cumsum(ww[o])
            i = int(np.searchsorted(c, level * c[-1]))
            return float(q[o][min(i, len(q) - 1)])

        t_lo = where(_TAIL_ZI if self.zi else _TAIL, _MASS)
        t_hi = where(1 - (_TAIL_LFP if model.lfp else _TAIL), 1 - _MASS)
        self.ok = bool(np.isfinite(t_lo) and np.isfinite(t_hi) and t_hi > t_lo)
        if self.log_time and self.ok:
            t_lo = max(t_lo, self.edge)
            self.ok = t_hi > self.edge and t_lo < t_hi
            if t_lo <= self.edge:
                # (a little above the edge, where log(t - edge) is finite)
                t_lo = self.edge + (t_hi - self.edge) * 1e-12
        # Each point's own support edge (past its offset), on the table's
        # scale: kinks in H
        kinks = np.unique(lo_edge + gamma) if self.log_time else np.empty(0)
        kinks = kinks[kinks > self.edge] if kinks.size > 1 else kinks[:0]
        self._kinks = self._v(kinks)
        self.evaluations = 0
        self.budget = max(_FEWEST, int(_BUDGET / max(N, 1)))
        if not self.ok:
            return
        self._exact = exact
        v_lo, v_hi = self._v(np.array([t_lo, t_hi]))
        self._build(float(v_lo), float(v_hi))
        self._exact = None

    # -- coordinates -------------------------------------------------------
    def _v(self, t: npt.NDArray) -> npt.NDArray:
        if self.log_time:
            with np.errstate(all="ignore"):
                return np.log(t - self.edge)
        return t

    def _t(self, v: npt.NDArray) -> npt.NDArray:
        return self.edge + np.exp(v) if self.log_time else v

    def _dtdv(self, t: npt.NDArray) -> npt.NDArray:
        return t - self.edge if self.log_time else np.ones_like(t)

    def _nodes(
        self, v: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """``log H``, its slope in ``v``, and ``H`` and ``h`` at ``v``,
        by the full sum."""
        self.evaluations += len(v)
        t = self._t(v)
        F, S, f, H = self._exact(t)
        with np.errstate(all="ignore"):
            h = f / S
            Y = np.log(H)
            D = h * self._dtdv(t) / H
        return Y, D, H, h

    # -- building ----------------------------------------------------------
    def _build(self, v_lo: float, v_hi: float) -> None:
        V = np.linspace(v_lo, v_hi, _START)
        Y, D, _, _ = self._nodes(V)
        # Intervals waiting to be checked (their ends), with the error
        # their parent was checked at (inf for the first ones)
        lo, hi = V[:-1].copy(), V[1:].copy()
        parent = np.full(lo.shape, np.inf)
        exact_iv: list[tuple[float, float]] = []
        min_width = (v_hi - v_lo) * 2.0**-30
        for _ in range(_ROUNDS):
            if lo.size == 0:
                break
            n = lo.size
            k = np.searchsorted(V, lo)
            # (two check points an interval, ``first < second``)
            first = lo + _CHECK_AT2 * (hi - lo)
            second = lo + _CHECK_AT * (hi - lo)
            at = np.concatenate([first, second])
            Ya, Da, Ha, ha = self._nodes(at)
            ya, dya = _hermite(at, V, Y, D, np.concatenate([k, k]))
            ea = self._error(at, ya, dya, Ha, ha, self.rtol)
            err = np.maximum(ea[:n], ea[n:])
            # An interval holding a point's own support edge (an offset) has
            # a kink in it, where the leading-order error shape the check
            # points rely on fails: it is held to a tenth of the tolerance
            kinked = np.searchsorted(
                self._kinks, lo, "right"
            ) < np.searchsorted(self._kinks, hi, "left")
            err = np.where(kinked, 10.0 * err, err)
            fin = np.isfinite(Ya) & np.isfinite(Da)
            finite = fin[:n] & fin[n:]
            ok = err <= 1.0
            room = self.evaluations + 2 * n <= self.budget
            # Hopeless: too narrow, out of budget, not finite, or splitting
            # did not lower the error at all (a kink, or a stretch where
            # H is flat to a rounding, which more nodes do not help)
            give_up = ~ok & (
                (hi - lo < min_width) | (not room) | ~finite | (err > parent)
            )
            exact_iv.extend(zip(lo[give_up].tolist(), hi[give_up].tolist()))
            # An interval that failed is split at its check points, which
            # become nodes; one that passed is kept as it is, so every
            # interval read from the table was checked itself
            split = ~ok & finite & ~give_up
            both = np.concatenate([split, split])
            V2 = np.concatenate([V, at[both]])
            Y2 = np.concatenate([Y, Ya[both]])
            D2 = np.concatenate([D, Da[both]])
            o = np.argsort(V2, kind="stable")
            V, Y, D = V2[o], Y2[o], D2[o]
            # Its three pieces are checked next, each with its error
            lo, hi, parent = (
                np.concatenate([lo[split], first[split], second[split]]),
                np.concatenate([first[split], second[split], hi[split]]),
                np.tile(err[split], 3),
            )
        else:
            exact_iv.extend(zip(lo.tolist(), hi.tolist()))
        self.V, self.Y, self.D = V, Y, D
        good = np.isfinite(Y[:-1]) & np.isfinite(Y[1:])
        good &= np.isfinite(D[:-1]) & np.isfinite(D[1:])
        # A monotone cubic on each interval (Fritsch and Carlson's
        # sufficient condition), so sf falls and qf has one root
        dy = Y[1:] - Y[:-1]
        w = V[1:] - V[:-1]
        with np.errstate(all="ignore"):
            a = D[:-1] * w / dy
            b = D[1:] * w / dy
        mono = (dy > 0) & (a >= 0) & (b >= 0) & (a * a + b * b <= 9.0)
        flat = (dy == 0) & (D[:-1] == 0) & (D[1:] == 0)
        good &= mono | flat
        for e_lo, e_hi in exact_iv:
            good &= ~((V[:-1] >= e_lo) & (V[1:] <= e_hi))
        self.good = good
        self.v_lo, self.v_hi = float(V[0]), float(V[-1])

    def _error(
        self,
        v: npt.NDArray,
        y: npt.NDArray,
        dy: npt.NDArray,
        H: npt.NDArray,
        h: npt.NDArray,
        rtol: float,
    ) -> npt.NDArray:
        """The largest relative error of ``sf``, ``ff`` and ``hf`` from the
        interpolated ``log H`` and its slope at ``v``, against the full
        sum's ``H`` and ``h``, as a multiple of ``rtol`` (inf where the
        values are not finite)."""
        with np.errstate(all="ignore"):
            Hi = np.exp(y)
            e_sf = np.abs(np.expm1(H - Hi))
            F = -np.expm1(-H)
            e_ff = np.abs(-np.expm1(-Hi) - F) / F
            hi = Hi * dy / self._dtdv(self._t(v))
            e_hf = np.where(h > 0, np.abs(hi - h) / h, np.abs(hi))
            err = np.maximum(np.maximum(e_sf, e_ff), e_hf / _HAZARD) / rtol
        bad = ~(np.isfinite(err) & np.isfinite(H) & (H > 0))
        return np.where(bad, np.inf, err)

    # -- reading -----------------------------------------------------------
    def read(
        self, t: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """``(where, H, h)`` at the 1-D ``t``: ``where`` is 1 where the
        table gives the values, 2 below the support's edge (``H`` and
        ``h`` then the known constants), 0 where the full sum must be
        used."""
        t = np.asarray(t, dtype=float)
        where = np.zeros(t.shape, dtype=np.int8)
        H = np.full(t.shape, np.nan)
        h = np.full(t.shape, np.nan)
        if not self.ok:
            return where, H, h
        if self.log_time:
            below = t < self.edge
            if self.zi:
                # (before 0 nothing has failed, the mass at 0 included)
                F0 = np.where(t < 0, 0.0, self.below[0])
                S0 = np.where(t < 0, 1.0, self.below[1])
            else:
                F0, S0 = np.zeros(t.shape), np.ones(t.shape)
            with np.errstate(divide="ignore"):
                H0 = np.where(F0 < 0.5, -np.log1p(-F0), -np.log(S0))
            where[below] = 2
            H[below] = H0[below] + 0.0
            h[below] = 0.0
        v = self._v(t)
        inside = (where == 0) & (v >= self.v_lo) & (v <= self.v_hi)
        if np.any(inside):
            vi = v[inside]
            k = np.clip(
                np.searchsorted(self.V, vi, side="right") - 1,
                0,
                len(self.V) - 2,
            )
            use = self.good[k]
            y, dy = _hermite(vi, self.V, self.Y, self.D, k)
            with np.errstate(all="ignore"):
                Hi = np.exp(y)
                hi = Hi * dy / self._dtdv(t[inside])
            idx = np.flatnonzero(inside)[use]
            where[idx] = 1
            H[idx] = Hi[use]
            h[idx] = np.maximum(hi[use], 0.0)
        return where, H, h
