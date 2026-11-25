r"""The likelihood of spells whose move out of a state is known only to
lie between two times (interval censored).

A unit seen in state :math:`a` at :math:`x_l` and in :math:`b` at
:math:`x_r` moved at an unknown time :math:`u` in :math:`(x_l, x_r]`
(directly: a model that may pass through other states unseen between
two looks is a panel model). Its likelihood integrates over :math:`u`:

.. math::
    \int_{x_l}^{x_r} S_a(u)\, h_{ab}(u)\; L_b(u)\, du,

:math:`S_a` the chance of staying in :math:`a` from its entry,
:math:`h_{ab}` the hazard of the move and :math:`L_b(u)` the likelihood
of what was seen of the unit after it, given that it entered :math:`b`
at :math:`u` -- its next spell (which may end in another
interval-censored move, and so on), or, where it has none, the chance of
staying in :math:`b` until :math:`x_r`. On the forward clock the hazards
are functions of the time since the start; on the reset clock, of the
time since entering the state, so that :math:`L_b(u)` depends on
:math:`u` through the next spell's duration (the sojourn densities are
convolved).

Each unit's chain of spells is a product of matrices, one per spell,
whose rows are the nodes of its entry time and whose columns are those
of its exit time (a single node where a time is known): the integrals by
tanh-sinh quadrature, which is accurate where a hazard is infinite at an
end of the interval (a Weibull with a shape below 1, at the start).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import expit

from surpyval.multistate._data import SpellData


def tanh_sinh(
    h: float = 1.0 / 8.0, reach: float = 3.25
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """The nodes of the tanh-sinh rule on (0, 1): each node's distance
    from 0 (``f``) and from 1 (``1 - f``, computed on its own so that it
    is accurate near 1), and the weights."""
    t = np.arange(-reach, reach + h / 2.0, h)
    g = 0.5 * np.pi * np.sinh(t)
    f = expit(2.0 * g)
    fr = expit(-2.0 * g)
    w = h * 0.5 * np.pi * np.cosh(t) * 2.0 * f * fr
    keep = (w > 0) & (f > 0) & (fr > 0)
    return f[keep], fr[keep], w[keep]


_F, _FR, _W = tanh_sinh()


class IntervalLikelihood:
    """The negative log-likelihood of spells, some with interval-censored
    exits, as a function of the transitions' models.

    ``hazard(model, t, rows, rate)`` gives a transition's cumulative
    hazard (its hazard with ``rate``) at the times ``t`` for the spells
    ``rows`` (their covariates); ``lows`` the start of each transition's
    support (an entry at or before it truncates nothing)."""

    def __init__(
        self,
        data: SpellData,
        clock: str,
        hazard: Any,
        lows: list,
    ) -> None:
        self.clock = clock
        self.hazard = hazard
        self.lows = np.asarray(lows, dtype=float)
        tm = data.tm
        self.frm, self.to = tm.from_index(), tm.to_index()
        self.n_states = tm.n_states
        size = data.x.size
        xl = data.x if data.xl is None else data.xl
        xr = data.x
        interval = xl < xr
        self.n = data.n

        # Link each interval-censored move to the spell it began: the
        # unit's spell in the state moved into that starts within the
        # interval.
        succ = np.full(size, -1, dtype=np.intp)
        prev = np.full(size, -1, dtype=np.intp)
        if data.ids_given:
            for k in np.flatnonzero(interval).tolist():
                cand = np.flatnonzero(
                    (data.cluster == data.cluster[k])
                    & (data.g == data.h[k])
                    & (data.tl >= xl[k])
                    & (data.tl <= xr[k])
                )
                cand = cand[cand != k]
                if cand.size > 1:
                    raise ValueError(
                        f"Row {k}'s move into {tm.states[data.h[k]]!r} in "
                        f"({xl[k]:g}, {xr[k]:g}] is followed by "
                        f"{cand.size} spells of its unit in that state "
                        "starting within the interval: give the spell it "
                        "began once."
                    )
                if cand.size == 1:
                    s = int(cand[0])
                    if xl[s] < xr[k]:
                        raise ValueError(
                            f"Row {s} enters {tm.states[data.g[s]]!r} by "
                            f"a move in ({xl[k]:g}, {xr[k]:g}] and leaves "
                            f"it by {xl[s]:g}, inside that interval: the "
                            "move out of a state cannot be before the "
                            "move into it. The next move's interval "
                            "starts at or after the end of the one "
                            "before."
                        )
                    succ[k] = s
                    prev[s] = k
        self.succ, self.prev = succ, prev

        # Each spell's exit nodes: v = xa + xo (from the left end), with
        # the same node from the right end, xr - xo_r; and its weights.
        self.xa = xl.astype(float)
        self.xr = xr.astype(float)
        exit_o, exit_or, exit_w = [], [], []
        for k in range(size):
            if interval[k]:
                L = xr[k] - xl[k]
                exit_o.append(L * _F)
                exit_or.append(L * _FR)
                exit_w.append(L * _W)
            else:
                exit_o.append(np.zeros(1))
                exit_or.append(np.zeros(1))
                exit_w.append(np.ones(1))
        # Each spell's entry nodes: u = ea - eo; "start" where it entered
        # at the start of the clock (no truncation).
        start = np.zeros(size, dtype=bool)
        ea = np.empty(size)
        entry_o = []
        for k in range(size):
            p = int(prev[k])
            if p >= 0:
                ea[k] = xr[p]
                entry_o.append(exit_or[p])
            else:
                entry_o.append(np.zeros(1))
                if np.isfinite(data.tl[k]):
                    ea[k] = data.tl[k]
                else:
                    ea[k] = 0.0
                    start[k] = True

        # The pairs (entry node, exit node) of every spell, flat.
        shapes, slices = [], []
        pe, px, ps, pr = [], [], [], []
        at = 0
        for k in range(size):
            m, q = entry_o[k].size, exit_o[k].size
            shapes.append((m, q))
            slices.append(slice(at, at + m * q))
            at += m * q
            pe.append(np.repeat(entry_o[k], q))
            px.append(np.tile(exit_o[k], m))
            ps.append(np.full(m * q, k, dtype=np.intp))
            pr.append(np.tile(np.log(exit_w[k]), m))
        self.shapes, self.slices = shapes, slices
        self.p_spell = np.concatenate(ps)
        self.p_eo = np.concatenate(pe)
        self.p_xo = np.concatenate(px)
        self.p_logw = np.concatenate(pr)
        self.p_g = data.g[self.p_spell]
        self.p_h = data.h[self.p_spell]
        self.p_start = start[self.p_spell]
        self.p_u = ea[self.p_spell] - self.p_eo
        self.p_v = self.xa[self.p_spell] + self.p_xo
        self.p_dur = (self.xa - ea)[self.p_spell] + self.p_xo + self.p_eo

        # The chance of staying in the state moved into until xr, for a
        # move with no spell after it.
        tails = np.flatnonzero(interval & (succ < 0))
        self.tail_spell = tails
        t_s, t_o = [], []
        for k in tails.tolist():
            t_s.append(np.full(exit_or[k].size, k, dtype=np.intp))
            t_o.append(exit_or[k])
        self.t_spell = (
            np.concatenate(t_s) if t_s else np.zeros(0, dtype=np.intp)
        )
        self.t_or = np.concatenate(t_o) if t_o else np.zeros(0)
        self.t_state = data.h[self.t_spell]
        self.t_u = self.xr[self.t_spell] - self.t_or
        self.t_slices = {}
        at = 0
        for k, o in zip(tails.tolist(), t_o):
            self.t_slices[int(k)] = slice(at, at + o.size)
            at += o.size

        # The chains: the spells linked from a spell with no linked
        # predecessor; a spell on its own (known entry, known exit) is a
        # single term.
        chained = interval | (prev >= 0)
        self.single = np.flatnonzero(~chained)
        self.chains = []
        for k in np.flatnonzero(interval & (prev < 0)).tolist():
            chain = [int(k)]
            while succ[chain[-1]] >= 0:
                chain.append(int(succ[chain[-1]]))
            self.chains.append(chain)
        # The chains in groups of the same shapes (and with or without a
        # tail), each spell's pair indices stacked, so that a group's
        # products are taken at once.
        groups: dict = {}
        for chain in self.chains:
            key = (
                tuple(self.shapes[k] for k in chain),
                chain[-1] in self.t_slices,
            )
            groups.setdefault(key, []).append(chain)
        self.groups = []
        for (shapes_, tail), chains in groups.items():
            idx = [
                np.array(
                    [
                        np.arange(
                            self.slices[c[j]].start, self.slices[c[j]].stop
                        )
                        for c in chains
                    ]
                ).reshape((len(chains),) + shapes_[j])
                for j in range(len(shapes_))
            ]
            tidx = (
                np.array(
                    [
                        np.arange(
                            self.t_slices[c[-1]].start,
                            self.t_slices[c[-1]].stop,
                        )
                        for c in chains
                    ]
                )
                if tail
                else None
            )
            n = self.n[[c[0] for c in chains]]
            self.groups.append((idx, tidx, n))

    # -- evaluation -----------------------------------------------------------

    def _log_kernel(self, models: list) -> tuple[npt.NDArray, npt.NDArray]:
        """The log of each pair's term (its quadrature weight included),
        and the log chance of each tail node."""
        logk = self.p_logw.copy()
        logt = np.zeros(self.t_spell.size)
        forward = self.clock == "forward"
        for h, m in enumerate(models):
            g, b = self.frm[h], self.to[h]
            sel = np.flatnonzero(self.p_g == g)
            if sel.size:
                rows = self.p_spell[sel]
                if forward:
                    v = self.p_v[sel]
                    dH = np.ravel(self.hazard(m, v, rows, False))
                    u = self.p_u[sel]
                    trunc = ~self.p_start[sel] & (u > self.lows[h])
                    if np.any(trunc):
                        dH = dH.copy()
                        dH[trunc] -= np.ravel(
                            self.hazard(m, u[trunc], rows[trunc], False)
                        )
                    at = v
                else:
                    at = self.p_dur[sel]
                    dH = np.ravel(self.hazard(m, at, rows, False))
                logk[sel] -= dH
                ev = self.p_h[sel] == b
                if np.any(ev):
                    rate = np.ravel(self.hazard(m, at[ev], rows[ev], True))
                    logk[sel[ev]] += np.log(rate)
            tsel = np.flatnonzero(self.t_state == g)
            if tsel.size:
                rows = self.t_spell[tsel]
                end = self.xr[rows]
                if forward:
                    H1 = np.ravel(self.hazard(m, end, rows, False))
                    u = self.t_u[tsel]
                    H0 = np.where(
                        u > self.lows[h],
                        np.ravel(self.hazard(m, u, rows, False)),
                        0.0,
                    )
                    logt[tsel] -= H1 - H0
                else:
                    logt[tsel] -= np.ravel(
                        self.hazard(m, self.t_or[tsel], rows, False)
                    )
        return logk, logt

    def neg_ll(self, models: list) -> float:
        with np.errstate(all="ignore"):
            logk, logt = self._log_kernel(models)
        total = 0.0
        if self.single.size:
            first = np.array([self.slices[k].start for k in self.single])
            total += float(np.sum(self.n[self.single] * logk[first]))
        for idx, tidx, n in self.groups:
            log_scale = np.zeros(n.size)
            vec: Any = None
            for K_idx in idx:
                K = logk[K_idx]
                top = np.max(K, axis=(1, 2))
                if not np.all(np.isfinite(top)):
                    return np.inf
                M = np.exp(K - top[:, None, None])
                log_scale += top
                vec = (
                    M[:, 0, :]
                    if vec is None
                    else np.einsum("gi,gij->gj", vec, M)
                )
                scale = np.max(vec, axis=1)
                if not np.all(scale > 0):
                    return np.inf
                vec = vec / scale[:, None]
                log_scale += np.log(scale)
            if tidx is not None:
                vec = vec * np.exp(logt[tidx])
            value = np.sum(vec, axis=1)
            if not np.all(value > 0):
                return np.inf
            total += float(np.sum(n * (log_scale + np.log(value))))
        return -total


def midpoint_data(data: SpellData, succ: npt.NDArray) -> SpellData:
    """The spells with each interval-censored move put at the middle of
    its interval (and the spell it began starting there): the data of
    the starting fits."""
    from dataclasses import replace

    xl = data.x if data.xl is None else data.xl
    mid = np.where(xl < data.x, 0.5 * (xl + data.x), data.x)
    tl = data.tl.copy()
    for k in np.flatnonzero(succ >= 0):
        tl[succ[k]] = mid[k]
    return replace(data, x=mid, tl=tl, xl=None)
