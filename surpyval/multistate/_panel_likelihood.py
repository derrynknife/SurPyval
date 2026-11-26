r"""The likelihood of states seen only at visits (panel data) when the
transition hazards are not constant.

On the forward clock (a non-homogeneous Markov model, each hazard a
function of the time since the start) a unit's likelihood is the product
over the gaps between its visits of elements of the transition
probability matrices :math:`P(t_{j-1}, t_j)`, found by product
integration of the Kolmogorov forward equations (:mod:`._forward`) on a
grid fixed for the search: every visit time is a node, and the nodes in
between are those the adaptive refinement needs at the starting values.
The units are carried through the grid together (a forward algorithm:
the chance of each state at the time, given what was seen so far).

On the reset clock (a semi-Markov model, each hazard a function of the
time since entering the state) the state alone does not carry the
future: the time it was entered does too. The model is then for
progressive models only (no way back to a state once left), so that the
paths between two visits are finite. The unit is carried as a
distribution over its state and its time of entry (on tanh-sinh nodes in
the gap where the entry happened); a move between visits integrates over
its unknown time, and a pass through states not seen nests the
integrals, one per state passed through.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate._data import SpellData, check_spells
from surpyval.multistate._interval import tanh_sinh
from surpyval.multistate.transitions import TransitionMatrix

_PANEL, _EXACT, _ENTRY = 0, 1, 2

# The rule for a new state's entry time in a gap, and the coarser one for
# the entry times of states passed through unseen (nested inside it).
_F, _FR, _W = tanh_sinh()
_FI, _FRI, _WI = tanh_sinh(h=1.0 / 4.0, reach=3.0)


# ---------------------------------------------------------------------------
# The structure of the transitions
# ---------------------------------------------------------------------------
def has_cycle(tm: TransitionMatrix) -> bool:
    """Whether a unit can come back to a state it has left."""
    K = tm.n_states
    out: list = [[] for _ in range(K)]
    for a, b in zip(tm.from_index(), tm.to_index()):
        out[a].append(b)
    colour = [0] * K

    def visit(v: int) -> bool:
        colour[v] = 1
        for w in out[v]:
            if colour[w] == 1 or (colour[w] == 0 and visit(w)):
                return True
        colour[v] = 2
        return False

    return any(colour[v] == 0 and visit(v) for v in range(K))


def all_paths(tm: TransitionMatrix) -> dict:
    """``{(a, b): [path, ...]}``: every path of states from ``a`` to
    ``b`` (``a != b``) of a model with no cycles."""
    K = tm.n_states
    out: list = [[] for _ in range(K)]
    for a, b in zip(tm.from_index(), tm.to_index()):
        out[a].append(b)
    paths: dict = {}

    def walk(path: list) -> None:
        for w in out[path[-1]]:
            new = path + [w]
            paths.setdefault((new[0], w), []).append(tuple(new))
            walk(new)

    for a in range(K):
        walk([a])
    return paths


def shortest_path(tm: TransitionMatrix, a: int, b: int) -> list | None:
    out: list = [[] for _ in range(tm.n_states)]
    for f, t in zip(tm.from_index(), tm.to_index()):
        out[f].append(t)
    prev = {a: -1}
    queue = deque([a])
    while queue:
        v = queue.popleft()
        if v == b:
            path = [b]
            while prev[path[-1]] >= 0:
                path.append(prev[path[-1]])
            return path[::-1]
        for w in out[v]:
            if w not in prev:
                prev[w] = v
                queue.append(w)
    return None


# ---------------------------------------------------------------------------
# Spells with each move put in the middle of its gap
# ---------------------------------------------------------------------------
def midpoint_spells(pdata: Any) -> SpellData:
    """The visits as spells, each change of state put in the middle of
    its gap (a pass through states not seen spread evenly over it, along
    the shortest path), an exactly timed move at its time: the times the
    predictions start from (the units' states at their first visits) and
    the span of the data. Refuses visits no allowed path can join."""
    tm = pdata.tm
    K = tm.n_states
    absorbing = np.zeros(K, dtype=bool)
    absorbing[[tm.index(s) for s in tm.absorbing]] = True
    rows: list = []  # (unit, state, tl, x, e)
    x, unit, obs, kind = pdata.x, pdata.unit, pdata.obs, pdata.kind
    first = np.r_[True, unit[1:] != unit[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], x.size]
    for lo, hi in zip(starts.tolist(), ends.tolist()):
        state = int(np.flatnonzero(obs[lo])[0])
        entered = float(x[lo])
        t_prev = float(x[lo])
        for r in range(lo + 1, hi):
            if absorbing[state]:
                break
            t = float(x[r])
            allowed = np.flatnonzero(obs[r])
            if state in allowed and kind[r] != _ENTRY:
                t_prev = t
                continue
            path = None
            for b in allowed.tolist():
                path = shortest_path(tm, state, b)
                if path is not None:
                    break
            if path is None:
                raise ValueError(
                    f"Unit {pdata.ids[unit[r]]!r} is seen in "
                    f"{tm.states[state]!r} at {t_prev:g} and then in "
                    f"{[tm.states[b] for b in allowed]!r} at {t:g}, but "
                    "no allowed transitions lead there: check "
                    "'transitions' (or the unit's states)."
                )
            moves = len(path) - 1
            if kind[r] == _PANEL:
                times = t_prev + (t - t_prev) * np.arange(1, moves + 1) / (
                    moves + 1
                )
            else:
                times = t_prev + (t - t_prev) * np.arange(1, moves + 1) / moves
            for j in range(moves):
                rows.append(
                    (
                        pdata.ids[unit[r]],
                        path[j],
                        entered,
                        float(times[j]),
                        path[j + 1],
                    )
                )
                entered = float(times[j])
            state = path[-1]
            t_prev = t
        if not absorbing[state] and t_prev > entered:
            rows.append((pdata.ids[unit[lo]], state, entered, t_prev, -1))
    if not rows:
        raise ValueError(
            "No unit is seen at two different times: panel data need at "
            "least two visits of a unit."
        )
    i, g, tl, xs, e = (list(c) for c in zip(*rows))
    return check_spells(
        np.array(xs, dtype=float),
        np.array([None if h < 0 else tm.states[h] for h in e], dtype=object),
        [tm.states[s] for s in g],
        np.array(tl, dtype=float),
        i,
        None,
        tm,
    )


# ---------------------------------------------------------------------------
# The forward clock: product integration on a fixed grid
# ---------------------------------------------------------------------------
class ForwardPanelLikelihood:
    """The negative log-likelihood of panel data on the forward clock,
    as a function of the transitions' models. ``hazards(models)`` gives
    the ``cumhaz(t, rate)`` function of :mod:`._forward`."""

    def __init__(
        self,
        pdata: Any,
        hazards: Any,
        breaks: npt.ArrayLike,
        tol: float,
    ) -> None:
        self.pdata = pdata
        self.hazards = hazards
        self.breaks = np.asarray(breaks, dtype=float).ravel()
        self.tol = tol
        tm = pdata.tm
        self.frm, self.to, self.K = tm.from_index(), tm.to_index(), tm.n_states
        x = pdata.x
        self.scale = float(x.max() - x.min()) or 1.0
        self.first = np.r_[True, pdata.unit[1:] != pdata.unit[:-1]]
        self.nodes = np.unique(x)

    def plan(self, models: list) -> None:
        """Fix the grid: every visit time, and the nodes the refinement
        adds between them at ``models``."""
        from surpyval.multistate import _forward

        cumhaz = self.hazards(models)
        pdata = self.pdata
        x = pdata.x
        breaks = self.breaks[(self.breaks > x.min()) & (self.breaks < x.max())]
        start = np.unique(np.concatenate([np.unique(x), breaks]))
        if start.size > 1:
            start = _forward._refine(
                cumhaz,
                start,
                self.scale,
                self.tol,
                self.frm,
                self.to,
                self.K,
                # (a floor far below tol: next to a hazard infinite at
                # 0 the intervals stop halving there)
                floor=self.tol * 1e-4,
            )
        self.nodes = start
        node = np.searchsorted(self.nodes, x)
        # Each gap between two visits of a unit: its nodes, its visit (the
        # row ending it), and its place in the unit's sequence of gaps.
        ends = np.flatnonzero(~self.first)
        self.gap_row = ends
        self.gap_a = node[ends - 1]
        self.gap_b = node[ends]
        unit = pdata.unit
        starts = np.flatnonzero(self.first)
        ordinal = ends - starts[unit[ends]]
        self.gap_ordinal = ordinal
        self.gap_unit = unit[ends]
        self.levels = int(
            np.max(self.gap_b - self.gap_a, initial=1)
        ).bit_length()

    def _gap_products(self, steps: npt.NDArray) -> npt.NDArray:
        """``P(t_{j-1}, t_j)`` of every gap, the product of its steps,
        from products over blocks of 2^l steps."""
        K = self.K
        table = [steps]
        for level in range(1, self.levels):
            half = 1 << (level - 1)
            prev = table[-1]
            if prev.shape[0] <= half:
                break
            table.append(prev[:-half] @ prev[half:])
        P = np.broadcast_to(np.eye(K), (self.gap_a.size, K, K)).copy()
        pos = self.gap_a.copy()
        rem = self.gap_b - self.gap_a
        for level in range(len(table) - 1, -1, -1):
            sel = np.flatnonzero(rem & (1 << level))
            if sel.size:
                P[sel] = P[sel] @ table[level][pos[sel]]
                pos[sel] += 1 << level
        return P

    def neg_ll(self, models: list) -> float:
        from surpyval.multistate import _forward

        pdata, K = self.pdata, self.K
        cumhaz = self.hazards(models)
        nodes = self.nodes
        H = _forward._checked(cumhaz, nodes)
        if nodes.size > 1:
            # (one Magnus step per interval: the grid was refined until
            # one step and two halves agree to the tolerance)
            steps = _forward._step(
                cumhaz,
                nodes[:-1],
                nodes[1:],
                H[:-1],
                H[1:],
                self.frm,
                self.to,
                K,
            )
        else:
            steps = np.zeros((0, K, K))
        P = self._gap_products(steps)
        rows = self.gap_row
        kind = pdata.kind[rows]
        obs = pdata.obs[rows].astype(float)
        exact = kind != _PANEL
        Q = np.zeros((rows.size, K, K))
        if np.any(exact):
            rate = _forward._checked(cumhaz, nodes[self.gap_b[exact]], True)
            Qe = np.zeros((rate.shape[0], K, K))
            Qe[:, self.frm, self.to] = rate
            Q[exact] = Qe
        # The matrix carrying the chance of each state at one visit to the
        # (unnormalised) chance of each at the next, given what was seen.
        M = P * obs[:, None, :]
        en = kind == _ENTRY
        if np.any(en):
            M[en] = (P[en] @ Q[en]) * obs[en][:, None, :]
        ex = kind == _EXACT
        if np.any(ex):
            dH = H[self.gap_b[ex]] - H[self.gap_a[ex]]
            out = np.zeros((dH.shape[0], K))
            np.add.at(out.T, self.frm, dH.T)
            R = Q[ex] + np.eye(K)
            M[ex] = np.exp(-out)[:, :, None] * R * obs[ex][:, None, :]
        alpha = pdata.obs[self.first].astype(float)
        total = 0.0
        for j in range(1, int(self.gap_ordinal.max(initial=0)) + 1):
            g = np.flatnonzero(self.gap_ordinal == j)
            u = self.gap_unit[g]
            new = np.einsum("uk,ukl->ul", alpha[u], M[g])
            sc = new.sum(axis=1)
            if not np.all(sc > 0) or not np.all(np.isfinite(sc)):
                return np.inf
            total += float(np.sum(np.log(sc)))
            alpha[u] = new / sc[:, None]
        return -total


# ---------------------------------------------------------------------------
# The reset clock: integrals over the unseen entry times
# ---------------------------------------------------------------------------
class _Evaluator:
    """The hazards a likelihood asks for, recorded on a first pass (in
    the order asked, the values not yet known) and then evaluated a
    transition at a time and handed back in the same order: the calls of
    one pass do not depend on the values."""

    def __init__(self, frm: npt.NDArray, n_states: int) -> None:
        self.frm = frm
        self.out_of = [np.flatnonzero(frm == a) for a in range(n_states)]
        self.requests: list = []  # (what, key, array)
        self.values: list | None = None
        self.counter = 0

    def _ask(self, what: str, key: int, d: npt.NDArray) -> npt.NDArray:
        if self.values is None:
            self.requests.append((what, key, d))
            return np.zeros(d.shape)
        v = self.values[self.counter]
        self.counter += 1
        return v

    def Hout(self, a: int, d: npt.NDArray) -> npt.NDArray:
        """The cumulative hazard of leaving ``a`` after ``d`` in it."""
        if self.out_of[a].size == 0:
            return np.zeros(np.shape(d))
        return self._ask("H", a, np.asarray(d, dtype=float))

    def rate(self, h: int, d: npt.NDArray) -> npt.NDArray:
        """The hazard of the transition ``h`` after ``d`` in its state."""
        return self._ask("h", h, np.asarray(d, dtype=float))

    def _layout(self) -> None:
        """Once: every transition's requested times in one array each,
        and where each request's values sit in them."""
        H_parts: dict = {}
        h_parts: dict = {}
        where: list = []  # per request: [(kind, h, start, stop), ...]
        size = {("H", h): 0 for h in range(self.frm.size)}
        size.update({("h", h): 0 for h in range(self.frm.size)})
        for what, key, d in self.requests:
            flat = d.ravel()
            hs = self.out_of[key].tolist() if what == "H" else [key]
            parts = H_parts if what == "H" else h_parts
            spots = []
            for h in hs:
                at = size[(what, h)]
                parts.setdefault(h, []).append(flat)
                size[(what, h)] = at + flat.size
                spots.append((what, h, at, at + flat.size))
            where.append(spots)
        self._times = {("H", h): np.concatenate(p) for h, p in H_parts.items()}
        self._times.update(
            {("h", h): np.concatenate(p) for h, p in h_parts.items()}
        )
        self._where = where

    def evaluate(self, models: list, hazard: Any) -> None:
        """Every request's value at ``models``; ``hazard(model, t,
        rate)``."""
        if not hasattr(self, "_where"):
            self._layout()
        got = {
            (what, h): np.ravel(hazard(models[h], t, what == "h"))
            for (what, h), t in self._times.items()
        }
        values = []
        for (_, _, d), spots in zip(self.requests, self._where):
            what, h, lo, hi = spots[0]
            v = got[(what, h)][lo:hi]
            for what, h, lo, hi in spots[1:]:
                v = v + got[(what, h)][lo:hi]
            values.append(v.reshape(d.shape))
        self.values = values
        self.counter = 0


class ResetPanelLikelihood:
    """The negative log-likelihood of panel data on the reset clock
    (progressive models only), each unit entering its first state at its
    first visit."""

    def __init__(self, pdata: Any, hazard: Any) -> None:
        tm = pdata.tm
        self.pdata = pdata
        self.hazard = hazard
        self.frm, self.to = tm.from_index(), tm.to_index()
        self.K = tm.n_states
        self.paths = all_paths(tm)
        self.index = {
            (int(a), int(b)): h
            for h, (a, b) in enumerate(zip(self.frm, self.to))
        }
        absorbing = np.zeros(self.K, dtype=bool)
        absorbing[[tm.index(s) for s in tm.absorbing]] = True
        self.absorbing = absorbing
        unit = pdata.unit
        first = np.r_[True, unit[1:] != unit[:-1]]
        starts = np.flatnonzero(first)
        self.units = list(zip(starts.tolist(), np.r_[starts[1:], unit.size]))

    # Times are carried as distances, so that a node next to an end of
    # its gap keeps its distance from it (a node can be within 1e-18 of
    # the end, closer than a time can be told from it).
    def _path(
        self,
        ev: _Evaluator,
        path: tuple,
        dE: npt.NDArray,
        dU: npt.NDArray,
    ) -> npt.NDArray:
        """The density of following ``path`` from its first state, in it
        for ``dE`` at the start ``lo`` of the gap, to entering its last
        state ``dU`` after ``lo`` (the states between entered in that
        time)."""
        a, c = path[0], path[1]
        h = self.index[(a, c)]
        if len(path) == 2:
            stay = ev.Hout(a, dU + dE) - ev.Hout(a, dE)
            return np.exp(-stay) * ev.rate(h, dU + dE)
        dv = dU[..., None] * _FI
        rest_gap = dU[..., None] * _FRI
        wV = dU[..., None] * _WI
        dE_ = dE[..., None]
        stay = ev.Hout(a, dv + dE_) - ev.Hout(a, dE_)
        first = np.exp(-stay) * ev.rate(h, dv + dE_)
        rest = self._path(ev, path[1:], np.zeros(rest_gap.shape), rest_gap)
        return np.sum(wV * first * rest, axis=-1)

    def _unit(self, ev: _Evaluator, lo: int, hi: int) -> float:
        """The log-likelihood of one unit's visits. Its state is a dict
        ``{state: [(anchor, off, w), ...]}``: entry times ``anchor -
        off`` with weights ``w``."""
        pdata = self.pdata
        x, obs, kind = pdata.x, pdata.obs, pdata.kind
        s0 = int(np.flatnonzero(obs[lo])[0])
        dist: dict = {s0: [(float(x[lo]), np.zeros(1), np.ones(1))]}
        logl = 0.0
        t_prev = float(x[lo])
        for r in range(lo + 1, hi):
            t = float(x[r])
            span = t - t_prev
            allowed = np.flatnonzero(obs[r]).tolist()
            new: dict = {}

            def put(b: int, part: tuple) -> None:
                new.setdefault(b, []).append(part)

            for a, parts in dist.items():
                for anchor, off, w in parts:
                    if self.absorbing[a]:
                        if a in allowed and kind[r] != _ENTRY:
                            put(a, (anchor, off, w))
                        continue
                    dE = (t_prev - anchor) + off  # in a at t_prev
                    stay = np.exp(-(ev.Hout(a, dE + span) - ev.Hout(a, dE)))
                    if a in allowed and kind[r] != _ENTRY:
                        put(a, (anchor, off, w * stay))
                    for b in allowed:
                        if b == a:
                            continue
                        if kind[r] == _EXACT:
                            h = self.index.get((a, b))
                            if h is None:
                                continue
                            dens = stay * ev.rate(h, dE + span)
                            put(b, (t, np.zeros(1), np.array([w @ dens])))
                            continue
                        paths = self.paths.get((a, b), [])
                        if not paths:
                            continue
                        if kind[r] == _ENTRY:
                            dU = np.full((dE.size, 1), span)
                            dens = sum(
                                self._path(ev, p, dE[:, None], dU)
                                for p in paths
                            )
                            put(b, (t, np.zeros(1), np.ravel(w @ dens)))
                            continue
                        dU = np.broadcast_to(span * _F, (dE.size, _F.size))
                        dens = sum(
                            self._path(ev, p, dE[:, None], dU) for p in paths
                        )
                        weights = (w @ dens) * (span * _W)
                        if self.absorbing[b]:
                            put(b, (t, np.zeros(1), np.array([weights.sum()])))
                        else:
                            surv = np.exp(-ev.Hout(b, span * _FR))
                            put(b, (t, span * _FR, weights * surv))
            total = sum(float(np.sum(p[2])) for ps in new.values() for p in ps)
            if ev.values is not None:
                if not (total > 0 and np.isfinite(total)):
                    return -np.inf
                logl += np.log(total)
                new = {
                    b: [(an, of, w / total) for an, of, w in ps]
                    for b, ps in new.items()
                }
            dist = new
            t_prev = t
        return logl

    def neg_ll(self, models: list) -> float:
        ev = getattr(self, "_ev", None)
        if ev is None:
            # (the hazards asked for do not depend on the parameters:
            # recorded once)
            ev = _Evaluator(self.frm, self.K)
            for lo, hi in self.units:
                self._unit(ev, lo, hi)
            self._ev = ev
        with np.errstate(all="ignore"):
            ev.evaluate(models, self.hazard)
            total = 0.0
            for lo, hi in self.units:
                value = self._unit(ev, lo, hi)
                if not np.isfinite(value):
                    return np.inf
                total += value
        return -total
