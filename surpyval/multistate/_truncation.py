r"""Right truncation in the parametric multi-state models: units seen
only because they reached a state by a cut-off.

A unit is in the data only if it reached the state ``reached`` by its
cut-off :math:`\tau` (a register of deaths, say, or of units that failed
by a date). Its likelihood is then the likelihood of its history given
that it got there in time,

.. math::
    \frac{L(\text{history})}{P(\text{reach } r \text{ by } \tau \mid
    \text{in } g_0 \text{ at } t_0)},

:math:`g_0` and :math:`t_0` the state and time of its first spell (its
entry: ``tl``, or the start of the clock). The denominator is the chance
of entering :math:`r` by :math:`\tau`, the first passage: the occupation
probability of :math:`r` at :math:`\tau` from :math:`g_0` at :math:`t_0`
in the model with every transition out of :math:`r` removed (so a unit
that has reached it stays). On the forward clock it is an element of the
product integral :math:`P(t_0, \tau)` of that model, on a grid planned
at the starting values and again at the estimate (as the panel models
do). With two states (one transition into an absorbing :math:`r`) it is
:math:`1 - S(\tau) / S(t_0)`, the univariate right-truncated (and left
truncated) likelihood's denominator.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate import _forward
from surpyval.multistate._data import SpellData

#: The accuracy of each row of the first-passage probabilities.
TOL = 1e-10


def check(data: SpellData, Z: npt.NDArray | None, clock: str) -> None:
    """Refuse what the truncation cannot take: the reset clock, and a
    unit whose covariates differ between its rows; and a unit whose
    history did not reach the state by its cut-off."""
    if data.tr is None:
        return
    tm = data.tm
    assert data.reached is not None
    name = tm.states[int(data.reached)]
    if clock == "reset":
        raise ValueError(
            "Right truncation (tr) is taken on the forward clock only: on "
            "the reset clock the chance of reaching "
            f"{name!r} by the cut-off needs the renewal equations solved "
            "within the likelihood, which is not offered. Use "
            "clock='forward', or leave the truncation out (and say that "
            "the fit ignores it)."
        )
    for u in np.unique(data.cluster[np.isfinite(data.tr)]).tolist():
        rows = np.flatnonzero(data.cluster == u)
        tau = data.tr[rows]
        if np.any(tau != tau[0]):
            raise ValueError(
                f"The rows of unit {u} have different cut-offs tr "
                f"({tau.min():g} to {tau.max():g}): a unit is truncated "
                "once, so give its cut-off on each of its rows."
            )
        if Z is not None and Z.shape[1] and np.any(Z[rows] != Z[rows[0]]):
            raise ValueError(
                f"Unit {u} is right truncated and its covariates differ "
                "between its rows: the chance of reaching "
                f"{name!r} by its cut-off is taken with its covariates "
                "fixed. Give the unit one row of covariates."
            )
        got = np.any(
            ((data.h[rows] == data.reached) & (data.x[rows] <= tau[0]))
            | (
                (data.g[rows] == data.reached)
                & (np.where(np.isfinite(data.tl[rows]), data.tl[rows], 0.0)
                   <= tau[0])
            )
        )  # fmt: skip
        if not got:
            raise ValueError(
                f"Unit {u} is right truncated at {tau[0]:g} (seen only "
                f"because it reached {name!r} by then), but its spells do "
                "not reach that state by then: correct its cut-off or "
                "its spells, or the state given in reached."
            )


class FirstPassage:
    """The truncation term of the log-likelihood: the sum over truncated
    units of ``n log P(reach by tau | first spell)``, on the forward
    clock."""

    def __init__(
        self,
        data: SpellData,
        hazard: Any,
        lows: list,
        scale: float,
        Z: npt.NDArray | None = None,
    ) -> None:
        tm = data.tm
        r = int(data.reached)  # type: ignore[arg-type]
        frm, to = tm.from_index(), tm.to_index()
        self.keep = np.flatnonzero(frm != r)
        self.frm, self.to = frm[self.keep], to[self.keep]
        self.k = tm.n_states
        self.r = r
        self.hazard = hazard
        self.scale = scale
        self.breaks = [float(v) for v in lows if np.isfinite(v)]
        # Each truncated unit: its first spell's state, entry and row, its
        # cut-off and count; grouped by (entry, covariate row).
        groups: dict = {}
        tau = data.tr
        assert tau is not None
        for u in np.unique(data.cluster[np.isfinite(tau)]).tolist():
            rows = np.flatnonzero(data.cluster == u)
            order = np.argsort(data.tl[rows], kind="stable")
            first = int(rows[order[0]])
            g0 = int(data.g[first])
            if g0 == r:
                continue  # already there at entry: the chance is 1
            t0 = float(data.tl[first]) if np.isfinite(data.tl[first]) else 0.0
            z = () if Z is None else tuple(np.ravel(Z[first]).tolist())
            entry = groups.setdefault((t0, z), (first, []))
            entry[1].append((g0, float(tau[first]), float(data.n[first])))
        # (one product integral for the units of one entry time and one
        # row of covariates, at all their cut-offs; ``row`` a row of
        # theirs, whose covariates the hazards take)
        self.groups = [
            (t0, row, np.array([g for g, _, _ in v]),
             np.array([t for _, t, _ in v]), np.array([n for _, _, n in v]))
            for (t0, _), (row, v) in groups.items()
        ]  # fmt: skip
        self.plans: list = []

    def _cumhaz(self, models: list, row: int) -> Any:
        kept = [models[h] for h in self.keep]

        def H(t: npt.NDArray, rate: bool) -> npt.NDArray:
            t = np.asarray(t, dtype=float).ravel()
            rows = np.full(t.size, row, dtype=np.intp)
            out = np.empty((t.size, len(kept)))
            with np.errstate(all="ignore"):
                for j, m in enumerate(kept):
                    out[:, j] = np.ravel(self.hazard(m, t, rows, rate))
            return out

        return H

    def plan(self, models: list) -> list:
        """Plan every group's grid at ``models``; the nodes, to compare
        with the next plan's."""
        self.plans = [
            _forward.plan(
                self._cumhaz(models, row),
                t0,
                np.unique(taus),
                self.scale,
                self.frm,
                self.to,
                self.k,
                tol=TOL,
                breaks=self.breaks,
            )
            for t0, row, _, taus, _ in self.groups
        ]
        return [p.nodes for p in self.plans]

    def log_chances(self, models: list) -> npt.NDArray:
        """``log P(reach by tau)`` of every truncated unit not already
        there at entry, in the order of the groups."""
        out = []
        for (t0, row, g0, taus, _), grid in zip(self.groups, self.plans):
            uniq, inverse = np.unique(taus, return_inverse=True)
            P = _forward.evaluate(
                self._cumhaz(models, row), grid, self.frm, self.to, self.k
            )
            p = P[np.ravel(inverse), g0, self.r]
            with np.errstate(divide="ignore"):
                out.append(np.log(np.where(p > 0, p, 0.0)))
        return np.concatenate(out) if out else np.zeros(0)

    def term(self, models: list) -> float:
        """``sum n log P``: added to the negative log-likelihood."""
        if not self.groups:
            return 0.0
        if not self.plans:
            self.plan(models)
        logp = self.log_chances(models)
        if not np.all(np.isfinite(logp)):
            return np.inf
        n = np.concatenate([n for *_, n in self.groups])
        return float(np.sum(n * logp))


class TruncatedLikelihood:
    """The likelihood of right-truncated spells: that of the spells
    (``base``, an ``IntervalLikelihood``, which also takes moves of
    unknown time) divided by each truncated unit's chance of reaching the
    state by its cut-off."""

    def __init__(self, base: Any, passage: FirstPassage) -> None:
        self.base = base
        self.passage = passage

    def plan(self, models: list) -> list:
        return self.passage.plan(models)

    def neg_ll(self, models: list) -> float:
        value = self.base.neg_ll(models)
        if not np.isfinite(value):
            return np.inf
        return float(value + self.passage.term(models))
