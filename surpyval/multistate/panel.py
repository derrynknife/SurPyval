r"""Multi-state Markov models for panel data: states seen only at visits.

Units move between states in continuous time, but each unit's state is
known only at the times it was looked at -- an inspection, a clinic
visit, a survey wave -- so the time of each move is known only to lie
between two visits, and a unit may pass through states between visits
that are never seen. The model is a time-homogeneous Markov chain: a
transition intensity :math:`q_{rs}` for each allowed move, so that the
probability of being in state :math:`s` a time :math:`t` after being in
:math:`r` is the :math:`(r, s)` element of the matrix exponential
:math:`P(t) = \exp(Q t)`. It is fitted by maximum likelihood (Kalbfleisch
and Lawless 1985; Jackson 2011, R's ``msm``).

Each unit's likelihood is the chance of the states it was seen in, given
the first: a product of elements of :math:`P` over the gaps between its
visits, computed by the forward recursion so that a visit that saw only
a set of states (censoring) is summed over. A move into an absorbing
state may also be timed exactly (``exact_entry``: the unit was in some
other state just before), or a row may record a move at the time it
happened (``obstype="exact"``).

Covariates act on the log intensities (a proportional-intensities model,
each transition with its own coefficients or one ``shared`` by all), and
intensities may change at given times (piecewise constant, ``pci``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate._base import check_bound_args, probability_bounds
from surpyval.multistate.transitions import TransitionMatrix
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

#: The ways a row can be observed: a snapshot of the state (``"panel"``),
#: a move timed exactly with the earlier state held until it
#: (``"exact"``), and an exactly timed entry into an absorbing state from
#: an unknown state (``"exact_entry"``), msm's ``obstype`` 1, 2 and 3.
OBSTYPES = ("panel", "exact", "exact_entry")
_PANEL, _EXACT, _ENTRY = 0, 1, 2

# A log intensity (in units of the mean gap between visits) below this,
# or a standardised coefficient beyond _RUNAWAY_BETA, has run off: the
# likelihood has no finite maximum in that direction.
_RUNAWAY_LOG_Q = -18.0
_RUNAWAY_BETA = 25.0
# Rows of the batched matrix exponentials computed at once.
_CHUNK = 4096


# ---------------------------------------------------------------------------
# The data
# ---------------------------------------------------------------------------
@dataclass
class _PanelData:
    """Checked panel data, sorted by unit and time."""

    x: npt.NDArray  # times
    unit: npt.NDArray  # unit codes 0..n_units-1
    ids: list  # each unit's label, by code
    ids_given: bool
    labels: list  # the observed state labels (or censoring codes)
    obs: npt.NDArray  # (N, K) bool: the states each row allows
    kind: npt.NDArray  # (N,) 0 panel, 1 exact, 2 exact entry
    Z: npt.NDArray  # (N, p) raw covariates
    tm: TransitionMatrix
    censored: dict  # code -> tuple of states
    exact_entry: tuple
    cuts: tuple  # pci cut times (raw)

    @property
    def size(self) -> int:
        return int(self.x.size)

    def intervals(self) -> npt.NDArray:
        """The rows that end a gap between two visits of one unit."""
        same = np.r_[False, self.unit[1:] == self.unit[:-1]]
        return np.flatnonzero(same)

    def first_rows(self) -> npt.NDArray:
        return np.flatnonzero(np.r_[True, self.unit[1:] != self.unit[:-1]])


def _labels(values: Any, name: str, size: int | None = None) -> npt.NDArray:
    from surpyval.utils.data_formats import missing_events

    if values is None:
        raise ValueError(f"'{name}' is needed.")
    if isinstance(values, (str, bytes)) or np.ndim(values) == 0:
        raise ValueError(f"'{name}' must have one value per row.")
    arr = np.empty(len(values), dtype=object)
    arr[:] = [v.item() if isinstance(v, np.generic) else v for v in values]
    if size is not None and arr.size != size:
        raise ValueError(
            f"'{name}' has {arr.size} values but 'x' has {size}: give one "
            "per row."
        )
    if np.any(missing_events(arr)):
        k = int(np.flatnonzero(missing_events(arr))[0])
        raise ValueError(
            f"'{name}' has missing values (the first is row {k}): every "
            "row needs one. A visit whose state is not fully known is "
            "given a code listed in 'censored'."
        )
    return arr


def _transition_matrix(transitions: Any) -> TransitionMatrix:
    if transitions is None:
        raise ValueError(
            "'transitions' is needed: with states seen only at visits the "
            "moves between them cannot be read off the data (a unit can "
            "pass through states unseen). Give a TransitionMatrix or a "
            "list of (from, to) pairs, such as "
            "TransitionMatrix.illness_death(recovery=True)."
        )
    if isinstance(transitions, TransitionMatrix):
        return transitions
    return TransitionMatrix(transitions)


def _check_censored(censored: Any, tm: TransitionMatrix) -> dict:
    if censored is None:
        return {}
    if not isinstance(censored, dict):
        raise ValueError(
            "'censored' must be a dict {code: states}, each code a value "
            "of 's' meaning the unit was in one of those states (for "
            "example {'alive': ['healthy', 'ill']}); got "
            f"{type(censored).__name__}."
        )
    out = {}
    for code, states in censored.items():
        if code in tm._index:
            raise ValueError(
                f"The censoring code {code!r} is also a state: use a code "
                "that is not one of the states "
                f"{list(tm.states)!r}."
            )
        if isinstance(states, (str, bytes)) or np.ndim(states) == 0:
            states = [states]
        states = tuple(states)
        if not states:
            raise ValueError(
                f"The censoring code {code!r} lists no states: give the "
                "states the unit may have been in."
            )
        for st in states:
            tm.index(st)
        out[code] = states
    return out


def _check_exact_entry(exact_entry: Any, tm: TransitionMatrix) -> tuple:
    if exact_entry is None:
        return ()
    if isinstance(exact_entry, (str, bytes)) or np.ndim(exact_entry) == 0:
        exact_entry = [exact_entry]
    out = []
    for st in exact_entry:
        tm.index(st)
        if st not in tm.absorbing:
            raise ValueError(
                f"'exact_entry' names {st!r}, which is not an absorbing "
                "state: an exactly timed entry from an unknown state is "
                "for states a unit never leaves (the absorbing states are "
                f"{list(tm.absorbing)!r}). Use obstype='exact' for a move "
                "seen when it happened."
            )
        out.append(st)
    return tuple(out)


def _check_cuts(pci: Any) -> tuple:
    if pci is None:
        return ()
    cuts = np.atleast_1d(np.asarray(pci, dtype=float))
    if cuts.ndim != 1 or cuts.size == 0 or not np.all(np.isfinite(cuts)):
        raise ValueError(
            "'pci' must be a list of finite times where the intensities "
            f"change; got {pci!r}."
        )
    if np.any(np.diff(cuts) <= 0):
        raise ValueError(
            f"'pci' must be in increasing order with no repeats; got "
            f"{list(cuts)!r}."
        )
    return tuple(float(c) for c in cuts)


def _check_panel(
    x: Any,
    s: Any,
    i: Any,
    Z: Any,
    obstype: Any,
    transitions: Any,
    censored: Any,
    exact_entry: Any,
    pci: Any,
) -> _PanelData:
    import pandas as pd

    tm = _transition_matrix(transitions)
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            "'x' must be one time per row (1-D): the time each state was "
            "seen."
        )
    size = x.size
    if size == 0:
        raise ValueError("No data: 'x' is empty.")
    if np.any(~np.isfinite(x)):
        k = int(np.flatnonzero(~np.isfinite(x))[0])
        raise ValueError(
            f"'x' has a missing or infinite time (the first is row {k}): "
            "every row is a visit at a known time."
        )
    s_arr = _labels(s, "s", size)
    ids = _labels(i, "i", size)
    cens = _check_censored(censored, tm)
    entry = _check_exact_entry(exact_entry, tm)
    cuts = _check_cuts(pci)

    K = tm.n_states
    obs = np.zeros((size, K), dtype=bool)
    unknown = []
    for r, lab in enumerate(s_arr):
        try:
            is_state = lab in tm._index
        except TypeError:
            is_state = False
        if is_state:
            obs[r, tm._index[lab]] = True
        elif lab in cens:
            for st in cens[lab]:
                obs[r, tm._index[st]] = True
        elif lab not in unknown:
            unknown.append(lab)
    if unknown:
        raise ValueError(
            f"'s' holds {unknown[:5]!r}, which are neither states (the "
            f"states are {list(tm.states)!r}) nor codes in 'censored'."
        )

    if obstype is None:
        kind = np.zeros(size, dtype=np.intp)
    else:
        ot = _labels(obstype, "obstype", size)
        bad = sorted({str(v) for v in ot if v not in OBSTYPES})
        if bad:
            raise ValueError(
                f"'obstype' holds {bad!r}; each row's obstype is one of "
                f"{list(OBSTYPES)!r}."
            )
        kind = np.array([OBSTYPES.index(v) for v in ot], dtype=np.intp)
    single = obs.sum(axis=1) == 1
    if entry:
        cols = [tm._index[st] for st in entry]
        hit = single & obs[:, cols].any(axis=1)
        kind = np.where(hit & (kind == _PANEL), _ENTRY, kind)
    absorbing = np.zeros(K, dtype=bool)
    absorbing[[tm._index[st] for st in tm.absorbing]] = True
    wrong = (kind == _ENTRY) & ~(single & (obs & absorbing).any(axis=1))
    if np.any(wrong):
        k = int(np.flatnonzero(wrong)[0])
        raise ValueError(
            f"Row {k} is an 'exact_entry' observation of {s_arr[k]!r}, "
            "which is not an absorbing state: an exactly timed entry from "
            "an unknown state is for a state the unit never leaves."
        )
    wrong = (kind == _EXACT) & ~single
    if np.any(wrong):
        k = int(np.flatnonzero(wrong)[0])
        raise ValueError(
            f"Row {k} is an 'exact' observation of the censoring code "
            f"{s_arr[k]!r}: a move seen when it happened is into one "
            "known state."
        )

    if Z is None:
        Zarr = np.zeros((size, 0))
    else:
        Zarr = np.asarray(Z, dtype=float)
        if Zarr.ndim == 1:
            Zarr = Zarr.reshape(-1, 1)
        if Zarr.ndim != 2 or Zarr.shape[0] != size:
            raise ValueError(
                f"Z must have one row of covariates per row ({size}); got "
                f"shape {np.shape(Z)}."
            )
        if np.any(~np.isfinite(Zarr)):
            rows = np.flatnonzero(~np.isfinite(Zarr).all(axis=1))
            raise ValueError(
                f"Z has missing or infinite covariates on {rows.size} rows "
                f"(the first is row {int(rows[0])}). A row cannot be "
                "dropped without changing the gaps between a unit's "
                "visits: impute them, or leave out the units they belong "
                "to."
            )

    codes, uniques = pd.factorize(pd.Series(ids, dtype=object))
    unit = np.asarray(codes, dtype=np.intp)
    order = np.lexsort((x, unit))
    x, unit = x[order], unit[order]
    obs, kind, Zarr = obs[order], kind[order], Zarr[order]
    single = single[order]
    s_sorted = s_arr[order]
    dup = (unit[1:] == unit[:-1]) & (x[1:] == x[:-1])
    if np.any(dup):
        k = int(np.flatnonzero(dup)[0])
        raise ValueError(
            f"Unit {uniques[unit[k]]!r} has two rows at time {x[k]:g}: "
            "give each visit once (one state per unit per time)."
        )
    first = np.r_[True, unit[1:] != unit[:-1]]
    bad_first = first & ~single
    if np.any(bad_first):
        k = int(np.flatnonzero(bad_first)[0])
        raise ValueError(
            f"Unit {uniques[unit[k]]!r} starts with the censoring code "
            f"{s_sorted[k]!r}: the model is conditional on each unit's "
            "first state, so the first visit must see one state."
        )
    kind = np.where(first, _PANEL, kind)
    return _PanelData(
        x=x,
        unit=unit,
        ids=list(uniques),
        ids_given=True,
        labels=list(s_sorted),
        obs=obs,
        kind=kind,
        Z=Zarr,
        tm=tm,
        censored=cens,
        exact_entry=entry,
        cuts=cuts,
    )


# ---------------------------------------------------------------------------
# The parameters: log intensities, covariate effects and period effects
# ---------------------------------------------------------------------------
class _Layout:
    """Where each parameter sits in the search vector ``theta``: the log
    intensities (one per transition), then each kept covariate's
    coefficients (one per transition, or one if shared), then each later
    period's log ratios (one per transition). ``design`` gives the
    derivatives of the log intensities by ``theta``, which are also their
    coefficients: ``log q = D @ theta``."""

    def __init__(
        self, n_trans: int, shared: tuple, n_cov: int, n_periods: int
    ):
        self.H = n_trans
        self.n_cov = n_cov
        self.shared = tuple(bool(v) for v in shared)
        self.n_periods = n_periods
        cov_index = []
        k = n_trans
        for c in range(n_cov):
            if self.shared[c]:
                cov_index.append(np.full(n_trans, k))
                k += 1
            else:
                cov_index.append(np.arange(k, k + n_trans))
                k += n_trans
        self.cov_index = cov_index
        self.period_index = []
        for _ in range(1, n_periods):
            self.period_index.append(np.arange(k, k + n_trans))
            k += n_trans
        self.P = k

    def design(self, U: npt.NDArray, period: npt.NDArray) -> npt.NDArray:
        """``(n, H, P)``: d log q_h / d theta_p for each row."""
        n = U.shape[0]
        H = self.H
        D = np.zeros((n, H, self.P))
        hh = np.arange(H)
        D[:, hh, hh] = 1.0
        for c in range(self.n_cov):
            idx = self.cov_index[c]
            D[:, hh, idx] += U[:, c : c + 1]
        for k in range(1, self.n_periods):
            idx = self.period_index[k - 1]
            on = (period == k).astype(float)[:, None]
            D[:, hh, idx] += on
        return D


def _Q(q: npt.NDArray, fr: npt.NDArray, to: npt.NDArray, K: int) -> Any:
    n = q.shape[0]
    Q = np.zeros((n, K, K))
    for h, (a, b) in enumerate(zip(fr, to)):
        Q[:, a, b] += q[:, h]
        Q[:, a, a] -= q[:, h]
    return Q


def _expm(A: npt.NDArray) -> npt.NDArray:
    from scipy.linalg import expm

    if A.shape[0] <= _CHUNK:
        return expm(A)
    return np.concatenate(
        [expm(A[k : k + _CHUNK]) for k in range(0, A.shape[0], _CHUNK)]
    )


def _by_theta(dX: npt.NDArray, D: npt.NDArray) -> npt.NDArray:
    """``(n, H, K, K)`` derivatives by each log intensity to ``(n, P, K,
    K)`` derivatives by theta (``D`` is ``(n, H, P)``)."""
    n, H, K, _ = dX.shape
    out = np.matmul(D.transpose(0, 2, 1), dX.reshape(n, H, K * K))
    return out.reshape(n, D.shape[2], K, K)


def _pieces(t0: npt.NDArray, t1: npt.NDArray, cuts: npt.NDArray) -> tuple:
    """Split each gap (t0, t1] at the cut times: (owner, position, dt,
    period) of each piece."""
    n = t0.size
    if cuts.size == 0:
        return (
            np.arange(n),
            np.zeros(n, dtype=np.intp),
            t1 - t0,
            np.zeros(n, dtype=np.intp),
        )
    k0 = np.searchsorted(cuts, t0, side="right")
    k1 = np.searchsorted(cuts, t1, side="left")
    count = np.maximum(k1 - k0, 0) + 1
    owner = np.repeat(np.arange(n), count)
    start = np.cumsum(count) - count
    pos = np.arange(owner.size) - start[owner]
    period = k0[owner] + pos
    lo = np.where(pos == 0, t0[owner], cuts[np.clip(period - 1, 0, None)])
    last = pos == count[owner] - 1
    hi = np.where(last, t1[owner], cuts[np.clip(period, 0, cuts.size - 1)])
    return owner, pos, hi - lo, period


class _Engine:
    """The matrices of a model with a given layout and transitions: the
    transition probabilities over gaps, with their derivatives by
    ``theta``."""

    def __init__(
        self,
        tm: TransitionMatrix,
        layout: _Layout,
        cuts: Any,
        rate_unit: float = 1.0,
    ):
        # the intensities' factor from the search's time unit to the
        # data's, for the densities of exactly timed moves
        self.rate_unit = float(rate_unit)
        self.K = tm.n_states
        self.fr = tm.from_index()
        self.to = tm.to_index()
        self.layout = layout
        self.cuts = np.asarray(cuts, dtype=float)
        H, K = layout.H, self.K
        E = np.zeros((H, K, K))
        E[np.arange(H), self.fr, self.to] = 1.0
        E[np.arange(H), self.fr, self.fr] = -1.0
        self.E = E

    def _q(self, theta: npt.NDArray, D: npt.NDArray) -> npt.NDArray:
        return np.exp(np.einsum("nhp,p->nh", D, theta))

    def _panel(
        self, q: npt.NDArray, dt: npt.NDArray, D: npt.NDArray, deriv: bool
    ) -> tuple:
        K, H = self.K, self.layout.H
        Q = _Q(q, self.fr, self.to, K) * dt[:, None, None]
        if not deriv:
            return _expm(Q), None
        n = q.shape[0]
        A = np.zeros((n, H, 2 * K, 2 * K))
        A[:, :, :K, :K] = Q[:, None]
        A[:, :, K:, K:] = Q[:, None]
        A[:, :, :K, K:] = (q * dt[:, None])[:, :, None, None] * self.E[None]
        B = _expm(A.reshape(n * H, 2 * K, 2 * K)).reshape(n, H, 2 * K, 2 * K)
        P = B[:, 0, :K, :K]
        dP = _by_theta(np.ascontiguousarray(B[:, :, :K, K:]), D)
        return P, dP

    def _stay(
        self, q: npt.NDArray, dt: npt.NDArray, D: npt.NDArray, deriv: bool
    ) -> tuple:
        K = self.K
        n = q.shape[0]
        out_rate = np.zeros((n, K))
        for h, a in enumerate(self.fr):
            out_rate[:, a] += q[:, h]
        stay = np.exp(-out_rate * dt[:, None])
        S = np.zeros((n, K, K))
        S[:, np.arange(K), np.arange(K)] = stay
        if not deriv:
            return S, None
        # d stay_a / d log q_h = -stay_a dt q_h for h leaving a
        dq = -(stay[:, self.fr] * dt[:, None] * q)  # (n, H)
        dS_h = np.zeros((n, self.layout.H, K, K))
        dS_h[:, np.arange(self.layout.H), self.fr, self.fr] = dq
        return S, _by_theta(dS_h, D)

    def _final(
        self, q: npt.NDArray, D: npt.NDArray, kind: npt.NDArray, deriv: bool
    ) -> tuple:
        """The factor of the time a move was seen: ``I + Qoff`` for an
        exact move (or a stay), ``Qoff`` for an exact entry."""
        K, H = self.K, self.layout.H
        n = q.shape[0]
        q = q * self.rate_unit
        F = np.zeros((n, K, K))
        F[:, self.fr, self.to] = q
        F[kind == _EXACT] += np.eye(K)
        if not deriv:
            return F, None
        dF_h = np.zeros((n, H, K, K))
        dF_h[:, np.arange(H), self.fr, self.to] = q
        return F, _by_theta(dF_h, D)

    def gaps(
        self,
        theta: npt.NDArray,
        U: npt.NDArray,
        t0: npt.NDArray,
        t1: npt.NDArray,
        kind: npt.NDArray | None = None,
        deriv: bool = True,
    ) -> tuple:
        """``M`` (n, K, K), the matrix of each gap (t0, t1] for a unit
        with standardised covariates ``U``, and ``dM`` (n, P, K, K) its
        derivatives by ``theta`` (None without ``deriv``)."""
        K, P = self.K, self.layout.P
        n = t0.size
        if kind is None:
            kind = np.zeros(n, dtype=np.intp)
        owner, pos, dt, period = _pieces(t0, t1, self.cuts)
        D = self.layout.design(U[owner], period)
        q = self._q(theta, D)
        stay = kind[owner] == _EXACT
        Mp = np.empty((owner.size, K, K))
        dMp: Any = np.empty((owner.size, P, K, K)) if deriv else None
        for sel, f in ((~stay, self._panel), (stay, self._stay)):
            if np.any(sel):
                m, dm = f(q[sel], dt[sel], D[sel], deriv)
                Mp[sel] = m
                if deriv:
                    dMp[sel] = dm
        M = np.broadcast_to(np.eye(K), (n, K, K)).copy()
        dM: Any = np.zeros((n, P, K, K)) if deriv else None
        for m in range(int(pos.max()) + 1 if pos.size else 0):
            sel = np.flatnonzero(pos == m)
            o = owner[sel]
            if m == 0:
                M[o] = Mp[sel]
                if deriv:
                    dM[o] = dMp[sel]
                continue
            if deriv:
                dM[o] = np.matmul(dM[o], Mp[sel][:, None]) + np.matmul(
                    M[o][:, None], dMp[sel]
                )
            M[o] = np.matmul(M[o], Mp[sel])
        timed = kind != _PANEL
        if np.any(timed):
            idx = np.flatnonzero(timed)
            last = np.flatnonzero(
                np.r_[owner[1:] != owner[:-1], True]
            )  # the last piece of each gap
            last = last[np.isin(owner[last], idx)]
            o = owner[last]
            F, dF = self._final(q[last], D[last], kind[o], deriv)
            if deriv:
                dM[o] = np.matmul(dM[o], F[:, None]) + np.matmul(
                    M[o][:, None], dF
                )
            M[o] = np.matmul(M[o], F)
        return M, dM


class _Likelihood:
    """The log-likelihood of panel data and its gradient, by the forward
    recursion over each unit's visits."""

    def __init__(
        self, data: _PanelData, engine: _Engine, U: npt.NDArray, scale: float
    ):
        rows = data.intervals()
        self.engine = engine
        self.t0 = data.x[rows - 1] / scale
        self.t1 = data.x[rows] / scale
        self.U = U[rows - 1]
        self.kind = data.kind[rows]
        self.emit = data.obs[rows].astype(float)
        unit = data.unit[rows]
        self.n_units = int(data.unit.max()) + 1
        firsts = data.first_rows()
        self.alpha0 = data.obs[firsts].astype(float)
        # the position of each gap in its unit's sequence
        start = np.searchsorted(rows, firsts, side="left")
        counts = np.diff(np.r_[start, rows.size])
        pos = np.arange(rows.size) - np.repeat(start, counts)
        self.layers = [
            (np.flatnonzero(pos == j), unit[pos == j])
            for j in range(int(pos.max()) + 1 if pos.size else 0)
        ]
        self.n_obs = int(rows.size)
        self._cache: dict = {}
        self._hess: dict = {}
        self._scores: tuple = (None, None)

    def unit_loglik(self, theta: npt.NDArray) -> npt.NDArray:
        M, _ = self.engine.gaps(
            theta, self.U, self.t0, self.t1, self.kind, deriv=False
        )
        alpha = self.alpha0.copy()
        ll = np.zeros(self.n_units)
        with np.errstate(divide="ignore", invalid="ignore"):
            for idx, u in self.layers:
                v = np.matmul(alpha[u][:, None], M[idx])[:, 0] * self.emit[idx]
                c = v.sum(axis=1)
                ll[u] += np.log(c)
                alpha[u] = v / c[:, None]
        return ll

    def __call__(self, theta: npt.NDArray, deriv: bool = True) -> tuple:
        theta = np.asarray(theta, dtype=float)
        key = (theta.tobytes(), deriv)
        if key in self._cache:
            return self._cache[key]
        P = self.engine.layout.P
        with np.errstate(all="ignore"):
            M, dM = self.engine.gaps(
                theta, self.U, self.t0, self.t1, self.kind, deriv=deriv
            )
            alpha = self.alpha0.copy()
            dalpha = np.zeros((self.n_units, P, alpha.shape[1]))
            ll = 0.0
            g = np.zeros(P)
            scores: Any = np.zeros((self.n_units, P)) if deriv else None
            for idx, u in self.layers:
                e = self.emit[idx]
                v = np.matmul(alpha[u][:, None], M[idx])[:, 0] * e
                c = v.sum(axis=1)
                ll += float(np.sum(np.log(c)))
                if deriv:
                    a = alpha[u]
                    dv = (
                        np.matmul(dalpha[u], M[idx])
                        + np.matmul(a[:, None, None, :], dM[idx])[:, :, 0]
                    ) * e[:, None, :]
                    dc = dv.sum(axis=2)
                    scores[u] += dc / c[:, None]
                    a_new = v / c[:, None]
                    dalpha[u] = (
                        dv / c[:, None, None]
                        - a_new[:, None, :] * (dc / c[:, None])[:, :, None]
                    )
                    alpha[u] = a_new
                else:
                    alpha[u] = v / c[:, None]
        if deriv:
            g = scores.sum(axis=0)
            self._scores = (theta.tobytes(), scores)
        out = (-ll if np.isfinite(ll) else np.inf, -g if deriv else None)
        if len(self._cache) > 8:
            self._cache.clear()
        self._cache[key] = out
        return out

    def fun(self, theta: npt.NDArray) -> float:
        return self(theta, deriv=False)[0]

    def scores(self, theta: npt.NDArray) -> npt.NDArray:
        """``(units, P)``: each unit's score (gradient of its
        log-likelihood)."""
        theta = np.asarray(theta, dtype=float)
        if self._scores[0] != theta.tobytes():
            self._cache.pop((theta.tobytes(), True), None)
            self(theta, deriv=True)
        return self._scores[1]

    def jac(self, theta: npt.NDArray) -> npt.NDArray:
        return self(theta, deriv=True)[1]

    def hess(self, theta: npt.NDArray) -> npt.NDArray:
        """Central differences of the exact gradient."""
        theta = np.asarray(theta, dtype=float)
        key = theta.tobytes()
        if key not in self._hess:
            self._hess = {key: self._differences(theta)}
        return self._hess[key]

    def _differences(self, theta: npt.NDArray) -> npt.NDArray:
        P = theta.size
        h = 1e-5 * np.maximum(np.abs(theta), 1.0)
        H = np.empty((P, P))
        for j in range(P):
            e = np.zeros(P)
            e[j] = h[j]
            H[:, j] = (self.jac(theta + e) - self.jac(theta - e)) / (2 * h[j])
        return 0.5 * (H + H.T)


def _crude_start(data: _PanelData, scale: float) -> npt.NDArray:
    """msm's crude start: each allowed move seen between two visits, over
    the time spent in its 'from' state (taken as the state at the start
    of each gap); half a move where none was seen."""
    tm = data.tm
    rows = data.intervals()
    single = data.obs.sum(axis=1) == 1
    a = np.argmax(data.obs[rows - 1], axis=1)
    b = np.argmax(data.obs[rows], axis=1)
    ok = single[rows - 1] & single[rows]
    dt = (data.x[rows] - data.x[rows - 1]) / scale
    K = tm.n_states
    time_in = np.bincount(a[single[rows - 1]], dt[single[rows - 1]], K)
    total = float(np.sum(dt)) or 1.0
    out = np.empty(tm.n_transitions)
    for h, (fa, fb) in enumerate(zip(tm.from_index(), tm.to_index())):
        n = float(np.sum(ok & (a == fa) & (b == fb)))
        t = time_in[fa] if time_in[fa] > 0 else total
        out[h] = max(n, 0.5) / t
    return np.log(out)


# ---------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------
class _PanelMultiStateBase(SerialisableMixin):
    """What the panel-data multi-state models share."""

    _regression = False

    #: The states and the allowed transitions.
    transitions: TransitionMatrix
    #: The covariates' names (none for :class:`PanelMultiState`).
    covariate_names: list
    #: For each covariate, whether one coefficient serves every
    #: transition.
    shared: tuple
    #: The times at which the intensities change (``pci``), if any.
    pci: tuple
    #: The absorbing states whose entry times are exact.
    exact_entry: tuple
    #: The codes of ``s`` that stand for a set of states.
    censored: dict
    #: The earliest visit: the default start of a prediction.
    start: float
    #: The last visit.
    last: float
    #: What the fit reached (see ``surpyval.utils.no_maximum``).
    maximum: str
    #: The number of gaps between visits (the likelihood's factors).
    n_obs: int

    _theta: npt.NDArray
    _cov_theta: npt.NDArray | None
    _neg_ll: float
    _scale: float
    _mean: npt.NDArray
    _sd: npt.NDArray
    _kept: npt.NDArray
    _p0: npt.NDArray
    _data: _PanelData

    # -- fitting ------------------------------------------------------------

    @classmethod
    def _fit(
        cls,
        data: _PanelData,
        covariate_names: list,
        shared: Any,
    ) -> Any:
        from surpyval.univariate.parametric.fitters import (
            is_local_minimum,
            preconditioned_bfgs,
        )
        from surpyval.univariate.regression._aliasing import (
            aliased_columns,
            covariate_columns,
            warn_aliased,
        )
        from surpyval.utils.warnings import ignoring_warnings

        rows = data.intervals()
        if rows.size == 0:
            raise ValueError(
                "No unit has two visits: the model learns from the change "
                "(or not) of state between a unit's visits, so each unit "
                "needs at least two rows."
            )
        p = data.Z.shape[1]
        shared_cols = _shared_columns(shared, covariate_names)
        Zi = data.Z[rows - 1]
        mean = Zi.mean(axis=0) if p else np.zeros(0)
        centred = Zi - mean
        constant = np.ptp(Zi, axis=0) == 0 if p else np.zeros(0, bool)
        aliased = (
            aliased_columns(centred.T @ centred, rows.size, constant)
            if p
            else np.zeros(0, dtype=int)
        )
        kept = np.setdiff1d(np.arange(p), aliased)
        if aliased.size:
            named = covariate_names != [f"z{j}" for j in range(p)]
            with covariate_columns(covariate_names if named else None):
                warn_aliased(
                    aliased,
                    "constant, or a combination of the other columns, over "
                    "the gaps between visits",
                )
        sd = Zi[:, kept].std(axis=0) if kept.size else np.zeros(0)
        sd = np.where(sd > 0, sd, 1.0)
        scale = float(np.mean(data.x[rows] - data.x[rows - 1]))

        out = cls()
        out.transitions = data.tm
        out.covariate_names = list(covariate_names)
        out.shared = tuple(j in shared_cols for j in range(p))
        out.pci = data.cuts
        out.exact_entry = data.exact_entry
        out.censored = dict(data.censored)
        out._data = data
        out._scale = scale
        out._mean = mean[kept]
        out._sd = sd
        out._kept = kept
        out._setup()
        lik = out._likelihood()

        theta0 = np.zeros(out._layout.P)
        theta0[: data.tm.n_transitions] = _crude_start(data, scale)
        bad = ~np.isfinite(lik.unit_loglik(theta0))
        if np.any(bad):
            units = [data.ids[k] for k in np.flatnonzero(bad)[:5]]
            raise ValueError(
                f"{int(bad.sum())} unit(s) have visits the allowed "
                f"transitions cannot explain (the first: {units!r}): a "
                "later state cannot be reached from an earlier one, or an "
                "exactly timed move is not an allowed transition. Correct "
                "the data, or allow the moves in 'transitions'."
            )
        n_obs = lik.n_obs
        with np.errstate(all="ignore"), ignoring_warnings():
            theta, converged = _scoring(lik, theta0)
            if not converged:
                res = preconditioned_bfgs(
                    lik.fun, theta, jac=lik.jac, floor=1.0, obj_scale=n_obs
                )
                if lik.fun(res.x) <= lik.fun(theta):
                    theta = np.asarray(res.x, dtype=float)
            if not is_local_minimum(
                lik.fun, lik.jac, lik.hess, theta, obj_scale=n_obs
            ):
                theta = _newton_polish(lik, theta)
        H = out._layout.H
        maximum = "verified"
        # the log intensities over every gap's covariates and periods (in
        # units of the mean gap): one that runs to 0, or to infinity, for
        # some of the units has no finite maximum
        owner, _, _, period = _pieces(lik.t0, lik.t1, lik.engine.cuts)
        logq = lik.engine.layout.design(lik.U[owner], period) @ theta
        names = data.tm.transition_names()
        runaway = [
            names[h]
            for h in range(H)
            if np.min(logq[:, h]) < _RUNAWAY_LOG_Q
            or np.max(logq[:, h]) > -_RUNAWAY_LOG_Q
        ]
        beta_runaway = np.abs(theta[H:]) > _RUNAWAY_BETA
        if runaway or np.any(beta_runaway):
            maximum = "no finite maximum"
        else:
            with np.errstate(all="ignore"), ignoring_warnings():
                ok = is_local_minimum(
                    lik.fun, lik.jac, lik.hess, theta, obj_scale=n_obs
                )
            if not ok:
                maximum = "unverified"
        if maximum == "no finite maximum":
            what = (
                f"the intensity of {', '.join(runaway)} runs to 0 or to "
                "infinity (for every unit, or for some covariate values or "
                "periods)"
                if runaway
                else "a covariate or period effect runs off to infinity"
            )
            warn_no_maximum(
                f"{what}: the data support no finite rate of that move",
                "The estimates of those parameters, their standard errors "
                "and bounds are meaningless",
                "leave the transition out of 'transitions', or the "
                "covariate or period out of the model",
            )
        elif maximum == "unverified":
            warn_unverified("The panel multi-state fit")
        out.maximum = maximum
        out.n_obs = n_obs
        out._theta = theta
        out._neg_ll = float(lik.fun(theta))
        out._cov_theta = _inverse_information(lik.hess(theta))
        return out

    def _setup(self) -> None:
        d = self._data
        tm = self.transitions
        self._layout = _Layout(
            tm.n_transitions,
            tuple(self.shared[j] for j in self._kept),
            int(self._kept.size),
            len(self.pci) + 1,
        )
        self._engine = _Engine(
            tm,
            self._layout,
            np.asarray(self.pci) / self._scale,
            1.0 / self._scale,
        )
        firsts = d.first_rows()
        counts = d.obs[firsts].sum(axis=0).astype(float)
        self._p0 = counts / counts.sum()
        self.start = float(np.min(d.x[firsts]))
        self.last = float(np.max(d.x))

    def _standardise(self, Z: npt.NDArray) -> npt.NDArray:
        return (Z[:, self._kept] - self._mean) / self._sd

    def _likelihood(self) -> _Likelihood:
        return _Likelihood(
            self._data,
            self._engine,
            self._standardise(self._data.Z),
            self._scale,
        )

    # -- parameters ---------------------------------------------------------

    @property
    def states(self) -> tuple:
        """The states, in order."""
        return self.transitions.states

    def _reported_map(self) -> tuple[npt.NDArray, npt.NDArray]:
        """``(A, b)``: the reported parameters on their log / coefficient
        scale are ``A @ theta + b`` (log intensities at ``Z = 0`` per unit
        time, coefficients per unit of each covariate, period log
        ratios)."""
        L = self._layout
        H = L.H
        A = np.eye(L.P)
        b = np.zeros(L.P)
        b[:H] = -np.log(self._scale)
        for c in range(L.n_cov):
            idx = L.cov_index[c]
            for h in range(H):
                A[h, idx[h]] -= self._mean[c] / self._sd[c]
            for k in np.unique(idx):
                A[k, k] = 1.0 / self._sd[c]
        return A, b

    def _phi(self, theta: npt.NDArray | None = None) -> npt.NDArray:
        A, b = self._reported_map()
        return A @ (self._theta if theta is None else theta) + b

    @property
    def params(self) -> npt.NDArray:
        """The fitted parameters, named by :attr:`parameter_names`: each
        transition's intensity (at ``Z = 0`` and in the first period,
        moves per unit of time), then the covariates' coefficients (log
        intensity ratios per unit of each covariate), then each later
        period's log intensity ratios."""
        phi = self._phi()
        H = self._layout.H
        return np.r_[np.exp(phi[:H]), phi[H:]]

    @property
    def parameter_names(self) -> list[str]:
        """The names of :attr:`params`: ``"<from> -> <to>: intensity"``,
        ``"<from> -> <to>: <covariate>"`` (or ``"<covariate>"`` where it
        is shared) and ``"<from> -> <to>: period [<start>, <end>)"``."""
        L = self._layout
        names = self.transitions.transition_names()
        out = [f"{t}: intensity" for t in names]
        cov_names = [self.covariate_names[j] for j in self._kept]
        for c in range(L.n_cov):
            if L.shared[c]:
                out.append(cov_names[c])
            else:
                out += [f"{t}: {cov_names[c]}" for t in names]
        cuts = list(self.pci)
        for k in range(1, L.n_periods):
            hi = cuts[k] if k < len(cuts) else np.inf
            out += [f"{t}: period [{cuts[k - 1]:g}, {hi:g})" for t in names]
        return out

    @property
    def covariance_names(self) -> list[str]:
        """The names of the rows of :meth:`covariance`: those of
        :attr:`params`."""
        return self.parameter_names

    def _estimates(self) -> npt.NDArray:
        return np.asarray(self.params, dtype=float)

    def _jacobian(self) -> npt.NDArray:
        """d params / d theta."""
        A, _ = self._reported_map()
        H = self._layout.H
        J = A.copy()
        J[:H] *= np.exp(self._phi()[:H])[:, None]
        return J

    def covariance(self) -> npt.NDArray:
        """
        The covariance of :attr:`params`: the inverse of the observed
        information (the Hessian of the negative log-likelihood, by
        central differences of its exact gradient), carried to the
        reported parameters by the delta method. NaN where the
        information is singular.
        """
        P = self._layout.P
        if self._cov_theta is None:
            return np.full((P, P), np.nan)
        J = self._jacobian()
        return J @ self._cov_theta @ J.T

    def standard_errors(self) -> npt.NDArray:
        """The standard errors of :attr:`params`."""
        from surpyval.utils.linalg import standard_errors_of

        return standard_errors_of(self.covariance())

    def _parameter_bounds(self) -> list[tuple]:
        H = self._layout.H
        return [(0, None)] * H + [(None, None)] * (self._layout.P - H)

    def _index(self, name: str) -> int:
        names = self.parameter_names
        if name not in names:
            raise ValueError(
                f"Unknown parameter {name!r}; the parameters are {names!r}."
            )
        return names.index(name)

    def param_cb(
        self, name: str, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> Any:
        """
        The Wald confidence bound on one parameter (named as in
        :attr:`parameter_names`): on the log scale for an intensity, so
        that it stays positive, and on the coefficient's own scale for a
        covariate or period effect (exponentiate those for bounds on the
        intensity ratio).

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.param_cb("ok -> worn: intensity").round(3)
        array([0.299, 2.539])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        from scipy.stats import norm

        k = self._index(name)
        est = float(self.params[k])
        cov = self._cov_theta
        if cov is None:
            se = np.nan
        else:
            J = self._jacobian()[k]
            se = float(np.sqrt(max(J @ cov @ J, 0.0)))
        z = norm.ppf(1 - alpha_ci / (2.0 if bound == "two-sided" else 1.0))
        if k < self._layout.H:
            rel = se / est
            lower, upper = est * np.exp(-z * rel), est * np.exp(z * rel)
        else:
            lower, upper = est - z * se, est + z * se
        if bound == "lower":
            return lower
        if bound == "upper":
            return upper
        return np.array([lower, upper])

    def summary(self, alpha_ci: float = 0.05) -> Any:
        """
        Every parameter with its standard error and two-sided Wald
        bounds (as :meth:`param_cb`), and for the covariate and period
        effects the intensity ratio, ``exp`` of the coefficient, with its
        bounds (R ``msm``'s ``hazard.msm``).

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.summary().round(3)["estimate"].tolist()
        [0.871, 0.374]
        """
        import pandas as pd

        check_alpha_ci(alpha_ci)
        names = self.parameter_names
        H = self._layout.H
        rows = []
        for k, name in enumerate(names):
            lo, hi = self.param_cb(name, alpha_ci)
            est = float(self.params[k])
            ratio = (np.nan,) * 3 if k < H else tuple(np.exp([est, lo, hi]))
            rows.append(
                (est, float(self.standard_errors()[k]), lo, hi) + ratio
            )
        return pd.DataFrame(
            rows,
            index=pd.Index(names, name="parameter"),
            columns=[
                "estimate",
                "standard error",
                "lower",
                "upper",
                "ratio",
                "ratio lower",
                "ratio upper",
            ],
        )

    def with_params(self, params: npt.ArrayLike) -> Any:
        """
        A copy of the model with the parameters ``params`` (in the order
        of :attr:`parameter_names`), for predictions at other values; the
        covariance is kept.
        """
        import copy

        values = np.asarray(params, dtype=float).ravel()
        P = self._layout.P
        if values.size != P:
            raise ValueError(
                f"params holds {P} values, {self.parameter_names}; got "
                f"{values.size}."
            )
        H = self._layout.H
        if np.any(values[:H] <= 0):
            raise ValueError("The intensities in params must be positive.")
        phi = np.r_[np.log(values[:H]), values[H:]]
        A, b = self._reported_map()
        out = copy.copy(self)
        out._theta = np.linalg.solve(A, phi - b)
        return out

    def neg_ll(self) -> float:
        """The negative log-likelihood at the fit (conditional on each
        unit's first state, as R ``msm``'s)."""
        return float(self._neg_ll)

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood, ``-neg_ll()``."""
        return -self.neg_ll()

    def aic(self) -> float:
        """Akaike's information criterion, ``2 K + 2 neg_ll``."""
        return float(2 * self._layout.P + 2 * self.neg_ll())

    def bic(self) -> float:
        """The Bayesian information criterion, ``2 neg_ll + K ln(n)``,
        ``n`` the number of gaps between visits (:attr:`n_obs`)."""
        return float(2 * self.neg_ll() + self._layout.P * np.log(self.n_obs))

    def aic_c(self) -> float:
        """The small-sample corrected AIC, with the ``n`` of :meth:`bic`;
        ``nan`` where ``n <= K + 1``."""
        from surpyval.univariate.information_criteria import corrected_aic

        return corrected_aic(self.aic(), self._layout.P, self.n_obs)

    # -- predictions --------------------------------------------------------

    def _start(self, s: float | None) -> float:
        if s is None:
            return self.start
        s = float(s)
        if not np.isfinite(s):
            raise ValueError(
                f"'s', the start of the prediction, must be a finite time; "
                f"got {s}."
            )
        return s

    def _start_distribution(self, from_state: Any) -> npt.NDArray:
        if from_state is None:
            return self._p0
        p0 = np.zeros(self.transitions.n_states)
        p0[self.transitions.index(from_state)] = 1.0
        return p0

    def _u(self, rows: npt.NDArray) -> npt.NDArray:
        return self._standardise(np.atleast_2d(rows))

    def _P(
        self, U: npt.NDArray, s: float, times: npt.NDArray, deriv: bool
    ) -> tuple:
        """P(s, t | u) at ``times`` (finite, at or after ``s``), one row
        of ``U`` each, with the derivatives by theta."""
        sc = self._scale
        t0 = np.full(times.size, s / sc)
        return self._engine.gaps(self._theta, U, t0, times / sc, deriv=deriv)

    def _intensities(self, u: npt.NDArray, at: float) -> tuple:
        """The intensities (per unit time) for one standardised row and
        the period holding ``at``, with ``D`` (H, P)."""
        cuts = np.asarray(self.pci, dtype=float)
        period = np.array([np.searchsorted(cuts, at, side="right")])
        D = self._layout.design(u.reshape(1, -1), period)[0]
        q = np.exp(D @ self._theta) / self._scale
        return q, D

    def _matrix(self, u: npt.NDArray, at: float) -> npt.NDArray:
        q, _ = self._intensities(u, at)
        tm = self.transitions
        return _Q(q[None], tm.from_index(), tm.to_index(), tm.n_states)[0]

    def _times(self, x: Any, s: float) -> tuple:
        flat = np.asarray(x, dtype=float).ravel()
        ok = np.isfinite(flat) & (flat >= s)
        return flat, ok

    def _occ(
        self,
        flat: npt.NDArray,
        rows: npt.NDArray,
        state: Any,
        s: float,
        from_state: Any,
        se: bool,
    ) -> npt.NDArray:
        """The occupation probability (or its delta-method standard
        error) at each time, for its covariate row."""
        b = self.transitions.index(state)
        p0 = self._start_distribution(from_state)
        out = np.full(flat.size, np.nan)
        ok = np.isfinite(flat) & (flat >= s) & np.all(np.isfinite(rows), 1)
        if not np.any(ok):
            return out
        U = self._u(rows[ok])
        P, dP = self._P(U, s, flat[ok], deriv=se)
        if not se:
            out[ok] = np.clip((p0 @ P)[:, b], 0.0, 1.0)
            return out
        if self._cov_theta is None:
            return out
        G = np.einsum("k,npk->np", p0, dP[:, :, :, b])
        var = np.einsum("np,pq,nq->n", G, self._cov_theta, G)
        out[ok] = np.sqrt(np.maximum(var, 0.0))
        return out

    def _cb(
        self,
        p: npt.NDArray,
        se: npt.NDArray,
        alpha_ci: float,
        bound: str,
        bound_type: str,
    ) -> npt.NDArray:
        return probability_bounds(
            np.ravel(p), np.ravel(se), alpha_ci, bound, bound_type
        )

    def _sojourn(self, state: Any, rows: npt.NDArray) -> tuple:
        """The log mean sojourn time and its gradient by theta."""
        if self.pci:
            raise ValueError(
                "The mean time in a state is not one number when the "
                "intensities change at the 'pci' times: it depends on when "
                "the state is entered. Use intensity_matrix(at=...) for "
                "the intensities of a period (-1 / its diagonal is the mean "
                "stay there while they last)."
            )
        a = self.transitions.index(state)
        leaving = np.flatnonzero(self.transitions.from_index() == a)
        if leaving.size == 0:
            raise ValueError(
                f"{state!r} is an absorbing state: a unit never leaves it, "
                "so its stay has no finite mean."
            )
        u = self._u(rows)[0]
        q, D = self._intensities(u, self.start)
        total = float(np.sum(q[leaving]))
        grad = -(q[leaving] @ D[leaving]) / total
        return -np.log(total), grad

    def _sojourn_cb(
        self, state: Any, rows: npt.NDArray, alpha_ci: float, bound: str
    ) -> Any:
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        from scipy.stats import norm

        if np.any(~np.isfinite(rows)):
            out = np.array([np.nan, np.nan])
            return out if bound == "two-sided" else np.nan
        log_m, g = self._sojourn(state, rows)
        cov = self._cov_theta
        se = np.nan if cov is None else float(np.sqrt(max(g @ cov @ g, 0)))
        z = norm.ppf(1 - alpha_ci / (2.0 if bound == "two-sided" else 1.0))
        lower, upper = np.exp(log_m - z * se), np.exp(log_m + z * se)
        if bound == "lower":
            return lower
        if bound == "upper":
            return upper
        return np.array([lower, upper])

    # -- serialisation ------------------------------------------------------

    def _data_dict(self) -> dict:
        d = self._data
        tm = self.transitions
        return {
            "transitions": tm.to_dict(),
            "x": d.x.tolist(),
            "s": [to_native(v) for v in d.labels],
            "i": [to_native(d.ids[u]) for u in d.unit],
            "obstype": [OBSTYPES[k] for k in d.kind],
            "Z": d.Z.tolist(),
            "censored": [
                [to_native(code), [to_native(st) for st in states]]
                for code, states in self.censored.items()
            ],
            "exact_entry": [to_native(st) for st in self.exact_entry],
            "pci": list(self.pci),
            "covariate_names": list(self.covariate_names),
            "shared": list(self.shared),
            "theta": self._theta.tolist(),
            "cov_theta": (
                None if self._cov_theta is None else self._cov_theta.tolist()
            ),
            "_neg_ll": self._neg_ll,
            "n_obs": self.n_obs,
            "scale": self._scale,
            "mean": self._mean.tolist(),
            "sd": self._sd.tolist(),
            "kept": self._kept.tolist(),
            **maximum_entry(self.maximum),
        }

    @classmethod
    def _from_data_dict(cls, model_dict: dict) -> Any:
        from surpyval.univariate.competing_risks.labels import (
            label_from_native,
        )

        tm = TransitionMatrix.from_dict(model_dict["transitions"])
        p = len(model_dict["covariate_names"])
        Z = (
            np.asarray(model_dict["Z"], dtype=float).reshape(-1, p)
            if p
            else None
        )
        data = _check_panel(
            model_dict["x"],
            [label_from_native(v) for v in model_dict["s"]],
            [label_from_native(v) for v in model_dict["i"]],
            Z,
            model_dict["obstype"],
            tm,
            {
                label_from_native(code): [label_from_native(v) for v in st]
                for code, st in model_dict["censored"]
            }
            or None,
            [label_from_native(v) for v in model_dict["exact_entry"]] or None,
            model_dict["pci"] or None,
        )
        out = cls()
        out.transitions = tm
        out.covariate_names = list(model_dict["covariate_names"])
        out.shared = tuple(bool(v) for v in model_dict["shared"])
        out.pci = data.cuts
        out.exact_entry = data.exact_entry
        out.censored = dict(data.censored)
        out._data = data
        out._scale = float(model_dict["scale"])
        out._mean = np.asarray(model_dict["mean"], dtype=float)
        out._sd = np.asarray(model_dict["sd"], dtype=float)
        out._kept = np.asarray(model_dict["kept"], dtype=int)
        out._setup()
        out._theta = np.asarray(model_dict["theta"], dtype=float)
        cov = model_dict.get("cov_theta")
        out._cov_theta = None if cov is None else np.asarray(cov, float)
        out._neg_ll = float(model_dict["_neg_ll"])
        out.n_obs = int(model_dict["n_obs"])
        out.maximum = restored_maximum(model_dict)
        return out

    def _repr_lines(self, title: str) -> list[str]:
        tm = self.transitions
        lines = [
            title,
            f"States: {list(tm.states)!r}",
            f"Units: {len(self._data.ids)}, visits: {self._data.size}",
            "Transition intensities (moves per unit time"
            + (", at Z = 0" if self._regression else "")
            + "):",
        ]
        for name, q in zip(tm.transition_names(), self.params):
            lines.append(f"    {name}: {q:.6g}")
        if self._layout.P > self._layout.H:
            lines.append("Effects (log intensity ratios):")
            for name, v in zip(
                self.parameter_names[self._layout.H :],
                self.params[self._layout.H :],
            ):
                lines.append(f"    {name}: {v:.6g}")
        lines.append(f"Log-likelihood: {self.log_likelihood:.6g}")
        return lines


def _shared_columns(shared: Any, names: list) -> tuple:
    """The columns of the covariates named in ``shared`` (by name or
    position), each with one coefficient for every transition."""
    if shared is None:
        return ()
    if isinstance(shared, dict):
        raise ValueError(
            "shared takes the covariates (names or column numbers) whose one "
            "coefficient applies to every transition, such as shared=0 or "
            "shared=['age']; sharing a coefficient among some transitions "
            "(a dict) is not built for the panel-data models."
        )
    if isinstance(shared, (str, int, np.integer)):
        shared = [shared]
    out = []
    for item in shared:
        if isinstance(item, (int, np.integer)) and not isinstance(item, bool):
            if not 0 <= int(item) < len(names):
                raise ValueError(
                    f"shared names column {item}, but Z has {len(names)} "
                    "column(s)."
                )
            out.append(int(item))
        elif item in names:
            out.append(names.index(item))
        else:
            raise ValueError(
                f"shared names {item!r}, which is not a covariate; the "
                f"covariates are {names!r}."
            )
    return tuple(sorted(set(out)))


def _scoring(lik: _Likelihood, theta: npt.NDArray) -> tuple:
    """Fisher scoring with the outer product of the units' scores (the
    BHHH matrix, which needs only the gradient), each step halved until
    the likelihood rises. Returns the answer and whether it converged."""
    f = lik.fun(theta)
    if not np.isfinite(f):
        return theta, False
    for _ in range(200):
        S = lik.scores(theta)
        g = -S.sum(axis=0)
        B = S.T @ S
        B += 1e-10 * np.trace(B) / B.shape[0] * np.eye(B.shape[0])
        try:
            step = np.linalg.solve(B, g)
        except np.linalg.LinAlgError:
            return theta, False
        if not np.all(np.isfinite(step)):
            return theta, False
        length = np.max(np.abs(step))
        if length > 2.0:
            step *= 2.0 / length
        for _ in range(30):
            trial = theta - step
            ft = lik.fun(trial)
            if np.isfinite(ft) and ft <= f:
                break
            step = step / 2
        else:
            return theta, False
        done = (
            f - ft < 1e-12 * max(1.0, abs(f)) and np.max(np.abs(step)) < 1e-6
        )
        theta, f = trial, ft
        if done:
            return theta, True
    return theta, False


def _newton_polish(lik: _Likelihood, theta: npt.NDArray) -> npt.NDArray:
    """A few Newton steps from the search's answer (on the Hessian at
    its answer), each kept only if it lowers the negative
    log-likelihood: BFGS's own stopping rule leaves the gradient a
    little short of 0."""
    f = lik.fun(theta)
    H = lik.hess(theta)
    try:
        np.linalg.cholesky(H)
    except np.linalg.LinAlgError:
        return theta
    for _ in range(6):
        step = np.linalg.solve(H, lik.jac(theta))
        trial = theta - step
        ft = lik.fun(trial)
        if not np.isfinite(ft) or ft > f + 1e-12 * max(abs(f), 1.0):
            break
        theta, f = trial, ft
        if np.max(np.abs(step)) < 1e-10:
            break
    return theta


def _inverse_information(H: npt.NDArray) -> npt.NDArray | None:
    if not np.all(np.isfinite(H)):
        return None
    try:
        np.linalg.cholesky(H)
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        warnings.warn(
            "The information matrix of the panel multi-state fit is not "
            "positive definite, so its standard errors and bounds are NaN. "
            "A transition, covariate or period may have too little data: "
            "simplify the model.",
            UserWarning,
            stacklevel=4,
        )
        return None
    return 0.5 * (cov + cov.T)


class PanelMultiState(_PanelMultiStateBase):
    r"""
    A multi-state Markov model for panel data: units whose state is seen
    only at visits (inspections, clinic visits, survey waves).

    Each allowed move from state :math:`r` to state :math:`s` has a
    constant *transition intensity* :math:`q_{rs}`: the rate of that move
    while a unit is in :math:`r`. The chance of being in :math:`s` a time
    :math:`t` after being in :math:`r` is element :math:`(r, s)` of
    :math:`P(t) = \exp(Q t)`, :math:`Q` the matrix of intensities, so a
    unit may make several moves between two visits, unseen. The
    intensities are fitted by maximum likelihood from the states seen at
    each visit (Kalbfleisch and Lawless 1985), as R's ``msm`` does: each
    unit contributes the probability of its sequence of states given its
    first, a product of elements of :math:`P` over the gaps between its
    visits.

    The mean stay in a state is :math:`-1 / q_{rr}`, and the probabilities
    of being in each state later follow from :math:`P(t)`; their bounds
    come from the delta method over the intensities.

    Call the class method ``PanelMultiState.fit`` (or ``fit_from_df``); it
    returns a fitted instance. For covariates on the intensities use
    :class:`PanelMultiStateRegression`.

    References
    ----------
    Kalbfleisch, J. D. and Lawless, J. F. (1985), "The analysis of panel
    data under a Markov assumption", *Journal of the American Statistical
    Association*, 80(392), 863-871.

    Jackson, C. H. (2011), "Multi-state models for panel data: the msm
    package for R", *Journal of Statistical Software*, 38(8), 1-29.
    """

    @classmethod
    @takes_unit
    def fit(
        cls,
        x: npt.ArrayLike,
        s: npt.ArrayLike,
        i: npt.ArrayLike,
        obstype: npt.ArrayLike | None = None,
        transitions: Any = None,
        censored: dict | None = None,
        exact_entry: Any = None,
        pci: Any = None,
    ) -> PanelMultiState:
        """
        Fit the transition intensities to the states seen at visits.

        Parameters
        ----------
        x : array_like
            The time of each visit (one row per visit).
        s : array_like
            The state seen at each visit; or a code from ``censored``
            where only a set of states is known.
        i : array_like
            The unit each visit belongs to.
        obstype : array_like, optional
            How each row was observed, one of ``'panel'`` (the default: a
            snapshot of the state at ``x``), ``'exact'`` (a move into
            ``s`` at exactly ``x``, the unit having stayed in its
            previous state until then; also a last row repeating the
            state, where it was seen to stay to ``x``) or
            ``'exact_entry'`` (an exactly timed entry into an absorbing
            state, from a state not seen). msm's ``obstype`` 1, 2 and 3.
            A unit's first row is its starting state, whatever it says.
        transitions : TransitionMatrix or list of (from, to)
            The allowed transitions (needed: moves cannot be read off
            visits).
        censored : dict, optional
            ``{code: states}``: a value of ``s`` that means the unit was
            in one of ``states`` (``{'alive': ['healthy', 'ill']}`` for a
            unit known to be alive at the end of follow-up, its state not
            seen).
        exact_entry : state or list of states, optional
            Absorbing states (death, scrapped) whose entry times are
            exact, from an unknown state just before: every row seeing
            them is ``obstype='exact_entry'`` (msm's ``deathexact``).
        pci : list of float, optional
            Times at which every intensity changes (piecewise-constant
            intensities): each period after the first multiplies each
            transition's intensity by its own factor, estimated.

        Returns
        -------
        PanelMultiState
            The fitted model.

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.params.round(3)
        array([0.871, 0.374])
        >>> model.occupancy(2.0, "failed", from_state="ok").round(3)
        np.float64(0.302)
        """
        data = _check_panel(
            x, s, i, None, obstype, transitions, censored, exact_entry, pci
        )
        return cls._fit(data, [], None)

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        s_col: str,
        i_col: str,
        obstype_col: str | None = None,
        transitions: Any = None,
        censored: dict | None = None,
        exact_entry: Any = None,
        pci: Any = None,
    ) -> PanelMultiState:
        """
        Fit the model to a DataFrame of visits, one row each; the columns
        are the arguments of :meth:`fit`.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.multistate import PanelMultiState
        >>> df = pd.DataFrame({
        ...     "unit": [1, 1, 1, 2, 2, 3, 3, 3],
        ...     "time": [0, 1, 2, 0, 2, 0, 1, 3],
        ...     "state": ["ok", "worn", "failed", "ok", "worn", "ok",
        ...               "ok", "worn"],
        ... })
        >>> model = PanelMultiState.fit_from_df(
        ...     df, "time", "state", "unit",
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.parameter_names
        ['ok -> worn: intensity', 'worn -> failed: intensity']
        """
        from surpyval.utils.dataframe import frame_column

        return cls.fit(
            frame_column(df, x_col, "x_col"),
            frame_column(df, s_col, "s_col"),
            frame_column(df, i_col, "i_col"),
            obstype=(
                None
                if obstype_col is None
                else frame_column(df, obstype_col, "obstype_col")
            ),
            transitions=transitions,
            censored=censored,
            exact_entry=exact_entry,
            pci=pci,
        )

    def intensity_matrix(self, at: float | None = None) -> npt.NDArray:
        """
        The fitted transition intensity matrix :math:`Q` (``(K, K)``,
        rows summing to 0): element ``[a, b]`` is the rate of the move
        from state ``a`` to ``b``, the diagonal minus the rate of leaving
        each state. With ``pci``, those of the period holding ``at``
        (default: the first).

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.intensity_matrix().round(3)
        array([[-0.871,  0.871,  0.   ],
               [ 0.   , -0.374,  0.374],
               [ 0.   ,  0.   ,  0.   ]])
        """
        at = self.start if at is None else float(at)
        return self._matrix(np.zeros(self._layout.n_cov), at)

    def transition_probabilities(
        self, x: npt.ArrayLike, s: float | None = None
    ) -> npt.NDArray:
        """
        The transition probability matrices :math:`P(s, t)` at the times
        ``x``: element ``[..., a, b]`` is the probability of being in
        state ``b`` at ``x`` for a unit in state ``a`` at ``s`` (by
        default the earliest visit). Without ``pci`` this is
        :math:`\\exp\\{Q (t - s)\\}` (R ``msm``'s ``pmatrix.msm``).
        NaN before ``s``.

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.transition_probabilities(1.0).round(3)
        array([[0.418, 0.472, 0.109],
               [0.   , 0.688, 0.312],
               [0.   , 0.   , 1.   ]])
        """
        s = self._start(s)
        k = self.transitions.n_states
        xs = np.asarray(x, dtype=float)
        flat, ok = self._times(xs, s)
        out = np.full((flat.size, k, k), np.nan)
        if np.any(ok):
            U = np.zeros((int(ok.sum()), self._layout.n_cov))
            out[ok] = self._P(U, s, flat[ok], deriv=False)[0]
        return out.reshape(xs.shape + (k, k))

    @keeps_query_shape
    def occupancy(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The state occupation probability: the probability of being in
        ``state`` at ``x``.

        Parameters
        ----------
        x : array_like
            The times.
        state : any
            The state.
        s : float, optional
            The time the prediction starts from. Defaults to the earliest
            visit.
        from_state : any, optional
            The unit's state at ``s``. By default the units' first states
            seen, in proportion.

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.occupancy([1, 2, 4], "worn").round(3)
        array([0.472, 0.522, 0.339])
        """
        s = self._start(s)
        flat = np.asarray(x, dtype=float).ravel()
        rows = np.zeros((flat.size, len(self.covariate_names)))
        return self._occ(flat, rows, state, s, from_state, se=False)

    @keeps_query_shape
    def occupancy_se(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The standard error of :meth:`occupancy`, by the delta method over
        the intensities (exact derivatives of the matrix exponential; the
        starting distribution taken as known).

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.occupancy_se(2.0, "failed").round(3)
        np.float64(0.181)
        """
        s = self._start(s)
        flat = np.asarray(x, dtype=float).ravel()
        rows = np.zeros((flat.size, len(self.covariate_names)))
        return self._occ(flat, rows, state, s, from_state, se=True)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "exp",
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on :meth:`occupancy`, from the
        delta-method standard error of :meth:`occupancy_se`. With
        ``from_state`` they are bounds on one element of
        :meth:`transition_probabilities`.

        Parameters
        ----------
        x, state, s, from_state
            As for :meth:`occupancy`.
        alpha_ci : float, optional
            One minus the confidence level (default 0.05).
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Which bounds.
        bound_type : ('exp', 'normal'), str, optional
            ``'exp'`` (the default) works on the log(-log) scale and
            keeps the bounds in (0, 1); ``'normal'`` is the estimate plus
            and minus ``z`` standard errors.

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.cb(2.0, "failed").round(3)
        array([0.041, 0.639])
        """
        check_bound_args(alpha_ci, bound, bound_type)
        p = self.occupancy(x, state, s, from_state)
        se = self.occupancy_se(x, state, s, from_state)
        return self._cb(p, se, alpha_ci, bound, bound_type)

    def mean_sojourn(self, state: Any) -> float:
        """
        The mean time a unit stays in ``state`` each time it enters it,
        :math:`-1 / q_{rr}` (R ``msm``'s ``sojourn.msm``). Refused for an
        absorbing state and for a model with ``pci``.

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> round(model.mean_sojourn("ok"), 3)
        1.148
        """
        rows = np.zeros((1, len(self.covariate_names)))
        return float(np.exp(self._sojourn(state, rows)[0]))

    def sojourn_cb(
        self, state: Any, alpha_ci: float = 0.05, bound: str = "two-sided"
    ) -> Any:
        """
        The Wald bound on :meth:`mean_sojourn`, on the log scale (so it
        stays positive).

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiState
        >>> model = PanelMultiState.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ... )
        >>> model.sojourn_cb("ok").round(3)
        array([0.394, 3.346])
        """
        rows = np.zeros((1, len(self.covariate_names)))
        return self._sojourn_cb(state, rows, alpha_ci, bound)

    def to_dict(self) -> dict:
        """
        Serialise the model to a plain, JSON-serialisable dict: the visits
        and the fitted parameters with their covariance. The reloaded
        model predicts, and bounds, exactly as the original, without
        refitting.
        """
        return stamp_schema({"model": "PanelMultiState", **self._data_dict()})

    @classmethod
    def from_dict(cls, model_dict: dict) -> PanelMultiState:
        """Rebuild the model from :meth:`to_dict`."""
        require_model_tag(
            model_dict, "PanelMultiState", "a panel multi-state model"
        )
        return cls._from_data_dict(model_dict)

    def __repr__(self) -> str:
        return "\n".join(
            self._repr_lines(
                "Panel multi-state model (a Markov model of states seen "
                "at visits)"
            )
        )


class PanelMultiStateRegression(_PanelMultiStateBase):
    r"""
    A multi-state Markov model for panel data with covariates on the
    transition intensities: :math:`q_{rs}(z) = q_{rs} \exp(\beta_{rs}^T
    z)`, each transition with its own coefficients, or one coefficient for
    every transition for the covariates named in ``shared``.

    Covariates may change from visit to visit: those of a row hold over
    the gap that starts at it (as in R ``msm``). Otherwise the model is
    :class:`PanelMultiState`; its predictions are for given covariates.

    Call the class method ``PanelMultiStateRegression.fit`` (or
    ``fit_from_df``); it returns a fitted instance.

    References
    ----------
    Marshall, G. and Jones, R. H. (1995), "Multi-state models and
    diabetic retinopathy", *Statistics in Medicine*, 14(18), 1975-1983.

    Jackson, C. H. (2011), "Multi-state models for panel data: the msm
    package for R", *Journal of Statistical Software*, 38(8), 1-29.
    """

    _regression = True

    @classmethod
    @takes_unit
    def fit(
        cls,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        s: npt.ArrayLike,
        i: npt.ArrayLike,
        obstype: npt.ArrayLike | None = None,
        transitions: Any = None,
        censored: dict | None = None,
        exact_entry: Any = None,
        pci: Any = None,
        shared: Any = None,
    ) -> PanelMultiStateRegression:
        """
        Fit the transition intensities and the covariates' effects on
        them to the states seen at visits.

        Parameters
        ----------
        x : array_like
            The time of each visit.
        Z : array_like
            The covariates, one row per visit; a row's values hold until
            the unit's next visit.
        s, i, obstype, transitions, censored, exact_entry, pci
            As for :meth:`PanelMultiState.fit`.
        shared : str, int or list, optional
            Covariates (names or column numbers) whose one coefficient
            applies to every transition.

        Returns
        -------
        PanelMultiStateRegression
            The fitted model.

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiStateRegression
        >>> model = PanelMultiStateRegression.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3, 0, 2],
        ...     Z=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn", "ok", "ok"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ...     shared=0,
        ... )
        >>> model.parameter_names
        ['ok -> worn: intensity', 'worn -> failed: intensity', 'z0']
        """
        from surpyval.multistate.regression import _covariates

        x_arr = np.asarray(x, dtype=float)
        if Z is None:
            raise ValueError(
                "Z is needed; without covariates use PanelMultiState."
            )
        _, names = _covariates(Z, int(np.size(x_arr)))
        data = _check_panel(
            x, s, i, Z, obstype, transitions, censored, exact_entry, pci
        )
        return cls._fit(data, names, shared)

    @classmethod
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        Z_cols: Any,
        s_col: str,
        i_col: str,
        obstype_col: str | None = None,
        transitions: Any = None,
        censored: dict | None = None,
        exact_entry: Any = None,
        pci: Any = None,
        shared: Any = None,
    ) -> PanelMultiStateRegression:
        """
        Fit the model to a DataFrame of visits; the columns are the
        arguments of :meth:`fit` (``Z_cols`` a column name or a list of
        them, which name the covariates).

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.multistate import PanelMultiStateRegression
        >>> df = pd.DataFrame({
        ...     "unit": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4],
        ...     "time": [0, 1, 2, 0, 2, 0, 1, 3, 0, 2],
        ...     "state": ["ok", "worn", "failed", "ok", "worn", "ok",
        ...               "ok", "worn", "ok", "ok"],
        ...     "heavy": [1, 1, 1, 0, 0, 1, 1, 1, 0, 0],
        ... })
        >>> model = PanelMultiStateRegression.fit_from_df(
        ...     df, "time", "heavy", "state", "unit",
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ...     shared="heavy",
        ... )
        >>> model.covariate_names
        ['heavy']
        """
        import pandas as pd

        from surpyval.utils.dataframe import frame_column

        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        Z = pd.DataFrame({c: frame_column(df, c, "Z_cols") for c in cols})
        return cls.fit(
            frame_column(df, x_col, "x_col"),
            Z,
            frame_column(df, s_col, "s_col"),
            frame_column(df, i_col, "i_col"),
            obstype=(
                None
                if obstype_col is None
                else frame_column(df, obstype_col, "obstype_col")
            ),
            transitions=transitions,
            censored=censored,
            exact_entry=exact_entry,
            pci=pci,
            shared=shared,
        )

    @property
    def betas(self) -> npt.NDArray:
        """``(transitions, covariates)``: each transition's coefficient
        of each covariate (a log intensity ratio per unit of it), the
        same down a shared covariate's column; NaN for an aliased one."""
        tm = self.transitions
        p = len(self.covariate_names)
        out = np.full((tm.n_transitions, p), np.nan)
        phi = self._phi()
        L = self._layout
        for c, j in enumerate(self._kept):
            out[:, j] = phi[L.cov_index[c]]
        return out

    @property
    def aliased(self) -> npt.NDArray:
        """The columns of ``Z`` that could not be estimated (constant, or
        a combination of others, over the gaps between visits)."""
        p = len(self.covariate_names)
        return np.setdiff1d(np.arange(p), self._kept)

    def _z(self, Z: Any) -> npt.NDArray:
        z = np.asarray(Z, dtype=float).ravel()
        if z.size != len(self.covariate_names):
            raise ValueError(
                f"Z must be one row of {len(self.covariate_names)} "
                "covariate(s)."
            )
        return z.reshape(1, -1)

    def _rows(self, Z: Any, size: int) -> npt.NDArray:
        p = len(self.covariate_names)
        Z = np.asarray(Z, dtype=float)
        if Z.ndim <= 1:
            Z = Z.reshape(1, -1)
        if Z.shape[1] != p:
            raise ValueError(
                f"Z must have {p} covariate column(s); got {Z.shape[1]}."
            )
        if Z.shape[0] == 1:
            return np.repeat(Z, size, axis=0)
        if Z.shape[0] != size:
            raise ValueError(
                "Z must be one covariate row for every time, or one row "
                f"per time ({size}); got {Z.shape[0]} rows."
            )
        return Z

    def intensity_matrix(
        self, Z: npt.ArrayLike, at: float | None = None
    ) -> npt.NDArray:
        """
        The transition intensity matrix :math:`Q(z)` for one covariate
        row ``Z`` (as :meth:`PanelMultiState.intensity_matrix`).

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiStateRegression
        >>> model = PanelMultiStateRegression.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3, 0, 2],
        ...     Z=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn", "ok", "ok"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ...     shared=0,
        ... )
        >>> model.intensity_matrix([1]).shape
        (3, 3)
        """
        z = self._z(Z)
        at = self.start if at is None else float(at)
        return self._matrix(self._u(z)[0], at)

    def transition_probabilities(
        self, x: npt.ArrayLike, Z: npt.ArrayLike, s: float | None = None
    ) -> npt.NDArray:
        """
        The predicted transition probability matrices :math:`P(s, t \\mid
        z)` at the times ``x`` for one covariate row ``Z``. NaN before
        ``s``.

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiStateRegression
        >>> model = PanelMultiStateRegression.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3, 0, 2],
        ...     Z=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn", "ok", "ok"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ...     shared=0,
        ... )
        >>> model.transition_probabilities(1.0, [1]).shape
        (3, 3)
        """
        s = self._start(s)
        z = self._z(Z)
        k = self.transitions.n_states
        xs = np.asarray(x, dtype=float)
        flat, ok = self._times(xs, s)
        out = np.full((flat.size, k, k), np.nan)
        if np.any(ok) and np.all(np.isfinite(z)):
            U = np.repeat(self._u(z), int(ok.sum()), axis=0)
            out[ok] = self._P(U, s, flat[ok], deriv=False)[0]
        return out.reshape(xs.shape + (k, k))

    @keeps_query_shape
    def occupancy(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The predicted state occupation probability: the probability of
        being in ``state`` at ``x`` for a unit with covariates ``Z`` (one
        row for every time, or one row per time; NaN for a row with a
        missing covariate). ``s`` and ``from_state`` are as for
        :meth:`PanelMultiState.occupancy`.

        Examples
        --------
        >>> from surpyval.multistate import PanelMultiStateRegression
        >>> model = PanelMultiStateRegression.fit(
        ...     x=[0, 1, 2, 3, 0, 1, 2, 0, 2, 3, 0, 1, 3, 0, 2],
        ...     Z=[0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        ...     s=["ok", "ok", "worn", "failed", "ok", "worn", "worn",
        ...        "ok", "worn", "failed", "ok", "ok", "worn", "ok", "ok"],
        ...     i=[1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
        ...     transitions=[("ok", "worn"), ("worn", "failed")],
        ...     shared=0,
        ... )
        >>> model.occupancy([2, 2], [[0], [1]], "failed").shape
        (2,)
        """
        s = self._start(s)
        flat = np.asarray(x, dtype=float).ravel()
        rows = self._rows(Z, flat.size)
        return self._occ(flat, rows, state, s, from_state, se=False)

    @keeps_query_shape
    def occupancy_se(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
    ) -> npt.NDArray:
        """
        The delta-method standard error of :meth:`occupancy`, over the
        intensities and the coefficients.
        """
        s = self._start(s)
        flat = np.asarray(x, dtype=float).ravel()
        rows = self._rows(Z, flat.size)
        return self._occ(flat, rows, state, s, from_state, se=True)

    @keeps_query_shape
    def cb(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        state: Any,
        s: float | None = None,
        from_state: Any = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        bound_type: str = "exp",
    ) -> npt.NDArray:
        """
        Pointwise confidence bounds on :meth:`occupancy` (as
        :meth:`PanelMultiState.cb`).
        """
        check_bound_args(alpha_ci, bound, bound_type)
        p = self.occupancy(x, Z, state, s, from_state)
        se = self.occupancy_se(x, Z, state, s, from_state)
        return self._cb(p, se, alpha_ci, bound, bound_type)

    def mean_sojourn(self, state: Any, Z: npt.ArrayLike) -> float:
        """The mean stay in ``state`` for a unit with covariates ``Z``
        (as :meth:`PanelMultiState.mean_sojourn`)."""
        z = self._z(Z)
        if np.any(~np.isfinite(z)):
            return np.nan
        return float(np.exp(self._sojourn(state, z)[0]))

    def sojourn_cb(
        self,
        state: Any,
        Z: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> Any:
        """The Wald bound on :meth:`mean_sojourn` (as
        :meth:`PanelMultiState.sojourn_cb`)."""
        return self._sojourn_cb(state, self._z(Z), alpha_ci, bound)

    def to_dict(self) -> dict:
        """Serialise the model to a plain, JSON-serialisable dict (as
        :meth:`PanelMultiState.to_dict`)."""
        return stamp_schema(
            {"model": "PanelMultiStateRegression", **self._data_dict()}
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> PanelMultiStateRegression:
        """Rebuild the model from :meth:`to_dict`."""
        require_model_tag(
            model_dict,
            "PanelMultiStateRegression",
            "a panel multi-state regression",
        )
        return cls._from_data_dict(model_dict)

    def __repr__(self) -> str:
        return "\n".join(
            self._repr_lines(
                "Panel multi-state regression (a Markov model of states "
                "seen at visits, covariates on the intensities)"
            )
        )
