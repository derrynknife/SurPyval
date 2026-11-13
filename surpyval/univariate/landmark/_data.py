"""Landmark data: from a :class:`~surpyval.SurpyvalData` of lifetimes, the
units still at risk at a landmark time, their covariates as known then,
and their follow-up cut at the landmark plus a horizon, as another
``SurpyvalData``."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt


def check_landmarks(at: Any, horizon: Any) -> tuple[npt.NDArray, float]:
    """The landmark times, sorted and distinct, and the horizon."""
    s = np.atleast_1d(np.asarray(at, dtype=float))
    if s.ndim != 1 or s.size == 0:
        raise ValueError(
            "'at' must be one landmark time or a 1-D list of them; got "
            "shape {}.".format(np.shape(at))
        )
    if not np.isfinite(s).all():
        raise ValueError(f"'at' must be finite times; got {s.tolist()}.")
    if np.unique(s).size != s.size:
        raise ValueError(
            f"'at' has repeated times ({s.tolist()}); give each once."
        )
    try:
        w = float(horizon)
    except (TypeError, ValueError):
        raise ValueError(
            "'horizon' must be one positive number (the window w: "
            "follow-up is cut at L + w); got {!r}.".format(horizon)
        ) from None
    if not (np.isfinite(w) and w > 0):
        raise ValueError(
            "'horizon' must be a positive, finite number (the window w: "
            "follow-up is cut at L + w); got {!r}.".format(horizon)
        )
    return np.sort(s), w


class Lives:
    """The lifetimes of ``life`` checked for landmarking: each unit's id,
    entry, time, event flag, count and covariates (fixed, time-varying
    and from the readings), with the covariates' names."""

    def __init__(self, life: Any) -> None:
        from surpyval.utils.surpyval_data import SurpyvalData

        if not isinstance(life, SurpyvalData):
            raise TypeError(
                "'life' must be a SurpyvalData of the units' lifetimes "
                "(SurpyvalData(x, c, n, tl=..., Z=...), from_dataframe or "
                "from_items); got {}.".format(type(life).__name__)
            )
        x = np.asarray(life.x, dtype=float)
        c = np.asarray(life.c)
        if x.ndim != 1 or np.isin(c, (-1, 2)).any():
            raise ValueError(
                "Landmarking needs each unit's time observed or right "
                "censored (c = 0 or 1); this data has left or interval "
                "censored units."
            )
        if np.isfinite(np.asarray(life.tr, dtype=float)).any():
            raise ValueError(
                "Landmarking takes no right truncation (tr); this data has "
                "right truncated units."
            )
        if getattr(life, "w", None) is not None:
            raise ValueError(
                "Landmarking with weights (w) is not built: refit without "
                "them."
            )
        if getattr(life, "e", None) is not None:
            raise ValueError(
                "Landmarking with event types (competing risks) is not "
                "built: give the data of one event (drop e, with the other "
                "events as censored)."
            )
        rows_n = x.size
        ids = life.ids if life.ids is not None else list(range(rows_n))
        tl = np.asarray(life.tl, dtype=float)
        n_row = np.asarray(life.n, dtype=float)
        # The units in order of first appearance, and each unit's rows in
        # the order given (SurpyvalData keeps a unit's rows in time order).
        order: dict = {}
        for r, u in enumerate(ids):
            order.setdefault(u, []).append(r)
        self.ids = list(order)
        k = len(self.ids)
        self.rows = [np.array(rs, dtype=int) for rs in order.values()]
        self.entry = np.empty(k)
        self.end = np.empty(k)
        self.event = np.zeros(k, dtype=bool)
        self.n = np.empty(k)
        # Each row's start: its tl where finite, else where the unit's
        # previous row ended (-inf for a first row with no entry).
        self.start = np.empty(rows_n)
        for j, rs in enumerate(self.rows):
            prev = -np.inf
            for q, r in enumerate(rs):
                if np.isfinite(tl[r]):
                    if q and tl[r] > prev:
                        raise ValueError(
                            "Landmarking needs each unit followed without "
                            "a break: unit {!r} has no row from {:g} to "
                            "{:g}. Add a row for the gap (its covariates "
                            "carried forward) if the unit was at risk "
                            "then, or end its follow-up at {:g}.".format(
                                self.ids[j], prev, tl[r], prev
                            )
                        )
                    self.start[r] = tl[r]
                else:
                    self.start[r] = prev
                if q < len(rs) - 1 and c[r] == 0:
                    raise ValueError(
                        "Unit {!r} has an event before its last row: "
                        "landmarking takes one event per unit (repeated "
                        "events are the recurrent models' data, in "
                        "surpyval.recurrent).".format(self.ids[j])
                    )
                prev = x[r]
            if np.unique(n_row[rs]).size > 1:
                raise ValueError(
                    "'n' counts identical units, so it must be the same on "
                    "every row of a unit; unit {!r} has {}.".format(
                        self.ids[j], sorted(set(n_row[rs].tolist()))
                    )
                )
            self.entry[j] = self.start[rs[0]]
            self.end[j] = x[rs[-1]]
            self.event[j] = c[rs[-1]] == 0
            self.n[j] = n_row[rs[0]]
        a = getattr(life, "a", None)
        if a is not None:
            last = np.array([rs[-1] for rs in self.rows], dtype=int)
            if (self.event & (np.asarray(a)[last] == 0)).any():
                raise ValueError(
                    "A unit's last event has a = 0 (a recurrent event, not "
                    "the end of its life): landmarking takes one terminal "
                    "event per unit; the recurrent models in "
                    "surpyval.recurrent take repeated events."
                )
        names: list[str] = []
        fixed = None
        if life.Z is not None:
            fixed = np.asarray(life.Z, dtype=float).reshape(rows_n, -1)
            bad = ~np.isfinite(fixed).all(axis=1)
            if bad.any():
                raise ValueError(
                    "{} row(s) have a missing (NaN) or infinite "
                    "covariate. A landmark data set keeps or drops whole "
                    "units, so give every unit its covariates, or leave "
                    "those units out of the data first.".format(int(bad.sum()))
                )
            names = (
                list(life.covariate_names)
                if life.covariate_names is not None
                else [f"Z{j}" for j in range(fixed.shape[1])]
            )
        self.fixed = fixed
        self.schedules = None
        tvc = getattr(life, "time_varying_covariates", None)
        if tvc:
            self.schedules = [tvc[u] for u in self.ids]
            width = self.schedules[0].p
            names += [f"time_varying_{j}" for j in range(width)]
        self.readings = _measurements(getattr(life, "m", None), self.ids)
        names += list(self.readings)
        self.names = names

    def _row_at(self, j: int, L: float) -> int:
        """The row of unit ``j`` whose covariates are in force at ``L``:
        its last row starting at or before ``L``."""
        rs = self.rows[j]
        k = int(np.searchsorted(self.start[rs], L, side="right")) - 1
        return int(rs[max(k, 0)])

    @property
    def p(self) -> int:
        return len(self.names)

    def at(self, L: float) -> tuple[npt.NDArray, npt.NDArray, int]:
        """The units at risk at ``L`` (entered by ``L`` and followed
        event-free past it), their covariates as known at ``L`` (the last
        value at or before ``L``), and how many were dropped for want of
        a reading by ``L``."""
        rows = np.flatnonzero((self.entry <= L) & (self.end > L))
        parts = []
        if self.fixed is not None:
            held = np.array([self._row_at(int(j), L) for j in rows], int)
            parts.append(self.fixed[held])
        if self.schedules is not None:
            parts.append(
                np.array(
                    [_value_at(self.schedules[r], L) for r in rows], float
                ).reshape(rows.size, -1)
            )
        dropped = 0
        if self.readings:
            m = np.column_stack(
                [
                    [_last_reading(series[r], L) for r in rows]
                    for series in self.readings.values()
                ]
            ).reshape(rows.size, -1)
            parts.append(m)
            keep = ~np.isnan(m).any(axis=1)
            dropped = int((~keep).sum())
            rows = rows[keep]
            parts = [part[keep] for part in parts]
        Z = (
            np.column_stack(parts).astype(float)
            if parts
            else np.empty((rows.size, 0))
        )
        return rows, Z, dropped


def _value_at(schedule: Any, L: float) -> npt.NDArray:
    """A unit's time-varying covariates at ``L``: the last value at or
    before ``L`` (a step taking effect at ``L`` counts)."""
    from surpyval.univariate.regression.tvc_path import CovariatePath

    if L < schedule.start:
        raise ValueError(
            "A time-varying covariate starts at {:g}, after the landmark "
            "{:g}: nothing says what it was then. Start each unit's "
            "schedule at or before its entry.".format(schedule.start, L)
        )
    if isinstance(schedule, CovariatePath):
        return np.asarray(schedule(L), dtype=float).ravel()
    starts, _, Z = schedule.segments(L + 1.0)
    k = int(np.searchsorted(starts, L, side="right")) - 1
    return Z[k]


def _measurements(m: Any, ids: list) -> dict:
    """Each measurement's readings per unit, ``{name: [(ages, values),
    ...]}`` in the order of ``ids``, sorted by age, from the data's
    :class:`~surpyval.Measurements` ``m`` (``{}`` without any). The one
    unnamed measurement is named ``"m"``."""
    if m is None:
        return {}
    out = {}
    for name in m.names:
        per_unit = m[name]
        series = []
        for u in ids:
            readings = per_unit.get(u, {})
            ages = np.array(sorted(readings), dtype=float)
            values = np.array([readings[a] for a in sorted(readings)], float)
            series.append((ages, values))
        out["m" if name is None else str(name)] = series
    return out


def _last_reading(reading: tuple, L: float) -> float:
    """The last reading at or before ``L`` (nan if none)."""
    ages, values = reading
    k = int(np.searchsorted(ages, L, side="right")) - 1
    return float(values[k]) if k >= 0 else np.nan


def stack(lives: Lives, at: npt.NDArray, horizon: float) -> dict:
    """The landmark rows at each landmark, stacked: one row per unit at
    risk at each landmark ``L``, entering at ``L``, followed to ``min(T, L
    + horizon)`` (an event only if it is by ``L + horizon``), with its
    covariates as known at ``L``."""
    unit, lm, Zs = [], [], []
    dropped = []
    for j, L in enumerate(at):
        rows, Z, gone = lives.at(float(L))
        unit.append(rows)
        lm.append(np.full(rows.size, j, dtype=int))
        Zs.append(Z)
        if gone:
            dropped.append((float(L), gone))
    if dropped:
        from surpyval.utils.warnings import caller_stacklevel

        warnings.warn(
            "Units at risk with no reading at or before the landmark were "
            "left out of it: "
            + ", ".join(f"{g} at L = {L:g}" for L, g in dropped)
            + ". Give each unit a reading by its first landmark, or "
            "choose later landmarks.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    u = np.concatenate(unit).astype(int)
    k = np.concatenate(lm).astype(int)
    Z = np.concatenate(Zs).reshape(u.size, lives.p)
    L = at[k]
    T = lives.end[u]
    x = np.minimum(T, L + horizon)
    c = np.where(lives.event[u] & (T <= L + horizon), 0, 1)
    return {
        "unit": u,
        "lm": k,
        "L": L,
        "x": x,
        "c": c,
        "n": lives.n[u],
        "Z": Z,
    }


def landmark(
    life: Any,
    at: npt.ArrayLike | float,
    horizon: float,
    interactions: bool = False,
) -> Any:
    """
    The landmark data of ``life`` at the landmark time(s) ``at``, as a
    :class:`~surpyval.SurpyvalData` (R ``dynpred``'s ``cutLM``).

    At a landmark ``L``, keep the units still at risk (entered by ``L`` and
    followed event-free past it), enter them at ``L`` (``tl = L``), cut
    their follow-up at ``L + horizon`` (an event after that, or a unit
    still running then, is right censored at ``L + horizon``), and give
    them their covariates as known at ``L``: the fixed ones, and the last
    value at or before ``L`` of the time-varying ones and of each
    measurement read on the units (``life.m``).
    Because only units at risk at ``L`` and only what was known at ``L``
    are used, a model of this data never borrows from the future: it has
    no immortal-time bias.

    With several landmarks the landmark data sets are stacked into one
    (the "super landmark data set" of van Houwelingen, 2007), in which a
    unit appears once for each landmark it is at risk at. The landmark is
    then a covariate column, ``L``, and with ``interactions=True`` so is
    each covariate times ``L`` (``"z*L"``): fit a Cox model to it
    (``CoxPH.fit(data)``, stratified by ``L`` or not) for a supermodel,
    with standard errors clustered on the unit, as each unit has several
    rows; :class:`~surpyval.LandmarkCox` does this and predicts from it.

    Parameters
    ----------
    life : SurpyvalData
        The units' lifetimes: time ``x``, ``c`` (0 the event, 1 still
        running), counts ``n``, entry ``tl``, unit ids ``i`` and
        covariates ``Z``. A covariate that changes is given as rows of
        the unit, ``(tl, x]`` stretches with ``Z`` fixed in each, only
        the last ending in the event (or as a ``StepSchedule`` or
        ``CovariatePath`` per unit, from ``from_items``); a gap between a
        unit's rows, or an event before its last row, is refused.
        Readings taken on the units are its measurements ``m``. The last
        reading of each measurement at or before ``L`` becomes a
        covariate named after it; a unit at
        risk with no reading of one by ``L`` is left out of that
        landmark, with a warning that counts them.
    at : float or array_like
        The landmark time(s) ``L``.
    horizon : float
        The window ``w``: follow-up is cut at ``L + w``.
    interactions : bool, optional
        With several landmarks, add each covariate times ``L`` as a
        column (``"z*L"``).

    Returns
    -------
    SurpyvalData
        One row per unit at risk at each landmark, ordered by landmark,
        with ``x``, ``c``, ``n``, ``tl`` (the landmark) and ``Z`` (named
        by ``covariate_names``). Each row's ``attributes`` hold its unit
        (``"i"``) and landmark (``"L"``); with one landmark the units are
        also its ``i``. (Stacked, a unit's rows overlap in time, each
        running from its landmark, which ``SurpyvalData`` does not allow
        for one unit's rows, so there the unit is in the attributes
        only.)

    Examples
    --------
    Three units with a fixed covariate; at the landmark 2 the first has
    failed already, and the follow-up of the others is cut at 2 + 3:

    >>> from surpyval import SurpyvalData, landmark
    >>> life = SurpyvalData(x=[1.5, 6, 4], c=[0, 0, 1], Z=[[1], [0], [1]],
    ...                     i=["a", "b", "c"])
    >>> data = landmark(life, at=2, horizon=3)
    >>> data.i, data.x, data.c, data.tl
    (['b', 'c'], array([5., 4.]), array([1, 1]), array([2., 2.]))

    Stacked over two landmarks, the landmark becomes a covariate:

    >>> data = landmark(life, at=[0, 2], horizon=3)
    >>> data.covariate_names
    ['Z0', 'L']
    >>> data.Z
    array([[1., 0.],
           [0., 0.],
           [1., 0.],
           [0., 2.],
           [1., 2.]])
    """
    from surpyval.utils.surpyval_data import SurpyvalData

    L_all, w = check_landmarks(at, horizon)
    lives = Lives(life)
    st = stack(lives, L_all, w)
    Z, names = st["Z"], list(lives.names)
    if L_all.size > 1:
        extra = [st["L"][:, None]]
        names.append("L")
        if interactions:
            extra.append(Z * st["L"][:, None])
            names += [f"{v}*L" for v in lives.names]
        Z = np.column_stack([Z] + extra)
    if st["unit"].size == 0:
        raise ValueError(
            "No unit is at risk at any of the landmarks {}: choose "
            "landmarks within the follow-up.".format(L_all.tolist())
        )
    unit_ids = [lives.ids[u] for u in st["unit"]]
    # One landmark: a unit is on one row, so its id is the row's i. Stacked,
    # a unit is on a row per landmark, and those rows overlap in time
    # (each runs from its landmark), which SurpyvalData refuses for one
    # unit's rows: the unit is then each row's attribute "i".
    data = SurpyvalData(
        x=st["x"],
        c=st["c"],
        n=st["n"],
        tl=st["L"],
        Z=Z if Z.shape[1] else None,
        i=unit_ids if L_all.size == 1 else None,
        covariate_names=names if Z.shape[1] else None,
        group_and_sort=False,
    )
    data.attributes = [
        {"i": u, "L": float(L)} for u, L in zip(unit_ids, st["L"])
    ]
    return data
