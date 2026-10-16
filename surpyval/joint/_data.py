"""The joint model's data: one row per unit (its event or censoring, its
truncation and its covariates) and one row per marker reading, checked
and gathered unit by unit (#891).

Every refusal names the argument and says how to put it right (principle
2). A unit is the whole of its rows, so a missing value anywhere in it is
refused rather than dropped (principle 3).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt


class JointData(NamedTuple):
    """The units and their readings, in the data's own units.

    ``ids`` are the units' ids, in the order of the other per-unit
    arrays: ``xl`` and ``xr`` the event time (equal) or the ends of its
    interval, ``c`` the censoring flag, ``tl`` and ``tr`` the
    truncation, ``Z`` the hazard and ``W`` the path covariates, and
    ``readings`` each unit's ``(ages, values)``, in order of age.
    """

    ids: list
    xl: npt.NDArray
    xr: npt.NDArray
    c: npt.NDArray
    tl: npt.NDArray
    tr: npt.NDArray
    Z: npt.NDArray
    W: npt.NDArray
    readings: list

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def n_readings(self) -> int:
        return int(sum(len(t) for t, _ in self.readings))


def _per_unit(name: str, value: Any, n: int, default: float) -> npt.NDArray:
    """A per-unit float array: a scalar for every unit, or one per unit."""
    if value is None:
        return np.full(n, default)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.ndim != 1 or arr.size != n:
        raise ValueError(
            f"{name} must be a number, or one value per unit (x has {n} "
            f"units); got shape {arr.shape}."
        )
    return arr


def _covariate_matrix(name: str, value: Any, n: int) -> npt.NDArray:
    """``(n, p)`` covariates: one row per unit (a 1-D array is one
    covariate)."""
    if value is None:
        return np.zeros((n, 0))
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"{name} must be numeric, one row per unit; code a category as "
            "indicator columns (fit_from_df does with a formula's C())."
        ) from None
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[0] != n:
        raise ValueError(
            f"{name} must have one row per unit ({n} units); got shape "
            f"{arr.shape}."
        )
    bad = ~np.isfinite(arr)
    if bad.any():
        rows = np.flatnonzero(bad.any(1))
        raise ValueError(
            f"{name} has missing or infinite values for {rows.size} unit(s) "
            f"(rows {rows[:5].tolist()}): a unit's covariates are part of "
            "the unit, so it cannot be dropped alone. Remove those units "
            "(and their readings), or fill in the values."
        )
    return arr


def _event_times(
    x: Any, c: Any, xl: Any, xr: Any
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """``(xl, xr, c)``: each unit's event time, or the ends of its
    interval, and its censoring flag."""
    if xl is not None or xr is not None:
        if x is not None:
            raise ValueError(
                "Give the event times as x, or as xl and xr, not both."
            )
        if xl is None or xr is None:
            raise ValueError("xl and xr must be given together.")
        lo = np.asarray(xl, dtype=float).reshape(-1)
        hi = np.asarray(xr, dtype=float).reshape(-1)
        if lo.size != hi.size:
            raise ValueError("xl and xr must have the same length.")
        if c is None:
            c = np.where(lo == hi, 0, 2)
    else:
        if x is None:
            raise ValueError(
                "x is required: each unit's failure or censoring time."
            )
        arr = np.asarray(x, dtype=float)
        if arr.ndim == 2 and arr.shape[1] == 2:
            lo, hi = arr[:, 0], arr[:, 1]
        elif arr.ndim <= 1:
            lo = hi = np.atleast_1d(arr)
        else:
            raise ValueError(
                "x must be one time per unit, or [lower, upper] pairs for "
                f"interval censored units; got shape {arr.shape}."
            )
    n = lo.size
    if n == 0:
        raise ValueError("x must not be empty: the model needs units.")
    c_arr = np.zeros(n, int) if c is None else np.asarray(c)
    c_arr = np.atleast_1d(c_arr)
    if c_arr.shape != (n,):
        raise ValueError(
            f"c must have one censoring flag per unit ({n}); got shape "
            f"{c_arr.shape}."
        )
    if not np.all(np.isin(c_arr, (-1, 0, 1, 2))):
        raise ValueError(
            "c must be 0 (failed at x), 1 (right censored), -1 (left "
            "censored) or 2 (interval censored); got "
            f"{sorted(set(np.asarray(c_arr).tolist()))}."
        )
    c_arr = c_arr.astype(int)
    if np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError(
            "x has missing (nan) times: a unit's time is part of the unit, "
            "so it cannot be dropped alone. Remove those units and their "
            "readings."
        )
    if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
        raise ValueError(
            "x must be finite: a unit still running at the end is right "
            "censored (c=1) at the age it was last seen."
        )
    interval = c_arr == 2
    if np.any(interval & ~(lo < hi)):
        raise ValueError(
            "An interval censored unit (c=2) needs a lower time below its "
            "upper one, [lower, upper]."
        )
    if np.any(~interval & (lo != hi)):
        raise ValueError(
            "Only an interval censored unit (c=2) may have a [lower, upper] "
            "pair as its time; set c=2 for those units."
        )
    if np.any(hi <= 0) or np.any(lo < 0):
        raise ValueError(
            "x must be positive: the Weibull baseline lives on (0, inf), and "
            "an age is measured from 0."
        )
    return lo, hi, c_arr


def _ids(i: Any, n: int) -> list:
    if i is None:
        return list(range(n))
    ids = np.asarray(i, dtype=object).reshape(-1).tolist()
    if len(ids) != n:
        raise ValueError(
            f"i must have one id per unit ({n} units); got {len(ids)}."
        )
    ids = [v.item() if hasattr(v, "item") else v for v in ids]
    seen: set = set()
    for v in ids:
        if v in seen:
            raise ValueError(
                f"i has the id {v!r} more than once: each row of x is one "
                "unit, and its readings are linked to it by marker_i."
            )
        seen.add(v)
    return ids


def _readings(marker_x: Any, marker_y: Any, marker_i: Any, ids: list) -> list:
    """Each unit's readings, ``(ages, values)`` in order of age."""
    if marker_x is None or marker_y is None or marker_i is None:
        raise ValueError(
            "marker_x, marker_y and marker_i are required: the ages, values "
            "and unit ids of the marker readings. With no marker, fit a "
            "regression such as surpyval.WeibullPH instead."
        )
    t = np.asarray(marker_x, dtype=float).reshape(-1)
    y = np.asarray(marker_y, dtype=float).reshape(-1)
    who = np.asarray(marker_i, dtype=object).reshape(-1)
    if not (t.size == y.size == who.size):
        raise ValueError(
            "marker_x, marker_y and marker_i must have the same length; got "
            f"{t.size}, {y.size} and {who.size}."
        )
    if t.size == 0:
        raise ValueError(
            "There are no marker readings: the joint model needs at least "
            "some. With none, fit a regression such as surpyval.WeibullPH."
        )
    for name, arr in (("marker_x", t), ("marker_y", y)):
        if not np.isfinite(arr).all():
            raise ValueError(
                f"{name} has {int((~np.isfinite(arr)).sum())} missing or "
                "infinite value(s): a reading is part of its unit's path, "
                "so it is not dropped silently. Remove those readings first "
                "(df.dropna(subset=[...]))."
            )
    if np.any(t < 0):
        raise ValueError(
            "marker_x must not be negative: it is the age of each reading."
        )
    index = {v: k for k, v in enumerate(ids)}
    pos = np.empty(t.size, dtype=int)
    for r, v in enumerate(who.tolist()):
        v = v.item() if hasattr(v, "item") else v
        if v not in index:
            raise ValueError(
                f"marker_i has the id {v!r}, which is not a unit of i: "
                "every reading must belong to a unit."
            )
        pos[r] = index[v]
    out = []
    for u in range(len(ids)):
        rows = np.flatnonzero(pos == u)
        order = np.argsort(t[rows], kind="stable")
        out.append((t[rows][order], y[rows][order]))
    return out


def _check_reading_ages(readings: list, ids: list, xr: npt.NDArray) -> None:
    late = [
        ids[u] for u, (t, _) in enumerate(readings) if t.size and t[-1] > xr[u]
    ]
    if late:
        raise ValueError(
            f"{len(late)} unit(s) have readings after their failure or "
            f"censoring time (ids {late[:5]}): a reading is taken while "
            "the unit is under observation, at or before x (the upper end "
            "of an interval)."
        )


def _truncation(
    tl: Any, tr: Any, xl: npt.NDArray, xr: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    n = xl.size
    lo = _per_unit("tl", tl, n, 0.0)
    hi = _per_unit("tr", tr, n, np.inf)
    lo = np.where(np.isneginf(lo), 0.0, lo)
    if np.isnan(lo).any() or np.isnan(hi).any():
        raise ValueError("tl and tr must not be missing (nan).")
    if np.any(lo < 0):
        raise ValueError(
            "tl must not be negative: it is the age at which a unit came "
            "under observation (0, the default, is no left truncation)."
        )
    if np.any(xl < lo) | np.any((xl == lo) & (lo > 0) & (xl == xr)):
        raise ValueError(
            "Each unit's time must be after its left truncation tl: a unit "
            "is only in the data because it survived to tl."
        )
    if np.any(xr > hi):
        raise ValueError(
            "Each unit's time must be at or before its right truncation tr."
        )
    return lo, hi


def joint_data(
    x: Any,
    c: Any,
    Z: Any,
    i: Any,
    tl: Any,
    tr: Any,
    xl: Any,
    xr: Any,
    marker_x: Any,
    marker_y: Any,
    marker_i: Any,
    path_Z: Any,
) -> JointData:
    """Check the arguments of ``JointModel.fit`` and gather them."""
    lo, hi, c_arr = _event_times(x, c, xl, xr)
    n = lo.size
    ids = _ids(i, n)
    Zm = _covariate_matrix("Z", Z, n)
    Wm = _covariate_matrix("path_Z", path_Z, n)
    t_lo, t_hi = _truncation(tl, tr, lo, hi)
    readings = _readings(marker_x, marker_y, marker_i, ids)
    _check_reading_ages(readings, ids, hi)
    if not np.any(c_arr != 1):
        raise ValueError(
            "Every unit is right censored: with no failures the survival "
            "part has nothing to estimate (its likelihood has no finite "
            "maximum). Model the readings alone, with "
            "surpyval.degradation.DegradationAnalysis, until failures are "
            "seen."
        )
    return JointData(ids, lo, hi, c_arr, t_lo, t_hi, Zm, Wm, readings)
