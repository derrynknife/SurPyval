"""Dates to durations: :func:`dates_to_durations`.

Field records are dates: when each unit started, when (if ever) it ended.
SurPyval's models work on durations as plain numbers in a unit the user
chooses, and refuse dates (``refuse_time_values``). This turns the one
into the other, with the censoring flags the fits take.

pandas is imported inside the function, as everywhere in the package
(``import surpyval`` does not load it).
"""

from __future__ import annotations

import datetime as _dt
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["dates_to_durations"]

# The length of each unit, in seconds. A year is the Julian year of
# 365.25 days, so that a duration in years is the duration in days over
# 365.25 whatever the calendar years it crosses.
_SECONDS = {
    "s": 1.0,
    "min": 60.0,
    "h": 3600.0,
    "D": 86400.0,
    "W": 7 * 86400.0,
    "Y": 365.25 * 86400.0,
}
_ALIASES = {
    "second": "s",
    "seconds": "s",
    "sec": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "hr": "h",
    "day": "D",
    "days": "D",
    "d": "D",
    "week": "W",
    "weeks": "W",
    "w": "W",
    "year": "Y",
    "years": "Y",
    "yr": "Y",
    "y": "Y",
}


def _unit_seconds(unit: str) -> float:
    key = _ALIASES.get(unit, unit) if isinstance(unit, str) else None
    if key not in _SECONDS:
        raise ValueError(
            f"unit={unit!r} is not one of 's', 'min', 'h', 'D' (the "
            "default), 'W' or 'Y' (or their names: 'hours', 'days', "
            "'weeks', 'years'). Months have no fixed length; use 'D' and "
            "divide by the length you mean."
        )
    return _SECONDS[key]


def _is_date_like(v: Any) -> bool:
    # NaT is a datetime.date subclass in pandas; None is a missing date
    return v is None or isinstance(v, (_dt.date, np.datetime64))


def _as_dates(values: Any, name: str) -> Any:
    """``values`` as a pandas ``DatetimeIndex`` (naive or time-zone
    aware), refusing anything that is not a date."""
    import pandas as pd

    if isinstance(values, (pd.Series, pd.Index)):
        values = values.to_numpy() if values.dtype.kind == "M" else values
        if isinstance(values, (pd.Series, pd.Index)):
            values = list(values)
    elif isinstance(values, (_dt.date, np.datetime64)):
        values = [values]
    arr = values if isinstance(values, np.ndarray) else list(values)
    if isinstance(arr, np.ndarray) and arr.dtype.kind == "M":
        return pd.DatetimeIndex(arr.ravel())
    if not all(_is_date_like(v) for v in np.asarray(arr, dtype=object).flat):
        raise ValueError(
            f"'{name}' must hold dates (datetime64, pandas Timestamp or "
            "datetime), with a blank (NaT or None) where a date is "
            "missing. Convert text first, saying how it is written: "
            f"pd.to_datetime({name}, format='%d/%m/%Y')."
        )
    try:
        return pd.DatetimeIndex(list(np.asarray(arr, dtype=object).flat))
    except (TypeError, ValueError) as error:
        if "tz" in str(error).lower() or "naive" in str(error).lower():
            raise ValueError(
                f"'{name}' mixes time-zone aware and naive dates. Give "
                "every date a time zone (.dt.tz_localize('UTC') for "
                "naive ones, or the zone they were written in) or none."
            ) from error
        raise


def _aware(index: Any) -> bool:
    return index.tz is not None


def _naive_or_utc(index: Any) -> Any:
    """Elapsed time between aware dates is real elapsed time (a clock
    change is not a missing hour), so they are compared in UTC."""
    return index.tz_convert("UTC") if _aware(index) else index


def dates_to_durations(
    start: npt.ArrayLike,
    end: npt.ArrayLike | None = None,
    cutoff: Any = None,
    unit: str = "D",
    event: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """Durations and censoring flags from start and end dates.

    In words: each unit's duration is the time from its start date to its
    end date. A unit with no end date is still running, so all that is
    known is that it lasted at least until the cut-off, the date the
    records stop; it is right censored there. The cut-off is the latest
    date in the records unless you give one. A unit that ended after the
    cut-off is also censored at it, since the records would not have
    shown the end.

    Parameters
    ----------
    start : array-like of dates
        When each unit started (``datetime64``, pandas ``Timestamp``,
        ``datetime``). No date may be missing.
    end : array-like of dates, optional
        When each unit ended. A blank (``NaT`` or ``None``) means it is
        still running. Omitted: every unit is still running.
    cutoff : date, optional
        The date the records stop. Default: the latest date seen, of the
        starts and the ends.
    unit : str, default "D"
        The unit of the durations: ``"s"``, ``"min"``, ``"h"`` (hours),
        ``"D"`` (days), ``"W"`` (weeks) or ``"Y"`` (years); their names
        (``"hours"``, ``"days"``, ...) work too. All are fixed lengths,
        and a year is 365.25 days. A month has no fixed length and is not
        offered.
    event : array-like of bool, optional
        Whether each unit's end date is a failure. Where it is false, the
        unit ended for another reason (a planned replacement, a sale) and
        is right censored at its end date. Default: every end date is a
        failure.

    Returns
    -------
    x : numpy.ndarray of float
        The durations, in ``unit``, not rounded.
    c : numpy.ndarray of int
        The censoring flags: 0 where the unit's failure was observed, 1
        where it is right censored (see the Conventions page).

    Raises
    ------
    ValueError
        If a start date is missing; if the dates mix time-zone aware and
        naive ones; if an end date or the cut-off is before its start
        (the count is given); or if ``unit`` is not one of those above.

    Notes
    -----
    Time-zone aware dates are compared in UTC, so the duration is the
    elapsed time whatever clock changes lie between. Dates without a time
    zone are compared as written. Give both start and end the same kind;
    a mix is refused.

    lifelines' ``datetimes_to_durations`` takes the same inputs (its
    ``fill_date`` is this ``cutoff``, and its ``E`` is ``1 - c``) and
    gives the same flags and, for whole days, the same durations; it
    returns whole units, truncating a part of a unit, where this returns
    the exact fraction.

    Examples
    --------
    >>> import pandas as pd
    >>> from surpyval.utils import dates_to_durations
    >>> start = pd.to_datetime(["2023-01-01", "2023-01-05", "2023-02-01"])
    >>> end = pd.to_datetime(["2023-03-01", None, "2023-04-11"])
    >>> x, c = dates_to_durations(start, end)
    >>> x
    array([59., 96., 69.])
    >>> c
    array([0, 1, 0])
    """
    import pandas as pd

    seconds = _unit_seconds(unit)
    s = _as_dates(start, "start")
    n = len(s)
    if s.isna().any():
        raise ValueError(
            f"'start' has {int(s.isna().sum())} missing date(s); a unit "
            "with no start date has no duration. Drop those rows first."
        )
    e = (
        pd.DatetimeIndex([pd.NaT] * n)
        if end is None
        else _as_dates(end, "end")
    )
    if len(e) != n:
        raise ValueError(
            f"'start' has {n} dates and 'end' has {len(e)}; give one end "
            "date (or a blank) for each start date."
        )
    ended = ~np.asarray(e.isna())
    if event is None:
        failed = ended.copy()
    else:
        flags = np.asarray(event)
        if flags.shape != (n,):
            raise ValueError(
                f"'event' must have one entry for each of the {n} units; "
                f"got shape {flags.shape}."
            )
        failed = ended & flags.astype(bool)

    if cutoff is not None:
        cut = _as_dates([cutoff], "cutoff")
        if cut.isna().any():
            raise ValueError("'cutoff' is a missing date; give a date.")
    kinds = {_aware(s)} | ({_aware(e)} if ended.any() else set())
    if cutoff is not None:
        kinds.add(_aware(cut))
    if len(kinds) > 1:
        raise ValueError(
            "The dates mix time-zone aware and naive ones (start, end and "
            "cutoff must all be the same). Give every date a time zone "
            "(.dt.tz_localize('UTC') for naive ones, or the zone they "
            "were written in) or none (.dt.tz_localize(None))."
        )
    s = _naive_or_utc(s)
    e = _naive_or_utc(e)
    if cutoff is None:
        seen = [s.max()] + ([e[ended].max()] if ended.any() else [])
        stop = max(seen)
    else:
        stop = _naive_or_utc(cut)[0]

    def to_ns(index: Any) -> npt.NDArray[np.int64]:
        # The same resolution for every input, whatever pandas picked
        return np.asarray(
            index.as_unit("ns").asi8 if hasattr(index, "as_unit") else index
        ).astype(np.int64)

    s_ns = to_ns(s)
    stop_ns = to_ns(pd.DatetimeIndex([stop]))[0]
    e_ns = np.where(ended, to_ns(e.fillna(stop)), stop_ns)

    if (e_ns[ended] < s_ns[ended]).any():
        k = int((e_ns[ended] < s_ns[ended]).sum())
        first = int(np.flatnonzero(ended & (e_ns < s_ns))[0])
        raise ValueError(
            f"{k} unit(s) end before they start (the first is row "
            f"{first}); check that start and end are not swapped or "
            "written in different formats."
        )
    if (s_ns > stop_ns).any():
        k = int((s_ns > stop_ns).sum())
        raise ValueError(
            f"{k} unit(s) start after the cutoff, so they have no "
            "duration within the records; drop them or move the cutoff."
        )
    # an end after the cutoff is not seen: censored at the cutoff
    seen_end = ended & (e_ns <= stop_ns)
    x_ns = np.where(seen_end, e_ns, stop_ns) - s_ns
    c = np.where(failed & seen_end, 0, 1).astype(np.int64)
    return x_ns / 1e9 / seconds, c
