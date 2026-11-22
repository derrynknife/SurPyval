"""Operations on whole datasets: combining (:func:`concat`), comparing
(:func:`same_data`), moving to another clock (:func:`rescale_time`) and
drawing a bootstrap sample of the units (:func:`resample`). The public
names are the methods of :class:`~surpyval.utils.surpyval_data.SurpyvalData`
(``SurpyvalData.concat``, ``data.equals``, ``data.rescale_time``,
``data.resample``); ``EventHistoryData`` shares the comparison only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.utils.warnings import (
    MaskedCauseWarning,
    ignoring_warnings,
    named,
)

# Comparing ------------------------------------------------------------

# What makes two datasets the same data. The arrays split from them for
# the likelihoods (x_o, ...) follow from these, and where a terminal flag
# was inferred (a_inferred) is how it was given, not what it is.
LIFETIME_FIELDS = (
    "x",
    "c",
    "n",
    "t",
    "w",
    "Z",
    "covariate_names",
    "e",
    "s",
    "a",
    "ids",
    "attributes",
    "time_varying_covariates",
)
HISTORY_FIELDS = (
    "x",
    "i",
    "c",
    "n",
    "tl",
    "tr",
    "e",
    "Z",
    "covariate_names",
    "attributes",
    "time_varying_covariates",
    "states",
    "window_map",
    "observation_windows",
)


def _close(a: Any, b: Any, rtol: float) -> bool:
    """Whether two values are the same, numbers to the relative tolerance
    ``rtol`` (blanks, NaN, equal to each other; infinities to themselves),
    containers entry by entry."""
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            return False
        if a.dtype.kind in "biuf" and b.dtype.kind in "biuf":
            a, b = a.astype(float), b.astype(float)
            if rtol == 0:
                return bool(np.array_equal(a, b, equal_nan=True))
            return bool(
                np.isclose(a, b, rtol=rtol, atol=0, equal_nan=True).all()
            )
        return all(_close(u, v, rtol) for u, v in zip(a.ravel(), b.ravel()))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(
            _close(a[k], b[k], rtol) for k in a
        )
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(
            _close(u, v, rtol) for u, v in zip(a, b)
        )
    numbers = (int, float, np.integer, np.floating)
    if (
        isinstance(a, numbers)
        and isinstance(b, numbers)
        and not isinstance(a, (bool, np.bool_))
        and not isinstance(b, (bool, np.bool_))
    ):
        a, b = float(a), float(b)
        if np.isnan(a) or np.isnan(b):
            return bool(np.isnan(a) and np.isnan(b))
        if rtol == 0 or not (np.isfinite(a) and np.isfinite(b)):
            return a == b
        return bool(np.isclose(a, b, rtol=rtol, atol=0))
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if isinstance(a, StepSchedule) and isinstance(b, StepSchedule):
        return (
            _close(a.period, b.period, rtol)
            and _close(a.edges, b.edges, rtol)
            and _close(a.Z, b.Z, rtol)
        )
    return bool(a == b)


def _same_measurements(a: Any, b: Any, rtol: float) -> bool:
    from surpyval.utils.units import same_unit

    if a is None or b is None:
        return a is None and b is None
    return (
        list(a._store) == list(b._store)
        and _close(a._store, b._store, rtol)
        and all(same_unit(a._units.get(k), b._units.get(k)) for k in a._store)
    )


def same_data(a: Any, b: Any, fields: tuple, rtol: float) -> bool:
    """Whether ``a`` and ``b`` (of one class) hold the same data: every
    one of ``fields``, the unit and the measurements."""
    from surpyval.utils.units import same_unit

    if type(a) is not type(b):
        return False
    rtol = _checked_rtol(rtol)
    if not same_unit(getattr(a, "unit", None), getattr(b, "unit", None)):
        return False
    if not _same_measurements(
        getattr(a, "m", None), getattr(b, "m", None), rtol
    ):
        return False
    return all(
        _close(getattr(a, f, None), getattr(b, f, None), rtol) for f in fields
    )


def _checked_rtol(rtol: Any) -> float:
    if (
        isinstance(rtol, (bool, np.bool_))
        or not isinstance(rtol, (int, float, np.integer, np.floating))
        or not np.isfinite(float(rtol))
        or float(rtol) < 0
    ):
        raise ValueError(
            "'rtol' is the relative tolerance numbers are compared to, a "
            f"finite number at or above 0 (0 for exactly equal); got {rtol!r}."
        )
    return float(rtol)


# Combining ------------------------------------------------------------


def _label(k: int) -> str:
    return f"dataset {k}"


def _refuse_mixed(datasets: list, has: Any, what: str, fix: str) -> None:
    """Refuse where some datasets have ``what`` and some do not."""
    flags = [bool(has(d)) for d in datasets]
    if any(flags) and not all(flags):
        with_ = [str(k) for k, f in enumerate(flags) if f]
        without = [str(k) for k, f in enumerate(flags) if not f]
        raise ValueError(
            f"Dataset{'s' if len(with_) > 1 else ''} {', '.join(with_)} "
            f"{'have' if len(with_) > 1 else 'has'} {what} and dataset"
            f"{'s' if len(without) > 1 else ''} {', '.join(without)} "
            f"{'have' if len(without) > 1 else 'has'} none: {fix}"
        )


def _check_units(datasets: list) -> Any:
    from surpyval.utils.units import same_unit

    first = datasets[0].unit
    for k, data in enumerate(datasets[1:], start=1):
        if same_unit(first, data.unit):
            continue
        if first is None or data.unit is None:
            known, blank = (0, k) if data.unit is None else (k, 0)
            unit = datasets[known].unit
            raise ValueError(
                f"{_label(known).capitalize()} is in {unit!r} and "
                f"{_label(blank)} has no unit: if it is in {unit!r} too, "
                f"label it with data.rescale_time(1, unit={unit!r}); "
                "otherwise put it on the same unit first, "
                "data.rescale_time(factor, unit=...)."
            )
        raise ValueError(
            f"Dataset 0 is in {first!r} and {_label(k)} in {data.unit!r}: "
            "put them on one unit first, data.rescale_time(factor, "
            f"unit={first!r}) (factor is the number of {data.unit!r} in "
            f"one {first!r})."
        )
    return first


def _check_covariates(datasets: list) -> Any:
    _refuse_mixed(
        datasets,
        lambda d: d.Z is not None,
        "covariates",
        "give every dataset the same covariates, or leave them out with "
        "data.without('covariates').",
    )
    first = datasets[0]
    if first.Z is None:
        return None
    width = _width(first.Z)
    for k, data in enumerate(datasets[1:], start=1):
        names, other = first.covariate_names, data.covariate_names
        if names != other or _width(data.Z) != width:
            raise ValueError(
                "The covariates differ: dataset 0 has "
                f"{_shown_covariates(first)} and {_label(k)} "
                f"{_shown_covariates(data)}. Give them the same "
                "covariates in the same order (data.with_covariates([...]) "
                "picks and orders them by name)."
            )
    return first.covariate_names


def _shown_covariates(data: Any) -> Any:
    if data.covariate_names is not None:
        return list(data.covariate_names)
    return f"{_width(data.Z)} unnamed column(s)"


def _width(Z: Any) -> int:
    Z = np.asarray(Z)
    return 1 if Z.ndim == 1 else Z.shape[1]


def _check_ids(datasets: list) -> None:
    _refuse_mixed(
        datasets,
        lambda d: d.ids is not None,
        "item ids (i)",
        "give every dataset its ids (i= or ids=, i_col= in from_dataframe), "
        "or build each without them.",
    )
    if datasets[0].ids is None:
        return
    seen: dict = {}
    for k, data in enumerate(datasets):
        units = set(data.ids)
        m = data.m
        if m is not None:
            units |= set(m.ids)
        for unit in units:
            seen.setdefault(unit, []).append(k)
    shared = [unit for unit, where in seen.items() if len(where) > 1]
    if shared:
        where = sorted({k for unit in shared for k in seen[unit]})
        raise ValueError(
            f"{len(shared)} item id{'s are' if len(shared) > 1 else ' is'} "
            f"in more than one dataset ({named(shared)}; in datasets "
            + ", ".join(map(str, where))
            + "). An id is one item, so the combined data would join "
            "their rows into one history: rename the ids first so each "
            "item is in one dataset (prefix them with the source, say), "
            "or, for the same items seen again, combine their records "
            "with data.update(...)."
        )


def _check_event_types(datasets: list) -> None:
    typed = [d.e is not None for d in datasets]
    if not any(typed) or all(typed):
        return
    untyped = [
        k
        for k, d in enumerate(datasets)
        if d.e is None and (np.asarray(d.c) != 1).any()
    ]
    if untyped:
        raise ValueError(
            "Some datasets have event types (e) and dataset"
            f"{'s' if len(untyped) > 1 else ''} "
            + ", ".join(map(str, untyped))
            + f" {'have' if len(untyped) > 1 else 'has'} failures without "
            "them, which would become failures of unknown type: give those "
            "failures their types (e=), or leave the types out of every "
            "dataset with data.without('event_types')."
        )


def _rows(data: Any) -> np.ndarray:
    x = np.asarray(data.x, dtype=float)
    return x if x.ndim == 2 else np.column_stack([x, x])


def _per_dataset(datasets: list, get: Any, blank: Any) -> list | None:
    """The values of a per-row field over every dataset, ``blank`` on the
    rows of a dataset without it; None where none has it."""
    if all(get(d) is None for d in datasets):
        return None
    out: list = []
    for d in datasets:
        value = get(d)
        out.extend([blank] * len(d.c) if value is None else list(value))
    return out


def _terminal(data: Any) -> list | None:
    """The terminal flags as given: blank where they were inferred, so
    the combined data infers them again (and marks them so)."""
    if data.a is None:
        return None
    inferred = data.a_inferred
    return [
        None if inferred is not None and inferred[r] else int(v)
        for r, v in enumerate(data.a)
    ]


def _keyed(datasets: list, get: Any, by_id: bool) -> dict | None:
    """A per-unit mapping (time-varying covariates, plain markers) over
    every dataset: by id, or, without ids, by row numbered on through the
    combined rows."""
    if all(get(d) is None for d in datasets):
        return None
    out: dict = {}
    offset = 0
    for d in datasets:
        value = get(d) or {}
        for key, item in value.items():
            out[key if by_id else offset + int(key)] = item
        offset += len(d.c)
    return out


def _measurements(datasets: list, by_id: bool) -> Any:
    from surpyval.utils.measurements import Measurements
    from surpyval.utils.units import same_unit

    held = [(k, d.m) for k, d in enumerate(datasets) if d.m is not None]
    if not held:
        return None
    names: list = []
    units: dict = {}
    for k, m in held:
        for name in m._store:
            if name not in names:
                names.append(name)
            unit = m._units.get(name)
            if name in units and not same_unit(units[name][1], unit):
                first, other = units[name]
                raise ValueError(
                    f"The measurement {name!r} is in {other!r} in dataset "
                    f"{first} and in {unit!r} in {_label(k)}: give it one "
                    "unit in both before combining them."
                )
            units.setdefault(name, (k, unit))
    if None in names and len(names) > 1:
        raise ValueError(
            "Some datasets have one unnamed measurement (markers) and some "
            "named ones: name the unnamed readings (m={name: readings}) so "
            "each reading is of a measurement the others know."
        )
    store: dict = {name: {} for name in names}
    offsets = np.cumsum([0] + [len(d.c) for d in datasets])
    for k, m in held:
        for name, readings in m._store.items():
            for key, values in readings.items():
                store[name][key if by_id else offsets[k] + int(key)] = dict(
                    values
                )
    return Measurements._raw(
        store,
        {n: u for n, (_, u) in units.items() if u is not None},
    )


def concat(datasets: Any) -> Any:
    """The rows of every dataset of ``datasets`` as one dataset; see
    :meth:`SurpyvalData.concat`."""
    from surpyval.utils.surpyval_data import SurpyvalData, _object_array

    if isinstance(datasets, SurpyvalData):
        datasets = [datasets]
    datasets = list(datasets)
    if not datasets:
        raise ValueError("concat needs at least one dataset; got none.")
    other = next(
        (d for d in datasets if not isinstance(d, SurpyvalData)), None
    )
    if other is not None:
        from surpyval.utils.event_history_data import EventHistoryData

        hint = (
            " An EventHistoryData is turned into lifetime rows with "
            "to_first_events() or to_gaps()."
            if isinstance(other, EventHistoryData)
            else ""
        )
        raise TypeError(
            "concat combines SurpyvalData datasets; got "
            f"{type(other).__name__}.{hint}"
        )
    unit = _check_units(datasets)
    names = _check_covariates(datasets)
    _check_ids(datasets)
    _refuse_mixed(
        datasets,
        lambda d: d.s is not None,
        "states (s)",
        "give every dataset its states, or leave them out with "
        "data.without('states').",
    )
    _refuse_mixed(
        datasets,
        lambda d: d.time_varying_covariates is not None,
        "time-varying covariates",
        "give every item its schedule, or leave them out with "
        "data.without('time_varying_covariates').",
    )
    _check_event_types(datasets)

    by_id = datasets[0].ids is not None
    x = np.vstack([_rows(d) for d in datasets])
    c = np.concatenate([np.asarray(d.c) for d in datasets])
    if not (c == 2).any():
        x = x[:, 0]
    n = np.concatenate([np.asarray(d.n) for d in datasets])
    tl = np.concatenate([np.asarray(d.tl, dtype=float) for d in datasets])
    tr = np.concatenate([np.asarray(d.tr, dtype=float) for d in datasets])
    truncated = np.isfinite(tl).any() or np.isfinite(tr).any()
    Z = None
    if datasets[0].Z is not None:
        Z = np.vstack(
            [
                np.asarray(d.Z, dtype=float).reshape(len(d.c), -1)
                for d in datasets
            ]
        )
    w = None
    if any(d.w is not None for d in datasets):
        # Unweighted rows stand for themselves: weight 1
        w = np.concatenate(
            [
                np.ones(len(d.c)) if d.w is None else np.asarray(d.w)
                for d in datasets
            ]
        )
    e = _per_dataset(datasets, lambda d: d.e, None)
    s = _per_dataset(datasets, lambda d: d.s, None)
    ids = _per_dataset(datasets, lambda d: d.ids, None)
    a: list | None = None
    if any(d.a is not None for d in datasets):
        a = []
        for d in datasets:
            flags = _terminal(d)
            a.extend([None] * len(d.c) if flags is None else flags)
    attributes = _per_dataset(datasets, lambda d: d.attributes, None)
    time_varying = _keyed(datasets, lambda d: d.time_varying_covariates, by_id)
    m = _measurements(datasets, by_id)
    # Rows kept by something with no place in the constructor stay in
    # order, so it can be put back by row
    keep_order = attributes is not None or (
        not by_id and (time_varying is not None or m is not None)
    )
    with ignoring_warnings(MaskedCauseWarning):
        # Warned of when each dataset was built
        out = SurpyvalData(
            x,
            c,
            n,
            tl=tl if truncated else None,
            tr=tr if truncated else None,
            Z=Z,
            covariate_names=names,
            e=None if e is None else _object_array(e),
            s=s,
            ids=ids,
            a=a,
            w=w,
            unit=unit,
            group_and_sort=not keep_order,
        )
    if attributes is not None:
        out.attributes = attributes
    if time_varying is not None:
        out.time_varying_covariates = time_varying
    if m is not None:
        out._m = m
    return out


# Another clock --------------------------------------------------------


def _rescaled_schedule(schedule: Any, factor: float, unit: Any) -> Any:
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if not isinstance(schedule, StepSchedule):
        raise ValueError(
            f"The time-varying covariates of item {unit!r} are a "
            f"{type(schedule).__name__}, a function of time that cannot be "
            "moved to another clock from outside: build it on the new "
            "clock, or leave the time-varying covariates out with "
            "data.without('time_varying_covariates') first."
        )
    return StepSchedule(
        schedule.edges / factor,
        schedule.Z,
        None if schedule.period is None else schedule.period / factor,
    )


def rescale_time(data: Any, factor: Any, unit: Any = None) -> Any:
    """``data`` with every time divided by ``factor``; see
    :meth:`SurpyvalData.rescale_time`."""
    from surpyval.univariate.parametric._rescale_time import checked_factor
    from surpyval.utils.measurements import Measurements
    from surpyval.utils.units import normalise_unit

    d = checked_factor(factor)
    rows = np.arange(len(data.c))
    x = np.asarray(data.x, dtype=float) / d
    t = np.asarray(data.t, dtype=float) / d
    out = data._take(rows, x=x, t=t)
    if unit is not None:
        out.unit = normalise_unit(unit)
    if data.time_varying_covariates is not None:
        out.time_varying_covariates = {
            key: _rescaled_schedule(schedule, d, key)
            for key, schedule in data.time_varying_covariates.items()
        }
    m = data.m
    if m is not None:
        out._m = Measurements._raw(
            {
                name: {
                    key: {float(age) / d: v for age, v in values.items()}
                    for key, values in readings.items()
                }
                for name, readings in m._store.items()
            },
            m._units,
        )
    return out


# Bootstrap ------------------------------------------------------------


def resample(data: Any, random_state: Any = None, size: Any = None) -> Any:
    """A bootstrap sample of the units of ``data``; see
    :meth:`SurpyvalData.resample`."""
    from surpyval.utils.rng import as_generator

    n = np.asarray(data.n, dtype=int)
    if data.ids is None:
        groups = [np.array([r]) for r in range(len(n))]
    else:
        rows_of: dict = {}
        for r, unit in enumerate(data.ids):
            rows_of.setdefault(unit, []).append(r)
        groups = [np.asarray(rows) for rows in rows_of.values()]
        uneven = [
            unit
            for unit, rows in rows_of.items()
            if len(set(n[rows].tolist())) > 1
        ]
        if uneven:
            raise ValueError(
                f"The rows of {len(uneven)} unit"
                f"{'s' if len(uneven) > 1 else ''} ({named(uneven)}) have "
                "different counts n, so how many units each id stands for "
                "is not clear: give every row of a unit the same count."
            )
    counts = np.array([n[rows[0]] for rows in groups])
    units = int(counts.sum())
    if size is None:
        size = units
    if (
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (int, np.integer))
        or size < 1
    ):
        raise ValueError(
            "'size' is how many units to draw, a whole number above 0 (the "
            f"number in the data, {units}, by default); got {size!r}."
        )
    rng = as_generator(random_state)
    # Each unit equally likely: a group of k identical units k times as
    # likely as one unit
    drawn = rng.choice(len(groups), size=int(size), p=counts / units)
    times = np.bincount(drawn, minlength=len(groups))
    picked = np.flatnonzero(times)
    rows = np.concatenate([groups[g] for g in picked])
    new_n = np.concatenate(
        [np.full(len(groups[g]), times[g], dtype=int) for g in picked]
    )
    return data._take(rows, n=new_n)
