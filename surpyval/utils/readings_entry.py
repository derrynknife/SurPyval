"""Readings as the input of the fits that take repeated readings per unit:
a :class:`~surpyval.utils.measurements.Measurements`, or a
:class:`~surpyval.utils.surpyval_data.SurpyvalData` that holds them as
``data.m``.

- The degradation fitters (``DegradationAnalysis``, ``WienerProcess``,
  ``GammaProcess``) take them as their first argument, in place of the
  arrays ``x``, ``y`` and ``i`` (:func:`takes_readings`).
- The joint models take them as their ``markers`` (or ``m``), matched to
  the life rows by unit (:func:`takes_joint_readings`,
  :func:`markers_by_row`).
"""

from __future__ import annotations

import functools
import inspect
import warnings
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])


def _plural(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def one_measurement(m: Any, what: str, data: bool = False) -> dict:
    """The one measurement's readings ``{i: {x: value}}`` of ``m``;
    refused where ``m`` holds several, naming them and how to keep one
    (``data.with_measurement`` where they came from a SurpyvalData,
    ``m.with_measurement`` otherwise)."""
    names = m.names
    if len(names) > 1:
        keep = (
            "data.with_measurement(name), or leave them out with "
            'data.without("m")'
            if data
            else "m.with_measurement(name)"
        )
        raise ValueError(
            f"{what} takes one measurement, and the readings hold "
            f"{len(names)} ({', '.join(map(str, names))}): keep one with "
            f"{keep}."
        )
    return m[names[0]]


def _id_array(ids: list) -> np.ndarray:
    """The units as a one-dimensional array, one per reading (an object
    array where an id is a tuple, which numpy would otherwise spread
    over a second dimension)."""
    if any(isinstance(k, tuple) for k in ids):
        out = np.empty(len(ids), dtype=object)
        out[:] = ids
        return out
    return np.asarray(ids)


def _warn_no_readings(units: list) -> None:
    from surpyval.utils.warnings import caller_stacklevel, named

    if not units:
        return
    warnings.warn(
        f"{_plural(len(units), 'unit')} "
        + ("has" if len(units) == 1 else "have")
        + f" no readings: {named(units)}. "
        + ("It is" if len(units) == 1 else "They are")
        + " left out of the fit, which fits each unit's readings; check "
        "the ids match (text '1' and the number 1 differ).",
        UserWarning,
        stacklevel=caller_stacklevel(),
    )


def _refuse_lost(data: Any) -> None:
    """Refuse what a degradation fit has no place for: counts, weights,
    truncation, event types, time-varying covariates, states and units
    of several rows (the readings are per unit)."""
    from surpyval.utils.surpyval_data import (
        _distinct_states,
        lost_message,
        several_rows_clause,
    )

    lost = []
    if np.any(np.asarray(data.n) != 1):
        lost.append("counts")
    if data.w is not None:
        lost.append("weights")
    if np.isfinite(np.asarray(data.tl, dtype=float)).any():
        lost.append("left truncation")
    if np.isfinite(np.asarray(data.tr, dtype=float)).any():
        lost.append("right truncation")
    if data.e is not None:
        lost.append("event types")
    if data.time_varying_covariates is not None:
        lost.append("time-varying covariates")
    if len(_distinct_states(data)) > 1:
        lost.append("states")
    several = several_rows_clause(data)
    if several is not None:
        lost.append(several)
    if lost:
        raise ValueError(lost_message(lost, False))


def degradation_readings(source: Any, what: str) -> tuple:
    """``(x, y, i, Z, names)`` of a degradation fit from ``source``, a
    Measurements or a SurpyvalData holding ``m``: one row per reading,
    grouped by unit (in the order of the Measurements' units, or of the
    data's rows) and in time order within a unit. ``Z`` is the data's
    covariates, each unit's row repeated on each of its readings (None
    without), and ``names`` their names (None where unnamed).

    A unit with no readings is left out, with a warning naming it; from a
    SurpyvalData, only the readings of units in its rows are used (a unit
    with readings and no rows was warned of when the readings were
    attached)."""
    from surpyval.utils.measurements import Measurements

    Z_rows = None
    names = None
    if isinstance(source, Measurements):
        store = one_measurement(source, what)
        units = list(store)
    else:
        data = source
        if data.m is None:
            raise ValueError(
                f"{what} fits the readings taken on each unit, and the "
                "SurpyvalData holds none: attach them with "
                "SurpyvalData(..., i=..., m=Measurements(...)), or pass the "
                "Measurements itself."
            )
        _refuse_lost(data)
        store = one_measurement(data.m, what, data=True)
        units = (
            list(data.ids)
            if data.ids is not None
            else list(range(len(data.x)))
        )
        if data.Z is not None:
            Z_rows = dict(zip(units, np.asarray(data.Z, dtype=float)))
            names = (
                None
                if data.covariate_names is None
                else [str(n) for n in data.covariate_names]
            )
    x: list = []
    y: list = []
    i: list = []
    Z: list = []
    empty = []
    for unit in units:
        readings = store.get(unit, {})
        if not readings:
            empty.append(unit)
            continue
        for t, value in sorted(readings.items()):
            x.append(t)
            y.append(value)
            i.append(unit)
            if Z_rows is not None:
                Z.append(Z_rows[unit])
    _warn_no_readings(empty)
    if not x:
        raise ValueError(f"{what} has no readings to fit.")
    return (
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        _id_array(i),
        None if Z_rows is None else np.asarray(Z, dtype=float),
        names,
    )


def _is_readings(value: Any) -> bool:
    from surpyval.utils.measurements import Measurements
    from surpyval.utils.surpyval_data import SurpyvalData

    return isinstance(value, (Measurements, SurpyvalData))


def takes_readings(method: F) -> F:
    """Let a degradation fit take the readings as a Measurements, or a
    SurpyvalData holding them (``data.m``), in place of its arrays ``x``,
    ``y`` and ``i``: ``WienerProcess.fit(m, threshold=100)``. ``y``, ``i``
    and ``Z`` are then left out (a SurpyvalData's covariates are each
    unit's stress, held over its readings). With arrays, ``y``, ``i`` and
    ``threshold`` are needed, as before."""
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError:
            return method(*args, **kwargs)
        arguments = bound.arguments
        x = arguments.get("x")
        what = "This fit"
        if not _is_readings(x):
            missing = [
                name
                for name in ("x", "y", "i", "threshold")
                if arguments.get(name) is None
            ]
            if missing:
                raise TypeError(
                    "This fit needs "
                    + ", ".join(repr(name) for name in missing)
                    + ": the readings as arrays x (times), y (values) and "
                    "i (units), or as a Measurements (or a SurpyvalData "
                    "holding them as m) in place of x, y and i; and the "
                    "failure threshold."
                )
            return method(*args, **kwargs)
        given = [
            name for name in ("y", "i", "Z") if arguments.get(name) is not None
        ]
        if given:
            kind = type(x).__name__
            raise TypeError(
                f"x is a {kind}, which holds the readings and their units: "
                "leave out "
                + ", ".join(repr(name) for name in given)
                + " (or pass arrays x, y and i)."
                + (
                    " The stress is each unit's covariates in a "
                    "SurpyvalData (SurpyvalData(..., i=..., Z=..., m=...)); "
                    "a stress that changes during a unit's test is given "
                    "with arrays, one row per reading."
                    if "Z" in given
                    else ""
                )
            )
        if arguments.get("threshold") is None:
            raise TypeError(
                "This fit needs 'threshold', the degradation level at which "
                "a unit has failed."
            )
        x_arr, y_arr, i_arr, Z, names = degradation_readings(x, what)
        arguments["x"], arguments["y"], arguments["i"] = x_arr, y_arr, i_arr
        if Z is not None:
            arguments["Z"] = Z

        def fit() -> Any:
            return method(*bound.args, **bound.kwargs)

        if names is None:
            model = fit()
        else:
            from surpyval.univariate.regression._aliasing import (
                covariate_columns,
            )

            with covariate_columns(names):
                model = fit()
            _name_stress(model, names)
        from surpyval.utils.surpyval_data import with_data_unit

        return with_data_unit(model, x)

    setattr(wrapper, "_takes_readings", True)
    return wrapper  # type: ignore[return-value]


def _name_stress(model: Any, names: list) -> None:
    """Record the stress columns' names on a degradation model, as its
    ``fit_from_df`` does, so it reads a DataFrame of stress by name."""
    try:
        model.Z_cols = list(names)
    except AttributeError:
        return
    life = getattr(model, "life_model", None)
    if life is not None and hasattr(life, "feature_names"):
        from surpyval.univariate.regression import (
            parametric_regression_model as prm,
        )

        if isinstance(life, prm.ParametricRegressionModel):
            life.feature_names = list(names)


def readings_column(m_col: Any, y_col: Any, default: str = "y") -> Any:
    """The readings' column of a degradation ``fit_from_df``: ``m_col``,
    or its other name ``y_col`` (giving both is a TypeError); ``default``
    where neither is given."""
    if m_col is not None and y_col is not None:
        raise TypeError(
            "m_col and y_col name the same column (the readings' values; "
            "y_col is its other name): give one."
        )
    if m_col is not None:
        return m_col
    return default if y_col is None else y_col


# -- the joint models ---------------------------------------------------------


def markers_by_row(markers: Any, i: Any, x: Any) -> Any:
    """The joint fit's ``markers`` as one ``{age: value}`` per row of
    ``x``: a Measurements is matched to the rows by unit, ``i`` (a unit
    with no readings gets ``{}``, as from a SurpyvalData; a unit with
    readings and no row is warned of and its readings are not used); any
    other ``markers`` is returned as it is, and then ``i`` must be left
    out."""
    from surpyval.utils.item_history import checked_id
    from surpyval.utils.measurements import Measurements, attached
    from surpyval.utils.warnings import named

    if not isinstance(markers, Measurements):
        if i is not None:
            raise TypeError(
                "i matches a Measurements' readings to the rows by unit; "
                "markers given as a list are one {age: value} per row, in "
                "the rows' order: leave i out, or give the readings as a "
                "Measurements."
            )
        return markers
    store = one_measurement(markers, "The joint model")
    if i is None:
        raise TypeError(
            "The readings (a Measurements) are matched to the rows by "
            "unit, never by position: give each row's unit as i=."
        )
    rows = len(np.atleast_1d(np.asarray(x, dtype=float)))
    values = i.tolist() if isinstance(i, np.ndarray) else list(i)
    ids = [checked_id(v) for v in values]
    if len(ids) != rows:
        raise ValueError(
            f"'i' must hold one unit per row ({rows}); got {len(ids)}."
        )
    seen: dict = {}
    for k in ids:
        seen[k] = seen.get(k, 0) + 1
    several = [k for k, count in seen.items() if count > 1]
    if several:
        raise ValueError(
            "The joint model takes one row per unit, with the unit's "
            f"readings; unit{'s' if len(several) > 1 else ''} "
            f"{named(several)} {'have' if len(several) > 1 else 'has'} "
            "several rows in i."
        )
    # The readings of a unit with no row are warned of, as when they are
    # attached to a SurpyvalData
    attached(SimpleNamespace(ids=ids), markers)
    return [dict(store.get(k, {})) for k in ids]


def takes_joint_readings(method: F) -> F:
    """Let a joint fit take its readings as ``m`` (the other name of
    ``markers``; giving both is a TypeError), and a Measurements beside
    a SurpyvalData without readings: they are attached to the data
    (``data.m``) and fitted as from data holding them."""
    signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = signature.bind_partial(*args, **kwargs)
        except TypeError:
            return method(*args, **kwargs)
        arguments = bound.arguments
        m = arguments.pop("m", None)
        given = "m"
        if m is not None:
            if arguments.get("markers") is not None:
                raise TypeError(
                    "markers and m are one argument (m is its other name): "
                    "give one."
                )
            arguments["markers"] = m
        else:
            given = "markers"
        readings = arguments.get("markers")
        from surpyval.utils.measurements import Measurements
        from surpyval.utils.surpyval_data import SurpyvalData

        x = arguments.get("x")
        if isinstance(x, SurpyvalData) and readings is not None:
            if x.m is not None:
                raise TypeError(
                    "x is a SurpyvalData, which holds the readings (data.m): "
                    f"leave out {given!r} (or pass arrays rather than a "
                    "SurpyvalData)."
                )
            if isinstance(readings, Measurements):
                one_measurement(readings, "The joint model")
                data = x[np.arange(len(x.x))]
                data.m = readings
                arguments["x"] = data
                del arguments["markers"]
        return method(*bound.args, **bound.kwargs)

    return wrapper  # type: ignore[return-value]
