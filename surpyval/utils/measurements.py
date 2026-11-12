"""Measurements: readings taken on a unit over its life (a vibration
level, a temperature, a wear or oil-debris reading, a biomarker), on their
own timeline, kept apart from the life events and linked to them by the
unit ``i``.

A :class:`Measurements` holds, for each measurement (one unnamed, or
several named ones), each unit's readings ``{x: value}``, with ``x`` on
the same clock as the life data's ``x``. A
:class:`~surpyval.utils.surpyval_data.SurpyvalData` carries one as
``data.m`` (``data.markers`` is the same store, read as the one
measurement's ``{i: {x: value}}``).
"""

from __future__ import annotations

from numbers import Number
from typing import Any

import numpy as np

# The parts of a SurpyvalData's ``without`` that drop its measurements:
# the old name and the two new ones, one meaning.
MEASUREMENT_PARTS = ("markers", "m", "measurements")


def _plural(k: int, word: str) -> str:
    return f"{k} {word}" + ("" if k == 1 else "s")


def _shown(name: Any) -> str:
    return "(unnamed)" if name is None else str(name)


def _time(unit: Any, name: Any, x: Any) -> float:
    """One reading's time: a finite number (any sign: the clock is the
    life data's, which may start below 0; a model checks each reading
    against its unit's observation when it fits them)."""
    where = _where(unit, name)
    if isinstance(x, (bool, np.bool_)) or not isinstance(
        x, (Number, np.number)
    ):
        raise ValueError(
            f"{where}: a reading's time must be a number on the life "
            f"data's clock (an age); got {x!r}."
        )
    value = float(x)  # type: ignore[arg-type]
    if not np.isfinite(value):
        raise ValueError(
            f"{where}: a reading's time must be finite; got {x!r}."
        )
    return value


def _value(unit: Any, name: Any, x: float, value: Any) -> float:
    """One reading: a finite number."""
    try:
        number = float(value) if np.ndim(value) == 0 else None
    except (TypeError, ValueError):
        number = None
    if (
        number is None
        or isinstance(value, (bool, np.bool_))
        or not np.isfinite(number)
    ):
        raise ValueError(
            f"{_where(unit, name)}: the reading at {x:g} must be one finite "
            f"number; got {value!r} (leave a reading not taken out)."
        )
    return number


def _where(unit: Any, name: Any) -> str:
    return (
        f"Unit {unit!r}"
        if name is None
        else f"Measurement {name!r}, unit {unit!r}"
    )


def _unit_id(unit: Any) -> Any:
    from surpyval.utils.item_history import checked_id

    try:
        return checked_id(unit)
    except ValueError as err:
        raise ValueError(
            str(err).replace("An item", "A unit in the measurements", 1)
        ) from None


def _check_ids(ids: list) -> None:
    """The units' ids: of one kind (all text or all numbers), as the life
    rows' ids must be."""
    from surpyval.utils.item_history import check_item_ids

    check_item_ids(list(dict.fromkeys(ids)), "set of readings")


def _check_name(name: Any) -> Any:
    from surpyval.utils.item_history import item_id

    name = item_id(name)
    if name is None or (isinstance(name, str) and not name.strip()):
        raise ValueError(
            "Each of several measurements needs a name (text, such as "
            "'vibration'); a single unnamed measurement is given as "
            "{i: {x: value}}."
        )
    try:
        hash(name)
    except TypeError:
        raise ValueError(
            f"A measurement's name must be text or a number; got {name!r}."
        ) from None
    return name


def _readings_of(unit: Any, name: Any, readings: Any) -> dict:
    if not isinstance(readings, dict):
        raise ValueError(
            f"{_where(unit, name)}: the readings must be a dict "
            f"{{x: value}}, one reading per time; got "
            f"{type(readings).__name__} (a pandas Series indexed by time: "
            "series.to_dict())."
        )
    out: dict = {}
    for x, value in readings.items():
        t = _time(unit, name, x)
        if t in out:
            raise ValueError(
                f"{_where(unit, name)}: two readings at time {t:g}; give "
                "one reading per time."
            )
        out[t] = _value(unit, name, t, value)
    return dict(sorted(out.items()))


def _one(readings: dict, name: Any) -> dict:
    """``{i: {x: value}}`` checked."""
    out = {}
    for unit, held in readings.items():
        unit = _unit_id(unit)
        out[unit] = _readings_of(unit, name, held)
    return out


def _shape(readings: dict) -> str | None:
    """Whether ``readings`` is ``{i: {x: value}}`` ("one") or ``{name:
    {i: {x: value}}}`` ("named"), told by what the innermost dicts hold:
    a number is a reading (one measurement), a dict is a unit's readings
    (named measurements). None where nothing says (no readings at all)."""
    found = set()
    for held in readings.values():
        if not isinstance(held, dict):
            raise ValueError(
                "Measurements are a dict {i: {x: value}} (one measurement) "
                "or {name: {i: {x: value}}} (several named ones); got "
                f"{type(held).__name__} where a dict was expected."
            )
        for inner in held.values():
            found.add("named" if isinstance(inner, dict) else "one")
    if len(found) > 1:
        raise ValueError(
            "Measurements mix the two shapes, {i: {x: value}} (one "
            "measurement) and {name: {i: {x: value}}} (several named ones): "
            "give one of them."
        )
    return found.pop() if found else None


def _checked_units(units: Any, names: list) -> dict:
    """``{name: label}``, each label as a model stores a unit."""
    from surpyval.utils.units import normalise_unit

    if units is None:
        return {}
    if not isinstance(units, dict):
        if len(names) != 1:
            raise ValueError(
                "units= is one label per measurement, {name: 'mm/s'}, for "
                f"several measurements ({', '.join(map(_shown, names))})."
            )
        units = {names[0]: units}
    unknown = [k for k in units if k not in names]
    if unknown:
        raise ValueError(
            "units= names measurements not given: "
            + ", ".join(repr(k) for k in unknown)
            + "; the measurements are "
            + ", ".join(map(_shown, names))
            + "."
        )
    out = {}
    for k, label in units.items():
        canonical = normalise_unit(label)
        if canonical is not None:
            out[k] = canonical
    return out


class Measurements:
    """
    Readings taken on units over their lives, on their own timeline: for
    each measurement (one unnamed, or several named ones such as
    ``"vibration"`` and ``"temperature"``), each unit's readings ``{x:
    value}``, with ``x`` on the life data's clock. Readings of different
    measurements need not share times or units. A
    :class:`~surpyval.utils.surpyval_data.SurpyvalData` carries them as
    ``data.m``, linked to its rows by the unit ids ``i``.

    Parameters
    ----------
    readings : dict
        ``{i: {x: value}}``, one measurement (the joint models' marker
        readings), unnamed unless ``name`` is given; or ``{name: {i: {x:
        value}}}``, several named ones. The shape is read from what the
        innermost dicts hold (numbers: one measurement; dicts: named
        ones); a dict holding no reading at all says neither and is
        refused. A unit may be listed with no readings (``{}``).
    name : str, optional
        The name of the one measurement in ``{i: {x: value}}``.
    units : str or dict, optional
        The unit each measurement is read in, ``{name: "mm/s"}`` (a
        label, as a model's ``unit``); one label where there is one
        measurement.

    Each time and reading is a finite number, one reading per time for a
    unit and measurement; unit ids are text or numbers, all of one kind.
    There is no censoring or count on a reading: a reading not taken is
    left out.

    Examples
    --------
    >>> from surpyval import Measurements
    >>> m = Measurements({
    ...     "vibration": {"A": {0: 2.1, 30: 2.6}, "B": {0: 1.9}},
    ...     "temperature": {"A": {10: 61.0}},
    ... })
    >>> m
    Measurements: vibration, temperature - 2 units, 4 readings
    >>> m["vibration"]["A"]
    {0.0: 2.1, 30.0: 2.6}
    >>> m.for_unit("A")
    {'vibration': {0.0: 2.1, 30.0: 2.6}, 'temperature': {10.0: 61.0}}
    """

    def __init__(
        self, readings: Any, name: Any = None, units: Any = None
    ) -> None:
        if isinstance(readings, Measurements):
            raise TypeError(
                "readings is already a Measurements; use it as it is."
            )
        if not isinstance(readings, dict):
            raise ValueError(
                "Measurements takes a dict, {i: {x: value}} (one "
                "measurement) or {name: {i: {x: value}}} (several named "
                f"ones); got {type(readings).__name__} (a long DataFrame: "
                "Measurements.from_dataframe)."
            )
        shape = _shape(readings)
        if shape is None:
            raise ValueError(
                "The measurements hold no readings, so neither their shape "
                "nor anything else can be read from them: give at least "
                "one reading."
            )
        if shape == "named":
            if name is not None:
                raise ValueError(
                    "name= names the one measurement of {i: {x: value}}; "
                    "these readings are {name: {i: {x: value}}}, already "
                    "named."
                )
            store = {
                _check_name(k): _one(v, _check_name(k))
                for k, v in readings.items()
            }
        else:
            if name is not None:
                name = _check_name(name)
            store = {name: _one(readings, name)}
        _check_ids([i for one in store.values() for i in one])
        self._store = store
        self._units = _checked_units(units, list(store))

    @classmethod
    def _raw(cls, store: dict, units: dict | None = None) -> Measurements:
        """A Measurements of ``store`` (``{name: {i: {x: value}}}``) as it
        is, not checked: the old ``markers`` set on a dataset, and what
        the package builds from checked readings."""
        out = cls.__new__(cls)
        out._store = store
        out._units = dict(units or {})
        return out

    @classmethod
    def from_dataframe(
        cls,
        df: Any,
        *,
        i_col: Any,
        x_col: Any,
        m_col: Any,
        m_name_col: Any = None,
        name: Any = None,
        units: Any = None,
    ) -> Measurements:
        """
        The readings in a long DataFrame, one row per reading: the unit
        (``i_col``), the time (``x_col``), the value (``m_col``) and,
        for several measurements in one frame, the measurement's name
        (``m_name_col``; else the frame is one measurement, unnamed or
        named ``name``). A blank cell is refused, naming the rows (leave
        a reading not taken out of the frame), and so is a second reading
        of a unit and measurement at one time.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval import Measurements
        >>> df = pd.DataFrame({
        ...     "unit": ["A", "A", "B", "A"],
        ...     "hours": [0, 30, 0, 10],
        ...     "reading": [2.1, 2.6, 1.9, 61.0],
        ...     "what": ["vibration"] * 3 + ["temperature"],
        ... })
        >>> m = Measurements.from_dataframe(
        ...     df, i_col="unit", x_col="hours", m_col="reading",
        ...     m_name_col="what",
        ... )
        >>> m.names, m.n_readings
        (['vibration', 'temperature'], 4)
        """
        from surpyval.utils.data_frames import _plain
        from surpyval.utils.dataframe import frame_column, require_frame
        from surpyval.utils.item_history import missing_type

        df = require_frame(df)
        if len(df) == 0:
            raise ValueError("The DataFrame has no rows.")
        if m_name_col is not None and name is not None:
            raise ValueError(
                "name= names the one measurement of a frame without "
                "m_name_col; with m_name_col each row names its own."
            )
        units_col = [_plain(v) for v in frame_column(df, i_col, "i_col")]
        times = [_plain(v) for v in frame_column(df, x_col, "x_col", True)]
        values = [_plain(v) for v in frame_column(df, m_col, "m_col")]
        names = (
            [name] * len(df)
            if m_name_col is None
            else [
                _plain(v) for v in frame_column(df, m_name_col, "m_name_col")
            ]
        )
        labels = list(df.index)
        for column, arg in (
            (units_col, "i_col"),
            (times, "x_col"),
            (values, "m_col"),
        ) + (((names, "m_name_col"),) if m_name_col is not None else ()):
            blank = [
                labels[k] for k, v in enumerate(column) if missing_type(v)
            ]
            if blank:
                raise ValueError(
                    f"{arg}={_col(df, arg, i_col, x_col, m_col, m_name_col)!r}"
                    f" has {_plural(len(blank), 'blank cell')}, in rows "
                    + _rows(blank)
                    + ": a reading needs its unit, time, value"
                    + (" and measurement" if m_name_col is not None else "")
                    + "; leave a reading not taken out of the frame."
                )
        store: dict = {}
        seen: dict = {}
        for k, (unit, x, value, which) in enumerate(
            zip(units_col, times, values, names)
        ):
            if which is not None:
                which = _check_name(which)
            unit = _unit_id(unit)
            t = _time(unit, which, x)
            key = (which, unit, t)
            if key in seen:
                raise ValueError(
                    f"{_where(unit, which)}: two readings at time {t:g} (rows "
                    f"{labels[seen[key]]!r} and {labels[k]!r}); give one "
                    "reading per time."
                )
            seen[key] = k
            store.setdefault(which, {}).setdefault(unit, {})[t] = _value(
                unit, which, t, value
            )
        for one in store.values():
            for unit in one:
                one[unit] = dict(sorted(one[unit].items()))
        _check_ids([i for one in store.values() for i in one])
        return cls._raw(store, _checked_units(units, list(store)))

    @property
    def names(self) -> list:
        """The measurements' names, in order (``[None]`` for one unnamed
        measurement)."""
        return list(self._store)

    @property
    def units(self) -> dict:
        """``{name: unit}``, the unit each measurement is read in (None
        where not given)."""
        return {k: self._units.get(k) for k in self._store}

    @property
    def ids(self) -> list:
        """The units with readings listed, in the order first given (a
        unit listed with no readings is one)."""
        return list(
            dict.fromkeys(i for one in self._store.values() for i in one)
        )

    @property
    def n_units(self) -> int:
        """The number of units listed (:attr:`ids`)."""
        return len(self.ids)

    @property
    def n_readings(self) -> int:
        """The number of readings, over every measurement and unit."""
        return sum(
            len(r) for one in self._store.values() for r in one.values()
        )

    def __getitem__(self, name: Any) -> dict:
        """One measurement's readings, ``{i: {x: value}}`` (``m[None]``
        for the unnamed one)."""
        if name not in self._store:
            raise KeyError(
                f"No measurement {name!r}; the measurements are "
                + ", ".join(map(_shown, self._store))
                + "."
            )
        return {i: dict(r) for i, r in self._store[name].items()}

    def for_unit(self, i: Any) -> dict:
        """The unit ``i``'s readings, ``{name: {x: value}}``, for each
        measurement that lists it."""
        from surpyval.utils.item_history import item_id

        i = item_id(i)
        out = {k: dict(one[i]) for k, one in self._store.items() if i in one}
        if not out:
            raise KeyError(f"No readings of unit {i!r}.")
        return out

    def to_dataframe(self) -> Any:
        """
        The readings as a long DataFrame, one row per reading: ``i``,
        ``x``, ``value``, and ``name`` where the measurements are named
        (a unit listed with no readings has no row).
        ``Measurements.from_dataframe(df, i_col="i", x_col="x",
        m_col="value", m_name_col="name")`` reads it back (without
        ``m_name_col`` for one unnamed measurement), with ``units=``.
        """
        import pandas as pd

        rows = [
            (i, x, v, k)
            for k, one in self._store.items()
            for i, readings in one.items()
            for x, v in readings.items()
        ]
        columns = {
            "i": [r[0] for r in rows],
            "x": np.array([r[1] for r in rows], dtype=float),
            "value": np.array([r[2] for r in rows], dtype=float),
        }
        if self.names != [None]:
            columns["name"] = [r[3] for r in rows]
        return pd.DataFrame(columns)

    def to_dict(self) -> dict:
        """
        The readings as a dictionary of plain values (strict JSON when
        dumped): each measurement's name, unit and readings, each unit's
        as ``[i, [[x, value], ...]]`` so that ids keep their type (a
        tuple id as ``{"tuple": [...]}``, as the datasets write them).
        :meth:`from_dict` reads it back.

        Examples
        --------
        >>> from surpyval import Measurements
        >>> m = Measurements({1: {0: 2.0}}, name="wear", units="mm")
        >>> Measurements.from_dict(m.to_dict()) == m
        True
        """
        from surpyval.utils.data_json import label

        out = []
        for k, one in self._store.items():
            entry: dict = {"name": label(k)}
            if self._units.get(k) is not None:
                entry["unit"] = self._units[k]
            entry["readings"] = [
                [label(i), [[x, v] for x, v in readings.items()]]
                for i, readings in one.items()
            ]
            out.append(entry)
        return {"class": "Measurements", "measurements": out}

    @classmethod
    def from_dict(cls, data_dict: dict) -> Measurements:
        """The readings back from their :meth:`to_dict` dictionary."""
        from surpyval.utils.data_json import label_back

        if not isinstance(data_dict, dict) or not isinstance(
            data_dict.get("measurements"), list
        ):
            raise ValueError(
                "Expected a serialised Measurements dict (with "
                "'measurements'), got "
                + (
                    "a dict without it"
                    if isinstance(data_dict, dict)
                    else type(data_dict).__name__
                )
                + "."
            )
        tag = data_dict.get("class")
        if tag is not None and tag != "Measurements":
            raise ValueError(
                f"This dictionary was written by {tag}, not Measurements."
            )
        store: dict = {}
        units: dict = {}
        for entry in data_dict["measurements"]:
            k = label_back(entry.get("name"))
            store[k] = {
                label_back(i): {float(x): float(v) for x, v in pairs}
                for i, pairs in entry["readings"]
            }
            if entry.get("unit") is not None:
                units[k] = entry["unit"]
        return cls._raw(store, units)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Measurements):
            return NotImplemented
        return (
            list(self._store) == list(other._store)
            and self._store == other._store
            and self.units == other.units
        )

    __hash__ = None  # type: ignore[assignment]

    def _describe(self) -> str:
        """``vibration (mm/s), temperature - 27 units, 412 readings``."""
        names = ", ".join(
            _shown(k)
            + ("" if self._units.get(k) is None else f" ({self._units[k]})")
            for k in self._store
        )
        return (
            f"{names} - {_plural(self.n_units, 'unit')}, "
            f"{_plural(self.n_readings, 'reading')}"
        )

    def __repr__(self) -> str:
        return f"Measurements: {self._describe()}"

    # What a dataset uses -----------------------------------------------

    @property
    def _plain(self) -> bool:
        """Whether these are the old ``markers``: one unnamed measurement
        with no unit label, which a dataset writes and shows exactly as
        it did."""
        return list(self._store) == [None] and not self._units

    def _single(self) -> dict:
        """The one measurement's ``{i: {x: value}}`` (the store itself),
        refused where there are several."""
        if len(self._store) != 1:
            raise ValueError(
                "data.markers is one measurement's readings, and this data "
                f"holds {len(self._store)} ("
                + ", ".join(map(_shown, self._store))
                + "): read one with data.m[name], or keep one with "
                "data.with_measurement(name)."
            )
        return next(iter(self._store.values()))

    def _only(self, name: Any) -> Measurements:
        """The measurement ``name`` alone."""
        if name not in self._store:
            raise KeyError(
                f"No measurement {name!r}; the measurements are "
                + ", ".join(map(_shown, self._store))
                + "."
            )
        units = {name: self._units[name]} if name in self._units else {}
        return Measurements._raw({name: self._store[name]}, units)

    def _for_units(self, ids: Any, windows: dict | None = None) -> Any:
        """The readings of the units ``ids`` (in that order); with
        ``windows`` (``{i: (start, end)}``, a unit whose observation a
        reshape moved), a moved unit keeps the readings inside its new
        window, as the old markers did. None where no unit is left."""
        wanted = list(dict.fromkeys(ids))
        windows = windows or {}
        store = {}
        for k, one in self._store.items():
            kept = {}
            for i in wanted:
                if i not in one:
                    continue
                readings = one[i]
                if i in windows:
                    start, end = windows[i]
                    readings = {
                        x: v for x, v in readings.items() if start <= x <= end
                    }
                kept[i] = readings
            store[k] = kept
        if not any(store.values()):
            return None
        return Measurements._raw(store, self._units)

    def _readings_of_unit(self, i: Any) -> int:
        return sum(len(one.get(i, {})) for one in self._store.values())


def _col(df: Any, arg: str, *names: Any) -> Any:
    return dict(zip(("i_col", "x_col", "m_col", "m_name_col"), names))[arg]


def _rows(labels: list) -> str:
    from surpyval.utils.warnings import named

    return named(labels)


def attached(data: Any, value: Any) -> Measurements | None:
    """``value`` (a Measurements, or a dict it is built from) checked
    against the life rows of ``data``: the rows need unit ids; a unit with
    readings but no life rows is warned of, by count and id, and kept."""
    import warnings

    from surpyval.utils.warnings import caller_stacklevel, named

    if value is None:
        return None
    if not isinstance(value, Measurements):
        value = Measurements(value)
    ids = getattr(data, "ids", None)
    if ids is None:
        raise ValueError(
            "m (measurements) are linked to the life rows by each row's "
            "unit: give the units as i= (or i_col= in from_dataframe)."
        )
    have = set(ids)
    missing = [i for i in value.ids if i not in have]
    if missing:
        warnings.warn(
            f"{_plural(len(missing), 'unit')} in m "
            + ("has" if len(missing) == 1 else "have")
            + f" no life rows: {named(missing)}. Their readings are kept "
            "(data.m), and a fit uses only the readings of units in the "
            "life rows; check the ids match (text '1' and the number 1 "
            "differ).",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    return value


def _windows(data: Any) -> dict:
    """``{i: (start, end)}``: the ages over which each unit of ``data``
    was observed (from its entry, else 0, to its last row's end; over all
    its rows where it has several)."""
    x = np.asarray(data.x, dtype=float)
    end = x[:, 1] if x.ndim == 2 else x
    tl = np.asarray(data.tl, dtype=float)
    start = np.where(np.isfinite(tl), tl, 0.0)
    out: dict = {}
    for i, a, b in zip(data.ids, start.tolist(), end.tolist()):
        if i in out:
            a, b = min(a, out[i][0]), max(b, out[i][1])
        out[i] = (a, b)
    return out


class MeasurementsMixin:
    """What a lifetime dataset does with its measurements ``m``: one
    store, read as ``data.m`` (the :class:`Measurements`) or as
    ``data.markers`` (the old name: the one measurement's ``{i: {x:
    value}}``)."""

    _m: Measurements | None = None

    @property
    def m(self) -> Measurements | None:
        """The readings taken on the units over their lives, on their own
        timeline (a :class:`Measurements`), linked to the rows by the unit
        ids ``i``; None where the data has none. ``data.markers`` is the
        same store."""
        return self.__dict__.get("_m")

    @m.setter
    def m(self, value: Any) -> None:
        self._m = attached(self, value)

    @property
    def markers(self) -> dict | None:
        """The marker readings, ``{i: {x: value}}`` (``{}`` for a unit
        with none), of the data's one measurement: the same store as
        ``data.m``. Refused where the data holds several measurements
        (read one with ``data.m[name]``). Setting it sets ``data.m`` to
        that one unnamed measurement, as given."""
        m = self.m
        return None if m is None else m._single()

    @markers.setter
    def markers(self, value: Any) -> None:
        self._m = None if value is None else Measurements._raw({None: value})

    def _plain_markers(self) -> dict | None:
        """The old ``markers`` where the measurements are one unnamed
        measurement with no unit label (written and shown exactly as
        before); None otherwise."""
        m = self.m
        return m._store[None] if m is not None and m._plain else None

    def with_measurement(self, name: Any) -> Any:
        """
        The data with only the measurement ``name`` in ``m``: a fit that
        takes one marker (the joint models) takes it as its ``markers``.

        Examples
        --------
        >>> from surpyval import SurpyvalData
        >>> data = SurpyvalData(
        ...     [40, 55], c=[0, 1], i=["A", "B"],
        ...     m={"vibration": {"A": {0: 2.1}, "B": {0: 1.9}},
        ...        "temperature": {"A": {10: 61.0}}},
        ... )
        >>> data.with_measurement("vibration").markers
        {'A': {0.0: 2.1}, 'B': {0.0: 1.9}}
        """
        m = self.m
        if m is None:
            raise ValueError("This data has no measurements (m).")
        out = self[np.arange(len(self.x))]  # type: ignore
        out._m = m._only(name)
        return out

    def _refuse_measurement_records(self, what: str) -> None:
        """Refuse to give records of data whose measurements are named, or
        have a unit label, which an ``ItemLifetime``'s markers (one
        unnamed measurement) have no place for."""
        m = self.m
        if m is not None and not m._plain:
            raise ValueError(
                f"{what} gives ItemLifetime records, whose markers are one "
                "unnamed measurement with no unit; this data's measurements "
                f"({m._describe()}) are kept in data.m. Use "
                "data.without('m') for records without them, and "
                "data.m.for_unit(i) for a unit's readings."
            )

    def _carry_measurements(self, out: Any) -> Any:
        """``out``, reshaped from this data's records, with this data's
        named measurements put back for the units it keeps (a unit whose
        observation moved keeps the readings inside its new window);
        the records carried only the old markers."""
        m = self.m
        if m is None or m._plain:
            return out
        if out.m is not None:
            raise ValueError(
                "These records carry markers (one unnamed measurement), and "
                f"this data's measurements are named ({m._describe()}): "
                "give the records without markers, and the readings as "
                "data.m."
            )
        if out.ids is None:
            return out
        before = _windows(self)
        after = _windows(out)
        moved = {i: w for i, w in after.items() if before.get(i) != w}
        out._m = m._for_units(out.ids, moved)
        return out

    def without(self, *parts: Any) -> Any:
        """
        The data without some of what it holds, to fit a model that does
        not take it: any of ``"covariates"``, ``"event_types"``,
        ``"time_varying_covariates"``, ``"truncation"``, ``"attributes"``,
        ``"states"`` (the rows' states ``s``) and the measurements
        (``"m"``, ``"measurements"``, or their old name ``"markers"``),
        as separate arguments or a list.
        """
        flat = (
            tuple(parts[0])
            if len(parts) == 1 and isinstance(parts[0], (list, tuple, set))
            else parts
        )
        m = self.m
        if m is None or m._plain or not set(flat) & set(MEASUREMENT_PARTS):
            return super().without(*parts)  # type: ignore[misc]
        from surpyval.utils.data_reshape import check_parts

        check_parts(flat, self._PARTS)  # type: ignore[attr-defined]
        out = self[np.arange(len(self.x))]  # type: ignore
        out._m = None
        rest = [p for p in flat if p not in MEASUREMENT_PARTS]
        return out.without(*rest) if rest else out
