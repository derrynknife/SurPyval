"""One item's lifetime, :class:`ItemLifetime`; a list of them is a dataset
(:meth:`SurpyvalData.from_items
<surpyval.utils.surpyval_data.SurpyvalData.from_items>`)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from surpyval.utils.item_history import (
    _age,
    _attributes,
    _covariate_field,
    _covariates,
    _schedule_field,
    _time_varying,
    _unit,
    _with_step,
    checked_id,
    comparable,
    covariate_value,
    freeze,
    frozen_setattr,
    frozen_setstate,
    hashable,
    item_id,
    late_covariates,
    missing_type,
)
from surpyval.utils.record_keywords import (
    LIFETIME_ALIASES,
    accepted_names,
    refuse_unknown_keywords,
)

__all__ = ["ItemLifetime"]

_OUTCOMES = (
    "observed",
    "right_censored",
    "left_censored",
    "interval_censored",
)


class ItemLifetime:
    """
    One item's lifetime: how it ended (or what is known of its failure),
    the window in which it could have been seen, what failed, and its
    covariates. A list of them makes a dataset,
    ``SurpyvalData.from_items([b7, b8, ...], counts=...)``, which the
    parametric, non-parametric and regression ``fit`` methods take. How
    many identical items there are is the dataset's ``counts``: an
    ``ItemLifetime`` is one item.

    Give exactly one outcome: ``observed``, ``right_censored``,
    ``left_censored`` or ``interval_censored``.

    Parameters
    ----------
    id : hashable
        The item's id (``.id``).
    observed : scalar, optional
        Its age at failure.
    right_censored : scalar, optional
        The age at which it was last seen working (a suspension).
    left_censored : scalar, optional
        An age by which it had already failed, at an unknown time.
    interval_censored : array like, optional
        ``[lower, upper]``: it failed between these ages.
    left_truncation : scalar, optional
        The age at which it came under observation: had it failed before,
        it would not be in the data.
    right_truncation : scalar, optional
        The age after which a failure would not have been seen.
    event_type : optional
        What failed (a failure mode or cause, for competing risks); not for
        a right censored item, which has not failed.
    covariates : scalar, array like or dict, optional
        Its covariates: a number, a row, or ``{name: value}`` (a pandas
        Series too), which names them (``covariate_names``);
        ``covariates`` is then the row of values, in that order.
    time_varying_covariates : dict, StepSchedule or CovariatePath, optional
        Covariates that change over its life: ``{age: values}`` (each
        value holding from its age on), a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
        or a continuously varying
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`.
        Each must start at or before the start of observation
        (``left_truncation``, else 0): nothing says what they were before
        their first age.
    markers : dict, optional
        Repeated measurements of one marker (a vibration level, a wear or
        oil-debris reading, a biomarker), ``{age: value}``: one reading
        per age, each a finite number taken while the item was observed,
        at or after the start of observation (``left_truncation``, else
        0) and at or before its outcome's age (the upper end of an
        interval). The joint models (``WeibullJointPH``) fit them with
        the lifetime; every other fit refuses data that has them
        (``data.without("markers")`` leaves them out).
    attributes : dict, optional
        ``{name: value}``, what describes the item without being fitted
        (a site, a batch, a product line: any values), for
        ``data.groupby(name)`` and ``data.select(where={name: value})``.
    unit : str, optional
        The unit its ages are in (``"hours"``, ``"km"``, ``"cycles"``), a
        label kept with the record (pint's canonical name for a unit it
        knows): a dataset takes it from its records, which must agree,
        and ``fit(data)`` gives it to the model.

    Examples
    --------
    >>> from surpyval import ItemLifetime, SurpyvalData
    >>> items = [
    ...     ItemLifetime("B7", observed=1820),
    ...     ItemLifetime("B8", right_censored=2400),
    ...     ItemLifetime("B9", interval_censored=[1000, 1500]),
    ... ]
    >>> data = SurpyvalData.from_items(items, counts=[1, 40, 2])
    >>> data.c, data.n
    (array([0, 1, 2]), array([ 1, 40,  2]))
    >>> data.to_items()[0] == items[0]
    True

    A pump's vibration readings, taken monthly while it ran:

    >>> pump = ItemLifetime(
    ...     "P1", right_censored=90, markers={0: 2.1, 30: 2.3}
    ... )
    >>> pump.update_markers(60, 2.6).markers
    {0.0: 2.1, 30.0: 2.3, 60.0: 2.6}
    """

    def __init__(
        self,
        id: Any,
        observed: Any = None,
        right_censored: Any = None,
        left_censored: Any = None,
        interval_censored: Any = None,
        *,
        left_truncation: Any = None,
        right_truncation: Any = None,
        event_type: Any = None,
        covariates: Any = None,
        time_varying_covariates: Any = None,
        markers: Any = None,
        attributes: Any = None,
        unit: Any = None,
        **unknown: Any,
    ) -> None:
        if unknown:
            refuse_unknown_keywords(
                "ItemLifetime",
                unknown,
                accepted_names(ItemLifetime.__init__, "self"),
                LIFETIME_ALIASES,
            )
        item = checked_id(id)
        self.id = item
        given = {
            name: value
            for name, value in zip(
                _OUTCOMES,
                (observed, right_censored, left_censored, interval_censored),
            )
            if value is not None
        }
        if len(given) != 1:
            raise ValueError(
                f"Item {item!r}: give exactly one outcome (observed, "
                "right_censored, left_censored or interval_censored); got "
                + (", ".join(given) if given else "none")
                + "."
            )
        ((outcome, value),) = given.items()
        self.outcome = outcome
        if outcome == "interval_censored":
            self.value: Any = _interval(item, value)
        else:
            self.value = _age(item, outcome, value)
        self.left_truncation = _age(item, "left_truncation", left_truncation)
        self.right_truncation = _age(
            item, "right_truncation", right_truncation
        )
        # A missing type (None, NaN, pandas NA, "") is None: on a failure,
        # a type not recorded (a masked cause); a numpy value is kept as
        # the plain value it holds.
        event_type = None if missing_type(event_type) else item_id(event_type)
        if event_type is not None and outcome == "right_censored":
            raise ValueError(
                f"Item {item!r}: a right censored item has not failed, so it "
                "has no event_type."
            )
        self.event_type = event_type
        self.covariates, self.covariate_names = _covariates(item, covariates)
        self.attributes = _attributes(item, attributes)
        self.unit = _unit(item, unit)
        self.time_varying_covariates, first, self._tvc_dict = _time_varying(
            item, time_varying_covariates
        )
        self._check(first)
        # The readings as two read-only arrays (frozen with the record), in
        # order of age; ``markers`` gives them as a dict.
        self._marker_ages, self._marker_values = self._checked_markers(markers)
        freeze(self)

    def __setattr__(self, name: str, value: Any) -> None:
        frozen_setattr(self, name, value)

    def __setstate__(self, state: dict) -> None:
        frozen_setstate(self, state)

    def attribute(self, name: str) -> Any:
        """The value of the attribute ``name`` (``attributes={name:
        value}``, what describes the item without being fitted: a site, a
        batch); None where the item does not have it.
        ``data.select(where={"site": "North"})`` keeps the items of one
        site, and ``data.groupby("site")`` splits the data by it."""
        if self.attributes is None:
            return None
        return self.attributes.get(name)

    def covariate(self, name: str) -> float:
        """The value of the covariate ``name`` (where the covariates are
        named, ``covariates={name: value}``): NaN where it was not
        recorded. ``data.select(where=lambda item: item.covariate("site")
        == 1)`` keeps the items of one site."""
        return covariate_value(self, name)

    @property
    def has_event_types(self) -> bool:
        """Whether this item's failure carries an event type (as an
        ``ItemHistory``'s ``has_event_types``)."""
        return self.event_type is not None

    # The outcome by name, as ItemHistory reads: its value, or None.
    @property
    def observed(self) -> "float | None":
        """The age at failure, if it failed at a known age."""
        return self.value if self.outcome == "observed" else None

    @property
    def right_censored(self) -> "float | None":
        """The age it was last seen working, if still running."""
        return self.value if self.outcome == "right_censored" else None

    @property
    def left_censored(self) -> "float | None":
        """The age by which it had failed, if that is all that is known."""
        return self.value if self.outcome == "left_censored" else None

    @property
    def interval_censored(self) -> "np.ndarray | None":
        """``[lower, upper]``, if it failed between two ages."""
        return self.value if self.outcome == "interval_censored" else None

    @property
    def markers(self) -> "dict | None":
        """The marker readings, ``{age: value}`` in order of age, or None
        where it has none."""
        ages = getattr(self, "_marker_ages", None)
        if ages is None or not len(ages):
            return None
        return dict(zip(ages.tolist(), self._marker_values.tolist()))

    def _window(self) -> "tuple[float, float]":
        """The ages over which it was observed: from the start of
        observation (``left_truncation``, else 0) to its outcome's age
        (the upper end of an interval)."""
        start = 0.0 if self.left_truncation is None else self.left_truncation
        end = (
            float(self.value[1])
            if self.outcome == "interval_censored"
            else float(self.value)
        )
        return start, end

    def _checked_markers(self, value: Any) -> "tuple[np.ndarray, np.ndarray]":
        """The readings ``{age: value}`` as arrays of ages and values in
        order of age (empty where none), each a finite number read inside
        the item's observation."""
        item = self.id
        if value is None:
            return np.empty(0), np.empty(0)
        if not isinstance(value, Mapping):
            raise ValueError(
                f"Item {item!r}: 'markers' must be a dict {{age: value}}, "
                "one reading per age (a pandas Series indexed by age: "
                "series.to_dict())."
            )
        ages, values = [], []
        for age, reading in value.items():
            age = _age(item, "markers", age)
            try:
                number = float(reading) if np.ndim(reading) == 0 else None
            except (TypeError, ValueError):
                number = None
            if number is None or not np.isfinite(number):
                raise ValueError(
                    f"Item {item!r}: the marker reading at age {age:g} must "
                    f"be one finite number; got {reading!r} (leave a "
                    "reading not taken out of markers)."
                )
            ages.append(age)
            values.append(number)
        ages_arr, values_arr = np.array(ages), np.array(values)
        if len(np.unique(ages_arr)) < len(ages_arr):
            raise ValueError(
                f"Item {item!r}: 'markers' has two readings at one age; "
                "give one reading per age."
            )
        start, end = self._window()
        early = ages_arr[ages_arr < start]
        if early.size:
            since = (
                f"left_truncation, {start:g}"
                if self.left_truncation is not None
                else "age 0"
            )
            raise ValueError(
                f"Item {item!r}: a marker reading at age {early.min():g} is "
                f"before the start of its observation ({since}): markers "
                "are the readings taken while it was observed."
            )
        late = ages_arr[ages_arr > end]
        if late.size:
            hint = (
                f" If it was still running at {late.max():g}, extend it "
                f"first: extend({late.max():g})."
                if self.outcome == "right_censored"
                else ""
            )
            raise ValueError(
                f"Item {item!r}: a marker reading at age {late.max():g} is "
                f"after its {self.outcome} age, {end:g}: markers are the "
                f"readings taken while it was observed.{hint}"
            )
        order = np.argsort(ages_arr, kind="stable")
        return ages_arr[order], values_arr[order]

    def _check(self, first_change: "float | None") -> None:
        item, tl, tr = self.id, self.left_truncation, self.right_truncation
        if tl is not None and tr is not None and not tl < tr:
            raise ValueError(
                f"Item {item!r}: left_truncation ({tl:g}) must be below "
                f"right_truncation ({tr:g})."
            )
        low, high = (
            (self.value[0], self.value[1])
            if self.outcome == "interval_censored"
            else (self.value, self.value)
        )
        if tl is not None and (
            low < tl or (self.outcome != "interval_censored" and low <= tl)
        ):
            raise ValueError(
                f"Item {item!r}: its {self.outcome} age is not after it came "
                f"under observation at {tl:g} (left_truncation)."
            )
        if tr is not None and high > tr:
            raise ValueError(
                f"Item {item!r}: its {self.outcome} age is after "
                f"right_truncation ({tr:g})."
            )
        if tr is not None and self.outcome == "right_censored" and high >= tr:
            # The dataset refuses it (xcnt_handler); refused here, naming
            # the item, so that every record that builds goes into a
            # dataset (#931).
            raise ValueError(
                f"Item {item!r}: right censored at {high:g}, at its "
                f"right_truncation ({tr:g}). An item right truncated at "
                f"{tr:g} is in the data because it failed by then, so it "
                f"cannot have been seen working at {high:g}. If {tr:g} is "
                "the end of its observation, that is right_censored alone, "
                "without right_truncation."
            )
        if first_change is not None:
            start = 0.0 if tl is None else tl
            if first_change > start:
                raise late_covariates(item, first_change, start)

    def _row(self) -> tuple[list, int]:
        """``([lower, upper], c)`` in the xcnt layout."""
        flag = {
            "observed": 0,
            "right_censored": 1,
            "left_censored": -1,
            "interval_censored": 2,
        }[self.outcome]
        if self.outcome == "interval_censored":
            return [float(self.value[0]), float(self.value[1])], flag
        return [self.value, self.value], flag

    # -- updates (each returns a new, checked ItemLifetime) ---------------

    def clone(self, id: Any = None, **changes: Any) -> "ItemLifetime":
        """
        A copy of this lifetime, under the id ``id`` (else the same id),
        with any of its arguments replaced by ``changes`` (an outcome
        given replaces its outcome).

        Examples
        --------
        >>> from surpyval import ItemLifetime
        >>> ItemLifetime("B7", observed=1820, covariates=1).clone("B8")
        ItemLifetime('B8', observed=1820.0, covariates=[1.0])
        """
        fields = self._fields()
        if any(name in changes for name in _OUTCOMES):
            fields.pop(self.outcome)
            if "right_censored" in changes:
                fields.pop("event_type", None)
        fields.update(changes)
        return ItemLifetime(self.id if id is None else id, **fields)

    def _still_running(self, what: str) -> float:
        if self.outcome != "right_censored":
            raise ValueError(
                f"Item {self.id!r} is {self.outcome}, not still running "
                f"(right_censored), so it cannot {what}."
            )
        return self.value

    def extend(self, age: Any) -> "ItemLifetime":
        """This lifetime, still running and seen working up to a later
        ``age``.

        >>> from surpyval import ItemLifetime
        >>> ItemLifetime("B8", right_censored=2400).extend(3000)
        ItemLifetime('B8', right_censored=3000.0)
        """
        last = self._still_running("be extended")
        age = _age(self.id, "age", age)
        if age < last:
            raise ValueError(
                f"Item {self.id!r}: extend moves it later, but {age:g} is "
                f"before it was last seen working, {last:g}."
            )
        return self.clone(right_censored=age)

    def fail(self, age: Any, event_type: Any = None) -> "ItemLifetime":
        """This lifetime, still running, failed at ``age``.

        >>> from surpyval import ItemLifetime
        >>> ItemLifetime("B8", right_censored=2400).fail(2550, "spall")
        ItemLifetime('B8', observed=2550.0, event_type='spall')
        """
        last = self._still_running("fail")
        age = _age(self.id, "age", age)
        if age < last:
            raise ValueError(
                f"Item {self.id!r}: it was seen working at {last:g}, so it "
                f"cannot have failed at {age:g}."
            )
        return self.clone(observed=age, event_type=event_type)

    def found_failed(self, age: Any, event_type: Any = None) -> "ItemLifetime":
        """This lifetime, still running, found failed at an inspection at
        ``age``: it failed between when it was last seen working and then.

        >>> from surpyval import ItemLifetime
        >>> ItemLifetime("B8", right_censored=2400).found_failed(2800)
        ItemLifetime('B8', interval_censored=[2400.0, 2800.0])
        """
        last = self._still_running("be found failed")
        age = _age(self.id, "age", age)
        if age <= last:
            hint = (
                f"; if it failed at {last:g}, use fail({last:g})"
                if age == last
                else ""
            )
            raise ValueError(
                f"Item {self.id!r} was last seen working at {last:g}, so "
                f"it cannot be found failed at {age:g}: an inspection that "
                f"finds it failed is after that{hint}."
            )
        return self.clone(interval_censored=[last, age], event_type=event_type)

    def update_covariates(self, age: Any, values: Any) -> "ItemLifetime":
        """This lifetime with its time-varying covariates taking ``values``
        from ``age`` on (a step added to its schedule)."""
        steps = _with_step(
            self.id,
            self.time_varying_covariates,
            age,
            values,
            self._as_dict(),
        )
        return self.clone(time_varying_covariates=steps)

    def update_markers(self, age: Any, value: Any) -> "ItemLifetime":
        """This lifetime with a marker reading ``value`` at ``age`` (in
        place of one already at that age). The reading must be inside its
        observation: for a new reading of an item still running, extend it
        first, ``item.extend(age).update_markers(age, value)``.

        >>> from surpyval import ItemLifetime
        >>> pump = ItemLifetime("P1", right_censored=60, markers={0: 2.1})
        >>> pump.extend(90).update_markers(90, 2.8).markers
        {0.0: 2.1, 90.0: 2.8}
        """
        readings = dict(self.markers or {})
        readings[_age(self.id, "age", age)] = value
        return self.clone(markers=readings)

    def _fields(self) -> dict:
        value = (
            self.value.tolist()
            if self.outcome == "interval_censored"
            else self.value
        )
        out: dict = {self.outcome: value}
        for name in ("left_truncation", "right_truncation", "event_type"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.covariates is not None:
            out["covariates"] = _covariate_field(
                self.covariates, self.covariate_names
            )
        if self.attributes is not None:
            out["attributes"] = dict(self.attributes)
        if self.unit is not None:
            out["unit"] = self.unit
        if self.time_varying_covariates is not None:
            out["time_varying_covariates"] = _schedule_field(
                self.time_varying_covariates, self._as_dict()
            )
        if self.markers is not None:
            out["markers"] = self.markers
        return out

    def to_dict(self) -> dict:
        """
        The record as a dictionary of plain Python values (strict JSON
        when dumped): its ``"class"``, ``"id"`` and the arguments that
        rebuild it, as :meth:`from_dict` reads them. A tuple id or event
        type is written as ``{"tuple": [...]}``; a step schedule of
        time-varying covariates by its ages and values; a
        ``CovariatePath`` is refused.

        Examples
        --------
        >>> from surpyval import ItemLifetime
        >>> ItemLifetime("B7", observed=1820).to_dict()
        {'class': 'ItemLifetime', 'id': 'B7', 'observed': 1820.0}
        """
        from surpyval.utils.data_json import record_dict

        return record_dict(self)

    @classmethod
    def from_dict(cls, record_dict: dict) -> "ItemLifetime":
        """The record back from its :meth:`to_dict` dictionary, checked
        as one built from its arguments.

        >>> from surpyval import ItemLifetime
        >>> item = ItemLifetime("B7", observed=1820)
        >>> ItemLifetime.from_dict(item.to_dict()) == item
        True
        """
        from surpyval.utils.data_json import record_from_dict

        return record_from_dict(cls, record_dict)

    def to_json(self, filepath: Any = None) -> "str | None":
        """The record as strict JSON text (:meth:`to_dict` dumped), or
        written to the file ``filepath``."""
        from surpyval.utils.data_json import dump

        return dump(self.to_dict(), filepath)

    @classmethod
    def from_json(cls, source: Any) -> "ItemLifetime":
        """The record back from JSON text (a string starting with ``{``)
        or the file holding it.

        >>> from surpyval import ItemLifetime
        >>> item = ItemLifetime("B7", observed=1820)
        >>> ItemLifetime.from_json(item.to_json()) == item
        True
        """
        from surpyval.utils.data_json import load

        return cls.from_dict(load(source))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ItemLifetime):
            return NotImplemented
        return self.id == other.id and self._compared() == other._compared()

    def __hash__(self) -> int:
        return hash((self.id, hashable(self._compared())))

    def _compared(self) -> dict:
        return comparable(self._fields(), self.time_varying_covariates)

    def _as_dict(self) -> bool:
        # A record pickled before #930 has no _tvc_dict: its schedule
        # showed as a dict.
        return getattr(self, "_tvc_dict", True)

    def __repr__(self) -> str:
        fields = "".join(
            f", {name}={value!r}" for name, value in self._fields().items()
        )
        return f"ItemLifetime({self.id!r}{fields})"


def _interval(item: Any, value: Any) -> np.ndarray:
    """``[lower, upper]``, finite, with lower < upper."""
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"Item {item!r}: 'interval_censored' must be numbers."
        )
    if arr.shape != (2,):
        raise ValueError(
            f"Item {item!r}: 'interval_censored' must be one [lower, upper] "
            "pair."
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            f"Item {item!r}: 'interval_censored' ages must be finite (open at "
            "the top is right_censored, at the bottom left_censored)."
        )
    if not arr[0] < arr[1]:
        raise ValueError(
            f"Item {item!r}: 'interval_censored' needs lower < upper."
        )
    return arr
