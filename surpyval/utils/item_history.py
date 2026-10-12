"""One item's recurrent event history, :class:`ItemHistory`; a list of
them is a dataset (:meth:`RecurrentEventData.from_items
<surpyval.utils.recurrent_event_data.RecurrentEventData.from_items>`)."""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["ItemHistory"]


class ItemHistory:
    """
    One repairable item's history: its failures, observed exactly, found
    in an interval or known only to have happened before an age, how long
    it was watched, the window in which it could be seen, and its
    covariates. Ages are the item's own age (time since new or first
    observed). A list of them makes a dataset:
    ``RecurrentEventData.from_items([pump, fan, ...])``, which every
    recurrent ``fit`` takes.

    Parameters
    ----------
    item : hashable
        The item's id.
    observed : array like, optional
        The ages at which it failed (a single failure may be a number;
        repeat an age for simultaneous failures).
    right_censored : scalar, optional
        Its age at the end of observation (when it was last checked), at
        or after every failure. Left out, observation is taken to have
        ended at its last failure.
    interval_censored : array like, optional
        ``[lower, upper]`` pairs: failures found between two ages, such as
        a last good and a failed inspection.
    interval_censored_counts : scalar or array like, optional
        How many failures each interval holds: one number for every
        interval or one per interval (default 1; an interval counted 0 is
        left out).
    left_censored : scalar, optional
        An age before which (since the start of observation) failures
        happened at unknown times, such as the first inspection.
    left_censored_counts : scalar, optional
        How many failures happened before ``left_censored`` (default 1).
    left_truncation : scalar, optional
        The age at which the item came under observation: failures before
        it are not known.
    right_truncation : scalar, optional
        The age after which its failures would not have been seen. With
        ``right_censored``, the two must be the same age.
    covariates : scalar or array like, optional
        Its covariates, constant over its history (a number or a row).
    observed_event_types, interval_censored_event_types : optional
        What each failure was (a failure mode or cause, for the
        cause-specific models): one type for every entry of the list, or
        one per entry.
    left_censored_event_types : optional
        The type of the failures counted by ``left_censored``.
    time_varying_covariates : dict, StepSchedule or CovariatePath, optional
        Covariates that change over its history: ``{age: values}``, each
        value holding from its age on (the first age at or before the
        start of observation, ``left_truncation``, else 0); a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`;
        or a continuously varying
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`
        (both of these hold their first value back to age 0). A dict is
        kept as a ``StepSchedule``. No recurrent model uses them yet
        (#905): every fit refuses data that has them.

    initial_state : optional
        The state the item was in when it came under observation (at
        ``left_truncation``, else age 0), for multi-state models; needed
        once any change of state is given.
    observed_states : dict, optional
        ``{age: state}``: changes of state seen as they happened.
    interval_censored_states : array like, optional
        ``[lower, upper, state]`` rows: a change into ``state`` somewhere
        between two inspections.
    left_censored_states : dict, optional
        ``{age: state}``: already in ``state`` at the first look, ``age``,
        having changed into it since the start of observation (at most
        one, before every other change).

    Event types, given for some failures, are needed for all of them (the
    end of observation has none). Changes of state are the item's path
    through its states: each into a state other than the one it was in,
    in order of age and not overlapping, and within its observation; the
    end of observation (``right_censored``, needed where there are no
    failures) is the age up to which it was seen in its last state. No
    model uses states yet (#887): every fit refuses data that has them.
    ``markers`` (repeated measurements, #891) are reserved for the joint
    models.

    Examples
    --------
    >>> from surpyval import ItemHistory, RecurrentEventData
    >>> pump = ItemHistory(
    ...     "P1",
    ...     observed=[420, 1310],
    ...     interval_censored=[[2100, 2300]],
    ...     interval_censored_counts=2,
    ...     right_censored=2600,
    ... )
    >>> fan = ItemHistory("F1", observed=[800], right_censored=3000)
    >>> data = RecurrentEventData.from_items([pump, fan])
    >>> data.items
    ['F1', 'P1']
    >>> data.to_items()[1] == pump
    True
    """

    def __init__(
        self,
        item: Any,
        observed: Any = None,
        right_censored: Any = None,
        *,
        interval_censored: Any = None,
        interval_censored_counts: Any = None,
        left_censored: Any = None,
        left_censored_counts: Any = None,
        left_truncation: Any = None,
        right_truncation: Any = None,
        covariates: Any = None,
        time_varying_covariates: Any = None,
        observed_event_types: Any = None,
        interval_censored_event_types: Any = None,
        left_censored_event_types: Any = None,
        initial_state: Any = None,
        observed_states: "dict | None" = None,
        interval_censored_states: Any = None,
        left_censored_states: "dict | None" = None,
    ) -> None:
        self.item = item
        ages = _ages(item, "observed", observed)
        types = _event_types(
            item,
            "observed_event_types",
            observed_event_types,
            len(ages),
            "observed",
        )
        order = np.argsort(ages, kind="stable")
        self.observed = ages[order]
        self.observed_event_types = None if types is None else types[order]
        self.right_censored = _age(item, "right_censored", right_censored)
        intervals = _intervals(item, interval_censored)
        counts = _counts(
            item,
            "interval_censored_counts",
            interval_censored_counts,
            len(intervals),
            allow_zero=True,
        )
        types = _event_types(
            item,
            "interval_censored_event_types",
            interval_censored_event_types,
            len(intervals),
            "interval_censored",
        )
        order = np.lexsort((intervals[:, 1], intervals[:, 0]))
        keep = counts[order] > 0
        self.interval_censored = intervals[order][keep]
        self.interval_censored_counts = counts[order][keep]
        self.interval_censored_event_types = (
            None if types is None else types[order][keep]
        )
        self.left_censored = _age(item, "left_censored", left_censored)
        if left_censored_counts is not None and left_censored is None:
            raise ValueError(
                f"Item {item!r}: 'left_censored_counts' was given without "
                "'left_censored', the age the failures happened before."
            )
        self.left_censored_counts = (
            None
            if self.left_censored is None
            else int(
                _counts(
                    item,
                    "left_censored_counts",
                    (
                        1
                        if left_censored_counts is None
                        else left_censored_counts
                    ),
                    1,
                    allow_zero=False,
                )[0]
            )
        )
        left_types = _event_types(
            item,
            "left_censored_event_types",
            left_censored_event_types,
            0 if self.left_censored is None else 1,
            "left_censored",
        )
        self.left_censored_event_types = (
            None if left_types is None else left_types[0]
        )
        self.left_truncation = _age(item, "left_truncation", left_truncation)
        self.right_truncation = _age(
            item, "right_truncation", right_truncation
        )
        self.covariates = _covariate_row(item, "covariates", covariates)
        self.time_varying_covariates, self._first_change = _time_varying(
            item, time_varying_covariates
        )
        self.initial_state = initial_state
        self.state_changes = _state_changes(
            item,
            observed_states,
            interval_censored_states,
            left_censored_states,
            0.0 if self.left_truncation is None else self.left_truncation,
        )
        self._check()

    # -- checks -------------------------------------------------------------

    @property
    def has_event_types(self) -> bool:
        """Whether this item's failures carry event types."""
        return (
            self.observed_event_types is not None
            or self.interval_censored_event_types is not None
            or self.left_censored_event_types is not None
        )

    def _check(self) -> None:
        item = self.item
        if self.has_event_types:
            untyped = [
                name
                for name, entries, types in (
                    (
                        "observed",
                        len(self.observed),
                        self.observed_event_types,
                    ),
                    (
                        "interval_censored",
                        len(self.interval_censored),
                        self.interval_censored_event_types,
                    ),
                    (
                        "left_censored",
                        int(self.left_censored is not None),
                        self.left_censored_event_types,
                    ),
                )
                if entries and types is None
            ]
            if untyped:
                raise ValueError(
                    f"Item {item!r}: event types were given for some failures "
                    "but not for "
                    + ", ".join(untyped)
                    + ": every failure needs its type ("
                    + ", ".join(f"'{name}_event_types'" for name in untyped)
                    + ")."
                )
        events = [*self.observed, *self.interval_censored.ravel()]
        if self.left_censored is not None:
            events.append(self.left_censored)
        if not events and self.right_censored is None:
            raise ValueError(
                f"Item {item!r} has no failures and no end of observation: "
                "give at least one"
                + (
                    " (a path through states needs the age up to which it "
                    "was seen, right_censored)."
                    if self.state_changes
                    else "."
                )
            )
        self._check_states()
        tl, tr = self.left_truncation, self.right_truncation
        if tl is not None and tr is not None and not tl < tr:
            raise ValueError(
                f"Item {item!r}: left_truncation ({tl:g}) must be below "
                f"right_truncation ({tr:g})."
            )
        last = max(events) if events else None
        end = self.right_censored
        if end is not None and last is not None and end < last:
            raise ValueError(
                f"Item {item!r}: the end of observation, {end:g}, is before "
                f"its last failure, {last:g}: right_censored is the age when "
                "the item was last checked, at or after every failure."
            )
        if end is not None and tr is not None and end != tr:
            raise ValueError(
                f"Item {item!r}: right_censored ({end:g}) and "
                f"right_truncation ({tr:g}) both end its observation, so "
                "they must be the same age; give one."
            )
        if self.left_censored is not None:
            later = [*self.observed, *self.interval_censored[:, 0]]
            if later and self.left_censored > min(later):
                raise ValueError(
                    f"Item {item!r}: left_censored ({self.left_censored:g}) "
                    "counts the failures before an age, so it must be at or "
                    f"before its other failures (the first at {min(later):g})."
                )
        if tl is not None:
            early = [
                *self.observed[self.observed <= tl],
                *self.interval_censored[self.interval_censored[:, 0] < tl, 0],
            ]
            if self.left_censored is not None and self.left_censored <= tl:
                early.append(self.left_censored)
            if end is not None and end <= tl:
                early.append(end)
            if early:
                raise ValueError(
                    f"Item {item!r} has a failure or end at {min(early):g}, "
                    f"at or before it came under observation at "
                    f"{tl:g} (left_truncation): it is observed over "
                    "(left_truncation, ...], so nothing can happen at or "
                    "before that age."
                )
        if tr is not None and last is not None and last > tr:
            raise ValueError(
                f"Item {item!r} has a failure at {last:g}, after "
                f"right_truncation ({tr:g}), past which none could be seen."
            )
        if self._first_change is not None:
            start = 0.0 if tl is None else tl
            first = self._first_change
            if first > start:
                raise ValueError(
                    f"Item {item!r}: its time-varying covariates start at age "
                    f"{first:g}, after the start of its observation "
                    f"({start:g}): give their values from {start:g} on."
                )

    @property
    def has_states(self) -> bool:
        """Whether this item's path through states is given."""
        return self.initial_state is not None

    def _check_states(self) -> None:
        item = self.item
        changes = self.state_changes
        if changes and self.initial_state is None:
            raise ValueError(
                f"Item {item!r}: changes of state were given without "
                "'initial_state', the state it was in when it came under "
                "observation."
            )
        if self.initial_state is None:
            return
        state = self.initial_state
        previous_end = -np.inf
        for kind, lower, upper, new in changes:
            if kind == "left_censored" and previous_end > -np.inf:
                raise ValueError(
                    f"Item {item!r}: left_censored_states counts a change "
                    "before the first look, so it must come before every "
                    "other change of state."
                )
            if new == state:
                raise ValueError(
                    f"Item {item!r}: a change into {new!r} at "
                    f"{_when(kind, lower, upper)} is from {new!r} itself; a "
                    "change of state is into another state."
                )
            if lower < previous_end or (
                kind == "observed" and lower == previous_end
            ):
                raise ValueError(
                    f"Item {item!r}: its changes of state overlap or are out "
                    f"of order at {_when(kind, lower, upper)}: each must "
                    "follow the one before."
                )
            state, previous_end = new, upper
        tl, tr = self.left_truncation, self.right_truncation
        for kind, lower, upper, new in changes:
            if tl is not None and (
                lower < tl or (kind == "observed" and lower <= tl)
            ):
                raise ValueError(
                    f"Item {item!r}: the change into {new!r} at "
                    f"{_when(kind, lower, upper)} is not after it came under "
                    f"observation ({tl:g}, left_truncation)."
                )
            if tr is not None and upper > tr:
                raise ValueError(
                    f"Item {item!r}: the change into {new!r} at "
                    f"{_when(kind, lower, upper)} is after right_truncation "
                    f"({tr:g})."
                )
        end = self.right_censored
        if changes and end is not None and end < changes[-1][2]:
            raise ValueError(
                f"Item {item!r}: the end of observation, {end:g}, is before "
                f"its last change of state, {changes[-1][2]:g}."
            )

    # -- rows for the dataset ----------------------------------------------

    def _rows(self) -> tuple[list, list, list, list]:
        """``([lower, upper] rows, c, n, event types)`` in handle_xicn's
        layout (the end of observation has no type: None)."""
        x: list = []
        c: list = []
        n: list = []
        e: list = []
        if self.left_censored is not None:
            x.append([self.left_censored, self.left_censored])
            c.append(-1)
            n.append(self.left_censored_counts)
            e.append(self.left_censored_event_types)
        observed_types = (
            [None] * len(self.observed)
            if self.observed_event_types is None
            else list(self.observed_event_types)
        )
        for age, kind in zip(self.observed, observed_types):
            x.append([age, age])
            c.append(0)
            n.append(1)
            e.append(kind)
        interval_types = (
            [None] * len(self.interval_censored)
            if self.interval_censored_event_types is None
            else list(self.interval_censored_event_types)
        )
        for (lower, upper), count, kind in zip(
            self.interval_censored,
            self.interval_censored_counts,
            interval_types,
        ):
            x.append([lower, upper])
            c.append(2)
            n.append(int(count))
            e.append(kind)
        if self.right_censored is not None:
            x.append([self.right_censored, self.right_censored])
            c.append(1)
            n.append(1)
            e.append(None)
        return x, c, n, e

    # -- updates (each returns a new, checked ItemHistory) ----------------

    def clone(self, item: Any = None, **changes: Any) -> "ItemHistory":
        """
        A copy of this history, under the id ``item`` (else the same id),
        with any of its arguments replaced by ``changes``.

        Examples
        --------
        >>> from surpyval import ItemHistory
        >>> pump = ItemHistory("P1", observed=[420, 1310], right_censored=2600)
        >>> pump.clone("P2", right_censored=3000)
        ItemHistory(
            'P2',
            observed=[420.0, 1310.0],
            right_censored=3000.0
        )
        """
        fields = self._fields()
        fields.update(changes)
        return ItemHistory(self.item if item is None else item, **fields)

    def extend(self, age: Any) -> "ItemHistory":
        """This history, seen working up to ``age``: its end of
        observation moved later (at or after its last event).

        >>> from surpyval import ItemHistory
        >>> ItemHistory("P1", observed=[420], right_censored=600).extend(900)
        ItemHistory(
            'P1',
            observed=[420.0],
            right_censored=900.0
        )
        """
        age = _age(self.item, "age", age)
        if self.right_censored is not None and age < self.right_censored:
            raise ValueError(
                f"Item {self.item!r}: extend moves the end of observation "
                f"later, but {age:g} is before its end, "
                f"{self.right_censored:g}."
            )
        return self.clone(right_censored=age)

    def _after_event(self, fields: dict, age: float) -> dict:
        # An event after the end of observation was seen, so the item was
        # watched until it: the end moves to it.
        end = fields.get("right_censored")
        if end is not None and age > end:
            fields["right_censored"] = age
        return fields

    def add_failure(self, age: Any, event_type: Any = None) -> "ItemHistory":
        """This history with another failure at ``age`` (of
        ``event_type``, needed where its failures have types).

        >>> from surpyval import ItemHistory
        >>> pump = ItemHistory("P1", observed=[420], right_censored=600)
        >>> pump.add_failure(700).right_censored
        700.0
        """
        age = _age(self.item, "age", age)
        fields = self._fields()
        fields["observed"] = [*fields.get("observed", []), age]
        if self.has_event_types or event_type is not None:
            fields["observed_event_types"] = [
                *fields.get(
                    "observed_event_types",
                    (
                        [None] * len(self.observed)
                        if self.observed_event_types is None
                        else list(self.observed_event_types)
                    ),
                ),
                event_type,
            ]
        return ItemHistory(self.item, **self._after_event(fields, age))

    def add_failures_between(
        self,
        lower: Any,
        upper: Any,
        failures: Any = 1,
        event_type: Any = None,
    ) -> "ItemHistory":
        """This history with ``failures`` failures found between two
        inspections, ``lower`` and ``upper``."""
        fields = self._fields()
        fields["interval_censored"] = [
            *fields.get("interval_censored", []),
            [lower, upper],
        ]
        counts = (
            list(self.interval_censored_counts)
            if len(self.interval_censored)
            else []
        )
        fields["interval_censored_counts"] = [*counts, failures]
        if self.has_event_types or event_type is not None:
            types = (
                [None] * len(self.interval_censored)
                if self.interval_censored_event_types is None
                else list(self.interval_censored_event_types)
            )
            fields["interval_censored_event_types"] = [*types, event_type]
        upper = _age(self.item, "upper", upper)
        return ItemHistory(self.item, **self._after_event(fields, upper))

    def change_state(self, age: Any, state: Any) -> "ItemHistory":
        """This history with a change into ``state`` seen at ``age``."""
        age = _age(self.item, "age", age)
        fields = self._fields()
        fields["observed_states"] = {
            **fields.get("observed_states", {}),
            age: state,
        }
        return ItemHistory(self.item, **self._after_event(fields, age))

    def change_state_between(
        self, lower: Any, upper: Any, state: Any
    ) -> "ItemHistory":
        """This history with a change into ``state`` somewhere between two
        inspections, ``lower`` and ``upper``."""
        fields = self._fields()
        fields["interval_censored_states"] = [
            *fields.get("interval_censored_states", []),
            [lower, upper, state],
        ]
        upper = _age(self.item, "upper", upper)
        return ItemHistory(self.item, **self._after_event(fields, upper))

    def update_covariates(self, age: Any, values: Any) -> "ItemHistory":
        """This history with its time-varying covariates taking ``values``
        from ``age`` on (a step added to its schedule; a first schedule
        starts at ``age``, which must then be at or before the start of
        observation)."""
        steps = _with_step(
            self.item, self.time_varying_covariates, age, values
        )
        return self.clone(time_varying_covariates=steps)

    # -- display and comparison ---------------------------------------------

    def _fields(self) -> dict:
        """The arguments that rebuild this history, without the defaults."""
        out: dict = {}
        if len(self.observed):
            out["observed"] = self.observed.tolist()
        if self.observed_event_types is not None and len(self.observed):
            out["observed_event_types"] = self.observed_event_types.tolist()
        if self.right_censored is not None:
            out["right_censored"] = self.right_censored
        if len(self.interval_censored):
            out["interval_censored"] = self.interval_censored.tolist()
            if (self.interval_censored_counts != 1).any():
                out["interval_censored_counts"] = (
                    self.interval_censored_counts.tolist()
                )
            if self.interval_censored_event_types is not None:
                out["interval_censored_event_types"] = (
                    self.interval_censored_event_types.tolist()
                )
        if self.left_censored is not None:
            out["left_censored"] = self.left_censored
            if self.left_censored_counts != 1:
                out["left_censored_counts"] = self.left_censored_counts
            if self.left_censored_event_types is not None:
                out["left_censored_event_types"] = (
                    self.left_censored_event_types
                )
        for name in ("left_truncation", "right_truncation"):
            if getattr(self, name) is not None:
                out[name] = getattr(self, name)
        if self.covariates is not None:
            out["covariates"] = self.covariates.tolist()
        if self.time_varying_covariates is not None:
            out["time_varying_covariates"] = _schedule_field(
                self.time_varying_covariates
            )
        if self.initial_state is not None:
            out["initial_state"] = self.initial_state
        observed = {
            u: st for k, _, u, st in self.state_changes if k == "observed"
        }
        if observed:
            out["observed_states"] = observed
        intervals = [
            [lo, u, st]
            for k, lo, u, st in self.state_changes
            if k == "interval_censored"
        ]
        if intervals:
            out["interval_censored_states"] = intervals
        left = {
            u: st for k, _, u, st in self.state_changes if k == "left_censored"
        }
        if left:
            out["left_censored_states"] = left
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ItemHistory):
            return NotImplemented
        return self.item == other.item and self._fields() == other._fields()

    __hash__ = None  # type: ignore[assignment]

    def __repr__(self) -> str:
        fields = "".join(
            f",\n    {name}={value!r}"
            for name, value in self._fields().items()
        )
        return f"ItemHistory(\n    {self.item!r}{fields}\n)"


# -- validation helpers -----------------------------------------------------


def _ages(item: Any, name: str, value: Any) -> np.ndarray:
    """A list of finite ages (a number is a list of one; None, none)."""
    if value is None:
        return np.empty(0)
    try:
        arr = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        raise ValueError(f"Item {item!r}: '{name}' must be numbers.")
    if arr.ndim != 1:
        raise ValueError(f"Item {item!r}: '{name}' must be a list of ages.")
    if not np.isfinite(arr).all():
        raise ValueError(f"Item {item!r}: '{name}' ages must be finite.")
    return arr


def _age(item: Any, name: str, value: Any) -> "float | None":
    """One finite age, or None."""
    if value is None:
        return None
    if np.ndim(value) != 0:
        raise ValueError(f"Item {item!r}: '{name}' must be one age.")
    try:
        age = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Item {item!r}: '{name}' must be a number.")
    if not np.isfinite(age):
        raise ValueError(f"Item {item!r}: '{name}' must be finite.")
    return age


def _intervals(item: Any, value: Any) -> np.ndarray:
    """``(k, 2)`` finite ``[lower, upper]`` pairs with lower < upper."""
    if value is None:
        return np.empty((0, 2))
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            f"Item {item!r}: 'interval_censored' must be numbers."
        )
    if arr.size == 0:
        return np.empty((0, 2))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"Item {item!r}: 'interval_censored' must be [lower, upper] "
            "pairs; for a single interval, [[lower, upper]]."
        )
    if not np.isfinite(arr).all():
        raise ValueError(
            f"Item {item!r}: 'interval_censored' ages must be finite (an "
            "interval open at the top is the end of observation, "
            "right_censored)."
        )
    if (arr[:, 0] >= arr[:, 1]).any():
        raise ValueError(
            f"Item {item!r}: each interval in 'interval_censored' needs "
            "lower < upper."
        )
    return arr


def _counts(
    item: Any, name: str, value: Any, k: int, allow_zero: bool
) -> np.ndarray:
    """``k`` whole-number counts: 1 each where not given, one for all, or
    one each."""
    if value is None:
        return np.ones(k, dtype=int)
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(f"Item {item!r}: '{name}' must be whole numbers.")
    if arr.ndim == 0:
        arr = np.full(k, float(arr))
    elif arr.ndim != 1 or len(arr) != k:
        raise ValueError(
            f"Item {item!r}: '{name}' must be one count for all, or one per "
            f"interval ({k}); got {np.shape(value)}."
        )
    least = 0 if allow_zero else 1
    if not np.isfinite(arr).all() or (arr < least).any() or (arr % 1).any():
        raise ValueError(
            f"Item {item!r}: '{name}' must be whole numbers, {least} or more."
        )
    return arr.astype(int)


def _covariate_row(item: Any, name: str, value: Any) -> "np.ndarray | None":
    """A finite 1-D row of covariates, or None."""
    if value is None:
        return None
    try:
        row = np.atleast_1d(np.asarray(value, dtype=float))
    except (TypeError, ValueError):
        raise ValueError(f"Item {item!r}: '{name}' must be numbers.")
    if row.ndim != 1 or not np.isfinite(row).all():
        raise ValueError(
            f"Item {item!r}: '{name}' must be one finite number or a row "
            "of them."
        )
    return row


def _time_varying(item: Any, value: Any) -> tuple[Any, "float | None"]:
    """The time-varying covariates as a StepSchedule or CovariatePath,
    and, for a ``{age: values}`` dict, its first age (checked against the
    start of observation); ``(None, None)`` where not given."""
    from surpyval.univariate.regression.tvc_path import CovariatePath
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if value is None:
        return None, None
    if isinstance(value, (StepSchedule, CovariatePath)):
        return value, None
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"Item {item!r}: 'time_varying_covariates' must be a dict "
            "{age: values} (the values holding from each age on), a "
            "StepSchedule or a CovariatePath."
        )
    ages = np.array([_age(item, "time_varying_covariates", a) for a in value])
    rows = [
        _covariate_row(item, "time_varying_covariates", v)
        for v in value.values()
    ]
    if len({len(row) for row in rows}) > 1:  # type: ignore[arg-type]
        raise ValueError(
            f"Item {item!r}: every age's 'time_varying_covariates' must have "
            "the same number of values."
        )
    order = np.argsort(ages)
    schedule = StepSchedule.from_changepoints(
        ages[order], np.vstack(rows)[order]
    )
    return schedule, float(ages.min())


def _schedule_field(schedule: Any) -> Any:
    """A StepSchedule without a period as the ``{age: values}`` dict that
    rebuilds it; anything else (a cyclic schedule, a CovariatePath) as
    itself."""
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if isinstance(schedule, StepSchedule) and schedule.period is None:
        return {
            float(age): row.tolist()
            for age, row in zip(schedule.edges[:-1], schedule.Z)
        }
    return schedule


def _event_types(
    item: Any, name: str, value: Any, k: int, kind: str
) -> "np.ndarray | None":
    """``k`` event types as an object array: one for all, or one each;
    None where not given."""
    if value is None:
        return None
    if k == 0 and kind == "left_censored":
        raise ValueError(
            f"Item {item!r}: '{name}' was given without 'left_censored'."
        )
    if isinstance(value, (str, bytes)) or np.ndim(value) == 0:
        types = np.empty(k, dtype=object)
        types[:] = [value] * k
    else:
        listed = list(value)
        if len(listed) != k:
            raise ValueError(
                f"Item {item!r}: '{name}' must be one type for every entry "
                f"of '{kind}', or one per entry ({k}); got {len(listed)}."
            )
        types = np.empty(k, dtype=object)
        types[:] = listed
    if any(t is None for t in types):
        raise ValueError(
            f"Item {item!r}: '{name}' cannot hold None: every failure needs "
            "its type."
        )
    return types


def _state_label(item: Any, name: str, state: Any) -> Any:
    """A state label: anything but None."""
    if state is None:
        raise ValueError(
            f"Item {item!r}: '{name}' cannot hold None as a state."
        )
    return state


def _state_changes(
    item: Any,
    observed: "dict | None",
    intervals: Any,
    left: "dict | None",
    start: float,
) -> list:
    """The changes of state as ``(kind, lower, upper, state)``, sorted by
    age; an observed change has ``lower == upper``, a left censored one
    runs from the start of observation."""
    changes: list = []
    for name, value in (
        ("observed_states", observed),
        ("left_censored_states", left),
    ):
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(
                f"Item {item!r}: '{name}' must be a dict {{age: state}}."
            )
        kind = name[: -len("_states")]
        for age, state in value.items():
            age = _age(item, name, age)
            changes.append(
                (
                    kind,
                    age if kind == "observed" else start,
                    age,
                    _state_label(item, name, state),
                )
            )
    if left is not None and len(left) > 1:
        raise ValueError(
            f"Item {item!r}: 'left_censored_states' holds at most one change "
            "(the state at the first look)."
        )
    if intervals is not None:
        for row in intervals:
            try:
                lower, upper, state = row
            except (TypeError, ValueError):
                raise ValueError(
                    f"Item {item!r}: 'interval_censored_states' must be "
                    "[lower, upper, state] rows."
                )
            lower = _age(item, "interval_censored_states", lower)
            upper = _age(item, "interval_censored_states", upper)
            if not lower < upper:  # type: ignore[operator]
                raise ValueError(
                    f"Item {item!r}: each row of 'interval_censored_states' "
                    "needs lower < upper."
                )
            changes.append(
                (
                    "interval_censored",
                    lower,
                    upper,
                    _state_label(item, "interval_censored_states", state),
                )
            )
    changes.sort(key=lambda change: (change[2], change[1]))
    return changes


def _when(kind: str, lower: float, upper: float) -> str:
    """Where a change of state happened, for a message."""
    if kind == "observed":
        return f"{upper:g}"
    if kind == "left_censored":
        return f"before {upper:g}"
    return f"({lower:g}, {upper:g}]"


def _with_step(item: Any, schedule: Any, age: float, values: Any) -> dict:
    """``schedule`` (None, or a StepSchedule without a period) with its
    values from ``age`` on replaced by ``values``, as the ``{age: values}``
    dict that builds it."""
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if schedule is not None and not (
        isinstance(schedule, StepSchedule) and schedule.period is None
    ):
        kind = type(schedule).__name__
        if isinstance(schedule, StepSchedule):
            kind += " with a period"
        raise ValueError(
            f"Item {item!r}: update_covariates adds a step to a step "
            f"schedule; its covariates are a {kind}, so give the whole path "
            "with clone(time_varying_covariates=...)."
        )
    steps = {} if schedule is None else _schedule_field(schedule)
    age = float(age)
    steps = {a: v for a, v in steps.items() if a < age}
    row = _covariate_row(item, "values", values)
    steps[age] = row.tolist()  # type: ignore[union-attr]
    return steps
