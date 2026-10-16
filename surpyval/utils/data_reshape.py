"""Reshaping a dataset item by item: each operation turns every item's
record (an :class:`~surpyval.utils.item_lifetime.ItemLifetime` or an
:class:`~surpyval.utils.item_history.ItemHistory`) into a new one, or
drops it, and the dataset is built again from the records -- so a
reshaped dataset passes the same checks as one built by hand. The
dataset methods (``without``, ``with_covariates``, ``for_event_type``,
``relabel_event_types``, ``select``, ``censor_at``, ``enter_at``) call
these."""

from __future__ import annotations

import warnings
from typing import Any, Callable

import numpy as np

from surpyval.utils.warnings import MaskedCauseWarning, caller_stacklevel

LIFETIME_PARTS = {
    "covariates": ("covariates",),
    "event_types": ("event_type",),
    "time_varying_covariates": ("time_varying_covariates",),
    "truncation": ("left_truncation", "right_truncation"),
    "attributes": ("attributes",),
}
HISTORY_PARTS = {
    "covariates": ("covariates",),
    "event_types": (
        "observed_event_types",
        "interval_censored_event_types",
        "left_censored_event_types",
    ),
    "time_varying_covariates": ("time_varying_covariates",),
    "truncation": ("left_truncation", "right_truncation"),
    "states": (
        "initial_state",
        "observed_states",
        "interval_censored_states",
        "left_censored_states",
    ),
    "attributes": ("attributes",),
}


def _rebuild(record: Any, fields: dict) -> Any:
    return type(record)(record.id, **fields)


def _is_history(record: Any) -> bool:
    from surpyval.utils.item_history import ItemHistory

    return isinstance(record, ItemHistory)


def check_parts(parts: tuple, allowed: dict) -> None:
    unknown = [p for p in parts if p not in allowed]
    if unknown or not parts:
        raise ValueError(
            "without() takes the parts to leave out, of "
            + ", ".join(repr(p) for p in allowed)
            + (
                f"; not {', '.join(repr(u) for u in unknown)}"
                if unknown
                else ""
            )
            + "."
        )


def without(record: Any, parts: tuple) -> Any:
    """``record`` without the named parts."""
    table = HISTORY_PARTS if _is_history(record) else LIFETIME_PARTS
    fields = record._fields()
    for part in parts:
        for name in table[part]:
            fields.pop(name, None)
    return _rebuild(record, fields)


def with_covariates(record: Any, columns: list) -> Any:
    """``record`` with only the covariate ``columns`` (indices)."""
    if record.covariates is None:
        raise ValueError(f"Item {record.id!r} has no covariates to pick from.")
    width = len(record.covariates)
    names = record.covariate_names
    named = [c for c in columns if not isinstance(c, (int, np.integer))]
    if named and names is None:
        raise ValueError(
            f"Covariates are picked by column number (0 to {width - 1}), "
            f"and these covariates have no names (name them with "
            f"covariates={{name: value}}); got {named!r}."
        )
    unknown = [c for c in named if c not in names]
    if unknown:
        raise ValueError(
            f"Item {record.id!r} has no covariates {unknown!r}; its "
            f"covariates are {list(names)!r}."
        )
    picked = [names.index(c) if c in named else c for c in columns]
    if len(
        {c % width for c in picked if isinstance(c, (int, np.integer))}
    ) < len(picked):
        twice = [c for c in columns if columns.count(c) > 1] or columns
        raise ValueError(
            f"Covariate {twice[0]!r} is picked more than once: each column "
            "once (the same column twice is not two covariates)."
        )
    columns = picked
    bad = [
        c
        for c in columns
        if not (isinstance(c, (int, np.integer)) and -width <= c < width)
    ]
    if bad:
        raise ValueError(
            f"Covariate columns {bad!r} are not among the {width} column(s) "
            f"(0 to {width - 1})."
        )
    fields = record._fields()
    fields["covariates"] = [record.covariates[c] for c in columns]
    if names is not None:
        fields["covariates"] = {
            names[c]: record.covariates[c] for c in columns
        }
    return _rebuild(record, fields)


def relabel(record: Any, mapping: dict) -> Any:
    """``record`` with its event types renamed by ``mapping``; a ``None``
    key renames the failures whose type was not recorded (masked causes),
    never a suspension."""
    fields = record._fields()
    if _is_history(record):
        if not record.has_event_types:
            return record
        names: tuple = (
            "observed_event_types",
            "interval_censored_event_types",
            "left_censored_event_types",
        )
        if record.left_censored is not None:
            fields.setdefault("left_censored_event_types", None)
    else:
        if record.outcome == "right_censored":
            return record
        names = ("event_type",)
        fields.setdefault("event_type", None)
    for name in names:
        if name not in fields:
            continue
        value = fields[name]
        if isinstance(value, list):
            fields[name] = [mapping.get(v, v) for v in value]
        else:
            fields[name] = mapping.get(value, value)
    return _rebuild(record, fields)


# -- one failure mode ---------------------------------------------------------


def for_event_type(
    record: Any, event_type: Any, masked: str = "censor"
) -> Any:
    """``record`` as data of one failure mode, ``event_type``: the other
    modes' failures stop being failures, as do failures whose type was not
    recorded (masked causes) unless ``masked="count"``, which counts them
    as of this type. A lifetime loses its right truncation (#923): it says
    the item's first failure, of any mode, came by that age, which says
    nothing of when this mode's would have come."""
    if _is_history(record):
        return _history_for_event_type(record, event_type, masked)
    fields = record._fields()
    fields.pop("right_truncation", None)
    if record.outcome == "right_censored":
        return _rebuild(record, fields)
    if record.event_type == event_type or (
        record.event_type is None and masked == "count"
    ):
        fields.pop("event_type", None)
        return _rebuild(record, fields)
    # A failure of another mode: for this mode, the item was last known
    # to be without it when it failed of the other one.
    fields.pop("event_type", None)
    fields.pop(record.outcome)
    if record.outcome == "observed":
        fields["right_censored"] = record.value
    elif record.outcome == "interval_censored":
        last_good = float(record.value[0])
        if (
            record.left_truncation is not None
            and last_good <= record.left_truncation
        ):
            return None
        fields["right_censored"] = last_good
    else:
        # Failed of another mode at an unknown age before its look: no age
        # is known at which it was still without this mode's failure.
        return None
    return _rebuild(record, fields)


def _history_for_event_type(
    history: Any, event_type: Any, masked: str = "censor"
) -> Any:
    if not history.has_event_types:
        return history
    if masked == "count":
        # Its failures of no recorded type are counted as of this one
        history = relabel(history, {None: event_type})
    fields = history._fields()
    ages = list(history.observed)
    types = list(history.observed_event_types)
    last = []
    keep = [a for a, t in zip(ages, types) if t == event_type]
    last += [a for a, t in zip(ages, types) if t != event_type]
    fields["observed"] = keep
    pairs = history.interval_censored.tolist()
    counts = history.interval_censored_counts.tolist()
    itypes = (
        list(history.interval_censored_event_types)
        if history.interval_censored_event_types is not None
        else [None] * len(pairs)
    )
    fields["interval_censored"] = [
        p for p, t in zip(pairs, itypes) if t == event_type
    ]
    fields["interval_censored_counts"] = [
        k for k, t in zip(counts, itypes) if t == event_type
    ]
    last += [p[1] for p, t in zip(pairs, itypes) if t != event_type]
    if history.left_censored is not None and (
        history.left_censored_event_types != event_type
    ):
        last.append(history.left_censored)
        for name in ("left_censored", "left_censored_counts"):
            fields.pop(name, None)
    for name in HISTORY_PARTS["event_types"]:
        fields.pop(name, None)
    # Selected on a failure of any type: not accounted for (warned of by
    # the dataset, as for lifetimes)
    fields.pop("right_truncation", None)
    # Observation went on at least to the last failure of another mode.
    if last and fields.get("right_censored") is None:
        kept = [*keep, *(p[1] for p in fields["interval_censored"])]
        fields["right_censored"] = max([*last, *kept])
    for name in ("observed", "interval_censored", "interval_censored_counts"):
        if not fields.get(name):
            fields.pop(name, None)
    return _rebuild(history, fields)


# -- windows ------------------------------------------------------------------


def censor_at(record: Any, age: float) -> Any:
    """``record`` as seen if observation stopped at ``age``."""
    if record.left_truncation is not None and record.left_truncation >= age:
        return None
    if _is_history(record):
        return _history_censor_at(record, age)
    fields = record._fields()
    outcome, value = record.outcome, record.value
    if outcome in ("observed", "right_censored"):
        if value <= age:
            return record
    elif outcome == "left_censored":
        if value <= age:
            return record
        return None  # failed before its look, which is after the window
    else:
        lower, upper = float(value[0]), float(value[1])
        if upper <= age:
            return record
        if lower < age:
            # Failed somewhere in (lower, upper], across the end of the
            # window: all that is known is that it worked at lower.
            if (
                record.left_truncation is not None
                and lower <= record.left_truncation
            ):
                return None
            age = lower
    fields.pop(outcome)
    fields.pop("event_type", None)
    fields["right_censored"] = age
    return _rebuild(record, fields)


def _spans(history: Any) -> list:
    """The spans of a history over which what happened is not known in
    time: its failures counted between two inspections and its changes of
    state between two inspections, as ``(lower, upper)``; a count or
    state known only before a first look spans from the start of
    observation to the look."""
    start = 0.0 if history.left_truncation is None else history.left_truncation
    spans = [(float(lo), float(hi)) for lo, hi in history.interval_censored]
    spans += [
        (float(lo), float(hi))
        for lo, hi, _ in history.interval_censored_states
    ]
    if history.left_censored is not None:
        spans.append((start, history.left_censored))
    spans += [(start, float(a)) for a in history.left_censored_states]
    return spans


def _history_censor_at(history: Any, age: float) -> Any:
    # Observation stops at age, or earlier where a span crosses it: what
    # happened in a span is known only as a whole, so the history is
    # known only up to the span's last good inspection, its lower end.
    spans = _spans(history)
    while True:
        crossing = [lo for lo, hi in spans if lo < age < hi]
        if not crossing:
            break
        age = min(crossing)
    start = 0.0 if history.left_truncation is None else history.left_truncation
    if age <= start:
        return None  # nothing about it is known in the window
    fields = history._fields()
    end = history.right_censored
    events = [*history.observed, *history.interval_censored[:, 1]]
    if history.left_censored is not None:
        events.append(history.left_censored)
    last = max(events) if events else None
    seen_to = end if end is not None else last
    if seen_to is not None and seen_to <= age:
        return history
    typed = history.observed_event_types
    keep = [k for k, a in enumerate(history.observed) if a <= age]
    fields["observed"] = [history.observed[k] for k in keep]
    if typed is not None:
        fields["observed_event_types"] = [typed[k] for k in keep]
    pairs = history.interval_censored
    inside = [k for k, (lo, hi) in enumerate(pairs) if hi <= age]
    fields["interval_censored"] = [pairs[k].tolist() for k in inside]
    fields["interval_censored_counts"] = [
        int(history.interval_censored_counts[k]) for k in inside
    ]
    if history.interval_censored_event_types is not None:
        fields["interval_censored_event_types"] = [
            history.interval_censored_event_types[k] for k in inside
        ]
    if history.left_censored is not None and history.left_censored > age:
        for name in (
            "left_censored",
            "left_censored_counts",
            "left_censored_event_types",
        ):
            fields.pop(name, None)
    if history.initial_state is not None:
        fields["observed_states"] = {
            a: s for a, s in history.observed_states.items() if a <= age
        } or None
        fields["interval_censored_states"] = [
            r for r in history.interval_censored_states if r[1] <= age
        ] or None
        fields["left_censored_states"] = {
            a: s for a, s in history.left_censored_states.items() if a <= age
        } or None
    fields["right_censored"] = age
    for name in (
        "observed",
        "observed_event_types",
        "interval_censored",
        "interval_censored_counts",
        "interval_censored_event_types",
    ):
        if name in fields and not fields[name]:
            fields.pop(name)
    return _rebuild(history, fields)


def enter_at(record: Any, age: float) -> Any:
    """``record`` as if observed only from ``age``: kept only if known to
    have reached ``age`` without failing (for a history: in a known
    state), with ``age`` as its left truncation."""
    if record.left_truncation is not None and record.left_truncation >= age:
        return record
    if _is_history(record):
        return _history_enter_at(record, age)
    if record.right_truncation is not None and record.right_truncation <= age:
        return None  # it failed by then
    fields = record._fields()
    outcome, value = record.outcome, record.value
    if outcome in ("observed", "right_censored"):
        if value <= age:
            return None
    elif outcome == "left_censored":
        return None  # it may have failed before age
    elif float(value[0]) < age:
        return None  # failed in an interval that reaches back before age
    fields["left_truncation"] = age
    return _rebuild(record, fields)


def _history_enter_at(history: Any, age: float) -> Any:
    # Observation starts at age, or later where a span crosses it: what
    # happened in a span is known only as a whole, so it can be counted
    # from the span's end.
    spans = _spans(history)
    while True:
        crossing = [hi for lo, hi in spans if lo < age < hi]
        if not crossing:
            break
        age = max(crossing)
    fields = history._fields()
    events = [*history.observed, *history.interval_censored[:, 1]]
    if history.left_censored is not None:
        events.append(history.left_censored)
    seen_to = history.right_censored
    if seen_to is None:
        seen_to = max(events) if events else None
    if seen_to is None or seen_to <= age:
        return None
    if history.initial_state is not None:
        state = history.state_at(age)
        if state is None:
            return None  # its state on entry is not known
        fields["initial_state"] = state
        fields["observed_states"] = {
            a: s for a, s in history.observed_states.items() if a > age
        } or None
        fields["interval_censored_states"] = [
            r for r in history.interval_censored_states if r[0] >= age
        ] or None
        fields.pop("left_censored_states", None)
    typed = history.observed_event_types
    keep = [k for k, a in enumerate(history.observed) if a > age]
    fields["observed"] = [history.observed[k] for k in keep]
    if typed is not None:
        fields["observed_event_types"] = [typed[k] for k in keep]
    pairs = history.interval_censored
    inside = [k for k, (lo, hi) in enumerate(pairs) if lo >= age]
    fields["interval_censored"] = [pairs[k].tolist() for k in inside]
    fields["interval_censored_counts"] = [
        int(history.interval_censored_counts[k]) for k in inside
    ]
    if history.interval_censored_event_types is not None:
        fields["interval_censored_event_types"] = [
            history.interval_censored_event_types[k] for k in inside
        ]
    for name in (
        "left_censored",
        "left_censored_counts",
        "left_censored_event_types",
    ):
        fields.pop(name, None)
    if fields.get("right_censored") is None:
        fields["right_censored"] = seen_to
    fields["left_truncation"] = age
    # Selected on a failure by tr, some of whose window is now before the
    # start of observation: not accounted for (warned of by the dataset)
    fields.pop("right_truncation", None)
    for name in (
        "observed",
        "observed_event_types",
        "interval_censored",
        "interval_censored_counts",
        "interval_censored_event_types",
    ):
        if name in fields and not fields[name]:
            fields.pop(name)
    return _rebuild(history, fields)


def select(records: list, ids: Any, where: "Callable | None") -> list:
    """The records with an id in ``ids`` (all, where None) for which
    ``where(record)`` is true (all, where None)."""
    from surpyval.utils.item_history import item_id

    if ids is not None:
        wanted = [item_id(i) for i in ids]
        have = {r.id for r in records}
        missing = [i for i in wanted if i not in have]
        if missing:
            raise KeyError(f"No items {missing!r} in the data.")
        wanted_set = set(wanted)
        records = [r for r in records if r.id in wanted_set]
    if where is not None:
        records = [r for r in records if where(r)]
    return records


def read_only(value: Any) -> Any:
    """``value``, where it is a writeable array, as a read-only view of
    it: the dataset's arrays cannot be changed in place. (An array that
    came from the caller is copied where it came in, so changing theirs
    does not change the dataset either.)"""
    if isinstance(value, np.ndarray) and value.flags.writeable:
        value = value.view()
        value.flags.writeable = False
    return value


def _failures(k: int) -> str:
    """'1 failure has' or 'k failures have'."""
    return f"{k} failure has" if k == 1 else f"{k} failures have"


class ReadOnlyArrays:
    """A dataset whose arrays are read-only (#927). Changing one in place
    (``data.c[2] = 1``) used to leave the arrays split from it (``x_o``,
    ``x_r``, ...) and the summary out of step with it, so two fits of the
    same data could differ; a changed dataset is made with the reshaping
    methods, or from changed arrays."""

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, read_only(value))

    def __setstate__(self, state: dict) -> None:
        for name, value in state.items():
            object.__setattr__(self, name, read_only(value))


def cut_age(method: str, age: Any, data: Any) -> float:
    """The ``age`` a cut-off (``censor_at``, ``enter_at``) is made at: one
    finite number, in the data's time unit (#932). A date, a duration, text
    or a per-item mapping is refused, saying what to give; so is an age at
    or before 0 where the data's ages are measured from 0 (every time and
    truncation bound at or above 0), which would leave nothing known of
    any item."""
    import datetime

    if isinstance(age, dict):
        raise ValueError(
            f"{method} takes one age for every item; a cut-off per item is "
            f"{method}({{id: age}}) (the items it leaves out are kept as "
            "they are)."
        )
    if isinstance(age, (datetime.timedelta, np.timedelta64)):
        raise ValueError(
            f"{method} takes an age (a number, in the data's time unit), "
            "not a duration: convert it, e.g. duration / "
            "pd.Timedelta(hours=1) for hours."
        )
    if isinstance(age, (datetime.date, np.datetime64)) or type(
        age
    ).__name__ in ("Timestamp", "datetime"):
        raise ValueError(
            f"{method} takes an age (a number, in the data's time unit), "
            "not a date: the same date is a different age for each item. "
            "Convert it to each item's age (date - its start date) and "
            f"give {method}({{id: age}})."
        )
    if isinstance(age, (str, bytes)) or np.ndim(age) != 0:
        raise ValueError(
            f"{method} takes one age (a number, in the data's time unit); "
            f"got {age!r}."
        )
    try:
        value = float(age)
    except (TypeError, ValueError):
        raise ValueError(
            f"{method} takes one age (a number, in the data's time unit); "
            f"got {age!r}."
        )
    if not np.isfinite(value):
        raise ValueError(f"{method} takes a finite age; got {value!r}.")
    if value <= 0 and _ages_from_zero(data):
        raise ValueError(
            f"{method}({value:g}): the data's ages are measured from 0 "
            "(none is below it), so a cut-off at or before 0 is before any "
            "item's observation, which would leave nothing. Give an age "
            "above 0."
        )
    return value


def _ages_from_zero(data: Any) -> bool:
    """Whether every time and truncation bound of ``data`` is at or above
    0: its ages are measured from 0, as a lifetime's are (a baseline
    defined on the real line may put times below 0)."""
    x = np.asarray(data.x, dtype=float)
    tl = np.asarray(data.tl, dtype=float)
    return bool((x >= 0).all() and (tl[np.isfinite(tl)] >= 0).all())


def per_item_ages(method: str, ages: Any, data: Any) -> dict:
    """``ages`` as ``{id: age}`` for a cut-off per item (#937): every id
    in the data, every age a finite number."""
    from surpyval.utils.item_history import item_id

    ids = getattr(data, "ids", None)
    if ids is None:
        raise ValueError(
            f"{method}({{id: age}}) needs item ids, and this data has none "
            "(its rows are sorted and grouped): build it with from_items, "
            "ids=, or from_observations(..., <kind>_ids=...)."
        )
    have = set(ids)
    out = {}
    for key, age in ages.items():
        key = item_id(key)
        if key not in have:
            raise KeyError(f"No item {key!r} in the data.")
        out[key] = cut_age(method, age, data)
    return out


def id_list(ids: Any, data: Any) -> list:
    """``ids`` for ``select`` as a list of ids (#932): a str, a number
    or an id of the data's is one id."""
    from surpyval.utils.item_history import item_id

    have = getattr(data, "ids", None)
    if isinstance(ids, (str, bytes)) or np.ndim(ids) == 0:
        return [item_id(ids)]
    if isinstance(ids, tuple) and have is not None and ids in set(have):
        return [ids]
    return [item_id(i) for i in ids]


class ReshapeMixin(ReadOnlyArrays):
    """The reshaping operations of a dataset (``SurpyvalData``,
    ``EventHistoryData``); each returns a new dataset, built again from
    the reshaped item records, or, for lifetime data, from its arrays
    (:mod:`surpyval.utils.lifetime_reshape`), which gives the same."""

    _PARTS: dict = {}

    def _records(self) -> tuple[list, list]:  # pragma: no cover - abstract
        raise NotImplementedError

    def _from_records(
        self, records: list, counts: list
    ) -> Any:  # pragma: no cover
        raise NotImplementedError

    def _map(self, fn: Callable) -> Any:
        records, counts = self._records()
        kept = [(fn(r), k) for r, k in zip(records, counts)]
        kept = [(r, k) for r, k in kept if r is not None]
        if not kept:
            raise ValueError("No items are left.")
        return self._from_records([r for r, _ in kept], [k for _, k in kept])

    @property
    def _on_arrays(self) -> bool:
        """Whether the reshapes work on the arrays (lifetime data)."""
        return self._PARTS is LIFETIME_PARTS

    def _event_types(self) -> set:
        e = getattr(self, "e", None)
        if e is None:
            raise ValueError("This data has no event types.")
        return {v for v in e if v is not None}

    def without(self, *parts: Any) -> Any:
        """
        The data without some of what it holds, to fit a model that does
        not take it: any of ``"covariates"``, ``"event_types"``,
        ``"time_varying_covariates"``, ``"truncation"`` (and ``"states"``
        for event history data), as separate arguments or a list. A plain
        ``Weibull.fit`` of data built with covariates and failure modes,
        say, is ``Weibull.fit(data.without("covariates", "event_types"))``.
        """
        if len(parts) == 1 and isinstance(parts[0], (list, tuple, set)):
            parts = tuple(parts[0])
        check_parts(parts, self._PARTS)
        if self._on_arrays:
            from surpyval.utils import lifetime_reshape

            return lifetime_reshape.without(self, parts)
        return self._map(lambda r: without(r, parts))

    def with_covariates(self, columns: Any) -> Any:
        """The data with only the covariate ``columns`` -- their names,
        where the covariates are named (``covariate_names``), or their
        column numbers -- in that order, to fit a regression on some of
        the covariates: ``data.with_covariates(["age", "arm"])``."""
        if np.isscalar(columns):
            columns = [columns]
        columns = list(columns)
        if getattr(self, "Z", None) is None:
            raise ValueError("This data has no covariates.")
        if not columns:
            raise ValueError(
                "with_covariates takes the columns to keep: give at least "
                "one, or use without('covariates') for none."
            )
        names = getattr(self, "covariate_names", None)
        unknown = [
            c
            for c in columns
            if not isinstance(c, (int, np.integer))
            and names is not None
            and c not in names
        ]
        if unknown:
            raise ValueError(
                f"This data has no covariates {unknown!r}; its covariates "
                f"are {list(names or [])!r}."
            )
        if self._on_arrays:
            from surpyval.utils import lifetime_reshape

            return lifetime_reshape.with_covariates(self, columns)
        return self._map(lambda r: with_covariates(r, columns))

    def for_event_type(self, event_type: Any, masked: str = "censor") -> Any:
        """
        The data of one failure mode, ``event_type`` (or of several taken
        as one, given as a list or set), as for a cause-specific analysis:
        for a lifetime, a failure of another mode becomes right censored
        at its age (at the last good inspection, for one found between
        inspections; one known only to have happened before an age is
        left out); for a history, the other modes' failures are removed
        and its observation runs at least to the last of them. The result
        has no event types.

        Lifetimes lose their right truncation, with a warning: an item is
        in right-truncated data because its first failure, of any mode,
        came by the truncation age, and a fit of one mode cannot account
        for a selection that depends on every mode, so the result is
        treated as not right truncated (an approximation). Kept, the
        truncation would turn each other mode's failure into a failure of
        this one before the truncation age. Left truncation is kept.

        Failures whose type was not recorded (masked causes) may or may
        not be of this type; ``masked`` says how they are taken:
        ``"censor"`` (the default) as not of it, with a warning that its
        rate is underestimated if some were; ``"count"`` as of it (then
        overestimated if some were not); ``"refuse"`` refuses the data.
        """
        types = self._event_types()
        wanted = (
            set(event_type)
            if isinstance(event_type, (list, set, frozenset))
            else {event_type}
        )
        if not wanted:
            raise ValueError("for_event_type takes at least one event type.")
        missing = [t for t in wanted if t not in types]
        if missing:
            raise ValueError(
                "No failures of type "
                + ", ".join(repr(t) for t in sorted(missing, key=str))
                + "; the data's types are "
                + ", ".join(repr(t) for t in sorted(types, key=str))
                + "."
            )
        options = ("censor", "count", "refuse")
        if masked not in options:
            raise ValueError(
                f"masked must be one of {options!r}; got {masked!r}."
            )
        what = (
            repr(event_type)
            if len(wanted) == 1
            else "one of "
            + ", ".join(repr(t) for t in sorted(wanted, key=str))
        )
        e, c = getattr(self, "e"), np.asarray(getattr(self, "c"))
        n = np.asarray(getattr(self, "n"))
        unrecorded = int(
            sum(n[k] for k in range(len(c)) if c[k] != 1 and e[k] is None)
        )
        if unrecorded and masked == "refuse":
            raise ValueError(
                f"{_failures(unrecorded)} no recorded event type (masked "
                f"causes), which may or may not be of type {what}: "
                "relabel them (relabel_event_types({None: ...})), leave "
                "their items out, or choose masked='censor' or 'count'."
            )
        if unrecorded and masked == "censor":
            warnings.warn(
                f"{unrecorded} failure{'' if unrecorded == 1 else 's'} of no "
                "recorded event type (masked causes) "
                f"{'is' if unrecorded == 1 else 'are'} taken as not of type "
                f"{what}: its rate "
                "is underestimated if some of them were. masked='count' "
                "takes them as of this type instead.",
                MaskedCauseWarning,
                stacklevel=caller_stacklevel(),
            )
        if np.isfinite(getattr(self, "tr")).any():
            warnings.warn(
                "for_event_type drops the data's right truncation: an item "
                "is in right-truncated data because it failed, of any mode, "
                "by the truncation age, which a fit of one mode cannot "
                "account for, so the result is treated as not right "
                "truncated (an approximation). Left truncation is kept.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        if self._on_arrays:
            from surpyval.utils import lifetime_reshape

            return lifetime_reshape.for_event_type(self, wanted, masked)
        if len(wanted) > 1:
            # Several modes as one: relabelled to one of them first
            one = next(iter(wanted))
            merged = self.relabel_event_types({t: one for t in wanted})
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return merged._map(lambda r: for_event_type(r, one, masked))
        return self._map(lambda r: for_event_type(r, event_type, masked))

    def relabel_event_types(self, mapping: dict) -> Any:
        """The data with its event types renamed by ``mapping`` (``{old:
        new}``): to merge failure modes, map several to one."""
        from surpyval.utils.item_history import missing_type

        # A missing key (None, NaN, "") names the masked causes
        mapping = {
            None if missing_type(k) else k: v for k, v in mapping.items()
        }
        types = self._event_types()
        unknown = [k for k in mapping if k is not None and k not in types]
        if unknown:
            raise ValueError(
                f"No failures of type(s) {unknown!r}; the data's types are "
                + ", ".join(repr(t) for t in sorted(types, key=str))
                + "."
            )
        if self._on_arrays:
            from surpyval.utils import lifetime_reshape

            return lifetime_reshape.relabel(self, mapping)
        return self._map(lambda r: relabel(r, mapping))

    def _levels(self, name: str, every: bool = True) -> list:
        """Each item's value of the attribute or covariate ``name``, in
        the order of the rows (lifetimes) or of ``ids`` (histories);
        where ``every``, an item without the attribute is refused, else
        its value is None."""
        attributes = getattr(self, "attributes", None)
        names = getattr(self, "covariate_names", None)
        if attributes is not None and any(
            a is not None and name in a for a in attributes
        ):
            missing = [
                i
                for i, a in zip(self._item_keys(), attributes)
                if a is None or a.get(name) is None
            ]
            if missing and every:
                from surpyval.utils.warnings import named

                raise ValueError(
                    f"Items {named(missing)} have no attribute {name!r}: "
                    "give it for every item, or select the items that "
                    "have it first."
                )
            return [None if a is None else a.get(name) for a in attributes]
        if names is not None and name in names:
            return self._covariate_levels(list(names).index(name))
        have = sorted(
            {k for a in attributes or [] if a is not None for k in a}, key=str
        )
        raise ValueError(
            f"No attribute or covariate {name!r}: the data's attributes are "
            + (", ".join(repr(k) for k in have) if have else "none")
            + " and its covariates "
            + (", ".join(repr(k) for k in names) if names else "none")
            + " (attributes={name: value} on the records, or "
            "from_dataframe(..., attribute_cols=[...]))."
        )

    def _item_keys(self) -> list:
        """One key per item, for messages: its id, else its row."""
        ids = getattr(self, "ids", None)
        if ids is not None:
            return list(ids)
        return list(range(len(getattr(self, "x"))))

    def _covariate_levels(self, column: int) -> list:
        raise NotImplementedError  # pragma: no cover

    def _rows_of_items(self, keep: np.ndarray) -> Any:
        raise NotImplementedError  # pragma: no cover

    def groupby(self, name: str) -> dict:
        """
        The data split by the attribute (or covariate) ``name``:
        ``{level: dataset}``, one dataset per value, in sorted order, for
        a fit per site, arm or product line:
        ``{site: Weibull.fit(part) for site, part in data.groupby("site")
        .items()}``. Every item needs the attribute.

        Examples
        --------
        >>> from surpyval import ItemLifetime, SurpyvalData
        >>> data = SurpyvalData.from_items([
        ...     ItemLifetime("A", observed=5, attributes={"site": "N"}),
        ...     ItemLifetime("B", observed=7, attributes={"site": "S"}),
        ...     ItemLifetime("C", right_censored=9, attributes={"site": "N"}),
        ... ])
        >>> {site: part.ids for site, part in data.groupby("site").items()}
        {'N': ['A', 'C'], 'S': ['B']}
        """
        levels = self._levels(name)
        out = {}
        for level in sorted(set(levels), key=str):
            keep = np.array([v == level for v in levels], dtype=bool)
            out[level] = self._rows_of_items(keep)
        return out

    def select(self, ids: Any = None, where: Any = None) -> Any:
        """
        Some of the items: those with an id in ``ids`` (one id, or a list
        of them; the items stay in the data's order), and for which
        ``where`` holds: a function of each item's record (an
        ``ItemLifetime`` or an ``ItemHistory``),
        ``data.select(where=lambda item: item.covariates[0] == 1)``, or a
        dict ``{name: value}`` of attributes (or covariates) the item
        must have, ``data.select(where={"site": "North"})``. Selecting by
        id needs item ids (``from_items``, ``ids=``).
        """
        if isinstance(where, dict):
            keep = np.ones(len(self._item_keys()), dtype=bool)
            for name, value in where.items():
                levels = self._levels(name, every=False)
                keep &= np.array([v == value for v in levels], dtype=bool)
            if ids is not None:
                wanted_ids = set(id_list(ids, self))
                keep &= np.array(
                    [k in wanted_ids for k in self._item_keys()], dtype=bool
                )
            if not keep.any():
                raise ValueError("No items are left.")
            return self._rows_of_items(keep)
        if ids is None and where is None:
            raise ValueError(
                "select takes the ids to keep (ids=), a test of each item's "
                "record (where=), or both."
            )
        have = getattr(self, "ids", None)
        rows: "np.ndarray | None" = None
        if ids is not None:
            if have is None:
                raise ValueError(
                    "select(ids=...) needs item ids, and this data has none "
                    "(its rows are sorted and grouped, so their numbers "
                    "would not be stable): build it with from_items, "
                    "ids=, or from_observations(..., <kind>_ids=...), or "
                    "select by where=."
                )
            wanted = id_list(ids, self)
            positions: dict = {}
            for k, i in enumerate(have):
                positions.setdefault(i, k)
            missing = [i for i in wanted if i not in positions]
            if missing:
                raise KeyError(
                    f"No item{'s' if len(missing) > 1 else ''} "
                    + ", ".join(repr(i) for i in missing)
                    + " in the data."
                )
            wanted_set = set(wanted)
            rows = np.array(
                [k for k, i in enumerate(have) if i in wanted_set], dtype=int
            )
        if where is None and rows is not None and self._on_arrays:
            from surpyval.utils import lifetime_reshape

            return lifetime_reshape.select(self, rows)
        records, counts = self._records()
        if rows is not None:
            kept_rows = set(rows.tolist())
            pairs = [
                (r, k)
                for j, (r, k) in enumerate(zip(records, counts))
                if j in kept_rows
            ]
        else:
            pairs = list(zip(records, counts))
        if where is not None:
            pairs = [(r, k) for r, k in pairs if where(r)]
        if not pairs:
            raise ValueError("No items are left.")
        return self._from_records([r for r, _ in pairs], [k for _, k in pairs])

    def censor_at(self, age: Any) -> Any:
        """
        The data as seen if observation stopped at ``age`` (administrative
        censoring, as at the end of a study or a warranty): what happened
        after ``age`` is no longer seen. A failure found between
        inspections either side of ``age`` becomes right censored at the
        last good inspection, since it may have come after ``age``; an item
        that came under observation at or after ``age`` is left out.

        ``age`` is one age for every item, in the data's time unit;
        ``{id: age}``, a cut-off per item (the items it leaves out are
        kept as they are); or a function of each item's record giving
        its cut-off: a study that ends on one date ends at a different
        age for each item, ``data.censor_at(lambda item: (end -
        item.attribute("installed")).days)``.

        A history's failures counted between two inspections, or a change
        of state between two, are known only as a whole: where ``age``
        falls between those inspections, the history is cut at the earlier
        one (and left out where that is before its first look).
        """
        if isinstance(age, dict):
            cuts = per_item_ages("censor_at", age, self)
            return self._map(
                lambda r: censor_at(r, cuts[r.id]) if r.id in cuts else r
            )
        if callable(age):
            return self._map(
                lambda r: censor_at(r, cut_age("censor_at", age(r), self))
            )
        value = cut_age("censor_at", age, self)
        if self._on_arrays:
            from surpyval.utils import lifetime_reshape

            return lifetime_reshape.censor_at(self, value)
        return self._map(lambda r: censor_at(r, value))

    def enter_at(self, age: Any) -> Any:
        """
        The data as if observed only from ``age`` (left truncation): only
        items known to have reached ``age`` without failing -- for a
        history, in a known state -- stay in, with ``age`` as the start of
        their observation; a failure found between inspections either side
        of ``age``, or known only to have happened before a look, may have
        come before it, so its item is left out. ``age`` is one age for
        every item, ``{id: age}``, an entry per item (the items it leaves
        out are kept as they are), or a function of each item's record
        giving its entry.

        A history goes on after its failures, so where ``age`` falls
        between two inspections that count failures, or a change of state,
        or before a first look, its observation starts instead at the later
        inspection, from which it is known again.
        """
        if isinstance(age, dict):
            cuts = per_item_ages("enter_at", age, self)
        elif callable(age):
            cuts = {
                r.id: cut_age("enter_at", age(r), self)
                for r in self._records()[0]
            }
        else:
            value = cut_age("enter_at", age, self)
            cuts = None
        if (
            self._PARTS is not LIFETIME_PARTS
            and np.isfinite(getattr(self, "tr")).any()
        ):
            warnings.warn(
                "enter_at drops the histories' right truncation: an item is "
                "in the data because it failed by its truncation age, "
                "possibly before the new start of observation, which the "
                "fit cannot account for, so the result is treated as not "
                "right truncated (an approximation).",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        if cuts is not None:
            return self._map(
                lambda r: enter_at(r, cuts[r.id]) if r.id in cuts else r
            )
        if self._on_arrays:
            from surpyval.utils import lifetime_reshape

            return lifetime_reshape.enter_at(self, value)
        return self._map(lambda r: enter_at(r, value))
