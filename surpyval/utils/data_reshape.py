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

from surpyval.utils.warnings import caller_stacklevel

LIFETIME_PARTS = {
    "covariates": ("covariates",),
    "event_types": ("event_type",),
    "time_varying_covariates": ("time_varying_covariates",),
    "truncation": ("left_truncation", "right_truncation"),
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
    columns = [names.index(c) if c in named else c for c in columns]
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
    """``record`` with its event types renamed by ``mapping``."""
    fields = record._fields()
    for name in (
        "event_type",
        "observed_event_types",
        "interval_censored_event_types",
        "left_censored_event_types",
    ):
        if name not in fields:
            continue
        value = fields[name]
        if isinstance(value, list):
            fields[name] = [mapping.get(v, v) for v in value]
        else:
            fields[name] = mapping.get(value, value)
    return _rebuild(record, fields)


# -- one failure mode ---------------------------------------------------------


def for_event_type(record: Any, event_type: Any) -> Any:
    """``record`` as data of one failure mode, ``event_type``: the other
    modes' failures stop being failures. A lifetime loses its right
    truncation (#923): it says the item's first failure, of any mode, came
    by that age, which says nothing of when this mode's would have come."""
    if _is_history(record):
        return _history_for_event_type(record, event_type)
    fields = record._fields()
    fields.pop("right_truncation", None)
    if record.outcome == "right_censored":
        return _rebuild(record, fields)
    if record.event_type == event_type:
        fields.pop("event_type")
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


def _history_for_event_type(history: Any, event_type: Any) -> Any:
    if not history.has_event_types:
        return history
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
    if history.right_truncation is not None:
        fields["right_truncation"] = age
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
    if record.right_truncation is not None and record.right_truncation <= age:
        return None
    if _is_history(record):
        return _history_enter_at(record, age)
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


class ReshapeMixin:
    """The reshaping operations of a dataset (``SurpyvalData``,
    ``EventHistoryData``); each returns a new dataset, built again from
    the reshaped item records."""

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

    def _event_types(self) -> set:
        e = getattr(self, "e", None)
        if e is None:
            raise ValueError("This data has no event types.")
        return {v for v in e if v is not None}

    def without(self, *parts: str) -> Any:
        """
        The data without some of what it holds, to fit a model that does
        not take it: any of ``"covariates"``, ``"event_types"``,
        ``"time_varying_covariates"``, ``"truncation"`` (and ``"states"``
        for event history data). A plain ``Weibull.fit`` of data built with
        covariates and failure modes, say, is
        ``Weibull.fit(data.without("covariates", "event_types"))``.
        """
        check_parts(parts, self._PARTS)
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
        return self._map(lambda r: with_covariates(r, columns))

    def for_event_type(self, event_type: Any) -> Any:
        """
        The data of one failure mode, ``event_type``, as for a
        cause-specific analysis: for a lifetime, a failure of another mode
        becomes right censored at its age (at the last good inspection,
        for one found between inspections; one known only to have happened
        before an age is left out); for a history, the other modes'
        failures are removed and its observation runs at least to the last
        of them. The result has no event types.

        Lifetimes lose their right truncation, with a warning: an item is
        in right-truncated data because its first failure, of any mode,
        came by the truncation age, and a fit of one mode cannot account
        for a selection that depends on every mode, so the result is
        treated as not right truncated (an approximation). Kept, the
        truncation would turn each other mode's failure into a failure of
        this one before the truncation age. Left truncation is kept.
        """
        types = self._event_types()
        if event_type not in types:
            raise ValueError(
                f"No failures of type {event_type!r}; the data's types are "
                + ", ".join(repr(t) for t in sorted(types, key=str))
                + "."
            )
        if (
            self._PARTS is LIFETIME_PARTS
            and np.isfinite(getattr(self, "tr")).any()
        ):
            warnings.warn(
                "for_event_type drops the data's right truncation: an item "
                "is in right-truncated data because its first failure, of "
                "any mode, came by the truncation age, which a fit of one "
                "mode cannot account for, so the result is treated as not "
                "right truncated (an approximation). Left truncation is "
                "kept.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
        return self._map(lambda r: for_event_type(r, event_type))

    def relabel_event_types(self, mapping: dict) -> Any:
        """The data with its event types renamed by ``mapping`` (``{old:
        new}``): to merge failure modes, map several to one."""
        types = self._event_types()
        unknown = [k for k in mapping if k not in types]
        if unknown:
            raise ValueError(
                f"No failures of type(s) {unknown!r}; the data's types are "
                + ", ".join(repr(t) for t in sorted(types, key=str))
                + "."
            )
        return self._map(lambda r: relabel(r, mapping))

    def select(self, ids: Any = None, where: "Callable | None" = None) -> Any:
        """
        Some of the items: those with an id in ``ids``, and for which
        ``where(item)`` is true (``where`` is given each item's record, an
        ``ItemLifetime`` or an ``ItemHistory``):
        ``data.select(where=lambda item: item.covariates[0] == 1)``.
        """
        records, counts = self._records()
        kept = select(records, ids, where)
        if not kept:
            raise ValueError("No items are left.")
        index = {id(r): k for r, k in zip(records, counts)}
        return self._from_records(kept, [index[id(r)] for r in kept])

    def censor_at(self, age: Any) -> Any:
        """
        The data as seen if observation stopped at ``age`` (administrative
        censoring, as at the end of a study or a warranty): what happened
        after ``age`` is no longer seen. A failure found between
        inspections either side of ``age`` becomes right censored at the
        last good inspection, since it may have come after ``age``; an item
        that came under observation at or after ``age`` is left out.

        A history's failures counted between two inspections, or a change
        of state between two, are known only as a whole: where ``age``
        falls between those inspections, the history is cut at the earlier
        one (and left out where that is before its first look).
        """
        age = float(age)
        return self._map(lambda r: censor_at(r, age))

    def enter_at(self, age: Any) -> Any:
        """
        The data as if observed only from ``age`` (left truncation): only
        items known to have reached ``age`` without failing -- for a
        history, in a known state -- stay in, with ``age`` as the start of
        their observation; a failure found between inspections either side
        of ``age``, or known only to have happened before a look, may have
        come before it, so its item is left out.

        A history goes on after its failures, so where ``age`` falls
        between two inspections that count failures, or a change of state,
        or before a first look, its observation starts instead at the later
        inspection, from which it is known again.
        """
        age = float(age)
        return self._map(lambda r: enter_at(r, age))
