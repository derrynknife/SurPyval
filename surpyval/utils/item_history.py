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
    time_varying_covariates : dict, optional
        ``{age: values}``: covariates that change over its history, each
        value holding from its age on; the first age must be at or before
        the start of observation (``left_truncation``, else 0). No model
        uses these yet (#905): every fit refuses data that has them.

    Event types, given for some failures, are needed for all of them (the
    end of observation has none). ``markers`` (repeated measurements,
    #891) and ``states`` (multi-state paths, #887) are reserved for the
    models that will use them.

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
        time_varying_covariates: "dict | None" = None,
        observed_event_types: Any = None,
        interval_censored_event_types: Any = None,
        left_censored_event_types: Any = None,
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
        self.time_varying_covariates = _step_covariates(
            item, time_varying_covariates
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
                "give at least one."
            )
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
        if self.time_varying_covariates is not None:
            start = 0.0 if tl is None else tl
            first = self.time_varying_covariates[0][0]
            if first > start:
                raise ValueError(
                    f"Item {item!r}: its time-varying covariates start at age "
                    f"{first:g}, after the start of its observation "
                    f"({start:g}): give their values from {start:g} on."
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
            ages, values = self.time_varying_covariates
            out["time_varying_covariates"] = {
                float(age): row.tolist() for age, row in zip(ages, values)
            }
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


def _step_covariates(
    item: Any, value: "dict | None"
) -> "tuple[np.ndarray, np.ndarray] | None":
    """``{age: values}`` as sorted ``(ages, values)``, values ``(k, p)``."""
    if value is None:
        return None
    if not isinstance(value, dict) or not value:
        raise ValueError(
            f"Item {item!r}: 'time_varying_covariates' must be a dict "
            "{age: values}, the values holding from each age on."
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
    return ages[order], np.vstack(rows)[order]


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
