"""One item's lifetime, :class:`ItemLifetime`; a list of them is a dataset
(:meth:`SurpyvalData.from_items
<surpyval.utils.surpyval_data.SurpyvalData.from_items>`)."""

from __future__ import annotations

from typing import Any

import numpy as np

from surpyval.utils.item_history import (
    _age,
    _covariate_field,
    _covariates,
    _schedule_field,
    _time_varying,
    _with_step,
    freeze,
    frozen_setattr,
    frozen_setstate,
    hashable,
    item_id,
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
        value holding from its age on, the first at or before the start of
        observation), a
        :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
        or a continuously varying
        :class:`~surpyval.univariate.regression.tvc_path.CovariatePath`.

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
    ) -> None:
        item = item_id(id)
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
        if event_type is not None and outcome == "right_censored":
            raise ValueError(
                f"Item {item!r}: a right censored item has not failed, so it "
                "has no event_type."
            )
        self.event_type = event_type
        self.covariates, self.covariate_names = _covariates(item, covariates)
        self.time_varying_covariates, first = _time_varying(
            item, time_varying_covariates
        )
        self._check(first)
        freeze(self)

    def __setattr__(self, name: str, value: Any) -> None:
        frozen_setattr(self, name, value)

    def __setstate__(self, state: dict) -> None:
        frozen_setstate(self, state)

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
        if first_change is not None:
            start = 0.0 if tl is None else tl
            if first_change > start:
                raise ValueError(
                    f"Item {item!r}: its time-varying covariates start at age "
                    f"{first_change:g}, after the start of its observation "
                    f"({start:g}): give their values from {start:g} on."
                )

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
        steps = _with_step(self.id, self.time_varying_covariates, age, values)
        return self.clone(time_varying_covariates=steps)

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
        if self.time_varying_covariates is not None:
            out["time_varying_covariates"] = _schedule_field(
                self.time_varying_covariates
            )
        return out

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ItemLifetime):
            return NotImplemented
        return self.id == other.id and self._fields() == other._fields()

    def __hash__(self) -> int:
        return hash((self.id, hashable(self._fields())))

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
