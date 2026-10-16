"""Predictions from item records (#940): a fitted model predicts for
each record of a dataset (or a list of records, or one record), from the
record's own covariates where the model has them, as a DataFrame
indexed by the item's id, with the option of conditioning on each
item's current age (a right censored item is still running).

``surpyval.utils.regression_data.named_row`` lets a record, a list of
records or a dataset stand as ``Z`` in a regression model's ``sf``,
``ff``, ``hf``, ``Hf``, ``df``, ``qf`` and ``mean``, so
``model.sf(t, item)`` is ``model.sf(t, item.covariates)`` in the order
the model was fitted in (by name, where the covariates are named).
"""

from __future__ import annotations

import inspect
from typing import Any

import numpy as np

# The quantities ``predict`` gives, and what each needs.
_AT_TIMES = ("sf", "ff", "hf", "Hf", "df")
_AT_P = ("qf",)
_PLAIN = ("mean",)


def _is_record(value: Any) -> bool:
    return type(value).__name__ in ("ItemLifetime", "ItemHistory")


def _is_dataset(value: Any) -> bool:
    return type(value).__name__ in ("SurpyvalData", "EventHistoryData")


def records_of(items: Any, what: str) -> list:
    """``items`` as a list of records: a dataset's records, a list of
    them, or one record (refused otherwise, naming ``what`` takes)."""
    if _is_dataset(items):
        return list(items.to_items())
    if _is_record(items):
        return [items]
    if isinstance(items, (list, tuple)):
        other = next((i for i in items if not _is_record(i)), None)
        if other is None and items:
            return list(items)
    raise TypeError(
        f"{what} takes a SurpyvalData or EventHistoryData, a list of "
        "ItemLifetime or ItemHistory records, or one record; got "
        f"{type(items).__name__}."
    )


def covariates_as_Z(items: Any, feature_names: "list | None") -> Any:
    """The covariates of ``items`` (a record, a list of records or a
    dataset) as a regression model's ``Z``: one row per record, by name
    where the model's covariates are named (a DataFrame), else in the
    records' order. A record without covariates is refused."""
    import pandas as pd

    if _is_dataset(items):
        if getattr(items, "Z", None) is None:
            raise ValueError(
                "This data has no covariates, which the model predicts "
                "from: give Z, or data built with them."
            )
        Z = np.asarray(items.Z, dtype=float)
        Z = Z[:, None] if Z.ndim == 1 else Z
        if type(items).__name__ == "EventHistoryData":
            first, _ = items.item_rows()
            Z = Z[first]
        names = getattr(items, "covariate_names", None)
        if names is not None and feature_names is not None:
            return pd.DataFrame(Z, columns=list(names))
        return Z
    records = records_of(items, "A prediction from records")
    missing = [r.id for r in records if r.covariates is None]
    if missing:
        from surpyval.utils.warnings import named

        raise ValueError(
            f"Items {named(missing)} have no covariates, which the model "
            "predicts from (covariates= on the record)."
        )
    named_rows = [r.covariate_names is not None for r in records]
    if all(named_rows) and feature_names is not None:
        return pd.DataFrame(
            [
                dict(zip(r.covariate_names, r.covariates.tolist()))
                for r in records
            ]
        )
    if any(named_rows) and feature_names is None:
        # Named on the record, unnamed in the model: by position
        pass
    rows = np.vstack([np.asarray(r.covariates, dtype=float) for r in records])
    return rows[0] if len(records) == 1 and _is_record(items) else rows


def _current_age(record: Any) -> "float | None":
    """The age a still-running item was last seen at: a right censored
    lifetime's age, or a history's end of observation; None for an item
    that has failed (a lifetime observed, left or interval censored)."""
    if type(record).__name__ == "ItemHistory":
        return record._end()
    return record.value if record.outcome == "right_censored" else None


def predict_records(
    model: Any,
    items: Any,
    times: Any = None,
    what: Any = "sf",
    given: Any = None,
    p: Any = None,
) -> Any:
    """The predictions of ``model`` for ``items`` as a DataFrame; see
    :meth:`PredictsRecordsMixin.predict`."""
    import pandas as pd

    records = records_of(items, "predict")
    wanted = [what] if isinstance(what, str) else list(what)
    known = _AT_TIMES + _AT_P + _PLAIN
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise ValueError(
            f"what must be among {known!r}; got "
            + ", ".join(repr(w) for w in unknown)
            + "."
        )
    if given not in (None, "age"):
        raise ValueError(
            "given takes None (the predictions as for a new item) or "
            f"'age' (given each item's current age); got {given!r}."
        )
    at_times = [w for w in wanted if w in _AT_TIMES]
    if at_times and times is None:
        raise ValueError(
            f"times= is needed for {', '.join(at_times)}: the ages to "
            "predict at."
        )
    if "qf" in wanted and p is None:
        raise ValueError("p= is needed for qf: the probabilities.")
    if given == "age" and any(w in wanted for w in _AT_P + _PLAIN):
        raise ValueError(
            "given='age' conditions the functions of time (sf, ff, hf, Hf, "
            "df) on each item's current age; qf and mean are not given "
            "an age here."
        )
    takes_Z = "Z" in inspect.signature(model.sf).parameters
    Z = None
    if takes_Z:
        Z = covariates_as_Z(records, getattr(model, "feature_names", None))
        Z = (
            np.atleast_2d(np.asarray(Z, dtype=float))
            if not hasattr(Z, "columns")
            else Z
        )

    def row_Z(k: int) -> Any:
        if Z is None:
            return None
        return Z.iloc[[k]] if hasattr(Z, "columns") else Z[k]

    def call(name: str, x: Any, k: int) -> np.ndarray:
        fn = getattr(model, name)
        if Z is None:
            return np.asarray(fn(x), dtype=float)
        return np.asarray(fn(x, row_Z(k)), dtype=float)

    times_arr = (
        None
        if times is None
        else np.atleast_1d(np.asarray(times, dtype=float))
    )
    p_arr = None if p is None else np.atleast_1d(np.asarray(p, dtype=float))
    frames = []
    for k, record in enumerate(records):
        age = _current_age(record) if given == "age" else None
        failed = given == "age" and age is None
        columns: dict = {}
        length = 1
        if times_arr is not None:
            columns["time"] = times_arr
            length = len(times_arr)
        elif p_arr is not None and "qf" in wanted:
            columns["p"] = p_arr
            length = len(p_arr)
        if given == "age":
            columns["age"] = np.full(length, np.nan if age is None else age)
        for name in wanted:
            if name in _AT_TIMES:
                assert times_arr is not None
                values = _conditional(
                    model, name, times_arr, k, age, failed, call
                )
            elif name == "qf":
                values = call("qf", p_arr, k)
                if length != len(values):
                    values = np.full(length, np.nan)
            else:
                mean = model.mean() if Z is None else model.mean(row_Z(k))
                values = np.full(length, float(np.asarray(mean).ravel()[0]))
            columns[name] = values
        frame = pd.DataFrame(columns)
        frame.insert(0, "id", [record.id] * length)
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True)
    return out.set_index("id")


def _conditional(
    model: Any,
    name: str,
    times: np.ndarray,
    k: int,
    age: "float | None",
    failed: bool,
    call: Any,
) -> np.ndarray:
    """``name`` at ``times`` for item ``k``: as for a new item where
    ``age`` is None; given survival to ``age`` otherwise (NaN for a
    failed item)."""
    if failed:
        return np.full(len(times), np.nan)
    values = call(name, times, k)
    if age is None:
        return values
    sf_age = float(call("sf", np.array([age]), k)[0])
    with np.errstate(divide="ignore", invalid="ignore"):
        if name == "sf":
            out = np.where(times <= age, 1.0, values / sf_age)
        elif name == "ff":
            sf = call("sf", times, k)
            out = np.where(times <= age, 0.0, 1 - sf / sf_age)
        elif name == "hf":
            out = np.where(times <= age, 0.0, values)
        elif name == "Hf":
            H_age = float(call("Hf", np.array([age]), k)[0])
            out = np.where(times <= age, 0.0, values - H_age)
        else:  # df
            out = np.where(times <= age, 0.0, values / sf_age)
    return np.asarray(out, dtype=float)


class PredictsRecordsMixin:
    """``predict`` from item records, for the lifetime models."""

    def predict(
        self,
        items: Any,
        times: Any = None,
        what: Any = "sf",
        given: Any = None,
        p: Any = None,
    ) -> Any:
        """
        Predictions for each item of ``items`` as a DataFrame, one row
        per item and time, indexed by the item's id: a model with
        covariates predicts from each record's own (by name where the
        covariates are named; see :meth:`sf`).

        Parameters
        ----------
        items : SurpyvalData, EventHistoryData, list of records or a record
            The items, as ``ItemLifetime`` or ``ItemHistory`` records.
        times : array like, optional
            The ages to predict ``sf``, ``ff``, ``hf``, ``Hf`` or ``df``
            at (a column ``time``).
        what : str or list of str, default "sf"
            Any of ``"sf"``, ``"ff"``, ``"hf"``, ``"Hf"``, ``"df"`` (at
            ``times``), ``"qf"`` (at ``p``) and ``"mean"``; one column
            each.
        given : {None, "age"}, optional
            ``"age"``: each prediction given the item's current age (a
            right censored lifetime's age, a history's end of
            observation; a column ``age``), the question for a running
            fleet: ``sf`` is ``S(t) / S(age)`` after the age and 1 up to
            it, ``Hf`` is ``H(t) - H(age)``, and so on. An item that has
            failed gets ``nan``. ``None``: as for a new item.
        p : array like, optional
            The probabilities for ``qf`` (a column ``p``).

        Returns
        -------
        pandas.DataFrame
            Indexed by ``id``, with ``time`` (or ``p``), ``age`` (with
            ``given="age"``) and a column per quantity.

        Examples
        --------
        >>> from surpyval import ItemLifetime, SurpyvalData, Weibull
        >>> model = Weibull.from_params([100, 2])
        >>> items = [
        ...     ItemLifetime("A", right_censored=50),
        ...     ItemLifetime("B", observed=30),
        ... ]
        >>> model.predict(items, times=[60, 80], given="age").round(3)
            time   age     sf
        id
        A   60.0  50.0  0.896
        A   80.0  50.0  0.677
        B   60.0   NaN    NaN
        B   80.0   NaN    NaN
        """
        return predict_records(self, items, times, what, given, p)
