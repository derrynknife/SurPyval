"""The dict and JSON forms of the data classes (#935): each record and
dataset writes a dictionary of plain Python values, strict JSON when
dumped (``to_dict``, ``to_json``), tagged with its class, and reads it
back (``from_dict``, ``from_json``); ``surpyval.from_dict`` and
``surpyval.from_json`` dispatch on the tag, as they do for models.

Ids and event types that are tuples (a gap's ``("A", 2)``) are written
as ``{"tuple": [...]}``, which JSON's lists would otherwise lose. A
step schedule of time-varying covariates is written by its ages and
values; a ``CovariatePath`` is a function and has no JSON form.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from surpyval.serialisation import (
    decode_non_finite,
    encode_non_finite,
    read_json,
)

# The ``"class"`` tags of the data dictionaries, and their readers.
DATA_CLASSES = {
    "SurpyvalData": "surpyval.utils.surpyval_data",
    "EventHistoryData": "surpyval.utils.event_history_data",
    "ItemLifetime": "surpyval.utils.item_lifetime",
    "ItemHistory": "surpyval.utils.item_history",
}


def label(value: Any) -> Any:
    """An id or event type as JSON keeps it: a tuple as
    ``{"tuple": [...]}``, a numpy value as the plain value it holds."""
    if isinstance(value, tuple):
        return {"tuple": [label(v) for v in value]}
    return value.item() if isinstance(value, np.generic) else value


def label_back(value: Any) -> Any:
    """An id or event type read back from JSON (:func:`label`)."""
    if isinstance(value, dict) and set(value) == {"tuple"}:
        return tuple(label_back(v) for v in value["tuple"])
    return value


def schedule_dict(schedule: Any, as_dict: bool) -> dict:
    """A record's time-varying covariates as a dictionary: a step
    schedule's ages and values (``{"ages": [...], "values": [[...]]}``
    where it was given as a ``{age: values}`` dict, else its ``edges``,
    ``Z`` and ``period``); a ``CovariatePath`` is refused."""
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if not isinstance(schedule, StepSchedule):
        raise ValueError(
            "Time-varying covariates given as a CovariatePath (a function) "
            "have no dict or JSON form: keep the record with pickle, or "
            "give the covariates as a StepSchedule or {age: values}."
        )
    if as_dict and schedule.period is None:
        return {
            "ages": schedule.edges[:-1].tolist(),
            "values": schedule.Z.tolist(),
        }
    return {
        "edges": schedule.edges.tolist(),
        "Z": schedule.Z.tolist(),
        "period": schedule.period,
    }


def schedule_back(value: Any) -> Any:
    """The time-varying covariates read back (:func:`schedule_dict`): a
    ``{age: values}`` dict, or a StepSchedule."""
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    if not isinstance(value, dict):
        raise ValueError(
            "'time_varying_covariates' must be a dict with 'ages' and "
            "'values', or 'edges', 'Z' and 'period'."
        )
    if "ages" in value:
        return {
            float(age): list(row)
            for age, row in zip(value["ages"], value["values"])
        }
    return StepSchedule(
        np.array(value["edges"], dtype=float),
        np.array(value["Z"], dtype=float),
        value.get("period"),
    )


def record_dict(record: Any) -> dict:
    """``record``'s ``to_dict``: its class, id and fields, JSON-safe."""
    fields = record._fields()
    out: dict = {"class": type(record).__name__, "id": label(record.id)}
    for name, value in fields.items():
        if name == "time_varying_covariates":
            # The schedule the record keeps (its field shows a dict where
            # one was given)
            out[name] = schedule_dict(
                record.time_varying_covariates, record._as_dict()
            )
        elif name.endswith("event_types") or name == "event_type":
            out[name] = (
                [label(v) for v in value]
                if isinstance(value, list)
                else label(value)
            )
        elif name == "covariates":
            out[name] = value  # a list, or {name: value}
        elif name in ("observed_states", "left_censored_states"):
            out[name] = [[float(age), label(st)] for age, st in value.items()]
        elif name == "interval_censored_states":
            out[name] = [[lo, hi, label(st)] for lo, hi, st in value]
        elif name == "initial_state":
            out[name] = label(value)
        else:
            out[name] = value
    extra = getattr(record, "unit", None)
    if extra is not None:
        out["unit"] = extra
    return encode_non_finite(out)


def record_from_dict(cls: Any, record_dict_: dict) -> Any:
    """A record of ``cls`` read back from its ``to_dict``."""
    if not isinstance(record_dict_, dict):
        raise ValueError(
            f"Expected a serialised {cls.__name__} dict, got "
            f"{type(record_dict_).__name__}."
        )
    fields = decode_non_finite(dict(record_dict_))
    tag = fields.pop("class", None)
    if tag is not None and tag != cls.__name__:
        raise ValueError(
            f"This dictionary was written by {tag}, not {cls.__name__}: "
            f"read it with {tag}.from_dict, or surpyval.from_dict."
        )
    if "id" not in fields:
        raise ValueError(f"A serialised {cls.__name__} needs its 'id'.")
    item = label_back(fields.pop("id"))
    kwargs: dict = {}
    for name, value in fields.items():
        if name == "time_varying_covariates":
            kwargs[name] = schedule_back(value)
        elif name.endswith("event_types") or name == "event_type":
            kwargs[name] = (
                [label_back(v) for v in value]
                if isinstance(value, list)
                else label_back(value)
            )
        elif name in ("observed_states", "left_censored_states"):
            kwargs[name] = {float(age): label_back(st) for age, st in value}
        elif name == "interval_censored_states":
            kwargs[name] = [[lo, hi, label_back(st)] for lo, hi, st in value]
        elif name == "initial_state":
            kwargs[name] = label_back(value)
        else:
            kwargs[name] = value
    return cls(item, **kwargs)


def dump(data_dict: dict, filepath: "str | os.PathLike | None") -> Any:
    """``data_dict`` as strict JSON text, or written to ``filepath``."""
    if filepath:
        Path(filepath).write_text(
            json.dumps(data_dict, indent=2, allow_nan=False)
        )
        return None
    return json.dumps(data_dict, allow_nan=False)


def load(source: Any) -> dict:
    """The dictionary in ``source``: JSON text (a string starting with
    ``{``), or the path of a file holding it."""
    loaded = read_json(source)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"Expected a serialised data dict, got {type(loaded).__name__}."
        )
    return loaded


def reader_of(data_dict: dict) -> Any:
    """The class whose ``from_dict`` reads ``data_dict``, by its
    ``"class"`` tag; None where it has no data tag."""
    import importlib

    tag = data_dict.get("class")
    if not isinstance(tag, str) or tag not in DATA_CLASSES:
        return None
    return getattr(importlib.import_module(DATA_CLASSES[tag]), tag)
