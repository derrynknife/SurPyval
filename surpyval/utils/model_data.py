"""A recurrent model's data as the fits work on it.

The recurrent fits work on an
:class:`~surpyval.utils.event_history_data.EventHistoryData` inside.
The package's own code reads a fitted model's data through
:func:`event_data`, not ``model.data``, so that what ``model.data``
hands users (a :class:`~surpyval.SurpyvalData` with each row's unit
``i``, the one container for several rows per unit, from v0.31) can
change without moving any number.

- :func:`event_data`, the model's data as the fits work on it;
- :func:`as_event_history`, data a user hands a diagnostic (a model's
  ``data``, or any ``SurpyvalData`` with units) in that form.
"""

from __future__ import annotations

from typing import Any

from surpyval.utils.event_history_data import EventHistoryData
from surpyval.utils.surpyval_data import SurpyvalData

__all__ = ["as_event_history", "event_data"]


def as_event_history(data: Any, what: str = "data") -> Any:
    """``data`` as the fits work on it: an ``EventHistoryData`` as it is,
    a ``SurpyvalData`` with units ``i`` read as each unit's history (as a
    recurrent ``fit`` reads it), and anything else (``None``, a model's
    own kind of data) as it is. ``what`` names the caller in a
    refusal."""
    if isinstance(data, SurpyvalData) and data.i is not None:
        from surpyval.utils.history_rows import event_history_from_rows

        return event_history_from_rows(data, what)
    return data


def event_data(model: Any) -> Any:
    """``model``'s data as the fits work on it (``None`` where it has
    none): what a recurrent model's fit kept; any other model's ``data``
    as it is."""
    held = getattr(model, "__dict__", {})
    if "_event_data" in held:
        return held["_event_data"]
    if isinstance(held.get("data"), EventHistoryData):
        return held["data"]  # kept as a plain attribute
    return getattr(model, "data", None)
