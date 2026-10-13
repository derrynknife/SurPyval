"""Deprecated: ``RecurrentEventData`` is now
:class:`~surpyval.utils.event_history_data.EventHistoryData`, in
``surpyval.utils.event_history_data`` (v0.25): it holds any item's event
history (recurrent events, changes of state, ...), not recurrent events
only. The old name and this module are removed in v0.26."""

from typing import Any


def __getattr__(name: str) -> Any:
    from surpyval.utils import event_history_data
    from surpyval.utils.deprecation import renamed_name

    if name == "RecurrentEventData":
        return renamed_name(
            "surpyval.utils.recurrent_event_data",
            "RecurrentEventData",
            "surpyval.utils.event_history_data.EventHistoryData",
            event_history_data.EventHistoryData,
        )
    try:
        return getattr(event_history_data, name)
    except AttributeError:
        raise AttributeError(
            f"module 'surpyval.utils.recurrent_event_data' has no attribute "
            f"{name!r}"
        ) from None
