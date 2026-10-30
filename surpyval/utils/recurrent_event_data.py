"""Removed in v0.26: ``RecurrentEventData`` is
:class:`~surpyval.utils.event_history_data.EventHistoryData` (renamed in
v0.25), in ``surpyval.utils.event_history_data``.

Importing this module raises an ``ImportError`` that says so, rather than
Python's bare "No module named 'surpyval.utils.recurrent_event_data'".
"""

raise ImportError(
    "surpyval.utils.recurrent_event_data was removed in v0.26; its "
    "RecurrentEventData is surpyval.EventHistoryData "
    "(from surpyval import EventHistoryData).",
    name=__name__,
)
