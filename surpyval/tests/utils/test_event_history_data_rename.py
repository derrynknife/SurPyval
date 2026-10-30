"""RecurrentEventData was renamed EventHistoryData in v0.25 and the old
name removed in v0.26: at the top level and in its old module, it raises
the error Python would, naming the new one. So do EventHistoryData's
``items`` and ``get_times_to_first_events``, deprecated with it."""

import importlib
import sys
import warnings

import pytest

import surpyval
from surpyval import EventHistoryData


def test_the_old_top_level_name_names_the_new_one():
    with pytest.raises(
        AttributeError,
        match="'RecurrentEventData': it was removed in v0.26; it is "
        "surpyval.EventHistoryData",
    ):
        surpyval.RecurrentEventData
    assert not hasattr(surpyval, "RecurrentEventData")
    assert "RecurrentEventData" not in surpyval.__all__


def test_from_import_of_the_old_name_names_the_new_one():
    with pytest.raises(
        ImportError,
        match="cannot import name 'RecurrentEventData': it was removed in "
        "v0.26",
    ):
        from surpyval import RecurrentEventData  # noqa: F401


def test_the_old_module_names_the_new_class():
    sys.modules.pop("surpyval.utils.recurrent_event_data", None)
    with pytest.raises(ImportError, match="surpyval.EventHistoryData"):
        importlib.import_module("surpyval.utils.recurrent_event_data")


def test_star_import_no_longer_gets_the_old_name():
    namespace: dict = {}
    with warnings.catch_warnings():
        # the data functions deprecated by date warn on a star import
        warnings.simplefilter("ignore", FutureWarning)
        warnings.simplefilter("error", DeprecationWarning)
        exec("from surpyval import *", namespace)
    assert "RecurrentEventData" not in namespace
    assert namespace["EventHistoryData"] is EventHistoryData


@pytest.mark.parametrize(
    "old, new",
    [("items", "'ids'"), ("get_times_to_first_events", "'to_first_events()'")],
)
def test_the_old_attributes_name_their_replacements(old, new):
    data = surpyval.handle_xicn([1, 2, 3], i=[1, 1, 2])
    with pytest.raises(AttributeError) as caught:
        getattr(data, old)
    message = str(caught.value)
    assert f"has no attribute '{old}'" in message
    assert "removed in v0.26" in message and new in message
    assert old not in dir(data)
    assert not hasattr(EventHistoryData, old)


def test_the_new_name_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert surpyval.EventHistoryData is EventHistoryData
