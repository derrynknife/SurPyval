"""RecurrentEventData is now EventHistoryData (v0.25): the old name, at
the top level and in its old module, is the same class, with a
DeprecationWarning, until v0.26."""

import pytest

import surpyval
from surpyval import EventHistoryData
from surpyval.utils.deprecation import REMOVED_IN


def test_the_old_top_level_name_warns_and_is_the_same_class():
    with pytest.warns(DeprecationWarning, match="EventHistoryData") as caught:
        old = surpyval.RecurrentEventData
    assert old is EventHistoryData
    assert f"v{REMOVED_IN}" in str(caught[0].message)
    assert caught[0].filename == __file__
    data = surpyval.handle_xicn([1, 2, 3], i=[1, 1, 1])
    assert isinstance(data, old)


def test_the_old_module_path_warns_and_forwards():
    with pytest.warns(DeprecationWarning, match="EventHistoryData"):
        from surpyval.utils.recurrent_event_data import RecurrentEventData
    assert RecurrentEventData is EventHistoryData
    from surpyval.utils import recurrent_event_data

    assert (
        recurrent_event_data.accepts_event_history_data
        is surpyval.utils.event_history_data.accepts_event_history_data
    )


def test_star_import_still_gets_the_old_name():
    namespace: dict = {}
    # and FutureWarnings for the data functions deprecated by date
    with pytest.warns((DeprecationWarning, FutureWarning)):
        exec("from surpyval import *", namespace)
    assert namespace["RecurrentEventData"] is EventHistoryData


def test_the_new_name_does_not_warn():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert surpyval.EventHistoryData is EventHistoryData
