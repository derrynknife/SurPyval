"""Usability fixes from using the data classes: look-up by id, one name
for ids, len, states read back, counts by id, messages about dates and
types, and summaries."""

import datetime

import numpy as np
import pandas as pd
import pytest

from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData

LIVES = [
    ItemLifetime("B7", observed=1820, event_type="spall"),
    ItemLifetime("B8", right_censored=2400),
]
HISTORIES = [
    ItemHistory("M1", observed=[420, 1310], right_censored=2600),
    ItemHistory("M2", observed=[980], right_censored=2900),
]


def test_look_up_an_item_by_id():
    data = SurpyvalData.from_items(LIVES)
    assert data.item("B8") == LIVES[1]
    assert EventHistoryData.from_items(HISTORIES).item("M2") == HISTORIES[1]
    with pytest.raises(KeyError, match="No item 'X'"):
        data.item("X")
    plain = SurpyvalData([3, 7], c=[0, 1])
    assert plain.item(1) == ItemLifetime(1, right_censored=7)
    with pytest.raises(KeyError, match="row numbers, 0 to 1"):
        plain.item(5)


def test_one_name_for_ids():
    assert LIVES[0].id == "B7"
    assert ItemLifetime(id="X", observed=1).id == "X"
    assert LIVES[0].clone(id="B9").id == "B9"
    data = EventHistoryData.from_items(HISTORIES)
    assert data.ids == ["M1", "M2"]
    assert SurpyvalData.from_items(LIVES).ids == ["B7", "B8"]
    with pytest.warns(DeprecationWarning, match="EventHistoryData.items"):
        assert data.items == ["M1", "M2"]
    assert len(data) == 5 and len(data.ids) == 2


def test_states_are_read_back_by_their_names():
    u = ItemHistory(
        "U1",
        initial_state="healthy",
        left_censored_states={300: "worn"},
        observed_states={900: "degraded"},
        interval_censored_states=[[1000, 1500, "failed"]],
        right_censored=2000,
        left_truncation=100,
    )
    assert u.observed_states == {900: "degraded"}
    assert u.interval_censored_states == [[1000, 1500, "failed"]]
    assert u.left_censored_states == {300: "worn"}
    assert [u.state_at(a) for a in (50, 150, 300, 950, 1200, 1700, 2500)] == [
        None,  # before observation
        None,  # before the first look, across a left censored change
        "worn",
        "degraded",
        None,  # between two inspections, across a change
        "failed",
        None,  # after the end of observation
    ]
    with pytest.raises(ValueError, match="has no states"):
        HISTORIES[0].state_at(5)


def test_counts_by_id():
    data = SurpyvalData.from_items(LIVES, counts={"B8": 40})
    assert data.n.tolist() == [1, 40]
    with pytest.raises(ValueError, match="names items not given: 'X'"):
        SurpyvalData.from_items(LIVES, counts={"X": 2})


@pytest.mark.parametrize(
    "age, match",
    [
        (pd.Timedelta(days=5), "is a duration"),
        (datetime.timedelta(hours=5), "is a duration"),
        (np.timedelta64(5, "D"), "is a duration"),
        (pd.Timestamp("2024-01-01"), "is a date"),
        (datetime.date(2024, 1, 1), "is a date"),
    ],
)
def test_dates_and_durations_say_how_to_convert(age, match):
    with pytest.raises(ValueError, match=match):
        ItemLifetime("A", observed=age)
    with pytest.raises(ValueError, match=match):
        ItemHistory("A", observed=[age])


def test_a_failure_added_without_its_type_is_a_masked_cause():
    typed = ItemHistory("P", observed=[5], observed_event_types="seal")
    assert typed.add_failure(9).observed_event_types.tolist() == ["seal", None]


def test_summaries():
    text = SurpyvalData.from_items(
        [
            ItemLifetime("A", observed=5, event_type="x", covariates=[1.0]),
            ItemLifetime("B", right_censored=9, covariates=[0.0]),
        ],
        counts=[2, 3],
    ).summary()
    assert text.splitlines() == [
        "SurpyvalData: 2 rows, 5 items (with item ids)",
        "  outcomes: 2 observed, 3 right censored",
        "  event types: x (1)",
        "  covariates: 1 column",
    ]
    history = EventHistoryData.from_items(
        [
            ItemHistory(
                "U",
                observed=[5],
                right_censored=9,
                left_truncation=1,
                initial_state="ok",
                observed_states={7: "worn"},
            )
        ]
    ).summary()
    assert "truncation: 2 left" in history
    assert "states: ok, worn" in history
