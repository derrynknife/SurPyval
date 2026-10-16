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
        "SurpyvalData: 5 items, 2 rows (with item ids)",
        "  outcomes: 2 observed, 3 right censored",
        "  event types: x (2)",
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
    assert "truncation: 1 left" in history
    assert "states: ok, worn" in history


# -- #934: display, counts and the small asymmetries --------------------


def test_issue_934_repr_is_the_size_and_the_first_rows():
    data = SurpyvalData.from_items(
        [
            ItemLifetime("a", observed=5, event_type="x", covariates={"z": 1}),
            ItemLifetime(
                "b", right_censored=9, covariates={"z": 2}, left_truncation=1
            ),
        ],
        counts=[40, 3],
    )
    lines = repr(data).splitlines()
    assert lines[0] == "SurpyvalData: 43 items, 2 rows"
    assert lines[1].split() == ["id", "x", "c", "n", "tl", "e", "z"]
    assert lines[2].split() == ["'a'", "5.0", "0", "40", "-inf", "'x'", "1.0"]
    assert str(data) == repr(data)
    html = data._repr_html_()
    assert html.startswith("<div>") and "<th>id</th>" in html
    many = SurpyvalData(list(range(1, 30)))
    assert repr(many).splitlines()[-1].startswith("  ... 19 more rows")
    history = EventHistoryData.from_items(HISTORIES)
    assert history._repr_html_().count("<tr>") == 1 + len(history)


def test_issue_934_summary_counts_items_not_rows():
    data = SurpyvalData.from_items(
        [
            ItemLifetime("a", observed=5, event_type="x", left_truncation=1),
            ItemLifetime("b", right_censored=9, right_truncation=20),
            ItemLifetime("c", observed=7),
        ],
        counts=[40, 3, 2],
    )
    text = data.summary()
    assert "SurpyvalData: 45 items, 3 rows" in text
    assert "truncation: 40 left, 3 right" in text
    assert "event types: x (40), not recorded (2)" in text
    history = EventHistoryData.from_items(
        [
            ItemHistory(
                "A",
                observed=[5, 9, 10],
                interval_censored=[[12, 14]],
                interval_censored_counts=3,
                right_censored=15,
                left_truncation=1,
                observed_event_types="seal",
                interval_censored_event_types="bearing",
            ),
            ItemHistory("B", right_censored=4),
        ]
    )
    text = history.summary()
    assert text.startswith("EventHistoryData: 2 items, 6 rows")
    assert "truncation: 1 left" in text
    assert "event types: bearing (3), seal (3)" in text


def test_issue_934_tab_completion_shows_the_api_not_the_internals():
    data = SurpyvalData([1, 2, 3], c=[0, 0, 1], Z=[[1], [2], [3]])
    names = dir(data)
    assert "x_o" not in names and "mask_i" not in names and "Z_o" not in names
    assert "to_items" in names and "x" in names and "Z" in names
    assert data.x_o.tolist() == [1.0, 2.0]  # still there
    history = EventHistoryData.from_items(HISTORIES)
    names = dir(history)
    assert "split_for_nhpp_likelihood" not in names and "items" not in names
    assert "to_first_events" in names and "ids" in names


def test_issue_934_small_asymmetries():
    data = SurpyvalData.from_items(LIVES, counts=[2, 5])
    assert data.event_types == ["spall"]
    assert SurpyvalData([1, 2]).event_types == []
    assert LIVES[0].has_event_types and not LIVES[1].has_event_types
    assert data.n_items == 7 and len(data) == 2
    assert [item.id for item in data.records()] == ["B7", "B8"]
    history = EventHistoryData.from_items(HISTORIES)
    assert history.n_items == 2 and len(history) == 5
    assert [item.id for item in history.records()] == ["M1", "M2"]
