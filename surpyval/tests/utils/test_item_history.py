"""ItemHistory: one repairable item's history, and RecurrentEventData
built from (and back to) a list of them."""

import numpy as np
import pytest

from surpyval import ItemHistory, RecurrentEventData, handle_xicn
from surpyval import recurrent as rc

PUMP = ItemHistory(
    "P1",
    observed=[420, 1310],
    interval_censored=[[2100, 2300]],
    interval_censored_counts=2,
    left_censored=200,
    right_censored=2600,
    left_truncation=150,
    covariates=[1, 0.2],
)
FAN = ItemHistory(
    "F1",
    observed=[800, 1900],
    right_censored=3000,
    left_truncation=0,
    covariates=[0, 0.5],
)


def test_from_items_is_the_arrays():
    data = RecurrentEventData.from_items([PUMP, FAN])
    by_hand = handle_xicn(
        [
            [800, 800],
            [1900, 1900],
            [3000, 3000],
            [200, 200],
            [420, 420],
            [1310, 1310],
            [2100, 2300],
            [2600, 2600],
        ],
        i=["F1"] * 3 + ["P1"] * 5,
        c=[0, 0, 1, -1, 0, 0, 2, 1],
        n=[1, 1, 1, 1, 1, 1, 2, 1],
        tl=[0] * 3 + [150] * 5,
        Z=[[0, 0.5]] * 3 + [[1, 0.2]] * 5,
    )
    for name in ("x", "i", "c", "n", "tl", "tr", "Z"):
        np.testing.assert_array_equal(
            getattr(data, name), getattr(by_hand, name)
        )


def test_exact_failures_only_give_one_column():
    data = RecurrentEventData.from_items(
        [ItemHistory("A", observed=[5, 9], right_censored=12)]
    )
    assert data.x.ndim == 1


def test_to_items_round_trips():
    data = RecurrentEventData.from_items([PUMP, FAN])
    items = data.to_items()
    assert items == [FAN, PUMP]
    again = RecurrentEventData.from_items(items)
    for name in ("x", "i", "c", "n", "tl", "tr", "Z"):
        np.testing.assert_array_equal(
            getattr(again, name), getattr(data, name)
        )


def test_from_histories_is_from_items():
    a = RecurrentEventData.from_histories(
        observed={"A": [5, 9]},
        right_censored={"A": 12, "B": 4},
        left_truncation={"B": 1},
    )
    b = RecurrentEventData.from_items(
        [
            ItemHistory("A", observed=[5, 9], right_censored=12),
            ItemHistory("B", right_censored=4, left_truncation=1),
        ]
    )
    for name in ("x", "i", "c", "n", "tl", "tr"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_an_interval_counted_zero_is_left_out():
    item = ItemHistory(
        "A",
        interval_censored=[[0, 10], [10, 20]],
        interval_censored_counts=[0, 3],
        right_censored=20,
    )
    assert item.interval_censored.tolist() == [[10, 20]]
    assert item.interval_censored_counts.tolist() == [3]


def test_repr_shows_what_was_given():
    text = repr(ItemHistory("A", observed=[5], right_censored=8))
    assert (
        text == "ItemHistory(\n    'A',\n    observed=[5.0],\n"
        "    right_censored=8.0\n)"
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "no failures and no end"),
        ({"observed": [5, 9], "right_censored": 7}, "before its last failure"),
        (
            {"observed": [5], "left_truncation": 5},
            "at or before it came under",
        ),
        (
            {"observed": [5, 30], "right_truncation": 20},
            "after right_truncation",
        ),
        (
            {"observed": [5], "right_censored": 8, "right_truncation": 9},
            "same age",
        ),
        (
            {"observed": [5], "left_censored": 7},
            "at or before its other failures",
        ),
        (
            {"left_censored_counts": 2, "observed": [5]},
            "without 'left_censored'",
        ),
        ({"interval_censored": [5, 9]}, "\\[\\[lower, upper\\]\\]"),
        ({"interval_censored": [[9, 5]]}, "lower < upper"),
        ({"interval_censored": [[5, np.inf]]}, "end of observation"),
        (
            {
                "interval_censored": [[5, 9]],
                "interval_censored_counts": [1, 2],
            },
            "one per",
        ),
        ({"observed": [5, np.nan]}, "finite"),
        (
            {"observed": [5], "left_truncation": 0, "right_truncation": 0},
            "must be below",
        ),
        (
            {"observed": [5], "time_varying_covariates": {1: [0]}},
            "start at age 1",
        ),
        (
            {"observed": [5], "time_varying_covariates": {0: [0], 2: [0, 1]}},
            "same number",
        ),
    ],
)
def test_an_item_refuses(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ItemHistory("A", **kwargs)


def test_the_dataset_needs_consistent_items():
    with pytest.raises(ValueError, match="given twice: 'A'"):
        RecurrentEventData.from_items(
            [ItemHistory("A", observed=[1]), ItemHistory("A", observed=[2])]
        )
    with pytest.raises(ValueError, match="missing \\['B'\\]"):
        RecurrentEventData.from_items(
            [
                ItemHistory("A", observed=[1], covariates=1),
                ItemHistory("B", observed=[2]),
            ]
        )
    with pytest.raises(ValueError, match="one type"):
        RecurrentEventData.from_items(
            [ItemHistory("A", observed=[1]), ItemHistory(2, observed=[2])]
        )
    with pytest.raises(TypeError, match="ItemHistory objects"):
        RecurrentEventData.from_items([{"observed": [1]}])


def test_interval_and_left_censored_items_fit_as_their_arrays():
    items = [
        ItemHistory(
            "A",
            observed=[50, 160],
            interval_censored=[[200, 300]],
            interval_censored_counts=2,
            right_censored=400,
        ),
        ItemHistory(
            "B",
            left_censored=100,
            left_censored_counts=2,
            observed=[250],
            right_censored=420,
        ),
        ItemHistory("C", observed=[80, 210, 330], right_censored=380),
    ]
    data = RecurrentEventData.from_items(items)
    np.testing.assert_allclose(
        rc.CrowAMSAA.fit(data).params,
        rc.CrowAMSAA.fit(data.x, data.i, data.c, data.n).params,
    )


def test_time_varying_covariates_are_kept_and_refused_by_every_fit():
    items = [
        ItemHistory(
            "A",
            observed=[5, 9],
            right_censored=12,
            time_varying_covariates={0: [1.0], 6: [2.0]},
        ),
        ItemHistory(
            "B",
            observed=[3],
            right_censored=10,
            time_varying_covariates={0: [0.5]},
        ),
    ]
    data = RecurrentEventData.from_items(items)
    ages, values = data.time_varying_covariates["A"]
    assert ages.tolist() == [0, 6] and values.tolist() == [[1.0], [2.0]]
    assert data.to_items() == items
    with pytest.raises(ValueError, match="#905"):
        rc.CrowAMSAA.fit(data)
    with pytest.raises(ValueError, match="#905"):
        rc.CrowAMSAA.fit_from_recurrent_data(data)
    with pytest.raises(ValueError, match="#905"):
        rc.NonParametricCounting.fit_from_recurrent_data(data)


# -- event types -----------------------------------------------------------


def test_event_types_go_with_their_failures_and_round_trip():
    items = [
        ItemHistory(
            "A",
            observed=[9, 5],
            observed_event_types=["seal", "bearing"],
            interval_censored=[[10, 20]],
            interval_censored_counts=2,
            interval_censored_event_types="seal",
            right_censored=25,
        ),
        ItemHistory(
            "B", observed=[7], observed_event_types="seal", right_censored=30
        ),
        ItemHistory("C", right_censored=40),  # no failures: needs no type
    ]
    data = RecurrentEventData.from_items(items)
    a = data.i == "A"
    assert list(data.e[a]) == ["bearing", "seal", "seal", None]
    assert data.to_items() == items


def test_cause_specific_fits_take_items_with_event_types():
    items = [
        ItemHistory(
            "A",
            observed=[1, 3],
            observed_event_types=["a", "b"],
            right_censored=5,
        ),
        ItemHistory(
            "B", observed=[2, 6], observed_event_types="a", right_censored=7
        ),
    ]
    data = RecurrentEventData.from_items(items)
    by_hand = rc.CauseSpecificMCF.fit(
        [1, 3, 5, 2, 6, 7],
        [1, 1, 1, 2, 2, 2],
        [0, 0, 1, 0, 0, 1],
        e=["a", "b", None, "a", "a", None],
    )
    np.testing.assert_allclose(
        rc.CauseSpecificMCF.fit(data).mcf([4, 6], "a"),
        by_hand.mcf([4, 6], "a"),
    )
    with pytest.raises(ValueError, match="event types"):
        rc.CrowAMSAA.fit(data)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {"observed": [1, 2], "observed_event_types": ["a"]},
            "one per entry \\(2\\)",
        ),
        (
            {"observed": [1], "observed_event_types": [None]},
            "cannot hold None",
        ),
        (
            {
                "observed": [1],
                "observed_event_types": "a",
                "interval_censored": [[2, 3]],
            },
            "'interval_censored_event_types'",
        ),
        (
            {"observed": [1], "left_censored_event_types": "a"},
            "without 'left_censored'",
        ),
    ],
)
def test_event_types_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ItemHistory("A", **kwargs)


def test_event_types_for_every_item_with_failures():
    with pytest.raises(ValueError, match="\\['B'\\]"):
        RecurrentEventData.from_items(
            [
                ItemHistory("A", observed=[1], observed_event_types="a"),
                ItemHistory("B", observed=[2]),
            ]
        )
