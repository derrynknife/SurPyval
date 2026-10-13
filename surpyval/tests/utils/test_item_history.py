"""ItemHistory: one repairable item's history, and EventHistoryData
built from (and back to) a list of them."""

import numpy as np
import pytest

from surpyval import EventHistoryData, ItemHistory, handle_xicn
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
    data = EventHistoryData.from_items([PUMP, FAN])
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
    data = EventHistoryData.from_items(
        [ItemHistory("A", observed=[5, 9], right_censored=12)]
    )
    assert data.x.ndim == 1


def test_to_items_round_trips():
    data = EventHistoryData.from_items([PUMP, FAN])
    items = data.to_items()
    assert items == [FAN, PUMP]
    again = EventHistoryData.from_items(items)
    for name in ("x", "i", "c", "n", "tl", "tr", "Z"):
        np.testing.assert_array_equal(
            getattr(again, name), getattr(data, name)
        )


def test_from_histories_is_from_items():
    a = EventHistoryData.from_histories(
        observed={"A": [5, 9]},
        right_censored={"A": 12, "B": 4},
        left_truncation={"B": 1},
    )
    b = EventHistoryData.from_items(
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
    with pytest.raises(ValueError, match="more than once: 'A'"):
        EventHistoryData.from_items(
            [ItemHistory("A", observed=[1]), ItemHistory("A", observed=[2])]
        )
    with pytest.raises(ValueError, match="missing \\['B'\\]"):
        EventHistoryData.from_items(
            [
                ItemHistory("A", observed=[1], covariates=1),
                ItemHistory("B", observed=[2]),
            ]
        )
    with pytest.raises(ValueError, match="one kind"):
        EventHistoryData.from_items(
            [ItemHistory("A", observed=[1]), ItemHistory(2, observed=[2])]
        )
    with pytest.raises(TypeError, match="ItemHistory objects"):
        EventHistoryData.from_items([{"observed": [1]}])


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
    data = EventHistoryData.from_items(items)
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
    data = EventHistoryData.from_items(items)
    schedule = data.time_varying_covariates["A"]
    assert schedule.edges.tolist() == [0, 6, np.inf]
    assert schedule.Z.tolist() == [[1.0], [2.0]]
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
    data = EventHistoryData.from_items(items)
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
    data = EventHistoryData.from_items(items)
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
        EventHistoryData.from_items(
            [
                ItemHistory("A", observed=[1], observed_event_types="a"),
                ItemHistory("B", observed=[2]),
            ]
        )


def test_time_varying_covariates_take_schedules_and_paths():
    from surpyval import CovariatePath, StepSchedule

    ramp = CovariatePath.from_points([0, 50], [0.0, 1.0])
    steps = StepSchedule.from_changepoints([0, 20], [[0.5], [1.5]])
    items = [
        ItemHistory(
            "A",
            observed=[5, 9],
            right_censored=12,
            time_varying_covariates=ramp,
        ),
        ItemHistory(
            "B", observed=[3], right_censored=10, time_varying_covariates=steps
        ),
        ItemHistory(
            "C",
            observed=[4],
            right_censored=11,
            time_varying_covariates={0: [1.0], 5: [2.0]},
        ),
    ]
    data = EventHistoryData.from_items(items)
    assert data.time_varying_covariates["A"] is ramp
    held = data.time_varying_covariates["B"]
    assert held.edges.tolist() == steps.edges.tolist()
    assert held.Z.tolist() == steps.Z.tolist()
    assert isinstance(data.time_varying_covariates["C"], StepSchedule)
    assert data.to_items() == items
    with pytest.raises(ValueError, match="#905"):
        rc.CrowAMSAA.fit(data)


def test_time_varying_covariates_need_the_same_number_of_values():
    from surpyval import StepSchedule

    with pytest.raises(ValueError, match="same number of values"):
        EventHistoryData.from_items(
            [
                ItemHistory(
                    "A", observed=[5], time_varying_covariates={0: [1]}
                ),
                ItemHistory(
                    "B",
                    observed=[3],
                    time_varying_covariates=StepSchedule.from_changepoints(
                        [0], [[1.0, 2.0]]
                    ),
                ),
            ]
        )


# -- states ----------------------------------------------------------------


def _path(**kwargs):
    base = {"initial_state": "healthy", "right_censored": 2000}
    base.update(kwargs)
    return ItemHistory("U1", **base)


def test_states_are_changes_with_censoring_and_round_trip():
    items = [
        ItemHistory(
            "U1",
            initial_state="healthy",
            left_censored_states={300: "worn"},
            observed_states={900: "degraded"},
            interval_censored_states=[[1000, 1500, "failed"]],
            right_censored=2000,
        ),
        ItemHistory("U2", initial_state="healthy", right_censored=2500),
    ]
    data = EventHistoryData.from_items(items)
    initial, changes = data.states["U1"]
    assert initial == "healthy"
    assert [(k, st) for k, _, _, st in changes] == [
        ("left_censored", "worn"),
        ("observed", "degraded"),
        ("interval_censored", "failed"),
    ]
    assert data.to_items() == items
    for fit in (rc.CrowAMSAA.fit, rc.CrowAMSAA.fit_from_recurrent_data):
        with pytest.raises(ValueError, match="#887"):
            fit(data)
    with pytest.raises(ValueError, match="states"):
        data.to_histories()


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {"initial_state": None, "observed_states": {9: "b"}},
            "without 'initial_state'",
        ),
        ({"observed_states": {900: "healthy"}}, "from 'healthy' itself"),
        (
            {"observed_states": {900: "a", 1200: "a"}},
            "from 'a' itself",
        ),
        (
            {
                "observed_states": {1200: "degraded"},
                "interval_censored_states": [[1000, 1500, "failed"]],
            },
            "overlap or are out of order",
        ),
        (
            {
                "observed_states": {500: "a"},
                "left_censored_states": {800: "b"},
            },
            "before every other change",
        ),
        ({"observed_states": {2500: "a"}}, "before its last change"),
        (
            {"observed_states": {100: "a"}, "left_truncation": 100},
            "not after it came under observation",
        ),
        ({"interval_censored_states": [[5, 5, "a"]]}, "lower < upper"),
        (
            {"interval_censored_states": [[5, "a"]]},
            "\\[lower, upper, state\\]",
        ),
        ({"observed_states": {900: None}}, "cannot hold None"),
        ({"left_censored_states": {300: "a", 400: "b"}}, "at most one"),
    ],
)
def test_states_are_checked(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _path(**kwargs)


def test_states_for_every_item_once_given():
    with pytest.raises(ValueError, match="'initial_state'"):
        EventHistoryData.from_items(
            [
                _path(observed_states={900: "degraded"}),
                ItemHistory("U2", right_censored=10),
            ]
        )


def test_a_path_needs_its_end_of_observation():
    with pytest.raises(ValueError, match="needs the age up to which"):
        ItemHistory("U1", initial_state="a", observed_states={5: "b"})
