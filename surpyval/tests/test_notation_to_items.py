"""SurpyvalData with item ids to the notation's Items and back."""

import warnings

import numpy as np
import pytest

from surpyval import ItemLifetime, StepSchedule, SurpyvalData
from surpyval.univariate.regression.tvc_path import CovariatePath
from surpyval.utils.notation import IC, LC, RC, Item, Items, O, Truncation

inf = np.inf


def _round_trip(data):
    items = data.to_notation()
    assert isinstance(items, Items)
    assert items.surpyval_data() == data
    assert SurpyvalData.from_notation(items) == data
    return items


def test_histories_with_terminal_flags_and_truncation_round_trip():
    data = SurpyvalData(
        x=[120, 310, [400, 450], 1000, 200, 200, 200, 800],
        c=[0, 0, 2, 1, 0, 0, 0, 0],
        i=["P1"] * 4 + ["F1"] * 4,
        tl=[0, -inf, -inf, -inf, 50, -inf, -inf, -inf],
        tr=[inf] * 4 + [2000] * 4,
        a=[0, 0, 0, 0, 0, 0, 0, 1],
        unit="hours",
    )
    items = _round_trip(data)
    assert items[1] == Item(
        "F1", 3 * O(200), O(800, terminal=True)
    ) / Truncation(50, 2000)
    assert items.text() == (
        "P1: 120, 310, [400, 450], 1000+ | (0, inf]\n"
        "F1: 3*200, 800 | (50, 2000]"
    )


def test_counts_types_covariates_and_readings_round_trip():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = Items(
            [
                Item(
                    "A",
                    2 * LC(50, event_type="seal"),
                    O(120, event_type="bearing"),
                    2 * IC(200, 300, event_type="seal"),
                    RC(400),
                    covariates={"temp": 60, "load": 1},
                    readings={"vib": {0: 1.2, 100: 1.9}, "oil": {50: 3.0}},
                ),
                Item(
                    "B",
                    O(80),
                    covariates={"temp": 70, "load": 2},
                    readings={"vib": {0: 1.0}},
                ),
            ]
        ).surpyval_data()
        _round_trip(data)


def test_lifetimes_with_ids_markers_and_schedules_round_trip():
    path = CovariatePath.from_points([0, 200], [0.0, 2.0])
    data = SurpyvalData.from_items(
        [
            ItemLifetime(
                "A",
                observed=100,
                markers={0: 1.2, 50: 1.9},
                time_varying_covariates=StepSchedule.from_changepoints(
                    [0, 50], [0.0, 1.0]
                ),
            ),
            ItemLifetime(
                "B",
                right_censored=120,
                markers={0: 1.0},
                time_varying_covariates=path,
            ),
        ]
    )
    items = _round_trip(data)
    assert items[1].time_varying_covariates is path  # passed through


def test_what_a_history_cannot_say_is_refused():
    cases = [
        (
            SurpyvalData(x=[5, 9], c=[0, 1], n=[3, 1], i=["A", "B"]),
            "counted 3",
        ),
        (
            SurpyvalData(
                x=[5, 9],
                c=[1, 0],
                i=["A", "A"],
                Z=[[0.0], [1.0]],
                covariate_names=["x"],
            ),
            "change between its rows",
        ),
        (
            SurpyvalData(x=[5, 9], i=["A", "B"], Z=[[0.0], [1.0]]),
            "unnamed",
        ),
        (SurpyvalData(x=[5, 9], i=["A", "B"], w=[1, 2]), "weights"),
        (
            SurpyvalData(
                x=[3, 5],
                tl=[0, 4],
                s=["well", "ill"],
                e=["ill", "dead"],
                i=["A", "A"],
            ),
            "states",
        ),
        (
            # a gap in watching: the second row starts after the first ends
            SurpyvalData(x=[5, 9], c=[0, 0], tl=[0, 7], i=["A", "A"]),
            "same data back",
        ),
    ]
    for data, message in cases:
        with pytest.raises(ValueError, match=message):
            data.to_notation()
    data = SurpyvalData.from_items([ItemLifetime("A", observed=5)])
    data.attributes = [{"site": "N"}]
    with pytest.raises(ValueError, match="attributes.*to_dict"):
        data.to_notation()
