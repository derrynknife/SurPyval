"""SurpyvalData.to_observations carries everything from_observations can
hold (weights, unit, attributes, readings, ids in their own rows) and
refuses, saying what to use, what it cannot (terminal flags, states,
time-varying covariates)."""

import numpy as np
import pytest

from surpyval import ItemLifetime, StepSchedule, SurpyvalData


def _round_trip(data):
    return SurpyvalData.from_observations(**data.to_observations())


def _same_rows(a, b):
    for x, y in zip(a.to_xcnt(), b.to_xcnt()):
        np.testing.assert_array_equal(x, y)


def test_ids_stay_with_their_own_rows():
    # from_observations used to sort the rows after the ids were listed,
    # pairing each id with another item's time.
    data = SurpyvalData.from_observations(
        observed=[5, 1],
        observed_ids=["a", "b"],
        right_censored=[3],
        right_censored_ids=["c"],
    )
    times = dict(zip(data.ids, data.x.tolist()))
    assert times == {"a": 5, "b": 1, "c": 3}
    back = _round_trip(data)
    assert dict(zip(back.ids, back.x.tolist())) == times


def test_weights_are_carried():
    data = SurpyvalData([1, 3, 5, 7], c=[0, 0, 1, 0], w=[0.5, 2, 1.5, 1])
    lists = data.to_observations()
    assert lists["observed_weights"].tolist() == [0.5, 2, 1]
    back = _round_trip(data)
    _same_rows(back, data)
    np.testing.assert_array_equal(back.w, data.w)


def test_weights_given_for_one_kind_are_needed_for_all():
    with pytest.raises(ValueError, match="'right_censored_weights' too"):
        SurpyvalData.from_observations(
            observed=[1, 2], observed_weights=[1, 2], right_censored=[3]
        )


def test_the_unit_is_carried():
    data = SurpyvalData([1, 3, 5], c=[0, 0, 1], unit="hours")
    assert data.to_observations()["unit"] == "hours"
    assert _round_trip(data).unit == "hours"


def test_attributes_are_carried():
    data = SurpyvalData.from_items(
        [
            ItemLifetime("P1", observed=120, attributes={"site": "north"}),
            ItemLifetime("P2", right_censored=300),
            ItemLifetime("P3", observed=80, attributes={"site": "south"}),
        ]
    )
    back = _round_trip(data)
    got = {i: a for i, a in zip(back.ids, back.attributes)}
    assert got == {
        "P1": {"site": "north"},
        "P2": None,
        "P3": {"site": "south"},
    }
    with pytest.raises(ValueError, match="dict of each item's attributes"):
        SurpyvalData.from_observations(observed=[1], observed_attributes=[3])


def test_readings_are_carried():
    readings = {"A": {1: 0.2, 4: 0.9}, "B": {2: 0.4}}
    data = SurpyvalData([5, 6], c=[0, 1], i=["A", "B"], m=readings)
    back = _round_trip(data)
    assert back.ids == ["A", "B"]
    assert back.markers == data.markers
    with pytest.raises(ValueError, match="'<kind>_ids'"):
        SurpyvalData.from_observations(observed=[5], m={"A": {1: 0.2}})


def test_what_from_observations_cannot_hold_is_refused_with_what_to_use():
    terminal = SurpyvalData(x=[2, 5, 9], i=["P1", "P1", "P1"])
    with pytest.raises(ValueError, match="terminal flags a.*to_dict"):
        terminal.to_observations()
    states = SurpyvalData(
        x=[3, 5], tl=[0, 3], s=["well", "ill"], e=["ill", None], i=["A", "A"]
    )
    with pytest.raises(ValueError, match="state s.*to_dict"):
        states.to_observations()
    steps = StepSchedule.from_changepoints([0, 2], [[0.0], [1.0]])
    tvc = SurpyvalData.from_items(
        [
            ItemLifetime("A", observed=5, time_varying_covariates=steps),
            ItemLifetime("B", right_censored=6, time_varying_covariates=steps),
        ]
    )
    with pytest.raises(ValueError, match="time-varying covariates.*to_dict"):
        tvc.to_observations()
