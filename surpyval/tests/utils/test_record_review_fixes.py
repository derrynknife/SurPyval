"""Fixes from the second review of the data classes: linear from_items,
event types never dropped on the time-varying route, slices and JSON
keeping what the data holds, frozen records, item ids from numpy, the
outcome by name on ItemLifetime, and clearer update and option errors."""

import pickle
import time

import numpy as np
import pytest

import surpyval as sp
from surpyval import (
    EventHistoryData,
    ItemHistory,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
)
from surpyval import recurrent as rc

STEPS = StepSchedule.from_changepoints([0, 3], [[0.0], [1.0]])
X = sp.Weibull.random(40, 10, 2, random_state=1)


# 1. linear in the number of items


def test_from_items_is_linear():
    small = [ItemLifetime(k, observed=1.0 + k % 97) for k in range(5_000)]
    big = [ItemLifetime(k, observed=1.0 + k % 97) for k in range(50_000)]
    t = time.perf_counter()
    SurpyvalData.from_items(small)
    t_small = time.perf_counter() - t
    t = time.perf_counter()
    SurpyvalData.from_items(big)
    t_big = time.perf_counter() - t
    # ten times the items: linear is ~10x, quadratic ~100x
    assert t_big < 40 * max(t_small, 1e-3), (t_small, t_big)


def test_repeated_ids_are_still_found():
    with pytest.raises(ValueError, match="more than once: 'A'"):
        SurpyvalData.from_items(
            [ItemLifetime("A", observed=1), ItemLifetime("A", observed=2)]
        )
    with pytest.raises(ValueError, match="more than once: 1"):
        EventHistoryData.from_items(
            [ItemHistory(1, observed=[1]), ItemHistory(1.0, observed=[2])]
        )


# 2. event types are never dropped on the way to fit_tvc


def test_event_types_are_refused_on_the_time_varying_route():
    items = [
        ItemLifetime(
            k, observed=x, event_type="a", time_varying_covariates=STEPS
        )
        for k, x in enumerate(X)
    ]
    with pytest.raises(ValueError, match="no event types"):
        sp.WeibullPH.fit(SurpyvalData.from_items(items))


# 3. slices keep ids, schedules, covariates and states


def test_surpyval_data_slices_keep_ids_and_schedules():
    data = SurpyvalData.from_items(
        [
            ItemLifetime(k, observed=x, time_varying_covariates=STEPS)
            for k, x in enumerate(X)
        ]
    )
    part = data[0:30]
    assert part.ids == list(range(30))
    assert set(part.time_varying_covariates) == set(range(30))
    with pytest.raises(ValueError, match="time-varying covariates"):
        sp.Weibull.fit(part)
    np.testing.assert_allclose(
        sp.WeibullPH.fit(part).params,
        sp.WeibullPH.fit(SurpyvalData.from_items(data.to_items()[:30])).params,
    )


def test_event_history_slices_keep_covariates_schedules_and_states():
    data = EventHistoryData.from_items(
        [
            ItemHistory(
                "A",
                observed=[5, 9],
                right_censored=12,
                covariates=[1.0],
                time_varying_covariates=STEPS,
                initial_state="ok",
                observed_states={7: "worn"},
            ),
            ItemHistory(
                "B",
                observed=[3],
                right_censored=10,
                covariates=[0.0],
                time_varying_covariates={0: [1.0]},
                initial_state="ok",
            ),
        ]
    )
    part = data[0:3]
    assert part.items == ["A"]
    assert part.Z.tolist() == [[1.0]] * 3
    assert set(part.time_varying_covariates) == {"A"}
    assert set(part.states) == {"A"}
    with pytest.raises(ValueError, match="#905"):
        rc.CrowAMSAA.fit(part)


# 4. JSON keeps event types and ids, and refuses schedules


def test_json_keeps_event_types_and_ids():
    data = SurpyvalData.from_items(
        [
            ItemLifetime("A", observed=5, event_type="x"),
            ItemLifetime("B", right_censored=9),
        ]
    )
    again = SurpyvalData.from_json(data.to_json())
    assert list(again.e) == ["x", None]
    assert again.ids == ["A", "B"]
    plain = SurpyvalData([1, 2, 3], c=[0, 0, 1], e=["a", "b", None])
    assert list(SurpyvalData.from_json(plain.to_json()).e) == ["a", "b", None]


def test_json_refuses_time_varying_covariates():
    data = SurpyvalData.from_items(
        [ItemLifetime("A", observed=5, time_varying_covariates=STEPS)]
    )
    with pytest.raises(ValueError, match="no JSON form"):
        data.to_json()
    assert pickle.loads(pickle.dumps(data)).time_varying_covariates


# 5. records are fixed once made, and hashable


@pytest.mark.parametrize(
    "record",
    [
        ItemLifetime("B", observed=10, covariates=[1.0]),
        ItemHistory("P", observed=[5, 9], right_censored=12),
    ],
)
def test_records_cannot_be_edited_in_place(record):
    with pytest.raises(AttributeError, match="fixed once made"):
        record.item = "C"
    arrays = [v for v in vars(record).values() if isinstance(v, np.ndarray)]
    assert arrays
    for arr in arrays:
        with pytest.raises(ValueError, match="read-only"):
            arr[...] = 0
    assert record.clone("C").item == "C"
    assert pickle.loads(pickle.dumps(record)) == record


def test_records_are_hashable_by_value():
    a = ItemLifetime("B", observed=10, covariates=[1.0])
    b = ItemLifetime("B", observed=10.0, covariates=np.array([1.0]))
    assert a == b and hash(a) == hash(b)
    assert len({a, b, a.clone("C")}) == 2
    h = ItemHistory("P", observed=[5, 9], right_censored=12)
    assert h in {h.clone()}


# 6. ids from numpy are plain ids


def test_numpy_ids_mix_with_plain_ones():
    data = SurpyvalData.from_items(
        [ItemLifetime(np.int64(1), observed=3), ItemLifetime(2, observed=4)]
    )
    assert data.ids == [1, 2] and type(data.ids[0]) is int
    data = SurpyvalData.from_items(
        [ItemLifetime(np.str_("a"), observed=3), ItemLifetime("b", observed=4)]
    )
    assert data.ids == ["a", "b"]
    with pytest.raises(ValueError, match="one kind"):
        SurpyvalData.from_items(
            [ItemLifetime("1", observed=3), ItemLifetime(2, observed=4)]
        )


# 7. the outcome by name


def test_lifetime_reads_its_outcome_by_name():
    item = ItemLifetime("B", interval_censored=[2, 4])
    assert item.interval_censored.tolist() == [2, 4]
    assert item.observed is None and item.right_censored is None
    assert ItemLifetime("A", observed=3).observed == 3
    assert ItemLifetime("A", left_censored=3).left_censored == 3


# 8. found_failed speaks of inspections


def test_found_failed_at_the_last_seen_age_points_to_fail():
    with pytest.raises(ValueError, match=r"use fail\(5\)"):
        ItemLifetime("B", right_censored=5).found_failed(5)


# 9. options the time-varying fit cannot take


def test_options_fit_tvc_cannot_take_are_named():
    data = SurpyvalData.from_items(
        [
            ItemLifetime(k, observed=x, time_varying_covariates=STEPS)
            for k, x in enumerate(X)
        ]
    )
    with pytest.raises(TypeError, match="takes no 'init'; it takes 'fixed'"):
        sp.WeibullAFT.fit(data, init=[10, 2, 0])
