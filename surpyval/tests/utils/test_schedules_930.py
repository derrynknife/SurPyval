"""#930: a covariate schedule starts at or before where time is measured
from, records keep the schedule they are given, and schedules compare by
contents."""

import copy
import pickle

import numpy as np
import pytest

from surpyval import (
    CovariatePath,
    ItemHistory,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
    WeibullPH,
)

LATE = StepSchedule.from_changepoints([5, 8], [[0.0], [1.0]])
GOOD = StepSchedule.from_changepoints([0, 8], [[0.0], [1.0]])


@pytest.mark.parametrize(
    "tvc",
    [
        LATE,
        {5: [0.0], 8: [1.0]},
        CovariatePath.from_points([5, 8], [0.0, 1.0]),
    ],
    ids=["schedule", "dict", "path"],
)
@pytest.mark.parametrize("record", [ItemLifetime, ItemHistory])
def test_a_record_refuses_covariates_starting_after_its_start(record, tvc):
    outcome = (
        {"observed": 10} if record is ItemLifetime else {"observed": [10]}
    )
    with pytest.raises(ValueError, match="start at age 5, after the start"):
        record("a", time_varying_covariates=tvc, **outcome)
    # From its entry on is enough.
    record("a", left_truncation=6, time_varying_covariates=tvc, **outcome)


def test_the_issue_example_clones_updates_and_reshapes():
    a = ItemLifetime("a", observed=10, time_varying_covariates=GOOD)
    assert a.clone("b").time_varying_covariates == GOOD
    updated = a.update_covariates(9, [2.0])
    assert updated.time_varying_covariates == StepSchedule.from_changepoints(
        [0, 8, 9], [[0.0], [1.0], [2.0]]
    )
    data = SurpyvalData.from_items(
        [a, ItemLifetime("c", right_censored=12, time_varying_covariates=GOOD)]
    )
    censored = data.censor_at(9)
    assert censored.item("c").time_varying_covariates == GOOD
    assert data.without("covariates").ids == ["a", "c"]
    h = ItemHistory("h", observed=[3, 7], time_varying_covariates=GOOD)
    assert h.clone() == h
    assert h.add_failure(9).time_varying_covariates == GOOD


def test_records_show_the_schedule_as_given_and_paste_back():
    a = ItemLifetime("a", observed=10, time_varying_covariates=GOOD)
    assert "StepSchedule.from_changepoints([0.0, 8.0]" in repr(a)
    assert eval(repr(a)) == a
    d = ItemLifetime("a", observed=10, time_varying_covariates={0: [0.0]})
    assert repr(d).endswith("time_varying_covariates={0.0: [0.0]})")
    assert d.update_covariates(4, [1.0]).time_varying_covariates == (
        StepSchedule.from_changepoints([0, 4], [[0.0], [1.0]])
    )
    assert "{0.0: [0.0], 4.0: [1.0]}" in repr(d.update_covariates(4, [1.0]))


def test_a_dict_and_its_schedule_give_equal_records():
    a = ItemLifetime("a", observed=10, time_varying_covariates=GOOD)
    d = ItemLifetime(
        "a", observed=10, time_varying_covariates={0: [0.0], 8: [1.0]}
    )
    assert a == d and hash(a) == hash(d)


def test_cyclic_schedules_compare_by_contents():
    c1 = StepSchedule.cyclic([0, 5], [[1.0], [0.0]], period=7)
    c2 = StepSchedule.cyclic([0, 5], [[1.0], [0.0]], period=7)
    assert c1 == c2 and c1 != StepSchedule.cyclic([0, 5], [1.0, 0.0], 8)
    with pytest.raises(TypeError):
        hash(c1)
    p = ItemLifetime("p", observed=20, time_varying_covariates=c1)
    q = ItemLifetime("p", observed=20, time_varying_covariates=c2)
    assert p == p.clone() == q and hash(p) == hash(q) == hash(p.clone())
    assert len({p, q}) == 1
    assert pickle.loads(pickle.dumps(p)) == p and copy.deepcopy(p) == p


@pytest.mark.parametrize(
    "schedule",
    [
        StepSchedule.from_changepoints([5, 8], [[0.0], [1.0]]),
        StepSchedule.cyclic([0, 5], [[1.0], [0.0]], period=7),
        StepSchedule.constant([2.0, 3.0]),
        StepSchedule.from_intervals([0, 1], [1, 2], [1.0, 2.0]),
        StepSchedule([1.0, 3.0], [[1.0]], period=4.0),
    ],
)
def test_a_schedule_repr_builds_it(schedule):
    assert eval(repr(schedule)) == schedule


def test_a_long_schedule_shows_in_short():
    long = StepSchedule.from_changepoints(np.arange(30), np.arange(30.0))
    assert repr(long) == "StepSchedule(step, 30 segments, p=1)"


def _model():
    rng = np.random.default_rng(0)
    Z = rng.binomial(1, 0.5, (100, 1)).astype(float)
    return WeibullPH.fit(100 * rng.weibull(2, 100), Z)


@pytest.mark.parametrize(
    "Z, xl",
    [
        (LATE, None),
        ([[0.0], [1.0]], [5, 8]),
        (CovariatePath.from_points([5, 8], [0.0, 1.0]), None),
        (
            ItemLifetime(
                "a",
                observed=10,
                left_truncation=6,
                time_varying_covariates=LATE,
            ),
            None,
        ),
    ],
    ids=["schedule", "arrays", "path", "item"],
)
def test_a_prediction_refuses_a_path_starting_after_0(Z, xl):
    model = _model()
    with pytest.raises(ValueError, match="start at 5, after the time origin"):
        model.sf_tvc([10.0], Z, xl=xl)


def test_a_path_has_no_value_before_its_first_point():
    path = CovariatePath.from_points([5, 8], [0.0, 1.0])
    assert path.start == 5.0
    with pytest.raises(ValueError, match="starts at 5"):
        path([1.0])
    np.testing.assert_allclose(path([5.0, 6.5]).ravel(), [0.0, 0.5])
    assert CovariatePath.from_callable(lambda t: t).start == -np.inf
    with pytest.raises(ValueError, match="start at 0"):
        CovariatePath.from_points([2, 24], [0.0, 1.0], period=24)


def test_a_fit_from_records_refuses_a_late_schedule():
    # Records refuse it; a dataset built around them is checked again.
    a = ItemLifetime("a", observed=10, time_varying_covariates=GOOD)
    b = ItemLifetime("b", observed=12, time_varying_covariates=GOOD)
    data = SurpyvalData.from_items([a, b])
    data.time_varying_covariates = {"a": LATE, "b": GOOD}
    from surpyval.utils.surpyval_data import _fit_tvc_from_data

    with pytest.raises(ValueError, match="start at age 5"):
        _fit_tvc_from_data(WeibullPH.fit_tvc, data, {})
