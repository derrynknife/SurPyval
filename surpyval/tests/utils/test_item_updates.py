"""Updating ItemLifetime and ItemHistory records: each update returns a
new, checked record and leaves the old one as it was."""

import pytest

from surpyval import (
    CovariatePath,
    EventHistoryData,
    ItemHistory,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
)

# -- ItemLifetime ------------------------------------------------------------


def test_clone_takes_a_new_id_and_changes():
    b7 = ItemLifetime("B7", observed=1820, covariates=[1.0], left_truncation=5)
    b8 = b7.clone("B8")
    assert b8 == ItemLifetime(
        "B8", observed=1820, covariates=[1.0], left_truncation=5
    )
    b9 = b7.clone("B9", right_censored=2000)
    assert b9.outcome == "right_censored" and b9.value == 2000
    assert b7.id == "B7" and b7.outcome == "observed"


def test_a_running_lifetime_extends_fails_or_is_found_failed():
    b = ItemLifetime("B", right_censored=100, covariates=[0.5])
    assert b.extend(150) == ItemLifetime(
        "B", right_censored=150, covariates=[0.5]
    )
    assert b.fail(120, "spall") == ItemLifetime(
        "B", observed=120, event_type="spall", covariates=[0.5]
    )
    assert b.found_failed(200) == ItemLifetime(
        "B", interval_censored=[100, 200], covariates=[0.5]
    )
    assert b == ItemLifetime("B", right_censored=100, covariates=[0.5])


@pytest.mark.parametrize(
    "call, match",
    [
        (lambda b: b.extend(50), "before it was last seen working"),
        (lambda b: b.fail(50), "cannot have failed at 50"),
        (lambda b: b.found_failed(50), "last seen working at 100"),
        (lambda b: b.fail(150).extend(200), "not still running"),
    ],
)
def test_lifetime_updates_are_checked(call, match):
    with pytest.raises(ValueError, match=match):
        call(ItemLifetime("B", right_censored=100))


def test_update_covariates_adds_a_step():
    b = ItemLifetime(
        "B", right_censored=100, time_varying_covariates={0: [1.0], 50: [2.0]}
    )
    updated = b.update_covariates(80, [3.0])
    assert updated.time_varying_covariates.edges.tolist()[:-1] == [0, 50, 80]
    replaced = b.update_covariates(40, [5.0])
    assert replaced.time_varying_covariates.edges.tolist()[:-1] == [0, 40]
    with pytest.raises(ValueError, match="CovariatePath"):
        b.clone(
            time_varying_covariates=CovariatePath.from_points([0, 9], [0, 1])
        ).update_covariates(5, [1.0])


# -- ItemHistory -------------------------------------------------------------


PUMP = ItemHistory("P1", observed=[420], right_censored=600)


def test_history_clone_and_extend():
    assert PUMP.clone("P2").id == "P2"
    assert PUMP.extend(900).right_censored == 900
    with pytest.raises(ValueError, match="before its end"):
        PUMP.extend(500)
    assert PUMP.right_censored == 600


def test_add_failure_before_and_after_the_end():
    inside = PUMP.add_failure(500)
    assert inside.observed.tolist() == [420, 500]
    assert inside.right_censored == 600
    after = PUMP.add_failure(700)
    assert after.right_censored == 700  # it was watched until it failed


def test_add_failure_keeps_its_type():
    typed = ItemHistory(
        "P1", observed=[420], observed_event_types="seal", right_censored=600
    )
    more = typed.add_failure(650, "bearing")
    assert more.observed_event_types.tolist() == ["seal", "bearing"]
    assert more.right_censored == 650
    # Without one, its type was not recorded: a masked cause (#928)
    assert typed.add_failure(650).observed_event_types.tolist() == [
        "seal",
        None,
    ]


def test_only_failures_are_added_by_method():
    # Counts between inspections and changes of state are given to the
    # constructor (or clone); there are no methods for them (#928)
    for name in (
        "add_failures_between",
        "change_state",
        "change_state_between",
    ):
        assert not hasattr(ItemHistory, name)


def test_updated_items_build_a_dataset():
    items = [PUMP.add_failure(500).extend(900), PUMP.clone("P2")]
    data = EventHistoryData.from_items(items)
    assert data.to_items() == items
    lifetimes = [
        ItemLifetime("A", right_censored=10).fail(12),
        ItemLifetime("A", right_censored=10).clone("B"),
    ]
    assert SurpyvalData.from_items(lifetimes).to_items() == lifetimes


def test_history_update_covariates():
    steps = StepSchedule.from_changepoints([0, 300], [[1.0], [2.0]])
    pump = PUMP.clone(time_varying_covariates=steps)
    updated = pump.update_covariates(500, [3.0])
    assert updated.time_varying_covariates.Z.ravel().tolist() == [1, 2, 3]
