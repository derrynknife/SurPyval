"""Event histories as lifetime data (#926): each item's first failure,
and the times between failures, as SurpyvalData that keep each item's
delayed entry, id, covariates and event types."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import EventHistoryData, ItemHistory, ItemLifetime, StepSchedule

PUMPS = EventHistoryData.from_items(
    [
        ItemHistory(
            "A",
            [5, 9],
            right_censored=12,
            left_truncation=2,
            covariates={"size": 1.0},
        ),
        ItemHistory("B", [4], right_censored=10, covariates={"size": 0.0}),
        ItemHistory("C", right_censored=8, covariates={"size": 1.0}),
    ]
)


def test_first_events_keep_entry_ids_and_covariates():
    first = PUMPS.to_first_events()
    assert first.ids == ["A", "B", "C"]
    assert first.item("A") == ItemLifetime(
        "A", observed=5, left_truncation=2, covariates={"size": 1.0}
    )
    assert first.item("C") == ItemLifetime(
        "C", right_censored=8, covariates={"size": 1.0}
    )
    assert first.covariate_names == ["size"]


def test_first_events_of_every_kind_with_types():
    data = EventHistoryData.from_items(
        [
            ItemHistory(
                "P",
                [3, 7],
                observed_event_types=["seal", "bearing"],
                right_censored=9,
            ),
            ItemHistory(
                "Q",
                interval_censored=[[2, 5]],
                interval_censored_event_types=["seal"],
                right_censored=8,
            ),
            ItemHistory(
                "R",
                [6],
                observed_event_types=["seal"],
                left_censored=4,
                left_censored_counts=1,
                left_censored_event_types="bearing",
                right_censored=9,
            ),
        ]
    )
    first = data.to_first_events()
    assert first.item("P") == ItemLifetime("P", observed=3, event_type="seal")
    assert first.item("Q") == ItemLifetime(
        "Q", interval_censored=[2, 5], event_type="seal"
    )
    assert first.item("R") == ItemLifetime(
        "R", left_censored=4, event_type="bearing"
    )


def test_gaps_renew_at_each_failure_and_censor_the_last():
    gaps = PUMPS.to_gaps()
    assert gaps.ids == [
        ("A", 1),
        ("A", 2),
        ("A", 3),
        ("B", 1),
        ("B", 2),
        ("C", 1),
    ]
    assert gaps.x.tolist() == [5, 4, 3, 4, 6, 8]
    assert gaps.c.tolist() == [0, 0, 1, 0, 1, 1]
    # Only the first gap of an item that entered late is truncated
    assert gaps.item(("A", 1)).left_truncation == 2
    assert gaps.item(("A", 2)).left_truncation is None
    assert gaps.covariate_names == ["size"]


def test_a_gap_fit_is_the_fit_of_the_same_arrays():
    gaps = PUMPS.to_gaps().without("covariates")
    fitted = sp.Weibull.fit(gaps)
    by_hand = sp.Weibull.fit(
        [5, 4, 3, 4, 6, 8],
        c=[0, 0, 1, 0, 1, 1],
        tl=[2, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf],
    )
    assert np.allclose(fitted.params, by_hand.params)


def test_gaps_need_each_failure_age():
    counted = EventHistoryData.from_items(
        [ItemHistory("Q", interval_censored=[[2, 5]], right_censored=8)]
    )
    with pytest.raises(ValueError, match="counted between inspections"):
        counted.to_gaps()
    varying = EventHistoryData.from_items(
        [
            ItemHistory(
                "T",
                [3],
                right_censored=9,
                time_varying_covariates=StepSchedule.from_changepoints(
                    [0, 5], [[0.0], [1.0]]
                ),
            )
        ]
    )
    with pytest.raises(ValueError, match="time-varying covariates"):
        varying.to_gaps()
    # The first event keeps them: they follow the item's age
    assert varying.to_first_events().item("T").time_varying_covariates


def test_a_gap_of_zero_is_refused():
    twice = EventHistoryData.from_items(
        [ItemHistory("D", [3, 3, 6], right_censored=9)]
    )
    with pytest.raises(ValueError, match="more than one failure at age 3"):
        twice.to_gaps()


def test_the_old_name_warns_and_gives_the_new_result():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = PUMPS.get_times_to_first_events()
    assert any(
        issubclass(w.category, DeprecationWarning)
        and "to_first_events" in str(w.message)
        for w in caught
    )
    assert old.ids == ["A", "B", "C"]


def test_a_lifetime_fit_points_to_them():
    with pytest.raises(TypeError, match="to_first_events.*to_gaps"):
        sp.Weibull.fit(PUMPS)
