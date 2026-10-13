"""A window cut through what is known only as a whole, records out in one
pass, counts in words, and covariates picked by column number."""

import time

import numpy as np
import pytest

from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData
from surpyval.utils.data_reshape import censor_at, enter_at

COUNTED = ItemHistory(
    "A",
    [100],
    interval_censored=[[400, 800]],
    interval_censored_counts=[2],
    right_censored=1000,
)
STATES = ItemHistory(
    "B",
    initial_state="ok",
    interval_censored_states=[[400, 800, "worn"]],
    right_censored=1000,
)
BEFORE_LOOK = ItemHistory(
    "C", [700], left_censored=300, left_censored_counts=1, right_censored=1000
)


# 1. a window that cuts a span ends or starts at the span's inspection


def test_censor_inside_counted_failures_ends_at_the_last_good_look():
    cut = censor_at(COUNTED, 600)
    assert cut.right_censored == 400
    assert cut.observed.tolist() == [100]
    assert len(cut.interval_censored) == 0


def test_enter_inside_counted_failures_starts_after_them():
    cut = enter_at(COUNTED, 600)
    assert cut.left_truncation == 800
    assert len(cut.observed) == 0 and len(cut.interval_censored) == 0
    assert cut.right_censored == 1000


def test_censor_inside_a_state_change_does_not_claim_the_old_state():
    cut = censor_at(STATES, 600)
    assert cut.right_censored == 400
    assert cut.state_at(400) == "ok"
    assert cut.state_at(600) is None


def test_enter_inside_a_state_change_starts_in_the_new_state():
    cut = enter_at(STATES, 600)
    assert cut.left_truncation == 800
    assert cut.initial_state == "worn"


def test_censor_before_a_first_look_knows_nothing():
    assert censor_at(BEFORE_LOOK, 200) is None


def test_enter_before_a_first_look_starts_at_the_look():
    cut = enter_at(BEFORE_LOOK, 200)
    assert cut.left_truncation == 300
    assert cut.left_censored is None
    assert cut.observed.tolist() == [700]


def test_windows_clear_of_spans_are_unchanged():
    assert censor_at(COUNTED, 900).right_censored == 900
    assert censor_at(COUNTED, 300).right_censored == 300
    assert enter_at(COUNTED, 300).left_truncation == 300
    assert enter_at(COUNTED, 900).left_truncation == 900


def test_dataset_windows_use_the_same_rule():
    data = EventHistoryData.from_items([COUNTED, BEFORE_LOOK])
    cut = data.censor_at(600)
    assert cut.item("A").right_censored == 400
    assert cut.item("C").right_censored == 600
    assert data.censor_at(200).ids == ["A"]


# 2. records out in one pass


def _best_time(f, runs=3):
    best = float("inf")
    for _ in range(runs):
        t = time.perf_counter()
        f()
        best = min(best, time.perf_counter() - t)
    return best


def _per_item(make, small, big):
    a, b = make(small), make(big)
    return (
        _best_time(a.to_items) / small,
        _best_time(b.to_items) / big,
    )


def test_lifetimes_to_items_is_linear():
    def make(k):
        return SurpyvalData.from_items(
            [ItemLifetime(j, observed=1.0 + j % 97) for j in range(k)]
        )

    per_small, per_big = _per_item(make, 1_000, 16_000)
    assert per_big < 8 * per_small, (per_small, per_big)


def test_histories_to_items_is_linear():
    def make(k):
        return EventHistoryData.from_items(
            [
                ItemHistory(j, [1.0, 2.0 + j % 7], right_censored=10)
                for j in range(k)
            ]
        )

    per_small, per_big = _per_item(make, 500, 8_000)
    assert per_big < 8 * per_small, (per_small, per_big)


def test_to_items_matches_item_by_item():
    data = EventHistoryData.from_items([BEFORE_LOOK, COUNTED])
    assert data.to_items() == [data.item(i) for i in data.ids]
    lives = SurpyvalData.from_items(
        [ItemLifetime("x", observed=5, covariates=[1, 2])]
    )
    assert lives.to_items() == [lives.item("x")]


# 5. one row, one item


def test_summaries_count_in_words():
    one = SurpyvalData.from_items([ItemLifetime("x", observed=5)])
    assert one.summary().startswith("SurpyvalData: 1 row, 1 item")
    history = EventHistoryData.from_items([COUNTED])
    assert history.summary().startswith("EventHistoryData: 1 item, 3 rows")


# 6. covariates are picked by column number


def test_covariates_by_name_say_they_go_by_number():
    data = SurpyvalData.from_items(
        [ItemLifetime("x", observed=5, covariates=[1, 2])]
    )
    with pytest.raises(ValueError, match="column number .* no names"):
        data.with_covariates(["age"])
    with pytest.raises(ValueError, match="column number"):
        data.with_covariates("age")
    assert np.asarray(data.with_covariates(1).Z).tolist() == [[2.0]]
    assert np.asarray(data.with_covariates(np.array([1, 0])).Z).tolist() == [
        [2.0, 1.0]
    ]
