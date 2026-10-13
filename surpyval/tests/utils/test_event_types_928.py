"""Event types (#928): masked causes, types on rows that have not failed,
tuple ids and types, several types at one inspection, and overlapping
inspections of different types."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import (
    EventHistoryData,
    ItemHistory,
    ItemLifetime,
    SurpyvalData,
    handle_xicn,
)
from surpyval.utils.warnings import MaskedCauseWarning


def _quiet(make):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return make()


# Masked causes: a failure whose type was not recorded

LIVES = [
    ItemLifetime("A", observed=100, event_type="seal"),
    ItemLifetime("B", observed=150, event_type=np.nan),
    ItemLifetime("C", observed=180, event_type="bearing"),
    ItemLifetime("D", right_censored=300, event_type=np.nan),
]


@pytest.mark.parametrize("missing", [None, np.nan, "", "  "])
def test_a_missing_type_is_none(missing):
    assert ItemLifetime("x", observed=1, event_type=missing).event_type is None
    assert (
        ItemLifetime("x", right_censored=1, event_type=missing).event_type
        is None
    )


def test_masked_causes_are_warned_of_once_by_item_and_summarised():
    with pytest.warns(MaskedCauseWarning, match="items 'B'") as caught:
        data = SurpyvalData.from_items(LIVES)
    assert len([w for w in caught if w.category is MaskedCauseWarning]) == 1
    assert "not recorded (1)" in data.summary()
    assert list(data.e) == ["seal", None, "bearing", None]


def test_fits_of_every_failure_count_masked_causes():
    data = _quiet(lambda: SurpyvalData.from_items(LIVES))
    plain = data.without("event_types")
    assert (plain.c == 0).sum() == 3


def test_for_event_type_takes_masked_causes_as_asked():
    data = _quiet(lambda: SurpyvalData.from_items(LIVES))
    with pytest.warns(MaskedCauseWarning, match="underestimated"):
        censored = data.for_event_type("seal")
    assert censored.item("B").outcome == "right_censored"
    counted = data.for_event_type("seal", masked="count")
    assert counted.item("B").outcome == "observed"
    assert counted.item("C").outcome == "right_censored"
    with pytest.raises(ValueError, match="masked causes"):
        data.for_event_type("seal", masked="refuse")
    with pytest.raises(ValueError, match="masked must be one of"):
        data.for_event_type("seal", masked="drop")


def test_relabel_gives_masked_causes_a_type_and_leaves_suspensions():
    data = _quiet(lambda: SurpyvalData.from_items(LIVES))
    relabelled = data.relabel_event_types({np.nan: "unknown"})
    assert list(relabelled.e) == ["seal", "unknown", "bearing", None]
    assert sp.CompetingRisks.fit(relabelled) is not None  # fits once typed


def test_competing_risks_refuse_masked_causes_saying_what_to_do():
    data = _quiet(lambda: SurpyvalData.from_items(LIVES))
    with pytest.raises(ValueError, match="masked causes.*relabel_event_types"):
        sp.CompetingRisks.fit(data)


def test_histories_with_masked_causes():
    with pytest.warns(MaskedCauseWarning, match="items 'P1'"):
        data = EventHistoryData.from_items(
            [
                ItemHistory(
                    "P1",
                    [300, 700],
                    observed_event_types=["seal", None],
                    right_censored=900,
                ),
                ItemHistory(
                    "P2",
                    [400],
                    observed_event_types="bearing",
                    right_censored=900,
                ),
            ]
        )
    seal = _quiet(lambda: data.for_event_type("seal"))
    assert seal.item("P1").observed.tolist() == [300]
    counted = data.for_event_type("seal", masked="count")
    assert counted.item("P1").observed.tolist() == [300, 700]
    assert data.relabel_event_types({None: "unknown"}).item(
        "P1"
    ).observed_event_types.tolist() == ["seal", "unknown"]


# Types on rows that have not failed


def test_a_type_on_a_suspension_is_dropped_with_a_warning():
    with pytest.warns(UserWarning, match="1 suspension .*rows 1"):
        data = SurpyvalData(
            [100, 150, 200], c=[0, 1, 0], e=["seal", "seal", "bearing"]
        )
    assert list(data.e) == ["seal", None, "bearing"]
    assert data.without("event_types").e is None
    with pytest.warns(UserWarning, match="suspension"):
        model = sp.CompetingRisks.fit(
            [100, 150, 200], e=["seal", "seal", "bearing"], c=[0, 1, 0]
        )
    assert model is not None


def test_a_type_on_an_end_of_observation_row_is_dropped_with_a_warning():
    with pytest.warns(UserWarning, match="suspension"):
        log = handle_xicn(
            [100, 250, 400],
            i=["P1"] * 3,
            c=[0, 0, 1],
            e=["seal", "bearing", "seal"],
        )
    assert list(log.e) == ["seal", "bearing", None]
    back = EventHistoryData.from_items(log.to_items())
    assert list(back.e) == list(log.e)


# Tuple ids and types


def test_tuple_ids_survive_slicing_and_json():
    gaps = EventHistoryData.from_items(
        [ItemHistory("A", [5, 9], right_censored=12)]
    ).to_gaps()
    assert gaps[0].ids == [("A", 1)]
    saved = SurpyvalData.from_json(gaps.to_json())
    assert saved.ids == gaps.ids
    assert saved.item(("A", 1)) == gaps.item(("A", 1))


def test_tuple_types_survive_slicing_and_json():
    data = SurpyvalData.from_items(
        [
            ItemLifetime("x", observed=3, event_type=("seal", "leak")),
            ItemLifetime("y", observed=4, event_type="bearing"),
        ]
    )
    assert list(data[0].e) == [("seal", "leak")]
    saved = SurpyvalData.from_json(data.to_json())
    assert saved.for_event_type(("seal", "leak")).c.tolist() == [0, 1]


# Several types at one inspection, and overlapping inspections

TWO_AT_ONCE = ItemHistory(
    "P2",
    interval_censored=[[1000, 1500], [1000, 1500]],
    interval_censored_event_types=["seal", "bearing"],
    right_censored=2000,
)
CROSSED = ItemHistory(
    "P3",
    interval_censored=[[1000, 1500], [1200, 1800]],
    interval_censored_event_types=["seal", "bearing"],
    right_censored=2000,
)


def test_two_types_found_at_one_inspection():
    data = EventHistoryData.from_items([TWO_AT_ONCE])
    assert data.for_event_type("seal").item(
        "P2"
    ).interval_censored.tolist() == [[1000, 1500]]
    # Without the types, one inspection that found two failures
    plain = data.without("event_types").item("P2")
    assert plain.interval_censored.tolist() == [[1000, 1500]]
    assert plain.interval_censored_counts.tolist() == [2]


def test_the_same_window_twice_for_one_type_is_one_count():
    twice = ItemHistory(
        "P",
        interval_censored=[[1000, 1500], [1000, 1500]],
        interval_censored_event_types="seal",
        right_censored=2000,
    )
    assert twice.interval_censored_counts.tolist() == [2]


def test_inspections_of_different_types_may_overlap():
    data = EventHistoryData.from_items([CROSSED])
    assert data.for_event_type("bearing").item(
        "P3"
    ).interval_censored.tolist() == [[1200, 1800]]
    with pytest.raises(ValueError, match="overlap"):
        data.without("event_types")


def test_inspections_of_one_type_may_not_overlap():
    with pytest.raises(ValueError, match="of type 'seal' .*overlap"):
        ItemHistory(
            "P",
            interval_censored=[[1000, 1500], [1200, 1800]],
            interval_censored_event_types="seal",
            right_censored=2000,
        )
    with pytest.raises(ValueError, match="counted twice"):
        ItemHistory(
            "P",
            [1200],
            observed_event_types="seal",
            interval_censored=[[1000, 1500]],
            interval_censored_event_types="seal",
            right_censored=2000,
        )
    with pytest.raises(ValueError, match="same inspection window"):
        handle_xicn(
            [[1000, 1500], [1000, 1500], [2000, 2000]],
            i=[1, 1, 1],
            c=[2, 2, 1],
            e=["seal", "seal", None],
        )


def test_a_fit_of_every_failure_refuses_crossed_inspections():
    data = EventHistoryData.from_items([TWO_AT_ONCE, CROSSED])
    with pytest.raises(ValueError, match="without\\('event_types'\\)"):
        sp.recurrent.CrowAMSAA.fit_from_recurrent_data(data)


def test_first_events_found_together_are_masked_causes():
    data = EventHistoryData.from_items(
        [
            TWO_AT_ONCE,
            ItemHistory(
                "P7",
                [1200],
                observed_event_types=["bearing"],
                interval_censored=[[1000, 1500]],
                interval_censored_event_types=["seal"],
                right_censored=2000,
            ),
            ItemHistory(
                "P9", [300, 700], observed_event_types=["seal", "bearing"]
            ),
        ]
    )
    with pytest.warns(MaskedCauseWarning, match="'P2', 'P7'"):
        first = data.to_first_events()
    assert first.item("P2") == ItemLifetime(
        "P2", interval_censored=[1000, 1500]
    )
    assert first.item("P7") == ItemLifetime(
        "P7", interval_censored=[1000, 1200]
    )
    assert first.item("P9") == ItemLifetime(
        "P9", observed=300, event_type="seal"
    )
    seal = data.for_event_type("seal").to_first_events()
    assert seal.item("P7") == ItemLifetime(
        "P7", interval_censored=[1000, 1500]
    )
