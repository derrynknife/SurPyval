"""Reshaping a dataset for another analysis: without, with_covariates,
for_event_type, relabel_event_types, select, censor_at, enter_at -- on
SurpyvalData and EventHistoryData, each built again from its records."""

import numpy as np
import pytest

import surpyval as sp
from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData
from surpyval import recurrent as rc

LIVES = [
    ItemLifetime("A", observed=5, event_type="seal", covariates=[1, 0.5]),
    ItemLifetime("B", observed=7, event_type="bearing", covariates=[0, 0.2]),
    ItemLifetime("C", right_censored=9, covariates=[1, 0.1]),
    ItemLifetime(
        "D",
        interval_censored=[4, 8],
        event_type="bearing",
        covariates=[0, 0.4],
    ),
    ItemLifetime("E", left_censored=3, event_type="seal", covariates=[1, 0.9]),
]
DATA = SurpyvalData.from_items(LIVES, counts=[2, 1, 3, 1, 1])


def test_without_drops_parts_and_keeps_counts():
    plain = DATA.without("covariates", "event_types")
    assert plain.Z is None and plain.e is None
    assert plain.n.tolist() == [2, 1, 3, 1, 1]
    assert plain.ids == ["A", "B", "C", "D", "E"]
    np.testing.assert_allclose(
        sp.Weibull.fit(plain).params,
        sp.Weibull.fit(DATA.x, DATA.c, DATA.n).params,
    )
    with pytest.raises(ValueError, match="'colour'"):
        DATA.without("colour")
    with pytest.raises(ValueError, match="parts to leave out"):
        DATA.without()


def test_with_covariates_picks_columns():
    one = DATA.without("event_types").with_covariates([1])
    assert one.Z.ravel().tolist() == [0.5, 0.2, 0.1, 0.4, 0.9]
    with pytest.raises(ValueError, match="not among the 2"):
        DATA.with_covariates([2])
    with pytest.raises(ValueError, match="no covariates"):
        DATA.without("covariates").with_covariates([0])


def test_for_event_type_censors_the_other_modes():
    seal = DATA.for_event_type("seal").without("covariates")
    # A seal: observed; B bearing at 7 -> censored at 7; C censored;
    # D bearing in [4, 8] -> censored at its last good look, 4;
    # E seal before 3 -> left censored.
    assert [seal.item(i) for i in ("A", "B", "D")] == [
        ItemLifetime("A", observed=5),
        ItemLifetime("B", right_censored=7),
        ItemLifetime("D", right_censored=4),
    ]
    assert seal.e is None
    bearing = DATA.for_event_type("bearing")
    # E failed of a seal at an unknown age before 3: left out
    assert bearing.ids == ["A", "B", "C", "D"]
    with pytest.raises(ValueError, match="No failures of type 'gear'"):
        DATA.for_event_type("gear")
    with pytest.raises(ValueError, match="no event types"):
        DATA.without("event_types").for_event_type("seal")


def test_relabel_merges_modes():
    merged = DATA.relabel_event_types({"seal": "leak", "bearing": "leak"})
    assert {v for v in merged.e if v} == {"leak"}
    with pytest.raises(ValueError, match="No failures of type"):
        DATA.relabel_event_types({"gear": "x"})


def test_select_by_id_and_by_record():
    assert DATA.select(ids=["B", "C"]).ids == ["B", "C"]
    site1 = DATA.select(where=lambda item: item.covariates[0] == 1)
    assert site1.ids == ["A", "C", "E"] and site1.n.tolist() == [2, 3, 1]
    with pytest.raises(KeyError, match="'X'"):
        DATA.select(ids=["X"])
    with pytest.raises(ValueError, match="No items are left"):
        DATA.select(where=lambda item: False)


def test_censor_at_stops_observation():
    cut = DATA.censor_at(6)
    assert cut.item("A") == LIVES[0]  # failed at 5, before the end
    assert cut.item("B") == ItemLifetime(
        "B", right_censored=6, covariates=[0, 0.2]
    )
    assert cut.item("C") == ItemLifetime(
        "C", right_censored=6, covariates=[1, 0.1]
    )
    # failed in [4, 8], across 6: known only to work at 4
    assert cut.item("D") == ItemLifetime(
        "D", right_censored=4, covariates=[0, 0.4]
    )
    assert cut.item("E") == LIVES[4]


def test_enter_at_keeps_items_known_to_reach_the_age():
    later = DATA.enter_at(4.5)
    # A (5), B (7), C (9) reached 4.5; D failed in [4, 8], maybe before;
    # E failed before 3
    assert later.ids == ["A", "B", "C"]
    assert later.item("A").left_truncation == 4.5
    np.testing.assert_allclose(later.tl, 4.5)


def test_plain_data_stays_without_ids():
    plain = SurpyvalData([1, 2, 3, 4], c=[0, 0, 1, 0])
    assert plain.censor_at(2.5).ids is None
    np.testing.assert_array_equal(plain.censor_at(2.5).c, [0, 0, 1, 1])


# -- event histories ------------------------------------------------------

HISTORIES = [
    ItemHistory(
        "M1",
        observed=[420, 1310, 2050],
        observed_event_types=["seal", "bearing", "seal"],
        right_censored=2600,
        covariates=[1.0],
        initial_state="ok",
        observed_states={1000: "worn"},
    ),
    ItemHistory(
        "M2",
        observed=[980, 2400],
        observed_event_types="seal",
        interval_censored=[[2500, 2700]],
        interval_censored_event_types="bearing",
        covariates=[0.0],
        initial_state="ok",
        interval_censored_states=[[1500, 1800, "worn"]],
    ),
]
EHD = EventHistoryData.from_items(HISTORIES)


def test_history_without_and_for_event_type():
    plain = EHD.without("event_types", "states", "covariates")
    assert rc.CrowAMSAA.fit(plain).params.shape == (2,)
    seal = EHD.for_event_type("seal").without("states", "covariates")
    m2 = seal.item("M2")
    # the bearing failures are gone, and M2 was watched to 2700
    assert m2.observed.tolist() == [980, 2400] and m2.right_censored == 2700
    assert seal.e is None


def test_history_censor_at_and_enter_at():
    cut = EHD.censor_at(2000)
    m1, m2 = cut.item("M1"), cut.item("M2")
    assert m1.observed.tolist() == [420, 1310] and m1.right_censored == 2000
    assert m2.interval_censored_states == [[1500, 1800, "worn"]]
    late = EHD.enter_at(1200)
    assert late.item("M1").initial_state == "worn"
    assert late.item("M1").observed.tolist() == [1310, 2050]
    assert late.item("M1").left_truncation == 1200
    # M2's state at 1600 is not known (it changed between 1500 and 1800),
    # so it is watched from 1800, when it is known to be worn
    m2 = EHD.enter_at(1600).item("M2")
    assert m2.left_truncation == 1800 and m2.initial_state == "worn"


def test_history_select_and_relabel():
    assert EHD.select(where=lambda h: len(h.observed) > 2).ids == ["M1"]
    merged = EHD.relabel_event_types({"bearing": "seal"})
    assert {v for v in merged.e if v is not None} == {"seal"}
    assert EHD.without("states").for_event_type("seal").states is None
