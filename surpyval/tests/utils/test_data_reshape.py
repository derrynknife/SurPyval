"""Reshaping a dataset for another analysis: without, with_covariates,
for_event_type, relabel_event_types, select, censor_at, enter_at -- on
SurpyvalData and EventHistoryData, each built again from its records."""

import numpy as np
import pytest

import surpyval as sp
from surpyval import EventHistoryData, ItemHistory, ItemLifetime, SurpyvalData
from surpyval import recurrent as rc
from surpyval.utils.warnings import MaskedCauseWarning

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


# #923: one mode of right-truncated lifetimes


def _truncated_modes():
    items = [
        ItemLifetime(k, observed=k, event_type="B", right_truncation=25)
        for k in range(1, 21)
    ]
    items += [
        ItemLifetime(
            100,
            observed=5,
            event_type="A",
            left_truncation=1,
            right_truncation=25,
        ),
        ItemLifetime(101, observed=7, event_type="A", right_truncation=25),
    ]
    return SurpyvalData.from_items(items)


def test_for_event_type_drops_right_truncation_with_a_warning():
    with pytest.warns(UserWarning, match="drops the data's right truncation"):
        only_a = _truncated_modes().for_event_type("A")
    # Another mode's failure is a suspension, not an A failure before 25
    assert only_a.item(3) == ItemLifetime(3, right_censored=3)
    assert only_a.item(100) == ItemLifetime(100, observed=5, left_truncation=1)
    assert not np.isfinite(only_a.tr).any()
    assert (only_a.c == 0).sum() == 2


def test_for_event_type_without_right_truncation_does_not_warn():
    import warnings

    data = SurpyvalData.from_items(
        [
            ItemLifetime(1, observed=2, event_type="A", left_truncation=1),
            ItemLifetime(2, observed=3, event_type="B"),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        only_a = data.for_event_type("A")
    assert only_a.item(1).left_truncation == 1


# -- #932: the array paths give what the records give; input checks ---------


def _varied(seed=0, n=60):
    """Lifetime data of every kind: each outcome, truncation on each side,
    typed, masked and untyped failures, named covariates and counts."""
    rng = np.random.default_rng(seed)
    items = []
    for k in range(n):
        tl = float(rng.integers(0, 3)) if rng.random() < 0.4 else None
        base = (tl or 0) + 1 + float(rng.integers(0, 10))
        kind = rng.choice(["observed", "right", "left", "interval"])
        tr = base + 20 if rng.random() < 0.3 else None
        event = (
            rng.choice(["seal", "bearing", None]) if kind != "right" else None
        )
        covariates = {"load": float(rng.random()), "site": float(k % 3)}
        if kind == "observed":
            item = ItemLifetime(
                f"P{k}",
                observed=base,
                left_truncation=tl,
                right_truncation=tr,
                event_type=event,
                covariates=covariates,
            )
        elif kind == "right":
            item = ItemLifetime(
                f"P{k}",
                right_censored=base,
                left_truncation=tl,
                right_truncation=tr,
                covariates=covariates,
            )
        elif kind == "left":
            item = ItemLifetime(
                f"P{k}",
                left_censored=base,
                left_truncation=tl,
                right_truncation=tr,
                event_type=event,
                covariates=covariates,
            )
        else:
            item = ItemLifetime(
                f"P{k}",
                interval_censored=[base, base + 3],
                left_truncation=tl,
                right_truncation=tr,
                event_type=event,
                covariates=covariates,
            )
        items.append(item)
    counts = rng.integers(1, 4, n)
    with pytest.warns(MaskedCauseWarning):
        return SurpyvalData.from_items(items, counts=counts)


def _same(a, b):
    for u, v in zip(a.to_xcnt(), b.to_xcnt()):
        np.testing.assert_array_equal(u, v)
    assert a.ids == b.ids
    assert (a.e is None) == (b.e is None)
    if a.e is not None:
        assert list(a.e) == list(b.e)
    assert (a.Z is None) == (b.Z is None)
    if a.Z is not None:
        np.testing.assert_array_equal(a.Z, b.Z)
    assert a.covariate_names == b.covariate_names
    assert a.x.ndim == b.x.ndim


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_issue_932_array_reshapes_equal_the_record_reshapes(seed):
    from surpyval.utils import data_reshape as by_record

    data = _varied(seed)
    ages = [2.5, 6.0, 11.0, 30.0]
    cases = [
        (
            lambda d: d.without("covariates"),
            lambda r: by_record.without(r, ("covariates",)),
        ),
        (
            lambda d: d.without("event_types", "truncation"),
            lambda r: by_record.without(r, ("event_types", "truncation")),
        ),
        (
            lambda d: d.with_covariates(["site"]),
            lambda r: by_record.with_covariates(r, ["site"]),
        ),
        (
            lambda d: d.with_covariates([1, 0]),
            lambda r: by_record.with_covariates(r, [1, 0]),
        ),
        (
            lambda d: d.relabel_event_types({"seal": "s", None: "unknown"}),
            lambda r: by_record.relabel(r, {"seal": "s", None: "unknown"}),
        ),
        (
            lambda d: d.for_event_type("seal"),
            lambda r: by_record.for_event_type(r, "seal"),
        ),
        (
            lambda d: d.for_event_type("bearing", masked="count"),
            lambda r: by_record.for_event_type(r, "bearing", "count"),
        ),
    ]
    for age in ages:
        cases.append(
            (
                lambda d, a=age: d.censor_at(a),
                lambda r, a=age: by_record.censor_at(r, a),
            )
        )
        cases.append(
            (
                lambda d, a=age: d.enter_at(a),
                lambda r, a=age: by_record.enter_at(r, a),
            )
        )
    for fast, slow in cases:
        try:
            expected = data._map(slow)
        except ValueError as error:
            # Nothing is left (enter_at past every item): both refuse
            with pytest.raises(ValueError, match=str(error)):
                fast(data)
            continue
        _same(fast(data), expected)
    # select by id keeps the data's order, as the records do
    wanted = [f"P{k}" for k in range(59, -1, -7)]
    _same(
        data.select(ids=wanted),
        data._map(lambda r: r if r.id in wanted else None),
    )


def test_issue_932_several_modes_as_one():
    both = DATA.for_event_type(["seal", "bearing"])
    assert both.e is None
    assert both.c.tolist() == [0, 0, 1, 2, -1]
    merged = DATA.relabel_event_types({"bearing": "seal"}).for_event_type(
        "seal"
    )
    _same(both, merged)
    with pytest.raises(ValueError, match="No failures of type 'x'"):
        DATA.for_event_type(["seal", "x"])


@pytest.mark.parametrize(
    "age, match",
    [
        (-5, r"censor_at\(-5\): the data's ages are measured from 0"),
        (0, r"censor_at\(0\): the data's ages"),
        ("1000", "takes one age .*got '1000'"),
        (np.nan, "finite age"),
        ([5, 6], "takes one age"),
    ],
)
def test_issue_932_cut_off_ages_are_checked(age, match):
    with pytest.raises(ValueError, match=match):
        DATA.censor_at(age)
    with pytest.raises(
        ValueError, match=match.replace("censor_at", "enter_at")
    ):
        DATA.enter_at(age)


def test_issue_932_dates_and_durations_say_how_to_convert():
    import datetime

    import pandas as pd

    with pytest.raises(ValueError, match="not a date: the same date is a"):
        DATA.censor_at(pd.Timestamp("2024-06-30"))
    with pytest.raises(ValueError, match="not a date"):
        DATA.enter_at(datetime.date(2024, 6, 30))
    with pytest.raises(ValueError, match="not a duration: convert it"):
        DATA.censor_at(pd.Timedelta(days=3))


def test_issue_932_real_line_data_takes_a_cut_off_below_zero():
    data = SurpyvalData([-3.0, 2.0, 5.0], c=[0, 0, 1])
    assert data.censor_at(-1).c.tolist() == [0, 1, 1]
    assert data.enter_at(-1).tl.tolist() == [-1.0, -1.0]


def test_issue_932_select_takes_one_id_and_keeps_the_order():
    assert DATA.select(ids="A").ids == ["A"]
    assert DATA.select(ids=["D", "A"]).ids == ["A", "D"]
    numbered = SurpyvalData.from_items(
        [ItemLifetime(k, observed=k + 1) for k in range(3)]
    )
    assert numbered.select(ids=1).ids == [1]
    with pytest.raises(KeyError, match="No items 'X', 'Y' in the data"):
        DATA.select(ids=["X", "A", "Y"])
    with pytest.raises(ValueError, match="select takes the ids"):
        DATA.select()
    # A tuple that is an id is one id; one that is not is a list of ids
    gaps = SurpyvalData.from_items(
        [
            ItemLifetime(("A", 1), observed=1),
            ItemLifetime(("A", 2), observed=2),
        ]
    )
    assert gaps.select(ids=("A", 1)).ids == [("A", 1)]
    assert DATA.select(ids=("A", "B")).ids == ["A", "B"]


def test_issue_932_without_takes_a_list():
    _same(
        DATA.without(["covariates", "event_types"]),
        DATA.without("covariates", "event_types"),
    )
    with pytest.raises(ValueError, match="takes the parts to leave out"):
        DATA.without([])


def test_issue_932_with_covariates_needs_a_column():
    with pytest.raises(ValueError, match="give at least one, or use without"):
        DATA.with_covariates([])


def test_issue_932_data_without_ids_says_so_when_selected_by_id():
    plain = SurpyvalData([5, 6, 7, 8], [0, 1, 0, 1])
    with pytest.raises(
        ValueError, match="needs item ids, and this data has none"
    ):
        plain.select(ids=[2, 3])
    assert plain.select(where=lambda item: item.value > 6).ids is None
    assert plain.item(2) == ItemLifetime(2, observed=7.0)


def test_issue_932_ids_on_the_constructor():
    data = SurpyvalData([9, 1, 5], c=[0, 1, 0], ids=["c", "a", "b"])
    assert data.ids == ["c", "a", "b"] and data.x.tolist() == [9.0, 1.0, 5.0]
    assert data.item("a") == ItemLifetime("a", right_censored=1.0)
    assert data.select(ids="b").x.tolist() == [5.0]
    with pytest.raises(ValueError, match="one id per entry"):
        SurpyvalData([9, 1, 5], ids=["c", "a"])
    with pytest.raises(ValueError, match="given more than once: 'a'"):
        SurpyvalData([9, 1, 5], ids=["a", "a", "b"])


def test_issue_932_reshapes_are_fast_at_fifty_thousand_items():
    import time

    rng = np.random.default_rng(0)
    x = rng.weibull(2, 50_000) * 100
    c = (x > 120).astype(int)
    data = SurpyvalData(x, c, ids=[f"P{k}" for k in range(len(x))])
    for op in (
        lambda: data.without("truncation"),
        lambda: data.select(ids=[f"P{k}" for k in range(0, 50_000, 2)]),
        lambda: data.censor_at(100),
    ):
        start = time.perf_counter()
        op()
        assert time.perf_counter() - start < 0.5


# -- #937: a cut-off per item, by dict or by a function of the record ----


def test_issue_937_cut_offs_per_item():
    data = SurpyvalData.from_items(
        [
            ItemLifetime("P1", observed=500, attributes={"installed": 100}),
            ItemLifetime(
                "P2", right_censored=1200, attributes={"installed": 0}
            ),
            ItemLifetime("P3", observed=900, attributes={"installed": 300}),
        ]
    )
    by_id = data.censor_at({"P2": 1000, "P3": 800})
    assert by_id.c.tolist() == [0, 1, 1]
    assert by_id.x.tolist() == [500.0, 1000.0, 800.0]
    # The same as combining per-item cuts
    parts = [
        data.select(ids="P1"),
        data.select(ids="P2").censor_at(1000),
        data.select(ids="P3").censor_at(800),
    ]
    joined = SurpyvalData.from_items(sum((p.to_items() for p in parts), []))
    assert joined.to_items() == by_id.to_items()
    # A study ending at calendar time 1000: each item's age at the end
    study_end = 1000
    by_record = data.censor_at(
        lambda item: study_end - item.attribute("installed")
    )
    assert by_record.x.tolist() == [500.0, 1000.0, 700.0]
    assert by_record.c.tolist() == [0, 1, 1]
    entered = data.enter_at({"P1": 200, "P3": 1000})
    assert entered.ids == ["P1", "P2"] and entered.tl.tolist() == [
        200.0,
        -np.inf,
    ]
    assert data.enter_at(lambda item: 200).tl.tolist() == [200.0] * 3
    with pytest.raises(KeyError, match="No item 'X'"):
        data.censor_at({"X": 5})
    with pytest.raises(ValueError, match="needs item ids, and this data"):
        SurpyvalData([1, 2]).censor_at({0: 1.5})
    with pytest.raises(ValueError, match="censor_at takes one age"):
        data.censor_at({"P1": "2024-01-01"})
    history = EventHistoryData.from_items(HISTORIES)
    cut = history.censor_at({"M1": 1500})
    assert (
        cut.item("M1").right_censored == 1500
        and cut.item("M2") == HISTORIES[1]
    )
