"""The notation's event types, covariates, histories, readings and
time-varying covariates (surpyval.utils.notation): each checked against
the same data built by hand or through the item records."""

import warnings

import numpy as np
import pytest

from surpyval import (
    EventHistoryData,
    ItemHistory,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
    Weibull,
)
from surpyval.recurrent import CrowAMSAA, NonParametricCounting
from surpyval.utils.notation import (
    IC,
    LC,
    RC,
    Item,
    Items,
    O,
    Observations,
    Truncation,
)

inf = np.inf


def _without_ids(data):
    data.ids = None
    return data


# --- 1. Event types and fixed covariates on observations -----------------


def test_event_types_and_covariates_map_as_the_item_records_do():
    obs = Observations(
        [
            O(10, event_type="bearing", covariates={"temp": 60, "load": 1}),
            RC(20, covariates={"load": 2, "temp": 70}),
            IC(5, 9, event_type="seal", covariates={"temp": 65, "load": 3}),
            LC(4, event_type="seal", covariates={"temp": 50, "load": 0}),
        ]
    )
    data = obs.surpyval_data()
    expected = SurpyvalData.from_items(
        [
            ItemLifetime(
                0,
                observed=10,
                event_type="bearing",
                covariates={"temp": 60, "load": 1},
            ),
            ItemLifetime(
                1, right_censored=20, covariates={"temp": 70, "load": 2}
            ),
            ItemLifetime(
                2,
                interval_censored=[5, 9],
                event_type="seal",
                covariates={"temp": 65, "load": 3},
            ),
            ItemLifetime(
                3,
                left_censored=4,
                event_type="seal",
                covariates={"temp": 50, "load": 0},
            ),
        ]
    )
    assert data == _without_ids(expected)
    assert data.covariate_names == ["temp", "load"]
    np.testing.assert_array_equal(data.Z[1], [70, 2])  # by name, not order


def test_counts_and_windows_keep_types_and_covariates():
    obs = Observations(
        [
            3 * O(10, event_type="a", covariates={"x": 1}),
            RC(20, covariates={"x": 0}) / Truncation(2, 40),
        ]
    )
    data = obs.surpyval_data()
    np.testing.assert_array_equal(data.n, [3, 1])
    np.testing.assert_array_equal(data.tl, [-inf, 2])
    assert list(data.e) == ["a", None]


def test_a_set_without_the_same_covariate_names_is_refused():
    with pytest.raises(ValueError, match="is missing 'load'"):
        Observations(
            [
                O(1, covariates={"temp": 1, "load": 2}),
                O(2, covariates={"temp": 1}),
            ]
        ).surpyval_data()
    with pytest.raises(ValueError, match="has 'humidity'"):
        Observations(
            [
                O(1, covariates={"temp": 1}),
                O(2, covariates={"temp": 1, "humidity": 2}),
            ]
        ).surpyval_data()
    with pytest.raises(
        ValueError, match=r"observation 1 \(failed at 2\) has none"
    ):
        Observations([O(1, covariates={"temp": 1}), O(2)]).surpyval_data()


def test_bad_types_and_covariates_are_refused_when_built():
    with pytest.raises(ValueError, match="has not failed"):
        RC(3, event_type="bearing")
    with pytest.raises(ValueError, match="has not failed"):
        IC(3, inf, event_type="bearing")  # an open interval is an RC
    with pytest.raises(ValueError, match="dict"):
        O(3, covariates=[1, 2])
    with pytest.raises(ValueError, match="must be a number"):
        O(3, covariates={"site": "north"})
    with pytest.raises(ValueError, match="finite"):
        O(3, covariates={"temp": inf})
    with pytest.raises(ValueError, match="names no covariate"):
        O(3, covariates={})


def test_types_and_covariates_take_part_in_equality_and_printout():
    assert O(1, event_type="a") != O(1)
    assert O(1, covariates={"a": 1, "b": 2}) == O(
        1, covariates={"b": 2, "a": 1}
    )
    assert hash(O(1, covariates={"a": 1, "b": 2})) == hash(
        O(1, covariates={"b": 2, "a": 1})
    )
    assert repr(O(10, event_type="bearing", covariates={"temp": 60})) == (
        "failed at 10 (type 'bearing'; temp = 60)"
    )


def test_text_leaves_types_and_covariates_out():
    obs = Observations(
        [
            O(10, event_type="a", covariates={"t": 1}),
            RC(20, covariates={"t": 2}),
        ]
    )
    assert obs.text() == "10, 20+"
    assert Observations.parse(obs.text()) == Observations([O(10), RC(20)])


def test_to_notation_keeps_named_covariates_and_event_types():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        data = SurpyvalData(
            [3, 5, 7],
            c=[0, 1, 0],
            Z=[[1.0], [2.0], [3.0]],
            covariate_names=["dose"],
            e=["a", None, None],
        )
        back = SurpyvalData.from_notation(data.to_notation())
    assert back == data
    assert data.to_notation()[0] == O(
        3, event_type="a", covariates={"dose": 1}
    )


def test_terminal_is_refused_outside_a_history():
    with pytest.raises(ValueError, match="Item"):
        Observations([O(1, terminal=True)]).surpyval_data()
    with pytest.raises(ValueError, match="end of watching"):
        RC(5, terminal=True)


def test_a_fit_of_typed_observations_is_the_fit_of_the_arrays():
    from surpyval.univariate.regression import WeibullPH

    obs = Observations(
        [
            O(t, covariates={"x": z})
            for t, z in [(3, 0), (5, 1), (7, 0), (2, 1), (9, 0), (4, 1)]
        ]
        + [RC(10, covariates={"x": 0})]
    )
    model = WeibullPH.fit(obs)
    arrays = WeibullPH.fit(
        x=[3, 5, 7, 2, 9, 4, 10],
        c=[0] * 6 + [1],
        Z=[[0], [1], [0], [1], [0], [1], [0]],
    )
    np.testing.assert_allclose(model.params, arrays.params, rtol=1e-10)


# --- 2. Histories --------------------------------------------------------


def _pump():
    return Item("P1", O(120), O(310), IC(400, 450), RC(1000)) / Truncation(0)


def test_a_history_is_the_same_data_built_by_hand():
    data = Items(
        [_pump(), Item("F1", 3 * O(200), O(800, terminal=True))]
    ).surpyval_data()
    expected = SurpyvalData(
        x=[120, 310, [400, 450], 1000, 200, 200, 200, 800],
        c=[0, 0, 2, 1, 0, 0, 0, 0],
        n=[1] * 8,
        i=["P1"] * 4 + ["F1"] * 4,
        tl=[0, -inf, -inf, -inf, -inf, -inf, -inf, -inf],
        a=[0, 0, 0, 0, 0, 0, 0, 1],
    )
    assert data == expected


def test_repeated_ages_are_simultaneous_failures_and_counts_stay_on_one_row():
    data = Items(
        [Item("A", 2 * LC(50), 3 * O(120), 2 * IC(200, 300))]
    ).surpyval_data()
    np.testing.assert_array_equal(data.c, [-1, 0, 0, 0, 2])
    np.testing.assert_array_equal(data.n, [2, 1, 1, 1, 2])


def test_events_are_kept_in_time_order_whatever_order_they_are_given():
    assert Item("A", RC(50), O(30), O(10)) == Item("A", O(10), O(30), RC(50))
    assert Item("A", RC(50), O(30), O(10)).text() == "A: 10, 30, 50+"


def test_a_history_fits_as_its_event_history_data_does():
    items = Items.parse("""
        P1: 120, 310, [400, 450], 1000+
        P2: 3*200, 900 | (100, inf]
        P3: 50, 400, 650, 1200+
        """)
    np.testing.assert_allclose(
        CrowAMSAA.fit(items).params,
        CrowAMSAA.fit(items.event_history_data()).params,
        rtol=1e-12,
    )
    exact = Items([item for item in items if item.id != "P1"])
    np.testing.assert_allclose(
        NonParametricCounting.fit(exact).mcf_hat,
        NonParametricCounting.fit(exact.event_history_data()).mcf_hat,
        rtol=1e-12,
    )


def test_the_bridge_is_the_item_history_records():
    items = Items(
        [_pump(), Item("F1", O(800), covariates={"x": 1.0})], unit="h"
    )
    with pytest.raises(ValueError, match="for some and not others"):
        items.surpyval_data()
    items = Items(
        [
            Item("P1", O(120), RC(1000), covariates={"x": 0.0})
            / Truncation(0),
            Item("F1", O(800), covariates={"x": 1.0}),
        ],
        unit="h",
    )
    expected = EventHistoryData.from_items(
        [
            ItemHistory(
                "P1",
                observed=[120],
                right_censored=1000,
                left_truncation=0,
                covariates={"x": 0.0},
            ),
            ItemHistory("F1", observed=[800], covariates={"x": 1.0}),
        ],
        unit="h",
    )
    assert items.event_history_data() == expected
    data = items.surpyval_data()
    assert data.unit == "hours" or data.unit == "h"
    np.testing.assert_array_equal(data.Z, [[0.0], [0.0], [1.0]])


def test_a_terminal_event_is_a_equal_1():
    data = Items([Item("A", O(100), O(800, terminal=True))]).surpyval_data()
    np.testing.assert_array_equal(data.a, [0, 1])
    assert not data.a_inferred.any()
    with pytest.raises(ValueError, match="terminal flag"):
        Items([Item("A", O(100), O(800, terminal=True))]).event_history_data()


def test_items_of_one_lifetime_each_are_lifetime_data():
    data = Items([Item("A", O(5)), Item("B", RC(9))]).surpyval_data()
    expected = SurpyvalData.from_items(
        [ItemLifetime("A", observed=5), ItemLifetime("B", right_censored=9)]
    )
    assert data == expected and data.a is None
    np.testing.assert_allclose(
        Weibull.fit(
            Items([Item(k, O(t)) for k, t in enumerate([3, 5, 7, 9])])
        ).params,
        Weibull.fit([3, 5, 7, 9]).params,
    )


def test_history_refusals_say_what_to_do():
    cases = [
        (lambda: Item("P", O(5), RC(3)), "at or after every event"),
        (lambda: Item("P", RC(9), RC(10)), "give one"),
        (lambda: Item("P", LC(5), LC(9)), "at most one"),
        (lambda: Item("P", O(5, terminal=True), O(7)), "last event"),
        (lambda: Item("P", O(5, terminal=True), RC(7)), "leave out the RC"),
        (lambda: Item("P", 2 * O(5, terminal=True)), "one event ends"),
        (
            lambda: Item("P", O(3) / Truncation(0)),
            r"Item\(\.\.\.\) / Truncation",
        ),
        (
            lambda: Item("P", O(3, covariates={"x": 1})),
            "covariates belong to the item",
        ),
        (lambda: Item("P", O(5)) / Truncation(6), "left_truncation"),
        (lambda: Item("P", [O(1)]), "one by one"),
        (lambda: Item("P"), "no failures"),
        (
            lambda: Items([Item("A", O(1)), Item("A", O(2))]),
            "'A' is given more than once",
        ),
        (lambda: Items(Item("A", O(1))), r"give \[item\]"),
        (lambda: Items([O(1)]), "Observations"),
    ]
    for build, message in cases:
        with pytest.raises(ValueError, match=message):
            build()


def test_a_list_of_items_given_to_a_fit_is_refused_naming_items():
    with pytest.raises(TypeError, match="Items"):
        Weibull.fit([Item("A", O(1)), Item("B", O(2))])


def test_items_join_slice_and_compare():
    a = Items([_pump()], unit="hours")
    b = a + Item("F1", O(8))
    assert len(b) == 2 and b.unit == "hours" and b[1].id == "F1"
    assert b[:1] == a
    with pytest.raises(ValueError, match="convert"):
        a + Items([Item("F2", O(1))], unit="km")


# --- 3. Readings and time-varying covariates -----------------------------


def test_named_readings_become_measurements():
    data = Items(
        [
            Item("A", O(100), readings={"vibration": {0: 1.2, 50: 1.9}}),
            Item("B", RC(120), readings={"vibration": {0: 1.0}}),
        ]
    ).surpyval_data()
    expected = SurpyvalData(
        x=[100, 120],
        c=[0, 1],
        i=["A", "B"],
        m={"vibration": {"A": {0: 1.2, 50: 1.9}, "B": {0: 1.0}}},
    )
    assert data == expected


def test_one_unnamed_series_is_the_markers():
    data = Items(
        [
            Item("A", O(100), readings={0: 1.2, 50: 1.9}),
            Item("B", RC(120), readings={0: 1.0}),
        ]
    ).surpyval_data()
    expected = SurpyvalData.from_items(
        [
            ItemLifetime("A", observed=100, markers={0: 1.2, 50: 1.9}),
            ItemLifetime("B", right_censored=120, markers={0: 1.0}),
        ]
    )
    assert data == expected
    assert data.markers == {"A": {0.0: 1.2, 50.0: 1.9}, "B": {0.0: 1.0}}


def test_readings_refusals():
    with pytest.raises(ValueError, match="mix"):
        Item("A", O(1), readings={"v": {0: 1}, 5: 2})
    with pytest.raises(ValueError, match="finite"):
        Item("A", O(1), readings={0: inf})
    with pytest.raises(ValueError, match="name every series"):
        Items(
            [
                Item("A", O(1), readings={0: 1.0}),
                Item("B", O(2), readings={"v": {0: 1.0}}),
            ]
        ).surpyval_data()
    with pytest.raises(ValueError, match="no readings"):
        Items([Item("A", O(1), readings={0: 1.0})]).event_history_data()


def test_time_varying_covariates_pass_through_unexpanded():
    a = StepSchedule.from_changepoints([0, 50], [1.0, 2.0])
    b = StepSchedule.from_changepoints([0], [1.0])
    data = Items(
        [
            Item("A", O(100), time_varying_covariates=a),
            Item("B", RC(120), time_varying_covariates=b),
        ]
    ).surpyval_data()
    expected = SurpyvalData.from_items(
        [
            ItemLifetime("A", observed=100, time_varying_covariates=a),
            ItemLifetime("B", right_censored=120, time_varying_covariates=b),
        ]
    )
    assert data == expected
    assert len(data.x) == 2  # one row per unit: the fit expands them
    assert data.time_varying_covariates["A"] == a
    with pytest.raises(ValueError, match="some items and not others"):
        Items(
            [Item("A", O(100), time_varying_covariates=a), Item("B", RC(120))]
        ).surpyval_data()


def test_a_fit_of_time_varying_covariates_is_the_records_fit():
    from surpyval.univariate.regression import WeibullPH

    rows = [
        (30, 0, [0, 20], [0.0, 1.0]),
        (45, 0, [0], [1.0]),
        (60, 1, [0, 10], [0.0, 1.0]),
        (25, 0, [0], [0.0]),
        (50, 0, [0, 40], [1.0, 0.0]),
    ]
    items, records = [], []
    for k, (t, c, times, values) in enumerate(rows):
        s = StepSchedule.from_changepoints(times, values)
        ev = RC(t) if c else O(t)
        items.append(Item(k, ev, time_varying_covariates=s))
        records.append(
            ItemLifetime(
                k,
                **({"right_censored": t} if c else {"observed": t}),
                time_varying_covariates=s,
            )
        )
    np.testing.assert_allclose(
        WeibullPH.fit(Items(items)).params,
        WeibullPH.fit(SurpyvalData.from_items(records)).params,
        rtol=1e-10,
    )


# --- 4. Text -------------------------------------------------------------


def test_a_history_line_reads_as_the_item():
    assert Items.parse("P1: 120, 310, [400, 450], 1000+ | (0, inf]") == Items(
        [_pump()]
    )
    assert Items.parse("A: 2*<50, 3*120, 2*[200, 300]") == Items(
        [Item("A", 2 * LC(50), 3 * O(120), 2 * IC(200, 300))]
    )


@pytest.mark.parametrize(
    "text",
    [
        "P1: 120, 310, [400, 450], 1000+ | (0, inf]",
        "P1: 120\nP2: 3*200, 900 | (100, 2000]",
        "unit 7: 1.5, 2.25e3+",
    ],
)
def test_items_text_round_trips(text):
    items = Items.parse(text)
    assert Items.parse(items.text()) == items


def test_text_leaves_out_what_stays_in_python():
    item = Item(
        "A",
        O(10, event_type="x"),
        O(20, terminal=True),
        covariates={"t": 1},
        readings={0: 1.0},
    )
    assert item.text() == "A: 10, 20"


def test_history_text_refusals():
    with pytest.raises(ValueError, match="colon"):
        Items.parse("120, 310")
    with pytest.raises(ValueError, match="one window"):
        Items.parse("P1: 120 | (0, inf], 310 | (0, inf]")
    with pytest.raises(ValueError, match="Can't read"):
        Items.parse("P1: 120, ?")
    with pytest.raises(ValueError, match="rename"):
        Item("a:b", O(1)).text()
