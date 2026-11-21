"""Operations on whole lifetime datasets: ``SurpyvalData.concat`` and
``+``, equality (``==``, ``equals``, on ``EventHistoryData`` too), the
DataFrame round trip with no column arguments, ``rescale_time`` and
``resample``; each against expectations built by hand."""

import numpy as np
import pandas as pd
import pytest

from surpyval import (
    EventHistoryData,
    ItemLifetime,
    StepSchedule,
    SurpyvalData,
    Weibull,
)

inf = np.inf


# Combining ------------------------------------------------------------


def test_concat_groups_and_sorts_as_the_constructor_does():
    a = SurpyvalData([10, 20, 30], c=[0, 0, 1], unit="hours")
    b = SurpyvalData([20, 25], unit="h")
    both = SurpyvalData.concat([a, b])
    expected = SurpyvalData(
        [10, 20, 30, 20, 25], c=[0, 0, 1, 0, 0], unit="hours"
    )
    assert both == expected
    np.testing.assert_array_equal(both.x, [10, 20, 25, 30])
    np.testing.assert_array_equal(both.n, [1, 2, 1, 1])
    assert (a + b) == both
    assert sum([a, b]) == both


def test_concat_carries_intervals_truncation_and_weights():
    a = SurpyvalData(x=[[1, 2], 3], c=[2, 0], tl=0.5, w=[2.0, 1.0])
    b = SurpyvalData([4, 5], c=[1, 0])
    out = a + b
    # Rows sorted by the constructor; the unweighted rows weigh 1
    rows = {
        (tuple(x), int(c), float(tl), float(w))
        for x, c, tl, w in zip(out.x, out.c, out.tl, out.w)
    }
    assert rows == {
        ((1.0, 2.0), 2, 0.5, 2.0),
        ((3.0, 3.0), 0, 0.5, 1.0),
        ((4.0, 4.0), 1, -inf, 1.0),
        ((5.0, 5.0), 0, -inf, 1.0),
    }


def test_concat_keeps_ids_event_types_covariates_and_attributes():
    a = SurpyvalData.from_items(
        [
            ItemLifetime(
                "A", observed=5, event_type="wear", covariates={"load": 1.0}
            ),
            ItemLifetime(
                "B",
                right_censored=7,
                covariates={"load": 2.0},
                attributes={"site": "north"},
            ),
        ]
    )
    b = SurpyvalData.from_items(
        [
            ItemLifetime(
                "C", observed=3, event_type="crack", covariates={"load": 0.5}
            )
        ],
        counts=[3],
    )
    out = a + b
    assert out.ids == ["A", "B", "C"]
    assert list(out.e) == ["wear", None, "crack"]
    assert out.covariate_names == ["load"]
    np.testing.assert_array_equal(out.Z[:, 0], [1.0, 2.0, 0.5])
    np.testing.assert_array_equal(out.n, [1, 1, 3])
    assert out.attributes == [None, {"site": "north"}, None]
    assert out.item("C") == b.item("C")


def test_concat_keeps_terminal_flags_and_measurements():
    a = SurpyvalData([2, 5, 9], i=["P1", "P1", "P1"], m={"P1": {1.0: 0.2}})
    b = SurpyvalData([4, 6], c=[0, 1], i=["Q", "Q"], a=[0, 0])
    out = a + b
    np.testing.assert_array_equal(out.a, [0, 0, 1, 0, 0])
    # Inferred where it was, given where it was given
    np.testing.assert_array_equal(
        out.a_inferred, [True, True, True, False, False]
    )
    assert out.markers == {"P1": {1.0: 0.2}}


def test_concat_refuses_a_shared_id_naming_it():
    p = SurpyvalData([5, 8], i=["P1", "P2"])
    q = SurpyvalData([6, 9], i=["P2", "P3"])
    with pytest.raises(ValueError, match=r"'P2'.*rename the ids first"):
        p + q


def test_concat_refuses_different_units_saying_how():
    a = SurpyvalData([1, 2], unit="hours")
    with pytest.raises(ValueError, match="rescale_time"):
        a + SurpyvalData([1, 2], unit="days")
    with pytest.raises(ValueError, match="no unit"):
        a + SurpyvalData([1, 2])


def test_concat_refuses_different_covariates():
    a = SurpyvalData([1, 2], Z=[[1.0], [2.0]], covariate_names=["load"])
    b = SurpyvalData([1, 2], Z=[[1.0], [2.0]], covariate_names=["temp"])
    with pytest.raises(ValueError, match=r"\['load'\].*\['temp'\]"):
        a + b
    with pytest.raises(ValueError, match="without\\('covariates'\\)"):
        a + SurpyvalData([3, 4])


def test_concat_refuses_ids_on_some_and_untyped_failures():
    with pytest.raises(ValueError, match="item ids"):
        SurpyvalData([1], i=["A"]) + SurpyvalData([2])
    typed = SurpyvalData([1, 2], e=["wear", "crack"])
    with pytest.raises(ValueError, match="event types"):
        typed + SurpyvalData([3])
    # Suspensions have no type: they combine
    out = typed + SurpyvalData([3], c=[1])
    assert list(out.e) == ["wear", "crack", None]


def test_concat_refuses_other_classes_and_nothing():
    with pytest.raises(ValueError, match="at least one"):
        SurpyvalData.concat([])
    with pytest.raises(TypeError, match="SurpyvalData"):
        SurpyvalData.concat([SurpyvalData([1]), [2, 3]])
    with pytest.raises(TypeError):
        SurpyvalData([1]) + [2]


# Equality -------------------------------------------------------------


def test_equality_compares_every_part():
    base = dict(x=[1, 2, 3], c=[0, 1, 0], i=["a", "b", "c"], unit="hours")
    data = SurpyvalData(**base)
    assert data == SurpyvalData(**base)
    assert data != SurpyvalData(**{**base, "c": [0, 0, 0]})
    assert data != SurpyvalData(**{**base, "i": ["a", "b", "d"]})
    assert data != SurpyvalData(**{**base, "unit": "days"})
    assert data != SurpyvalData(**{**base, "e": ["x", None, "y"]})
    assert data != SurpyvalData(**{**base, "w": [1, 1, 2]})
    assert data != [1, 2, 3]
    with pytest.raises(TypeError):
        hash(data)


def test_equals_takes_a_tolerance():
    a = SurpyvalData([1.0, 2.0, 3.0])
    b = SurpyvalData([1.0, 2.0, 3.0 * (1 + 1e-12)])
    assert a != b
    assert a.equals(b)
    assert not a.equals(b, rtol=0)
    assert not a.equals(SurpyvalData([1.0, 2.0, 3.1]), rtol=1e-3)
    with pytest.raises(ValueError, match="rtol"):
        a.equals(b, rtol=-1)


def test_event_history_data_equality():
    def make(ages):
        return EventHistoryData.from_histories(
            observed={"A": ages, "B": [3.0]},
            right_censored={"A": 12, "B": 8},
        )

    assert make([5.0, 9.0]) == make([5.0, 9.0])
    assert make([5.0, 9.0]) != make([5.0, 9.5])
    assert make([5.0, 9.0]).equals(make([5.0, 9.0 + 1e-12]))
    assert make([5.0, 9.0]) != SurpyvalData([5.0, 9.0])


# DataFrame round trip ---------------------------------------------------


def _round_trips():
    return {
        "plain": lambda: SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1]),
        "interval": lambda: SurpyvalData(
            x=[1, [2, 3], 4, 5], c=[0, 2, 1, -1], n=[1, 2, 1, 1]
        ),
        "truncated": lambda: SurpyvalData([2, 3, 4], tl=1, tr=[9, 9, inf]),
        "weighted": lambda: SurpyvalData([1, 2, 3], w=[0.5, 1, 2]),
        "covariates": lambda: SurpyvalData(
            [1, 2, 3],
            Z=[[1, 0], [0, 1], [1, 1]],
            covariate_names=["a", "load"],
        ),
        "unnamed covariates": lambda: SurpyvalData(
            [1, 2, 3], Z=[[1.0], [0], [1]]
        ),
        "items": lambda: SurpyvalData(
            [3, 1, 2], c=[0, 1, 0], i=["u1", "u2", "u3"], unit="hours"
        ),
        "event types": lambda: SurpyvalData(
            [3, 1, 2], c=[0, 1, 0], e=["crack", None, "wear"]
        ),
        "several rows per item": lambda: SurpyvalData(
            x=[2, 5, 9], i=["P1", "P1", "P1"]
        ),
        "attributes": lambda: SurpyvalData.from_items(
            [
                ItemLifetime("A", observed=5, attributes={"site": "x"}),
                ItemLifetime("B", right_censored=3, attributes={"site": "y"}),
            ]
        ),
        "states": lambda: SurpyvalData(
            x=[3, 5, 7, 4],
            tl=[0, 3, 0, 0],
            s=["well", "ill", "well", "well"],
            e=["ill", "dead", None, "dead"],
            i=["A", "A", "B", "C"],
        ),
    }


ROUND_TRIPS = _round_trips()


@pytest.mark.parametrize("case", sorted(ROUND_TRIPS))
def test_dataframe_round_trip_with_no_column_arguments(case):
    data = ROUND_TRIPS[case]()
    again = SurpyvalData.from_dataframe(data.to_dataframe())
    assert again == data


def test_dataframe_round_trip_through_a_file_reads_the_names(tmp_path):
    data = SurpyvalData(
        x=[1, [2, 3], 4], c=[0, 2, 1], Z=[[1.0], [2.0], [3.0]], tl=0.5
    )
    path = tmp_path / "data.csv"
    data.to_dataframe().to_csv(path, index=False)
    again = SurpyvalData.from_dataframe(pd.read_csv(path))
    assert again == data


def test_from_dataframe_refuses_columns_it_cannot_place():
    df = pd.DataFrame({"x": [1, 2], "c": [0, 1], "site": ["a", "b"]})
    with pytest.raises(ValueError, match="'site'.*attribute_cols"):
        SurpyvalData.from_dataframe(df)
    with pytest.raises(ValueError, match="no 'x'"):
        SurpyvalData.from_dataframe(pd.DataFrame({"hours": [1, 2]}))


def test_from_dataframe_attributes_stay_with_their_rows():
    # The rows used to be sorted after the attributes were read, which
    # gave each row another row's attributes
    df = pd.DataFrame({"x": [5, 1, 3], "site": ["a", "b", "c"]})
    data = SurpyvalData.from_dataframe(df, x_col="x", attribute_cols=["site"])
    pairs = {(float(x), a["site"]) for x, a in zip(data.x, data.attributes)}
    assert pairs == {(5.0, "a"), (1.0, "b"), (3.0, "c")}


# Another clock ----------------------------------------------------------


def test_rescale_time_divides_every_time():
    schedule = StepSchedule.from_changepoints([0, 48], [0.0, 1.0])
    data = SurpyvalData.from_items(
        [
            ItemLifetime("A", observed=48, left_truncation=24),
            ItemLifetime("B", interval_censored=[24, 72]),
        ],
        unit="hours",
    )
    days = data.rescale_time(24, unit="days")
    np.testing.assert_array_equal(days.x, [[2, 2], [1, 3]])
    np.testing.assert_array_equal(days.tl, [1, -inf])
    assert days.unit == "days"
    assert data.unit == "hours"  # unchanged
    assert days.ids == ["A", "B"]

    tvc = SurpyvalData([48, 96], i=["A", "B"]).rescale_time(24)
    assert tvc.ids == ["A", "B"]
    with_tvc = SurpyvalData.from_items(
        [ItemLifetime("A", observed=96, time_varying_covariates=schedule)]
    ).rescale_time(24)
    np.testing.assert_array_equal(
        with_tvc.time_varying_covariates["A"].edges, [0, 2, inf]
    )


def test_rescale_time_moves_readings_and_keeps_the_rest():
    data = SurpyvalData(
        [40, 80],
        c=[0, 1],
        i=["A", "B"],
        w=[1.0, 2.0],
        m={"A": {20.0: 1.5}, "B": {40.0: 2.5}},
    )
    out = data.rescale_time(0.5)
    np.testing.assert_array_equal(out.x, [80, 160])
    assert out.markers == {"A": {40.0: 1.5}, "B": {80.0: 2.5}}
    np.testing.assert_array_equal(out.w, data.w)
    np.testing.assert_array_equal(out.c, data.c)
    assert data.rescale_time(1) == data


def test_rescale_time_matches_the_model_method():
    data = SurpyvalData([24, 48, 96, 120], c=[0, 0, 1, 0], unit="hours")
    by_model = Weibull.fit(data).rescale_time(24, unit="days")
    by_data = Weibull.fit(data.rescale_time(24, unit="days"))
    np.testing.assert_allclose(by_data.params, by_model.params, rtol=1e-6)


@pytest.mark.parametrize("factor", [0, -2, inf, "24", True])
def test_rescale_time_refuses_a_factor_not_above_zero(factor):
    with pytest.raises(ValueError, match="finite number above 0"):
        SurpyvalData([1, 2]).rescale_time(factor)


# Bootstrap --------------------------------------------------------------


def test_resample_draws_units_by_their_counts():
    data = SurpyvalData([1, 2, 3], c=[0, 1, 0], n=[3, 1, 1])
    boot = data.resample(random_state=7)
    assert int(boot.n.sum()) == 5
    # Every row is one of the data's rows
    rows = {(float(x), int(c)) for x, c in zip(data.x, data.c)}
    assert {(float(x), int(c)) for x, c in zip(boot.x, boot.c)} <= rows
    assert boot == data.resample(random_state=7)
    assert int(data.resample(random_state=7, size=50).n.sum()) == 50


def test_resample_matches_the_draw_by_hand():
    data = SurpyvalData([1, 2, 3], c=[0, 1, 0], n=[3, 1, 1])
    drawn = np.random.default_rng(11).choice(3, size=5, p=[0.6, 0.2, 0.2])
    times = np.bincount(drawn, minlength=3)
    boot = data.resample(random_state=11)
    expected = {float(x): int(k) for x, k in zip(data.x, times) if k > 0}
    assert {float(x): int(k) for x, k in zip(boot.x, boot.n)} == expected


def test_resample_moves_an_items_rows_together():
    data = SurpyvalData(
        x=[2, 5, 9, 4, 6],
        c=[0, 0, 0, 0, 1],
        i=["P1", "P1", "P1", "Q", "Q"],
        Z=[[1.0], [1.0], [1.0], [0.0], [0.0]],
    )
    for seed in range(10):
        boot = data.resample(random_state=seed)
        counts: dict = {}
        for unit, k in zip(boot.ids, boot.n):
            counts.setdefault(unit, set()).add(int(k))
        # All of an item's rows, each counted the times it was drawn
        assert boot.ids.count("P1") in (0, 3)
        assert boot.ids.count("Q") in (0, 2)
        assert all(len(k) == 1 for k in counts.values())
        assert sum(next(iter(k)) for k in counts.values()) == 2


def test_resample_refuses_a_bad_size():
    with pytest.raises(ValueError, match="size"):
        SurpyvalData([1, 2]).resample(size=0)
