"""Measurements in the data model: ``Measurements`` (readings taken on
units over their lives, on their own timeline) and ``SurpyvalData``'s
``m``, the same store as its ``markers``."""

import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import ItemLifetime, Measurements, SurpyvalData


def _named():
    return {
        "vibration": {"A": {0: 2.1, 30: 2.6}, "B": {0: 1.9, 15: 2.0}},
        "temperature": {"A": {10: 61.0}, "C": {5: 58.0, 25: 60.5}},
    }


def _data(**kwargs):
    return SurpyvalData(
        [40, 55, 70], c=[0, 1, 0], i=["A", "B", "C"], m=_named(), **kwargs
    )


def _items():
    return [
        ItemLifetime("A", observed=50, markers={20: 2.0, 0: 1.0}),
        ItemLifetime(
            "B",
            right_censored=80,
            left_truncation=10,
            markers={10: 1.5, 40: 2.5, 70: 3.0},
        ),
        ItemLifetime("C", right_censored=90),
    ]


# -- construction ------------------------------------------------------------


def test_one_measurement_from_units_readings():
    m = Measurements({"A": {30: 2.6, 0: 2.1}, "B": {}})
    assert m.names == [None]
    assert m.ids == ["A", "B"]
    assert (m.n_units, m.n_readings) == (2, 2)
    # In order of time, as floats
    assert m[None] == {"A": {0.0: 2.1, 30.0: 2.6}, "B": {}}
    assert repr(m) == "Measurements: (unnamed) - 2 units, 2 readings"


def test_one_measurement_named():
    m = Measurements({1: {0: 2.0}}, name="wear")
    assert m.names == ["wear"]
    assert m["wear"] == {1: {0.0: 2.0}}


def test_several_named_measurements():
    m = Measurements(_named())
    assert m.names == ["vibration", "temperature"]
    assert m.ids == ["A", "B", "C"]
    assert (m.n_units, m.n_readings) == (3, 7)
    assert m.for_unit("A") == {
        "vibration": {0.0: 2.1, 30.0: 2.6},
        "temperature": {10.0: 61.0},
    }
    assert repr(m) == (
        "Measurements: vibration, temperature - 3 units, 7 readings"
    )
    with pytest.raises(KeyError, match="'pressure'.*vibration"):
        m["pressure"]
    with pytest.raises(KeyError, match="'Z'"):
        m.for_unit("Z")


def test_the_two_shapes_are_told_apart_or_refused():
    with pytest.raises(ValueError, match="hold no readings"):
        Measurements({"A": {}, "B": {}})
    with pytest.raises(ValueError, match="mix the two shapes"):
        Measurements({"A": {0: 1.0}, "vibration": {"B": {0: 1.0}}})
    with pytest.raises(ValueError, match="already\\s+named"):
        Measurements(_named(), name="x")
    with pytest.raises(ValueError, match="takes a dict"):
        Measurements([1, 2])


def test_units_labels():
    m = Measurements(_named(), units={"vibration": "mm/s"})
    assert m.units["temperature"] is None
    assert m.units["vibration"] == "mm/s"
    assert "vibration (mm/s)" in repr(m)
    one = Measurements({"A": {0: 1.0}}, units="mm")
    assert one.units == {None: "mm"}
    with pytest.raises(ValueError, match="one label per measurement"):
        Measurements(_named(), units="mm")
    with pytest.raises(ValueError, match="'pressure'"):
        Measurements(_named(), units={"pressure": "bar"})


@pytest.mark.parametrize(
    "readings, match",
    [
        ({"A": {np.inf: 1.0}}, "Unit 'A'.*finite"),
        ({"A": {"10": 1.0}}, "Unit 'A'.*must be a number"),
        ({"A": {0: np.nan}}, "Unit 'A'.*at 0 must be one finite"),
        ({"A": {0: "high"}}, "Unit 'A'.*one finite number"),
        ({"A": {0: [1, 2]}}, "Unit 'A'.*one finite number"),
        ({"A": [1, 2]}, "a dict where a dict was expected|got list"),
        ({None: {0: 1.0}}, "missing"),
        ({"A": {0: 1.0}, 1: {0: 1.0}}, "one kind"),
        (
            {"vibration": {"A": {0: np.nan}}},
            "Measurement 'vibration', unit 'A'",
        ),
    ],
)
def test_bad_readings_are_refused_naming_where(readings, match):
    with pytest.raises(ValueError, match=match):
        Measurements(readings)


def test_negative_times_are_allowed():
    # The life data's clock may start below 0 (left truncation below 0)
    assert Measurements({"A": {-5: 1.0}})[None] == {"A": {-5.0: 1.0}}


# -- DataFrames ---------------------------------------------------------------


def _frame():
    return pd.DataFrame(
        {
            "unit": ["A", "A", "B", "A", "C", "C"],
            "hours": [0, 30, 0, 10, 5, 25],
            "reading": [2.1, 2.6, 1.9, 61.0, 58.0, 60.5],
            "what": ["vibration"] * 3 + ["temperature"] * 3,
        }
    )


def test_from_dataframe_with_several_names():
    m = Measurements.from_dataframe(
        _frame(),
        i_col="unit",
        x_col="hours",
        m_col="reading",
        m_name_col="what",
        units={"vibration": "mm/s"},
    )
    assert m.names == ["vibration", "temperature"]
    assert m["temperature"] == {
        "A": {10.0: 61.0},
        "C": {5.0: 58.0, 25.0: 60.5},
    }
    df = m.to_dataframe()
    assert list(df.columns) == ["i", "x", "value", "name"]
    again = Measurements.from_dataframe(
        df,
        i_col="i",
        x_col="x",
        m_col="value",
        m_name_col="name",
        units=m.units,
    )
    assert again == m


def test_from_dataframe_of_one_measurement_round_trips():
    m = Measurements({1: {0: 2.0, 5: 2.5}, 2: {3: 1.0}})
    df = m.to_dataframe()
    assert list(df.columns) == ["i", "x", "value"]
    again = Measurements.from_dataframe(
        df, i_col="i", x_col="x", m_col="value"
    )
    assert again == m
    assert again.ids == [1, 2] and isinstance(again.ids[0], int)


def test_from_dataframe_refusals_name_the_rows():
    df = _frame()
    df.loc[2, "reading"] = np.nan
    with pytest.raises(ValueError, match="m_col='reading'.*rows 2"):
        Measurements.from_dataframe(
            df, i_col="unit", x_col="hours", m_col="reading"
        )
    df = _frame()
    df.loc[3, "what"] = "vibration"
    df.loc[3, "hours"] = 30
    with pytest.raises(ValueError, match="rows 1 and 3"):
        Measurements.from_dataframe(
            df,
            i_col="unit",
            x_col="hours",
            m_col="reading",
            m_name_col="what",
        )
    with pytest.raises(ValueError, match="x_col='when' is not a column"):
        Measurements.from_dataframe(
            _frame(), i_col="unit", x_col="when", m_col="reading"
        )
    with pytest.raises(TypeError):
        Measurements.from_dataframe(  # no censoring or counts on readings
            _frame(),
            i_col="unit",
            x_col="hours",
            m_col="reading",
            c_col="c",
        )


# -- dict and JSON -----------------------------------------------------------


def test_dict_and_json_keep_id_and_name_types():
    m = Measurements(
        {1: {("A", 2): {0: 1.0}}, 2: {("B", 1): {3: 2.0}}},
        units={1: "mm"},
    )
    text = json.dumps(m.to_dict(), allow_nan=False)
    again = Measurements.from_dict(json.loads(text))
    assert again == m
    assert again.names == [1, 2]
    assert again.ids == [("A", 2), ("B", 1)]
    assert sp.from_dict(json.loads(text)) == m


def test_from_dict_refuses_other_dictionaries():
    with pytest.raises(ValueError, match="serialised Measurements"):
        Measurements.from_dict({"x": [1]})


def test_equality():
    assert Measurements(_named()) == Measurements(_named())
    assert Measurements(_named()) != Measurements({"A": {0: 2.1}})
    assert Measurements(_named()) != Measurements(
        _named(), units={"vibration": "mm/s"}
    )


@pytest.mark.parametrize("typed", ["mm/s", "°C", "bar", "h"])
def test_unit_labels_print_and_round_trip_as_typed(typed):
    # Decided 13 Nov 2026: never pint's "millimeter / second"
    m = Measurements(_named(), units={"vibration": typed})
    assert m.units["vibration"] == typed
    assert f"vibration ({typed})" in repr(m)
    saved = m.to_dict()
    assert saved["measurements"][0]["unit"] == typed
    back = Measurements.from_dict(saved)
    assert back.units["vibration"] == typed and back == m
    data = SurpyvalData([40, 55, 70], c=[0, 1, 0], i=["A", "B", "C"], m=m)
    assert f"vibration ({typed})" in data.summary()
    assert SurpyvalData.from_json(data.to_json()).m.units["vibration"] == (
        typed
    )
    assert SurpyvalData.from_dict(data.to_dict()).m.units["vibration"] == (
        typed
    )
    again = Measurements.from_dataframe(
        m.to_dataframe(),
        i_col="i",
        x_col="x",
        m_col="value",
        m_name_col="name",
        units=m.units,
    )
    assert again.units["vibration"] == typed


def test_a_file_with_pints_unit_name_still_loads():
    saved = Measurements(_named(), units={"vibration": "mm/s"}).to_dict()
    saved["measurements"][0]["unit"] = "millimeter / second"
    old = Measurements.from_dict(saved)
    assert old.units["vibration"] == "millimeter / second"
    # The same unit, read with pint
    assert old == Measurements(_named(), units={"vibration": "mm/s"})
    assert old != Measurements(_named(), units={"vibration": "m/s"})


def test_a_unit_label_pint_does_not_know_is_kept():
    m = Measurements(_named(), units={"vibration": "counts per sweep"})
    assert m.units["vibration"] == "counts per sweep"
    with pytest.raises(TypeError, match="unit must be text"):
        Measurements(_named(), units={"vibration": 5})
    with pytest.raises(ValueError, match="leave it out"):
        Measurements(_named(), units={"vibration": "  "})


# -- attached to life data ----------------------------------------------------


@pytest.mark.parametrize("given", [_named, lambda: Measurements(_named())])
def test_attached_as_m(given):
    data = SurpyvalData(
        [40, 55, 70], c=[0, 1, 0], i=["A", "B", "C"], m=given()
    )
    assert data.m == Measurements(_named())


def test_m_needs_unit_ids():
    with pytest.raises(ValueError, match="give the units as i="):
        SurpyvalData([40, 55], m=_named())


def test_units_without_life_rows_are_warned_of_and_kept():
    readings = {f"U{k}": {0: 1.0} for k in range(7)}
    with pytest.warns(
        UserWarning, match="6 units in m have no life rows"
    ) as w:
        data = SurpyvalData([5.0], i=["U0"], m=readings)
    assert "'U1', 'U2', 'U3', 'U4', 'U5' and 1 more" in str(w[0].message)
    assert data.m.n_units == 7


def test_from_dataframe_takes_m():
    df = pd.DataFrame({"unit": ["A", "B", "C"], "hours": [40, 55, 70]})
    data = SurpyvalData.from_dataframe(
        df, i_col="unit", x_col="hours", m=_named()
    )
    assert data.m == Measurements(_named())
    with pytest.raises(ValueError, match="i="):
        SurpyvalData.from_dataframe(df, x_col="hours", m=_named())


# -- one store, two names -----------------------------------------------------


def test_m_and_markers_are_one_store():
    data = SurpyvalData([40, 55], i=["A", "B"], m={"A": {0: 1.0}})
    assert data.markers == {"A": {0.0: 1.0}}
    data.markers = {"A": {}, "B": {5.0: 2.0}}
    assert data.m == Measurements({"A": {}, "B": {5: 2.0}})
    data.m = {"B": {1: 3.0}}
    assert data.markers == {"B": {1.0: 3.0}}
    data.markers = None
    assert data.m is None
    # One named measurement is the markers too
    data.m = Measurements({"A": {0: 1.0}}, name="wear")
    assert data.markers == {"A": {0.0: 1.0}}


def test_markers_of_several_measurements_are_refused_naming_m():
    with pytest.raises(ValueError, match=r"data\.m\[name\].*with_measurement"):
        _data().markers


def test_markers_from_items_are_unchanged():
    data = SurpyvalData.from_items(_items())
    assert data.markers == {
        "A": {0.0: 1.0, 20.0: 2.0},
        "B": {10.0: 1.5, 40.0: 2.5, 70.0: 3.0},
        "C": {},
    }
    assert data.m._plain
    assert data._markers_by_row() == list(data.markers.values())
    assert "markers" in data.to_dict() and "m" not in data.to_dict()
    assert "schema" not in data.to_dict()
    assert data.to_items()[1].markers == {10.0: 1.5, 40.0: 2.5, 70.0: 3.0}
    assert "markers: 5 readings on 2 items" in data.summary()
    assert "markers" in repr(data)
    again = SurpyvalData.from_dict(data.to_dict())
    assert again.markers == data.markers


def test_today_markers_byte_identical_whichever_way_they_are_set():
    old = SurpyvalData.from_items(_items())
    new = SurpyvalData(old.x, old.c, old.n, t=old.t, i=old.ids, m=old.markers)
    assert repr(new) == repr(old)
    assert new.summary() == old.summary()
    assert new.to_dict() == old.to_dict()
    assert new._markers_by_row() == old._markers_by_row()


def test_old_pickles_with_markers_load():
    data = SurpyvalData([40, 55], i=["A", "B"])
    state = dict(data.__dict__)
    state["markers"] = {"A": {0.0: 1.0}, "B": {}}
    old = SurpyvalData.__new__(SurpyvalData)
    old.__setstate__(state)
    assert old.markers == {"A": {0.0: 1.0}, "B": {}}
    again = pickle.loads(pickle.dumps(_data()))
    assert again.m == _data().m


# -- named measurements through the dataset -----------------------------------


def test_repr_summary_and_dict_of_named_measurements():
    data = _data()
    assert "  m: vibration, temperature - 3 units, 7 readings" in (
        data.summary()
    )
    assert " m" in repr(data).splitlines()[1]
    d = data.to_dict()
    assert "markers" not in d and d["schema"] == 4
    again = SurpyvalData.from_json(json.dumps(d))
    assert again.m == data.m and again.ids == data.ids


def test_dict_of_one_measurement_with_a_unit_label_is_under_m():
    data = SurpyvalData(
        [40], i=["A"], m=Measurements({"A": {0: 1.0}}, units="mm")
    )
    assert "m" in data.to_dict() and "markers" not in data.to_dict()
    assert SurpyvalData.from_dict(data.to_dict()).m == data.m


def test_subsets_keep_the_readings_of_the_units_kept():
    data = _data()
    assert data[[0, 2]].m.ids == ["A", "C"]
    assert data[[0, 2]].m["vibration"] == {"A": {0.0: 2.1, 30.0: 2.6}}
    kept = data.select(ids=["B"])
    assert kept.m["vibration"] == {"B": {0.0: 1.9, 15.0: 2.0}}
    assert data.drop(["A"]).m.ids == ["B", "C"]


def test_groupby_keeps_each_groups_readings():
    data = SurpyvalData(
        [40, 55, 70],
        i=["A", "B", "C"],
        Z=[1.0, 0.0, 1.0],
        covariate_names=["z"],
        m=_named(),
    )
    parts = data.groupby("z")
    assert parts[1.0].m.ids == ["A", "C"]
    assert parts[0.0].m.ids == ["B"]


def test_reshapes_trim_only_a_moved_units_readings():
    data = _data()
    cut = data.censor_at(20)
    assert cut.m.for_unit("A") == {
        "vibration": {0.0: 2.1},
        "temperature": {10.0: 61.0},
    }
    entered = data.enter_at(8)
    assert entered.m["vibration"]["B"] == {15.0: 2.0}
    # A unit whose observation did not move keeps every reading, even one
    # outside it (the data was given that way)
    outside = SurpyvalData(
        [10, 20],
        i=["A", "B"],
        m={"vibration": {"A": {12: 1.0}, "B": {1: 1.0}}, "t": {"B": {2: 1.0}}},
    )
    assert outside.censor_at(15).m["vibration"]["A"] == {12.0: 1.0}


def test_without_drops_the_measurements_under_each_name():
    data = _data()
    for part in ("m", "measurements", "markers"):
        out = data.without(part)
        assert out.m is None and out.ids == data.ids
    plain = SurpyvalData.from_items(_items())
    assert plain.without("m").markers is None
    assert plain.without("measurements").markers is None
    with_z = SurpyvalData(
        [40, 55, 70], i=["A", "B", "C"], Z=[1.0, 0.0, 1.0], m=_named()
    )
    out = with_z.without("covariates", "m")
    assert out.Z is None and out.m is None
    assert with_z.without("covariates").m == with_z.m


def test_records_of_named_measurements_are_refused():
    data = _data()
    for call in (
        data.to_items,
        lambda: data.item("A"),
        lambda: list(data.records()),
    ):
        with pytest.raises(ValueError, match=r"data\.m"):
            call()
    assert len(data.without("m").to_items()) == 3


def test_update_refuses_records_with_markers_into_named_data():
    data = _data()
    with pytest.raises(ValueError, match="carry markers"):
        data.update([ItemLifetime("D", observed=5, markers={0: 1.0})])
    assert data.update([ItemLifetime("D", observed=5)]).m.ids == list("ABC")


def test_with_measurement():
    data = _data()
    one = data.with_measurement("temperature")
    assert one.m.names == ["temperature"]
    assert one.markers == {"A": {10.0: 61.0}, "C": {5.0: 58.0, 25.0: 60.5}}
    assert data.m.names == ["vibration", "temperature"]
    with pytest.raises(KeyError, match="'pressure'"):
        data.with_measurement("pressure")
    with pytest.raises(ValueError, match="no measurements"):
        SurpyvalData([1, 2]).with_measurement("x")


# -- fits ---------------------------------------------------------------------


def test_fits_without_markers_refuse_measurements_as_before():
    with pytest.raises(ValueError, match='without\\("markers"\\)'):
        sp.Weibull.fit(_data())
    sp.Weibull.fit(_data().without("m"))


def _fleet():
    from surpyval.tests.univariate.joint._fleet import fleet

    return fleet(60, seed=8)


def test_a_joint_fit_of_m_equals_the_fit_of_markers():
    old = _fleet()
    new = SurpyvalData(
        old.x,
        old.c,
        old.n,
        t=old.t,
        Z=old.Z,
        covariate_names=old.covariate_names,
        i=old.ids,
        m=Measurements(old.markers, name="wear"),
    )
    a = sp.WeibullJointPH.fit(old)
    b = sp.WeibullJointPH.fit(new)
    assert np.array_equal(a.params, b.params)


def test_a_joint_fit_refuses_several_measurements_until_one_is_chosen():
    old = _fleet()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = SurpyvalData(
            old.x,
            old.c,
            old.n,
            t=old.t,
            Z=old.Z,
            covariate_names=old.covariate_names,
            i=old.ids,
            m={"wear": old.markers, "temperature": {old.ids[0]: {0: 20.0}}},
        )
    with pytest.raises(ValueError, match="with_measurement\\(name\\)"):
        sp.WeibullJointPH.fit(data)
    a = sp.WeibullJointPH.fit(data.with_measurement("wear"))
    b = sp.WeibullJointPH.fit(old)
    assert np.array_equal(a.params, b.params)


def test_extend_running_and_update_of_data_with_markers():
    # Data with readings goes through the records; these two used to take
    # its records for event histories
    data = SurpyvalData.from_items(_items())
    assert data.extend_running(100).x.tolist() == [50.0, 100.0, 100.0]
    assert data.extend_running(100).markers == data.markers
    updated = data.update([ItemLifetime("D", observed=5)])
    assert updated.ids == ["A", "B", "C", "D"]
    named = _data().extend_running(100)
    assert named.m == _data().m


# -- several rows per unit (a) and rows left out (x with xl, xr) -------------


def _repairs(m):
    # Unit A repaired at 2 and 5, scrapped at 9; B still running at 4
    return SurpyvalData(
        x=[2, 5, 9, 4], c=[0, 0, 0, 1], i=["A", "A", "A", "B"], m=m
    )


def _readings():
    return {
        "vib": {"A": {0: 1.0, 3: 2.0}, "B": {1: 1.5}},
        "temp": {"A": {2: 50.0}},
    }


def test_m_is_per_unit_when_a_unit_has_several_rows():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = _repairs(_readings())
    assert data.a.tolist() == [0, 0, 1, 0]
    assert data.m.ids == ["A", "B"]
    # The unit's count shows on its first row only
    rows = repr(data).splitlines()[2:]
    assert rows[0].endswith("3 readings") and rows[1].endswith("  ")
    assert rows[3].endswith("1 reading")
    assert "  m: vib, temp - 2 units, 4 readings" in data.summary()
    plain = _repairs({"A": {0: 1.0}, "B": {1: 2.0}})
    rows = repr(plain).splitlines()[2:]
    assert rows[0].endswith("1 reading") and not rows[1].endswith("reading")
    assert "markers: 2 readings on 2 items" in plain.summary()


def test_a_unit_keeps_its_readings_while_any_of_its_rows_is_kept():
    data = _repairs(_readings())
    assert data[[1]].m.ids == ["A"]
    assert data[[1]].m["vib"] == {"A": {0.0: 1.0, 3.0: 2.0}}
    assert data[[3]].m.ids == ["B"]
    assert data.select(ids=["A"]).m.ids == ["A"]
    # A unit's window is the span of its rows: cut at 4, A keeps 0 to 4
    cut = data.censor_at(4)
    assert cut.m.for_unit("A") == {
        "vib": {0.0: 1.0, 3.0: 2.0},
        "temp": {2.0: 50.0},
    }
    assert data.censor_at(2.5).m.for_unit("A") == {
        "vib": {0.0: 1.0},
        "temp": {2.0: 50.0},
    }


def test_dict_with_a_and_m_round_trips_and_is_schema_4():
    data = _repairs(_readings())
    d = json.loads(data.to_json())
    assert d["schema"] == 4 and "a" in d and "m" in d
    again = SurpyvalData.from_dict(d)
    assert again.m == data.m
    assert again.a.tolist() == data.a.tolist()
    assert again.a_inferred.tolist() == data.a_inferred.tolist()
    plain = _repairs({"A": {0: 1.0}, "B": {1: 2.0}})
    d = plain.to_dict()
    assert d["schema"] == 4 and "markers" in d
    assert SurpyvalData.from_dict(d).markers == plain.markers


def test_no_fit_is_handed_a_units_readings_once_per_row():
    plain = _repairs({"A": {0: 1.0}, "B": {1: 2.0}})
    with pytest.raises(ValueError, match="one row per unit"):
        sp.WeibullJointPH.fit(plain)
    from surpyval.utils.surpyval_data import _markers_taken

    # Even a fit that would take the terminal flags (and so passes the
    # one-row-per-unit check) is not handed the readings row by row
    with pytest.raises(ValueError, match="one row per unit"):
        _markers_taken(plain, {"x", "c", "markers", "a"})


def test_a_unit_whose_only_row_was_left_out_is_warned_of_once():
    with pytest.warns(UserWarning) as caught:
        data = SurpyvalData(
            x=[3, None],
            xl=[1, 2],
            xr=[4, 6],
            i=["A", "B"],
            m={"A": {0: 1.0}, "B": {1: 1.0}},
        )
    messages = [str(w.message) for w in caught]
    assert "left out" in messages[0]
    assert messages[1].startswith("1 unit in m has no life rows: 'A'. Its")
    assert data.ids == ["B"] and data.m.ids == ["A", "B"]


# -- reshapes of units with several rows and measurements -------------------


def _stays(states, named=False):
    readings = {1: {1.0: 3.0}, 2: {1.0: 2.0, 2.5: 2.2}}
    if named:
        readings = {"vib": readings}
    kwargs = dict(x=[2, 6, 3, 8, 4], tl=[0, 2, 0, 3, 0], i=[1, 1, 2, 2, 3])
    if states:
        kwargs.update(
            e=["ill", "dead", "ill", None, "dead"],
            s=["well", "ill", "well", "ill", "well"],
        )
    else:
        kwargs.update(c=[0, 0, 0, 1, 0])
    return SurpyvalData(m=readings, **kwargs)


@pytest.mark.parametrize("named", [False, True])
@pytest.mark.parametrize("states", [True, False])
def test_reshapes_of_units_with_several_rows_keep_m_by_unit(states, named):
    data = _stays(states, named)
    one = (lambda m: m["vib"]) if named else (lambda m: m[None])
    for part in ("m", "markers", "measurements"):
        out = data.without(part)
        assert out.m is None and out.ids == data.ids
        assert out.x.tolist() == data.x.tolist()
    assert one(data.select(ids=[2]).m) == {2: {1.0: 2.0, 2.5: 2.2}}
    # Cut at 2.2: unit 2 keeps the reading at 1, not the one at 2.5
    cut = data.censor_at(2.2)
    assert one(cut.m)[2] == {1.0: 2.0}
    assert one(cut.m)[1] == {1.0: 3.0}
    entered = data.enter_at(1.5)
    assert one(entered.m)[2] == {2.5: 2.2}
    assert one(data.extend_running(20).m) == one(data.m)
    if states:
        dropped = data.without("states")
    else:
        dropped = data.without("truncation")
    assert one(dropped.m) == one(data.m)
