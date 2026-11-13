"""The joint models take their readings as a ``Measurements`` matched to
the life rows by unit: ``fit(x, c, ..., i=ids, markers=m)`` (or ``m=m``),
``fit(data, markers=m)`` for a SurpyvalData without readings, and
``fit_from_df(df, ..., i_col=..., m=m)``. Each fit is the one from the
old per-row marker list exactly. The fixture is the conformance suite's
``joint_data`` (40 items, a 0/1 covariate, some counted twice), which fits
in about a second.
"""

import numpy as np
import pandas as pd
import pytest

from surpyval import (
    JointPH,
    Measurements,
    SurpyvalData,
    Weibull,
    WeibullJointPH,
)
from surpyval.tests.conformance.registry_fixtures import joint_data

IDS = [f"pump-{k}" for k in range(40)]


@pytest.fixture(scope="module")
def d():
    return joint_data()


@pytest.fixture(scope="module")
def ref(d):
    return WeibullJointPH.fit(
        d["x"], c=d["c"], n=d["n"], Z=d["Z"], markers=d["markers"]
    )


def _readings(d, order=None):
    order = range(len(IDS)) if order is None else order
    return Measurements({IDS[k]: dict(d["markers"][k]) for k in order})


def _same(a, b):
    """Identical parameters, covariance, dictionary (which holds no data)
    and a dynamic prediction."""
    np.testing.assert_array_equal(a.params, b.params)
    np.testing.assert_array_equal(a.covariance(), b.covariance())
    assert a.to_dict() == b.to_dict()
    item = {0.0: 3.0, 60.0: 3.2, 120.0: 3.5}
    t = np.array([200.0, 400.0])
    np.testing.assert_array_equal(
        a.predict([1.0], item, given=120.0).sf(t),
        b.predict([1.0], item, given=120.0).sf(t),
    )


def test_arrays_with_measurements_equal_the_marker_list(d, ref):
    got = WeibullJointPH.fit(
        d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS, markers=_readings(d)
    )
    _same(ref, got)


def test_m_is_the_other_name_of_markers(d, ref):
    got = WeibullJointPH.fit(
        d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS, m=_readings(d)
    )
    _same(ref, got)
    with pytest.raises(TypeError, match="give one"):
        WeibullJointPH.fit(d["x"], i=IDS, m=_readings(d), markers=_readings(d))


def test_units_are_matched_by_id_not_position(d, ref):
    shuffled = np.random.default_rng(0).permutation(len(IDS))
    got = WeibullJointPH.fit(
        d["x"],
        c=d["c"],
        n=d["n"],
        Z=d["Z"],
        i=np.array(IDS),
        markers=_readings(d, shuffled),
    )
    _same(ref, got)


def test_a_surpyval_data_with_measurements_beside_it(d, ref):
    data = SurpyvalData(d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS)
    _same(ref, WeibullJointPH.fit(data, markers=_readings(d)))
    _same(ref, WeibullJointPH.fit(data, m=_readings(d)))
    # the same as the data holding them
    held = SurpyvalData(
        d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS, m=_readings(d)
    )
    _same(ref, WeibullJointPH.fit(held))
    with pytest.raises(TypeError, match=r"leave out 'm'"):
        WeibullJointPH.fit(held, m=_readings(d))
    with pytest.raises(TypeError, match=r"leave out 'markers'"):
        WeibullJointPH.fit(held, markers=_readings(d))


def test_a_named_measurement_and_several(d, ref):
    named = Measurements(
        {IDS[k]: dict(d["markers"][k]) for k in range(40)}, name="vibration"
    )
    _same(
        ref,
        WeibullJointPH.fit(
            d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS, m=named
        ),
    )
    several = Measurements(
        {
            "vibration": {IDS[k]: dict(d["markers"][k]) for k in range(40)},
            "temperature": {IDS[0]: {0.0: 60.0}},
        }
    )
    with pytest.raises(
        ValueError, match=r"vibration, temperature.*m.with_measurement"
    ):
        WeibullJointPH.fit(d["x"], i=IDS, m=several)
    data = SurpyvalData(d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS)
    with pytest.raises(ValueError, match=r"m.with_measurement"):
        WeibullJointPH.fit(data, m=several)
    _same(
        ref,
        WeibullJointPH.fit(
            d["x"],
            c=d["c"],
            n=d["n"],
            Z=d["Z"],
            i=IDS,
            m=several.with_measurement("vibration"),
        ),
    )


def test_units_missing_on_either_side(d):
    # An item with no readings: as {} in the list (and as a SurpyvalData
    # gives it), silently.
    markers = [dict(r) for r in d["markers"]]
    markers[3] = {}
    old = WeibullJointPH.fit(
        d["x"], c=d["c"], n=d["n"], Z=d["Z"], markers=markers
    )
    readings = Measurements(
        {IDS[k]: dict(d["markers"][k]) for k in range(40) if k != 3}
    )
    got = WeibullJointPH.fit(
        d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS, m=readings
    )
    _same(old, got)
    # A unit with readings but no row: warned of, naming it, and not used.
    extra = Measurements(
        {
            **{IDS[k]: dict(d["markers"][k]) for k in range(40)},
            "spare": {0.0: 1.0},
        }
    )
    with pytest.warns(UserWarning, match=r"1 unit in m has no life rows"):
        got = WeibullJointPH.fit(
            d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS, m=extra
        )
    np.testing.assert_array_equal(
        got.params,
        WeibullJointPH.fit(
            d["x"], c=d["c"], n=d["n"], Z=d["Z"], markers=d["markers"]
        ).params,
    )


def test_refusals(d):
    with pytest.raises(TypeError, match=r"never by position: give .* i="):
        WeibullJointPH.fit(d["x"], c=d["c"], m=_readings(d))
    with pytest.raises(TypeError, match="leave i out"):
        WeibullJointPH.fit(d["x"], c=d["c"], i=IDS, markers=d["markers"])
    with pytest.raises(ValueError, match="one unit per row"):
        WeibullJointPH.fit(d["x"], i=IDS[:5], m=_readings(d))
    with pytest.raises(ValueError, match="several rows in i"):
        WeibullJointPH.fit(d["x"], i=["a"] * 40, m=_readings(d))
    # a SurpyvalData without ids cannot be matched by unit
    with pytest.raises(ValueError, match="give the units as i="):
        WeibullJointPH.fit(SurpyvalData(d["x"], c=d["c"]), m=_readings(d))


def test_fit_from_df_with_a_readings_frame(d, ref):
    life = pd.DataFrame(
        {
            "unit": IDS,
            "age": d["x"],
            "running": d["c"],
            "count": d["n"],
            "site": np.asarray(d["Z"])[:, 0],
        }
    )
    long = pd.DataFrame(
        [
            {"unit": IDS[k], "day": t, "level": v}
            for k in range(40)
            for t, v in d["markers"][k].items()
        ]
    )
    m = Measurements.from_dataframe(
        long, i_col="unit", x_col="day", m_col="level"
    )
    got = WeibullJointPH.fit_from_df(
        life,
        x_col="age",
        c_col="running",
        n_col="count",
        Z_cols=["site"],
        i_col="unit",
        m=m,
    )
    np.testing.assert_array_equal(got.params, ref.params)
    np.testing.assert_array_equal(got.covariance(), ref.covariance())
    # the old column of {age: value} per item is unchanged
    life["readings"] = list(d["markers"])
    old = WeibullJointPH.fit_from_df(
        life,
        x_col="age",
        c_col="running",
        n_col="count",
        Z_cols=["site"],
        markers_col="readings",
    )
    np.testing.assert_array_equal(old.params, ref.params)
    with pytest.raises(TypeError, match="give one"):
        WeibullJointPH.fit_from_df(
            life, x_col="age", markers_col="readings", i_col="unit", m=m
        )


def test_other_joint_fitters_take_measurements(d):
    fitter = JointPH(Weibull, link="slope")
    old = fitter.fit(
        d["x"], c=d["c"], n=d["n"], Z=d["Z"], markers=d["markers"]
    )
    new = fitter.fit(
        d["x"], c=d["c"], n=d["n"], Z=d["Z"], i=IDS, m=_readings(d)
    )
    np.testing.assert_array_equal(old.params, new.params)
