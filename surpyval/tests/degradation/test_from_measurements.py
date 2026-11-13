"""The degradation fitters take their readings as a ``Measurements``, or a
``SurpyvalData`` holding them as ``data.m``, in place of the arrays ``x``,
``y`` and ``i``; the fit is the arrays' fit exactly.

Wired: ``DegradationAnalysis``, ``WienerProcess``, ``GammaProcess`` (their
``fit``; their ``fit_from_df`` also names the readings' column ``m_col``).
Not wired: ``DestructiveDegradation`` (one reading per unit, censored
readings, no unit ids: arrays are its natural form).
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from surpyval import Measurements, SurpyvalData
from surpyval.degradation import (
    DegradationAnalysis,
    GammaProcess,
    WienerProcess,
)
from surpyval.tests.conformance.registry_fixtures import (
    path_data,
    process_data,
)
from surpyval.tests.degradation.test_degradation_analysis import adt_data

FITTERS = {
    "DegradationAnalysis": (DegradationAnalysis, path_data, 150.0),
    "WienerProcess": (WienerProcess, process_data, 100.0),
    "GammaProcess": (GammaProcess, process_data, 100.0),
}


def _store(x, y, i, order=None):
    """``{i: {x: value}}`` of the arrays, units in ``order`` (default as
    first seen)."""
    store: dict = {}
    for t, v, k in zip(x, y, i):
        store.setdefault(k.item() if hasattr(k, "item") else k, {})[
            float(t)
        ] = float(v)
    if order is not None:
        store = {k: store[k] for k in order}
    return store


def _same(a, b, Z=None):
    """Two degradation models fitted to the same readings in the same
    order: identical dictionaries (they hold the readings, the parameters
    and the life model), covariances and predictions."""
    assert a.to_dict() == b.to_dict()
    if hasattr(a, "covariance"):
        np.testing.assert_array_equal(a.covariance(), b.covariance())
    t = np.array([200.0, 400.0, 800.0])
    if Z is None:
        np.testing.assert_array_equal(a.sf(t), b.sf(t))
    else:
        np.testing.assert_array_equal(a.sf(t, Z), b.sf(t, Z))


@pytest.mark.parametrize("name", list(FITTERS))
def test_measurements_fit_equals_the_arrays_fit(name):
    fitter, data, threshold = FITTERS[name]
    d = data()
    ref = fitter.fit(d["x"], d["y"], d["i"], threshold=threshold)
    m = Measurements(_store(d["x"], d["y"], d["i"]))
    _same(ref, fitter.fit(m, threshold=threshold))
    # positionally too: the threshold after the two left-out arrays
    _same(ref, fitter.fit(m, None, None, threshold))


@pytest.mark.parametrize("name", list(FITTERS))
def test_surpyval_data_fit_equals_the_arrays_fit(name):
    fitter, data, threshold = FITTERS[name]
    d = data()
    ref = fitter.fit(d["x"], d["y"], d["i"], threshold=threshold)
    units = list(dict.fromkeys(d["i"].tolist()))
    life = SurpyvalData(
        np.full(len(units), 5000.0),
        c=np.ones(len(units)),
        i=units,
        m=_store(d["x"], d["y"], d["i"]),
    )
    _same(ref, fitter.fit(life, threshold=threshold))


@pytest.mark.parametrize("name", list(FITTERS))
def test_units_in_any_order_give_the_same_fit(name):
    # The readings are matched by unit: a Measurements listing the units
    # in another order gives the fit of the arrays in that order, the
    # same numbers.
    fitter, data, threshold = FITTERS[name]
    d = data()
    units = list(dict.fromkeys(d["i"].tolist()))
    shuffled = units[::-1]
    ref = fitter.fit(d["x"], d["y"], d["i"], threshold=threshold)
    got = fitter.fit(
        Measurements(_store(d["x"], d["y"], d["i"], shuffled)),
        threshold=threshold,
    )
    t = np.array([200.0, 400.0, 800.0])
    np.testing.assert_allclose(got.sf(t), ref.sf(t), rtol=1e-12)


def test_rul_prediction_is_the_same():
    d = path_data()
    ref = DegradationAnalysis.fit(d["x"], d["y"], d["i"], threshold=150.0)
    got = DegradationAnalysis.fit(
        Measurements(_store(d["x"], d["y"], d["i"])), threshold=150.0
    )
    a = ref.predict_rul([100.0, 200.0], [40.0, 75.0], random_state=1)
    b = got.predict_rul([100.0, 200.0], [40.0, 75.0], random_state=1)
    assert (a.rul, a.rul_interval) == (b.rul, b.rul_interval)
    w = WienerProcess.fit(*[process_data()[k] for k in "xyi"], 100.0)
    p = process_data()
    w2 = WienerProcess.fit(
        Measurements(_store(p["x"], p["y"], p["i"])), threshold=100.0
    )
    assert repr(w.predict_rul(40.0)) == repr(w2.predict_rul(40.0))
    assert w.predict_rul(40.0).rul == w2.predict_rul(40.0).rul


def test_text_ids_and_a_named_measurement():
    d = process_data()
    ids = np.array([f"unit-{k}" for k in d["i"]])
    ref = WienerProcess.fit(d["x"], d["y"], ids, threshold=100.0)
    m = Measurements(_store(d["x"], d["y"], ids), name="wear", units="mm")
    _same(ref, WienerProcess.fit(m, threshold=100.0))


def _two():
    d = process_data()
    wear = _store(d["x"], d["y"], d["i"])
    oil = {k: {0.0: 1.0, 10.0: 2.0} for k in wear}
    return d, Measurements({"wear": wear, "oil": oil})


def test_several_measurements_are_refused_naming_the_choices():
    d, m = _two()
    with pytest.raises(ValueError, match=r"wear, oil.*m.with_measurement"):
        WienerProcess.fit(m, threshold=100.0)
    ref = WienerProcess.fit(d["x"], d["y"], d["i"], threshold=100.0)
    _same(ref, WienerProcess.fit(m.with_measurement("wear"), threshold=100.0))
    life = SurpyvalData(np.full(5, 500.0), c=np.ones(5), i=list(range(5)), m=m)
    with pytest.raises(ValueError, match=r"data.with_measurement\(name\)"):
        GammaProcess.fit(life, threshold=100.0)
    _same(
        ref,
        WienerProcess.fit(life.with_measurement("wear"), threshold=100.0),
    )


def test_units_with_no_readings_are_warned_of_and_left_out():
    d = process_data()
    store = _store(d["x"], d["y"], d["i"])
    ref = WienerProcess.fit(d["x"], d["y"], d["i"], threshold=100.0)
    # a unit in the life rows with no readings
    life = SurpyvalData(
        np.full(6, 500.0), c=np.ones(6), i=list(range(5)) + [99], m=store
    )
    with pytest.warns(UserWarning, match=r"1 unit has no readings: 99"):
        got = WienerProcess.fit(life, threshold=100.0)
    _same(ref, got)
    # a unit listed in the Measurements with none
    with pytest.warns(UserWarning, match=r"1 unit has no readings: 'x'"):
        got = WienerProcess.fit(
            Measurements({**{str(k): v for k, v in store.items()}, "x": {}}),
            threshold=100.0,
        )
    # a unit with readings but no life row: warned when attached, and its
    # readings are not used
    with pytest.warns(UserWarning, match="no life rows"):
        life = SurpyvalData(
            np.full(4, 500.0), c=np.ones(4), i=[0, 1, 2, 3], m=store
        )
    kept = d["i"] < 4
    _same(
        WienerProcess.fit(
            d["x"][kept], d["y"][kept], d["i"][kept], threshold=100.0
        ),
        WienerProcess.fit(life, threshold=100.0),
    )


def test_the_data_s_covariates_are_each_unit_s_stress():
    x, y, i, Z = adt_data(seed=1)
    ref = DegradationAnalysis.fit(x, y, i, threshold=100.0, Z=Z)
    units = list(dict.fromkeys(i.tolist()))
    Z_unit = [Z[i == k][0] for k in units]
    life = SurpyvalData(
        np.full(len(units), 500.0),
        c=np.ones(len(units)),
        i=units,
        Z=Z_unit,
        m=_store(x, y, i),
    )
    got = DegradationAnalysis.fit(life, threshold=100.0)
    np.testing.assert_array_equal(got.life_model.params, ref.life_model.params)
    np.testing.assert_array_equal(got.sf(50.0, [1.0]), ref.sf(50.0, [1.0]))
    # named covariates name the stress, as fit_from_df's Z_cols do
    named = SurpyvalData.from_dataframe(
        pd.DataFrame({"x": 500.0, "c": 1, "i": units, "stress": Z_unit}),
        x_col="x",
        c_col="c",
        i_col="i",
        Z_cols=["stress"],
        m=_store(x, y, i),
    )
    model = DegradationAnalysis.fit(named, threshold=100.0)
    assert model.Z_cols == ["stress"]
    np.testing.assert_array_equal(
        model.sf(50.0, pd.DataFrame({"stress": [1.0]})),
        ref.sf(50.0, [1.0]),
    )


def test_process_stress_from_the_data():
    d = process_data()
    stress = {0: 0.0, 1: 0.0, 2: 1.0, 3: 1.0, 4: 1.0}
    Z = np.array([stress[k] for k in d["i"]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ref = WienerProcess.fit(d["x"], d["y"], d["i"], threshold=100.0, Z=Z)
        life = SurpyvalData(
            np.full(5, 500.0),
            c=np.ones(5),
            i=list(stress),
            Z=list(stress.values()),
            m=_store(d["x"], d["y"], d["i"]),
        )
        got = WienerProcess.fit(life, threshold=100.0)
    _same(ref, got, Z=[1.0])


def test_other_data_arguments_beside_the_readings_are_refused():
    d = process_data()
    m = Measurements(_store(d["x"], d["y"], d["i"]))
    with pytest.raises(TypeError, match=r"leave out 'y', 'i'"):
        WienerProcess.fit(m, d["y"], d["i"], threshold=100.0)
    with pytest.raises(TypeError, match=r"leave out 'Z'.*SurpyvalData"):
        DegradationAnalysis.fit(m, threshold=150.0, Z=np.ones(55))
    with pytest.raises(TypeError, match="needs 'threshold'"):
        GammaProcess.fit(m)
    with pytest.raises(TypeError, match=r"needs 'i'"):
        GammaProcess.fit(d["x"], d["y"], threshold=100.0)


def test_data_the_fit_has_no_place_for_is_refused():
    d = process_data()
    store = _store(d["x"], d["y"], d["i"])
    with pytest.raises(ValueError, match="holds none"):
        WienerProcess.fit(SurpyvalData([1.0, 2.0], i=[0, 1]), threshold=100.0)
    counted = SurpyvalData(
        np.full(5, 500.0), c=np.ones(5), n=[1, 2, 1, 1, 1], i=[0, 1, 2, 3, 4]
    )
    counted.m = store
    with pytest.raises(ValueError, match="takes no counts"):
        WienerProcess.fit(counted, threshold=100.0)
    truncated = SurpyvalData(
        np.full(5, 500.0), c=np.ones(5), tl=1.0, i=[0, 1, 2, 3, 4], m=store
    )
    with pytest.raises(ValueError, match="truncation"):
        DegradationAnalysis.fit(truncated, threshold=150.0)


@pytest.mark.parametrize("name", list(FITTERS))
def test_fit_from_df_names_the_readings_m_col_or_y_col(name):
    fitter, data, threshold = FITTERS[name]
    d = data()
    df = pd.DataFrame({"unit": d["i"], "t": d["x"], "reading": d["y"]})
    ref = fitter.fit(d["x"], d["y"], d["i"], threshold=threshold)
    new = fitter.fit_from_df(
        df, i_col="unit", x_col="t", m_col="reading", threshold=threshold
    )
    old = fitter.fit_from_df(
        df, i_col="unit", x_col="t", y_col="reading", threshold=threshold
    )
    _same(ref, new)
    _same(ref, old)
    # the default column is still "y"
    plain = pd.DataFrame(d)
    _same(ref, fitter.fit_from_df(plain, threshold=threshold))
    with pytest.raises(TypeError, match="give one"):
        fitter.fit_from_df(
            df,
            i_col="unit",
            x_col="t",
            m_col="reading",
            y_col="reading",
            threshold=threshold,
        )
    # the long frame Measurements.from_dataframe reads
    m = Measurements.from_dataframe(
        df, i_col="unit", x_col="t", m_col="reading"
    )
    _same(ref, fitter.fit(m, threshold=threshold))
