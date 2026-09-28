"""Regression tests for the degradation part of #375 (missing values) and
for #374 (predicting from a DataFrame after ``fit_from_df``).

#375, NaN in, NaN out (element by element), except for the methods that
describe one unit's history, which raise:

* Gamma/Wiener ``sf``/``ff``/``Hf``/``df``/``hf`` at a ``nan`` time gave
  ``sf = 1`` and ``ff = df = 0``, and ``qf(nan)`` gave ``inf``.
* Process ``predict_rul(current_degradation=nan)`` never returned.
* ``DegradationModel.qf`` gave ``inf`` for a ``nan`` covariate or ``p``
  (and read only the first row of a multi-row ``Z``).
* Process ``sf``/``mean`` raised on a ``nan`` stress, where the covariate
  degradation model returns ``nan``; so did the step-stress model.
* ``InducedFailureDistribution`` gave ``ff = 0`` at a ``nan`` time and
  ``qf(nan)`` raised; the bootstrap ``cb`` raised on a ``nan`` stress.

#374: ``fit_from_df`` records ``Z_cols``, and every method that takes ``Z``
reads a DataFrame by those names, before and after ``to_dict``.
"""

import json
import threading

import numpy as np
import pandas as pd
import pytest

from surpyval import StepSchedule
from surpyval.degradation import (
    DegradationAnalysis,
    DegradationModel,
    GammaProcess,
    GammaProcessModel,
    WienerProcess,
    WienerProcessModel,
)

PROCESSES = [GammaProcess, WienerProcess]
REFUSAL = "fitted without covariate names"


def _returns(fn, seconds=20.0):
    """Run ``fn`` and return ``(finished, result or exception)``, giving up
    after ``seconds`` -- so a hang fails the test instead of the run."""
    box = {}

    def run():
        try:
            box["out"] = fn()
        except Exception as e:  # handed to the caller
            box["out"] = e

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    return not worker.is_alive(), box.get("out")


# -- data ------------------------------------------------------------------


def _process_data():
    """Monotone paths of 24 units at four stress levels (one per unit)."""
    rng = np.random.default_rng(0)
    t = np.arange(1, 11) * 5.0
    xs, ys, ids, zs = [], [], [], []
    for u in range(24):
        stress = [0.0, 0.5, 1.0, 1.5][u % 4]
        inc = rng.gamma(0.5 * np.exp(0.5 * stress) * 5, 1.0, size=t.size)
        xs.append(t)
        ys.append(np.cumsum(inc))
        ids.append(np.full(t.size, u))
        zs.append(np.full(t.size, stress))
    return tuple(np.concatenate(v) for v in (xs, ys, ids, zs))


def _adt_data():
    """Linear paths whose rate grows with stress (constant per unit)."""
    rng = np.random.default_rng(0)
    t = np.arange(1, 11) * 5.0
    xs, ys, ids, zs = [], [], [], []
    for u in range(24):
        stress = [0.0, 0.5, 1.0, 1.5][u % 4]
        rate = 0.5 * np.exp(0.8 * stress) * np.exp(rng.normal(0, 0.1))
        xs.append(t)
        ys.append(10 + rng.normal(0, 1) + rate * t + rng.normal(0, 0.5, 10))
        ids.append(np.full(t.size, u))
        zs.append(np.full(t.size, stress))
    return tuple(np.concatenate(v) for v in (xs, ys, ids, zs))


# a step-stress profile: z = 0 up to t = 10, then z = 1
def _step_data():
    rng = np.random.default_rng(1)
    t = np.arange(1.0, 21.0)
    z = np.where(t <= 10, 0.0, 1.0)
    tau = np.cumsum(np.exp(0.7 * z))
    xs, ys, ids, zs = [], [], [], []
    for u in range(10):
        a, b = rng.normal([1.0, 0.5], [0.2, 0.05])
        xs.append(t)
        ys.append(a + b * tau + rng.normal(0, 0.1, t.size))
        ids.append(np.full(t.size, u))
        zs.append(z)
    return tuple(np.concatenate(v) for v in (xs, ys, ids, zs))


@pytest.fixture(scope="module")
def process_frame():
    x, y, i, z = _process_data()
    return pd.DataFrame({"t": x, "y": y, "unit": i, "temp": z, "other": 1.0})


@pytest.fixture(scope="module", params=PROCESSES, ids=["gamma", "wiener"])
def process_model(request, process_frame):
    """A stressed process model fitted from arrays."""
    df = process_frame
    return request.param.fit(
        df["t"], df["y"], df["unit"], threshold=60.0, Z=df["temp"]
    )


@pytest.fixture(scope="module", params=PROCESSES, ids=["gamma", "wiener"])
def process_models(request, process_frame):
    """The same fit from arrays and from a DataFrame (``Z_cols``)."""
    df = process_frame
    fitter = request.param
    arrays = fitter.fit(
        df["t"], df["y"], df["unit"], threshold=60.0, Z=df["temp"]
    )
    named = fitter.fit_from_df(
        df, x="t", y="y", i="unit", Z_cols="temp", threshold=60.0
    )
    return arrays, named


@pytest.fixture(scope="module")
def adt_frame():
    x, y, i, z = _adt_data()
    return pd.DataFrame({"t": x, "y": y, "unit": i, "temp": z})


@pytest.fixture(scope="module")
def adt_models(adt_frame):
    df = adt_frame
    arrays = DegradationAnalysis.fit(
        df["t"], df["y"], df["unit"], threshold=100.0, Z=df["temp"]
    )
    named = DegradationAnalysis.fit_from_df(
        df, x="t", y="y", i="unit", Z_cols="temp", threshold=100.0
    )
    return arrays, named


@pytest.fixture(scope="module")
def linked_models(adt_frame):
    df = adt_frame
    kwargs = dict(threshold=100.0, links={"b": "log"})
    arrays = DegradationAnalysis.fit(
        df["t"], df["y"], df["unit"], Z=df["temp"], **kwargs
    )
    named = DegradationAnalysis.fit_from_df(
        df, x="t", y="y", i="unit", Z_cols=["temp"], **kwargs
    )
    return arrays, named


@pytest.fixture(scope="module")
def clock_models():
    x, y, i, z = _step_data()
    df = pd.DataFrame({"t": x, "y": y, "unit": i, "load": z})
    kwargs = dict(threshold=12.0, acceleration="clock", stress_ref=[0.0])
    arrays = DegradationAnalysis.fit(x, y, i, Z=z, **kwargs)
    named = DegradationAnalysis.fit_from_df(
        df, x="t", y="y", i="unit", Z_cols="load", **kwargs
    )
    return arrays, named


def _row(value, name="temp"):
    return pd.DataFrame({"unused": [-1.0], name: [value]})


# -- #375: process models ------------------------------------------------


@pytest.mark.parametrize("fn", ["sf", "ff", "Hf", "hf", "df"])
def test_process_nan_time_is_nan(process_model, fn):
    model = process_model
    t = np.array([np.nan, 50.0])
    out = getattr(model, fn)(t, Z=[0.5])
    # it was sf = 1, ff = Hf = hf = df = 0
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(getattr(model, fn)(50.0, Z=[0.5]))
    assert np.isnan(getattr(model, fn)(np.nan, Z=[0.5]))
    # under a stress profile too
    profile = StepSchedule.from_changepoints([0.0, 20.0], [[0.0], [1.0]])
    along = getattr(model, fn)(t, Z=profile)
    assert np.isnan(along[0]) and np.isfinite(along[1])


@pytest.mark.parametrize("fitter", PROCESSES)
def test_unstressed_process_nan_time_is_nan(fitter):
    x, y, i, _ = _process_data()
    model = fitter.fit(x, y, i, threshold=60.0)
    assert np.isnan(model.sf(np.nan))
    assert np.isnan(model.ff([np.nan, 40.0])[0])
    assert np.isnan(model.qf(np.nan))


def test_process_qf_of_nan_is_nan(process_model):
    model = process_model
    out = model.qf([np.nan, 0.5], Z=[0.5])
    # qf(nan) was inf
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(model.qf(0.5, Z=[0.5]))


def test_process_predict_rul_refuses_a_nan_degradation(process_model):
    model = process_model
    finished, out = _returns(lambda: model.predict_rul(np.nan, Z=[0.5]))
    # it never returned
    assert finished
    assert isinstance(out, ValueError)
    assert "current_degradation" in str(out)


@pytest.mark.parametrize(
    "model_class", [GammaProcessModel, WienerProcessModel]
)
def test_process_quantile_search_is_bounded(model_class):
    # no start of the bracket search can hang it: a nan distance used to
    # double a nan bracket forever
    model = model_class(1.0, 1.0, 10.0)
    finished, out = _returns(lambda: model._quantile(0.5, np.nan))
    assert finished
    assert out == np.inf
    with np.errstate(invalid="ignore"):
        assert model.predict_rul(-np.inf).rul == np.inf


def test_process_nan_stress_is_nan(process_model):
    model = process_model
    t = np.array([50.0, 100.0])
    # sf and mean raised "Z must contain only finite values"
    assert np.isnan(model.sf(t, Z=[np.nan])).all()
    assert np.isnan(model.ff(t, Z=[np.nan])).all()
    assert np.isnan(model.hf(t, Z=[np.nan])).all()
    assert np.isnan(model.mean(Z=[np.nan]))
    assert np.isnan(model.qf([0.0, 0.5], Z=[np.nan])).all()
    assert np.isnan(model.acceleration_factor([np.nan]))
    assert np.isnan(model.random(3, random_state=0, Z=[np.nan])).all()
    # an infinite stress is still refused
    with pytest.raises(ValueError, match="finite"):
        model.sf(t, Z=[np.inf])


def test_process_predict_rul_refuses_a_nan_stress(process_model):
    model = process_model
    with pytest.raises(ValueError, match="Z must contain only finite"):
        model.predict_rul(20.0, Z=[np.nan])


# -- #375: DegradationModel ----------------------------------------------


def test_adt_qf_nan_covariate_or_p_is_nan(adt_models):
    model, _ = adt_models
    # both were inf
    assert np.isnan(model.qf([0.1, 0.5], Z=[np.nan])).all()
    out = model.qf([np.nan, 0.5], Z=[0.5])
    assert np.isnan(out[0])
    assert out[1] == pytest.approx(model.qf(0.5, Z=[0.5]))
    # each p is paired with its row of Z, as sf pairs x; only row 0 was
    # used
    rows = model.qf(0.5, Z=np.array([[np.nan], [0.5], [1.0]]))
    assert np.isnan(rows[0])
    assert rows[1:] == pytest.approx(
        [model.qf(0.5, Z=[0.5]), model.qf(0.5, Z=[1.0])]
    )
    assert rows[1] > rows[2]
    with pytest.raises(ValueError, match="pairs each probability"):
        model.qf([0.1, 0.5], Z=[[0.5], [1.0], [1.5]])


def test_adt_nan_covariate_mean_random_and_cb(adt_models):
    model, _ = adt_models
    assert np.isnan(model.mean(Z=[np.nan]))
    assert np.isnan(model.random(3, Z=[np.nan], random_state=0)).all()
    # the bootstrap dropped every refit and raised
    band = model.cb(
        [50.0, 100.0],
        method="bootstrap",
        n_boot=20,
        random_state=0,
        Z=[np.nan],
    )
    assert band.shape == (2, 2) and np.isnan(band).all()
    band = model.cb(
        [np.nan, 100.0], method="bootstrap", n_boot=20, random_state=0, Z=[0.5]
    )
    assert np.isnan(band[0]).all() and np.isfinite(band[1]).all()


def test_induced_life_nan_time_and_p():
    x, y, i, _ = _adt_data()
    model = DegradationAnalysis.fit(x, y, i, threshold=100.0)
    induced = model.induced_life(n_samples=500, random_state=0)
    # ff was 0 (sf 1) at a nan time, and qf(nan) raised
    assert np.isnan(induced.ff(np.nan)) and np.isnan(induced.sf(np.nan))
    ff = induced.ff([np.nan, 100.0])
    assert np.isnan(ff[0]) and ff[1] == induced.ff(100.0)
    qf = induced.qf([np.nan, 0.5])
    assert np.isnan(qf[0]) and qf[1] == induced.qf(0.5)


def test_clock_model_nan_stress_is_nan(clock_models):
    model, _ = clock_models
    t = np.array([5.0, 10.0])
    # the step-stress model raised, like the process models
    assert np.isnan(model.sf(t, Z=[np.nan])).all()
    assert np.isnan(model.hf(t, Z=[np.nan])).all()
    assert np.isnan(model.qf([0.1, 0.5], Z=[np.nan])).all()
    assert np.isnan(model.mean(Z=[np.nan]))
    assert np.isnan(model.acceleration_factor([np.nan]))
    sf = model.sf(np.array([np.nan, 5.0]), Z=[0.0])
    assert np.isnan(sf[0]) and np.isfinite(sf[1])


def test_clock_model_one_unit_methods_refuse_nan(clock_models):
    model, _ = clock_models
    x, y, i, z = _step_data()
    unit = i == 0
    with pytest.raises(ValueError, match="finite"):
        model.predict_rul(x[unit], y[unit], Z=np.r_[np.nan, z[unit][1:]])
    with pytest.raises(ValueError, match="finite"):
        model.predict_rul(x[unit], y[unit], Z=z[unit], Z_future=[np.nan])
    with pytest.raises(ValueError, match="finite"):
        model.predict_failure_time(
            x[unit], y[unit], Z=z[unit], Z_future=[np.nan]
        )
    with pytest.raises(ValueError, match="finite"):
        model.induced_life(n_samples=100, random_state=0, Z=[np.nan])


def test_linked_population_nan_covariate(linked_models):
    model, _ = linked_models
    names = model.path_param_fixed_names
    # only the stress-dependent parameter (b) is nan; it raised
    mean = model.path_param_link_mean([np.nan])
    assert np.isfinite(mean[0]) and np.isnan(mean[1])
    assert mean[0] == pytest.approx(model.path_param_link_mean([0.5])[0])
    median = model.path_param_median([np.nan])
    assert np.isfinite(median[0]) and np.isnan(median[1])
    assert names is not None
    x, y, _, _ = _adt_data()
    # one unit's prediction still refuses a missing stress
    with pytest.raises(ValueError, match="finite"):
        model.predict_rul(x[:5], y[:5], Z=[np.nan], random_state=0)
    with pytest.raises(ValueError, match="finite"):
        model.induced_life(n_samples=100, random_state=0, Z=[np.nan])


# -- #374: DataFrames by name --------------------------------------------


def test_process_fit_from_df_matches_arrays(process_models):
    arrays, named = process_models
    assert arrays.Z_cols is None and named.Z_cols == ["temp"]
    assert named.params == pytest.approx(arrays.params)
    t = np.array([30.0, 60.0])
    for fn in ["sf", "ff", "df", "hf", "Hf"]:
        assert getattr(named, fn)(t, Z=_row(0.5)) == pytest.approx(
            getattr(arrays, fn)(t, Z=[0.5])
        )
    assert named.qf(0.5, Z=_row(0.5)) == pytest.approx(arrays.qf(0.5, Z=[0.5]))
    assert named.mean(Z=_row(0.5)) == pytest.approx(arrays.mean(Z=[0.5]))
    assert named.acceleration_factor(_row(1.0)) == pytest.approx(
        arrays.acceleration_factor([1.0])
    )
    assert named.random(3, _row(0.5), random_state=0) == pytest.approx(
        arrays.random(3, [0.5], random_state=0)
    )
    rul = named.predict_rul(20.0, Z=_row(0.5))
    assert rul.rul == pytest.approx(arrays.predict_rul(20.0, Z=[0.5]).rul)
    assert np.isnan(named.sf(50.0, Z=_row(np.nan)))
    with pytest.raises(ValueError, match="not in dataframe columns"):
        named.sf(50.0, Z=pd.DataFrame({"load": [0.5]}))


def test_process_names_survive_serialisation(process_models):
    _, named = process_models
    as_dict = named.to_dict()
    assert as_dict["Z_cols"] == ["temp"]
    # an older reader ignores the key, so the schema is unchanged
    assert as_dict["schema"] == 1
    restored = type(named).from_dict(json.loads(json.dumps(as_dict)))
    assert restored.Z_cols == ["temp"]
    assert restored.sf(50.0, Z=_row(0.5)) == pytest.approx(
        named.sf(50.0, Z=[0.5])
    )
    assert (
        "Z_cols"
        not in type(named)
        .from_dict({k: v for k, v in as_dict.items() if k != "Z_cols"})
        .to_dict()
    )


def test_process_fitted_from_arrays_refuses_a_dataframe(process_model):
    arrays = process_model
    # a DataFrame was read by position, silently
    with pytest.raises(ValueError, match=REFUSAL) as info:
        arrays.sf(50.0, Z=_row(0.5))
    assert "Process.fit_from_df" in str(info.value)
    with pytest.raises(ValueError, match=REFUSAL):
        arrays.predict_rul(20.0, Z=_row(0.5))


def test_adt_fit_from_df_reads_a_dataframe_by_name(adt_models):
    arrays, named = adt_models
    assert named.Z_cols == ["temp"] and arrays.Z_cols is None
    Z = pd.DataFrame({"other": [1.0, 2.0], "temp": [0.5, 1.0]})
    t = np.array([50.0, 100.0])
    for fn in ["sf", "ff", "df", "hf", "Hf"]:
        assert getattr(named, fn)(t, Z) == pytest.approx(
            getattr(arrays, fn)(t, [[0.5], [1.0]])
        )
    assert named.qf(0.5, Z) == pytest.approx(arrays.qf(0.5, [[0.5], [1.0]]))
    assert named.mean(Z.iloc[:1]) == pytest.approx(arrays.mean([0.5]))
    assert named.random(4, Z.iloc[:1], random_state=0) == pytest.approx(
        arrays.random(4, [0.5], random_state=0)
    )
    band = named.cb(
        [50.0], method="bootstrap", n_boot=10, random_state=0, Z=Z.iloc[:1]
    )
    assert band == pytest.approx(
        arrays.cb(
            [50.0], method="bootstrap", n_boot=10, random_state=0, Z=[0.5]
        )
    )
    # the life model reads it by name too
    assert named.life_model.sf(t, Z) == pytest.approx(
        arrays.sf(t, [[0.5], [1.0]])
    )
    # a missing value is nan in its row only
    Z_nan = pd.DataFrame({"temp": [np.nan, 1.0]})
    sf = named.sf(t, Z_nan)
    assert np.isnan(sf[0])
    assert sf[1] == pytest.approx(arrays.sf(t, [[0.5], [1.0]])[1])


def test_adt_names_survive_serialisation(adt_models):
    _, named = adt_models
    as_dict = named.to_dict()
    assert as_dict["Z_cols"] == ["temp"] and as_dict["schema"] == 1
    restored = DegradationModel.from_dict(json.loads(json.dumps(as_dict)))
    assert restored.Z_cols == ["temp"]
    Z = pd.DataFrame({"temp": [0.5]})
    assert restored.sf([50.0], Z) == pytest.approx(named.sf([50.0], [0.5]))
    assert restored.life_model.sf([50.0], Z) == pytest.approx(
        named.sf([50.0], [0.5])
    )


def test_adt_fitted_from_arrays_refuses_a_dataframe(adt_models):
    arrays, _ = adt_models
    with pytest.raises(ValueError, match=REFUSAL) as info:
        arrays.sf([50.0], pd.DataFrame({"temp": [0.5]}))
    assert "DegradationAnalysis.fit_from_df" in str(info.value)
    x, y, i, _ = _adt_data()
    plain = DegradationAnalysis.fit(x, y, i, threshold=100.0)
    with pytest.raises(ValueError, match="no covariates"):
        plain.sf([50.0], pd.DataFrame({"temp": [0.5]}))


def test_linked_model_reads_a_dataframe_by_name(linked_models):
    arrays, named = linked_models
    row = _row(0.5)
    assert named.path_param_link_mean(row) == pytest.approx(
        arrays.path_param_link_mean([0.5])
    )
    assert named.path_param_median(row) == pytest.approx(
        arrays.path_param_median([0.5])
    )
    x, y, _, _ = _adt_data()
    pred = named.predict_rul(x[:5], y[:5], Z=row, random_state=0)
    ref = arrays.predict_rul(x[:5], y[:5], Z=[0.5], random_state=0)
    assert pred.failure_time == pytest.approx(ref.failure_time)
    induced = named.induced_life(n_samples=200, random_state=0, Z=row)
    assert induced.median() == pytest.approx(
        arrays.induced_life(n_samples=200, random_state=0, Z=[0.5]).median()
    )
    restored = DegradationModel.from_dict(named.to_dict())
    assert restored.path_param_median(row) == pytest.approx(
        arrays.path_param_median([0.5])
    )
    with pytest.raises(ValueError, match=REFUSAL):
        arrays.path_param_median(row)


def test_clock_model_reads_histories_by_name(clock_models):
    arrays, named = clock_models
    assert named.Z_cols == ["load"]
    x, y, i, z = _step_data()
    unit = i == 3
    history = pd.DataFrame({"load": z[unit], "unit": 3})
    future = pd.DataFrame({"load": [1.0]})
    t = np.array([5.0, 15.0])
    assert named.sf(t, Z=future) == pytest.approx(arrays.sf(t, Z=[1.0]))
    assert named.acceleration_factor(future) == pytest.approx(
        arrays.acceleration_factor([1.0])
    )
    assert named.mean(Z=future) == pytest.approx(arrays.mean(Z=[1.0]))
    ft = named.predict_failure_time(
        x[unit], y[unit], Z=history, Z_future=future
    )
    assert ft == pytest.approx(
        arrays.predict_failure_time(x[unit], y[unit], Z=z[unit], Z_future=[1])
    )
    rl = named.predict_remaining_life(x[unit], y[unit], Z=history)
    assert rl == pytest.approx(
        arrays.predict_remaining_life(x[unit], y[unit], Z=z[unit])
    )
    pred = named.predict_rul(
        x[unit], y[unit], Z=history, Z_future=future, random_state=0
    )
    ref = arrays.predict_rul(
        x[unit], y[unit], Z=z[unit], Z_future=[1.0], random_state=0
    )
    assert pred.failure_time == pytest.approx(ref.failure_time)
    induced = named.induced_life(n_samples=200, random_state=0, Z=future)
    assert induced.stress == [1.0]
    band = named.cb(t, method="bootstrap", n_boot=5, random_state=0, Z=future)
    assert band == pytest.approx(
        arrays.cb(t, method="bootstrap", n_boot=5, random_state=0, Z=[1.0])
    )
    restored = DegradationModel.from_dict(named.to_dict())
    assert restored.sf(t, Z=future) == pytest.approx(arrays.sf(t, Z=[1.0]))
    with pytest.raises(ValueError, match=REFUSAL):
        arrays.predict_rul(x[unit], y[unit], Z=history, random_state=0)
