"""Serialisation of the fitted degradation models.

The degradation result classes round-trip through ``to_dict``/``from_dict``
(and the JSON file variants): the Wiener and gamma stochastic-process models
(a few floats each), the Monte-Carlo ``InducedFailureDistribution`` (including
its ``inf`` never-fails mass), and the full ``DegradationModel`` (raw data,
per-unit paths, population summaries, and its fitted life model -- plain or
accelerated). Each restored model reproduces its predictions exactly.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval
from surpyval.degradation import (
    PATH_MODELS,
    DegradationAnalysis,
    DegradationModel,
    DestructiveDegradation,
    DestructiveDegradationModel,
    GammaProcess,
    GammaProcessModel,
    InducedFailureDistribution,
    WienerProcess,
    WienerProcessModel,
)
from surpyval.tests._helpers import json_round_trip, linear_degradation_paths


def _monotone_process_data(seed=1, n_units=15):
    rng = np.random.default_rng(seed)
    xs, ys, ii = [], [], []
    for u in range(n_units):
        t = np.arange(0, 20, 2.0)
        y = np.cumsum(np.abs(rng.normal(1.0, 0.5, t.size)))
        xs.append(t)
        ys.append(y)
        ii.append(np.full(t.size, u))
    return (np.concatenate(z) for z in (xs, ys, ii))


def _wiener_process_data(seed=2, n_units=15):
    rng = np.random.default_rng(seed)
    xs, ys, ii = [], [], []
    for u in range(n_units):
        t = np.arange(0, 20, 2.0)
        y = np.cumsum(rng.normal(1.0, 1.0, t.size))
        xs.append(t)
        ys.append(y)
        ii.append(np.full(t.size, u))
    return (np.concatenate(z) for z in (xs, ys, ii))


# -- process models -------------------------------------------------------


def test_gamma_process_round_trip():
    xg, yg, ig = _monotone_process_data()
    model = GammaProcess.fit(xg, yg, ig, threshold=15.0)
    restored = GammaProcessModel.from_dict(json_round_trip(model.to_dict()))
    t = np.array([5.0, 10.0, 15.0])
    assert np.allclose(model.sf(t), restored.sf(t))
    assert np.allclose(model.ff(t), restored.ff(t))
    assert (model.alpha, model.beta, model.threshold) == (
        restored.alpha,
        restored.beta,
        restored.threshold,
    )


def test_wiener_process_round_trip(tmp_path):
    xw, yw, iw = _wiener_process_data()
    model = WienerProcess.fit(xw, yw, iw, threshold=20.0)
    fp = tmp_path / "wiener.json"
    model.to_json(fp)
    restored = WienerProcessModel.from_json(fp)
    t = np.array([5.0, 10.0, 15.0])
    assert np.allclose(model.sf(t), restored.sf(t))
    assert np.isclose(model.mean(), restored.mean())


def test_process_model_rejects_wrong_dict():
    with pytest.raises(ValueError, match="GammaProcessModel"):
        GammaProcessModel.from_dict({"model": "Other"})
    with pytest.raises(ValueError, match="WienerProcessModel"):
        WienerProcessModel.from_dict({"model": "Other"})


# -- induced failure distribution -----------------------------------------


def test_induced_failure_distribution_round_trip():
    x, y, i = linear_degradation_paths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(x, y, i, threshold=150)
    induced = model.induced_life(n_samples=5000, random_state=3)
    restored = InducedFailureDistribution.from_dict(
        json_round_trip(induced.to_dict())
    )
    t = np.array([300.0, 450.0, 600.0])
    assert np.allclose(induced.ff(t), restored.ff(t))
    assert np.isclose(induced.median(), restored.median())
    assert np.isclose(induced.prob_never_fails, restored.prob_never_fails)


def test_induced_never_fails_mass_survives_round_trip():
    # a straddling-zero slope population leaves a genuine inf mass
    rng = np.random.default_rng(5)
    xs, ys, ids = [], [], []
    for u in range(60):
        a = rng.normal(0.0, 0.2)
        b = rng.normal(0.2, 0.5)  # some units never reach the threshold
        t = np.arange(0, 20, 2.0)
        y = a + b * t + rng.normal(0, 0.4, t.size)
        xs.append(t)
        ys.append(y)
        ids.append(np.full(t.size, u))
    x, y, i = (np.concatenate(z) for z in (xs, ys, ids))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(x, y, i, threshold=30.0, path="linear")
    induced = model.induced_life(n_samples=20000, random_state=6)
    assert induced.prob_never_fails > 0.05
    restored = InducedFailureDistribution.from_dict(induced.to_dict())
    assert np.isclose(induced.prob_never_fails, restored.prob_never_fails)
    assert np.isinf(restored.mean()) == np.isinf(induced.mean())


# -- DegradationModel -----------------------------------------------------


def test_degradation_model_round_trip():
    x, y, i = linear_degradation_paths()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(x, y, i, threshold=150)
    restored = DegradationModel.from_dict(json_round_trip(model.to_dict()))
    t = np.array([300.0, 450.0, 600.0])
    assert np.allclose(model.sf(t), restored.sf(t))
    assert np.allclose(model.ff(t), restored.ff(t))
    assert np.allclose(model.qf(0.5), restored.qf(0.5))
    # per-unit path evaluation matches
    assert np.allclose(
        model.path([100, 200], model.units[0]),
        restored.path([100, 200], restored.units[0]),
    )
    # bootstrap bounds still work (the raw data was kept)
    band = restored.cb(t, method="analytic")
    assert np.asarray(band).shape == (3, 2)


def test_degradation_model_json_file(tmp_path):
    x, y, i = linear_degradation_paths(seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(x, y, i, threshold=150)
    fp = tmp_path / "deg.json"
    model.to_json(fp)
    restored = DegradationModel.from_json(fp)
    t = np.array([300.0, 500.0])
    assert np.allclose(model.sf(t), restored.sf(t))


def test_degradation_model_accelerated_round_trip():
    # the life model is a ParametricRegressionModel; it must round-trip too
    rng = np.random.default_rng(4)
    xs, ys, ii, ZZ = [], [], [], []
    uid = 0
    for stress in [0.0, 0.5, 1.0]:
        for _ in range(12):
            b = (1.0 + stress) * rng.normal(1.0, 0.1)
            t = np.arange(0, 20, 2.0)
            y = b * t + rng.normal(0, 0.4, t.size)
            xs.append(t)
            ys.append(y)
            ii.append(np.full(t.size, uid))
            ZZ.append(np.full(t.size, stress))
            uid += 1
    x, y, i, Z = (np.concatenate(z) for z in (xs, ys, ii, ZZ))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(
            x, y, i, threshold=30.0, path="linear", Z=Z
        )
    restored = DegradationModel.from_dict(json_round_trip(model.to_dict()))
    assert restored.is_accelerated
    t = np.array([10.0, 20.0, 30.0])
    for stress in ([0.0], [0.5], [1.0]):
        assert np.allclose(model.sf(t, Z=stress), restored.sf(t, Z=stress))


def test_degradation_model_rejects_wrong_dict():
    with pytest.raises(ValueError, match="DegradationModel"):
        DegradationModel.from_dict({"model": "Other"})


def _stress_data(seed=0):
    rng = np.random.default_rng(seed)
    x, y, i, Z = [], [], [], []
    for u in range(12):
        s = [1.0, 2.0, 3.0][u % 3]
        rate = rng.lognormal(np.log(0.5 * s), 0.2)
        for t in np.arange(1.0, 11.0):
            x.append(t)
            y.append(rate * t + rng.normal(0, 0.1))
            i.append(u)
            Z.append([s])
    return tuple(map(np.array, (x, y, i, Z)))


@pytest.mark.parametrize("distribution", ["LogNormal", "Weibull", "WeibullPH"])
def test_a_reloaded_model_gives_the_same_bootstrap_bounds(distribution):
    # from_dict left the refit fitter unset, so every bootstrap refit of a
    # reloaded model failed
    import surpyval as surv

    x, y, i, Z = _stress_data()
    dist = getattr(surv, distribution)
    kwargs, at = {}, {}
    if distribution != "LogNormal":
        kwargs, at = {"Z": Z}, {"Z": [2.0]}
    model = DegradationAnalysis.fit(
        x, y, i, threshold=3.0, path="linear", distribution=dist, **kwargs
    )
    restored = surv.from_dict(json.loads(json.dumps(model.to_dict())))
    t = np.array([5.0, 8.0])
    band = model.cb(t, method="bootstrap", n_boot=25, random_state=1, **at)
    again = restored.cb(t, method="bootstrap", n_boot=25, random_state=1, **at)
    np.testing.assert_allclose(again, band)


# ---------------------------------------------------------------------------
# Every built-in path model round-trips through
# ``DegradationModel`` serialisation (the offset-exponential
# model used to fail to reload: its display name was stored
# and only the registry key resolved), and
# ``DestructiveDegradationModel`` is a full serialisable model
# that stores its fit data, so a reloaded model's bootstrap
# ``cb`` works.
# ---------------------------------------------------------------------------


# Per-path data: (true parameters, threshold). Each unit's parameters are
# jittered around these, and the threshold is reached inside or a little
# past the measurement window.
_PATH_CASES = {
    "linear": ([1.0, 0.5], 12.0),
    "quadratic": ([1.0, 0.3, 0.02], 15.0),
    "exponential": ([1.0, 0.08], 8.0),
    "offset-exponential": ([20.0, -18.0, -0.05], 15.0),
    "power": ([0.5, 1.2], 15.0),
    "logarithmic": ([1.0, 3.0], 12.0),
    "lloyd-lipow": ([10.0, 8.0], 9.0),
    "gompertz": ([20.0, 3.0, 0.1], 15.0),
    "michaelis-menten": ([20.0, 10.0], 12.0),
}


def _path_data(key: str) -> tuple:
    params, threshold = _PATH_CASES[key]
    model = PATH_MODELS[key]
    rng = np.random.default_rng(0)
    t = np.arange(1.0, 31.0, 3.0)
    xs, ys, ii = [], [], []
    for u in range(8):
        p = np.asarray(params) * (1 + rng.normal(0, 0.03, len(params)))
        y = model.path(t, *p) + rng.normal(0, 0.02, t.size)
        xs.append(t)
        ys.append(y)
        ii.append(np.full(t.size, u))
    x, y, i = (np.concatenate(z) for z in (xs, ys, ii))
    return x, y, i, threshold


def test_path_cases_cover_every_built_in_path():
    assert set(_PATH_CASES) == set(PATH_MODELS)


@pytest.mark.parametrize("key", sorted(PATH_MODELS))
def test_every_built_in_path_round_trips(key):
    x, y, i, threshold = _path_data(key)
    with warnings.catch_warnings():
        # a few units on some paths trip the population-covariance
        # warnings; they are not what is being tested here
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(x, y, i, threshold=threshold, path=key)
    d = json_round_trip(model.to_dict())
    assert d["path_model"] == key
    restored = DegradationModel.from_dict(d)
    assert restored.path_model is model.path_model
    via_package = surpyval.from_dict(d)
    assert via_package.path_model is model.path_model

    grid = np.quantile(model.pseudo_failure_times, [0.25, 0.5, 0.75])
    assert np.allclose(model.sf(grid), restored.sf(grid))
    assert np.allclose(model.ff(grid), restored.ff(grid))
    unit = model.units[0]
    t = np.array([5.0, 15.0, 25.0])
    assert np.allclose(model.path(t, unit), restored.path(t, unit))
    # a new unit's pseudo failure time comes from refitting the path model
    new = i == i[0]
    assert np.isclose(
        model.predict_failure_time(x[new], y[new]),
        restored.predict_failure_time(x[new], y[new]),
    )


def test_old_dict_with_display_name_still_loads():
    # dictionaries written before the fix stored the display name
    x, y, i, threshold = _path_data("offset-exponential")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(
            x, y, i, threshold=threshold, path="offset-exponential"
        )
    d = json_round_trip(model.to_dict())
    d["path_model"] = "Offset Exponential"
    restored = DegradationModel.from_dict(d)
    assert restored.path_model is model.path_model
    grid = np.array([10.0, 20.0, 30.0])
    assert np.allclose(model.sf(grid), restored.sf(grid))


def _destructive_model() -> DestructiveDegradationModel:
    rng = np.random.default_rng(1)
    x = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
    y = np.exp(4.0 - 0.02 * x + rng.normal(0, 0.1, 24))
    c = np.zeros(24, dtype=int)
    c[0] = 1  # one specimen did not break at the maximum load
    return DestructiveDegradation.fit(
        x, y, threshold=20, c=c, transform="best"
    )


def test_destructive_round_trip_keeps_data_and_bounds():
    model = _destructive_model()
    restored = DestructiveDegradationModel.from_dict(
        json_round_trip(model.to_dict())
    )
    t = np.array([30.0, 50.0, 70.0])
    assert np.allclose(model.sf(t), restored.sf(t))
    assert np.allclose(
        model.degradation_quantile(0.1, t),
        restored.degradation_quantile(0.1, t),
    )
    assert restored.transform_scores == pytest.approx(model.transform_scores)
    assert restored._neg_ll == pytest.approx(model._neg_ll)
    for k in ("x", "y", "c"):
        assert np.array_equal(model.data[k], restored.data[k])
    # same data and seed: the bootstrap band is reproduced exactly
    assert np.allclose(
        model.cb(t, n_boot=20, random_state=3),
        restored.cb(t, n_boot=20, random_state=3),
    )


def test_destructive_json_file_and_package_dispatch(tmp_path):
    model = _destructive_model()
    fp = tmp_path / "destructive.json"
    model.to_json(fp)
    for restored in (
        DestructiveDegradationModel.from_json(fp),
        surpyval.from_json(fp),
        surpyval.from_dict(model.to_dict()),
    ):
        assert isinstance(restored, DestructiveDegradationModel)
        t = np.array([30.0, 50.0])
        assert np.allclose(model.sf(t), restored.sf(t))
        assert np.allclose(
            model.cb(t, n_boot=10, random_state=0),
            restored.cb(t, n_boot=10, random_state=0),
        )
    d = model.to_dict()
    assert d["schema"] == surpyval.serialisation.required_schema(d)


def test_destructive_old_dict_without_data_still_loads():
    model = _destructive_model()
    d = json_round_trip(model.to_dict())
    for key in ("data", "neg_ll", "transform_scores"):
        del d[key]
    restored = DestructiveDegradationModel.from_dict(d)
    t = np.array([30.0, 50.0])
    assert np.allclose(model.sf(t), restored.sf(t))
    with pytest.raises(ValueError, match="fit data"):
        restored.cb(t, n_boot=5)
