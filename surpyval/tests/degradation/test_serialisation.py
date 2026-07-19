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

from surpyval.degradation import (
    DegradationAnalysis,
    DegradationModel,
    GammaProcess,
    GammaProcessModel,
    InducedFailureDistribution,
    WienerProcess,
    WienerProcessModel,
)


def _rt(d):
    return json.loads(json.dumps(d))


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


def _linear_deg_data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.tile(np.arange(100, 1100, 100), 4).astype(float)
    slopes = np.repeat([0.31, 0.28, 0.44, 0.37], 10)
    i = np.repeat([1, 2, 3, 4], 10)
    y = 10 + slopes * x + rng.normal(0, 1, x.size)
    return x, y, i


# -- process models -------------------------------------------------------


def test_gamma_process_round_trip():
    xg, yg, ig = _monotone_process_data()
    model = GammaProcess.fit(xg, yg, ig, threshold=15.0)
    restored = GammaProcessModel.from_dict(_rt(model.to_dict()))
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
    x, y, i = _linear_deg_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(x, y, i, threshold=150)
    induced = model.induced_life(n_samples=5000, random_state=3)
    restored = InducedFailureDistribution.from_dict(_rt(induced.to_dict()))
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
    x, y, i = _linear_deg_data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = DegradationAnalysis.fit(x, y, i, threshold=150)
    restored = DegradationModel.from_dict(_rt(model.to_dict()))
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
    x, y, i = _linear_deg_data(seed=2)
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
    restored = DegradationModel.from_dict(_rt(model.to_dict()))
    assert restored.is_accelerated
    t = np.array([10.0, 20.0, 30.0])
    for stress in ([0.0], [0.5], [1.0]):
        assert np.allclose(model.sf(t, Z=stress), restored.sf(t, Z=stress))


def test_degradation_model_rejects_wrong_dict():
    with pytest.raises(ValueError, match="DegradationModel"):
        DegradationModel.from_dict({"model": "Other"})
