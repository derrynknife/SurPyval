"""Regression tests for the second round of degradation bug fixes.

* Every built-in path model round-trips through ``DegradationModel``
  serialisation (the offset-exponential model used to fail to reload,
  because its display name "Offset Exponential" was stored and only the
  registry key "offset-exponential" resolved).
* ``DestructiveDegradationModel`` is a full serialisable model: it has
  ``to_json`` / ``from_json``, dispatches through ``surpyval.from_dict``,
  and stores its fit data so a reloaded model's bootstrap ``cb`` works.
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
    get_path_model,
)
from surpyval.degradation.path_models import path_model_key


def _rt(d: dict) -> dict:
    return json.loads(json.dumps(d))


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
    d = _rt(model.to_dict())
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
    d = _rt(model.to_dict())
    d["path_model"] = "Offset Exponential"
    restored = DegradationModel.from_dict(d)
    assert restored.path_model is model.path_model
    grid = np.array([10.0, 20.0, 30.0])
    assert np.allclose(model.sf(grid), restored.sf(grid))


def test_get_path_model_accepts_display_names():
    for key, model in PATH_MODELS.items():
        assert get_path_model(model.name) is model
        assert get_path_model(model.name.upper()) is model
        assert path_model_key(model) == key
    with pytest.raises(ValueError):
        get_path_model("Offset  Exponential")


# -- destructive degradation --------------------------------------------------


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
    restored = DestructiveDegradationModel.from_dict(_rt(model.to_dict()))
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
        model.cb(t, n_boot=20, seed=3), restored.cb(t, n_boot=20, seed=3)
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
            model.cb(t, n_boot=10, seed=0), restored.cb(t, n_boot=10, seed=0)
        )
    assert model.to_dict()["schema"] == surpyval.serialisation.SCHEMA_VERSION


def test_destructive_old_dict_without_data_still_loads():
    model = _destructive_model()
    d = _rt(model.to_dict())
    for key in ("data", "neg_ll", "transform_scores"):
        del d[key]
    restored = DestructiveDegradationModel.from_dict(d)
    t = np.array([30.0, 50.0])
    assert np.allclose(model.sf(t), restored.sf(t))
    with pytest.raises(ValueError, match="fit data"):
        restored.cb(t, n_boot=5)
