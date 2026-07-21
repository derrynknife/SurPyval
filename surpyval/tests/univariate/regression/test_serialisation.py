"""Serialisation of the parametric regression models.

The fixed-form regression families -- Accelerated Failure Time, Proportional
Hazards, Proportional Odds and (parametric) Additive Hazards -- round-trip
through ``to_dict``/``from_dict`` (and the JSON file variants): the restored
model predicts identically to the original, its parameter names line up, and
-- when a covariance was stored -- it reproduces the same confidence bounds
without needing the original data. Accelerated Life parameter-substitution
models built on a built-in life model (Power, Eyring, ...) also round-trip,
rebuilt from the distribution and life-model names. Only a genuinely custom
life model whose parameterisation is not a fixed name map is refused, and an
untrusted dict cannot resolve an arbitrary distribution.
"""

import json

import numpy as np
import pytest

from surpyval import (
    AFT,
    AH,
    PH,
    PO,
    AcceleratedLife,
    Eyring,
    Power,
    Weibull,
)
from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)


def _fit_data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    lin = 0.5 * Z[:, 0] - 0.3 * Z[:, 1]
    x = np.abs(Weibull.random(n, 10.0, 2.0) * np.exp(-lin / 2.0)) + 0.1
    c = np.zeros(n)
    return x, Z, c


FAMILIES = {
    "AFT": AFT,
    "PH": PH,
    "PO": PO,
    "AH": AH,
}


@pytest.mark.parametrize("name", list(FAMILIES))
def test_predictions_round_trip(name):
    x, Z, c = _fit_data()
    model = FAMILIES[name](Weibull).fit(x, Z=Z, c=c)
    restored = ParametricRegressionModel.from_dict(model.to_dict())

    xs = np.array([2.0, 5.0, 10.0])
    Zq = np.array([0.4, -0.2])
    for fn in ("sf", "ff", "df", "hf", "Hf"):
        a = np.asarray(getattr(model, fn)(xs, Zq), dtype=float)
        b = np.asarray(getattr(restored, fn)(xs, Zq), dtype=float)
        assert np.allclose(a, b, rtol=1e-10, atol=1e-12), fn
    assert np.allclose(model.params, restored.params)
    assert model.parameter_names() == restored.parameter_names()
    assert model.kind == restored.kind
    assert model.distribution.name == restored.distribution.name


@pytest.mark.parametrize("name", list(FAMILIES))
def test_confidence_bounds_round_trip(name):
    x, Z, c = _fit_data()
    model = FAMILIES[name](Weibull).fit(x, Z=Z, c=c)
    d = model.to_dict()
    # a well-behaved fit stores a finite covariance
    assert "covariance" in d
    restored = ParametricRegressionModel.from_dict(d)

    xs = np.array([2.0, 5.0, 10.0])
    Zq = np.array([0.4, -0.2])
    cb1 = np.asarray(model.cb(xs, Zq, on="sf"), dtype=float)
    cb2 = np.asarray(restored.cb(xs, Zq, on="sf"), dtype=float)
    assert np.allclose(cb1, cb2, rtol=1e-6, atol=1e-8)
    se1 = model.standard_errors()
    se2 = restored.standard_errors()
    assert np.allclose(se1, se2, rtol=1e-6, atol=1e-8, equal_nan=True)
    pc1 = model.param_cb(model.parameter_names()[-1])
    pc2 = restored.param_cb(restored.parameter_names()[-1])
    assert np.allclose(pc1, pc2, rtol=1e-6, atol=1e-8)


def test_json_file_round_trip(tmp_path):
    x, Z, c = _fit_data()
    model = AFT(Weibull).fit(x, Z=Z, c=c)
    fp = tmp_path / "model.json"
    model.to_json(fp)
    restored = ParametricRegressionModel.from_json(fp)
    xs = np.array([2.0, 5.0, 10.0])
    Zq = np.array([0.4, -0.2])
    assert np.allclose(
        np.asarray(model.sf(xs, Zq), dtype=float),
        np.asarray(restored.sf(xs, Zq), dtype=float),
    )
    # the dict is genuinely JSON-serialisable
    with open(fp) as f:
        assert json.load(f)["parameterization"] == "parametric-regression"


def test_to_dict_is_json_serialisable():
    x, Z, c = _fit_data()
    model = PH(Weibull).fit(x, Z=Z, c=c)
    # must not raise
    json.dumps(model.to_dict())


def test_phi_round_trip():
    x, Z, c = _fit_data()
    model = AFT(Weibull).fit(x, Z=Z, c=c)
    restored = ParametricRegressionModel.from_dict(model.to_dict())
    Zq = np.array([0.4, -0.2])
    assert np.allclose(
        np.asarray(model.phi(Zq), dtype=float),
        np.asarray(restored.phi(Zq), dtype=float),
    )


def _al_data(seed=1, per=40, stresses=(1.0, 2.0, 3.0, 4.0)):
    rng = np.random.default_rng(seed)
    xs, Zs = [], []
    for s in stresses:
        life = 500.0 * s**-1.2
        xs.append(life * rng.weibull(2.2, per))
        Zs.append(np.full(per, s))
    x = np.concatenate(xs)
    Z = np.concatenate(Zs).reshape(-1, 1)
    return x, Z, np.zeros_like(x)


@pytest.mark.parametrize("life_model", [Power, Eyring])
def test_accelerated_life_round_trips(life_model):
    x, Z, c = _al_data()
    model = AcceleratedLife(Weibull, life_model).fit(x, Z=Z, c=c)

    d = model.to_dict()
    assert d["kind"] == "Accelerated Life"
    assert d["life_model_name"] == life_model.name
    json.dumps(d)  # JSON-safe

    restored = ParametricRegressionModel.from_dict(d)
    assert np.allclose(model.params, restored.params)

    xq = np.linspace(1.0, 400.0, 12)
    for z in (1.5, 2.5, 3.5):
        Zq = np.full((xq.size, 1), z)
        assert np.allclose(
            model.sf(xq, Zq), restored.sf(xq, Zq), equal_nan=True
        )
    Zphi = np.array([[1.5], [2.5], [3.5]])
    assert np.allclose(
        np.asarray(model.phi(Zphi), dtype=float),
        np.asarray(restored.phi(Zphi), dtype=float),
    )


def test_accelerated_life_dispatches_through_package():
    import surpyval

    x, Z, c = _al_data()
    model = AcceleratedLife(Weibull, Power).fit(x, Z=Z, c=c)
    restored = surpyval.from_dict(model.to_dict())
    assert isinstance(restored, ParametricRegressionModel)
    assert restored.kind == "Accelerated Life"


def test_custom_life_model_is_refused():
    # A user-defined life model is not in the built-in registry and its
    # parameterisation cannot be rebuilt from a name, so it is refused.
    class _CustomLife(LifeModel):
        def __init__(self):
            super().__init__(
                "MyCustomLink", {"a": 0, "b": 1}, ((0, None),) * 2
            )

        def phi(self, Z, *params):
            return params[0] * Z ** params[1]

        def phi_init(self, life, Z):
            return [1.0, 1.0]

    x, Z, c = _al_data()
    model = AcceleratedLife(Weibull, _CustomLife()).fit(x, Z=Z, c=c)
    with pytest.raises(NotImplementedError, match="built-in life model"):
        model.to_dict()


def test_from_dict_rejects_unknown_distribution():
    x, Z, c = _fit_data()
    d = AFT(Weibull).fit(x, Z=Z, c=c).to_dict()
    d["distribution"] = "os"  # a real surpyval-importable name, not a dist
    with pytest.raises(ValueError, match="Unknown distribution"):
        ParametricRegressionModel.from_dict(d)


def test_from_dict_rejects_wrong_parameterization():
    with pytest.raises(ValueError, match="parametric-regression"):
        ParametricRegressionModel.from_dict({"parameterization": "parametric"})


def test_restored_without_covariance_refuses_bounds():
    x, Z, c = _fit_data()
    d = AFT(Weibull).fit(x, Z=Z, c=c).to_dict()
    d.pop("covariance", None)  # simulate a predictions-only dict
    restored = ParametricRegressionModel.from_dict(d)
    # predictions still work
    sf = np.asarray(restored.sf(5.0, [0.1, 0.1]), dtype=float)
    assert np.isfinite(sf).all()
    # but bounds are unavailable without the covariance or data
    with pytest.raises(ValueError):
        restored.cb(5.0, [0.1, 0.1])


def test_fixed_and_feature_names_preserved():
    x, Z, c = _fit_data()
    model = AFT(Weibull).fit(x, Z=Z, c=c, fixed={"alpha": 10.0})
    restored = ParametricRegressionModel.from_dict(model.to_dict())
    assert restored.fixed == {"alpha": 10.0}
    # a fixed parameter is held at its value
    assert restored.params[0] == pytest.approx(10.0)


def test_dataframe_fit_metadata_preserved():
    import pandas as pd

    x, Z, c = _fit_data()
    df = pd.DataFrame({"x": x, "c": c, "z0": Z[:, 0], "z1": Z[:, 1]})
    model = AFT(Weibull).fit_from_df(
        df, x_col="x", c_col="c", Z_cols=["z0", "z1"]
    )
    restored = ParametricRegressionModel.from_dict(model.to_dict())
    assert restored.feature_names == ["z0", "z1"]
    # array-Z prediction matches the original
    xs = np.array([2.0, 5.0])
    Zq = np.array([0.4, -0.2])
    assert np.allclose(
        np.asarray(model.sf(xs, Zq), dtype=float),
        np.asarray(restored.sf(xs, Zq), dtype=float),
    )
