"""Serialisation of the fitted recurrent-event models.

The recurrence result classes -- the parametric intensity fit
(``ParametricRecurrenceModel``), the nonparametric MCF
(``NonParametricCounting``), the proportional-intensity regression
(``ProportionalIntensityModel``), and the competing-risks containers
(``CauseSpecificMCF``, ``CauseSpecificNHPP``) -- round-trip through
``to_dict``/``from_dict`` (and the JSON file variants). The intensity model is
stateless, so each stores its name plus the fitted parameters (or, for the
MCF, the step arrays), and the reloaded model reproduces every prediction
exactly.
"""

import json

import numpy as np
import pytest

from surpyval.recurrent import (
    CrowAMSAA,
    Duane,
    NonParametricCounting,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)
from surpyval.recurrent.competing_risks import (
    CauseSpecificMCF,
    CauseSpecificNHPP,
)
from surpyval.recurrent.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)


def _rt(d):
    """JSON round-trip a dict (proves it is JSON-serialisable)."""
    return json.loads(json.dumps(d))


def _pi_data(seed=0, n_items=25):
    rng = np.random.default_rng(seed)
    xs, ii, cc, ZZ = [], [], [], []
    for item in range(n_items):
        z = rng.normal(0, 1)
        for _ in range(int(rng.integers(2, 6))):
            xs.append(float(rng.uniform(0, 500)))
            ii.append(item)
            cc.append(0)
            ZZ.append([z])
        xs.append(500.0)
        ii.append(item)
        cc.append(1)
        ZZ.append([z])
    return np.array(xs), np.array(ii), np.array(cc), np.array(ZZ)


# -- ParametricRecurrenceModel --------------------------------------------


@pytest.mark.parametrize("fitter", [CrowAMSAA, Duane])
def test_parametric_recurrence_round_trip(fitter):
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1000, 40))
    model = fitter.fit(x)
    restored = ParametricRecurrenceModel.from_dict(_rt(model.to_dict()))
    xt = np.array([100.0, 500.0, 900.0])
    assert np.allclose(model.cif(xt), restored.cif(xt))
    assert np.allclose(model.iif(xt), restored.iif(xt))
    assert np.allclose(model.params, restored.params)
    assert model.dist.name == restored.dist.name


def test_parametric_recurrence_json_file(tmp_path):
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0, 1000, 30))
    model = CrowAMSAA.fit(x)
    fp = tmp_path / "nhpp.json"
    model.to_json(fp)
    restored = ParametricRecurrenceModel.from_json(fp)
    xt = np.array([200.0, 800.0])
    assert np.allclose(model.cif(xt), restored.cif(xt))


def test_parametric_recurrence_rejects_wrong_model():
    with pytest.raises(ValueError, match="ParametricRecurrenceModel"):
        ParametricRecurrenceModel.from_dict({"model": "Other"})


def test_parametric_recurrence_rejects_unknown_dist():
    d = {"model": "ParametricRecurrenceModel", "dist": "Nope", "params": [1.0]}
    with pytest.raises(ValueError, match="Unknown recurrence intensity"):
        ParametricRecurrenceModel.from_dict(d)


# -- NonParametricCounting (MCF) ------------------------------------------


def test_mcf_round_trip():
    x = np.array([5, 10, 15, 4, 9, 12, 20], dtype=float)
    i = np.array([1, 1, 1, 2, 2, 3, 3])
    c = np.array([0, 0, 1, 0, 1, 0, 1])
    model = NonParametricCounting.fit(x, i, c)
    restored = NonParametricCounting.from_dict(_rt(model.to_dict()))
    grid = np.array([6.0, 11.0, 16.0])
    assert np.allclose(model.mcf(grid), restored.mcf(grid), equal_nan=True)
    assert np.allclose(model.x, restored.x)
    assert np.allclose(model.var, restored.var)


def test_mcf_json_file(tmp_path):
    x = np.array([5, 10, 15, 4, 9], dtype=float)
    i = np.array([1, 1, 1, 2, 2])
    c = np.array([0, 0, 1, 0, 1])
    model = NonParametricCounting.fit(x, i, c)
    fp = tmp_path / "mcf.json"
    model.to_json(fp)
    restored = NonParametricCounting.from_json(fp)
    grid = np.array([6.0, 11.0])
    assert np.allclose(model.mcf(grid), restored.mcf(grid), equal_nan=True)


# -- ProportionalIntensityModel -------------------------------------------


def test_proportional_intensity_nhpp_round_trip():
    x, i, c, Z = _pi_data(seed=3)
    model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=Duane)
    restored = type(model).from_dict(_rt(model.to_dict()))
    xt = np.array([100.0, 300.0, 450.0])
    Zq = np.array([0.5])
    assert np.allclose(model.cif(xt, Zq), restored.cif(xt, Zq))
    assert np.allclose(model.iif(xt, Zq), restored.iif(xt, Zq))
    assert np.allclose(model.params, restored.params)
    assert np.allclose(model.coeffs, restored.coeffs)


def test_proportional_intensity_hpp_round_trip():
    x, i, c, Z = _pi_data(seed=4)
    model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    assert model.kind == "HPP"
    restored = type(model).from_dict(_rt(model.to_dict()))
    xt = np.array([100.0, 300.0])
    Zq = np.array([0.5])
    assert np.allclose(model.cif(xt, Zq), restored.cif(xt, Zq))
    assert np.allclose(model.iif(xt, Zq), restored.iif(xt, Zq))


def test_proportional_intensity_rejects_wrong_model():
    with pytest.raises(ValueError, match="ProportionalIntensityModel"):
        from surpyval.recurrent.regression.proportional_intensity import (
            ProportionalIntensityModel,
        )

        ProportionalIntensityModel.from_dict({"model": "Other"})


# -- Cause-specific containers --------------------------------------------


def test_cause_specific_mcf_round_trip():
    x = np.array([5, 10, 15, 4, 9, 12, 20], dtype=float)
    i = np.array([1, 1, 1, 2, 2, 3, 3])
    c = np.array([0, 0, 1, 0, 1, 0, 1])
    e = np.array(["A", "B", "A", "A", "B", "B", "A"])
    model = CauseSpecificMCF.fit(x, i, c, e=e)
    restored = CauseSpecificMCF.from_dict(_rt(model.to_dict()))
    assert restored.event_types == model.event_types
    grid = np.array([6.0, 11.0, 16.0])
    for cause in model.event_types:
        assert np.allclose(
            model.mcf(grid, cause),
            restored.mcf(grid, cause),
            equal_nan=True,
        )


def test_cause_specific_nhpp_round_trip():
    rng = np.random.default_rng(5)
    xs, ii, cc, es = [], [], [], []
    for item in range(20):
        for _ in range(int(rng.integers(2, 5))):
            xs.append(float(rng.uniform(0, 400)))
            ii.append(item)
            cc.append(0)
            es.append(rng.choice(["A", "B"]))
        xs.append(400.0)
        ii.append(item)
        cc.append(1)
        es.append(None)
    model = CauseSpecificNHPP.fit(
        np.array(xs), np.array(ii), np.array(cc), e=np.array(es, dtype=object)
    )
    restored = CauseSpecificNHPP.from_dict(_rt(model.to_dict()))
    assert restored.dist.name == model.dist.name
    assert restored.event_types == model.event_types
    xt = np.array([50.0, 200.0, 350.0])
    for cause in model.event_types:
        assert np.allclose(model.cif(xt, cause), restored.cif(xt, cause))


def test_cause_specific_nhpp_json_file(tmp_path):
    rng = np.random.default_rng(6)
    xs, ii, cc, es = [], [], [], []
    for item in range(15):
        for _ in range(int(rng.integers(2, 5))):
            xs.append(float(rng.uniform(0, 400)))
            ii.append(item)
            cc.append(0)
            es.append(rng.choice(["A", "B"]))
        xs.append(400.0)
        ii.append(item)
        cc.append(1)
        es.append(None)
    model = CauseSpecificNHPP.fit(
        np.array(xs), np.array(ii), np.array(cc), e=np.array(es, dtype=object)
    )
    fp = tmp_path / "csnhpp.json"
    model.to_json(fp)
    restored = CauseSpecificNHPP.from_json(fp)
    xt = np.array([100.0, 300.0])
    for cause in model.event_types:
        assert np.allclose(model.cif(xt, cause), restored.cif(xt, cause))
