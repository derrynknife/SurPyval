"""Serialisation of the competing-risks and mixture models.

``MixtureModel`` (an EM mixture of a base family), ``FineGrayModel`` (the
subdistribution-hazard regression), ``ParametricCompetingRisks`` (one
distribution per cause) and the nonparametric ``CompetingRisks`` all round-trip
through ``to_dict``/``from_dict`` (and the JSON file variants): each stores its
fitted parameters (or step arrays / per-cause sub-models) and the reloaded
model reproduces its predictions exactly.
"""

import json

import numpy as np
import pytest

from surpyval import MixtureModel, Weibull
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    FineGray,
    ParametricCompetingRisks,
)
from surpyval.univariate.competing_risks.regression.fine_gray import (
    FineGrayModel,
)


def _rt(d):
    return json.loads(json.dumps(d))


def _cr_data(seed=0, n=150):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    x = np.abs(rng.weibull(1.3, n) * 15) + 0.2
    e = rng.choice([1, 2], n)
    c = (rng.random(n) < 0.2).astype(int)
    e = np.where(c == 1, None, e)
    return x, Z, e, c


# -- MixtureModel ---------------------------------------------------------


def test_mixture_model_round_trip():
    x = np.concatenate(
        [Weibull.random(80, 10, 3), Weibull.random(80, 50, 4)]
    )
    model = MixtureModel(dist=Weibull, m=2)
    model.fit(x=x)
    restored = MixtureModel.from_dict(_rt(model.to_dict()))
    t = np.array([5.0, 20.0, 50.0])
    assert np.allclose(model.sf(t), restored.sf(t))
    assert np.allclose(model.ff(t), restored.ff(t))
    assert np.allclose(model.df(t), restored.df(t))
    assert np.isclose(model.mean(), restored.mean())
    assert np.allclose(model.params, restored.params)
    assert np.allclose(model.w, restored.w)


def test_mixture_model_json_file(tmp_path):
    x = np.concatenate(
        [Weibull.random(60, 8, 3), Weibull.random(60, 40, 5)]
    )
    model = MixtureModel(dist=Weibull, m=2)
    model.fit(x=x)
    fp = tmp_path / "mix.json"
    model.to_json(fp)
    restored = MixtureModel.from_json(fp)
    t = np.array([5.0, 20.0])
    assert np.allclose(model.sf(t), restored.sf(t))


def test_mixture_model_guards():
    with pytest.raises(ValueError, match="MixtureModel"):
        MixtureModel.from_dict({"model": "Other"})
    with pytest.raises(ValueError, match="Unknown distribution"):
        MixtureModel.from_dict(
            {"model": "MixtureModel", "dist": "os", "m": 2,
             "params": [[1.0, 1.0]], "w": [1.0]}
        )


# -- FineGrayModel --------------------------------------------------------


def test_fine_gray_round_trip():
    x, Z, e, c = _cr_data()
    model = FineGray.fit(x, Z, e, c=c, cause=1)
    restored = FineGrayModel.from_dict(_rt(model.to_dict()))
    t = np.array([2.0, 5.0, 10.0])
    Zq = np.array([0.3, -0.2])
    assert np.allclose(model.cif(t, Zq), restored.cif(t, Zq))
    assert np.allclose(model.sf(t, Zq), restored.sf(t, Zq))
    assert np.allclose(model.beta, restored.beta)
    assert restored.cause == model.cause


def test_fine_gray_json_file(tmp_path):
    x, Z, e, c = _cr_data(seed=2)
    model = FineGray.fit(x, Z, e, c=c, cause=1)
    fp = tmp_path / "fg.json"
    model.to_json(fp)
    restored = FineGrayModel.from_json(fp)
    t = np.array([3.0, 7.0])
    Zq = np.array([0.1, 0.1])
    assert np.allclose(model.cif(t, Zq), restored.cif(t, Zq))


def test_fine_gray_guard():
    with pytest.raises(ValueError, match="FineGrayModel"):
        FineGrayModel.from_dict({"model": "Other"})


# -- ParametricCompetingRisks ---------------------------------------------


def test_parametric_competing_risks_round_trip():
    x, _, e, c = _cr_data(seed=3)
    model = ParametricCompetingRisks.fit(x, e, c=c)
    restored = ParametricCompetingRisks.from_dict(_rt(model.to_dict()))
    assert restored.causes == model.causes
    t = np.array([2.0, 5.0, 10.0])
    for cause in model.causes:
        assert np.allclose(
            model.Hf(t, event=cause), restored.Hf(t, event=cause)
        )
        assert np.allclose(
            model.hf(t, event=cause), restored.hf(t, event=cause)
        )


def test_parametric_competing_risks_guard():
    with pytest.raises(ValueError, match="ParametricCompetingRisks"):
        ParametricCompetingRisks.from_dict({"model": "Other"})


# -- nonparametric CompetingRisks -----------------------------------------


def test_competing_risks_round_trip():
    x, _, e, c = _cr_data(seed=4)
    model = CompetingRisks.fit(x, e, c=c)
    restored = CompetingRisks.from_dict(_rt(model.to_dict()))
    assert list(restored.event_idx_map) == list(model.event_idx_map)
    t = np.array([2.0, 5.0, 10.0])
    for event in model.event_idx_map:
        assert np.allclose(model.cif(t, event), restored.cif(t, event))
        assert np.allclose(model.sf(t, event), restored.sf(t, event))


def test_competing_risks_json_file(tmp_path):
    x, _, e, c = _cr_data(seed=5)
    model = CompetingRisks.fit(x, e, c=c)
    fp = tmp_path / "cr.json"
    model.to_json(fp)
    restored = CompetingRisks.from_json(fp)
    t = np.array([3.0, 8.0])
    event = list(model.event_idx_map)[0]
    assert np.allclose(model.cif(t, event), restored.cif(t, event))


def test_competing_risks_guard():
    with pytest.raises(ValueError, match="CompetingRisks"):
        CompetingRisks.from_dict({"model": "Other"})
