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
import pandas as pd
import pytest

import surpyval
from surpyval import MixtureModel, Weibull
from surpyval.serialisation import required_schema
from surpyval.tests._helpers import json_round_trip
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
    ParametricCompetingRisks,
)
from surpyval.univariate.competing_risks.regression.fine_gray import (
    FineGrayModel,
)


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
    x = np.concatenate([Weibull.random(80, 10, 3), Weibull.random(80, 50, 4)])
    model = MixtureModel(dist=Weibull, m=2)
    model.fit(x=x)
    restored = MixtureModel.from_dict(json_round_trip(model.to_dict()))
    t = np.array([5.0, 20.0, 50.0])
    assert np.allclose(model.sf(t), restored.sf(t))
    assert np.allclose(model.ff(t), restored.ff(t))
    assert np.allclose(model.df(t), restored.df(t))
    assert np.isclose(model.mean(), restored.mean())
    assert np.allclose(model.params, restored.params)
    assert np.allclose(model.w, restored.w)


def test_mixture_model_json_file(tmp_path):
    x = np.concatenate([Weibull.random(60, 8, 3), Weibull.random(60, 40, 5)])
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
            {
                "model": "MixtureModel",
                "dist": "os",
                "m": 2,
                "params": [[1.0, 1.0]],
                "w": [1.0],
            }
        )


# -- FineGrayModel --------------------------------------------------------


def test_fine_gray_round_trip():
    x, Z, e, c = _cr_data()
    model = FineGray.fit(x, Z, e, c=c, event=1)
    restored = FineGrayModel.from_dict(json_round_trip(model.to_dict()))
    t = np.array([2.0, 5.0, 10.0])
    Zq = np.array([0.3, -0.2])
    assert np.allclose(model.cif(t, Zq), restored.cif(t, Zq))
    assert np.allclose(model.sf(t, Zq), restored.sf(t, Zq))
    assert np.allclose(model.beta, restored.beta)
    assert restored.cause == model.cause


def test_fine_gray_json_file(tmp_path):
    x, Z, e, c = _cr_data(seed=2)
    model = FineGray.fit(x, Z, e, c=c, event=1)
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
    restored = ParametricCompetingRisks.from_dict(
        json_round_trip(model.to_dict())
    )
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
    restored = CompetingRisks.from_dict(json_round_trip(model.to_dict()))
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


# ---------------------------------------------------------------------------
# ``CompetingRisksProportionalHazards`` round-trips through
# ``to_dict`` / JSON, with numpy-integer causes and formula
# metadata, and its dict is BSON-native.
# ---------------------------------------------------------------------------


def _cr_data_crph(seed=0, n=150):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    t_a = rng.exponential(1 / (0.1 * np.exp(Z @ [0.7, 0.2])))
    t_b = rng.exponential(1 / (0.05 * np.exp(Z @ [-0.3, 0.1])))
    t_c = rng.uniform(0, 20, n)
    x = np.minimum.reduce([t_a, t_b, t_c]).round(2)
    e = np.where(
        t_c < np.minimum(t_a, t_b), None, np.where(t_a < t_b, "a", "b")
    )
    return x, Z, e


T = np.array([0.0, 0.5, 2.0, 5.0, 10.0, 30.0])
ZS = [[0.0, 0.0], [1.0, -2.0], [3.0, 3.0]]


def _assert_same_predictions(model, restored):
    for z in ZS:
        for ev in ["a", "b"]:
            for f in ["cif", "sf", "ff", "Hf"]:
                np.testing.assert_array_equal(
                    getattr(model, f)(T, z, ev), getattr(restored, f)(T, z, ev)
                )
            if model.model == "Cox":
                for f in ["hf", "df"]:
                    np.testing.assert_array_equal(
                        getattr(model, f)(T, z, ev),
                        getattr(restored, f)(T, z, ev),
                    )
        if model.model == "Cox":
            for f in ["sf", "Hf", "hf"]:
                np.testing.assert_array_equal(
                    getattr(model, f)(T, z), getattr(restored, f)(T, z)
                )
    np.testing.assert_array_equal(model.betas, restored.betas)
    np.testing.assert_array_equal(model.beta, restored.beta)
    assert restored.event_idx_map == model.event_idx_map


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_round_trips_through_json(how):
    x, Z, e = _cr_data_crph()
    model = CompetingRisksProportionalHazards.fit(x, Z, e, model=how)
    d = model.to_dict()
    assert d["model"] == "CompetingRisksProportionalHazards"
    assert d["schema"] == required_schema(d)
    restored = surpyval.from_dict(json.loads(json.dumps(d)))
    assert type(restored) is CompetingRisksProportionalHazards
    assert restored.model == how
    assert restored.results is None
    _assert_same_predictions(model, restored)
    # class-level reader too
    again = CompetingRisksProportionalHazards.from_dict(d)
    _assert_same_predictions(model, again)


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_to_json_file(tmp_path, how):
    x, Z, e = _cr_data_crph(seed=1)
    model = CompetingRisksProportionalHazards.fit(x, Z, e, model=how)
    path = tmp_path / "crph.json"
    model.to_json(path)
    _assert_same_predictions(model, surpyval.from_json(path))
    _assert_same_predictions(
        model, CompetingRisksProportionalHazards.from_json(path)
    )


def test_crph_numpy_integer_causes_round_trip():
    # numpy-integer cause labels (as np.where produces) must serialise to
    # JSON, including inside the per-cause Fine-Gray models
    x, Z, e = _cr_data_crph(seed=2)
    codes = np.array(
        [None if v is None else np.int64(1 if v == "a" else 2) for v in e],
        dtype=object,
    )
    for how in ["Cox", "Fine-Gray"]:
        model = CompetingRisksProportionalHazards.fit(x, Z, codes, model=how)
        restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
        for ev in [1, 2]:
            np.testing.assert_array_equal(
                model.cif(T, [0.5, -1.0], ev), restored.cif(T, [0.5, -1.0], ev)
            )


def test_crph_formula_metadata_round_trips():
    x, Z, e = _cr_data_crph(seed=3)
    frame = pd.DataFrame({"t": x, "cause": e, "z1": Z[:, 0], "z2": Z[:, 1]})
    model = CompetingRisksProportionalHazards.fit_from_df(
        frame, x_col="t", e_col="cause", formula="z1 + z2"
    )
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.feature_names == model.feature_names == ["z1", "z2"]
    assert restored.formula == str(model.formula)
    _assert_same_predictions(model, restored)


def test_crph_from_dict_rejects_other_models():
    x, Z, e = _cr_data_crph()
    fg = surpyval.univariate.competing_risks.FineGray.fit(x, Z, e, event="a")
    with pytest.raises(ValueError, match="CompetingRisksProportionalHazards"):
        CompetingRisksProportionalHazards.from_dict(fg.to_dict())


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_dict_is_bson_native(how):
    # MongoDB's encoder rejects numpy scalars that json.dumps tolerates
    bson = pytest.importorskip("bson")
    x, Z, e = _cr_data_crph(seed=4)
    model = CompetingRisksProportionalHazards.fit(x, Z, e, model=how)
    doc = bson.decode(bson.encode(model.to_dict()))
    _assert_same_predictions(model, surpyval.from_dict(doc))
