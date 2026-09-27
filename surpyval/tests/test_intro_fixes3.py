"""Regression tests for the third review of package serialisation
(``surpyval.serialisation``) and the degenerate distributions."""

import json

import numpy as np
import pytest

import surpyval
from surpyval import KaplanMeier, Weibull
from surpyval.univariate.parametric import InstantlyOccurs, NeverOccurs


@pytest.fixture
def weibull_dict():
    return Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0]).to_dict()


@pytest.mark.parametrize("key", ["distribution", "params", "how"])
def test_missing_key_names_the_key(weibull_dict, key):
    # Used to be a bare KeyError from inside Parametric.from_dict.
    del weibull_dict[key]
    with pytest.raises(ValueError, match=f"no '{key}' entry"):
        surpyval.from_dict(weibull_dict)


def test_missing_key_in_a_tagged_model():
    Z = np.array([[0.0], [1.0], [0.0], [1.0], [1.0]])
    d = surpyval.CoxPH.fit([1, 2, 3, 4, 5.0], Z).to_dict()
    del d["beta"]
    with pytest.raises(ValueError, match="no 'beta' entry"):
        surpyval.from_dict(d)
    d = KaplanMeier.fit([1, 2, 3, 4]).to_dict()
    del d["model"]
    with pytest.raises(ValueError, match="no 'model' entry"):
        surpyval.from_dict(d)


@pytest.mark.parametrize("schema", ["2", 2.0, 1.0, "1", True, None, -1])
def test_schema_must_be_a_non_negative_integer(weibull_dict, schema):
    # "2" and 2.0 used to slip past the version check.
    weibull_dict["schema"] = schema
    with pytest.raises(ValueError, match="schema"):
        surpyval.from_dict(weibull_dict)


def test_integer_schemas_still_read(weibull_dict):
    for schema in (0, 1, np.int64(1)):
        weibull_dict["schema"] = schema
        surpyval.from_dict(weibull_dict)
    del weibull_dict["schema"]
    surpyval.from_dict(weibull_dict)


def test_newer_schema_with_unknown_model_asks_for_upgrade():
    with pytest.raises(ValueError, match="Upgrade surpyval"):
        surpyval.from_dict({"model": "FromTheFuture", "schema": 99})


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(params=[-5.0, 2.0]), "alpha=-5.0 .* outside its bounds"),
        (dict(params=[5.0, -2.0]), "beta=-2.0 .* outside its bounds"),
        (dict(params=[np.nan, 2.0]), "NaN"),
        (dict(lfp=True, p=1.5), "'p'=1.5 is a proportion"),
        (dict(zi=True, f0=-0.1), "'f0'=-0.1 is a proportion"),
    ],
)
def test_invalid_parameters_are_refused(weibull_dict, change, match):
    weibull_dict.update(change)
    with pytest.raises(ValueError, match=match):
        surpyval.from_dict(weibull_dict)


def test_class_from_json_gets_the_same_checks(tmp_path, weibull_dict):
    path = tmp_path / "w.json"
    weibull_dict["params"] = [-5.0, 2.0]
    path.write_text(json.dumps(weibull_dict))
    with pytest.raises(ValueError, match="outside its bounds"):
        Weibull.fit([1, 2, 3.0]).from_json(path)
    weibull_dict["params"] = [5.0, 2.0]
    weibull_dict["schema"] = "1"
    path.write_text(json.dumps(weibull_dict))
    with pytest.raises(ValueError, match="schema"):
        Weibull.fit([1, 2, 3.0]).from_json(path)


@pytest.mark.parametrize("dist", [NeverOccurs, InstantlyOccurs])
def test_degenerate_to_json_round_trip(tmp_path, dist):
    path = tmp_path / "d.json"
    dist.to_json(path)
    assert surpyval.from_json(path) is dist
    assert dist.from_json(path) is dist
    assert surpyval.from_dict(dist.to_dict()) is dist


def test_degenerate_from_dict_checks_the_tag(tmp_path):
    with pytest.raises(ValueError, match="InstantlyOccurs"):
        InstantlyOccurs.from_dict(NeverOccurs.to_dict())
    path = tmp_path / "n.json"
    NeverOccurs.to_json(path)
    with pytest.raises(ValueError, match="InstantlyOccurs"):
        InstantlyOccurs.from_json(path)
