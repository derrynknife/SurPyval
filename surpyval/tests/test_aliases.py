"""The other names: ``Gauss`` for ``Normal``, ``Galton`` for ``LogNormal``
and ``life_models.Arrhenius`` for ``life_models.Exponential``.

A model fitted with an alias carries the alias (its repr, ``to_dict``, a
JSON or pickle round trip) and is otherwise identical to one fitted with
the canonical name: the same numbers to the last bit, and the same
answer wherever the package asks which distribution or life model a
model has.
"""

import json
import pickle
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import life_models
from surpyval.utils.aliases import (
    DISTRIBUTION_ALIASES,
    LIFE_MODEL_ALIASES,
    canonical_life_model_name,
    canonical_name,
)
from surpyval.utils.no_maximum import (
    NoFiniteMaximumError,
)

X = [1.0, 2.0, 3.0, 4.0, 5.0, 6.5]
# A temperature test, in kelvin, and the lives at each temperature
T = [300.0, 300, 300, 350, 350, 350, 400, 400, 400]
LIFE = [2900.0, 4100, 5200, 1500, 2100, 2600, 800, 1100, 1400]

PAIRS = [(sp.Gauss, sp.Normal), (sp.Galton, sp.LogNormal)]
IDS = ["Gauss", "Galton"]

#: Dicts written before the aliases round-tripped by name (v0.26, from
#: ``dev`` at b250dc48): an alias fit wrote the alias's name, and a model
#: built with ``life_models.Arrhenius`` (then ``Exponential`` itself)
#: wrote ``"Exponential"``. ``AcceleratedLife(Galton, ...)`` was refused
#: then, so no dict of one exists.
OLD_GAUSS = {
    "parameterization": "parametric",
    "distribution": "Gauss",
    "how": "MLE",
    "param_names": ["mu", "sigma"],
    "params": [3.5833333333333335, 1.8352262954621035],
    "lfp": False,
    "p": 1.0,
    "zi": False,
    "f0": 0.0,
    "offset": False,
    "gamma": 0.0,
    "hess_inv": [
        [0.5613425925925928, 3.4983738823443476e-17],
        [3.4983738823443476e-17, 0.2806712962962964],
    ],
    "covariance": [
        [0.5613425925925928, 3.4983738823443476e-17],
        [3.4983738823443476e-17, 0.2806712962962964],
    ],
    "_neg_ll": 12.156637973312485,
    "maximum": "verified",
    "data_summary": "6 units: 6 events at 6 unique times",
    "ic_n": 6.0,
    "schema": 1,
}
OLD_GALTON = {
    "parameterization": "parametric",
    "distribution": "Galton",
    "how": "MLE",
    "param_names": ["mu", "sigma"],
    "params": [1.109882319947273, 0.6207677006535425],
    "lfp": False,
    "p": 1.0,
    "zi": False,
    "f0": 0.0,
    "offset": False,
    "gamma": 0.0,
    "hess_inv": [
        [0.06422542302911435, 1.831825785844596e-18],
        [5.495477357533788e-18, 0.03211271151455719],
    ],
    "covariance": [
        [0.06422542302911435, 1.831825785844596e-18],
        [5.495477357533788e-18, 0.03211271151455719],
    ],
    "_neg_ll": 12.312135078593105,
    "maximum": "verified",
    "data_summary": "6 units: 6 events at 6 unique times",
    "ic_n": 6.0,
    "schema": 1,
}


def _json(d):
    return json.loads(json.dumps(d, allow_nan=False))


def _quiet(f, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*args, **kwargs)


# -- the names ----------------------------------------------------------------


@pytest.mark.parametrize("alias, canon", PAIRS, ids=IDS)
def test_alias_keeps_its_name_and_has_the_canonical_one(alias, canon):
    assert alias.name in DISTRIBUTION_ALIASES
    assert alias.canonical_name == canon.name == canon.canonical_name
    assert type(alias) is type(canon)
    assert canonical_name(alias) == canonical_name(canon) == canon.name
    assert canonical_name(alias.name) == canon.name
    assert canonical_name(alias.name.lower()) == canon.name
    assert repr(alias) == f"{alias.name}: parametric fitter"


def test_other_distributions_are_their_own_canonical_name():
    for dist in (sp.Weibull, sp.Exponential, sp.Gamma, sp.Normal):
        assert dist.canonical_name == dist.name
        assert canonical_name(dist) == dist.name
    assert canonical_name("Weibull") == "Weibull"
    assert canonical_name(object()) is None


def test_arrhenius_names():
    assert LIFE_MODEL_ALIASES == {"Arrhenius": "Exponential"}
    assert life_models.Arrhenius.name == "Arrhenius"
    assert canonical_life_model_name(life_models.Arrhenius) == "Exponential"
    assert canonical_life_model_name("Arrhenius") == "Exponential"
    assert canonical_life_model_name(life_models.Power) == "Power"
    assert life_models.Power.canonical_name == "Power"


# -- fitting: the same numbers ------------------------------------------------


@pytest.mark.parametrize("alias, canon", PAIRS, ids=IDS)
def test_fit_is_bit_identical(alias, canon):
    a, b = alias.fit(X), canon.fit(X)
    assert a.dist.name == alias.name
    np.testing.assert_array_equal(a.params, b.params)
    np.testing.assert_array_equal(a.covariance(), b.covariance())
    assert a.neg_ll() == b.neg_ll()
    assert a.aic() == b.aic()
    xs = np.array([0.5, 2.0, 4.0, 8.0])
    for fn in ("sf", "ff", "df", "hf", "Hf"):
        np.testing.assert_array_equal(getattr(a, fn)(xs), getattr(b, fn)(xs))
    np.testing.assert_array_equal(a.cb(xs), b.cb(xs))
    # Galton's restricted mean was a quadrature, not the closed form
    assert a.rmst(4.0) == b.rmst(4.0)


@pytest.mark.parametrize("alias, canon", PAIRS, ids=IDS)
def test_data_with_no_maximum_are_refused_as_for_the_canonical(alias, canon):
    # An exact failure at 1 observable only up to 1, one at 2 and one
    # known to be before 3: no finite maximum. The alias used to skip the
    # check and return a spike, marked unverified.
    data = dict(c=[0, 0, -1], tr=[1, np.inf, np.inf])
    for dist in (canon, alias):
        with pytest.raises(NoFiniteMaximumError, match=dist.name):
            _quiet(dist.fit, [1.0, 2, 3], **data)


@pytest.mark.parametrize("alias, canon", PAIRS, ids=IDS)
def test_fit_best_takes_the_alias_as_its_distribution(alias, canon):
    best = _quiet(sp.fit_best, X, include=[alias.name])
    assert best.dist is canon
    lower = _quiet(sp.fit_best, X, include=alias.name.lower())
    assert lower.dist is canon
    table = _quiet(sp.fit_best, X, exclude=[alias.name], return_table=True)
    assert canon.name not in set(table[1]["model"])


# -- saving and loading -------------------------------------------------------


@pytest.mark.parametrize("alias, canon", PAIRS, ids=IDS)
def test_json_round_trip_keeps_the_alias(alias, canon, tmp_path):
    model = alias.fit(X)
    d = model.to_dict()
    assert d["distribution"] == alias.name
    assert d == {**canon.fit(X).to_dict(), "distribution": alias.name}
    restored = sp.from_dict(_json(d))
    assert restored.dist is alias
    assert f"Distribution        : {alias.name}" in repr(restored)
    assert repr(restored) == repr(model)
    np.testing.assert_array_equal(restored.params, model.params)
    assert restored.to_dict() == d
    fp = tmp_path / "model.json"
    model.to_json(fp)
    assert sp.from_json(fp).dist is alias


@pytest.mark.parametrize("alias, canon", PAIRS, ids=IDS)
def test_pickle_round_trip_keeps_the_alias(alias, canon):
    model = alias.fit(X)
    restored = pickle.loads(pickle.dumps(model))
    assert restored.dist.name == alias.name
    assert restored.dist.canonical_name == canon.name
    assert repr(restored) == repr(model)
    np.testing.assert_array_equal(restored.params, model.params)
    np.testing.assert_array_equal(restored.sf(X), model.sf(X))
    fitter = pickle.loads(pickle.dumps(alias))
    assert fitter.name == alias.name
    assert fitter.canonical_name == canon.name


@pytest.mark.parametrize(
    "old, alias, canon",
    [(OLD_GAUSS, sp.Gauss, sp.Normal), (OLD_GALTON, sp.Galton, sp.LogNormal)],
    ids=IDS,
)
def test_dicts_saved_before_still_load(old, alias, canon):
    restored = sp.from_dict(_json(old))
    assert restored.dist is alias
    # And give what a fit now gives
    np.testing.assert_allclose(restored.params, canon.fit(X).params)
    np.testing.assert_allclose(restored.sf(X), canon.fit(X).sf(X))


# -- accelerated life ---------------------------------------------------------


def _al(dist, life_model):
    return _quiet(sp.AcceleratedLife(dist, life_model).fit, LIFE, T)


@pytest.mark.parametrize(
    "dist, life_model",
    [
        (sp.Galton, life_models.Exponential),
        (sp.LogNormal, life_models.Arrhenius),
        (sp.Galton, life_models.Arrhenius),
        (sp.Gauss, life_models.Linear),
        (sp.Weibull, life_models.Arrhenius),
    ],
    ids=["Galton", "Arrhenius", "both", "Gauss", "Weibull-Arrhenius"],
)
def test_accelerated_life_with_an_alias(dist, life_model):
    canon_dist = getattr(sp, canonical_name(dist))
    canon_life = getattr(life_models, canonical_life_model_name(life_model))
    model, plain = _al(dist, life_model), _al(canon_dist, canon_life)
    np.testing.assert_array_equal(model.params, plain.params)
    assert model.neg_ll() == plain.neg_ll()
    assert model.distribution.name == dist.name
    assert model.reg_model.name == life_model.name
    assert f"{life_model.name} life model" in repr(
        sp.AcceleratedLife(dist, life_model)
    )

    d = model.to_dict()
    assert d["distribution"] == dist.name
    assert d["life_model_name"] == life_model.name
    restored = sp.from_dict(_json(d))
    assert restored.distribution.name == dist.name
    assert restored.reg_model.name == life_model.name
    assert repr(restored) == repr(model)
    np.testing.assert_array_equal(restored.params, model.params)
    Z = np.array([[320.0]])
    np.testing.assert_array_equal(
        restored.sf([1000.0], Z), plain.sf([1000.0], Z)
    )


def test_arrhenius_life_model_pickles_with_its_name():
    restored = pickle.loads(pickle.dumps(life_models.Arrhenius))
    assert restored.name == "Arrhenius"
    assert restored.canonical_name == "Exponential"


def test_arrhenius_dict_saved_before_loads_as_exponential():
    # Before, Arrhenius was the Exponential life model itself, so its
    # dicts say "Exponential" and load as that.
    plain = _al(sp.Weibull, life_models.Exponential)
    old = plain.to_dict()
    assert old["life_model_name"] == "Exponential"
    restored = sp.from_dict(_json(old))
    assert restored.reg_model is life_models.Exponential
    np.testing.assert_array_equal(restored.params, plain.params)


def test_arrhenius_diagnosis_is_the_exponentials():
    # The comparison of life models leaves out the model's own, under
    # either name, and the rest of the diagnosis is the same.
    a = _quiet(_al(sp.Weibull, life_models.Arrhenius).diagnose).table
    b = _quiet(_al(sp.Weibull, life_models.Exponential).diagnose).table
    assert "Exponential" not in set(a["life_model"])
    assert list(a["life_model"].replace("Arrhenius", "Exponential")) == list(
        b["life_model"]
    )
    np.testing.assert_array_equal(
        a["statistic"].astype(float), b["statistic"].astype(float)
    )


# -- the rest of the package takes the alias as its distribution --------------


@pytest.mark.parametrize("alias, canon", PAIRS, ids=IDS)
def test_regression_families_take_the_alias(alias, canon):
    # Refused by name before
    assert sp.Distributional(alias).dist is alias
    if canon is sp.LogNormal:
        assert "Galton latency" in repr(sp.MixtureCure(alias))


def test_test_planning_takes_galton_as_the_lognormal():
    from surpyval.test_planning._information import family_of

    assert family_of(sp.Galton) == family_of(sp.LogNormal)


@pytest.mark.parametrize("alias, canon", PAIRS, ids=IDS)
def test_diagnosis_is_the_canonicals(alias, canon):
    a = _quiet(alias.fit(X).diagnose).table
    b = _quiet(canon.fit(X).diagnose).table
    assert a.shape == b.shape
    assert list(a["check"]) == list(b["check"])
    np.testing.assert_array_equal(
        a["statistic"].astype(float), b["statistic"].astype(float)
    )
