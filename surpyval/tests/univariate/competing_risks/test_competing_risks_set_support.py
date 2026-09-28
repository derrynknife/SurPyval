"""``CompetingRisks.set_support``: an explicit support for the
non-parametric competing-risks estimate (principle 11)."""

import copy
import json
import warnings

import numpy as np
import pytest

import surpyval
from surpyval.univariate.competing_risks import CompetingRisks

X = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10])
E = ["a", "b", "a", None, "a", "b", "a", None, "b", None]
METHODS = ("Nelson-Aalen", "Kaplan-Meier")
START = {"sf": 1.0, "ff": 0.0, "Hf": 0.0, "hf": 0.0, "df": 0.0}
START.update({"cif": 0.0, "iif": 0.0})


def _calls():
    out = [(f, e) for f in ("sf", "ff", "Hf", "hf", "df") for e in (None,)]
    out += [(f, e) for f in ("sf", "Hf", "hf") for e in ("a", "b")]
    out += [(f, e) for f in ("cif", "iif") for e in ("a", "b")]
    return out


def _f(model, fname, x, event):
    if fname in ("cif", "iif"):
        return np.asarray(getattr(model, fname)(x, event), float)
    return np.asarray(getattr(model, fname)(x, event=event), float)


@pytest.fixture(params=METHODS)
def model(request):
    return CompetingRisks.fit(X, E, method=request.param)


def test_set_support_chains_and_a_fit_has_none(model):
    assert model.support is None
    assert model.set_support(0, 20) is model
    assert model.support == (0.0, 20.0)


@pytest.mark.parametrize("fname, event", _calls())
def test_every_region(model, fname, event):
    bounded = copy.deepcopy(model).set_support(-5.0, 20.0)
    inside = np.linspace(1.0, 10.0, 19)
    np.testing.assert_array_equal(
        _f(bounded, fname, inside, event), _f(model, fname, inside, event)
    )
    for q in ([-5.1], [20.1], [np.nan], [-np.inf], [np.inf]):
        assert np.isnan(_f(bounded, fname, q, event)).all()
    start = _f(bounded, fname, [-5.0, 0.0, 0.999], event)
    np.testing.assert_array_equal(start, START[fname])
    assert not np.signbit(start).any()
    after = _f(bounded, fname, [10.5, 20.0], event)
    np.testing.assert_array_equal(after, _f(model, fname, 10.0, event)[0])


def test_infinite_bounds(model):
    bounded = copy.deepcopy(model).set_support(-np.inf, np.inf)
    np.testing.assert_array_equal(
        bounded.sf([-np.inf, np.inf]), [1.0, model.sf(10.0)[0]]
    )
    np.testing.assert_array_equal(
        bounded.cif([-np.inf, np.inf], "a"), [0.0, model.cif(10.0, "a")[0]]
    )


def test_unbounded_holds_its_value_forever(model):
    # What set_support changes: without it the estimate holds everywhere.
    assert model.cif(1e9, "a")[0] == model.cif(10.0, "a")[0]
    assert model.sf(-1e9)[0] == 1.0


def test_no_raw_numpy_warning(model):
    bounded = copy.deepcopy(model).set_support(-np.inf, 30)
    q = np.array([-np.inf, -1.0, 1.0, 5.5, 10.0, 30.0, 31.0, np.inf, np.nan])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for fname, event in _calls():
            _f(bounded, fname, q, event)


def test_negative_times():
    x = np.array([-5.0, -3.0, -1.0, 0.0, 2.0])
    model = CompetingRisks.fit(x, ["a", "b", None, "a", "b"])
    bounded = copy.deepcopy(model).set_support(-10, 5)
    np.testing.assert_array_equal(
        bounded.cif([-11, -10, -6, 3, 5, 6], "a"),
        [np.nan, 0.0, 0.0, model.cif(2.0, "a")[0], model.cif(2.0, "a")[0]]
        + [np.nan],
    )
    with pytest.raises(ValueError, match="first time"):
        model.set_support(-4, 5)


@pytest.mark.parametrize(
    "lower, upper, match",
    [
        (2.0, 20.0, r"'lower' \(2.0\) is above the first time \(1.0\)"),
        (0.0, 9.0, r"'upper' \(9.0\) is below the last time \(10.0\)"),
        (5.0, 5.0, "'lower' must be below 'upper'"),
        (np.nan, 5.0, "not NaN"),
    ],
)
def test_invalid_bounds_are_refused(lower, upper, match):
    model = CompetingRisks.fit(X, E)
    with pytest.raises(ValueError, match=match):
        model.set_support(lower, upper)
    assert model.support is None


@pytest.mark.parametrize("support", [(0.0, 20.0), (-np.inf, np.inf)])
def test_serialisation_round_trip(model, support):
    bounded = copy.deepcopy(model).set_support(*support)
    d = bounded.to_dict()
    assert d["schema"] == 2
    text = json.dumps(d, allow_nan=False)
    restored = surpyval.from_dict(json.loads(text))
    assert restored.support == support
    q = np.array([-1.0, 0.5, 3.5, 12.0, 25.0, np.inf])
    for fname, event in _calls():
        np.testing.assert_array_equal(
            _f(restored, fname, q, event), _f(bounded, fname, q, event)
        )


def test_without_bounds_the_dictionary_is_unchanged(model):
    d = model.to_dict()
    assert "support" not in d
    assert d["schema"] == 1
    assert surpyval.from_dict(d).support is None
