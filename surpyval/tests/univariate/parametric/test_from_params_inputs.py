"""``from_params`` takes a plain number for a one-parameter distribution
and refuses one for more, naming how many are needed; ``parameter_roles``
and ``distribution`` say what a model is."""

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.parametric.parametric_fitter import ParametricFitter


def _families():
    seen = {}
    for name in dir(sp):
        obj = getattr(sp, name)
        if isinstance(obj, ParametricFitter) and obj.parameter_names:
            seen.setdefault(obj.canonical_name, obj)
    return sorted(seen.values(), key=lambda f: f.name)


ONE = [f for f in _families() if f.k == 1]
MORE = [f for f in _families() if f.k > 1]

# A value inside each one-parameter family's bounds.
VALUE = {
    "Bernoulli": 0.3,
    "ExactEventTime": 5.0,
    "Exponential": 1 / 20000,
    "FixedEventProbability": 0.3,
    "Geometric": 0.3,
    "Poisson": 2.0,
    "Rayleigh": 3.0,
}


def test_every_one_parameter_family_is_covered():
    assert {f.name for f in ONE} == set(VALUE)


@pytest.mark.parametrize("family", ONE, ids=lambda f: f.name)
def test_a_one_parameter_family_takes_a_number(family):
    value = VALUE[family.name]
    bare = family.from_params(value)
    listed = family.from_params([value])
    np.testing.assert_array_equal(bare.params, listed.params)
    np.testing.assert_array_equal(bare.params, [value])
    assert family.from_params(np.float64(value)).params.shape == (1,)


@pytest.mark.parametrize("family", MORE, ids=lambda f: f.name)
def test_a_number_for_several_parameters_is_refused_with_the_count(family):
    names = ", ".join(family.parameter_names)
    with pytest.raises(ValueError) as err:
        family.from_params(2.0)
    message = str(err.value)
    assert f"{family.name} takes {family.k} parameters ({names})" in message
    assert "got 1 number" in message


def test_the_wrong_number_of_parameters_is_named():
    with pytest.raises(ValueError, match=r"takes 2 parameters \(alpha, beta"):
        sp.Weibull.from_params([1, 2, 3])
    with pytest.raises(ValueError, match="takes 1 parameter "):
        sp.Exponential.from_params([1, 2])


def test_a_listed_model_is_unchanged():
    model = sp.Weibull.from_params([10, 4])
    np.testing.assert_array_equal(model.params, [10.0, 4.0])
    assert model.sf(10) == pytest.approx(np.exp(-1))


@pytest.mark.parametrize("family", _families(), ids=lambda f: f.name)
def test_every_parameter_has_a_recorded_role(family):
    roles = family.parameter_roles
    assert list(roles) == family.parameter_names
    assert "parameter" not in roles.values()


def test_the_roles_tell_a_rate_from_a_shape():
    assert sp.Gamma.parameter_roles["beta"] == "rate"
    assert sp.Weibull.parameter_roles["beta"] == "shape"
    # Gamma's mean is alpha / beta: beta is a rate
    assert sp.Gamma.from_params([2, 0.5]).mean() == pytest.approx(4.0)
    assert sp.Gamma.from_params([2, 0.5]).parameter_roles == {
        "alpha": "shape",
        "beta": "rate",
    }


def test_a_model_names_its_distribution():
    model = sp.Weibull.fit([1.0, 2.0, 3.0, 4.0, 6.0])
    assert type(model).__name__ == "Parametric"
    assert model.distribution == "Weibull"
    assert sp.LogNormal.from_params([1, 1]).distribution == "LogNormal"
