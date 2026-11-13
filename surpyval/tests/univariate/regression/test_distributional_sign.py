"""The sign of a distributional regression's coefficients (v0.27).

v0.26 gave every coefficient the link's sign, ``g(theta(z)) = g(theta) +
b'z``, so that a positive coefficient on a scale meant a longer life,
the opposite of SurPyval's AFT models. Since v0.27 each parameter's
coefficients carry SurPyval's sign, ``g(theta(z)) = g(theta) + s b'z``:
``s = -1`` for a parameter whose increase makes every quantile later (a
scale, a location), ``+1`` otherwise, so a positive coefficient means a
shorter life wherever the parameter has a direction.

The saved models in ``reference/data/v026_distributional.json`` were
written by SurPyval 0.26 from PyPI (its ``to_dict`` and ``pickle``), with
the predictions it gave; loaded here they must predict the same.
"""

import base64
import json
import pickle
from pathlib import Path

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.regression.distributional import _links
from surpyval.univariate.regression.distributional.fitter import SUPPORTED

V026 = (
    Path(__file__).parents[2]
    / "reference"
    / "data"
    / "v026_distributional.json"
)

# A value of each distribution's parameters to move from.
BASE = {
    "Exponential": [0.5],
    "Rayleigh": [2.0],
    "Weibull": [3.0, 1.5],
    "Gamma": [2.0, 0.7],
    "LogNormal": [1.0, 0.5],
    "LogLogistic": [3.0, 2.0],
    "Normal": [5.0, 1.0],
    "Gumbel": [5.0, 1.0],
    "GumbelLEV": [5.0, 1.0],
    "Logistic": [5.0, 1.0],
    "Beta": [2.0, 3.0],
    "ExpoWeibull": [3.0, 1.5, 2.0],
}


@pytest.mark.parametrize("name", SUPPORTED)
def test_each_sign_is_its_parameters_effect_on_life(name):
    # Raising a parameter with s = -1 makes every quantile later, one
    # listed "shorter" every quantile earlier; the others (s = +1, no
    # direction) move some quantiles each way.
    dist = getattr(sp, name)
    p = np.linspace(0.01, 0.99, 99)
    for k, parameter in enumerate(dist.parameter_names):
        up = list(BASE[name])
        up[k] *= 1.1
        dq = dist.qf(p, *up) - dist.qf(p, *BASE[name])
        effect = _links.EFFECT[name].get(parameter)
        if effect == "longer":
            assert np.all(dq > 0), parameter
            assert _links.sign(name, parameter) == -1.0
        elif effect == "shorter":
            assert np.all(dq < 0), parameter
            assert _links.sign(name, parameter) == 1.0
        else:
            assert np.any(dq > 0) and np.any(dq < 0), parameter
            assert _links.sign(name, parameter) == 1.0


def test_each_kind_of_parameter_moves_as_documented():
    z = np.array([[0.0], [1.0]])
    rng = np.random.default_rng(1)
    zz = rng.uniform(0, 1, 200)
    # A scale: alpha e^(-b z); a shape with no direction: beta e^(b z)
    x = 10 * np.exp(-0.5 * zz) * rng.weibull(1.5, 200)
    model = sp.WeibullDistributional.fit(x, zz)
    a, b, ca, cb = model.params
    np.testing.assert_allclose(
        model.parameters_at(z), [[a, b], [a * np.exp(-ca), b * np.exp(cb)]]
    )
    assert ca > 0  # the scale fell with z: a shorter life
    # A rate: lambda e^(b z)
    expo = sp.ExponentialDistributional.fit(x, zz)
    lam, c = expo.params
    np.testing.assert_allclose(
        expo.parameters_at(z)[:, 0], lam * np.exp([0, c])
    )
    assert c > 0
    # A location: mu - b z
    y = rng.normal(20 - 3 * zz, 1.0)
    normal = sp.NormalDistributional.fit(y, zz, covariates={"mu": [0]})
    mu, _, cm = normal.params
    np.testing.assert_allclose(normal.parameters_at(z)[:, 0], [mu, mu - cm])
    np.testing.assert_allclose(cm, 3.0, atol=0.3)
    # A Gamma's shape orders lives: alpha e^(-b z)
    g = sp.GammaDistributional.fit(x, zz, covariates={"alpha": [0]})
    ga, gb, gc = g.params
    np.testing.assert_allclose(g.parameters_at(z)[:, 0], ga * np.exp([0, -gc]))
    assert g.coefficient_signs == {"alpha": -1.0, "beta": 1.0}


def _cases():
    with open(V026) as f:
        saved = json.load(f)
    assert saved["surpyval"] == "0.26"
    return saved["models"]


# The bounds take a central-difference Jacobian (a step of about 6e-6),
# whose rounding differs between platforms by about 1e-10 of a bound: the
# values stored by v0.26 on one platform are met to 1e-8 on another
# (2.5e-10 seen on a tail probability of 0.01).
BOUND_RTOL = 1e-8


def _check_predictions(model, case):
    rows, t = case["rows"], case["t"]
    np.testing.assert_allclose(model.sf(t, rows), case["sf"], rtol=1e-12)
    np.testing.assert_allclose(model.cb(t, rows), case["cb"], rtol=BOUND_RTOL)
    np.testing.assert_allclose(
        model.parameters_at(rows), case["parameters_at"], rtol=1e-12
    )
    np.testing.assert_allclose(
        model.parameter_cb(case["parameter"], rows),
        case["parameter_cb"],
        rtol=BOUND_RTOL,
    )
    np.testing.assert_allclose(
        model.standard_errors(), case["standard_errors"], rtol=1e-10
    )
    # The coefficient's sign, and so its bound, is the other way round
    # where its parameter has s = -1 (the scale and the location here)
    k = model.k_dist
    s = np.ones(len(model.params))
    j = k
    for sign, cols in zip(
        model.coefficient_signs.values(), model.covariate_positions
    ):
        s[j : j + len(cols)] = sign
        j += len(cols)
    np.testing.assert_allclose(model.params, s * case["params"], rtol=1e-15)
    lo, hi = case["param_cb"]
    expected = [lo, hi] if s[k] == 1 else [-hi, -lo]
    np.testing.assert_allclose(
        model.param_cb(case["coefficient"]), expected, rtol=BOUND_RTOL
    )


@pytest.mark.parametrize("i", [0, 1])
def test_a_v026_dict_loads_and_predicts_as_it_did(i):
    case = _cases()[i]
    model = sp.from_dict(case["dict"])
    _check_predictions(model, case)
    assert -model.neg_ll() == pytest.approx(-case["neg_ll"], rel=1e-15)
    # Saved again it carries its signs, as schema 4 (which a v0.26 reader
    # refuses), and loads unchanged
    d = model.to_dict()
    assert d["schema"] == 4
    assert d["coefficient_signs"] == list(model.coefficient_signs.values())
    _check_predictions(sp.from_dict(d), case)


@pytest.mark.parametrize("i", [0, 1])
def test_a_v026_pickle_loads_and_predicts_as_it_did(i):
    case = _cases()[i]
    model = pickle.loads(base64.b64decode(case["pickle"]))
    _check_predictions(model, case)
    # The fixed coefficient moved with its sign
    for name, value in case["dict"]["fixed"].items():
        k = model.parameter_names.index(name)
        assert model.fixed[name] == pytest.approx(
            value * model.params[k] / case["params"][k]
        )
    # It pickles again with its signs
    again = pickle.loads(pickle.dumps(model))
    _check_predictions(again, case)
    np.testing.assert_allclose(model.neg_ll(), case["neg_ll"], rtol=1e-12)


def test_a_dict_with_bad_signs_is_refused():
    d = _cases()[0]["dict"]
    d = {**d, "coefficient_signs": [0.5, 1.0]}
    with pytest.raises(ValueError, match="coefficient_signs"):
        sp.from_dict(d)
