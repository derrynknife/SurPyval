"""The general log-linear life model, L(Z) = c exp(beta'Z) (#530, #345).

``AcceleratedLife(Weibull, GeneralLogLinear).fit`` raised autograd's
"array is not broadcastable" for any data: ``phi`` returned a 0-d life
for a stress row, whose gradient through the fit's ``where`` had the
wrong shape. The model also had no constant factor, so L(0) was 1 in
whatever units the times were in, and its parameter map and bounds were
callables of Z rather than the dict and tuple ``LifeModel`` declares.
"""

import json
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.regression.accelerated_life import LIFE_MODELS


def _data():
    rng = np.random.default_rng(0)
    Z = np.column_stack(
        [
            np.repeat([1.0, 2.0, 3.0], 40),
            np.tile(np.repeat([0.5, 1.5], 20), 3),
        ]
    )
    life = np.exp(4.0 - 0.5 * Z[:, 0] + 0.3 * Z[:, 1])
    x = life * rng.weibull(2.0, len(life))
    c = (x > np.quantile(x, 0.85)).astype(int)
    return x, Z, c


GLL = sp.AcceleratedLife(sp.Weibull, sp.life_models.GeneralLogLinear)


def test_fits_and_is_the_weibull_aft():
    # The fit raised a ValueError from autograd (#530). With Weibull,
    # alpha = c exp(beta'Z) is the Weibull AFT model (whose alpha is
    # alpha exp(-beta'Z)): the same maximum.
    x, Z, c = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = GLL.fit(x, Z, c=c)
    aft = sp.WeibullAFT.fit(x, Z, c=c)
    np.testing.assert_allclose(model._neg_ll, aft._neg_ll, rtol=1e-9)
    np.testing.assert_allclose(
        model.params[1:],
        [aft.params[1], aft.params[0], *(-aft.params[2:])],
        rtol=1e-4,
    )
    q = np.array([5.0, 20.0, 40.0])
    np.testing.assert_allclose(
        model.sf(q, Z[[0, 50, 100]]), aft.sf(q, Z[[0, 50, 100]]), rtol=1e-5
    )


def test_resolved_for_the_columns():
    # The fitted model carries a life model with a fixed parameter map
    # and bounds, of the types LifeModel declares (#345).
    x, Z, c = _data()
    model = GLL.fit(x, Z, c=c)
    lm = model.reg_model
    assert lm.n_stresses == 2
    assert lm.phi_param_map == {"c": 0, "coef_0": 1, "coef_1": 2}
    assert lm.phi_bounds == ((0, None), (None, None), (None, None))
    assert sp.life_models.GeneralLogLinear.n_stresses is None
    assert sp.life_models.GeneralLogLinear.resolve(3).n_stresses == 3
    assert lm.resolve(5) is lm
    # A 1-D query is one row of two stresses.
    np.testing.assert_allclose(
        model.sf(10.0, [2.0, 0.5]), model.sf(10.0, [[2.0, 0.5]])
    )
    # Three columns: three coefficients.
    rng = np.random.default_rng(1)
    Z3 = np.column_stack([Z, rng.normal(size=len(x))])
    assert GLL.fit(x, Z3, c=c).reg_model.n_stresses == 3


@pytest.mark.parametrize(
    "dist",
    [
        sp.Weibull,
        sp.LogNormal,
        sp.Exponential,
        sp.Gamma,
        sp.Normal,
        sp.Gumbel,
        sp.Logistic,
    ],
)
def test_621_a_continuous_stress_fits_every_baseline(dist):
    # With a continuous third column every row is its own stress level,
    # and a level's own fit fails on one row: its start took the mean time
    # as the life parameter itself, so a LogNormal's mu was the time and
    # its life exp(time), and the fit raised "The log-likelihood is not
    # finite at the initial parameter values" (#621).
    x, Z, c = _data()
    rng = np.random.default_rng(1)
    Z3 = np.column_stack([Z, rng.normal(size=len(x))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = sp.AcceleratedLife(dist, sp.life_models.GeneralLogLinear).fit(
            x, Z3, c=c
        )
    assert model.maximum == "verified"
    assert model.reg_model.n_stresses == 3


def test_621_the_lognormal_is_the_lognormal_aft():
    x, Z, c = _data()
    rng = np.random.default_rng(1)
    Z3 = np.column_stack([Z, rng.normal(size=len(x))])
    model = sp.AcceleratedLife(
        sp.LogNormal, sp.life_models.GeneralLogLinear
    ).fit(x, Z3, c=c)
    aft = sp.LogNormalAFT.fit(x, Z3, c=c)
    np.testing.assert_allclose(model._neg_ll, aft._neg_ll, rtol=1e-9)


def test_scale_invariant():
    # The constant factor c carries the units (principle 6): without it
    # L(0) was 1 whatever the time unit.
    x, Z, c = _data()
    a, b = GLL.fit(x, Z, c=c), GLL.fit(1000.0 * x, Z, c=c)
    np.testing.assert_allclose(b.params[2], 1000.0 * a.params[2], rtol=1e-4)
    np.testing.assert_allclose(b.params[3:], a.params[3:], rtol=1e-4)


def test_round_trips():
    x, Z, c = _data()
    model = GLL.fit(x, Z, c=c)
    assert "GeneralLogLinear" in LIFE_MODELS
    d = json.loads(model.to_json())
    assert d["n_stresses"] == 2
    back = sp.from_json(model.to_json())
    assert back.reg_model.phi_param_map == model.reg_model.phi_param_map
    q = np.array([5.0, 20.0])
    np.testing.assert_array_equal(
        back.sf(q, Z[[0, 100]]), model.sf(q, Z[[0, 100]])
    )
    del d["n_stresses"]
    with pytest.raises(ValueError, match="n_stresses"):
        sp.from_dict(d)


def test_repeated_column_is_aliased():
    # The second copy of a column is aliased (#503): nan, and the fit
    # otherwise that without it.
    x, Z, c = _data()
    ref = GLL.fit(x, Z, c=c)
    with pytest.warns(UserWarning, match="cannot be estimated"):
        model = GLL.fit(x, np.c_[Z, Z[:, 1]], c=c)
    np.testing.assert_array_equal(model.aliased, [3])
    assert np.isnan(model.params[-1])
    np.testing.assert_allclose(model.params[:-1], ref.params, rtol=1e-6)
