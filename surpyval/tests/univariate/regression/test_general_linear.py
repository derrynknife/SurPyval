"""The general linear life model, L(Z) = a + beta'Z.

On a location parameter (``Normal``, ``Logistic``, ``Gumbel`` ``mu``) it
is R's ``survreg(Surv(y, status) ~ z1 + z2, dist = "gaussian")`` (and
``"logistic"``, ``"extreme"``): the time itself, not its log, with the
covariates shifting the distribution. The reference values come from
``scripts/reference/reference_r_general_linear.R``, which builds the same
deterministic data.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

import surpyval as sp
from surpyval.univariate.regression.accelerated_life import LIFE_MODELS

GL = sp.life_models.GeneralLinear


def _data():
    # The rows of reference_r_general_linear.R: two covariates, times on
    # both sides of 0, 18 of 60 right censored.
    i = np.arange(1, 61)
    z1 = (i % 7) - 3.0
    z2 = np.round(np.sin(i), 3)
    eps = norm.ppf(((i * 0.6180339887) % 1) * 0.98 + 0.01)
    y = 2 + 1.5 * z1 - 2.5 * z2 + 3 * eps
    cens = 4 + 2 * np.cos(i)
    c = (y > cens).astype(int)
    return np.minimum(y, cens), np.column_stack([z1, z2]), c


# survreg(Surv(time, status) ~ z1 + z2, dist = ...) with rel.tolerance
# 1e-12: (intercept, z1, z2), scale, log-likelihood. survreg's intercept
# and coefficients are a and beta with the same sign, and its scale is
# the distribution's sigma ("extreme" is the smallest extreme value
# distribution, SurPyval's Gumbel).
R_SURVREG = {
    "Normal": (
        [1.79023552638, 1.00201381618, -2.4867809011],
        2.54322964183,
        -110.468306979,
    ),
    "Logistic": (
        [1.74862722265, 1.05240864314, -2.45877058977],
        1.56055172905,
        -112.436605228,
    ),
    "Gumbel": (
        [2.94597422932, 0.981051743028, -2.35199815731],
        2.24513507502,
        -114.125934298,
    ),
}


@pytest.mark.parametrize("dist", sorted(R_SURVREG))
def test_matches_survreg(dist):
    # Tolerance: 1e-5 relative on the coefficients and scale, 1e-8 on the
    # log-likelihood (both fits converged far tighter).
    x, Z, c = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = sp.AcceleratedLife(getattr(sp, dist), GL).fit(x, Z, c=c)
    coef, sigma, loglik = R_SURVREG[dist]
    assert model.maximum == "verified"
    np.testing.assert_allclose(model.phi_params, coef, rtol=1e-5)
    np.testing.assert_allclose(model.params[1], sigma, rtol=1e-5)
    np.testing.assert_allclose(-model._neg_ll, loglik, rtol=1e-8)


def test_a_shift_of_the_data_moves_a_alone():
    # A location model does not depend on where 0 is: adding 100 to every
    # time adds 100 to a, and the coefficients and sigma stay.
    x, Z, c = _data()
    fitter = sp.AcceleratedLife(sp.Normal, GL)
    a, b = fitter.fit(x, Z, c=c), fitter.fit(x + 100.0, Z, c=c)
    np.testing.assert_allclose(b.phi_params[0], a.phi_params[0] + 100.0)
    np.testing.assert_allclose(b.phi_params[1:], a.phi_params[1:], rtol=1e-6)
    np.testing.assert_allclose(b.params[1], a.params[1], rtol=1e-6)
    q = np.array([0.0, 3.0])
    np.testing.assert_allclose(
        b.sf(q + 100.0, Z[[0, 30]]), a.sf(q, Z[[0, 30]]), rtol=1e-6
    )


def test_positive_coefficient_is_a_later_life():
    # The sign of Linear and GeneralLogLinear: L(Z) = a + beta'Z.
    assert GL.resolve(2).phi(np.array([[1.0, 2.0]]), 10.0, 3.0, -1.0) == [11.0]
    x, Z, c = _data()
    model = sp.AcceleratedLife(sp.Normal, GL).fit(x, Z, c=c)
    assert model.phi_params[1] > 0
    assert model.sf(2.0, [1.0, 0.0]) > model.sf(2.0, [0.0, 0.0])


def test_one_column_is_linear():
    x, Z, c = _data()
    gl = sp.AcceleratedLife(sp.Normal, GL).fit(x, Z[:, :1], c=c)
    lin = sp.AcceleratedLife(sp.Normal, sp.life_models.Linear).fit(
        x, Z[:, 0], c=c
    )
    np.testing.assert_allclose(gl._neg_ll, lin._neg_ll, rtol=1e-9)
    np.testing.assert_allclose(gl.params, lin.params, rtol=1e-5)


def test_resolved_for_the_columns():
    x, Z, c = _data()
    model = sp.AcceleratedLife(sp.Normal, GL).fit(x, Z, c=c)
    lm = model.reg_model
    assert lm.n_stresses == 2
    assert lm.phi_param_map == {"a": 0, "coef_0": 1, "coef_1": 2}
    assert lm.phi_bounds == ((None, None),) * 3
    assert GL.n_stresses is None
    assert GL.resolve(3).n_stresses == 3
    assert lm.resolve(5) is lm
    np.testing.assert_allclose(
        model.sf(1.0, [2.0, 0.5]), model.sf(1.0, [[2.0, 0.5]])
    )


def test_named_by_the_dataframe_columns():
    x, Z, c = _data()
    df = pd.DataFrame({"x": x, "c": c, "temp": Z[:, 0], "load": Z[:, 1]})
    model = sp.AcceleratedLife(sp.Normal, GL).fit_from_df(
        df, x_col="x", c_col="c", Z_cols=["temp", "load"]
    )
    assert model.reg_model.phi_param_map == {"a": 0, "temp": 1, "load": 2}
    ref = sp.AcceleratedLife(sp.Normal, GL).fit(x, Z, c=c)
    np.testing.assert_allclose(model.phi_params, ref.phi_params, rtol=1e-6)


def test_round_trips():
    x, Z, c = _data()
    model = sp.AcceleratedLife(sp.Logistic, GL).fit(x, Z, c=c)
    assert LIFE_MODELS["GeneralLinear"] is GL
    d = json.loads(model.to_json())
    assert d["n_stresses"] == 2
    back = sp.from_json(model.to_json())
    assert back.reg_model.phi_param_map == model.reg_model.phi_param_map
    q = np.array([-2.0, 3.0])
    np.testing.assert_array_equal(
        back.sf(q, Z[[0, 30]]), model.sf(q, Z[[0, 30]])
    )
    del d["n_stresses"]
    with pytest.raises(ValueError, match="n_stresses"):
        sp.from_dict(d)


def test_repeated_column_is_aliased():
    x, Z, c = _data()
    fitter = sp.AcceleratedLife(sp.Normal, GL)
    ref = fitter.fit(x, Z, c=c)
    with pytest.warns(UserWarning, match="life is linear in the stresses"):
        model = fitter.fit(x, np.c_[Z, Z[:, 1]], c=c)
    np.testing.assert_array_equal(model.aliased, [3])
    assert np.isnan(model.params[-1])
    np.testing.assert_allclose(model.params[:-1], ref.params, rtol=1e-6)
