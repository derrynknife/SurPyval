"""Robust all or none: with ``robust=True`` every uncertainty of a
parametric regression rests on the robust (sandwich) covariance.

``summary(robust=True)``, ``param_cb``, ``cb`` and ``quantile_cb`` with
``robust=True`` and ``robust_p_values()`` agree with the robust standard
errors (checked here against Wald intervals and p-values written out from
``robust_standard_errors`` and ``robust_covariance``); without
``robust=True`` everything stays model-based, as before; and what has no
robust form refuses rather than mixing the two.
"""

import numpy as np
import pytest
from scipy.stats import norm

import surpyval as sp
from surpyval import WeibullAFT, WeibullPH
from surpyval.tests._helpers import rossi_with_censoring

Z975 = norm.ppf(0.975)
COLS = ["fin", "age", "prio"]


@pytest.fixture(scope="module")
def rossi():
    df = rossi_with_censoring()
    return df


@pytest.fixture(scope="module")
def model(rossi):
    return WeibullAFT.fit_from_df(
        rossi, x_col="week", c_col="censored", Z_cols=COLS
    )


def _delta_se(f, params, cov, h=1e-6):
    """The delta-method standard error of ``f`` at ``params`` with
    ``cov``, its gradient by central differences."""
    params = np.asarray(params, dtype=float)
    g = np.zeros((np.size(f(params)), params.size))
    for j in range(params.size):
        step = h * max(abs(params[j]), 1.0)
        up, down = params.copy(), params.copy()
        up[j] += step
        down[j] -= step
        g[:, j] = (np.ravel(f(up)) - np.ravel(f(down))) / (2 * step)
    return np.sqrt(np.einsum("ij,jk,ik->i", g, cov, g))


def test_param_cb_and_p_values_agree_with_the_robust_standard_errors(model):
    se = model.robust_standard_errors()
    names = model.parameter_names
    k = model.k_dist
    # The coefficients: b +- z se, and 2 Phi(-|b / se|)
    for i in range(k, len(names)):
        b = model.params[i]
        lo, hi = model.param_cb(names[i], robust=True)
        np.testing.assert_allclose(
            [lo, hi], [b - Z975 * se[i], b + Z975 * se[i]], rtol=1e-12
        )
    p = model.robust_p_values()
    expected = 2 * norm.sf(np.abs(model.params[k:] / se[k:]))
    np.testing.assert_allclose(p[k:], expected, rtol=1e-12)
    assert np.all(np.isnan(p[:k]))
    # The positive scale: on the log scale, a / e^(z se / a), a e^(z se / a)
    a = model.params[0]
    np.testing.assert_allclose(
        model.param_cb("alpha", robust=True),
        a * np.exp(np.array([-1, 1]) * Z975 * se[0] / a),
        rtol=1e-12,
    )
    # They differ from the model-based ones
    assert not np.allclose(se, model.standard_errors())
    assert not np.allclose(p[k:], model.p_values[k:])


def test_summary_robust_is_param_cb_and_robust_p_values(model):
    table = model.summary(robust=True)
    lower, upper = table["coef lower 95%"], table["coef upper 95%"]
    for j, name in enumerate(model.parameter_names):
        bounds = model.param_cb(name, robust=True)
        np.testing.assert_allclose(
            [lower.iloc[j], upper.iloc[j]], bounds, rtol=1e-12
        )
    np.testing.assert_allclose(
        table["se(coef)"], model.robust_standard_errors(), rtol=1e-12
    )
    k = model.k_dist
    np.testing.assert_allclose(
        table["p"].to_numpy()[k:], model.robust_p_values()[k:], rtol=1e-12
    )


def test_cb_and_quantile_cb_use_the_robust_covariance(model):
    x = np.array([10.0, 30.0, 52.0])
    Z = np.array([[0.0, 25.0, 3.0], [1.0, 40.0, 0.0], [0.0, 20.0, 8.0]])
    V = model.robust_covariance()
    params = np.asarray(model.params, dtype=float)
    fitter = model.model
    # The Weibull band is a Wald bound on log H
    u = np.log(model.Hf(x, Z))
    se = _delta_se(lambda p: np.log(fitter.Hf(x, Z, *p)), params, V)
    expected = np.exp(
        -np.exp(u[:, None] + np.array([1, -1]) * Z975 * se[:, None])
    )
    np.testing.assert_allclose(
        model.cb(x, Z, robust=True), expected, rtol=1e-6
    )
    # Model-based by default, as before: the same by hand with covariance()
    se_m = _delta_se(
        lambda p: np.log(fitter.Hf(x, Z, *p)), params, model.covariance()
    )
    expected_m = np.exp(
        -np.exp(u[:, None] + np.array([1, -1]) * Z975 * se_m[:, None])
    )
    np.testing.assert_allclose(model.cb(x, Z), expected_m, rtol=1e-6)
    assert not np.allclose(model.cb(x, Z), model.cb(x, Z, robust=True))
    # The B10 life: a Wald bound on its log, se(H) / (h t)
    t = model.qf(0.1, Z)
    se_H = _delta_se(lambda p: fitter.Hf(t, Z, *p), params, V)
    se_log_t = se_H / (model.hf(t, Z) * t)
    expected_q = t[:, None] * np.exp(
        np.array([-1, 1]) * Z975 * se_log_t[:, None]
    )
    np.testing.assert_allclose(
        model.quantile_cb(0.1, Z, robust=True), expected_q, rtol=1e-6
    )


def test_clustered(rossi, model):
    groups = np.arange(len(rossi)) // 4
    se = model.robust_standard_errors(cluster=groups)
    b = model.params[-1]
    np.testing.assert_allclose(
        model.param_cb(model.parameter_names[-1], robust=True, cluster=groups),
        [b - Z975 * se[-1], b + Z975 * se[-1]],
        rtol=1e-12,
    )
    table = model.summary(robust=True, cluster=groups)
    np.testing.assert_allclose(
        table["p"].to_numpy()[model.k_dist :],
        model.robust_p_values(cluster=groups)[model.k_dist :],
        rtol=1e-12,
    )
    # Clustering changes the band
    x, Z = [20.0], [[1.0, 30.0, 2.0]]
    assert not np.allclose(
        model.cb(x, Z, robust=True),
        model.cb(x, Z, robust=True, cluster=groups),
    )


def test_a_ph_model_with_a_centred_fit(rossi):
    # A baseline reported at 0 behind a centred fit: the robust band is
    # computed in the centred parameterisation, and agrees with the
    # robust covariance of the reported parameters by the delta method.
    model = WeibullPH.fit_from_df(
        rossi, x_col="week", c_col="censored", Z_cols=COLS
    )
    x, Z = np.array([20.0]), np.array([[1.0, 30.0, 2.0]])
    params = np.asarray(model.params, dtype=float)
    se = _delta_se(
        lambda p: np.log(model.model.Hf(x, Z, *p)),
        params,
        model.robust_covariance(),
    )
    u = np.log(model.Hf(x, Z))
    expected = np.exp(-np.exp(u + np.array([1, -1]) * Z975 * se))
    np.testing.assert_allclose(
        model.cb(x, Z, robust=True).ravel(), expected, rtol=1e-6
    )


def test_refusals(model):
    with pytest.raises(ValueError, match="only with robust=True"):
        model.cb([10.0], [[0, 25, 3]], cluster=np.zeros(432))
    with pytest.raises(ValueError, match="only with robust=True"):
        model.param_cb("alpha", cluster=np.zeros(432))
    for method in ("lr", "bootstrap"):
        with pytest.raises(ValueError, match="no robust form"):
            model.cb([10.0], [[0, 25, 3]], method=method, robust=True)
        with pytest.raises(ValueError, match="no robust form"):
            model.param_cb("alpha", method=method, robust=True)
        with pytest.raises(ValueError, match="no robust form"):
            model.quantile_cb(0.1, [[0, 25, 3]], method=method, robust=True)
    # A restored model has no data for the sandwich: refused, not mixed
    restored = sp.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="need the data"):
        restored.cb([10.0], [[0, 25, 3]], robust=True)
    with pytest.raises(ValueError, match="need the data"):
        restored.param_cb("alpha", robust=True)
    with pytest.raises(ValueError, match="need the data"):
        restored.robust_p_values()
    # ...while its model-based bounds are as before
    np.testing.assert_allclose(
        restored.cb([10.0], [[0, 25, 3]]), model.cb([10.0], [[0, 25, 3]])
    )


def test_a_weighted_fit_is_robust_either_way():
    # With weights the model-based covariance is already the sandwich
    rng = np.random.default_rng(4)
    z = rng.normal(size=(120, 1))
    x = 10 * np.exp(-0.3 * z[:, 0]) * rng.weibull(1.5, 120)
    w = rng.uniform(0.5, 2.0, 120)
    model = WeibullAFT.fit(x, z, w=w)
    np.testing.assert_allclose(
        model.cb([8.0], [[0.5]], robust=True), model.cb([8.0], [[0.5]])
    )
    np.testing.assert_allclose(
        model.param_cb("coef_0", robust=True), model.param_cb("coef_0")
    )
