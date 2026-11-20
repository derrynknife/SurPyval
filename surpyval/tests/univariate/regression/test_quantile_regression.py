"""Censored quantile regression (Peng and Huang 2008; Portnoy 2003).

The model is linear in the covariates at each quantile of the log time,
``log Q(tau | Z) = b0(tau) + b(tau)'Z``, fitted on a grid of levels. The
references here are independent of the fit: ordinary quantile regression
solved as its own (primal) linear programme, the Kaplan-Meier and
Nelson-Aalen estimates the two estimators reduce to with no covariates,
and the true quantiles of simulated data.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import linprog
from scipy.stats import norm

import surpyval as sp
from surpyval import QuantileRegression

METHODS = ("peng-huang", "portnoy")


def _hetero(N, seed, cens_mu=3.0):
    """log T = 2 + 0.5 z1 - 0.3 z2 + (0.3 + 0.2 z1) eps, eps standard
    normal: the tau quantile's slope on z1 is 0.5 + 0.2 Phi^-1(tau), and on
    z2 -0.3 at every level. Independent log-normal censoring."""
    rng = np.random.default_rng(seed)
    z1 = rng.uniform(0, 2, N)
    z2 = rng.binomial(1, 0.5, N).astype(float)
    eps = rng.normal(size=N)
    T = np.exp(2.0 + 0.5 * z1 - 0.3 * z2 + (0.3 + 0.2 * z1) * eps)
    C = np.exp(rng.normal(cens_mu, 0.7, N))
    c = (T > C).astype(int)
    return np.minimum(T, C), np.column_stack([z1, z2]), c


def _truth(tau):
    q = norm.ppf(tau)
    return np.array([2.0 + 0.3 * q, 0.5 + 0.2 * q, -0.3])


def _quantile_lp(X, y, tau):
    """Ordinary quantile regression as its primal linear programme:
    min sum tau u + (1 - tau) v subject to X b + u - v = y."""
    n, p = X.shape
    cost = np.r_[np.zeros(p), np.full(n, tau), np.full(n, 1 - tau)]
    A = np.hstack([X, np.eye(n), -np.eye(n)])
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    res = linprog(cost, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    return res.x[:p]


def _fit(x, Z, c=None, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return QuantileRegression.fit(x, Z, c=c, **kw)


# ---------------------------------------------------------------------------
# The estimators against independent references
# ---------------------------------------------------------------------------
def test_portnoy_without_censoring_is_ordinary_quantile_regression():
    # No unit is censored, so nothing is reweighted: each level is the
    # ordinary quantile regression of log time (Koenker and Bassett).
    rng = np.random.default_rng(1)
    N = 120
    Z = rng.uniform(-1, 1, size=(N, 2))
    y = 1.0 + Z @ [0.4, -0.2] + rng.normal(0, 0.4, N)
    model = _fit(np.exp(y), Z, method="portnoy", step=0.05)
    X = np.column_stack([np.ones(N), Z])
    for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
        np.testing.assert_allclose(
            model.coefficients(tau), _quantile_lp(X, y, tau), atol=1e-8
        )


def _one_sample(seed, N=300):
    rng = np.random.default_rng(seed)
    T = np.exp(2 + rng.normal(0, 0.6, N))
    C = np.exp(rng.normal(2.3, 0.6, N))
    return np.minimum(T, C), (T > C).astype(int)


@pytest.mark.parametrize(
    "method, reference",
    [("portnoy", sp.KaplanMeier), ("peng-huang", sp.FlemingHarrington)],
)
@pytest.mark.parametrize("seed", [0, 5])
def test_without_covariates_is_the_non_parametric_quantile(
    method, reference, seed
):
    # With only the intercept, Portnoy's estimator is the Kaplan-Meier
    # quantile (Efron's redistribution to the right) and Peng and Huang's
    # the quantile of exp(-Nelson-Aalen), each up to the grid's resolution:
    # the estimate at tau lies where the reference curve passes tau, to
    # within a step of the grid (0.004) and a unit's share (1/300).
    x, c = _one_sample(seed)
    model = _fit(x, np.zeros((x.size, 0)), c=c, method=method, step=0.004)
    ref = reference.fit(x, c=c)
    q = np.exp(model.intercepts)
    below, at = ref.ff(q * (1 - 1e-9)), ref.ff(q)
    miss = np.maximum(below - model.taus, model.taus - at).clip(0)
    assert miss.max() < 0.004 + 1 / 300, miss.max()
    # It stops where the reference stops: at the last event.
    assert model.tau_max <= ref.ff(x.max()) + 1e-12


@pytest.mark.parametrize("method", METHODS)
def test_recovers_varying_coefficients(method):
    x, Z, c = _hetero(3000, 11)
    model = _fit(x, Z, c=c, method=method)
    for tau in (0.25, 0.5, 0.75):
        np.testing.assert_allclose(
            model.coefficients(tau), _truth(tau), atol=0.06
        )
    # The spread grows with z1, so its effect on the B10 life is weaker
    # than on the 90th percentile.
    assert model.coefficients(0.1)[1] < model.coefficients(0.9)[1] - 0.2


# ---------------------------------------------------------------------------
# Data invariances
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", METHODS)
def test_counts_equal_repeated_rows(method):
    x, Z, c = _hetero(150, 2)
    n = np.tile([1, 2, 3], 50)
    counted = _fit(x, Z, c=c, n=n, method=method)
    repeated = _fit(
        np.repeat(x, n),
        np.repeat(Z, n, axis=0),
        c=np.repeat(c, n),
        method=method,
    )
    np.testing.assert_array_equal(counted.taus, repeated.taus)
    np.testing.assert_allclose(counted.coefs, repeated.coefs, rtol=1e-10)


@pytest.mark.parametrize("method", METHODS)
def test_order_and_time_unit_do_not_matter(method):
    x, Z, c = _hetero(200, 3)
    model = _fit(x, Z, c=c, method=method)
    order = np.random.default_rng(0).permutation(x.size)
    shuffled = _fit(x[order], Z[order], c=c[order], method=method)
    np.testing.assert_array_equal(model.coefs, shuffled.coefs)
    # Times in other units: the intercept moves by log(k), nothing else.
    hours = _fit(24 * x, Z, c=c, method=method)
    np.testing.assert_array_equal(model.taus, hours.taus)
    np.testing.assert_allclose(hours.coefs, model.coefs, atol=1e-9)
    np.testing.assert_allclose(
        hours.intercepts, model.intercepts + np.log(24), atol=1e-9
    )
    p = [0.1, 0.5]
    np.testing.assert_allclose(
        hours.qf(p, [1.0, 0.0]), 24 * model.qf(p, [1.0, 0.0]), rtol=1e-9
    )


# ---------------------------------------------------------------------------
# Predictions
# ---------------------------------------------------------------------------
def test_quantile_is_the_linear_model_at_the_grid():
    x, Z, c = _hetero(800, 4)
    model = _fit(x, Z, c=c)
    row = np.array([1.0, 1.0])
    b = model.coefficients(0.1)
    # Where the quantiles at the levels below do not cross it, the B10 is
    # exp(b0 + b'Z) itself.
    lp = model.intercepts[:10] + model.coefs[:10] @ row
    assert np.all(np.diff(lp) > 0)
    np.testing.assert_allclose(
        model.qf(0.1, row), np.exp(b[0] + b[1:] @ row), rtol=1e-12
    )


def test_ff_and_qf_are_one_proper_step_distribution():
    x, Z, c = _hetero(400, 5)
    model = _fit(x, Z, c=c, method="portnoy")
    rows = np.array([[0.2, 0.0], [1.8, 1.0], [1.0, 0.0]])
    t = np.r_[np.geomspace(0.5, 200, 400), 1e9]
    for row in rows:
        F = model.ff(t, row)
        assert np.all(np.diff(F) >= 0) and F[0] == 0
        assert F[-1] == pytest.approx(model.tau_max)
        np.testing.assert_allclose(model.sf(t, row), 1 - F)
        np.testing.assert_allclose(model.Hf(t, row), -np.log(1 - F))
        # qf is the first time ff reaches p.
        p = np.array([0.05, 0.1, 0.33, 0.5, 0.77])
        q = model.qf(p, row)
        assert np.all(model.ff(q, row) >= p - 1e-9)
        assert np.all(model.ff(q * (1 - 1e-9), row) < p)
    # Past the last level, qf is nan; before 0 nothing has failed.
    assert np.isnan(model.qf(min(model.tau_max + 0.01, 0.999), rows[0]))
    assert model.ff(0.0, rows[0]) == 0.0


def test_rows_paired_with_times():
    x, Z, c = _hetero(300, 6)
    model = _fit(x, Z, c=c)
    rows = np.array([[0.5, 0.0], [1.5, 1.0], [1.0, 1.0]])
    t = np.array([5.0, 8.0, 12.0])
    together = model.sf(t, rows)
    alone = [model.sf(t[k], rows[k]) for k in range(3)]
    np.testing.assert_array_equal(together, alone)
    with pytest.raises(ValueError, match="covariate rows"):
        model.sf([1.0, 2.0], rows)


def test_rossi_b10_by_covariates():
    # The B10 (tenth percentile) of the time to re-arrest: financial aid
    # lengthens it, prior convictions shorten it.
    df = sp.datasets.load_rossi_static()
    x, c = df["week"].values, 1 - df["arrest"].values
    model = _fit(x, df[["fin", "age", "prio"]].values, c=c)
    b = model.coefficients(0.1)
    assert b[1] > 0 and b[3] < 0
    df_model = QuantileRegression.fit_from_df(
        df.assign(c=c),
        x_col="week",
        Z_cols=["fin", "age", "prio"],
        c_col="c",
    )
    np.testing.assert_array_equal(df_model.coefs, model.coefs)
    rows = df[["fin", "age", "prio"]].iloc[:3]
    np.testing.assert_array_equal(
        df_model.qf(0.1, rows), model.qf(0.1, rows.values)
    )


# ---------------------------------------------------------------------------
# Coefficients and refusals
# ---------------------------------------------------------------------------
def test_coefficients_are_the_step_process():
    x, Z, c = _hetero(300, 7)
    model = _fit(x, Z, c=c, step=0.05)
    np.testing.assert_allclose(model.taus[:3], [0.05, 0.1, 0.15])
    np.testing.assert_array_equal(
        model.coefficients(0.07), model.coefficients(0.1)
    )
    assert model.coefficients([0.1, 0.5]).shape == (2, 3)
    with pytest.raises(ValueError, match="past the last quantile"):
        model.coefficients(min(model.tau_max + 0.05, 0.99))
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        model.coefficients(1.5)


def test_heavy_censoring_stops_the_process():
    # Follow-up ends at the median life of the units with z = 0, the
    # shortest lived, and before it for the others: no quantile much above
    # the median can be estimated for anyone.
    rng = np.random.default_rng(8)
    z = rng.uniform(0, 1, 400)
    T = np.exp(1 + 0.5 * z + 0.3 * rng.normal(size=400))
    end = np.exp(1.0)
    model = _fit(np.minimum(T, end), z, c=(T > end).astype(int))
    assert model.tau_max < 0.6
    assert np.isnan(model.qf(0.9, [0.5]))
    assert model.sf(1e6, [0.5]) == pytest.approx(1 - model.tau_max)


def test_aliased_column():
    x, Z, c = _hetero(200, 9)
    ref = _fit(x, Z, c=c)
    with pytest.warns(UserWarning, match="cannot be estimated"):
        model = QuantileRegression.fit(x, np.c_[Z, Z[:, 0] * 2], c=c)
    np.testing.assert_array_equal(model.aliased, [2])
    assert np.isnan(model.coefs[:, 2]).all()
    np.testing.assert_allclose(model.coefs[:, :2], ref.coefs, rtol=1e-12)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"c": [-1] + [0] * 9}, "only observed"),
        ({"x": np.r_[0.0, np.arange(1.0, 10)]}, "must be positive"),
        ({"c": np.ones(10, int)}, "at least one event"),
        ({"method": "crq"}, "'method' must be one of"),
        ({"step": 0.0}, "'step' must be a number"),
    ],
)
def test_refusals_name_the_problem(kwargs, match):
    args = {"x": np.arange(1.0, 11), "Z": np.linspace(0, 1, 10)}
    args.update(kwargs)
    with pytest.raises(ValueError, match=match):
        QuantileRegression.fit(**args)


# ---------------------------------------------------------------------------
# Bootstrap bounds
# ---------------------------------------------------------------------------
def test_bootstrap_bounds():
    x, Z, c = _hetero(250, 10)
    # (A coarse grid keeps the refits quick.)
    model = _fit(x, Z, c=c, step=0.05)
    kw = dict(n_boot=30, random_state=4)
    ci = model.bootstrap_ci(0.5, alpha_ci=0.1, **kw)
    assert ci.shape == (3, 2) and np.all(ci[:, 0] < ci[:, 1])
    assert np.all((ci[:, 0] < _truth(0.5)) & (_truth(0.5) < ci[:, 1]))
    np.testing.assert_array_equal(ci, model.bootstrap_ci(0.5, 0.1, **kw))
    table = model.summary(0.5, alpha_ci=0.1, **kw)
    np.testing.assert_array_equal(table["coef lower 90%"], ci[:, 0])

    # A one-sided 90% bound on a B10 life is the end of the two-sided 80%.
    rows = np.array([[1.0, 0.0], [0.5, 1.0]])
    two = model.quantile_cb(0.1, rows, alpha_ci=0.2, **kw)
    lower = model.quantile_cb(0.1, rows, alpha_ci=0.1, bound="lower", **kw)
    np.testing.assert_array_equal(lower, two[:, 0])
    assert np.all(lower < model.qf(0.1, rows))

    # The bounds on ff and Hf are those on sf, transformed.
    t = [6.0, 9.0]
    sf = model.cb(t, rows[0], **kw)
    np.testing.assert_allclose(
        model.cb(t, rows[0], on="ff", **kw), 1 - sf[:, ::-1]
    )
    np.testing.assert_allclose(
        model.cb(t, rows[0], on="Hf", **kw), -np.log(sf[:, ::-1])
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def test_round_trip_keeps_predictions_and_bootstrap():
    x, Z, c = _hetero(150, 12)
    df = pd.DataFrame({"t": x, "c": c, "dose": Z[:, 0], "group": Z[:, 1]})
    model = QuantileRegression.fit_from_df(
        df, x_col="t", formula="dose + group", c_col="c", method="portnoy"
    )
    restored = sp.from_json(model.to_json())
    assert type(restored) is type(model) and restored.method == "portnoy"
    rows = df[["dose", "group"]].iloc[:4]
    np.testing.assert_array_equal(restored.qf(0.3, rows), model.qf(0.3, rows))
    np.testing.assert_array_equal(
        restored.bootstrap_ci(0.3, n_boot=5, random_state=1),
        model.bootstrap_ci(0.3, n_boot=5, random_state=1),
    )
    assert "Portnoy" in repr(restored)
