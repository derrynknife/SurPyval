"""The second start of the penalised solver.

A Weibull AFT lasso fit on the Rossi data (``fin``, ``age``, ``prio``) at
the penalty ``0.12087690781156954 * 10**-2.25`` (the fourth of a
five-penalty path) used to stop near its start, every coefficient barely
moved from 0: its penalised objective, 1.58917, was above the 1.58111
reached at the larger penalty 0.00382, which no minimum can be (the
minimum of the penalised objective cannot fall as the penalty grows). The
solver now runs a second search from the unpenalised fit whenever the
first fails its optimality check, and keeps the better answer.

No R package fits a penalised parametric AFT model (glmnet's survival
family is the Cox model; ``penalized`` has no AFT; ``penAFT`` is the
semiparametric Gehan AFT), so the answer is checked by what a minimum must
satisfy, computed here from the model's own negative log-likelihood by
finite differences, independently of the solver: the gradient of the
baseline's parameters is 0, and each non-zero coefficient's gradient is
minus the penalty times its sign (each zero one's no larger than the
penalty); and along the path the objective falls as the penalty falls.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.datasets import load_rossi_static
from surpyval.univariate.regression import _penalised
from surpyval.utils.surpyval_data import SurpyvalData

COLS = ["fin", "age", "prio"]
LAMBDA_MAX = 0.12087690781156954
STALLED = LAMBDA_MAX * 10**-2.25  # 0.00067974...


def _rossi():
    df = load_rossi_static()
    x = df["week"].to_numpy(float)
    c = 1 - df["arrest"].to_numpy()
    return x, df[COLS], c


def _objective(model, x, Z, c):
    """The penalised objective at ``model``'s parameters, from the
    model's negative log-likelihood: a function of the baseline's
    parameters and the standardised coefficients."""
    data = SurpyvalData(x, c, group_and_sort=False)
    data.add_covariates(np.asarray(Z, dtype=float))
    N = float(np.sum(data.n))
    scale = model.penalty.scale
    l1, l2 = model.penalty.weights()

    def f(theta):
        b_std = theta[2:]
        nll = sp.WeibullAFT.neg_ll(data, *theta[:2], *(b_std / scale))
        return (
            float(nll) / N
            + float(np.sum(l1 * np.abs(b_std)))
            + 0.5 * float(np.sum(l2 * b_std**2))
        )

    theta = np.asarray(model.params, dtype=float).copy()
    theta[2:] = theta[2:] * scale
    return f, theta


def _smooth_gradient(f, theta, l1):
    """Central differences of the objective less its L1 part."""
    b0 = theta[2:]

    def smooth(th):
        return f(th) - float(np.sum(l1 * np.abs(th[2:])))

    g = np.empty(theta.size)
    for j in range(theta.size):
        h = 1e-6 * max(1.0, abs(theta[j]))
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        if j >= 2 and b0[j - 2] != 0:
            # stay on the coefficient's side of 0
            assert abs(b0[j - 2]) > h
        g[j] = (smooth(up) - smooth(down)) / (2 * h)
    return g


def _fit(lam):
    x, Z, c = _rossi()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return sp.WeibullAFT.fit_penalised(
            x, Z, c=c, penalizer=lam, l1_ratio=1
        )


def test_the_stalled_lasso_fit_reaches_its_minimum():
    model = _fit(STALLED)
    assert model.maximum == "verified"
    x, Z, c = _rossi()
    f, theta = _objective(model, x, Z, c)
    # The objective the model reports is its own, and below the stall.
    assert f(theta) == pytest.approx(model.res.fun, rel=1e-12)
    assert f(theta) < 1.5795
    # The optimality conditions, by finite differences of the likelihood.
    l1, _ = model.penalty.weights()
    g = _smooth_gradient(f, theta, l1)
    np.testing.assert_allclose(g[:2], 0, atol=1e-6)
    b = theta[2:]
    assert np.all(b != 0)
    np.testing.assert_allclose(g[2:], -l1 * np.sign(b), atol=1e-6)
    # Close to its neighbours on the path, as a continuous path must be.
    for near in (STALLED * 1.05, STALLED / 1.05):
        np.testing.assert_allclose(
            model.params[2:], _fit(near).params[2:], rtol=2e-3
        )


def test_the_objective_falls_as_the_penalty_falls():
    lams = LAMBDA_MAX * np.logspace(0, -3, 5)
    x, Z, c = _rossi()
    values = []
    for lam in lams:
        model = _fit(lam)
        assert model.maximum == "verified"
        f, theta = _objective(model, x, Z, c)
        values.append(f(theta))
    # Largest penalty first: each next value is no larger.
    assert np.all(np.diff(values) <= 1e-12)


def test_the_second_search_runs_only_when_the_first_fails(monkeypatch):
    calls = []
    once = _penalised._solve_once

    def counted(*args, **kwargs):
        res = once(*args, **kwargs)
        calls.append(res.verified)
        return res

    monkeypatch.setattr(_penalised, "_solve_once", counted)
    _fit(0.00382)  # a fit that converges from its start
    assert calls == [True]
    calls.clear()
    _fit(STALLED)
    assert calls == [False, True]
