"""
Tests for truncation handling in the univariate parametric regression
fitters: proportional hazards, proportional odds, accelerated failure time
and accelerated life (parameter substitution).

With covariates fixed at zero the regression term ``phi(Z) = exp(beta'Z)``
collapses to one, so each model must reproduce the truncation-corrected
baseline fit produced by the plain parametric distribution. This anchors
the regression likelihoods to SurPyval's known-correct base implementation.
"""

import numpy as np
import pytest

from surpyval import Weibull, WeibullPH, WeibullPO
from surpyval.univariate.regression import AFT, AcceleratedLife
from surpyval.univariate.regression.accelerated_life import Power


def _left_truncated_data(seed=0, n=1000, alpha=10.0, beta=2.0, tl=3.0):
    rng = np.random.default_rng(seed)
    x = alpha * rng.weibull(beta, n)
    x = x[x > tl]
    t = np.column_stack([np.full(x.size, tl), np.full(x.size, np.inf)])
    Z = np.zeros((x.size, 1))
    return x, Z, t, tl


@pytest.mark.parametrize("fitter", [WeibullPH, WeibullPO, AFT(Weibull)])
def test_left_truncation_matches_baseline(fitter):
    # phi(0) = 1, so the regression fit must equal the truncated baseline.
    x, Z, t, tl = _left_truncated_data()
    baseline = Weibull.fit(x=x, tl=tl)
    model = fitter.fit(x=x, Z=Z, t=t)

    assert np.allclose(model.params[:2], baseline.params, atol=1e-2)
    # The regression coefficient should be ~0 for a constant covariate.
    assert np.allclose(model.params[2:], 0.0, atol=1e-2)


@pytest.mark.parametrize("fitter", [WeibullPH, WeibullPO, AFT(Weibull)])
def test_truncation_changes_estimate(fitter):
    # Ignoring left truncation biases the scale upward; correcting for it
    # must pull the estimate back toward the truth.
    x, Z, t, tl = _left_truncated_data()
    truncated = fitter.fit(x=x, Z=Z, t=t)
    naive = fitter.fit(x=x, Z=Z)

    assert not np.allclose(truncated.params[:2], naive.params[:2], atol=1e-2)
    # Truncation-corrected scale should be the smaller (less biased) one.
    assert truncated.params[0] < naive.params[0]


def test_interval_censoring_matches_baseline():
    # The interval-censoring term must equal F(xr) - F(xl), matching the
    # base parametric fit when covariates are zero.
    rng = np.random.default_rng(1)
    x = 10.0 * rng.weibull(2.0, 400)
    xl = np.floor(x)
    xr = np.ceil(x + 1e-9)
    baseline = Weibull.fit(xl=xl, xr=xr)

    xc = [[a, b] for a, b in zip(xl, xr)]
    c = np.full(x.size, 2)
    Z = np.zeros((x.size, 1))
    model = WeibullPH.fit(x=xc, Z=Z, c=c)

    assert np.allclose(model.params[:2], baseline.params, atol=1e-2)


def test_accelerated_life_handles_truncation():
    # AcceleratedLife (parameter substitution) must accept truncated data and
    # produce a different fit than when truncation is ignored.
    rng = np.random.default_rng(2)
    tl = 2.0
    xs, Zs, ts = [], [], []
    for stress, alpha in [(1.0, 8.0), (2.0, 14.0)]:
        xi = alpha * rng.weibull(2.0, 500)
        xi = xi[xi > tl]
        xs.append(xi)
        Zs.append(np.full((xi.size, 1), stress))
        ts.append(
            np.column_stack(
                [np.full(xi.size, tl), np.full(xi.size, np.inf)]
            )
        )
    x = np.concatenate(xs)
    Z = np.concatenate(Zs)
    t = np.concatenate(ts)

    with_trunc = AcceleratedLife(Weibull, Power).fit(x=x, Z=Z, t=t)
    ignore_trunc = AcceleratedLife(Weibull, Power).fit(x=x, Z=Z)

    assert np.isfinite(with_trunc.neg_ll())
    assert not np.allclose(with_trunc.params, ignore_trunc.params, atol=1e-2)


def test_right_truncation_runs():
    # Right truncation (tr finite, tl = -inf) must be handled without error
    # and recover the baseline fit for zero covariates.
    rng = np.random.default_rng(3)
    tr = 18.0
    x = 10.0 * rng.weibull(2.0, 1000)
    x = x[x < tr]
    t = np.column_stack([np.full(x.size, -np.inf), np.full(x.size, tr)])
    Z = np.zeros((x.size, 1))

    baseline = Weibull.fit(x=x, tr=tr)
    model = WeibullPH.fit(x=x, Z=Z, t=t)

    assert np.allclose(model.params[:2], baseline.params, atol=1e-2)
