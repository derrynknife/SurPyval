"""Confidence bounds for the pseudo-failure-time degradation model.

The degradation analysis is a two-stage estimator (per-unit path fit ->
extrapolate to threshold -> fit a life model to the pseudo failure times), so
the plain life-model MLE covariance -- which treats the pseudo failure times as
exact -- gives intervals that are too narrow. These tests pin the analytic
delta-method correction (and its bootstrap cross-check): that it *widens* the
bounds via the first-stage uncertainty, that it agrees with the bootstrap, and
that it restores coverage of the true reliability the MLE-only bounds lose.
"""

import warnings

import numpy as np
import pytest

from surpyval import Weibull
from surpyval.degradation import DegradationAnalysis
from surpyval.degradation._bounds import (
    _delta_se,
    _life_loglik,
    _logit_bound,
    _num_hessian,
)


def _linear_degradation(seed, n_units=20, noise=2.0, n_points=5, t_max=8.0):
    """Units with linear degradation crossing a threshold of 100 at a
    Weibull-distributed true time, measured only out to ``t_max`` (so the
    threshold is reached by extrapolation)."""
    rng = np.random.default_rng(seed)
    alpha, beta, thr = 50.0, 3.0, 100.0
    T = alpha * rng.weibull(beta, n_units)
    t_meas = np.linspace(1.0, t_max, n_points)
    xs, ys, ids = [], [], []
    for u, Tu in enumerate(T):
        slope = thr / Tu
        y = slope * t_meas + rng.normal(0.0, noise, t_meas.size)
        xs.append(t_meas)
        ys.append(y)
        ids.append(np.full(t_meas.size, u))
    return (
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ids),
        thr,
        (alpha, beta),
    )


def _fit(seed, **kw):
    x, y, i, thr, truth = _linear_degradation(seed, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = DegradationAnalysis.fit(
            x, y, i, threshold=thr, path="linear", distribution=Weibull
        )
    return m, truth


def _mle_only_se(model, x):
    # The life-model MLE covariance alone (H^{-1}), i.e. the bound the current
    # life_model.cb would use, treating pseudo failure times as exact.
    phi = np.asarray(model.life_model.params, dtype=float)
    t = model.pseudo_failure_times
    cov = np.linalg.inv(
        -_num_hessian(lambda p: _life_loglik(model, p, t), phi)
    )
    x = np.atleast_1d(np.asarray(x, dtype=float))
    return _delta_se(lambda p: model.life_model.dist.sf(x, *p), phi, cov)


# --- the mechanism: the correction widens the covariance ------------------


def test_analytic_covariance_is_wider_than_mle():
    m, _ = _fit(0)
    cov_corr = m.life_parameter_covariance("analytic")
    phi = np.asarray(m.life_model.params, dtype=float)
    t = m.pseudo_failure_times
    cov_mle = np.linalg.inv(
        -_num_hessian(lambda p: _life_loglik(m, p, t), phi)
    )
    se_corr = np.sqrt(np.diag(cov_corr))
    se_mle = np.sqrt(np.diag(cov_mle))
    # Every parameter's standard error grows once the first-stage uncertainty
    # is folded in.
    assert np.all(se_corr > se_mle)


def test_exact_fits_give_no_correction():
    # Two points per unit and no measurement noise => each path fits exactly,
    # measurement_var is 0, so there is no first-stage uncertainty and the
    # analytic covariance reduces to the plain MLE covariance.
    rng = np.random.default_rng(1)
    thr = 100.0
    xs, ys, ids = [], [], []
    for u in range(15):
        slope = rng.uniform(2.0, 6.0)
        t = np.array([1.0, 3.0])
        xs.append(t)
        ys.append(slope * t)  # exact line, no noise
        ids.append(np.full(2, u))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = DegradationAnalysis.fit(
            np.concatenate(xs),
            np.concatenate(ys),
            np.concatenate(ids),
            threshold=thr,
            path="linear",
            distribution=Weibull,
        )
    assert m.measurement_var == 0.0
    cov_corr = m.life_parameter_covariance("analytic")
    phi = np.asarray(m.life_model.params, dtype=float)
    t = m.pseudo_failure_times
    cov_mle = np.linalg.inv(
        -_num_hessian(lambda p: _life_loglik(m, p, t), phi)
    )
    assert np.allclose(cov_corr, cov_mle)


# --- shape / range / direction of the bounds ------------------------------


def test_cb_brackets_and_bounded():
    m, _ = _fit(2)
    x = np.array([20.0, 35.0, 50.0])
    cb = m.cb(x, on="sf")
    assert cb.shape == (3, 2)
    sf = m.sf(x)
    assert np.all(cb[:, 0] <= sf) and np.all(sf <= cb[:, 1])
    assert np.all(cb >= 0) and np.all(cb <= 1)
    assert np.all(cb[:, 0] <= cb[:, 1])


@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_cb_on_variants_bracket(on):
    m, _ = _fit(3)
    x = np.array([25.0, 45.0])
    est = getattr(m, on)(x)
    cb = m.cb(x, on=on)
    assert np.all(cb[:, 0] <= est) and np.all(est <= cb[:, 1])
    if on in ("sf", "ff"):
        assert np.all(cb >= 0) and np.all(cb <= 1)
    else:
        assert np.all(cb >= 0)


def test_one_sided_bounds_ordered():
    m, _ = _fit(4)
    x = np.array([30.0, 45.0])
    for on in ("sf", "ff", "Hf"):
        est = getattr(m, on)(x)
        lower = m.cb(x, on=on, bound="lower")
        upper = m.cb(x, on=on, bound="upper")
        assert np.all(lower <= est) and np.all(est <= upper)


# --- analytic vs bootstrap ------------------------------------------------


def test_analytic_and_bootstrap_agree():
    m, _ = _fit(5)
    x = np.array([25.0, 40.0])
    an = m.cb(x, on="sf", method="analytic")
    bs = m.cb(x, on="sf", method="bootstrap", n_boot=150, seed=7)
    # Same ballpark: the two 95% intervals overlap substantially at each point.
    for a, b in zip(an, bs):
        lo = max(a[0], b[0])
        hi = min(a[1], b[1])
        overlap = max(0.0, hi - lo)
        assert overlap > 0.4 * (a[1] - a[0])


# --- coverage: the correction restores what the MLE-only bound loses ------


def test_coverage_improves_over_mle_only():
    # In a heavy-extrapolation regime (few noisy points, threshold far beyond
    # the measurement window) the MLE-only bound under-covers the true
    # reliability; the analytic bound folds the first-stage error back in and
    # covers at least as well, close to nominal.
    t0 = 45.0
    true_sf = float(Weibull.from_params([50.0, 3.0]).sf(t0))
    an_hits, mle_hits = 0, 0
    reps = 80
    for s in range(reps):
        m, _ = _fit(1000 + s, n_units=18, noise=3.0, n_points=3, t_max=4.0)
        an = m.cb([t0], on="sf", method="analytic")[0]
        se = _mle_only_se(m, [t0])
        sf_hat = m.sf([t0])
        mlo = _logit_bound(sf_hat, se, 0.05, "lower")[0]
        mhi = _logit_bound(sf_hat, se, 0.05, "upper")[0]
        an_hits += an[0] <= true_sf <= an[1]
        mle_hits += mlo <= true_sf <= mhi
    an_cov = an_hits / reps
    mle_cov = mle_hits / reps
    # Analytic reaches (at least) nominal coverage and never covers worse than
    # the uncorrected bound.
    assert an_cov >= 0.90
    assert mle_cov <= an_cov


# --- guards ---------------------------------------------------------------


def test_cb_rejects_bad_arguments():
    m, _ = _fit(6)
    with pytest.raises(ValueError, match="`on` must be one of"):
        m.cb([10.0], on="nonsense")
    with pytest.raises(ValueError, match="`bound` must be"):
        m.cb([10.0], bound="sideways")
    with pytest.raises(ValueError, match="`method` must be"):
        m.cb([10.0], method="magic")


def test_analytic_rejects_lfp_life_model():
    # The analytic correction is only implemented for a plain life model; a
    # limited-failure-population component must route to the bootstrap.
    m, _ = _fit(7)
    m.life_model.p = 0.8  # pretend an LFP was fitted
    with pytest.raises(ValueError, match="limited-failure-population"):
        m.cb([10.0], on="sf", method="analytic")
