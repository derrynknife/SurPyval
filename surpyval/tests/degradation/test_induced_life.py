"""The Lu-Meeker induced failure-time distribution.

``DegradationModel.induced_life`` derives the population failure-time
distribution directly from the fitted path-parameter distribution -- drawing
``theta ~ N(path_param_mean, path_param_cov)`` and pushing each draw through
the path model's ``inv_path(threshold)`` -- instead of via each unit's noisy
pseudo failure time. These tests check that it agrees with the pseudo-failure
life fit on well-behaved data, that its distribution methods are internally
consistent, that a non-increasing population produces a "never fails" mass,
that it is reproducible, and that it is refused for accelerated models.
"""

import numpy as np
import pytest

from surpyval.degradation import DegradationAnalysis
from surpyval.degradation.degradation_analysis import (
    InducedFailureDistribution,
)


def _simulate_linear(threshold, units, seed, b_mean=1.0, b_sd=0.25):
    rng = np.random.default_rng(seed)
    xs, ys, ids = [], [], []
    for u in range(units):
        a = rng.normal(0.0, 0.2)
        b = rng.normal(b_mean, b_sd)
        t = np.arange(0, 20, 2.0)
        y = a + b * t + rng.normal(0, 0.4, t.size)
        xs.append(t)
        ys.append(y)
        ids.append(np.full(t.size, u))
    return (np.concatenate(z) for z in (xs, ys, ids))


def test_induced_life_agrees_with_pseudo_failure_fit():
    x, y, i = _simulate_linear(threshold=30.0, units=50, seed=0)
    model = DegradationAnalysis.fit(x, y, i, threshold=30.0, path="linear")

    induced = model.induced_life(n_samples=20000, random_state=1)
    assert isinstance(induced, InducedFailureDistribution)
    # true crossing time is ~ threshold / b_mean = 30; both routes should
    # land near it and near each other
    assert abs(induced.median() - float(model.qf(0.5))) < 3.0
    assert abs(induced.qf(0.1) - float(model.qf(0.1))) < 4.0
    # the induced CDF tracks the pseudo-failure CDF across the body
    t = np.array([20.0, 30.0, 40.0])
    assert np.allclose(induced.ff(t), model.ff(t), atol=0.1)
    assert induced.prob_never_fails == 0.0


def test_induced_distribution_is_internally_consistent():
    x, y, i = _simulate_linear(threshold=30.0, units=40, seed=2)
    model = DegradationAnalysis.fit(x, y, i, threshold=30.0, path="linear")
    induced = model.induced_life(n_samples=15000, random_state=3)

    t = np.array([15.0, 30.0, 45.0])
    assert np.allclose(induced.sf(t) + induced.ff(t), 1.0)
    # ff is non-decreasing
    grid = np.linspace(0, 80, 40)
    assert np.all(np.diff(induced.ff(grid)) >= -1e-12)
    # qf inverts ff to within a Monte-Carlo tolerance
    for p in (0.25, 0.5, 0.75):
        assert abs(induced.ff(induced.qf(p)) - p) < 0.02
    # scalar in -> scalar out
    assert np.isscalar(induced.ff(30.0)) and np.isscalar(induced.sf(30.0))
    # resampling reproduces the mean
    draws = induced.random(20000, random_state=4)
    assert abs(np.mean(draws) - induced.mean()) < 1.0


def test_induced_life_reports_never_fails_mass():
    # A population whose slope straddles zero: units with b <= 0 never reach
    # the (positive) threshold, so they contribute a defective mass.
    x, y, i = _simulate_linear(
        threshold=30.0, units=60, seed=5, b_mean=0.2, b_sd=0.5
    )
    model = DegradationAnalysis.fit(x, y, i, threshold=30.0, path="linear")
    induced = model.induced_life(n_samples=20000, random_state=6)
    assert induced.prob_never_fails > 0.05
    # the mean is infinite once some draws never fail
    assert np.isinf(induced.mean())
    # a quantile beyond the failing mass is infinite
    assert np.isinf(induced.qf(1.0 - induced.prob_never_fails / 2.0))
    # but a low quantile is finite
    assert np.isfinite(induced.qf(0.1))


def test_induced_life_is_reproducible():
    x, y, i = _simulate_linear(threshold=30.0, units=40, seed=7)
    model = DegradationAnalysis.fit(x, y, i, threshold=30.0, path="linear")
    a = model.induced_life(n_samples=5000, random_state=42)
    b = model.induced_life(n_samples=5000, random_state=42)
    assert np.array_equal(a.samples, b.samples)
    assert a.median() == b.median()


def test_induced_life_works_for_nonlinear_path():
    # Exponential path y = a * exp(b t); still recovers a sensible life.
    rng = np.random.default_rng(8)
    xs, ys, ids = [], [], []
    for u in range(50):
        a = rng.normal(1.0, 0.1)
        b = rng.normal(0.15, 0.02)
        t = np.arange(0, 16, 2.0)
        y = a * np.exp(b * t) * (1 + rng.normal(0, 0.03, t.size))
        xs.append(t)
        ys.append(y)
        ids.append(np.full(t.size, u))
    x, y, i = (np.concatenate(z) for z in (xs, ys, ids))
    model = DegradationAnalysis.fit(
        x, y, i, threshold=10.0, path="exponential"
    )
    induced = model.induced_life(n_samples=10000, random_state=9)
    assert induced.median() > 0
    assert np.isfinite(induced.median())


def test_induced_life_rejected_for_accelerated_model():
    # Build a covariate (ADT) model and confirm induced_life refuses it.
    rng = np.random.default_rng(10)
    xs, ys, ids, Zs = [], [], [], []
    uid = 0
    for stress in [0.0, 0.5, 1.0]:
        for _ in range(15):
            b = (1.0 + stress) * rng.normal(1.0, 0.1)
            t = np.arange(0, 20, 2.0)
            y = b * t + rng.normal(0, 0.4, t.size)
            xs.append(t)
            ys.append(y)
            ids.append(np.full(t.size, uid))
            Zs.append(np.full(t.size, stress))
            uid += 1
    x, y, i, Z = (np.concatenate(a) for a in (xs, ys, ids, Zs))
    model = DegradationAnalysis.fit(
        x, y, i, threshold=30.0, path="linear", Z=Z
    )
    with pytest.raises(ValueError, match="accelerated"):
        model.induced_life()
