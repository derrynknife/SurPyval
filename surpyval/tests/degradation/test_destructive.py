r"""
Destructive degradation modelling (#153).

Each unit yields one destructive ``(time, degradation)`` measurement, so the
population degradation distribution is fit directly as a location-scale
regression on a time transform, and the lifetime distribution is induced by
crossing a threshold. These tests pin recovery of the known trend/scale, the
threshold-to-lifetime mapping (both directions), censored measurements, the
AICc transform selection, bootstrap bounds, and serialisation.
"""

import json

import numpy as np
import pytest

from surpyval import LogNormal, Normal
from surpyval.degradation import (
    DestructiveDegradation,
    DestructiveDegradationModel,
)


def _increasing(seed=0, n=400, a=1.0, b=0.05, sigma=0.25):
    # log Y ~ Normal(a + b t, sigma): positive, increasing degradation.
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, 40, n)
    y = np.exp(a + b * t + rng.normal(0, sigma, n))
    return t, y


def test_recovers_trend_and_scale():
    t, y = _increasing()
    Df = np.exp(1.0 + 0.05 * 30)
    m = DestructiveDegradation.fit(t, y, threshold=Df, distribution=LogNormal)
    assert m.direction == "increasing"
    assert np.allclose(m.beta, [1.0, 0.05], atol=0.03)
    assert abs(m.sigma - 0.25) < 0.03


def test_threshold_induces_lifetime_increasing():
    t, y = _increasing()
    # threshold = median degradation at t=30 -> ~50% failed by t=30
    Df = np.exp(1.0 + 0.05 * 30)
    m = DestructiveDegradation.fit(t, y, threshold=Df, distribution=LogNormal)
    assert abs(float(m.sf(30.0)) - 0.5) < 0.05
    # reliability is monotone non-increasing in time
    tt = np.linspace(1, 60, 40)
    s = m.sf(tt)
    assert np.all(np.diff(s) <= 1e-9)
    assert np.allclose(m.ff(tt) + m.sf(tt), 1.0)
    assert np.all(m.df(tt) >= -1e-9)


def test_decreasing_direction_auto_detected():
    # strength loss: Y ~ Normal(100 - 1.5 t, 5); fails when strength <= Df
    rng = np.random.default_rng(1)
    n = 300
    t = rng.uniform(0, 40, n)
    y = 100.0 - 1.5 * t + rng.normal(0, 5.0, n)
    Df = 100.0 - 1.5 * 25
    m = DestructiveDegradation.fit(t, y, threshold=Df, distribution=Normal)
    assert m.direction == "decreasing"
    assert abs(float(m.sf(25.0)) - 0.5) < 0.06
    assert np.all(np.diff(m.sf(np.linspace(1, 40, 30))) <= 1e-9)


def test_explicit_direction_override():
    t, y = _increasing()
    Df = np.exp(1.0 + 0.05 * 30)
    m = DestructiveDegradation.fit(
        t, y, threshold=Df, distribution=LogNormal, direction="increasing"
    )
    assert m.direction == "increasing"


def test_scalar_and_vector_output():
    t, y = _increasing()
    Df = np.exp(1.0 + 0.05 * 30)
    m = DestructiveDegradation.fit(t, y, threshold=Df, distribution=LogNormal)
    assert np.isscalar(m.sf(30.0)) or np.ndim(m.sf(30.0)) == 0
    assert m.sf(np.array([10.0, 30.0, 50.0])).shape == (3,)


def test_right_censored_measurements():
    # cap the measurement at a ceiling; values above are right-censored.
    t, y = _increasing(seed=2)
    cap = np.exp(1.0 + 0.05 * 35)
    c = (y > cap).astype(int)
    yc = np.where(c == 1, cap, y)
    Df = np.exp(1.0 + 0.05 * 30)
    m = DestructiveDegradation.fit(
        t, yc, threshold=Df, c=c, distribution=LogNormal
    )
    # censoring is accounted for, so the slope is still ~recovered
    assert np.allclose(m.beta, [1.0, 0.05], atol=0.04)


def test_ignoring_censoring_biases_the_fit():
    # A sanity check that the censoring actually does something: treating the
    # capped values as observed pulls the slope down vs the honest fit.
    t, y = _increasing(seed=2)
    cap = np.exp(1.0 + 0.05 * 35)
    c = (y > cap).astype(int)
    yc = np.where(c == 1, cap, y)
    Df = np.exp(1.0 + 0.05 * 30)
    honest = DestructiveDegradation.fit(
        t, yc, threshold=Df, c=c, distribution=LogNormal
    )
    naive = DestructiveDegradation.fit(
        t, yc, threshold=Df, distribution=LogNormal
    )
    assert honest.beta[1] > naive.beta[1]


def test_transform_best_selects_by_aicc():
    t, y = _increasing()
    Df = np.exp(1.0 + 0.05 * 30)
    m = DestructiveDegradation.fit(
        t, y, threshold=Df, distribution=LogNormal, transform="best"
    )
    # linear data -> linear transform should win, and all scores are recorded
    assert m.transform == "linear"
    assert set(m.transform_scores) == {
        "linear",
        "log",
        "sqrt",
        "reciprocal",
    }


def test_bootstrap_cb_brackets_point_estimate():
    t, y = _increasing()
    Df = np.exp(1.0 + 0.05 * 30)
    m = DestructiveDegradation.fit(t, y, threshold=Df, distribution=LogNormal)
    ts = np.array([20.0, 30.0, 40.0])
    band = m.cb(ts, on="sf", n_boot=120, seed=3)
    assert band.shape == (3, 2)
    assert np.all(band[:, 0] <= band[:, 1])
    point = m.sf(ts)
    # the point estimate lies within the (generous) two-sided band
    assert np.all(band[:, 0] - 1e-6 <= point) and np.all(
        point <= band[:, 1] + 1e-6
    )


def test_round_trip_serialisation():
    t, y = _increasing()
    Df = np.exp(1.0 + 0.05 * 30)
    m = DestructiveDegradation.fit(t, y, threshold=Df, distribution=LogNormal)
    restored = DestructiveDegradationModel.from_dict(
        json.loads(json.dumps(m.to_dict()))
    )
    tt = np.array([15.0, 30.0, 45.0])
    assert np.allclose(m.sf(tt), restored.sf(tt))
    assert np.allclose(
        m.degradation_quantile(0.5, tt), restored.degradation_quantile(0.5, tt)
    )


def test_validation_errors():
    with pytest.raises(ValueError, match="at least 3"):
        DestructiveDegradation.fit([1.0, 2.0], [1.0, 2.0], threshold=5.0)
    t, y = _increasing(n=10)
    with pytest.raises(ValueError, match="c must be"):
        DestructiveDegradation.fit(t, y, threshold=5.0, c=np.full(10, 3))
    with pytest.raises(ValueError, match="same length"):
        DestructiveDegradation.fit(t, y[:-1], threshold=5.0)


def test_from_dict_rejects_wrong_model():
    with pytest.raises(ValueError, match="DestructiveDegradationModel"):
        DestructiveDegradationModel.from_dict({"model": "Other"})
