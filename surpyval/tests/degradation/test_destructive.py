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
from typing import Any, cast

import numpy as np
import pytest

from surpyval import Logistic, LogNormal, Normal
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
    band = m.cb(ts, on="sf", n_boot=120, random_state=3)
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


# ---------------------------------------------------------------------------
# Input checks, and any location-scale distribution
# round-trips.
# ---------------------------------------------------------------------------


def _destructive_data() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    age = rng.uniform(1, 40, 60)
    return age, 100 - 1.5 * age + rng.normal(0, 5, 60)


def test_destructive_bad_input_refused() -> None:
    age, y = _destructive_data()
    with pytest.raises(ValueError, match="positive support"):
        DestructiveDegradation.fit(age, y - 60, threshold=1.0)
    with pytest.raises(ValueError, match="threshold"):
        DestructiveDegradation.fit(
            age, y, threshold=np.nan, distribution=Normal
        )
    with pytest.raises(ValueError, match="finite"):
        DestructiveDegradation.fit(
            np.r_[age, np.nan],
            np.r_[y, 50.0],
            threshold=40.0,
            distribution=Normal,
        )
    for transform in ("log", "reciprocal"):
        with pytest.raises(ValueError, match="time transform"):
            DestructiveDegradation.fit(
                np.r_[0.0, age],
                np.r_[100.0, y],
                threshold=40.0,
                distribution=Normal,
                transform=transform,
            )
    # "best" skips the transforms that are not finite at t = 0
    best = DestructiveDegradation.fit(
        np.r_[0.0, age],
        np.r_[100.0, y],
        threshold=40.0,
        distribution=Normal,
        transform="best",
    )
    assert best.transform_scores is not None
    assert set(best.transform_scores) == {"linear", "sqrt"}
    # a 0-d threshold is a number
    assert (
        DestructiveDegradation.fit(
            age, y, threshold=cast(Any, np.array(40.0)), distribution=Normal
        ).threshold
        == 40.0
    )


def test_destructive_any_distribution_round_trips() -> None:
    age, y = _destructive_data()
    model = DestructiveDegradation.fit(
        age, y, threshold=40.0, distribution=Logistic
    )
    restored = DestructiveDegradationModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    t = np.array([20.0, 40.0])
    assert np.allclose(restored.sf(t), model.sf(t))
    by_name = DestructiveDegradation.fit(
        age, y, threshold=40.0, distribution="Logistic"
    )
    assert np.allclose(by_name.sf(t), model.sf(t))


# -- #564: the fit says what it reached -------------------------------------


def test_564_destructive_fit_records_and_saves_its_maximum(monkeypatch):
    import warnings

    import surpyval.degradation._maximum as maximum_module

    t, y = _increasing(n=60)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = DestructiveDegradation.fit(t, y, threshold=10.0)
        best = DestructiveDegradation.fit(
            t, y, threshold=10.0, transform="best"
        )
    assert model.maximum == "verified" and best.maximum == "verified"
    assert DestructiveDegradationModel.from_dict(model.to_dict()).maximum == (
        "verified"
    )
    old = model.to_dict()
    del old["maximum"]
    assert DestructiveDegradationModel.from_dict(old).maximum == "unknown"
    # noise-free readings: no finite maximum, and only that warning
    with pytest.warns(UserWarning, match="No finite maximum") as caught:
        flat = DestructiveDegradation.fit(t, np.exp(1 + 0.05 * t), 10.0)
    assert flat.maximum == "no finite maximum" and len(caught) == 1
    # an answer that cannot be verified says so, once
    monkeypatch.setattr(
        maximum_module, "verify_or_polish", lambda f, r, n, **k: (r, False)
    )
    with pytest.warns(UserWarning, match="did not reach a verified") as w:
        stalled = DestructiveDegradation.fit(t, y, threshold=10.0)
    assert stalled.maximum == "unverified" and len(w) == 1
    assert "destructive degradation fit" in str(w[0].message)


def _strength_model(**kwargs: Any) -> Any:
    rng = np.random.default_rng(1)
    x = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
    y = np.exp(4.0 - 0.02 * x + rng.normal(0, 0.1, 24))
    return DestructiveDegradation.fit(x, y, threshold=20, **kwargs)


def test_666_qf_inverts_ff_and_random_mean_follow():
    model = _strength_model()
    p = np.array([0.01, 0.1, 0.5, 0.9, 0.99])
    q = model.qf(p)
    np.testing.assert_allclose(model.ff(q), p, rtol=1e-8)
    assert np.all(np.diff(q) > 0)
    assert model.qf(0.0) == 0.0 and np.isinf(model.qf(1.0))
    assert np.isnan(model.qf(np.nan))
    with pytest.warns(UserWarning, match="outside"):
        assert np.isnan(model.qf(1.5))
    # the mean is the integral of sf; the draws are around it
    from scipy.integrate import quad

    direct, _ = quad(lambda t: float(model.sf(t)), 0, 200, limit=200)
    assert model.mean() == pytest.approx(direct, rel=1e-6)
    draws = model.random(4000, random_state=0)
    assert draws.mean() == pytest.approx(model.mean(), rel=0.01)
    np.testing.assert_array_equal(draws, model.random(4000, random_state=0))


def test_666_hf_is_df_over_sf():
    model = _strength_model()
    t = np.array([45.0, 50.0, 55.0])
    np.testing.assert_allclose(model.hf(t), model.df(t) / model.sf(t))
    assert model.hf(500.0) == np.inf


def test_666_qf_inf_where_ff_levels_off_and_refuses_a_falling_ff():
    # a reciprocal transform levels off: some units never cross
    model = _strength_model(transform="reciprocal")
    limit = float(model.ff(1e12))
    assert limit < 1.0
    assert np.isinf(model.qf(min(1.0, limit + 0.01)))
    assert np.isinf(model.mean())
    # a direction against the fitted trend: ff falls with time
    wrong = _strength_model(direction="increasing")
    with pytest.raises(ValueError, match="moves away from the threshold"):
        wrong.qf(0.5)


def test_746_Hf_where_sf_is_one_is_plus_zero():
    # Below the threshold's reach sf is 1, and -log(1) was -0.0, in Hf
    # and in its bounds (#746).
    rng = np.random.default_rng(1)
    x = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
    y = np.exp(4.0 - 0.02 * x + rng.normal(0, 0.1, 24))
    m = DestructiveDegradation.fit(x, y, threshold=20)
    assert np.all(m.sf([0.0, 1.0]) == 1)
    H = m.Hf([0.0, 1.0])
    assert np.all(H == 0) and not np.any(np.signbit(H))
    for bound in ("two-sided", "lower", "upper"):
        b = m.cb([0.0], on="Hf", bound=bound)
        assert not np.any(np.signbit(b))
