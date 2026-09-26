"""Regression tests for the third round of degradation bug fixes.

* The analytic two-sided ``cb`` puts ``alpha_ci / 2`` in each tail (it was
  a 90% band labelled 95%).
* Wiener ``sf``/``ff``/``hf`` stay finite when ``2 D mu / sigma**2`` is
  large, and ``sigma = 0`` is refused clearly.
* Units already past the threshold at their first measurement are failed
  (left censored), not survivors; ``predict_rul`` / ``predict_failure_time``
  / ``induced_life`` treat them as failed at time zero.
* ``GammaProcessModel.mean`` is right when ``beta * threshold`` is large;
  zero gamma increments are censored at the measurement resolution.
* Quadratic ``inv_path`` is stable for near-zero curvature.
* ``predict_rul`` quantiles reaching into the never-fails mass are ``inf``.
* Single-stress ADT fits, bad destructive input, an unseeded ``random``,
  noise-free REML clocks and a handful of argument checks.
"""

import json
import warnings
from typing import Any, cast

import numpy as np
import pytest
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammainc

from surpyval import Logistic, Normal
from surpyval.degradation import (
    DegradationAnalysis,
    DegradationModel,
    DestructiveDegradation,
    DestructiveDegradationModel,
    GammaProcess,
    GammaProcessModel,
    QuadraticPath,
    WienerProcess,
    WienerProcessModel,
)


def _linear_units(
    n: int = 6, seed: int = 0, noise: float = 0.3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rising linear units measured at t = 1..8, crossing 15 around t = 14."""
    rng = np.random.default_rng(seed)
    t = np.arange(1.0, 9.0)
    xs, ys, ids = [], [], []
    for u in range(n):
        b = rng.normal(1.0, 0.2)
        xs.append(t)
        ys.append(1 + b * t + rng.normal(0, noise, t.size))
        ids.append(np.full(t.size, u))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)


def _with_extra_units() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The linear units plus one already past 15 at t = 1 (unit 6) and one
    below it trending away (unit 7)."""
    x, y, i = _linear_units()
    rng = np.random.default_rng(1)
    t = np.arange(1.0, 9.0)
    x = np.concatenate([x, t, t])
    y = np.concatenate(
        [
            y,
            16 + t + rng.normal(0, 0.3, t.size),
            5 - 0.5 * t + rng.normal(0, 0.3, t.size),
        ]
    )
    i = np.concatenate([i, np.full(t.size, 6), np.full(t.size, 7)])
    return x, y, i


# -- two-sided analytic bounds ----------------------------------------------


def test_analytic_two_sided_cb_puts_half_alpha_in_each_tail() -> None:
    x, y, i = _linear_units(12)
    model = DegradationAnalysis.fit(x, y, i, threshold=15.0)
    t = np.array([10.0, 14.0, 18.0])
    for on in ("sf", "ff", "Hf"):
        two = model.cb(t, on=on)
        lower = model.cb(t, on=on, bound="lower", alpha_ci=0.025)
        upper = model.cb(t, on=on, bound="upper", alpha_ci=0.025)
        assert np.allclose(two[:, 0], lower)
        assert np.allclose(two[:, 1], upper)
    # the old band equalled the 5% one-sided bound (z = 1.645)
    assert not np.allclose(
        model.cb(t)[:, 0], model.cb(t, bound="lower", alpha_ci=0.05)
    )


# -- Wiener process ---------------------------------------------------------


def test_wiener_life_finite_for_large_drift_to_noise() -> None:
    model = WienerProcessModel(1.0, 0.3, 35.0)
    nu, lam = 35.0, 35.0**2 / 0.09
    ref = stats.invgauss(mu=nu / lam, scale=lam)
    t = np.array([30.0, 35.0, 40.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sf = model.sf(t)
        ff = model.ff(t)
        hf = np.asarray(model.hf(np.array([35.0, 1e3, 1e6])))
        Hf = np.asarray(model.Hf(np.array([40.0, 100.0])))
    assert np.allclose(sf, ref.sf(t), rtol=1e-9)
    assert np.allclose(ff, ref.cdf(t), rtol=1e-9)
    # the hazard tends to mu**2 / (2 sigma**2) far in the tail
    assert np.isfinite(hf).all()
    assert hf[-1] == pytest.approx(1.0 / (2 * 0.09), rel=1e-3)
    assert np.isfinite(Hf).all() and Hf[1] > Hf[0] > 0


def test_wiener_realistic_fit_quantiles_and_rul() -> None:
    rng = np.random.default_rng(0)
    t = np.tile(np.arange(0, 61.0), 5)
    i = np.repeat(np.arange(5), 61)
    y = np.hstack(
        [np.r_[0, np.cumsum(rng.normal(1, 0.3, 60))] for _ in range(5)]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = WienerProcess.fit(t, y, i, threshold=40)
        sf = model.sf([35.0, 40.0, 45.0])
        q = model.qf([0.1, 0.5])
        rul = model.predict_rul(20.0)
    assert np.isfinite(sf).all() and np.all(np.diff(sf) < 0)
    assert np.allclose(model.ff(q), [0.1, 0.5])
    assert 15 < rul.rul < 25


def test_wiener_noise_free_is_refused() -> None:
    with pytest.raises(ValueError, match="sigma is 0"):
        WienerProcess.fit([0, 1, 2, 3], [0, 1, 2, 3], [1, 1, 1, 1], 10.0)
    with pytest.raises(ValueError, match="sigma must be positive"):
        WienerProcessModel(1.0, 0.0, 10.0)


def test_process_predict_rul_alpha_ci_validated() -> None:
    model = WienerProcessModel(0.5, 0.4, 10.0)
    with pytest.raises(ValueError, match="alpha_ci"):
        model.predict_rul(2.0, alpha_ci=1.5)


# -- units already past the threshold ---------------------------------------


def test_unit_past_threshold_at_start_is_left_censored() -> None:
    x, y, i = _with_extra_units()
    with pytest.warns(UserWarning) as caught:
        model = DegradationAnalysis.fit(x, y, i, threshold=15.0)
    messages = " ".join(str(w.message) for w in caught)
    assert "already past the threshold" in messages
    assert "never reach" in messages  # unit 7, trending away
    assert model.c[6] == -1 and model.pseudo_failure_times[6] == 1.0
    assert model.c[7] == 1
    assert "Failed Before Start : 1" in repr(model)
    # the life likelihood (and so the analytic bounds) include the unit
    band = model.cb([5.0, 10.0])
    assert np.isfinite(band).all()
    # and it round-trips
    restored = DegradationModel.from_dict(
        json.loads(json.dumps(model.to_dict()))
    )
    assert np.array_equal(restored.c, model.c)


def test_left_censored_at_first_positive_time_when_starting_at_zero() -> None:
    x, y, i = _linear_units()
    t = np.arange(0.0, 8.0)
    x = np.concatenate([x, t])
    y = np.concatenate([y, 16 + t])
    i = np.concatenate([i, np.full(t.size, 9)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = DegradationAnalysis.fit(x, y, i, threshold=15.0)
    assert model.c[-1] == -1 and model.pseudo_failure_times[-1] == 1.0


def test_predictions_for_a_trajectory_already_past_threshold() -> None:
    # units whose starting levels vary widely, so the posterior of a new
    # unit's intercept follows its own (already failed) trajectory
    rng = np.random.default_rng(2)
    t = np.arange(1.0, 9.0)
    x = np.tile(t, 10)
    i = np.repeat(np.arange(10), t.size)
    a = np.repeat(rng.normal(4.0, 5.0, 10), t.size)
    b = np.repeat(rng.normal(1.0, 0.2, 10), t.size)
    y = a + b * x + rng.normal(0, 0.3, x.size)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = DegradationAnalysis.fit(x, y, i, threshold=30.0)
    t = np.arange(1.0, 9.0)
    t = np.arange(1.0, 9.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pred = model.predict_rul(t, 31 + t, random_state=0)
        crossing = model.predict_failure_time(t, 31 + t)
    assert pred.prob_failed == 1.0
    assert pred.prob_never_fails == 0.0
    assert pred.failure_time == 0.0
    assert pred.rul == -8.0
    assert crossing == pytest.approx(-1.0)
    # a unit below the threshold moving away still never reaches it
    with pytest.warns(UserWarning, match="never reaches"):
        assert np.isnan(model.predict_failure_time(t, 20 - 0.5 * t))


def test_induced_life_counts_started_draws_as_failures_at_zero() -> None:
    x, y, i = _with_extra_units()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = DegradationAnalysis.fit(x, y, i, threshold=15.0)
    induced = model.induced_life(random_state=0)
    at_zero = induced.ff(0.0)
    assert at_zero > 0
    assert np.all(induced.samples[induced.samples <= 0] == 0.0)
    # every draw is a failure at zero, a positive crossing, or never fails
    assert induced.prob_never_fails + induced.ff(1e9) == pytest.approx(1.0)


# -- gamma process ----------------------------------------------------------


def test_gamma_mean_with_large_beta_threshold() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean = GammaProcessModel(3.0, 1500.0, 30.0).mean()
    # E[T] ~ (beta D + 1/2) / alpha for a gamma process
    assert mean == pytest.approx((1500.0 * 30.0 + 0.5) / 3.0, rel=1e-6)
    assert GammaProcessModel(3.0, 1.5, 30.0).mean() == pytest.approx(
        (45.0 + 0.5) / 3.0, rel=1e-4
    )


def _gamma_data(
    step: "float | None",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    xs, ys, ids = [], [], []
    for u in range(10):
        t = np.arange(0, 21.0)
        dy = rng.gamma(2.0, 1 / 4.0, 20)
        xs.append(t)
        ys.append(np.r_[0, np.cumsum(dy)])
        ids.append(np.full(21, u))
    x, y, i = np.concatenate(xs), np.concatenate(ys), np.concatenate(ids)
    if step is not None:
        y = np.round(y / step) * step
    return x, y, i


def test_gamma_zero_increments_censored_at_resolution() -> None:
    x, y, i = _gamma_data(0.2)
    dt = np.concatenate([np.diff(x[i == u]) for u in range(10)])
    dy = np.concatenate([np.diff(y[i == u]) for u in range(10)])
    zero = dy == 0
    assert zero.sum() > 0
    model = GammaProcess.fit(x, y, i, threshold=10.0)

    # the censored likelihood, maximised directly
    def nll(v: np.ndarray) -> float:
        a, b = np.exp(v)
        pos = stats.gamma.logpdf(dy[~zero], a * dt[~zero], scale=1 / b).sum()
        cens = np.log(gammainc(a * dt[zero], b * 0.2)).sum()
        return -(pos + cens)

    ref = minimize(
        nll, [0.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-9}
    )
    assert np.allclose([model.alpha, model.beta], np.exp(ref.x), rtol=1e-4)
    # the old 1e-12 nudge cut alpha about six-fold; now it stays near the
    # fit to the unrounded data
    exact = GammaProcess.fit(*_gamma_data(None), threshold=10.0)
    assert model.alpha == pytest.approx(exact.alpha, rel=0.25)
    # an explicit resolution is used as given
    coarse = GammaProcess.fit(x, y, i, threshold=10.0, resolution=0.4)
    assert coarse.alpha != pytest.approx(model.alpha)


def test_gamma_no_zero_increments_unchanged_and_resolution_checked() -> None:
    x, y, i = _gamma_data(None)
    a = GammaProcess.fit(x, y, i, threshold=10.0)
    b = GammaProcess.fit(x, y, i, threshold=10.0, resolution=0.5)
    assert (a.alpha, a.beta) == (b.alpha, b.beta)
    with pytest.raises(ValueError, match="resolution"):
        GammaProcess.fit(*_gamma_data(0.2), threshold=10.0, resolution=-1.0)
    with pytest.raises(ValueError, match="every increment is zero"):
        GammaProcess.fit([0, 1, 2], [1.0, 1.0, 1.0], [1, 1, 1], 10.0)


def test_gamma_zero_increments_with_stress() -> None:
    rng = np.random.default_rng(3)
    xs, ys, ids, Zs = [], [], [], []
    for u, z in enumerate(np.repeat([0.0, 1.0], 8)):
        t = np.arange(0, 21.0)
        dy = rng.gamma(2.0 * np.exp(0.7 * z), 1 / 4.0, 20)
        xs.append(t)
        ys.append(np.round(np.r_[0, np.cumsum(dy)], 1))
        ids.append(np.full(21, u))
        Zs.append(np.full(21, z))
    x, y, i, Z = (np.concatenate(v) for v in (xs, ys, ids, Zs))
    model = GammaProcess.fit(x, y, i, threshold=10.0, Z=Z, stress_ref=[0.0])
    assert model.gamma is not None
    assert model.gamma[0] == pytest.approx(0.7, abs=0.2)


# -- path models, quantiles, input checks -----------------------------------


def test_quadratic_inv_path_near_zero_curvature() -> None:
    x = np.arange(1.0, 11.0)
    params = QuadraticPath.fit(x, 2 + 3 * x)
    assert QuadraticPath.inv_path(50.0, *params) == pytest.approx(16.0)
    for c in (1e-17, -1e-17, 1e-10):
        assert QuadraticPath.inv_path(100.0, 0.0, 1.0, c) == pytest.approx(
            100.0, rel=1e-6
        )
    # genuine roots are unchanged
    assert QuadraticPath.inv_path(100.0, 0.0, -1.0, 1.0) == pytest.approx(
        (1 + np.sqrt(401)) / 2
    )


def test_predict_rul_quantiles_inf_without_warnings() -> None:
    rng = np.random.default_rng(0)
    xs, ys, ids = [], [], []
    for u in range(8):
        t = np.arange(1.0, 11.0)
        xs.append(t)
        ys.append(1 + rng.normal(0.05, 0.2) * t + rng.normal(0, 0.3, t.size))
        ids.append(np.full(t.size, u))
    x, y, i = map(np.concatenate, (xs, ys, ids))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = DegradationAnalysis.fit(x, y, i, threshold=5.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        pred = model.predict_rul([1, 2, 3], [1.0, 0.9, 0.8], random_state=0)
        flat = model.predict_rul([1, 2, 3], [1.0, 0.5, 0.0], random_state=0)
    assert pred.failure_time_interval[1] == np.inf
    assert pred.rul_interval[1] == np.inf
    assert np.isfinite(pred.failure_time)
    assert flat.prob_never_fails > 0.5
    assert flat.failure_time == np.inf and flat.rul == np.inf


def test_single_stress_level_refused() -> None:
    x, y, i = _linear_units()
    with pytest.raises(ValueError, match="two\\s+distinct stress levels"):
        DegradationAnalysis.fit(x, y, i, threshold=15.0, Z=np.ones(x.size))
    with pytest.raises(ValueError, match="stress effect cannot be estimated"):
        DegradationAnalysis.fit(
            x, y, i, threshold=15.0, Z=np.ones(x.size), links={"b": "log"}
        )


def test_plain_random_honours_seed() -> None:
    x, y, i = _linear_units()
    model = DegradationAnalysis.fit(x, y, i, threshold=15.0)
    a = model.random(5, random_state=0)
    b = model.random(5, random_state=0)
    assert np.array_equal(a, b)
    assert np.all(a > 0)


def test_minor_argument_checks() -> None:
    x, y, i = _linear_units()
    model = DegradationAnalysis.fit(
        x, y, i, threshold=cast(Any, np.array(15.0))
    )
    assert model.threshold == 15.0
    with pytest.raises(ValueError, match="n_samples"):
        model.predict_rul([1.0, 2.0], [2.0, 3.0], n_samples=0)
    for alpha in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError, match="alpha_ci"):
            model.predict_rul([1.0, 2.0], [2.0, 3.0], alpha_ci=alpha)
    with pytest.raises(ValueError, match="not one of the model's units"):
        model.path([1.0], 42)


# -- REML clock without noise -----------------------------------------------


def test_reml_clock_on_noise_free_data_refused_early() -> None:
    levels = np.array([0.0, 0.5, 1.0])
    times = np.arange(1.0, 31.0)
    zrow = np.select([times <= 10, times <= 20], levels[:2], levels[2])
    af = np.exp(2.0 * zrow)
    rng = np.random.default_rng(0)
    xs, ys, ids, Zs = [], [], [], []
    for u in range(6):
        a, b = rng.normal([1.0, 0.2], [0.2, 0.03])
        tau = np.cumsum(np.diff(np.r_[0.0, times]) * af)
        xs.append(times)
        ys.append(a + b * tau)
        ids.append(np.full(times.size, u))
        Zs.append(zrow)
    x, y, i, Z = map(np.concatenate, (xs, ys, ids, Zs))
    for noise in (0.0, 1e-8):
        yy = y + noise * rng.normal(size=y.size)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            with pytest.raises(
                ValueError, match="population_method='moments'"
            ):
                DegradationAnalysis.fit(
                    x,
                    yy,
                    i,
                    threshold=20.0,
                    Z=Z,
                    acceleration="clock",
                    stress_ref=[0.0],
                    population_method="reml",
                )
    # moments recovers the clock exactly
    model = DegradationAnalysis.fit(
        x, y, i, threshold=20.0, Z=Z, acceleration="clock", stress_ref=[0.0]
    )
    assert model.gamma is not None
    assert model.gamma[0] == pytest.approx(2.0)


# -- destructive degradation -------------------------------------------------


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
