"""Stage 3 of #155: accelerated and step-stress process models."""

import json

import numpy as np
import pytest
from scipy.integrate import quad

import surpyval
from surpyval import StepSchedule
from surpyval.degradation import (
    GammaProcess,
    GammaProcessModel,
    WienerProcess,
    WienerProcessModel,
)

# Arrhenius covariate z = 1/T; a three-step profile 323K -> 348K -> 373K
T_LEVELS = np.array([323.0, 348.0, 373.0])
Z_LEVELS = 1.0 / T_LEVELS
Z_USE = Z_LEVELS[0]
G_TRUE = -5000.0
STEPS = [0.0, 100.0, 200.0]
PROFILE = StepSchedule.from_changepoints(STEPS, Z_LEVELS)
MU, SIGMA = 0.05, 0.12
ALPHA, BETA = 0.2, 4.0
THRESHOLD = 12.0


def af(z):
    return np.exp(G_TRUE * (np.asarray(z) - Z_USE))


def stress_at(t):
    t = np.asarray(t, dtype=float)
    return np.select([t <= 100, t <= 200], Z_LEVELS[:2], Z_LEVELS[2])


def step_stress_data(kind, n_units=40, seed=1, dt=5.0):
    """Every unit follows the same step profile, measured every dt."""
    rng = np.random.default_rng(seed)
    times = np.arange(dt, 300.0 + 1e-9, dt)
    z = stress_at(times)
    xs, ys, ii, ZZ = [], [], [], []
    for k in range(n_units):
        dtau = af(z) * dt
        if kind == "wiener":
            inc = rng.normal(MU * dtau, SIGMA * np.sqrt(dtau))
        else:
            inc = rng.gamma(ALPHA * dtau, 1.0 / BETA)
        xs.append(np.concatenate([[0.0], times]))
        ys.append(np.concatenate([[0.0], np.cumsum(inc)]))
        ii.append(np.full(times.size + 1, k))
        ZZ.append(np.concatenate([[Z_LEVELS[0]], z]))
    return tuple(np.concatenate(v) for v in (xs, ys, ii, ZZ))


def simulated_first_passage(kind, n=3000, seed=7, h=0.2, t_end=260.0):
    """Brute-force first passage along PROFILE on a fine grid."""
    rng = np.random.default_rng(seed)
    grid = np.arange(h, t_end + 1e-9, h)
    dtau = af(stress_at(grid)) * h
    if kind == "wiener":
        inc = rng.normal(MU * dtau, SIGMA * np.sqrt(dtau), (n, grid.size))
    else:
        inc = rng.gamma(ALPHA * dtau, 1.0 / BETA, (n, grid.size))
    hit = np.cumsum(inc, axis=1) >= THRESHOLD
    return np.where(hit.any(axis=1), grid[hit.argmax(axis=1)], np.inf)


FITTERS = {"wiener": WienerProcess, "gamma": GammaProcess}


@pytest.fixture(scope="module", params=["wiener", "gamma"])
def fitted(request):
    kind = request.param
    x, y, i, Z = step_stress_data(kind)
    model = FITTERS[kind].fit(
        x, y, i, threshold=THRESHOLD, Z=Z, stress_ref=[Z_USE]
    )
    return kind, model


# -- recovering the truth ---------------------------------------------------


def test_step_stress_fit_recovers_the_truth(fitted):
    kind, model = fitted
    assert model.is_accelerated
    assert model.stress_ref == pytest.approx([Z_USE])
    assert model.gamma[0] == pytest.approx(G_TRUE, abs=400)
    if kind == "wiener":
        assert model.mu == pytest.approx(MU, rel=0.05)
        assert model.sigma == pytest.approx(SIGMA, rel=0.05)
    else:
        assert model.alpha == pytest.approx(ALPHA, rel=0.1)
        assert model.beta == pytest.approx(BETA, rel=0.1)
    assert model.acceleration_factor([Z_USE]) == pytest.approx(1.0)


@pytest.mark.parametrize("kind", ["wiener", "gamma"])
def test_constant_stress_levels_between_units(kind):
    # classic ADT: each unit at one constant stress, three levels
    rng = np.random.default_rng(3)
    times = np.arange(0.0, 60.0 + 1e-9, 2.0)
    xs, ys, ii, ZZ = [], [], [], []
    for k, z in enumerate(np.repeat(Z_LEVELS, 15)):
        dtau = af(z) * np.diff(times)
        if kind == "wiener":
            inc = rng.normal(MU * dtau, SIGMA * np.sqrt(dtau))
        else:
            inc = rng.gamma(ALPHA * dtau, 1.0 / BETA)
        xs.append(times)
        ys.append(np.concatenate([[0.0], np.cumsum(inc)]))
        ii.append(np.full(times.size, k))
        ZZ.append(np.full(times.size, z))
    x, y, i, Z = (np.concatenate(v) for v in (xs, ys, ii, ZZ))
    model = FITTERS[kind].fit(x, y, i, threshold=THRESHOLD, Z=Z)
    assert model.gamma[0] == pytest.approx(G_TRUE, abs=600)
    # the default reference is the mean stress over the increments
    assert model.stress_ref == pytest.approx([np.mean(Z_LEVELS)])


def test_two_covariates():
    # temperature and voltage, each stepping at its own time
    rng = np.random.default_rng(5)
    g_true = np.array([-4000.0, 1.5])
    times = np.arange(0.0, 90.0 + 1e-9, 3.0)
    xs, ys, ii, ZZ = [], [], [], []
    for k in range(30):
        temp = np.where(times < 30 + k % 3 * 10, 1 / 333.0, 1 / 363.0)
        volt = np.where(times < 50 + k % 2 * 15, 0.0, 0.8)
        z = np.column_stack([temp, volt])
        rate = np.exp((z[1:] - [1 / 333.0, 0.0]) @ g_true)
        dtau = rate * np.diff(times)
        inc = rng.normal(MU * dtau, SIGMA * np.sqrt(dtau))
        xs.append(times)
        ys.append(np.concatenate([[0.0], np.cumsum(inc)]))
        ii.append(np.full(times.size, k))
        ZZ.append(z)
    x, y, i = (np.concatenate(v) for v in (xs, ys, ii))
    model = WienerProcess.fit(
        x, y, i, threshold=5.0, Z=np.vstack(ZZ), stress_ref=[1 / 333.0, 0.0]
    )
    assert model.gamma == pytest.approx(g_true, rel=0.15)
    assert model.mu == pytest.approx(MU, rel=0.1)


# -- life under a stress profile ----------------------------------------------


def test_life_under_profile_matches_brute_force_simulation(fitted):
    kind, model = fitted
    sim = simulated_first_passage(kind)
    for t in (130.0, 145.0, 160.0):
        assert float(model.ff(t, Z=PROFILE)) == pytest.approx(
            np.mean(sim <= t), abs=0.04
        )
    assert model.mean(Z=PROFILE) == pytest.approx(np.mean(sim), rel=0.02)


def test_constant_stress_is_a_rescaled_clock(fitted):
    _, model = fitted
    hot = [Z_LEVELS[2]]
    a = model.acceleration_factor(hot)
    t = np.array([5.0, 20.0, 40.0])
    assert np.allclose(model.ff(t, Z=hot), model.ff(a * t, Z=[Z_USE]))
    assert model.mean(Z=hot) == pytest.approx(model.mean(Z=[Z_USE]) / a)
    assert np.allclose(
        model.qf([0.1, 0.5, 0.9], Z=hot),
        np.asarray(model.qf([0.1, 0.5, 0.9], Z=[Z_USE])) / a,
    )
    # a constant StepSchedule is the same thing
    const = StepSchedule.constant(hot)
    assert np.allclose(model.ff(t, Z=const), model.ff(t, Z=hot))


def test_profile_distribution_is_internally_consistent(fitted):
    _, model = fitted
    u = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    q = model.qf(u, Z=PROFILE)
    assert np.all(np.diff(q) > 0)
    assert np.allclose(model.ff(q, Z=PROFILE), u, atol=1e-8)
    assert np.allclose(model.sf(q, Z=PROFILE) + model.ff(q, Z=PROFILE), 1.0)
    # the density is the derivative of ff, away from the step times
    t = np.array([60.0, 130.0, 150.0, 170.0])
    h = 1e-4
    slope = (model.ff(t + h, Z=PROFILE) - model.ff(t - h, Z=PROFILE)) / (2 * h)
    assert np.allclose(model.df(t, Z=PROFILE), slope, rtol=1e-3, atol=1e-8)
    assert np.allclose(
        model.hf(t, Z=PROFILE),
        model.df(t, Z=PROFILE) / model.sf(t, Z=PROFILE),
    )
    mean = quad(lambda s: float(model.sf(s, Z=PROFILE)), 0, np.inf)[0]
    assert model.mean(Z=PROFILE) == pytest.approx(mean, rel=1e-6)
    # sampling along the profile
    draws = model.random(20_000, random_state=3, Z=PROFILE)
    assert np.median(draws) == pytest.approx(
        float(model.qf(0.5, Z=PROFILE)), rel=0.02
    )


def test_predict_rul_with_stress(fitted):
    _, model = fitted
    hot = [Z_LEVELS[2]]
    a = model.acceleration_factor(hot)
    at_ref = model.predict_rul(6.0, Z=[Z_USE])
    at_hot = model.predict_rul(6.0, Z=hot)
    assert at_hot.rul == pytest.approx(at_ref.rul / a)
    assert np.allclose(at_hot.rul_interval, np.array(at_ref.rul_interval) / a)
    # a profile starting now: 20 time units at 348K, then 373K -- slower
    # than going straight to 373K
    profile = StepSchedule.from_changepoints([0, 20], [[Z_LEVELS[1]], hot])
    stepped = model.predict_rul(6.0, Z=profile)
    assert stepped.rul > at_hot.rul
    # already past the threshold
    assert model.predict_rul(THRESHOLD + 1, Z=hot).prob_already_failed == 1


def test_stressed_models_round_trip(fitted):
    _, model = fitted
    d = json.loads(json.dumps(model.to_dict()))
    assert d["gamma"] == pytest.approx(model.gamma.tolist())
    assert d["stress_ref"] == pytest.approx(model.stress_ref.tolist())
    restored = surpyval.from_dict(d)
    assert type(restored) is type(model)
    t = np.array([100.0, 150.0])
    assert np.array_equal(restored.ff(t, Z=PROFILE), model.ff(t, Z=PROFILE))
    assert "Stress coefficients" in repr(model)
    assert "Mean life (ref.)" in repr(model)


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize("fitter", [WienerProcess, GammaProcess])
def test_stress_fit_validation(fitter):
    x, y, i, Z = step_stress_data(
        "wiener" if fitter is WienerProcess else "gamma", n_units=5
    )
    with pytest.raises(ValueError, match="one row per measurement"):
        fitter.fit(x, y, i, threshold=THRESHOLD, Z=Z[:-1])
    with pytest.raises(ValueError, match="finite"):
        fitter.fit(x, y, i, threshold=THRESHOLD, Z=np.where(Z > 0, np.nan, Z))
    with pytest.raises(ValueError, match="cannot be estimated"):
        fitter.fit(x, y, i, threshold=THRESHOLD, Z=np.full_like(Z, Z_USE))
    with pytest.raises(ValueError, match="only meaningful with Z"):
        fitter.fit(x, y, i, threshold=THRESHOLD, stress_ref=[Z_USE])
    with pytest.raises(ValueError, match="single stress row"):
        fitter.fit(x, y, i, threshold=THRESHOLD, Z=Z, stress_ref=[1.0, 2.0])


def test_gamma_stress_fit_still_requires_monotone_paths():
    x, y, i, Z = step_stress_data("wiener", n_units=5)
    with pytest.raises(ValueError, match="monotone"):
        GammaProcess.fit(x, y, i, threshold=THRESHOLD, Z=Z)


def test_stress_argument_errors(fitted):
    _, model = fitted
    for call in (
        lambda: model.ff(10.0),
        lambda: model.sf(10.0),
        lambda: model.df(10.0),
        lambda: model.qf(0.5),
        lambda: model.mean(),
        lambda: model.random(5),
        lambda: model.predict_rul(3.0),
    ):
        with pytest.raises(ValueError, match="depends on stress"):
            call()
    with pytest.raises(ValueError, match="single stress row"):
        model.ff(10.0, Z=[Z_USE, Z_USE])
    with pytest.raises(ValueError, match="covariate"):
        model.ff(
            10.0,
            Z=StepSchedule.from_changepoints([0.0], [[Z_USE, 0.0]]),
        )


def test_stress_free_models_refuse_Z():
    x, y, i, Z = step_stress_data("gamma", n_units=5)
    plain = GammaProcess.fit(x, y, i, threshold=THRESHOLD)
    assert not plain.is_accelerated
    assert plain.gamma is None and plain.stress_ref is None
    with pytest.raises(ValueError, match="without stress"):
        plain.ff(10.0, Z=[Z_USE])
    with pytest.raises(ValueError, match="without stress"):
        plain.acceleration_factor([Z_USE])
    assert "gamma" not in plain.to_dict()
    with pytest.raises(ValueError, match="together"):
        WienerProcessModel(1.0, 1.0, 5.0, gamma=[1.0])
    with pytest.raises(ValueError, match="together"):
        GammaProcessModel(1.0, 1.0, 5.0, stress_ref=[1.0])
