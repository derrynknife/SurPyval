"""Stage 3 of #155, route 2: step-stress general-path degradation."""

import json

import matplotlib
import numpy as np
import pytest
from scipy.integrate import quad

from surpyval import AFT, StepSchedule, Weibull
from surpyval.degradation import DegradationAnalysis, DegradationModel
from surpyval.degradation.population import (
    _reml_pieces,
    _reml_pieces_woodbury,
    _unit_summaries,
    reml_estimate,
    reml_estimate_woodbury,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Arrhenius covariate z = 1/T; a three-step profile 50C -> 75C -> 100C
Z_LEVELS = 1.0 / np.array([323.0, 348.0, 373.0])
Z_USE = Z_LEVELS[0]
G_TRUE = -5000.0
PROFILE = StepSchedule.from_changepoints([0.0, 100.0, 200.0], Z_LEVELS)
MU = np.array([1.0, 0.02])  # linear path a + b * tau
SD = np.array([0.2, 0.003])
NOISE = 0.3
THRESHOLD = 15.0


def af(z):
    return np.exp(G_TRUE * (np.asarray(z) - Z_USE))


def stress_at(t):
    return np.select([t <= 100, t <= 200], Z_LEVELS[:2], Z_LEVELS[2])


def simulate(
    n_units=30,
    seed=0,
    stepped=True,
    dt=10.0,
    noise=NOISE,
    path="linear",
    return_params=False,
):
    rng = np.random.default_rng(seed)
    t = np.arange(dt, 300.0 + 1e-9, dt)
    if path == "linear":
        params = rng.normal(MU, SD, size=(n_units, 2))
    else:
        params = rng.normal([1.0, 0.002], [0.1, 0.0003], size=(n_units, 2))
    xs, ys, ii, ZZ = [], [], [], []
    for k, (a, b) in enumerate(params):
        z = stress_at(t) if stepped else np.full(t.size, Z_LEVELS[k % 3])
        tau = np.cumsum(np.diff(np.concatenate([[0.0], t])) * af(z))
        y = a + b * tau if path == "linear" else a * np.exp(b * tau)
        xs.append(t)
        ys.append(y + rng.normal(0.0, noise, t.size))
        ii.append(np.full(t.size, k))
        ZZ.append(z)
    out = tuple(np.concatenate(v) for v in (xs, ys, ii, ZZ))
    return (out, params) if return_params else out


def fit(x, y, i, Z, **kwargs):
    kwargs.setdefault("threshold", THRESHOLD)
    kwargs.setdefault("stress_ref", [Z_USE])
    return DegradationAnalysis.fit(
        x, y, i, Z=Z, acceleration="clock", **kwargs
    )


@pytest.fixture(scope="module")
def data():
    return simulate()


@pytest.fixture(scope="module", params=["moments", "reml"])
def model(request, data):
    x, y, i, Z = data
    return fit(x, y, i, Z, population_method=request.param)


# -- recovering the truth ---------------------------------------------------


def test_step_stress_fit_recovers_the_truth(model):
    assert model.is_accelerated
    assert model.acceleration == "clock"
    assert model.gamma[0] == pytest.approx(G_TRUE, abs=150)
    assert model.path_param_mean == pytest.approx(MU, rel=0.1)
    assert np.sqrt(np.diag(model.path_param_cov)) == pytest.approx(SD, rel=0.3)
    assert np.sqrt(model.measurement_var) == pytest.approx(NOISE, rel=0.05)
    assert model.acceleration_factor([Z_USE]) == pytest.approx(1.0)


def test_noise_free_step_stress_is_recovered_exactly():
    (x, y, i, Z), ab = simulate(n_units=6, noise=0.0, return_params=True)
    m = fit(x, y, i, Z)
    assert m.gamma[0] == pytest.approx(G_TRUE, rel=1e-6)
    # the pseudo failure times are the true reference-stress lifetimes
    assert m.pseudo_failure_times == pytest.approx(
        (THRESHOLD - ab[:, 0]) / ab[:, 1], rel=1e-6
    )
    # and each unit's fitted path runs through its data in calendar time
    for unit in range(6):
        mask = i == unit
        assert m.path(x[mask], unit) == pytest.approx(y[mask], abs=1e-6)


def test_exponential_path():
    x, y, i, Z = simulate(n_units=20, noise=0.05, path="exponential")
    m = fit(x, y, i, Z, threshold=5.0, path="exponential")
    assert m.gamma[0] == pytest.approx(G_TRUE, abs=150)
    assert m.path_param_mean == pytest.approx([1.0, 0.002], rel=0.1)


def test_constant_stresses_need_the_mixed_model():
    x, y, i, Z = simulate(n_units=30, stepped=False)
    with pytest.raises(ValueError, match="population_method='reml'"):
        fit(x, y, i, Z)
    m = fit(x, y, i, Z, population_method="reml")
    assert m.gamma[0] == pytest.approx(G_TRUE, abs=500)
    assert m.path_param_mean == pytest.approx(MU, rel=0.1)


def test_two_covariates():
    # temperature and voltage stepping at different times
    rng = np.random.default_rng(4)
    g_true = np.array([-4000.0, 1.5])
    ref = np.array([1 / 333.0, 0.0])
    t = np.arange(5.0, 120.0 + 1e-9, 5.0)
    xs, ys, ii, ZZ = [], [], [], []
    for k in range(24):
        temp = np.where(t <= 30 + 10 * (k % 3), 1 / 333.0, 1 / 363.0)
        volt = np.where(t <= 60 + 15 * (k % 2), 0.0, 0.8)
        z = np.column_stack([temp, volt])
        tau = np.cumsum(5.0 * np.exp((z - ref) @ g_true))
        a, b = rng.normal(MU, SD)
        xs.append(t)
        ys.append(a + b * tau + rng.normal(0.0, 0.1, t.size))
        ii.append(np.full(t.size, k))
        ZZ.append(z)
    x, y, i = (np.concatenate(v) for v in (xs, ys, ii))
    m = fit(x, y, i, np.vstack(ZZ), stress_ref=ref)
    assert m.gamma == pytest.approx(g_true, rel=0.1)


def test_default_reference_stress_is_the_mean_interval_stress(data):
    x, y, i, Z = data
    m = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, Z=Z, acceleration="clock"
    )
    assert m.stress_ref == pytest.approx([Z.mean()])
    assert m.acceleration_factor(m.stress_ref) == pytest.approx(1.0)


# -- life under a stress ----------------------------------------------------


def test_life_under_the_profile_matches_the_population(model):
    # the truth: each unit crosses at its reference-stress time
    # (D - a) / b, reached in calendar time along the profile's clock
    rng = np.random.default_rng(5)
    ab = rng.normal(MU, SD, size=(100_000, 2))
    tau_star = (THRESHOLD - ab[:, 0]) / ab[:, 1]
    knots_t = np.array([0.0, 100.0, 200.0])
    knots_tau = np.concatenate([[0.0], np.cumsum(100.0 * af(Z_LEVELS[:2]))])
    life = np.where(
        tau_star <= knots_tau[-1],
        np.interp(tau_star, knots_tau, knots_t),
        200.0 + (tau_star - knots_tau[-1]) / af(Z_LEVELS[2]),
    )
    assert model.mean(Z=PROFILE) == pytest.approx(life.mean(), rel=0.03)
    assert float(model.qf(0.5, Z=PROFILE)[0]) == pytest.approx(
        np.median(life), rel=0.03
    )


def test_constant_stress_is_a_rescaled_clock(model):
    hot = [Z_LEVELS[2]]
    a = model.acceleration_factor(hot)
    t = np.array([20.0, 60.0, 100.0])
    assert np.allclose(model.sf(t, Z=hot), model.sf(a * t, Z=[Z_USE]))
    assert np.allclose(model.ff(t, Z=hot), model.life_model.ff(a * t))
    assert model.mean(Z=hot) == pytest.approx(model.life_model.mean() / a)
    assert np.allclose(
        model.qf([0.1, 0.9], Z=hot), model.life_model.qf([0.1, 0.9]) / a
    )
    const = StepSchedule.constant(hot)
    assert np.allclose(model.sf(t, Z=const), model.sf(t, Z=hot))
    assert model.mean(Z=const) == pytest.approx(model.mean(Z=hot), rel=1e-6)


def test_life_under_a_profile_is_consistent(model):
    u = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
    q = model.qf(u, Z=PROFILE)
    assert np.all(np.diff(q) > 0)
    assert np.allclose(model.ff(q, Z=PROFILE), u, atol=1e-8)
    assert np.allclose(model.sf(q, Z=PROFILE), 1 - u, atol=1e-8)
    t = np.array([150.0, 220.0, 240.0, 260.0])
    h = 1e-4
    slope = (model.ff(t + h, Z=PROFILE) - model.ff(t - h, Z=PROFILE)) / (2 * h)
    assert np.allclose(model.df(t, Z=PROFILE), slope, rtol=1e-4, atol=1e-10)
    assert np.allclose(
        model.hf(t, Z=PROFILE),
        model.df(t, Z=PROFILE) / model.sf(t, Z=PROFILE),
    )
    assert np.allclose(model.Hf(t, Z=PROFILE), -np.log(model.sf(t, Z=PROFILE)))
    mean = quad(
        lambda s: float(model.sf(s, Z=PROFILE)[0]),
        0,
        np.inf,
        points=None,
        limit=200,
    )[0]
    assert model.mean(Z=PROFILE) == pytest.approx(mean, rel=1e-4)
    draws = model.random(20_000, Z=PROFILE, random_state=3)
    assert np.median(draws) == pytest.approx(
        float(model.qf(0.5, Z=PROFILE)[0]), rel=0.01
    )


# -- the fitted model -------------------------------------------------------


def test_round_trip_and_repr(model):
    d = json.loads(json.dumps(model.to_dict()))
    assert d["acceleration"] == "clock"
    assert d["gamma"] == pytest.approx(model.gamma.tolist())
    restored = DegradationModel.from_dict(d)
    t = np.array([100.0, 200.0, 250.0])
    assert np.array_equal(restored.sf(t, Z=PROFILE), model.sf(t, Z=PROFILE))
    assert repr(restored) == repr(model)
    assert "Stress coefficients" in repr(model)
    assert "Weibull (reference stress)" in repr(model)
    for unit in (0, 5):
        assert np.array_equal(restored.path(t, unit), model.path(t, unit))


def test_other_models_serialise_as_before():
    x, y, i, Z = simulate(n_units=6, stepped=False)
    plain = DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD)
    assert "acceleration" not in plain.to_dict()
    assert plain.gamma is None and plain.acceleration is None


def test_path_and_plot_use_the_units_clock(model, data):
    x, y, i, Z = data
    mask = i == 3
    fitted = model.path(x[mask], 3)
    # the fitted path follows the data (noise 0.3) in calendar time
    assert np.sqrt(np.mean((fitted - y[mask]) ** 2)) < 2 * NOISE
    # beyond the last measurement the last stress is held
    later = model.path(np.array([300.0, 310.0]), 3)
    b = model.path_params[3, 1]
    assert later[1] - later[0] == pytest.approx(
        b * 10.0 * model.acceleration_factor([Z_LEVELS[2]])
    )
    fig, ax = plt.subplots()
    model.plot(ax=ax)
    assert len(ax.lines) == len(model.units) + 1
    plt.close(fig)


# -- validation -------------------------------------------------------------


def test_clock_argument_validation(data):
    x, y, i, Z = data
    with pytest.raises(ValueError, match="must be None or 'clock'"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, Z=Z, acceleration="time"
        )
    with pytest.raises(ValueError, match="only used with it"):
        DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD, stress_ref=[1])
    with pytest.raises(ValueError, match="Z must be given"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, acceleration="clock"
        )
    with pytest.raises(ValueError, match="cannot be combined"):
        fit(x, y, i, Z, links={"b": "log"})
    with pytest.raises(ValueError, match="path='best'"):
        fit(x, y, i, Z, path="best")
    with pytest.raises(ValueError, match="plain distribution"):
        fit(x, y, i, Z, distribution=AFT(Weibull))
    with pytest.raises(ValueError, match="non-negative"):
        fit(x - 20.0, y, i, Z)
    with pytest.raises(ValueError, match="one row per measurement"):
        fit(x, y, i, Z[:-1])
    with pytest.raises(ValueError, match="finite"):
        fit(x, y, i, np.where(x > 50, np.nan, Z))
    with pytest.raises(ValueError, match="cannot be estimated"):
        fit(x, y, i, np.full_like(Z, Z_USE))
    with pytest.raises(ValueError, match="single stress row"):
        fit(x, y, i, Z, stress_ref=[1.0, 2.0])


def test_varying_stress_without_the_clock_points_to_it(data):
    x, y, i, Z = data
    with pytest.raises(ValueError, match="acceleration='clock'"):
        DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD, Z=Z)


def test_prediction_argument_validation(model):
    for call in (
        lambda: model.sf(10.0),
        lambda: model.qf(0.5),
        lambda: model.mean(),
        lambda: model.random(5),
    ):
        with pytest.raises(ValueError, match="depends on stress"):
            call()
    with pytest.raises(ValueError, match="single stress row"):
        model.sf(10.0, Z=[Z_USE, Z_USE])
    with pytest.raises(ValueError, match="covariate"):
        model.sf(10.0, Z=StepSchedule.constant([Z_USE, 0.0]))
    for call in (
        lambda: model.predict_rul([10.0], [1.2]),
        lambda: model.predict_failure_time([10.0, 20.0], [1.2, 1.4]),
        lambda: model.predict_remaining_life([10.0, 20.0], [1.2, 1.4]),
    ):
        with pytest.raises(ValueError, match="stress history"):
            call()
    with pytest.raises(ValueError, match="one row of 1 covariate"):
        model.predict_rul([10.0, 20.0, 30.0], [1.2, 1.4, 1.6], Z=[1, 2])
    with pytest.raises(ValueError, match="non-negative"):
        model.predict_failure_time([-10.0, 20.0], [1.2, 1.4], Z=[Z_USE])
    with pytest.raises(ValueError, match="depends on stress"):
        model.induced_life()
    for call in (
        lambda: model.cb([100.0], Z=PROFILE),
        lambda: model.life_parameter_covariance(),
    ):
        with pytest.raises(NotImplementedError, match="bootstrap"):
            call()


def test_models_without_a_clock_refuse_its_arguments():
    x, y, i, _ = simulate(n_units=6, stepped=False)
    plain = DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD)
    with pytest.raises(ValueError, match="Z_future"):
        plain.predict_rul([10.0], [1.2], Z_future=[Z_USE])
    with pytest.raises(ValueError, match="Z_future"):
        plain.predict_failure_time([10.0, 20.0], [1.2, 1.4], Z_future=[1])
    with pytest.raises(ValueError, match="takes Z only"):
        plain.predict_failure_time([10.0, 20.0], [1.2, 1.4], Z=[Z_USE])


# -- predictions for a new unit (part B) -----------------------------------


@pytest.fixture(scope="module")
def exact_model():
    """Noise-free training data: the clock is recovered exactly."""
    x, y, i, Z = simulate(n_units=8, noise=0.0)
    return fit(x, y, i, Z)


def _new_unit(until=150.0, a=0.9, b=0.022):
    """A noise-free unit on the test profile up to ``until``, and its true
    failure time if the profile then carries on (100 C from 200 h)."""
    t = np.arange(10.0, until + 1e-9, 10.0)
    z = stress_at(t)
    y = a + b * np.cumsum(10.0 * af(z))
    tau_star = (THRESHOLD - a) / b
    knots_tau = np.concatenate([[0.0], np.cumsum(100.0 * af(Z_LEVELS[:2]))])
    if tau_star <= knots_tau[-1]:
        truth = np.interp(tau_star, knots_tau, [0.0, 100.0, 200.0])
    else:
        truth = 200.0 + (tau_star - knots_tau[-1]) / af(Z_LEVELS[2])
    return t, y, z, truth


def test_failure_time_along_the_history_and_a_planned_future(exact_model):
    t, y, z, truth = _new_unit()
    # from 150 h: 50 more hours at 75 C, then 100 C -- the test profile
    plan = StepSchedule.from_changepoints(
        [0, 50], [[Z_LEVELS[1]], [Z_LEVELS[2]]]
    )
    ft = exact_model.predict_failure_time(t, y, Z=z, Z_future=plan)
    assert ft == pytest.approx(truth, rel=1e-5)
    rl = exact_model.predict_remaining_life(t, y, Z=z, Z_future=plan)
    assert rl == pytest.approx(truth - 150.0, rel=1e-5)
    # holding the last stress (75 C) instead ages the unit more slowly
    held = exact_model.predict_failure_time(t, y, Z=z)
    assert held > ft


def test_constant_stress_history_is_a_rescaled_clock(model):
    # any fitted clock gives the exact answer for a unit held at one
    # stress: its rate absorbs the acceleration factor
    hot = [Z_LEVELS[2]]
    t = np.arange(5.0, 31.0, 5.0)
    y = 0.9 + 0.022 * af(Z_LEVELS[2]) * t
    truth = (THRESHOLD - 0.9) / 0.022 / af(Z_LEVELS[2])
    assert model.predict_failure_time(t, y, Z=hot) == pytest.approx(truth)
    rows = np.full(t.size, Z_LEVELS[2])
    assert model.predict_failure_time(t, y, Z=rows) == pytest.approx(truth)


def test_predict_rul_on_the_clock(model):
    t, y, z, truth = _new_unit()
    plan = StepSchedule.from_changepoints(
        [0, 50], [[Z_LEVELS[1]], [Z_LEVELS[2]]]
    )
    noisy = y + np.random.default_rng(9).normal(0.0, NOISE, y.size)
    pred = model.predict_rul(t, noisy, Z=z, Z_future=plan, random_state=1)
    lo, hi = pred.failure_time_interval
    assert lo < truth < hi
    assert pred.rul == pytest.approx(pred.failure_time - 150.0)
    assert pred.prob_failed == 0.0
    # the posterior is on the reference-stress path parameters
    assert pred.posterior_mean == pytest.approx([0.9, 0.022], rel=0.15)
    # low-noise training data and a long, noise-free trajectory: the
    # posterior settles on the truth
    x, y_low, i, Z = simulate(n_units=8, noise=0.01)
    low_noise = fit(x, y_low, i, Z)
    t2, y2, z2, truth2 = _new_unit(until=190.0)
    rest = StepSchedule.from_changepoints(
        [0, 10], [[Z_LEVELS[1]], [Z_LEVELS[2]]]
    )
    settled = low_noise.predict_rul(
        t2, y2, Z=z2, Z_future=rest, random_state=0
    )
    assert settled.failure_time == pytest.approx(truth2, rel=2e-3)
    # a unit already past the threshold
    done = model.predict_rul(t, y + 20.0, Z=z, random_state=0)
    assert done.prob_failed > 0.99


def test_induced_life_under_a_stress(model):
    ref = model.induced_life(Z=[Z_USE], random_state=0)
    hot = model.induced_life(Z=[Z_LEVELS[2]], random_state=0)
    assert np.allclose(
        hot.samples, ref.samples / model.acceleration_factor([Z_LEVELS[2]])
    )
    assert hot.stress == pytest.approx([Z_LEVELS[2]])
    induced = model.induced_life(Z=PROFILE, random_state=0)
    assert induced.stress is None
    assert induced.median() == pytest.approx(
        float(model.qf(0.5, Z=PROFILE)[0]), rel=0.03
    )


def test_bootstrap_bounds_under_a_profile(model):
    t = np.array([200.0, 240.0])
    band = model.cb(t, Z=PROFILE, method="bootstrap", n_boot=20, seed=1)
    assert band.shape == (2, 2)
    assert np.all(band[:, 0] <= band[:, 1])
    sf = model.sf(t, Z=PROFILE)
    assert np.all((band[:, 0] <= sf + 0.05) & (sf - 0.05 <= band[:, 1]))
    restored = DegradationModel.from_dict(model.to_dict())
    with pytest.raises(RuntimeError, match="restored from a dict"):
        restored.cb(t, Z=PROFILE, method="bootstrap", n_boot=5, seed=1)


def test_acceleration_factor_needs_a_clock_model():
    x, y, i, _ = simulate(n_units=6, stepped=False)
    plain = DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD)
    with pytest.raises(ValueError, match="acceleration='clock'"):
        plain.acceleration_factor([Z_USE])


# -- the Woodbury REML ------------------------------------------------------


def _lmm(seed=0, units=8, n=10):
    rng = np.random.default_rng(seed)
    ys, xs = [], []
    for _ in range(units):
        t = np.sort(rng.uniform(0, 10, n))
        X = np.column_stack([np.ones(n), t])
        theta = rng.normal([2.0, 0.5], [0.5, 0.1])
        xs.append(X)
        ys.append(X @ theta + rng.normal(0, 0.3, n))
    return ys, xs


def test_woodbury_objective_is_the_reml_objective():
    ys, xs = _lmm()
    rng = np.random.default_rng(1)
    for _ in range(5):
        z = rng.normal(size=4) * 0.5
        full = _reml_pieces(z, ys, xs, 2, xs)
        fast = _reml_pieces_woodbury(z, _unit_summaries(ys, xs, xs), 2)
        for a, b in zip(full, fast):
            assert np.allclose(a, b, rtol=1e-9)


def test_woodbury_estimate_matches_reml_estimate():
    ys, xs = _lmm(seed=2)
    cov0, s20 = np.diag([0.2, 0.01]), 0.1
    mean, cov, s2, ok = reml_estimate(ys, xs, cov0, s20)
    mean_w, cov_w, s2_w, ok_w, _ = reml_estimate_woodbury(
        ys, xs, cov0, s20, a_mat_list=xs
    )
    assert ok and ok_w
    assert mean_w == pytest.approx(mean, rel=1e-5)
    assert cov_w == pytest.approx(cov, rel=1e-3, abs=1e-8)
    assert s2_w == pytest.approx(s2, rel=1e-5)


def test_reml_on_paths_with_very_different_time_scales():
    # exponential paths whose time scales differ ~8x between units -- what a
    # clock makes of units run at different constant stresses. The REML step
    # used to run for many minutes on this (an absolute function tolerance
    # the objective's round-off could not meet); it is now well under a
    # second.
    rng = np.random.default_rng(0)
    t = np.arange(10.0, 300.0 + 1e-9, 10.0)
    xs, ys, ii = [], [], []
    for k in range(20):
        tau = t * [1.0, 3.04, 7.93][k % 3]
        a, b = rng.normal([1.0, 0.002], [0.1, 0.0003])
        xs.append(tau)
        ys.append(a * np.exp(b * tau) + rng.normal(0, 0.05, t.size))
        ii.append(np.full(t.size, k))
    x, y, i = (np.concatenate(v) for v in (xs, ys, ii))
    m = DegradationAnalysis.fit(
        x, y, i, threshold=5.0, path="exponential", population_method="reml"
    )
    assert m.path_param_mean == pytest.approx([1.0, 0.002], rel=0.05)
    assert np.sqrt(m.measurement_var) == pytest.approx(0.05, rel=0.05)
