"""Analytic two-stage bounds of the accelerated degradation models.

An accelerated (covariate) model fits a regression life model to the pseudo
failure times; a step-stress (``acceleration="clock"``) model fits a plain
life model to pseudo failure times on a clock that is itself estimated. The
analytic bounds fold the first-stage uncertainty -- each unit's path fit and,
for the clock, the shared estimate of the stress coefficients -- into the
life-model covariance by the delta method. These tests pin the algebra
against direct computations, the limits it must reach, and its agreement with
the bootstrap and with a simulation.
"""

import warnings

import numpy as np
import pytest

from surpyval.degradation import DegradationAnalysis, DegradationModel
from surpyval.degradation._bounds import (
    _plain_information_inverse,
    clock_first_stage,
)
from surpyval.tests.degradation import test_step_stress as ss
from surpyval.tests.degradation.test_degradation_analysis import adt_data


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call()


def _overlap(a, b):
    """The fraction of band ``a`` that band ``b`` covers, at each row."""
    lo = np.maximum(a[:, 0], b[:, 0])
    hi = np.minimum(a[:, 1], b[:, 1])
    return np.maximum(hi - lo, 0.0) / (a[:, 1] - a[:, 0])


# -- accelerated (covariate) life fits ---------------------------------------


@pytest.fixture(scope="module")
def adt_model():
    x, y, i, Z = adt_data(seed=1)
    return _quiet(
        lambda: DegradationAnalysis.fit(x, y, i, threshold=100.0, Z=Z)
    )


def _exact_adt():
    """Linear paths through every reading exactly: no first-stage noise."""
    rng = np.random.default_rng(0)
    t = np.array([5.0, 10.0, 20.0])
    xs, ys, ids, Zs = [], [], [], []
    for unit in range(24):
        z = float(unit % 3)
        b = 0.5 * np.exp(0.6 * z + 0.2 * rng.standard_normal())
        xs.append(t)
        ys.append(10.0 + b * t)
        ids.append(np.full(t.size, unit))
        Zs.append(np.full(t.size, z))
    x, y, i, Z = (np.concatenate(v) for v in (xs, ys, ids, Zs))
    return _quiet(
        lambda: DegradationAnalysis.fit(x, y, i, threshold=100.0, Z=Z)
    )


def test_adt_correction_adds_a_positive_semidefinite_term(adt_model):
    cov = adt_model.life_parameter_covariance()
    plain = adt_model.life_model.covariance()
    assert cov.shape == plain.shape == (3, 3)
    extra = cov - plain
    scale = np.sqrt(np.outer(np.diag(plain), np.diag(plain)))
    assert np.linalg.eigvalsh(extra / scale).min() > -1e-10
    assert np.all(np.diag(cov) > np.diag(plain))


def test_adt_exact_paths_give_the_regression_bound():
    model = _exact_adt()
    x = np.array([100.0, 150.0])
    np.testing.assert_allclose(
        model.life_parameter_covariance(),
        model.life_model.covariance(),
        rtol=1e-8,
        atol=0.0,
    )
    for on in ("sf", "ff", "Hf"):
        np.testing.assert_allclose(
            model.cb(x, Z=[1.0], on=on),
            model.life_model.cb(x, [1.0], on=on),
            rtol=1e-6,
        )


@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
def test_adt_band_brackets_and_contains_the_regression_band(adt_model, on):
    x = np.array([110.0, 125.0, 140.0])
    est = getattr(adt_model, on)(x, Z=[0.5])
    band = adt_model.cb(x, Z=[0.5], on=on)
    plain = adt_model.life_model.cb(x, [0.5], on=on)
    assert band.shape == (3, 2)
    assert np.all(band[:, 0] <= est) and np.all(est <= band[:, 1])
    # the first-stage term only widens the band
    assert np.all(band[:, 0] <= plain[:, 0] + 1e-12)
    assert np.all(band[:, 1] >= plain[:, 1] - 1e-12)
    lower = adt_model.cb(x, Z=[0.5], on=on, bound="lower", alpha_ci=0.025)
    upper = adt_model.cb(x, Z=[0.5], on=on, bound="upper", alpha_ci=0.025)
    np.testing.assert_allclose(band, np.stack([lower, upper], axis=-1))


def test_adt_pairs_times_and_stress_rows(adt_model):
    x = np.array([110.0, 90.0, 70.0])
    rows = [[0.5], [1.0], [1.5]]
    together = adt_model.cb(x, Z=rows)
    for k in range(3):
        np.testing.assert_allclose(
            together[k], adt_model.cb(x[k], Z=rows[k]), rtol=1e-10
        )


def test_adt_analytic_needs_the_stress(adt_model):
    with pytest.raises(ValueError, match="covariate vector Z"):
        adt_model.cb(100.0)


def test_adt_analytic_survives_a_round_trip(adt_model):
    restored = DegradationModel.from_dict(adt_model.to_dict())
    x = np.array([110.0, 125.0])
    np.testing.assert_allclose(
        restored.cb(x, Z=[0.5]), adt_model.cb(x, Z=[0.5]), rtol=1e-8
    )


def test_adt_analytic_and_bootstrap_agree(adt_model):
    x = np.array([110.0, 125.0])
    an = adt_model.cb(x, Z=[0.5])
    bs = _quiet(
        lambda: adt_model.cb(
            x, Z=[0.5], method="bootstrap", n_boot=100, random_state=7
        )
    )
    assert np.all(_overlap(an, bs) > 0.6)


# -- step-stress (acceleration="clock") fits ---------------------------------


@pytest.fixture(scope="module")
def clock_model():
    x, y, i, Z = ss.simulate()
    return ss.fit(x, y, i, Z)


def _dense_first_stage(model):
    """``(Cov(t), Cov(t, gamma), Cov(gamma))`` of a linear-path clock model
    from the whole joint least-squares Jacobian, inverted at once."""
    gamma = np.asarray(model.gamma, dtype=float)
    z_ref = np.asarray(model.stress_ref, dtype=float)
    n, p, q = len(model.units), 2, gamma.size
    blocks, rows_g = [], []
    L = np.zeros((n, n * p + q))
    for k, unit in enumerate(model.units):
        mask = model.i == unit
        order = np.argsort(model.x[mask])
        x = model.x[mask][order]
        s = model.Z[mask][order] - z_ref
        rate = np.exp(s @ gamma)
        dt = np.diff(np.concatenate([[0.0], x]))
        tau = np.cumsum(dt * rate)
        dtau = np.cumsum((dt * rate)[:, None] * s, axis=0)
        a, b = model.path_params[k]
        # the path a + b tau: Jacobian [1, tau], and b dtau/dgamma
        J = np.zeros((x.size, n * p + q))
        J[:, k * p] = 1.0
        J[:, k * p + 1] = tau
        J[:, n * p :] = b * dtau
        blocks.append(J)
        if model.c[k] == 0:
            # t = (threshold - a) / b
            L[k, k * p] = -1.0 / b
            L[k, k * p + 1] = -(model.threshold - a) / b**2
        else:
            L[k, n * p :] = dtau[np.argmax(tau)]
        rows_g.append(dtau)
    J = np.vstack(blocks)
    cov = model.measurement_var * np.linalg.inv(J.T @ J)
    return (
        L @ cov @ L.T,
        L @ cov[:, n * p :],
        cov[n * p :, n * p :],
    )


def test_clock_first_stage_is_the_joint_least_squares_covariance():
    x, y, i, Z = ss.simulate(n_units=8, seed=4)
    model = ss.fit(x, y, i, Z)
    got = clock_first_stage(model)
    want = _dense_first_stage(model)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=0.0)


def test_clock_censored_unit_moves_with_the_clock():
    x, y, i, Z = ss.simulate(n_units=8, seed=3)
    y = y.copy()
    # unit 0 drifts away from the threshold: right censored at its last
    # reference-stress time
    mask = i == 0
    y[mask] = 2.0 - 0.001 * x[mask] + 0.01 * np.sin(x[mask])
    model = _quiet(lambda: ss.fit(x, y, i, Z))
    assert model.c[0] == 1
    got = clock_first_stage(model)
    want = _dense_first_stage(model)
    for g, w in zip(got, want):
        np.testing.assert_allclose(g, w, rtol=1e-5, atol=0.0)
    cov_t, cov_tg, cov_g = got
    # its variance is the clock's alone
    assert cov_t[0, 0] > 0 and np.all(cov_tg[0] != 0)
    band = model.cb([200.0, 240.0], Z=ss.PROFILE)
    assert np.all(np.isfinite(band))


def test_clock_exact_paths_have_no_first_stage_term():
    # readings on the paths: the noise left is that of the clock's
    # numerical search, so the first stage adds nothing the life fit would
    # notice
    x, y, i, Z = ss.simulate(n_units=6, noise=0.0)
    model = ss.fit(x, y, i, Z)
    assert model.measurement_var < 1e-12 * np.var(y)
    cov = model.life_parameter_covariance()
    phi = np.asarray(model.life_model.params, dtype=float)
    np.testing.assert_allclose(
        cov[:2, :2], _plain_information_inverse(model, phi), rtol=1e-9
    )
    assert np.sqrt(cov[2, 2]) < 1e-9 * abs(model.gamma[0])
    # and none at all where the noise is exactly zero
    model.measurement_var = 0.0
    cov_t, cov_tg, cov_g = clock_first_stage(model)
    assert not cov_t.any() and not cov_tg.any() and not cov_g.any()


def test_clock_covariance_orders_life_then_gamma(clock_model):
    cov = clock_model.life_parameter_covariance()
    k = len(clock_model.life_model.params)
    q = len(clock_model.gamma)
    assert cov.shape == (k + q, k + q)
    np.testing.assert_allclose(cov, cov.T)
    assert (
        np.linalg.eigvalsh(cov / np.sqrt(np.outer(*[np.diag(cov)] * 2)))[0]
        > -1e-10
    )
    _, _, cov_g = clock_first_stage(clock_model)
    np.testing.assert_array_equal(cov[k:, k:], cov_g)


def test_clock_gamma_standard_error_matches_simulation():
    # 100 data sets from the model of the step-stress tests: the spread of
    # the estimated stress coefficient against the analytic standard error
    estimates, ses = [], []
    for seed in range(100, 200):
        x, y, i, Z = ss.simulate(seed=seed)
        model = ss.fit(x, y, i, Z)
        estimates.append(model.gamma[0])
        ses.append(np.sqrt(model.life_parameter_covariance()[-1, -1]))
    ratio = np.mean(ses) / np.std(estimates, ddof=1)
    assert 0.85 < ratio < 1.15


@pytest.mark.parametrize("on", ["sf", "ff", "Hf"])
@pytest.mark.parametrize("stress", ["profile", "row"])
def test_clock_band_brackets_the_estimate(clock_model, on, stress):
    Z = ss.PROFILE if stress == "profile" else [ss.Z_LEVELS[1]]
    x = np.array([150.0, 200.0, 240.0])
    est = getattr(clock_model, on)(x, Z=Z)
    band = clock_model.cb(x, Z=Z, on=on)
    assert band.shape == (3, 2)
    assert np.all(band[:, 0] <= est) and np.all(est <= band[:, 1])
    lower = clock_model.cb(x, Z=Z, on=on, bound="lower", alpha_ci=0.025)
    upper = clock_model.cb(x, Z=Z, on=on, bound="upper", alpha_ci=0.025)
    np.testing.assert_allclose(band, np.stack([lower, upper], axis=-1))


def test_clock_band_at_time_zero_is_certain(clock_model):
    band = clock_model.cb([0.0], Z=ss.PROFILE)
    np.testing.assert_array_equal(band, [[1.0, 1.0]])


def test_clock_analytic_survives_a_round_trip(clock_model):
    restored = DegradationModel.from_dict(clock_model.to_dict())
    x = np.array([200.0, 240.0])
    np.testing.assert_allclose(
        restored.cb(x, Z=ss.PROFILE),
        clock_model.cb(x, Z=ss.PROFILE),
        rtol=1e-8,
    )


def test_reml_clock_refuses_the_analytic_bound():
    x, y, i, Z = ss.simulate(n_units=12)
    model = ss.fit(x, y, i, Z, population_method="reml")
    with pytest.raises(NotImplementedError, match="population_method"):
        model.cb([200.0], Z=ss.PROFILE)
    with pytest.raises(NotImplementedError, match="method='bootstrap'"):
        model.life_parameter_covariance()
