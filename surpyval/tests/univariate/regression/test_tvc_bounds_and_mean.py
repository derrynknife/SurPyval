"""Time-varying covariates along a path: bounds and mean life (#172), and
accelerated life along steps and paths.

- ``cb_tvc``: the delta method on the logit of ``sf_tvc``, on a quadrature
  mesh held at the fitted parameters, against the delta method through the
  closed-form cumulative hazard of a ramp.
- ``mean_tvc``: the integral of ``sf_tvc``, against closed forms.
- Accelerated life along steps and paths is cumulative exposure.
"""

import copy
import warnings

import numpy as np
import pytest
from scipy.special import exp1
from scipy.stats import norm

import surpyval as sp
from surpyval import CovariatePath, StepSchedule
from surpyval.utils.linalg import delta_method_se


def _model(F, params):
    """A fitted ``F`` with its parameters replaced by ``params``."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 1))
    x = sp.Weibull.random(60, 10, 2, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = F.fit(x, Z)
    model.params = np.asarray(params, dtype=float)
    model.center = None
    return model


def _ramp(a0, b0, top=1e6):
    """``Z(u) = a0 + b0 u`` up to ``top``."""
    return CovariatePath.from_points([0, top], [a0, a0 + top * b0])


# -- cb_tvc --------------------------------------------------------------


@pytest.fixture(scope="module")
def exp_ph():
    rng = np.random.default_rng(3)
    Z = rng.uniform(0, 2, (300, 1))
    x = rng.exponential(10 * np.exp(-0.7 * Z[:, 0]))
    return sp.ExponentialPH.fit(x, Z)


def test_cb_tvc_is_the_delta_method_through_the_path_integral(exp_ph):
    # Along Z(u) = b u, H = lam e^{-beta c} (e^{beta b t} - 1) / (beta b)
    # in the parameters cb works in (the centred fit's, centre c): the
    # bounds from the quadrature on its frozen mesh are those through the
    # closed form, to the delta method's own step error.
    b0 = 0.1
    t = np.array([2.0, 5.0, 10.0, 20.0])
    params, center, cov = exp_ph._inference_state()
    c = 0.0 if center is None else float(np.ravel(center)[0])

    # On the Exponential's band scale, log H (#504).
    def log_H(p):
        H = p[0] * np.exp(-p[1] * c) * np.expm1(p[1] * b0 * t) / (p[1] * b0)
        return np.log(H)

    se = delta_method_se(log_H, params, cov)
    z = norm.ppf(0.975)
    ref = np.stack(
        [
            np.exp(-np.exp(log_H(params) + z * se)),
            np.exp(-np.exp(log_H(params) - z * se)),
        ],
        axis=-1,
    )
    np.testing.assert_allclose(
        exp_ph.cb_tvc(t, _ramp(0.0, b0)), ref, rtol=1e-9, atol=0
    )


def test_cb_tvc_conditional(exp_ph):
    # Survival to t <= given is certain: the bound is [1, 1]. After it the
    # bounds bracket sf_tvc(given=), along a schedule and along a path.
    t = np.array([2.0, 5.0, 10.0, 20.0])
    for Z in (
        StepSchedule.from_changepoints([0, 5, 10], [[0.0], [1.0], [0.5]]),
        _ramp(0.0, 0.1),
    ):
        b = exp_ph.cb_tvc(t, Z, given=5.0)
        s = exp_ph.sf_tvc(t, Z, given=5.0)
        np.testing.assert_array_equal(b[:2], 1.0)
        assert np.all((b[2:, 0] < s[2:]) & (s[2:] < b[2:, 1]))
        # A one-sided 95% bound lies inside the two-sided one.
        lower = exp_ph.cb_tvc(t, Z, given=5.0, bound="lower")
        assert np.all((b[2:, 0] < lower[2:]) & (lower[2:] < s[2:]))
        assert np.isnan(exp_ph.cb_tvc(t, Z, given=np.nan)).all()


def test_cb_tvc_refusals(exp_ph):
    with pytest.raises(ValueError, match="not the hazard or the density"):
        exp_ph.cb_tvc([1.0], _ramp(0, 1), on="hf")
    with pytest.raises(ValueError, match="bound"):
        exp_ph.cb_tvc([1.0], _ramp(0, 1), bound="both")
    restored = copy.copy(exp_ph)
    restored.res = None
    with pytest.raises(ValueError, match="Confidence bounds"):
        restored.cb_tvc([1.0], _ramp(0, 1))


# -- mean_tvc ------------------------------------------------------------


LAM, BETA, B0 = 0.1, 0.8, 0.1


def _ramp_mean(given=0.0):
    # S(t | g) = exp(-c1 e^{c g} (e^{c (t - g)} - 1)), c = beta b, c1 = lam
    # / c, whose integral is e^{c1'} E1(c1') / c with c1' = c1 e^{c g}.
    c = BETA * B0
    c1 = LAM / c * np.exp(c * given)
    return np.exp(c1) * exp1(c1) / c


def test_mean_tvc_along_a_ramp_in_closed_form():
    model = _model(sp.ExponentialPH, [LAM, BETA])
    ramp = _ramp(0.0, B0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mean = model.mean_tvc(ramp)
        mrl = model.mean_tvc(ramp, given=7.0)
    assert abs(mean / _ramp_mean() - 1) < 1e-10, mean
    assert abs(mrl / _ramp_mean(7.0) - 1) < 1e-10, mrl
    assert np.isnan(model.mean_tvc(ramp, given=np.nan))


def test_mean_tvc_midpoint_steps_converge_to_the_ramp():
    # A step schedule is integrated as its piecewise-constant path: the
    # mean of N midpoint steps of the ramp converges to the ramp's at
    # O(N^-2).
    model = _model(sp.ExponentialPH, [LAM, BETA])
    err = []
    for n in (10, 100, 1000):
        e = np.linspace(0, 80, n + 1)
        mid = B0 * 0.5 * (e[:-1] + e[1:])
        steps = StepSchedule.from_changepoints(
            np.r_[e[:-1], 80.0], np.r_[mid, B0 * 80]
        )
        err.append(abs(model.mean_tvc(steps) / _ramp_mean() - 1))
    assert err[0] > 50 * err[1] > 2500 * err[2], err
    assert err[2] < 1e-5, err


def test_mean_tvc_cyclic_schedule_is_its_periodic_path():
    model = _model(sp.WeibullPH, [10.0, 2.0, 0.7])
    cyc = StepSchedule.cyclic([0, 2], [[1.0], [0.0]], period=3)
    path = CovariatePath.from_points(
        [0, 2, 2, 3], [1.0, 1.0, 0.0, 0.0], period=3
    )
    assert abs(model.mean_tvc(cyc) / model.mean_tvc(path) - 1) < 1e-12


def test_mean_tvc_is_infinite_when_survival_levels_off():
    # Z(u) = -0.2 u drives the PH hazard to 0: H converges, and a
    # fraction of units never fails.
    model = _model(sp.WeibullPH, [100.0, 2.0, 1.0])
    down = CovariatePath.from_callable(lambda u: -0.2 * u)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mean = model.mean_tvc(down)
    assert mean == np.inf
    assert len(caught) == 1, [str(w.message) for w in caught]
    assert caught[0].filename == __file__
    sf_end = model.sf_tvc(1e6, down)
    assert "still {:.4g}".format(sf_end) in str(caught[0].message)
    assert "never fails" in str(caught[0].message)


def test_mean_tvc_below_zero_support():
    # A Normal baseline has mass below 0, where the value at 0 holds: the
    # mean takes off the area under F there, so a constant path gives the
    # distribution's mean. For a Normal AFT model at z that is mu / phi(z).
    model = _model(sp.NormalAFT, [5.0, 3.0, 0.4])
    mean = model.mean_tvc(StepSchedule.constant([0.5]))
    assert abs(mean - 5.0 / np.exp(0.4 * 0.5)) < 1e-9


# -- accelerated life along steps and paths --------------------------------


@pytest.mark.parametrize(
    "dist", [sp.Weibull, sp.Exponential, sp.Gamma, sp.LogNormal]
)
def test_accelerated_life_ramp_is_cumulative_exposure(dist):
    # L(V) = A V^n along V = a + b u ages a unit by
    # psi = int du / L = (V^{1-n} - a^{1-n}) / (A b (1 - n)), and S is the
    # distribution at unit life at psi (Nelson's cumulative exposure).
    rng = np.random.default_rng(0)
    stress = np.repeat([1.0, 2.0, 3.0], 60)
    x = rng.weibull(2.0, 180) * 50 / stress
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        al = sp.AcceleratedLife(dist, sp.life_models.Power).fit(x=x, Z=stress)
    A, n = al.params[al.k_dist :]
    a0, b0 = 1.0, 0.05
    t = np.array([1.0, 5.0, 20.0, 60.0])
    V = a0 + b0 * t
    psi = (V ** (1 - n) - a0 ** (1 - n)) / (A * b0 * (1 - n))
    unit = np.array(al.params[: al.k_dist], dtype=float)
    unit[al.model.param_map[al.model.life_parameter]] = (
        al.model.param_transform(1.0)
    )
    H = np.asarray(dist.Hf(psi, *unit), dtype=float)
    ramp = _ramp(a0, b0, top=1000)
    np.testing.assert_allclose(al.Hf_tvc(t, ramp), H, rtol=1e-12)
    # Midpoint steps converge to it at O(N^-2).
    err = []
    for N in (100, 1000):
        e = np.linspace(0, 60, N + 1)
        mid = a0 + b0 * 0.5 * (e[:-1] + e[1:])
        steps = StepSchedule.from_changepoints(e[:-1], mid)
        err.append(np.max(np.abs(al.Hf_tvc(t, steps) / H - 1)))
    assert 80 < err[0] / err[1] < 120, err
    # cb_tvc at a constant stress is cb, and brackets the ramp's survival.
    flat = CovariatePath.from_points([0], [2.0])
    np.testing.assert_allclose(
        al.cb_tvc(t, flat), al.cb(t, [2.0]), rtol=1e-8, atol=1e-12
    )
    b = al.cb_tvc(t, ramp)
    s = al.sf_tvc(t, ramp)
    assert np.all((b[:, 0] < s) & (s < b[:, 1]))


def test_periodic_shortcut_for_a_time_scaling_family():
    # A sawtooth 0 -> 1 every unit of time: the accelerated age over a
    # period is (e^beta - 1) / beta, so psi(t) = k (e^beta - 1) / beta +
    # (e^{beta r} - 1) / beta. Only one period is integrated, so 10^7
    # periods cost no more than one, where a hazard family (whose baseline
    # ages) needs a panel per period and refuses.
    alpha, shape, beta = 100.0, 2.0, 0.7
    model = _model(sp.WeibullAFT, [alpha, shape, beta])
    saw = CovariatePath.from_points([0, 1], [0.0, 1.0], period=1.0)
    t = np.array([0.3, 5.5, 77.25, 400.0, 1e7 + 0.5])
    k, r = np.floor(t), t - np.floor(t)
    psi = k * np.expm1(beta) / beta + np.expm1(beta * r) / beta
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        H = model.Hf_tvc(t, saw)
    np.testing.assert_allclose(H, (psi / alpha) ** shape, rtol=1e-12)
    ph = _model(sp.WeibullPH, [alpha, shape, beta])
    with pytest.raises(ValueError, match="quadrature panels"):
        ph.Hf_tvc(t, saw)
    # given: from the conditioning age, with the shortcut on both ends.
    S = model.sf_tvc(t[2:], saw, given=t[1])
    np.testing.assert_allclose(S, np.exp(-(H[2:] - H[1])), rtol=1e-10, atol=0)


def test_cb_below_the_support_is_the_estimate():
    # Before 0 nothing has happened: sf is 1, and so is its bound. An
    # additive hazard's beta'Z x is not 0 at x < 0, and cb gave a band
    # around a survival of about 0.9 there (cb_tvc's constant-path
    # property found it).
    rng = np.random.default_rng(1)
    Z = rng.uniform(0, 1, (300, 1))
    x = rng.weibull(2, 300) * 10 / (1 + Z[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.WeibullAH.fit(x, Z)
    t = np.array([-1.0, 0.0, 5.0])
    b = model.cb(t, [1.0])
    np.testing.assert_array_equal(b[:2], 1.0)
    np.testing.assert_array_equal(model.sf(t[:2], [1.0]), 1.0)
    assert b[2, 0] < model.sf(5.0, [1.0]) < b[2, 1]


def test_mean_tvc_additive_hazard_turning_negative():
    # h = h0 + beta z with h0 -> 0 (a Weibull shape below 1) and beta z < 0:
    # H falls without limit, survival rises above 1, and the mean is
    # infinite. Two warnings, both pointing here -- the negative hazard
    # (#376) and the infinite mean -- and no raw numpy overflow.
    model = _model(sp.WeibullAH, [10.0, 0.5, 0.1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mean = model.mean_tvc(StepSchedule.constant([-1.0]))
    assert mean == np.inf
    messages = [str(w.message) for w in caught]
    assert len(caught) == 2, messages
    assert all(w.filename == __file__ for w in caught), messages
    assert "negative" in messages[0] and "not fallen to 0" in messages[1]
