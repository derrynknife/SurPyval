r"""
Continuously varying covariates: ``sf_tvc`` / ``Hf_tvc`` along a
``CovariatePath`` (#172, phase 1).

Along a path the cumulative hazard is the integral of the model's hazard,
:math:`H(t) = \int_0^t h(u \mid Z(u))\, du` (for AFT the accelerated age
:math:`\psi(t) = \int_0^t e^{\beta' Z(u)}\, du`, fed through the baseline),
evaluated by adaptive Gauss-Kronrod quadrature to a relative error of
about 1e-10; for Cox it is the baseline jumps weighted by the path at each
jump, exactly. These tests check:

- analytic cases for every family (the quadrature against closed forms);
- the #170 step sum on finer and finer midpoint steps converging to the
  path result, at O(N^-2);
- a flat path giving the matching ``StepSchedule`` to rounding, in every
  family and for Cox, and a Cox path giving the step schedule split at
  its event times exactly;
- conditional survival, periodic paths, declared and undeclared kinks and
  jumps;
- the accuracy warning when the target is missed, and the panel limit;
- shapes, missing values and validation.
"""

import warnings
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma, gammainc

import surpyval as sp
from surpyval import CovariatePath, CoxPH, StepSchedule
from surpyval.univariate.regression import tvc_path
from surpyval.univariate.regression.accelerated_life import (
    AcceleratedLife,
    Power,
)

NOT_CALLABLE: Any = 3.0

# The engine's stated target: relative error on H.
RTOL = 1e-10

T = np.linspace(0.05, 30, 200)


def _model(F, params, p=1):
    """A fitted ``F`` with its parameters replaced by ``params``, so the
    analytic cases have round numbers."""
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, p))
    x = sp.Weibull.random(60, 10, 2, random_state=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = F.fit(x, Z)
    model.params = np.asarray(params, dtype=float)
    model.center = None
    return model


def _ramp(a0, b0):
    """``Z(u) = a0 + b0 u`` up to u = 1000."""
    return CovariatePath.from_points([0, 1000], [a0, a0 + 1000 * b0])


def _rel(got, ref):
    return float(np.max(np.abs(got - ref) / np.abs(ref)))


# -- analytic cases, every family -------------------------------------------


@pytest.mark.parametrize("a0, b0", [(0.0, 0.1), (1.0, -0.05)])
def test_exponential_ph_linear_ramp(a0, b0):
    # H = lam e^{beta a} (e^{beta b t} - 1) / (beta b)
    lam, beta = 0.1, 0.8
    model = _model(sp.ExponentialPH, [lam, beta])
    c = beta * b0
    ref = lam * np.exp(beta * a0) * np.expm1(c * T) / c
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _rel(model.Hf_tvc(T, _ramp(a0, b0)), ref) < RTOL
        assert _rel(model.sf_tvc(T, _ramp(a0, b0)), np.exp(-ref)) < RTOL


@pytest.mark.parametrize("shape", [0.5, 2.0])
def test_weibull_ph_falling_ramp_incomplete_gamma(shape):
    # A shape below 1 has a hazard singular at 0: the graded panels.
    alpha, beta, a0, b0 = 10.0, 0.7, 1.0, -0.1
    model = _model(sp.WeibullPH, [alpha, shape, beta])
    c = -beta * b0
    ref = (
        (shape / alpha**shape)
        * np.exp(beta * a0)
        * gamma(shape)
        * gammainc(shape, c * T)
        / c**shape
    )
    assert _rel(model.Hf_tvc(T, _ramp(a0, b0)), ref) < RTOL


def test_weibull_aft_ramp_is_cumulative_exposure():
    # psi = e^{beta a} (e^{beta b t} - 1) / (beta b), H = (psi / alpha)^k
    alpha, shape, beta, a0, b0 = 10.0, 1.5, 0.4, 0.0, 0.1
    model = _model(sp.WeibullAFT, [alpha, shape, beta])
    c = beta * b0
    psi = np.exp(beta * a0) * np.expm1(c * T) / c
    assert _rel(model.Hf_tvc(T, _ramp(a0, b0)), (psi / alpha) ** shape) < RTOL


def test_weibull_ah_ramp():
    # H = H0 + beta (a t + b t^2 / 2)
    alpha, shape, beta, a0, b0 = 10.0, 2.0, 0.01, 0.5, 0.02
    model = _model(sp.WeibullAH, [alpha, shape, beta])
    ref = (T / alpha) ** shape + beta * (a0 * T + 0.5 * b0 * T**2)
    assert _rel(model.Hf_tvc(T, _ramp(a0, b0)), ref) < RTOL


def test_po_exponential_baseline_closed_form():
    # With phi(Z(u)) = 1 + kappa (e^{lam u} - 1) the PO hazard form
    # h = O0' / (O0 + phi) integrates to log1p((1+k)(e^{lam t}-1))/(1+k).
    alpha, beta, kappa = 10.0, 0.5, 0.3
    lam = 1 / alpha
    model = _model(sp.WeibullPO, [alpha, 1.0, beta])
    path = CovariatePath.from_callable(
        lambda u: np.log1p(kappa * np.expm1(lam * u)) / beta
    )
    ref = np.log1p((1 + kappa) * np.expm1(lam * T)) / (1 + kappa)
    assert _rel(model.Hf_tvc(T, path), ref) < RTOL


@pytest.mark.parametrize("shape", [0.5, 2.0])
def test_po_ramp_matches_quadrature_of_the_hazard_form(shape):
    # Brute force: scipy quad of h0 / (F0 + phi S0) along the ramp, written
    # from the definition, not the model's functions.
    alpha, beta = 10.0, 0.8
    model = _model(sp.WeibullPO, [alpha, shape, beta])
    tt = np.array([2.0, 5.0, 10.0, 20.0, 40.0])

    def h(u):
        s0 = np.exp(-((u / alpha) ** shape))
        h0 = (shape / alpha) * (u / alpha) ** (shape - 1)
        return h0 / (1 - s0 + np.exp(beta * (-1 + 0.1 * u)) * s0)

    ref = np.array(
        [quad(h, 0, t, epsabs=0, epsrel=1e-13, limit=2000)[0] for t in tt]
    )
    assert _rel(model.Hf_tvc(tt, _ramp(-1.0, 0.1)), ref) < RTOL


def test_sine_path_matches_scipy_quad():
    lam, beta = 0.1, 0.9
    model = _model(sp.ExponentialPH, [lam, beta])

    def f(u):
        return np.sin(2 * np.pi * u / 3.0)

    tt = np.array([1.0, 5.0, 12.3, 30.0])
    ref = np.array(
        [
            quad(
                lambda u: lam * np.exp(beta * f(u)),
                0,
                t,
                epsabs=0,
                epsrel=1e-13,
                limit=2000,
            )[0]
            for t in tt
        ]
    )
    assert _rel(model.Hf_tvc(tt, CovariatePath.from_callable(f)), ref) < RTOL
    # Periodic with the pattern given over one period only.
    one = CovariatePath.from_callable(f, period=3.0)
    assert _rel(model.Hf_tvc(tt, one), ref) < RTOL


def test_multivariate_path():
    # Two covariates, one ramp each: phi = e^{b1 z1 + b2 z2} is again an
    # exponential in u.
    lam, b1, b2 = 0.1, 0.5, -0.3
    model = _model(sp.ExponentialPH, [lam, b1, b2], p=2)
    path = CovariatePath.from_points([0, 10], [[0.0, 1.0], [1.0, 3.0]])
    tt = np.array([2.0, 7.0, 10.0])
    c = 0.1 * b1 + 0.2 * b2
    ref = lam * np.exp(b2) * np.expm1(c * tt) / c
    assert _rel(model.Hf_tvc(tt, path), ref) < RTOL


# -- the #170 step sum is the limit case -------------------------------------


@pytest.mark.parametrize(
    "F, params",
    [
        (sp.ExponentialPH, [0.1, 0.8]),
        (sp.WeibullAFT, [10.0, 1.5, 0.4]),
        (sp.WeibullAH, [10.0, 2.0, 0.01]),
        (sp.WeibullPO, [10.0, 2.0, 0.8]),
    ],
)
def test_midpoint_steps_converge_to_the_path(F, params):
    # (A linear path would make the additive hazards' midpoint sum exact.)
    model = _model(F, params)
    path = CovariatePath.from_callable(
        lambda u: 0.5 + 0.1 * u + 0.3 * np.sin(u / 2)
    )
    tt = np.array([5.0, 12.0, 20.0])
    H = model.Hf_tvc(tt, path)
    errors = []
    for n in (10, 100, 1000):
        e = np.linspace(0, 20, n + 1)
        mid = path(0.5 * (e[:-1] + e[1:]))
        sched = StepSchedule.from_changepoints(e[:-1], mid)
        errors.append(_rel(model.Hf_tvc(tt, sched), H))
    # O(N^-2): a tenfold refinement cuts the error about a hundredfold.
    assert errors[0] > errors[1] > 50 * errors[2], errors
    assert errors[2] < 1e-5


# -- a flat path is the step schedule ----------------------------------------

JUMPS = CovariatePath.from_points(
    [0, 5, 5, 12, 12], [0.3, 0.3, 1.0, 1.0, -0.2]
)
STEPS = StepSchedule.from_changepoints([0, 5, 12], [0.3, 1.0, -0.2])

FAMILIES = [
    (sp.WeibullPH, [10, 0.5, 0.7]),
    (sp.WeibullAFT, [10, 1.5, 0.4]),
    (sp.WeibullAH, [10, 2, 0.01]),
    (sp.WeibullPO, [10, 2, 0.5]),
    # baselines defined below 0: the value at 0 is held before it
    (sp.NormalPH, [10, 3, 0.3]),
    (sp.LogisticAFT, [10, 3, 0.3]),
    (sp.GumbelPO, [10, 3, 0.3]),
    (sp.NormalAH, [10, 3, 0.01]),
    (sp.LogNormalAH, [2, 1, 0.01]),
]
FAMILY_IDS = [
    "WeibullPH",
    "WeibullAFT",
    "WeibullAH",
    "WeibullPO",
    "NormalPH",
    "LogisticAFT",
    "GumbelPO",
    "NormalAH",
    "LogNormalAH",
]


@pytest.mark.parametrize("F, params", FAMILIES, ids=FAMILY_IDS)
def test_flat_path_is_the_step_schedule(F, params):
    model = _model(F, params)
    tt = np.concatenate([[-3.0, 0.0, 5.0, 12.0], T])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        np.testing.assert_allclose(
            model.Hf_tvc(tt, JUMPS),
            model.Hf_tvc(tt, STEPS),
            rtol=1e-13,
            atol=1e-15,
        )
        # The conditional survival integrates from given on (no
        # subtraction), so it agrees to rounding, not bit for bit.
        np.testing.assert_allclose(
            model.sf_tvc(tt, JUMPS, given=4.0),
            model.sf_tvc(tt, STEPS, given=4.0),
            rtol=1e-12,
            atol=1e-300,
        )
    # Only the additive model's own warning (#376: NormalAH's hazard is
    # negative far below its mean), and never a raw numpy warning.
    for w in caught:
        assert str(w.message).startswith("The additive hazard"), w.message


def test_constant_path_is_sf():
    model = _model(sp.WeibullPH, [10, 2, 0.7])
    tt = np.array([-1.0, 0.0, 1.0, 7.0, 20.0])
    np.testing.assert_allclose(
        model.sf_tvc(tt, CovariatePath.from_points([0], [0.4])),
        model.sf(tt, [0.4]),
        rtol=1e-13,
    )


def _cox(seed=3, n=80):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    x = rng.exponential(5 * np.exp(-0.5 * Z[:, 0] + 0.3 * Z[:, 1]))
    return CoxPH.fit(x, Z)


def test_cox_flat_path_is_the_step_schedule():
    model = _cox()
    # A jump exactly at an event time: the old value is at risk there
    # ((start, stop] rows, #259), in both.
    jump = float(np.sort(model.x)[20])
    path = CovariatePath.from_points(
        [0, jump, jump, 6, 6], [[0.3, 1], [0.3, 1], [1, -1], [1, -1], [0, 0]]
    )
    sched = StepSchedule.from_changepoints(
        [0, jump, 6], [[0.3, 1], [1, -1], [0, 0]]
    )
    tt = np.concatenate([[-1.0, 0.0, jump], np.sort(model.x), [100.0]])
    np.testing.assert_array_equal(
        model.Hf_tvc(tt, path), model.Hf_tvc(tt, sched)
    )
    np.testing.assert_allclose(
        model.sf_tvc(tt, path, given=1.5),
        model.sf_tvc(tt, sched, given=1.5),
        rtol=1e-14,
    )


def test_cox_path_is_the_schedule_split_at_the_event_times():
    # Only the covariate at the baseline jump times matters, so a ramp is
    # exactly the step schedule that takes the ramp's value at each jump.
    model = _cox()
    ramp = CovariatePath.from_points([0, 10], [[0, 0], [1, 2]])
    bt = model.x
    sched = StepSchedule.from_changepoints(
        np.concatenate([[0.0], bt]), ramp(np.concatenate([bt, [bt[-1] + 1]]))
    )
    tt = np.concatenate([[0.0], np.sort(model.x), [100.0]])
    np.testing.assert_array_equal(
        model.Hf_tvc(tt, ramp), model.Hf_tvc(tt, sched)
    )
    # ... and the weighted sum of the jumps.
    weight = model.h0 * np.exp(ramp(bt) @ model.beta)
    np.testing.assert_allclose(
        model.Hf_tvc([3.0], ramp), [weight[bt <= 3.0].sum()], rtol=1e-14
    )


# -- conditional survival ----------------------------------------------------


# NormalAH's hazard is negative far below its mean (#376).
@pytest.mark.filterwarnings("ignore:The additive hazard")
@pytest.mark.parametrize("F, params", FAMILIES, ids=FAMILY_IDS)
def test_given_is_the_ratio_of_survivals(F, params):
    model = _model(F, params)
    path = CovariatePath.from_callable(
        lambda u: 0.5 + 0.5 * np.sin(u / 2), breakpoints=[3.0]
    )
    tt = np.array([4.0, 6.0, 9.0, 15.0])
    np.testing.assert_allclose(
        model.sf_tvc(tt, path, given=4.0),
        model.sf_tvc(tt, path) / model.sf_tvc(4.0, path),
        rtol=1e-11,
    )
    # A conditioning age at or before 0 conditions on nothing (for a
    # baseline that starts at 0) or divides by sf(given).
    np.testing.assert_allclose(
        model.sf_tvc(tt, path, given=-1.0),
        model.sf_tvc(tt, path) / model.sf_tvc(-1.0, path),
        rtol=1e-11,
    )
    assert np.isnan(model.sf_tvc(tt, path, given=np.nan)).all()


def test_given_sums_from_given_without_cancellation():
    # Late in life H(given) is large and the increment small: summing from
    # given keeps the digits a subtraction loses.
    lam, beta = 1.0, 0.5
    model = _model(sp.ExponentialPH, [lam, beta])
    path = _ramp(0.0, 0.1)
    g, x = 60.0, 60.0 + 1e-6
    c = beta * 0.1
    exact = lam * (np.exp(c * x) - np.exp(c * g)) / c
    got = -np.log(model.sf_tvc(x, path, given=g))
    assert abs(got / exact - 1) < 1e-8


def test_cox_given_is_the_ratio_of_survivals():
    model = _cox()
    ramp = CovariatePath.from_points([0, 10], [[0, 0], [1, 2]])
    tt = np.array([1.0, 2.0, 4.0, 8.0])
    np.testing.assert_allclose(
        model.sf_tvc(tt, ramp, given=1.0),
        model.sf_tvc(tt, ramp) / model.sf_tvc(1.0, ramp),
        rtol=1e-13,
    )
    assert np.isnan(model.sf_tvc(tt, ramp, given=np.nan)).all()


# -- kinks, jumps and periods ------------------------------------------------


def _kink_H(t, lam, beta):
    # Z = |u - 7| / 5, exponential PH
    k = beta / 5
    before = lam / k * (np.exp(7 * k) - np.exp(k * (7 - np.minimum(t, 7))))
    after = lam / k * np.expm1(k * np.maximum(t - 7, 0))
    return before + after


@pytest.mark.parametrize("breakpoints", [[7.0], None])
def test_callable_kink_declared_or_found(breakpoints):
    lam, beta = 0.1, 0.9
    model = _model(sp.ExponentialPH, [lam, beta])
    path = CovariatePath.from_callable(
        lambda u: np.abs(u - 7) / 5, breakpoints=breakpoints
    )
    tt = np.array([1.0, 5.0, 12.3, 30.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _rel(model.Hf_tvc(tt, path), _kink_H(tt, lam, beta)) < RTOL


@pytest.mark.parametrize("breakpoints", [[7.3], None])
def test_callable_jump_declared_or_found(breakpoints):
    lam, beta = 0.1, 0.9
    model = _model(sp.ExponentialPH, [lam, beta])
    path = CovariatePath.from_callable(
        lambda u: (u > 7.3).astype(float), breakpoints=breakpoints
    )
    tt = np.array([1.0, 5.0, 12.3, 30.0])
    ref = lam * (tt + np.expm1(beta) * np.maximum(tt - 7.3, 0))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _rel(model.Hf_tvc(tt, path), ref) < RTOL


def test_from_points_jumps_equal_the_step_schedule():
    # The same changes given as jumps of a path and as change-points.
    model = _model(sp.WeibullPH, [10, 2, 0.7])
    path = CovariatePath.from_points(
        [0, 2, 2, 4, 4, 9, 9], [0.1, 0.1, 0.8, 0.8, -0.3, -0.3, 0.5]
    )
    sched = StepSchedule.from_changepoints([0, 2, 4, 9], [0.1, 0.8, -0.3, 0.5])
    np.testing.assert_allclose(
        model.sf_tvc(T, path), model.sf_tvc(T, sched), rtol=1e-13
    )


def test_jump_value_is_the_left_value():
    path = CovariatePath.from_points([0, 5, 5], [0.0, 1.0, 3.0])
    np.testing.assert_array_equal(path([5.0]), [[1.0]])
    np.testing.assert_array_equal(path([5.0 + 1e-12]).round(6), [[3.0]])


def test_periodic_points_equal_the_unrolled_path():
    model = _model(sp.WeibullPH, [10, 2, 0.7])
    saw = CovariatePath.from_points([0, 4, 10], [0.0, 1.0, 0.2], period=10)
    unrolled = CovariatePath.from_points(
        [0, 4, 10, 10, 14, 20, 20, 24, 30, 30, 34, 40],
        [0.0, 1.0, 0.2, 0.0, 1.0, 0.2, 0.0, 1.0, 0.2, 0.0, 1.0, 0.2],
    )
    tt = np.array([3.0, 10.0, 17.5, 29.0, 35.0])
    np.testing.assert_allclose(
        model.Hf_tvc(tt, saw), model.Hf_tvc(tt, unrolled), rtol=1e-13
    )
    # The value before the jump at each period's end is the pattern's end.
    np.testing.assert_allclose(
        saw([10.0, 20.0, 25.0]).ravel(), [0.2, 0.2, 1 - 0.8 / 6]
    )


def test_breakpoints_repeat_with_the_period():
    path = CovariatePath.from_points([0, 5], [0.0, 1.0], period=8)
    np.testing.assert_array_equal(path.breakpoints(20), [5, 8, 13, 16])
    call = CovariatePath.from_callable(np.sin, breakpoints=[2.0], period=4)
    np.testing.assert_array_equal(call.breakpoints(9), [2, 4, 6, 8])


# -- warnings and the panel limit --------------------------------------------


def test_accuracy_warning_once_with_counts(monkeypatch):
    # Two rounds cannot close in on an undeclared jump.
    monkeypatch.setattr(tvc_path, "_MAX_ROUNDS", 2)
    model = _model(sp.ExponentialPH, [0.1, 0.9])
    path = CovariatePath.from_callable(lambda u: (u > 7.3).astype(float))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.sf_tvc(np.array([1.0, 5.0, 12.3, 30.0]), path)
    assert len(caught) == 1, [str(w.message) for w in caught]
    w = caught[0]
    assert w.category is RuntimeWarning
    assert w.filename == __file__
    assert "at 2 of the 4 query time(s)" in str(w.message)
    assert "after 2 rounds" in str(w.message)
    assert "breakpoints" in str(w.message)
    # Declaring the jump resolves it.
    fixed = CovariatePath.from_callable(
        lambda u: (u > 7.3).astype(float), breakpoints=[7.3]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.Hf_tvc(np.array([1.0, 5.0, 12.3, 30.0]), fixed)


def test_accuracy_warning_at_the_panel_limit(monkeypatch):
    # A path that oscillates without limit near 7 cannot be integrated to
    # the target: refinement stops at the panel limit, and warns.
    monkeypatch.setattr(tvc_path, "_PANEL_CAP", 2000)
    model = _model(sp.ExponentialPH, [0.1, 0.9])
    path = CovariatePath.from_callable(lambda u: np.sin(1 / (u - 7)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        H = model.Hf_tvc(np.array([5.0, 12.0]), path)
    assert len(caught) == 1, [str(w.message) for w in caught]
    assert caught[0].filename == __file__
    assert "limit of 2,000 panels" in str(caught[0].message)
    assert "at 1 of the 2 query time(s)" in str(caught[0].message)
    assert np.isfinite(H).all()


def test_too_many_panels_raises():
    model = _model(sp.ExponentialPH, [0.1, 0.9])
    fast = CovariatePath.from_points(
        [0, 5e-4, 1e-3], [0.0, 1.0, 0.0], period=1e-3
    )
    with pytest.raises(ValueError, match="more than the limit of 1,000,000"):
        model.sf_tvc([1e4], fast)
    with pytest.raises(ValueError, match="repeats every 0.001"):
        model.sf_tvc([1e4], fast)


def test_ah_negative_hazard_warns_once_along_a_path():
    # h0 + beta Z(u) turns negative as the ramp falls (#376).
    model = _model(sp.WeibullAH, [10.0, 2.0, 0.05])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.sf_tvc(T, _ramp(0.0, -0.5))
    assert len(caught) == 1, [str(w.message) for w in caught]
    assert "additive hazard" in str(caught[0].message)
    assert caught[0].filename == __file__


def test_no_raw_numpy_warnings():
    # A log-time baseline at t = 0, and a hazard singular at 0.
    for F, params in [
        (sp.LogNormalPH, [2, 1, 0.3]),
        (sp.WeibullPH, [10, 0.3, 0.3]),
        (sp.GammaAFT, [2, 1, 0.3]),
    ]:
        model = _model(F, params)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            model.sf_tvc(np.array([0.0, 1e-300, 1.0, 50.0]), _ramp(0, 0.1))


# -- shapes, missing values and refusals -------------------------------------


@pytest.mark.parametrize("model", ["parametric", "cox"])
def test_shape_in_shape_out(model):
    m = _model(sp.WeibullPH, [10, 2, 0.7]) if model == "parametric" else None
    if m is None:
        m = _cox()
        path = CovariatePath.from_points([0, 10], [[0, 0], [1, 2]])
    else:
        path = _ramp(0.0, 0.1)
    scalar = m.sf_tvc(3.0, path)
    assert np.shape(scalar) == ()
    grid = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    got = m.sf_tvc(grid, path)
    assert got.shape == (2, 3)
    np.testing.assert_allclose(got.ravel(), m.sf_tvc(grid.ravel(), path))
    for empty in (np.array([]), np.empty((0, 3))):
        assert m.Hf_tvc(empty, path).shape == empty.shape
    # A missing time is nan, element by element.
    got = m.sf_tvc([1.0, np.nan, 4.0], path)
    assert np.isnan(got[1]) and np.isfinite(got[[0, 2]]).all()
    np.testing.assert_allclose(got[[0, 2]], m.sf_tvc([1.0, 4.0], path))
    assert np.isnan(m.sf_tvc([np.nan], path)).all()


def test_path_call_shapes_and_missing_time():
    path = CovariatePath.from_points([0, 10], [[0.0, 1.0], [1.0, 3.0]])
    assert path([1.0, 2.0]).shape == (2, 2)
    assert path(5.0).shape == (1, 2)
    assert np.isnan(path([np.nan])).all()
    assert path.p == 2
    assert repr(path) == "CovariatePath(points, 2 knot(s), p=2)"


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: CovariatePath.from_points([0, 1], [0.0, np.nan]), "'values'"),
        (lambda: CovariatePath.from_points([0, np.nan], [0, 1]), "'times'"),
        (lambda: CovariatePath.from_points([0, 2, 1], [0, 1, 2]), "non-dec"),
        (
            lambda: CovariatePath.from_points([0, 1, 1, 1], [0, 1, 2, 3]),
            "more than twice",
        ),
        (lambda: CovariatePath.from_points([0, 1], [0, 1, 2]), "same length"),
        (
            lambda: CovariatePath.from_points([0, 12], [0, 1], period=10),
            "period",
        ),
        (lambda: CovariatePath.from_points([], []), "at least one"),
        (lambda: CovariatePath.from_callable(NOT_CALLABLE), "'func'"),
        (lambda: CovariatePath.from_callable(np.sin, p=0), "'p'"),
        (
            lambda: CovariatePath.from_callable(np.sin, breakpoints=[np.inf]),
            "'breakpoints'",
        ),
        (lambda: CovariatePath.from_callable(np.sin, period=-1), "'period'"),
    ],
)
def test_validation(build, match):
    with pytest.raises(ValueError, match=match):
        build()


def test_callable_output_is_checked_when_used():
    model = _model(sp.WeibullPH, [10, 2, 0.7])
    wrong_shape = CovariatePath.from_callable(
        lambda u: np.ones((u.size, 3)), p=1
    )
    with pytest.raises(ValueError, match="'func' must return"):
        model.sf_tvc([1.0], wrong_shape)
    missing = CovariatePath.from_callable(
        lambda u: np.where(u > 3, np.nan, 0.0)
    )
    with pytest.raises(ValueError, match="'func' returned a missing"):
        model.sf_tvc([5.0], missing)
    # A scalar is a constant path.
    const = CovariatePath.from_callable(lambda u: 0.4)
    np.testing.assert_allclose(
        model.sf_tvc(T, const), model.sf(T, [0.4]), rtol=1e-13
    )


def test_refusals():
    model = _model(sp.WeibullPH, [10, 2, 0.7])
    with pytest.raises(ValueError, match="xl must not be given"):
        model.sf_tvc([1.0], _ramp(0, 1), xl=[0.0])
    with pytest.raises(ValueError, match="the path has 2 covariate"):
        model.sf_tvc([1.0], CovariatePath.from_points([0], [[0.0, 1.0]]))
    cox = _cox()
    with pytest.raises(ValueError, match="xl must not be given"):
        cox.sf_tvc([1.0], _ramp(0, 1), xl=[0.0])
    with pytest.raises(ValueError, match="the path has 1 covariate"):
        cox.sf_tvc([1.0], _ramp(0, 1))
    rng = np.random.default_rng(0)
    stress = rng.uniform(1, 3, 100)
    x = rng.weibull(2, 100) * 10 / stress
    al = AcceleratedLife(sp.Weibull, Power).fit(x=x, Z=stress)
    with pytest.raises(NotImplementedError):
        al.sf_tvc([1.0], _ramp(1, 0.1))
