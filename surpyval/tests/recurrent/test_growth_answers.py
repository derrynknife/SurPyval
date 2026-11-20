"""The answers asked of a fitted growth model (``remaining_faults``,
``next_failure_sf_cb``, ``time_to_intensity``, ``time_to_reliability``
and their bounds), on every fitted Poisson process.

Each answer against its closed form, the bounds against the bounds they
are made from, and the refusals' words. The bounds' coverage is studied
in ``calibration/test_coverage_growth_answers.py``.
"""

import numpy as np
import pytest
from scipy.stats import norm

from surpyval.recurrent import (
    HPP,
    CoxLewis,
    CrowAMSAA,
    Duane,
    GoelOkumoto,
    InflectionS,
    MusaOkumoto,
    YamadaDelayedS,
)
from surpyval.recurrent.parametric.growth_answers import (
    NeverReached,
    last_time_above,
)

WEEKS = np.column_stack([np.arange(8.0), np.arange(1.0, 9.0)])
COUNTS = [12, 9, 8, 5, 4, 4, 2, 1]


@pytest.fixture(scope="module")
def fitted():
    return GoelOkumoto.fit(WEEKS, c=[2] * 8, n=COUNTS)


# -- remaining faults -------------------------------------------------------
def test_remaining_faults_closed_forms():
    t = np.array([0.0, 5.0, 20.0, 80.0])
    a, b, psi = 100.0, 0.05, 3.0
    np.testing.assert_allclose(
        GoelOkumoto.from_params([a, b]).remaining_faults(t),
        a * np.exp(-b * t),
        rtol=1e-13,
    )
    np.testing.assert_allclose(
        YamadaDelayedS.from_params([a, b]).remaining_faults(t),
        a * (1 + b * t) * np.exp(-b * t),
        rtol=1e-13,
    )
    decay = np.exp(-b * t)
    np.testing.assert_allclose(
        InflectionS.from_params([a, b, psi]).remaining_faults(t),
        a * (1 + psi) * decay / (1 + psi * decay),
        rtol=1e-13,
    )
    # a falling Cox-Lewis intensity levels off at exp(alpha) / -beta
    np.testing.assert_allclose(
        CoxLewis.from_params([1.0, -0.1]).remaining_faults(t),
        np.exp(1.0 - 0.1 * t) / 0.1,
        rtol=1e-13,
    )
    assert GoelOkumoto.from_params([a, b]).remaining_faults(5.0).shape == ()


@pytest.mark.parametrize(
    "model",
    [
        MusaOkumoto.from_params([2.0, 0.05]),
        CrowAMSAA.from_params([10.0, 0.6]),
        Duane.from_params([0.6, 2.0]),
        HPP.from_params([0.3]),
        CoxLewis.from_params([0.0, 0.1]),
    ],
)
def test_remaining_faults_refused_without_a_total(model):
    for call in (model.remaining_faults, model.remaining_faults_cb):
        with pytest.raises(ValueError) as error:
            call(10.0)
        message = str(error.value)
        assert "needs a model with a finite number of faults" in message
        assert model.dist.name in message
        assert "expects failures without limit" in message
        assert "GoelOkumoto, YamadaDelayedS or InflectionS" in message
        assert "cif(t2) - cif(t1)" in message


def test_remaining_faults_cb_is_the_delta_method_on_the_log(fitted):
    a, b = fitted.params
    t = np.array([4.0, 8.0, 12.0])
    left = a * np.exp(-b * t)
    np.testing.assert_allclose(fitted.remaining_faults(t), left, rtol=1e-12)
    cov = fitted.covariance()
    # the gradient of log(a exp(-b t)) is (1 / a, -t)
    grad = np.column_stack([np.full(3, 1 / a), -t])
    se_log = np.sqrt(np.einsum("ij,jk,ik->i", grad, cov, grad))
    z = norm.ppf(0.975)
    expected = np.column_stack(
        [left * np.exp(-z * se_log), left * np.exp(z * se_log)]
    )
    np.testing.assert_allclose(
        fitted.remaining_faults_cb(t), expected, rtol=1e-6
    )
    one = norm.ppf(0.9)
    np.testing.assert_allclose(
        fitted.remaining_faults_cb(t, alpha_ci=0.1, bound="upper"),
        left * np.exp(one * se_log),
        rtol=1e-6,
    )
    assert fitted.remaining_faults_cb(8.0).shape == (2,)
    with pytest.raises(ValueError, match="alpha_ci"):
        fitted.remaining_faults_cb(8.0, alpha_ci=2)
    with pytest.raises(ValueError, match="bound"):
        fitted.remaining_faults_cb(8.0, bound="both")


def test_bounds_need_a_fit():
    model = GoelOkumoto.from_params([100.0, 0.05])
    for call in (
        lambda: model.remaining_faults_cb(10.0),
        lambda: model.next_failure_sf_cb(1.0, age=[10.0]),
        lambda: model.time_to_intensity_cb(0.5),
        lambda: model.time_to_reliability_cb(0.9, 1.0),
    ):
        with pytest.raises(ValueError):
            call()


# -- reliability over a mission ----------------------------------------------
def test_next_failure_sf_is_the_mission_reliability():
    model = GoelOkumoto.from_params([100.0, 0.05])
    t, x = np.array([10.0, 60.0]), np.array([1.0, 5.0, 20.0])
    gained = 100.0 * np.exp(-0.05 * t[:, None]) * -np.expm1(-0.05 * x)
    np.testing.assert_allclose(
        model.next_failure_sf(x, age=t), np.exp(-gained), rtol=1e-13
    )


def test_next_failure_sf_cb(fitted):
    x = np.array([0.5, 1.0, 4.0])
    ages = [8.0, 10.0]
    point = fitted.next_failure_sf(x, age=ages)
    cb = fitted.next_failure_sf_cb(x, age=ages)
    assert cb.shape == (2, 3, 2)
    assert np.all(cb[..., 0] < point) and np.all(point < cb[..., 1])
    assert np.all((cb > 0) & (cb < 1))
    # the delta method on the log of the failures expected in the mission
    a, b = fitted.params
    cov = fitted.covariance()
    z = norm.ppf(0.975)
    for k, t in enumerate(ages):
        gained = a * np.exp(-b * t) * -np.expm1(-b * x)
        grad = np.column_stack(
            [
                np.full(3, 1 / a),
                -t + x * np.exp(-b * x) / -np.expm1(-b * x),
            ]
        )
        se_log = np.sqrt(np.einsum("ij,jk,ik->i", grad, cov, grad))
        np.testing.assert_allclose(
            cb[k, :, 0], np.exp(-gained * np.exp(z * se_log)), rtol=1e-6
        )
        np.testing.assert_allclose(
            cb[k, :, 1], np.exp(-gained * np.exp(-z * se_log)), rtol=1e-6
        )
    # the shapes of next_failure_sf, and the one-sided bounds
    assert fitted.next_failure_sf_cb(0.5, age=ages).shape == (2, 2)
    lower = fitted.next_failure_sf_cb(
        x, age=ages, alpha_ci=0.025, bound="lower"
    )
    np.testing.assert_allclose(lower, cb[..., 0], rtol=1e-12)
    assert fitted.next_failure_sf_cb(0.5, age=ages, bound="upper").shape == (
        2,
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        fitted.next_failure_sf_cb(-1.0, age=ages)


def test_next_failure_sf_cb_defaults_to_the_end_of_the_test():
    d = GoelOkumoto.from_params([60.0, 0.04]).time_terminated_simulation_data(
        80.0, items=3, random_state=5
    )
    model = GoelOkumoto.fit(d.x, d.i, d.c)
    np.testing.assert_array_equal(
        model.next_failure_sf_cb(5.0),
        model.next_failure_sf_cb(5.0, age=[80.0, 80.0, 80.0]),
    )


# -- testing to a target ------------------------------------------------------
def test_time_to_intensity_closed_forms():
    a, b = 100.0, 0.05
    assert GoelOkumoto.from_params([a, b]).time_to_intensity(
        0.5
    ) == pytest.approx(np.log(a * b / 0.5) / b, rel=1e-10)
    lambda0, theta = 2.0, 0.05
    assert MusaOkumoto.from_params([lambda0, theta]).time_to_intensity(
        0.05
    ) == pytest.approx((lambda0 / 0.05 - 1) / (lambda0 * theta), rel=1e-10)
    alpha, beta = 10.0, 0.6
    assert CrowAMSAA.from_params([alpha, beta]).time_to_intensity(
        0.05
    ) == pytest.approx(
        (0.05 * alpha**beta / beta) ** (1 / (beta - 1)), rel=1e-10
    )
    assert CoxLewis.from_params([0.0, -0.1]).time_to_intensity(
        0.05
    ) == pytest.approx(np.log(0.05) / -0.1, rel=1e-10)


@pytest.mark.parametrize(
    "model, peak",
    [
        (YamadaDelayedS.from_params([100.0, 0.1]), 10.0),
        (InflectionS.from_params([100.0, 0.1, 4.0]), np.log(4.0) / 0.1),
    ],
)
def test_time_to_intensity_is_on_the_way_down(model, peak):
    # An S-shaped intensity passes a level twice: the answer is the
    # second time, after the peak.
    target = 0.5 * float(model.iif(peak))
    t = model.time_to_intensity(target)
    assert t > peak
    assert float(model.iif(t)) == pytest.approx(target, rel=1e-10)
    assert np.all(model.iif(t * np.array([1.01, 2.0, 10.0])) < target)
    # a level above the peak is never exceeded
    assert model.time_to_intensity(2.0 * float(model.iif(peak))) == 0.0


def test_time_to_reliability_closed_forms():
    a, b, x, target = 100.0, 0.05, 10.0, 0.9
    model = GoelOkumoto.from_params([a, b])
    t = model.time_to_reliability(target, x)
    assert t == pytest.approx(
        np.log(a * -np.expm1(-b * x) / -np.log(target)) / b, rel=1e-10
    )
    assert model.next_failure_sf(x, age=[t])[0] == pytest.approx(target)
    lambda0, theta = 2.0, 0.05
    k = lambda0 * theta
    musa = MusaOkumoto.from_params([lambda0, theta])
    assert musa.time_to_reliability(target, 1.0) == pytest.approx(
        (k * 1.0 / np.expm1(-theta * np.log(target)) - 1) / k, rel=1e-10
    )
    # met from the start
    assert model.time_to_reliability(1e-30, 1.0) == 0.0


@pytest.mark.parametrize(
    "model, how",
    [
        (HPP.from_params([0.3]), "does not come down that far"),
        (CrowAMSAA.from_params([10.0, 1.5]), "is rising"),
        (CoxLewis.from_params([0.0, 0.1]), "does not come down that far"),
    ],
)
def test_a_target_never_reached_is_refused(model, how):
    with pytest.raises(ValueError) as error:
        model.time_to_intensity(0.05)
    message = str(error.value)
    assert f"The fitted {model.dist.name} model never reaches" in message
    assert "an intensity = 0.05" in message and how in message
    assert "Testing longer will not get there" in message
    assert "trend_test()" in message
    with pytest.raises(ValueError, match="a reliability over a mission of 1"):
        model.time_to_reliability(0.9, 1.0)
    # a constant rate below the target has met it from the start
    assert HPP.from_params([0.3]).time_to_intensity(0.5) == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, "fast", None])
def test_targets_are_checked(bad):
    model = GoelOkumoto.from_params([100.0, 0.05])
    with pytest.raises(ValueError, match="target must be a number above 0"):
        model.time_to_intensity(bad)
    with pytest.raises(ValueError, match="target must be a number above 0"):
        model.time_to_reliability(bad, 1.0)
    with pytest.raises(ValueError, match="mission's length"):
        model.time_to_reliability(0.9, bad)


def test_reliability_target_is_below_one():
    model = GoelOkumoto.from_params([100.0, 0.05])
    with pytest.raises(ValueError, match="above 0 and below 1"):
        model.time_to_reliability(1.0, 1.0)


def test_time_bounds_invert_the_curve_bounds(fitted):
    target = 0.5
    t = fitted.time_to_intensity(target)
    lower, upper = fitted.time_to_intensity_cb(target)
    assert lower < t < upper
    # at each end, the intensity's bound of that side is the target
    assert fitted.iif_cb(lower, 0.025, "lower") == pytest.approx(
        target, rel=1e-8
    )
    assert fitted.iif_cb(upper, 0.025, "upper") == pytest.approx(
        target, rel=1e-8
    )
    assert fitted.time_to_intensity_cb(target, 0.025, "upper") == upper
    assert fitted.time_to_intensity_cb(target, 0.025, "lower") == lower
    wide = fitted.time_to_intensity_cb(target, alpha_ci=0.01)
    assert wide[0] < lower and wide[1] > upper

    t = fitted.time_to_reliability(0.9, 0.5)
    lower, upper = fitted.time_to_reliability_cb(0.9, 0.5)
    assert lower < t < upper
    # the reliability's lower bound reaches the target at the upper time
    assert fitted.next_failure_sf_cb(
        0.5, age=[upper], alpha_ci=0.025, bound="lower"
    )[0] == pytest.approx(0.9, rel=1e-8)
    assert fitted.next_failure_sf_cb(
        0.5, age=[lower], alpha_ci=0.025, bound="upper"
    )[0] == pytest.approx(0.9, rel=1e-8)


def test_an_upper_time_bound_can_be_infinite(fitted):
    # The upper bound of a Goel-Okumoto intensity is a b exp(-b t + z
    # se(t)), and far out se(t) is t se(b): it comes down only where b
    # is more than z of its standard errors above 0. At a level past
    # that, the data do not rule out that the target is never reached.
    b, se_b = fitted.params[1], fitted.standard_errors()[1]
    ratio = b / se_b
    falls = 2 * norm.sf(0.8 * ratio)
    never = 2 * norm.sf(1.2 * ratio)
    assert np.isfinite(fitted.time_to_intensity_cb(0.5, alpha_ci=falls)[1])
    lower, upper = fitted.time_to_intensity_cb(0.5, alpha_ci=never)
    assert np.isfinite(lower) and upper == np.inf
    assert fitted.time_to_intensity_cb(0.5, never / 2, "upper") == np.inf
    assert fitted.time_to_reliability_cb(0.9, 0.5, alpha_ci=never)[1] == np.inf


def test_last_time_above():
    def falling(t):
        return 8.0 * np.exp(-0.5 * np.asarray(t))

    assert last_time_above(falling, 1.0, 1.0) == pytest.approx(
        2 * np.log(8.0), rel=1e-12
    )
    # whatever the scale it is given
    for scale in (1e-3, 1e3):
        assert last_time_above(falling, 1.0, scale) == pytest.approx(
            2 * np.log(8.0), rel=1e-10
        )
    assert last_time_above(falling, 10.0, 1.0) == 0.0
    with pytest.raises(NeverReached, match="still above"):
        last_time_above(lambda t: 2.0 + 0 * np.asarray(t), 1.0, 1.0)
    with pytest.raises(NeverReached, match="rising"):
        last_time_above(lambda t: np.asarray(t) ** 0.5, 1.0, 1.0)
    # below the level now but rising
    with pytest.raises(NeverReached, match="rising"):
        last_time_above(lambda t: 1e-30 * np.asarray(t), 1.0, 1.0)


def test_answers_do_not_depend_on_the_unit_of_time(fitted):
    scaled = GoelOkumoto.fit(WEEKS * 168.0, c=[2] * 8, n=COUNTS)
    assert scaled.time_to_intensity(0.5 / 168.0) == pytest.approx(
        168.0 * fitted.time_to_intensity(0.5), rel=1e-4
    )
    np.testing.assert_allclose(
        scaled.time_to_reliability_cb(0.9, 84.0),
        168.0 * fitted.time_to_reliability_cb(0.9, 0.5),
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        scaled.remaining_faults_cb(8.0 * 168.0),
        fitted.remaining_faults_cb(8.0),
        rtol=1e-3,
    )


def test_expected_failures_far_out_keep_their_digits(fitted):
    # At b t = 60 the two counts of a mission are equal in doubles; the
    # log of their difference is then summed from the intensity, and is
    # the closed form log(a) - b t + log(1 - exp(-b x)) all the same.
    a, b = fitted.params
    start = np.array([1.0, 30.0, 60.0, 500.0]) / b
    for mission in (0.01 / b, 1.0 / b, 300.0 / b):
        x = np.full(4, mission)
        exact = np.log(a) - b * start + np.log(-np.expm1(-b * x))
        np.testing.assert_allclose(
            fitted._log_gained(start, x)(fitted.params), exact, rtol=1e-9
        )
