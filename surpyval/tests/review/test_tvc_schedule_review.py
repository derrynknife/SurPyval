"""Targeted review of ``univariate/regression/tvc_schedule.py`` (#399).

Each test pins a bug found by reading the module and its consumers
(``sf_tvc``/``Hf_tvc`` and the degradation stress clock) adversarially. They
were strict expected failures until the bug was fixed; #433, #434 and
#435 are fixed and run as ordinary regression tests.
"""

import numpy as np
import pytest

import surpyval as surv
from surpyval import StepSchedule, StepValuedError
from surpyval.degradation import DegradationAnalysis


def _ph_data():
    rng = np.random.default_rng(0)
    Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
    x = 100 * rng.weibull(2, 200) * np.exp(-0.25 * Z[:, 0])
    return x, Z


def test_aft_sf_tvc_ignores_the_path_before_time_zero():
    # #433: a constant path from -10 gave sf(20) = 0.8626, not
    # sf(20, Z) = 0.9362 (ten units of age before 0 were counted).
    x, Z = _ph_data()
    model = surv.WeibullAFT.fit(x, Z)
    q = np.array([20.0, 50.0, 100.0])
    early = StepSchedule.from_changepoints([-10.0], [1.0])
    np.testing.assert_allclose(
        model.sf_tvc(q, early), model.sf(q, [1.0]), rtol=1e-6
    )


def _step_stress_model():
    rng = np.random.default_rng(3)
    z_levels = 1 / np.array([323.0, 348.0, 373.0])
    times = np.arange(10.0, 300.0 + 1e-9, 10.0)
    z = np.select([times <= 100, times <= 200], z_levels[:2], z_levels[2])
    af = np.exp(-5000.0 * (z - z_levels[0]))
    xs, ys, ids, Zs = [], [], [], []
    for unit in range(8):
        a, b = rng.normal([1.0, 0.02], [0.2, 0.003])
        tau = np.cumsum(10.0 * af)
        xs.append(times)
        ys.append(a + b * tau + rng.normal(0, 0.3, times.size))
        ids.append(np.full(times.size, unit))
        Zs.append(z)
    xs_, ys_, ids_, Zs_ = (np.concatenate(v) for v in (xs, ys, ids, Zs))
    model = DegradationAnalysis.fit(
        xs_,
        ys_,
        ids_,
        threshold=15.0,
        Z=Zs_,
        acceleration="clock",
        stress_ref=[z_levels[0]],
    )
    return model, z_levels[2]


def test_stress_clock_ignores_the_path_before_time_zero():
    # #433: the clock started at the schedule's first edge; a constant path
    # from -10 gave F(100) = 0.662, not F(100, Z) = 0.489.
    model, z = _step_stress_model()
    t = np.array([50.0, 100.0, 150.0])
    early = StepSchedule.from_changepoints([-10.0], [z])
    np.testing.assert_allclose(
        model.ff(t, Z=early), model.ff(t, Z=[z]), rtol=1e-6
    )


def test_expression_boolean_operators_return_an_operand():
    # #434: 'a and b' / 'a or b' gave a bool, so this was 1.0 everywhere.
    expr = "(t > 50) and 2.0 or 1.0"
    schedule = StepSchedule.from_expression(expr, 100)
    starts, _, Z = schedule.segments(100.0)
    got = Z[np.searchsorted(starts, [10.0, 60.0], side="right") - 1, 0]
    np.testing.assert_array_equal(got, [1.0, 2.0])


@pytest.mark.parametrize(
    "expr", ["round(t / 10, ndigits=1)", "round(t / 10, 1)"], ids=str
)
def test_expression_keyword_arguments_are_used(expr):
    # #434: the keyword was dropped (Z = 0 at t = 1, 2); the positional
    # form raised a TypeError (every constant is read as a float).
    schedule = StepSchedule.from_expression(expr, 30)
    starts, _, Z = schedule.segments(30.0)
    got = Z[np.searchsorted(starts, [0.0, 1.0, 2.0], side="right") - 1, 0]
    np.testing.assert_allclose(got, [0.0, 0.1, 0.2])


@pytest.mark.parametrize(
    "expr, name",
    [
        ("floor(t, digits=1)", "digits"),
        ("max(1, floor(t), default=3)", "default"),
        ("round(t / 10, ndigits=0.5)", "ndigits"),
    ],
    ids=str,
)
def test_expression_keyword_it_cannot_honour_is_refused(expr, name):
    # #434: an unusable keyword was dropped silently.
    with pytest.raises(ValueError, match=name):
        StepSchedule.from_expression(expr, 10)


def test_expression_operand_returned_by_or_must_be_stepped():
    # #434: with 'or' returning its operand, '(t > 5) or t' is t itself
    # after the first second, so it is not step-valued.
    with pytest.raises(StepValuedError):
        StepSchedule.from_expression("(t > 5) or t", 10)
    # 'and' returns an earlier operand only when it is 0.
    schedule = StepSchedule.from_expression("t and 1", 10)
    np.testing.assert_array_equal(schedule.Z[:, 0], [0.0, 1.0])


def test_parametric_sf_tvc_missing_given_is_nan():
    # #435 item 1: it returned the unconditional survival (0.936 at 20).
    x, Z = _ph_data()
    model = surv.WeibullPH.fit(x, Z)
    schedule = StepSchedule.constant([1.0])
    assert np.isnan(model.sf_tvc([20.0, 50.0], schedule, given=np.nan)).all()


@pytest.mark.parametrize(
    "fitter", ["WeibullPH", "WeibullAH", "WeibullAFT"], ids=str
)
def test_sf_tvc_2d_query_keeps_its_shape(fitter):
    # #435 item 2: a broadcasting ValueError for a 2-D query.
    x, Z = _ph_data()
    model = getattr(surv, fitter).fit(x, Z)
    schedule = StepSchedule.constant([1.0])
    grid = np.array([[20.0, 50.0], [60.0, 70.0]])
    got = model.sf_tvc(grid, schedule)
    assert got.shape == grid.shape
    np.testing.assert_allclose(
        got.ravel(), model.sf_tvc(grid.ravel(), schedule)
    )


def test_sf_tvc_scalar_query_gives_a_scalar():
    # #435 item 3: shape (1,) where WeibullPH.sf(20.0, Z) is a scalar.
    x, Z = _ph_data()
    model = surv.WeibullPH.fit(x, Z)
    schedule = StepSchedule.constant([1.0])
    assert np.shape(model.sf(20.0, [1.0])) == ()
    assert np.shape(model.sf_tvc(20.0, schedule)) == ()


@pytest.mark.parametrize("fitter", ["WeibullPH", "CoxPH"], ids=str)
def test_sf_tvc_at_time_zero_is_one(fitter):
    # #435 item 4: 'x must contain a positive time' where sf(0, Z) = 1.
    x, Z = _ph_data()
    model = getattr(surv, fitter).fit(x, Z)
    schedule = StepSchedule.constant([1.0])
    np.testing.assert_allclose(model.sf_tvc([0.0], schedule), [1.0])
    assert model.sf_tvc(0.0, schedule) == 1.0
    np.testing.assert_allclose(model.sf_tvc([-5.0, 0.0], schedule), 1.0)


def test_aft_sf_tvc_before_zero_is_sf_for_a_baseline_below_zero():
    # #435 item 4: with a Normal baseline, sf_tvc(-5) was the survival at
    # 0 (0.9725), not sf(-5, Z) = 0.9801.
    x, Z = _ph_data()
    model = surv.NormalAFT.fit(x, Z)
    schedule = StepSchedule.constant([1.0])
    q = np.array([-5.0, 0.0, 20.0])
    np.testing.assert_allclose(
        model.sf_tvc(q, schedule), model.sf(q, [1.0]), rtol=1e-12
    )


def test_stress_clock_starts_at_zero():
    # #433: the clock is 0 at time 0 whatever the schedule's first edge.
    from surpyval.degradation._clock import StressClock

    for first in (-10.0, 0.0, 5.0):
        schedule = StepSchedule.from_changepoints([first, 20.0], [1.0, 2.0])
        clock = StressClock(lambda z: float(z[0]), 1, schedule)
        np.testing.assert_allclose(
            clock.tau(np.array([0.0, 10.0, 30.0])), [0.0, 10.0, 40.0]
        )
