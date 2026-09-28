"""Targeted review of ``univariate/regression/tvc_schedule.py`` (#399).

Each test pins a bug found by reading the module and its consumers
(``sf_tvc``/``Hf_tvc`` and the degradation stress clock) adversarially. They
are strict expected failures until the bug is fixed.
"""

import numpy as np
import pytest

import surpyval as surv
from surpyval import StepSchedule
from surpyval.degradation import DegradationAnalysis


def _ph_data():
    rng = np.random.default_rng(0)
    Z = rng.binomial(1, 0.5, (200, 1)).astype(float)
    x = 100 * rng.weibull(2, 200) * np.exp(-0.25 * Z[:, 0])
    return x, Z


@pytest.mark.xfail(
    strict=True,
    reason="#433: AFT sf_tvc counts a schedule's time before 0 as age; a "
    "constant path from -10 gives sf(20) = 0.8626, not sf(20, Z) = 0.9362",
)
def test_aft_sf_tvc_ignores_the_path_before_time_zero():
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


@pytest.mark.xfail(
    strict=True,
    reason="#433: the degradation stress clock starts at the schedule's "
    "first edge, not 0; a constant path from -10 gives F(100) = 0.662, "
    "not F(100, Z) = 0.489",
)
def test_stress_clock_ignores_the_path_before_time_zero():
    model, z = _step_stress_model()
    t = np.array([50.0, 100.0, 150.0])
    early = StepSchedule.from_changepoints([-10.0], [z])
    np.testing.assert_allclose(
        model.ff(t, Z=early), model.ff(t, Z=[z]), rtol=1e-6
    )


@pytest.mark.xfail(
    strict=True,
    reason="#434: from_expression evaluates 'a and b' / 'a or b' to a bool, "
    "not an operand: '(t > 50) and 2.0 or 1.0' is 1.0 everywhere, "
    "Python gives 2.0 after t = 50",
)
def test_expression_boolean_operators_return_an_operand():
    expr = "(t > 50) and 2.0 or 1.0"
    schedule = StepSchedule.from_expression(expr, 100)
    starts, _, Z = schedule.segments(100.0)
    got = Z[np.searchsorted(starts, [10.0, 60.0], side="right") - 1, 0]
    np.testing.assert_array_equal(got, [1.0, 2.0])


@pytest.mark.xfail(
    strict=True,
    reason="#434: from_expression drops keyword arguments silently: "
    "'round(t / 10, ndigits=1)' is evaluated as round(t / 10), so Z at "
    "t = 1, 2 is 0, not 0.1, 0.2",
)
def test_expression_keyword_arguments_are_used_or_refused():
    expr = "round(t / 10, ndigits=1)"
    try:
        schedule = StepSchedule.from_expression(expr, 30)
    except ValueError:
        return  # refusing the keyword is an acceptable fix
    starts, _, Z = schedule.segments(30.0)
    got = Z[np.searchsorted(starts, [0.0, 1.0, 2.0], side="right") - 1, 0]
    np.testing.assert_allclose(got, [0.0, 0.1, 0.2])


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


@pytest.mark.xfail(
    strict=True,
    reason="#435: sf_tvc(0) raises 'x must contain a positive time' "
    "(PH and Cox) where sf(0, Z) = 1",
)
@pytest.mark.parametrize("fitter", ["WeibullPH", "CoxPH"], ids=str)
def test_sf_tvc_at_time_zero_is_one(fitter):
    x, Z = _ph_data()
    model = getattr(surv, fitter).fit(x, Z)
    schedule = StepSchedule.constant([1.0])
    np.testing.assert_allclose(model.sf_tvc([0.0], schedule), [1.0])
