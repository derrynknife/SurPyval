r"""
Time-varying-covariate *evaluation*: ``StepSchedule`` and ``sf_tvc`` (#170).

Where ``fit_tvc`` (#150) reshapes start-stop data to *fit* a proportional- or
additive-hazards model, ``sf_tvc`` evaluates an *already-fitted* model along a
piecewise-constant covariate path ``Z(t)``. The covariate path is described by
a :class:`StepSchedule`, built either structurally (change-points, intervals, a
cyclic pattern) or from a step-valued expression string whose step-ness is
proved statically before it is evaluated.

The evaluation is exact for these families because the cumulative hazard is
additive over disjoint segments, so along a step path it is the sum of the
per-segment increments -- and therefore collapses to the ordinary ``sf`` when
the covariate is constant.
"""

import numpy as np
import pytest

from surpyval import WeibullAFT, WeibullAH, WeibullPH, WeibullPO
from surpyval.univariate.regression import (
    CoxPH,
    StepSchedule,
    StepValuedError,
)

# -- StepSchedule (structural) --------------------------------------------


def test_constant_schedule_is_single_segment():
    s = StepSchedule.constant([0.5])
    starts, ends, Z = s.segments(10.0)
    assert starts.tolist() == [0.0]
    assert ends.tolist() == [10.0]
    assert Z.tolist() == [[0.5]]
    assert s.p == 1


def test_from_changepoints_segments():
    s = StepSchedule.from_changepoints([0, 500, 800], [0.3, 0.9, 0.3])
    starts, ends, Z = s.segments(1000.0)
    assert starts.tolist() == [0.0, 500.0, 800.0]
    assert ends.tolist() == [500.0, 800.0, 1000.0]
    assert Z.ravel().tolist() == [0.3, 0.9, 0.3]


def test_from_intervals_matches_changepoints():
    a = StepSchedule.from_intervals([0, 4], [4, 10], [[0.2], [0.9]])
    b = StepSchedule.from_changepoints([0, 4], [[0.2], [0.9]])
    for sa, sb in zip(a.segments(8.0), b.segments(8.0)):
        assert np.allclose(sa, sb)


def test_from_intervals_rejects_gaps():
    with pytest.raises(ValueError, match="contiguous"):
        StepSchedule.from_intervals([0, 5], [4, 10], [[0.2], [0.9]])


def test_cyclic_duty_cycle_repeats_to_horizon():
    # 8 "on" (0.9) then 16 "off" (0.3), period 24.
    s = StepSchedule.cyclic([0, 8], [[0.9], [0.3]], 24)
    starts, ends, Z = s.segments(50.0)
    assert starts.tolist() == [0.0, 8.0, 24.0, 32.0, 48.0]
    assert ends.tolist() == [8.0, 24.0, 32.0, 48.0, 50.0]
    assert Z.ravel().tolist() == [0.9, 0.3, 0.9, 0.3, 0.9]


def test_multivariate_schedule():
    s = StepSchedule.from_changepoints([0, 5], [[0.1, 1.0], [0.2, 2.0]])
    assert s.p == 2
    starts, ends, Z = s.segments(10.0)
    assert Z.tolist() == [[0.1, 1.0], [0.2, 2.0]]


def test_schedule_validation():
    with pytest.raises(ValueError, match="strictly increasing"):
        StepSchedule(np.array([0.0, 0.0, 1.0]), np.array([[1.0], [2.0]]))
    with pytest.raises(ValueError, match="one more entry"):
        StepSchedule(np.array([0.0, 1.0]), np.array([[1.0], [2.0]]))


# -- StepSchedule (expression + step guard) -------------------------------


def test_expression_duty_cycle_matches_cyclic():
    expr = StepSchedule.from_expression(
        "0.9 if t % 24 < 8 else 0.3", horizon=50, resolution=1.0
    )
    cyc = StepSchedule.cyclic([0, 8], [[0.9], [0.3]], 24)
    se, _, ze = expr.segments(50.0)
    sc, _, zc = cyc.segments(50.0)
    assert np.allclose(se, sc)
    assert np.allclose(ze.ravel(), zc.ravel())


def test_expression_stepped_doubling():
    s = StepSchedule.from_expression(
        "0.3 * 2 ** floor(t / 1000)", horizon=3500, resolution=1.0
    )
    starts, ends, Z = s.segments(3500.0)
    assert starts.tolist() == [0.0, 1000.0, 2000.0, 3000.0]
    assert np.allclose(Z.ravel(), [0.3, 0.6, 1.2, 2.4])


def test_expression_multivariate():
    s = StepSchedule.from_expression(
        ["0.3 if t < 10 else 0.9", "1.0"], horizon=20, resolution=1.0
    )
    assert s.p == 2
    _, _, Z = s.segments(20.0)
    # second covariate constant at 1.0 throughout
    assert np.allclose(Z[:, 1], 1.0)


@pytest.mark.parametrize(
    "bad",
    [
        "0.3 + 1e-4 * t",  # continuous ramp
        "0.3 * 2 ** (t / 1000)",  # smooth doubling
        "sin(t)",  # unknown / continuous fn
        "t",  # bare t
    ],
)
def test_expression_rejects_continuous(bad):
    with pytest.raises(StepValuedError):
        StepSchedule.from_expression(bad, horizon=100, resolution=1.0)


def test_expression_accepts_constant_and_phase():
    # both are step-valued and must be accepted
    StepSchedule.from_expression("0.3", horizon=10)
    StepSchedule.from_expression("0.3 if t < 500 else 0.9", horizon=1000)


# -- sf_tvc on fitted PH / AH models --------------------------------------


def _fit(F, seed=0, n=300):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 1))
    x = np.abs(rng.weibull(1.6, n) * 10 * np.exp(-0.4 * Z[:, 0])) + 0.5
    c = np.zeros(n)
    return F.fit(x=x, Z=Z, c=c)


@pytest.mark.parametrize("F", [WeibullPH, WeibullAH, WeibullAFT])
def test_constant_schedule_reduces_to_sf(F):
    m = _fit(F)
    z = 0.7
    x = np.array([2.0, 5.0, 9.0])
    a = m.sf_tvc(x, StepSchedule.constant([z]))
    b = np.asarray(m.sf(x, np.array([[z]])), dtype=float).ravel()
    assert np.allclose(a, b)


@pytest.mark.parametrize("F", [WeibullPH, WeibullAH, WeibullAFT])
def test_array_form_matches_schedule(F):
    m = _fit(F)
    xl = np.array([0.0, 4.0])
    Z = np.array([[0.2], [0.9]])
    a = m.sf_tvc([3.0, 6.0], Z, xl=xl)
    b = m.sf_tvc([3.0, 6.0], StepSchedule.from_changepoints(xl, Z))
    assert np.allclose(a, b)


@pytest.mark.parametrize("F", [WeibullPH, WeibullAH])
def test_hf_tvc_equals_manual_telescoping(F):
    m = _fit(F)
    xl = np.array([0.0, 4.0])
    Z = np.array([[0.2], [0.9]])
    # H(6) = [Hf(4, z0) - 0] + [Hf(6, z1) - Hf(4, z1)]
    Hf = lambda t, z: float(  # noqa: E731
        np.asarray(
            m.model.Hf(np.array([t]), np.array([z]), *m.params)
        ).ravel()[0]
    )
    manual = (Hf(4.0, [0.2]) - 0.0) + (Hf(6.0, [0.9]) - Hf(4.0, [0.9]))
    got = m.Hf_tvc([6.0], Z, xl=xl)[0]
    assert np.isclose(got, manual)


@pytest.mark.parametrize("F", [WeibullPH, WeibullAH, WeibullAFT])
def test_conditional_survival(F):
    m = _fit(F)
    sched = StepSchedule.from_changepoints([0.0, 4.0], [[0.2], [0.9]])
    cond = m.sf_tvc([6.0], sched, given=3.0)
    manual = m.sf_tvc([6.0], sched) / m.sf_tvc([3.0], sched)
    assert np.allclose(cond, manual)


def test_aft_hf_tvc_equals_accumulated_accelerated_age():
    # AFT rescales time: each segment contributes phi(z) * width of accelerated
    # age, then the baseline cumulative hazard is evaluated once at the total.
    m = _fit(WeibullAFT)
    xl = np.array([0.0, 4.0])
    Z = np.array([[0.2], [0.9]])
    beta = np.asarray(m.params[m.k_dist :])
    dist_params = m.params[: m.k_dist]
    psi = np.exp(0.2 * beta[0]) * 4.0 + np.exp(0.9 * beta[0]) * (6.0 - 4.0)
    manual = float(
        np.asarray(
            m.model.Hf_dist(np.array([psi]), *dist_params), dtype=float
        ).ravel()[0]
    )
    got = m.Hf_tvc([6.0], Z, xl=xl)[0]
    assert np.isclose(got, manual)


def test_sf_tvc_survival_decreasing_and_bounded():
    m = _fit(WeibullPH)
    sched = StepSchedule.cyclic([0, 8], [[0.0], [1.5]], 24)
    x = np.linspace(1, 100, 50)
    s = m.sf_tvc(x, sched)
    assert np.all(s <= 1.0) and np.all(s >= 0.0)
    assert np.all(np.diff(s) <= 1e-12)  # monotone non-increasing


def test_po_rejects_tvc_evaluation():
    m = _fit(WeibullPO)
    with pytest.raises(NotImplementedError, match="proportional-odds"):
        m.sf_tvc([2.0], StepSchedule.constant([0.5]))


def test_schedule_covariate_count_checked():
    m = _fit(WeibullPH)  # one covariate
    two = StepSchedule.constant([0.5, 0.5])
    with pytest.raises(ValueError, match="covariate"):
        m.sf_tvc([2.0], two)


def test_end_to_end_fit_tvc_then_sf_tvc():
    # Fit on time-varying start-stop data, then evaluate survival along a path.
    rng = np.random.default_rng(7)
    n = 200
    i, xl, xr, c, Z = [], [], [], [], []
    for s in range(n):
        split = 3.0
        end = split + np.abs(rng.weibull(1.4)) * 6.0 + 0.5
        z0 = rng.normal(0, 1)
        i += [s, s]
        xl += [0.0, split]
        xr += [split, end]
        c += [1, 0]
        Z += [[z0], [z0 + 0.3]]
    model = WeibullPH.fit_tvc(
        np.array(i), np.array(xl), np.array(xr), np.array(c), np.array(Z)
    )
    sched = StepSchedule.from_changepoints([0.0, 3.0], [[0.0], [0.3]])
    s = model.sf_tvc([2.0, 5.0, 8.0], sched)
    assert s.shape == (3,)
    assert np.all((s >= 0) & (s <= 1)) and s[0] >= s[-1]


# -- sf_tvc on the semi-parametric Cox model ------------------------------
#
# Cox shares the sf_tvc / StepSchedule convention with the parametric
# families; sf_tvc is the schedule-oriented counterpart of the existing
# interval-oriented predict_tvc, and the two must agree.


def _fit_cox_tvc(seed=3, n=40):
    rng = np.random.default_rng(seed)
    ident, xl, xr, c, Z = [], [], [], [], []
    for s in range(n):
        split = 3.0
        end = split + np.abs(rng.weibull(1.4)) * 6.0 + 0.5
        z0 = rng.normal(0, 1)
        ident += [s, s]
        xl += [0.0, split]
        xr += [split, end]
        c += [1, 0]
        Z += [[z0], [z0 + 0.2]]
    return CoxPH.fit_tvc(
        np.array(ident), np.array(xl), np.array(xr), np.array(c), np.array(Z)
    )


def test_cox_sf_tvc_matches_predict_tvc():
    m = _fit_cox_tvc()
    xl = np.array([0.0, 3.0])
    xr = np.array([3.0, 8.0])
    Z = np.array([[0.5], [0.7]])
    times = np.array([2.0, 5.0, 7.5])
    _, sf_pred, _ = m.predict_tvc(xl, xr, Z, times=times)
    sf_new = m.sf_tvc(times, StepSchedule.from_intervals(xl, xr, Z))
    assert np.allclose(sf_new, sf_pred)


def test_cox_array_form_matches_schedule():
    m = _fit_cox_tvc()
    times = np.array([2.0, 5.0, 7.5])
    a = m.sf_tvc(times, np.array([[0.5], [0.7]]), xl=np.array([0.0, 3.0]))
    b = m.sf_tvc(times, StepSchedule.from_changepoints([0.0, 3.0], [0.5, 0.7]))
    assert np.allclose(a, b)


def test_cox_constant_reduces_to_sf_above_first_event():
    # Cox's sf clamps the left tail (via _get_idx) to the first jump, so the
    # constant reduction holds at/above the first event time, where the
    # Breslow baseline is well defined.
    m = _fit_cox_tvc()
    z = 0.6
    t = m.x[m.x > 0][:5]
    a = m.sf_tvc(t, StepSchedule.constant([z]))
    b = np.asarray(m.sf(t, np.array([z])), dtype=float).ravel()
    assert np.allclose(a, b)


def test_cox_conditional_survival():
    m = _fit_cox_tvc()
    sched = StepSchedule.from_intervals([0.0, 3.0], [3.0, 8.0], [[0.5], [0.7]])
    cond = m.sf_tvc([7.5], sched, given=2.0)
    manual = m.sf_tvc([7.5], sched) / m.sf_tvc([2.0], sched)
    assert np.allclose(cond, manual)


def test_cox_expression_schedule():
    m = _fit_cox_tvc()
    es = StepSchedule.from_expression(
        "0.9 if t < 3 else 0.3", horizon=8, resolution=0.5
    )
    s = m.sf_tvc([2.0, 6.0], es)
    assert np.all((s >= 0) & (s <= 1)) and s[0] >= s[1]


def test_cox_covariate_count_checked():
    m = _fit_cox_tvc()  # one covariate
    with pytest.raises(ValueError, match="covariate"):
        m.sf_tvc([2.0], StepSchedule.constant([0.5, 0.5]))


def test_cox_stratified_rejects_sf_tvc():
    rng = np.random.default_rng(0)
    n = 120
    Z = rng.normal(0, 1, (n, 1))
    x = np.abs(rng.weibull(1.5, n) * 10) + 0.5
    c = np.zeros(n)
    strata = (Z[:, 0] > 0).astype(int)
    m = CoxPH.fit(x=x, Z=Z, c=c, strata=strata)
    with pytest.raises(NotImplementedError, match="stratified"):
        m.sf_tvc([2.0], StepSchedule.constant([0.5]))
