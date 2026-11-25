"""Reliability growth planning: the idealised curve, the chance a growth
test demonstrates a requirement, PM2-Continuous and PM2-Discrete.

References: MIL-HDBK-189C (14 June 2011), sections 5.2, 5.3, 5.5 and 5.6;
its SPLAN examples 1 and 2 (tables II and III) for the idealised curve and
the demonstration chance. The handbook read its chances from the tables of
Ellner and Mioduski (1992) by double linear interpolation, so they agree
with the exact sum to about 0.03; the exact sum is checked against a
simulation of the test itself.
"""

import json

import numpy as np
import pytest
from scipy import stats

from surpyval.recurrent.parametric.crow_amsaa import (
    crow_time_terminated_coefficients,
)
from surpyval.test_planning import (
    IdealisedGrowthCurve,
    demonstrated_mtbf,
    growth_demonstration_probability,
    growth_test_time,
    idealised_growth_curve,
    pm2_continuous,
    pm2_discrete,
)

# -- the idealised curve ------------------------------------------------------


def test_curve_closed_form():
    curve = idealised_growth_curve(
        68.0, 500.0, growth_rate=0.23, test_time=2800.0
    )
    expected = 68.0 / 0.77 * (2800.0 / 500.0) ** 0.23
    assert curve.final_mtbf == pytest.approx(expected, rel=1e-14)
    assert curve.mtbf(2800.0) == pytest.approx(expected, rel=1e-14)
    assert curve.mtbf(100.0) == 68.0  # the first phase
    # Duane: cumulative MTBF M_I (t / t1)^alpha, instantaneous over 1 - a.
    t = np.array([600.0, 1000.0, 2800.0])
    np.testing.assert_allclose(
        curve.cumulative_mtbf(t), 68.0 * (t / 500.0) ** 0.23, rtol=1e-14
    )
    np.testing.assert_allclose(
        curve.mtbf(t) * 0.77, curve.cumulative_mtbf(t), rtol=1e-14
    )
    # The derivative of the expected failures is 1 / M(t) after t1.
    h = 1e-3
    for ti in t:
        slope = (
            curve.expected_failures(ti + h) - curve.expected_failures(ti - h)
        ) / (2 * h)
        assert 1.0 / slope == pytest.approx(curve.mtbf(ti), rel=1e-7)
    # Continuous expected failures at t1.
    assert curve.expected_failures(500.0) == pytest.approx(500.0 / 68.0)


def test_curve_solves_each_quantity():
    base = idealised_growth_curve(
        50.0, 400.0, growth_rate=0.35, test_time=6000.0
    )
    by_time = idealised_growth_curve(
        50.0, 400.0, growth_rate=0.35, final_mtbf=base.final_mtbf
    )
    by_rate = idealised_growth_curve(
        50.0, 400.0, final_mtbf=base.final_mtbf, test_time=6000.0
    )
    assert by_time.test_time == pytest.approx(6000.0, rel=1e-12)
    assert by_rate.growth_rate == pytest.approx(0.35, rel=1e-12)


def test_handbook_approximate_growth_rate_is_close():
    # MIL-HDBK-189C's closed form for the growth rate (from the second-
    # order expansion of -ln(1 - alpha)) is close to the exact root.
    mi, t1, mf, T = 68.0, 500.0, 130.0, 2800.0
    lt = np.log(T / t1)
    approx = -lt - 1 + np.sqrt((1 + lt) ** 2 + 2 * np.log(mf / mi))
    exact = idealised_growth_curve(
        mi, t1, final_mtbf=mf, test_time=T
    ).growth_rate
    assert exact == pytest.approx(approx, abs=0.01)


def test_phase_mtbf_is_test_time_over_expected_failures():
    curve = idealised_growth_curve(
        68.0, 500.0, growth_rate=0.23, test_time=2800.0
    )
    table = curve.phase_mtbf([500.0, 1200.0, 2000.0, 2800.0])
    assert table["mtbf"].iloc[0] == pytest.approx(68.0)
    n = curve.expected_failures(np.array([0.0, 500.0, 1200.0, 2000.0, 2800]))
    np.testing.assert_allclose(
        table["mtbf"], np.diff([0, 500, 1200, 2000, 2800]) / np.diff(n)
    )
    # The phase average lies between the curve at its two ends.
    for _, row in table.iloc[1:].iterrows():
        assert curve.mtbf(row.start + 1e-9) < row.mtbf < curve.mtbf(row.end)


# MIL-HDBK-189C table II (SPLAN example 1: M_I = 68 over t_I = 500,
# T = 2800, TR = 100 at 80% confidence): growth rate, M(T), E(N), P(A).
TABLE_II = [
    (0.14, 100, 32.6, 0.15),
    (0.20, 120, 29.2, 0.37),
    (0.23, 130, 28.0, 0.48),
    (0.25, 139, 26.9, 0.58),
    (0.30, 163, 24.5, 0.77),
    (0.35, 191, 22.6, 0.90),
    (0.40, 226, 20.6, 0.96),
]


@pytest.mark.parametrize("rate, mtbf, failures, chance", TABLE_II)
def test_splan_example_1(rate, mtbf, failures, chance):
    curve = idealised_growth_curve(
        68.0, 500.0, growth_rate=rate, test_time=2800.0
    )
    # The handbook rounds the growth rate it prints to two places.
    assert curve.final_mtbf == pytest.approx(mtbf, rel=0.012)
    assert curve.expected_failures(2800.0) == pytest.approx(
        failures, rel=0.012
    )
    # The handbook's chances are read from tables by interpolation.
    exact = growth_demonstration_probability(
        mtbf / 100.0, failures, alpha_ci=0.2
    )
    assert exact == pytest.approx(chance, abs=0.035)


def test_splan_example_1_oc_curve_rows():
    curve = idealised_growth_curve(
        68.0, 500.0, growth_rate=0.23, test_time=2800.0
    )
    oc = curve.oc_curve(100.0, [r for r, *_ in TABLE_II], alpha_ci=0.2)
    assert np.all(np.diff(oc["probability"]) > 0)
    assert oc["probability"].iloc[2] == pytest.approx(
        curve.demonstration_probability(100.0, alpha_ci=0.2)
    )


# MIL-HDBK-189C table III (SPLAN example 2: M_I = 48 over 500, growth
# rate 0.3): T, M(T), E(N), P(A) ("<" rows are bounds read from tables).
TABLE_III = [
    (3000, 117.4, 36.5),
    (4000, 128.0, 44.6),
    (5000, 136.8, 52.2),
    (5500, 140.8, 55.8),
    (5400, 140.0, 55.1),
    (5300, 139.2, 54.4),
    (5375, 139.8, 54.9),
]


@pytest.mark.parametrize("T, mtbf, failures", TABLE_III)
def test_splan_example_2_curve(T, mtbf, failures):
    curve = idealised_growth_curve(48.0, 500.0, growth_rate=0.3, test_time=T)
    assert curve.final_mtbf == pytest.approx(mtbf, abs=0.06)
    assert curve.expected_failures(T) == pytest.approx(failures, abs=0.06)


def test_splan_example_2_test_time():
    # The handbook solved T = 5375 by table interpolation (its P(A) 0.800
    # there); the exact chance there is 0.813, and the exact test time
    # for 0.8 is about 5270, 2% shorter.
    curve = growth_test_time(
        48.0, 500.0, 0.3, 100.0, probability=0.8, alpha_ci=0.2
    )
    assert curve.test_time == pytest.approx(5375.0, rel=0.03)
    assert curve.demonstration_probability(
        100.0, alpha_ci=0.2
    ) == pytest.approx(0.8, abs=0.005)
    at_5375 = idealised_growth_curve(
        48.0, 500.0, growth_rate=0.3, test_time=5375.0
    ).demonstration_probability(100.0, alpha_ci=0.2)
    assert at_5375 == pytest.approx(0.80, abs=0.02)


def _simulated_chance(curve, requirement, alpha_ci, runs, seed):
    """The test itself: a power-law process with the curve's shape and
    expected failures, fitted, its Crow lower bound against the
    requirement; given at least one failure."""
    rng = np.random.default_rng(seed)
    T = curve.test_time
    beta = 1 - curve.growth_rate
    mu = float(curve.expected_failures(T))
    passed = seen = 0
    for _ in range(runs):
        n = rng.poisson(mu)
        if n == 0:
            continue
        seen += 1
        t = T * rng.random(n) ** (1 / beta)
        m_hat = T * np.log(T / t).sum() / n**2
        lower = m_hat * crow_time_terminated_coefficients(n, alpha_ci)[0]
        passed += lower >= requirement
    return passed / seen


def test_demonstration_chance_matches_simulation():
    curve = idealised_growth_curve(
        68.0, 500.0, growth_rate=0.23, test_time=2800.0
    )
    exact = curve.demonstration_probability(100.0, alpha_ci=0.2)
    simulated = _simulated_chance(curve, 100.0, 0.2, 20000, seed=20261125)
    # 20000 runs: a standard error of 0.0035.
    assert simulated == pytest.approx(exact, abs=0.012)


def test_consumer_risk_at_most_alpha():
    for mu in (2.0, 10.0, 40.0, 150.0):
        assert growth_demonstration_probability(1.0, mu, alpha_ci=0.2) <= 0.2
    # And it rises with the discrimination ratio and with the failures.
    assert growth_demonstration_probability(
        1.5, 30.0
    ) < growth_demonstration_probability(2.0, 30.0)
    assert growth_demonstration_probability(
        1.5, 30.0
    ) < growth_demonstration_probability(1.5, 60.0)


def test_curve_refusals():
    with pytest.raises(ValueError, match="exactly two"):
        idealised_growth_curve(68.0, 500.0, growth_rate=0.2)
    with pytest.raises(ValueError, match="longer than initial_time"):
        idealised_growth_curve(68.0, 500.0, growth_rate=0.2, test_time=400)
    with pytest.raises(ValueError, match="above initial_mtbf"):
        idealised_growth_curve(68.0, 500.0, final_mtbf=60.0, test_time=900)
    with pytest.raises(ValueError, match="jump at the end"):
        idealised_growth_curve(68.0, 500.0, growth_rate=0.5, final_mtbf=100)
    with pytest.raises(ValueError, match="growth_rate"):
        idealised_growth_curve(68.0, 500.0, growth_rate=1.2, test_time=900)


def test_curve_to_dict_and_repr():
    curve = idealised_growth_curve(
        68.0, 500.0, growth_rate=0.23, test_time=2800.0
    )
    d = curve.to_dict()
    assert d["result"] == "IdealisedGrowthCurve"
    json.dumps(d, allow_nan=False)
    assert IdealisedGrowthCurve(
        **{k: v for k, v in d.items() if k != "result"}
    ) == (curve)
    assert "Growth rate" in repr(curve)


# -- PM2-Continuous -----------------------------------------------------------


@pytest.fixture
def plan():
    # MIL-HDBK-189C figure 27: M_I = 25, M_G = 90, MS = 0.95, FEF 0.8.
    return pm2_continuous(
        25.0, 90.0, 30000.0, management_strategy=0.95, fef=0.8
    )


def test_pm2_continuous_ends(plan):
    assert plan.mtbf(0.0) == pytest.approx(25.0, rel=1e-14)
    assert plan.mtbf(30000.0) == pytest.approx(90.0, rel=1e-12)
    assert plan.growth_potential == pytest.approx(25.0 / (1 - 0.76))
    assert plan.goal_to_potential == pytest.approx(90 / (25 / 0.24))
    assert plan.mtbf(1e12) == pytest.approx(plan.growth_potential, rel=1e-6)
    # The handbook's beta, in terms of M_G / M_GP.
    expected = (1 - 25 / 90) / (25 / 90 - 25 / plan.growth_potential) / 3e4
    assert plan.beta == pytest.approx(expected, rel=1e-14)


def test_pm2_continuous_identities(plan):
    t = np.array([10.0, 1000.0, 20000.0])
    # mu'(t) = h(t); theta = 1 - h / lambda_B.
    h = 1e-4
    slope = (plan.expected_modes(t + h) - plan.expected_modes(t - h)) / (2 * h)
    np.testing.assert_allclose(slope, plan.new_mode_rate(t), rtol=1e-7)
    np.testing.assert_allclose(
        plan.fraction_surfaced(t),
        1 - plan.new_mode_rate(t) / plan.lambda_b,
        rtol=1e-12,
    )
    # rho = lambda_I - mu_d (lambda_B - h(t)).
    np.testing.assert_allclose(
        plan.intensity(t),
        1 / 25 - 0.8 * (plan.lambda_b - plan.new_mode_rate(t)),
        rtol=1e-12,
    )


def test_pm2_continuous_matches_its_mode_model():
    """The handbook's assumptions: many B modes with gamma rates, each
    first seen at an exponential time. As the number of modes grows (the
    gamma shape falling in step), the expected modes seen by t, the rate
    of the unseen ones and the intensity once the seen ones are fixed are
    PM2's."""
    plan = pm2_continuous(
        25.0, 90.0, 30000.0, management_strategy=0.95, fef=0.8
    )
    rng = np.random.default_rng(1125)
    k, reps = 20000, 200
    shape = plan.lambda_b / (k * plan.beta)
    t = np.array([2000.0, 10000.0, 30000.0])
    modes = np.zeros(t.size)
    unseen = np.zeros(t.size)
    for _ in range(reps):
        rates = rng.gamma(shape, plan.beta, size=k)
        with np.errstate(divide="ignore", over="ignore"):
            first = rng.exponential(1 / rates)  # a zero rate: never
        seen = first[None, :] <= t[:, None]
        modes += seen.sum(axis=1)
        unseen += np.where(seen, 0.0, rates[None, :]).sum(axis=1)
    modes /= reps
    unseen /= reps
    np.testing.assert_allclose(modes, plan.expected_modes(t), rtol=0.03)
    np.testing.assert_allclose(unseen, plan.new_mode_rate(t), rtol=0.03)
    rho = plan.lambda_a + 0.2 * (plan.lambda_b - unseen) + unseen
    np.testing.assert_allclose(1 / rho, plan.mtbf(t), rtol=0.01)


def test_pm2_continuous_targets(plan):
    table = plan.targets([5000.0, 15000.0, 30000.0], lags=[1000.0, 2000, 0])
    assert len(table) == 4
    assert table["mtbf"].iloc[0] == 25.0
    np.testing.assert_allclose(
        table["mtbf"].iloc[1:], plan.mtbf(np.array([4000.0, 13000, 30000]))
    )
    assert table["mtbf"].iloc[-1] == pytest.approx(90.0)
    assert np.all(np.diff(table["mtbf"]) > 0)
    # Lags lower the steps.
    no_lag = plan.targets([5000.0, 15000.0, 30000.0])
    assert np.all(no_lag["mtbf"].iloc[1:-1] > table["mtbf"].iloc[1:-1])
    with pytest.raises(ValueError, match="past the planned test time"):
        plan.targets([5000.0, 40000.0])
    with pytest.raises(ValueError, match="increasing"):
        plan.targets([5000.0, 4000.0])


def test_pm2_demonstration_probability(plan):
    # Enter at 90 (10% degraded: 81) and demonstrate 65 at 80% in a test
    # of 1000 hours: the most failures allowed, from the chi-squared
    # bound, then the Poisson chance at 81.
    chance = plan.demonstration_probability(
        65.0, 1000.0, alpha_ci=0.2, degradation=0.1
    )
    c = max(
        k
        for k in range(40)
        if demonstrated_mtbf(1000.0, k, alpha_ci=0.2) >= 65.0
    )
    assert chance == pytest.approx(stats.poisson.cdf(c, 1000.0 / 81.0))
    assert plan.demonstration_probability(65.0, 10.0) == 0.0


def test_pm2_continuous_refusals():
    with pytest.raises(ValueError, match="growth potential"):
        pm2_continuous(25.0, 110.0, 3e4, management_strategy=0.95, fef=0.8)
    with pytest.raises(ValueError, match="above initial_mtbf"):
        pm2_continuous(25.0, 20.0, 3e4, management_strategy=0.95, fef=0.8)
    with pytest.raises(ValueError, match="management_strategy"):
        pm2_continuous(25.0, 50.0, 3e4, management_strategy=0.0, fef=0.8)


def test_pm2_continuous_to_dict(plan):
    d = plan.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["beta"] == plan.beta
    assert "Growth potential" in repr(plan)


# -- PM2-Discrete -------------------------------------------------------------


@pytest.fixture
def dplan():
    # MIL-HDBK-189C section 5.6.6: R_I = 0.8987 and R_G = 0.9639.
    return pm2_discrete(0.8987, 0.9639, 400, management_strategy=0.9, fef=0.8)


def test_pm2_discrete_ends(dplan):
    assert dplan.reliability(1) == pytest.approx(0.8987, rel=1e-14)
    assert dplan.reliability(400) == pytest.approx(0.9639, rel=1e-12)
    assert dplan.growth_potential == pytest.approx(0.8987 ** (1 - 0.72))
    assert dplan.reliability(1e12) == pytest.approx(
        dplan.growth_potential, rel=1e-9
    )
    assert dplan.r_a * dplan.r_b == pytest.approx(0.8987)
    assert dplan.fraction_surfaced(1) == 0.0
    assert dplan.new_mode_probability(1) == pytest.approx(1 - dplan.r_b)


def test_pm2_discrete_matches_its_mode_model(dplan):
    """The handbook's assumptions: ``k`` independent B modes whose chances
    on a trial are a beta sample (shape ``n``, mean ``x / n`` with
    ``k x / n = -ln R_B``), each first seen at a geometric trial. Its
    exact expectations, from beta integrals, tend to PM2-Discrete's as
    ``k`` grows: at a million modes they agree to 1e-5."""
    from scipy.special import betaln

    k, n = 10**6, dplan.n
    x = -np.log(dplan.r_b) * n / k
    t = np.array([1.0, 10.0, 100.0, 400.0])
    log_b = betaln(x, n - x)
    mean = x / n
    # E[P (1 - P)^(t - 1)]: a mode not seen before t fails on trial t.
    fails_unseen = np.exp(betaln(x + 1, n - x + t - 1) - log_b)
    fails_seen = mean - fails_unseen
    seen_by = 1 - np.exp(betaln(x, n - x + t) - log_b)
    np.testing.assert_allclose(k * seen_by, dplan.expected_modes(t), rtol=1e-4)
    np.testing.assert_allclose(
        1 - (1 - fails_unseen) ** k,
        dplan.new_mode_probability(t),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        (1 - (1 - fails_seen) ** k) / (1 - (1 - mean) ** k),
        dplan.fraction_surfaced(t),
        rtol=1e-4,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        dplan.r_a * (1 - mean + 0.8 * fails_seen) ** k,
        dplan.reliability(t),
        rtol=1e-5,
    )


def test_pm2_discrete_targets(dplan):
    table = dplan.targets([100, 250, 402], lags=2)
    assert table["reliability"].iloc[0] == 0.8987
    np.testing.assert_allclose(
        table["reliability"].iloc[1:],
        dplan.reliability(np.array([99.0, 249.0, 401.0])),
    )
    assert table["reliability"].iloc[-1] == pytest.approx(
        float(dplan.reliability(401.0))
    )


def test_pm2_discrete_refusals():
    with pytest.raises(ValueError, match="growth potential"):
        pm2_discrete(0.9, 0.99, 400, management_strategy=0.9, fef=0.8)
    with pytest.raises(ValueError, match="more than 1"):
        pm2_discrete(0.9, 0.95, 1, management_strategy=0.9, fef=0.8)
    with pytest.raises(ValueError, match="trial number"):
        pm2_discrete(
            0.9, 0.95, 400, management_strategy=0.9, fef=0.8
        ).reliability(0)


# -- plots ---------------------------------------------------------------


def test_plots_draw():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import surpyval as surv

    curve = idealised_growth_curve(
        68.0, 500.0, growth_rate=0.23, test_time=2800.0
    )
    rng = np.random.default_rng(3)
    n = rng.poisson(float(curve.expected_failures(2800.0)))
    times = np.sort(2800.0 * rng.random(n) ** (1 / 0.77))
    model = surv.CrowAMSAA.fit(
        np.append(times, 2800.0), c=np.append(np.zeros(n), 1)
    )
    ax = curve.plot(phase_ends=[500, 1500, 2800], model=model, requirement=100)
    assert len(ax.lines) >= 3
    plan = pm2_continuous(
        25.0, 90.0, 30000.0, management_strategy=0.95, fef=0.8
    )
    plan.plot(phase_ends=[5000, 15000, 30000], lags=1000, requirement=65)
    pm2_discrete(0.8987, 0.9639, 400, management_strategy=0.9, fef=0.8).plot(
        phase_ends=[100, 250, 402], lags=2
    )
    with pytest.raises(ValueError, match="iif"):
        curve.plot(model=object())
    plt.close("all")
