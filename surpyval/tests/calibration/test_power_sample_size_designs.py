"""Simulated power at the sample sizes of ``test_planning.sample_size``.

Each study plans a study with one of the functions, simulates it at the
units planned, runs the test the plan is for, and compares the rejection
rate with the power the module promises at those units (its ``_power``
function). The formulas are asymptotic, so each power check carries a
slack of 0.03 (``check_rate``'s ``slack``); the size checks of the
non-inferiority and equivalence tests carry 0.01. The RMST study also
compares the analytic variance of a group's Kaplan-Meier RMST with its
variance over the simulated studies (to 8%, relative).

Seeds, replicates and tolerances were fixed before the first run.

- Cox model, continuous covariate correlated with another (R^2 = 0.3):
  the Wald test of its coefficient (Hsieh and Lavori 2000).
- Cox model, two-level covariate (30% at the first level) correlated
  with a continuous one: the Wald test of its coefficient.
- Non-inferiority: the upper 95% limit of the Cox hazard ratio below the
  margin 1.3, power at a true ratio of 1 and size at 1.3.
- Equivalence: the 90% interval inside (0.8, 1.25), power at 1 and size
  at 1.25.
- RMST: Weibull against a piecewise exponential, uniform entry, follow-up
  and dropout; the test of ``surpyval.rmst_diff``.
- Cure: mixture cure groups differing in the cured fraction, and in the
  latency hazard only; the log-rank test.
"""

import numpy as np
from scipy.stats import norm

import surpyval as sp
from surpyval.test_planning import (
    cox_power,
    cox_sample_size,
    cure_logrank_power,
    cure_logrank_sample_size,
    equivalence_power,
    equivalence_sample_size,
    noninferiority_power,
    noninferiority_sample_size,
    rmst_power,
    rmst_sample_size,
)
from surpyval.tests.calibration._montecarlo import check_rate

ALPHA = 0.05
POWER_SLACK = 0.03
SIZE_SLACK = 0.01


def _wald_rejects(x, c, Z):
    model = sp.CoxPH.fit(x, Z, c)
    z = model.beta[0] / model.standard_errors()[0]
    return abs(z) > norm.ppf(1 - ALPHA / 2)


# --- Cox regression -------------------------------------------------------


def _cox_continuous(rng, n):
    rho = np.sqrt(0.3)
    z1 = rng.normal(size=n)
    z2 = rho * z1 + np.sqrt(1 - rho**2) * rng.normal(size=n)
    rate = np.exp(np.log(1.35) * z1 + np.log(1.2) * z2)
    t = rng.exponential(1 / rate)
    return np.minimum(t, 1.0), (t > 1.0).astype(int), np.column_stack([z1, z2])


def test_cox_continuous_adjusted_power():
    pilot = np.random.default_rng(601)
    _, c, _ = _cox_continuous(pilot, 200000)
    event_probability = 1 - c.mean()
    n = cox_sample_size(1.35, event_probability, sd=1.0, r_squared=0.3)
    target = cox_power(1.35, n, event_probability, sd=1.0, r_squared=0.3)
    rng = np.random.default_rng(602)
    reps = 2000
    rejected = sum(
        _wald_rejects(*_cox_continuous(rng, n)) for _ in range(reps)
    )
    print(f"n {n}, event probability {event_probability:.4f}")
    check_rate(rejected, reps, target, "Cox continuous, R^2 0.3", POWER_SLACK)


def _cox_binary(rng, n):
    latent = rng.normal(size=n)
    z2 = 0.6 * latent + 0.8 * rng.normal(size=n)
    x1 = (latent > norm.ppf(0.7)).astype(float)  # 30% at the first level
    rate = np.exp(np.log(1.6) * x1 + np.log(1.2) * z2)
    t = rng.exponential(1 / rate)
    return np.minimum(t, 1.0), (t > 1.0).astype(int), np.column_stack([x1, z2])


def test_cox_binary_adjusted_power():
    pilot = np.random.default_rng(611)
    _, c, Z = _cox_binary(pilot, 200000)
    event_probability = 1 - c.mean()
    r_squared = np.corrcoef(Z[:, 0], Z[:, 1])[0, 1] ** 2
    kw = dict(proportion=0.3, r_squared=r_squared)
    n = cox_sample_size(1.6, event_probability, **kw)
    target = cox_power(1.6, n, event_probability, **kw)
    rng = np.random.default_rng(612)
    reps = 2000
    rejected = sum(_wald_rejects(*_cox_binary(rng, n)) for _ in range(reps))
    print(f"n {n}, R^2 {r_squared:.4f}, event p {event_probability:.4f}")
    check_rate(rejected, reps, target, "Cox two-level, adjusted", POWER_SLACK)


# --- Non-inferiority and equivalence --------------------------------------


def _hazard_ratio_interval(rng, n, hazard_ratio, z):
    """Two exponential groups censored at ln 2 (half the second group
    fail); the Cox interval of the hazard ratio, first to second."""
    group = np.repeat([1.0, 0.0], n)
    rate = np.where(group == 1, hazard_ratio, 1.0)
    t = rng.exponential(1 / rate)
    end = np.log(2.0)
    model = sp.CoxPH.fit(
        np.minimum(t, end), group[:, None], (t > end).astype(int)
    )
    beta, se = model.beta[0], model.standard_errors()[0]
    return np.exp(beta - z * se), np.exp(beta + z * se)


def _event_probability(hazard_ratio):
    return 1 - 0.5**hazard_ratio, 0.5


def test_noninferiority_power_and_size():
    margin, reps = 1.3, 2000
    z = norm.ppf(1 - ALPHA / 2)
    n = noninferiority_sample_size(margin, _event_probability(1.0))
    target = noninferiority_power(margin, n, _event_probability(1.0))
    rng = np.random.default_rng(621)
    shown = sum(
        _hazard_ratio_interval(rng, n[0], 1.0, z)[1] < margin
        for _ in range(reps)
    )
    print(f"n {n}")
    check_rate(shown, reps, target, "non-inferiority power", POWER_SLACK)
    rng = np.random.default_rng(622)
    shown = sum(
        _hazard_ratio_interval(rng, n[0], margin, z)[1] < margin
        for _ in range(reps)
    )
    check_rate(shown, reps, ALPHA / 2, "non-inferiority size", SIZE_SLACK)


def test_equivalence_power_and_size():
    margin, reps, alpha_ci = 1.25, 2000, 0.1
    z = norm.ppf(1 - alpha_ci / 2)
    n = equivalence_sample_size(
        margin, _event_probability(1.0), alpha_ci=alpha_ci
    )
    target = equivalence_power(
        margin, n, _event_probability(1.0), alpha_ci=alpha_ci
    )
    rng = np.random.default_rng(631)
    shown = 0
    for _ in range(reps):
        lower, upper = _hazard_ratio_interval(rng, n[0], 1.0, z)
        shown += (lower > 1 / margin) and (upper < margin)
    print(f"n {n}")
    check_rate(shown, reps, target, "equivalence power", POWER_SLACK)
    rng = np.random.default_rng(632)
    shown = 0
    for _ in range(reps):
        lower, upper = _hazard_ratio_interval(rng, n[0], margin, z)
        shown += (lower > 1 / margin) and (upper < margin)
    check_rate(shown, reps, alpha_ci / 2, "equivalence size", SIZE_SLACK)


# --- RMST ------------------------------------------------------------------

ACCRUAL, FOLLOW_UP, DROPOUT, TAU = 2.0, 4.0, 0.05, 5.0
WEIBULL = sp.Weibull.from_params([8.0, 1.3])


def _piecewise_sf(t):
    t = np.asarray(t, dtype=float)
    hazard = np.where(t < 2.0, 0.15 * t, 0.3 + 0.25 * (t - 2.0))
    return np.exp(-hazard)


def _piecewise_random(rng, n):
    e = rng.exponential(size=n)
    return np.where(e < 0.3, e / 0.15, 2.0 + (e - 0.3) / 0.25)


def _censor(rng, t):
    entry = rng.uniform(0, ACCRUAL, t.size)
    end = ACCRUAL + FOLLOW_UP - entry
    dropout = rng.exponential(1 / DROPOUT, t.size)
    censor = np.minimum(end, dropout)
    return np.minimum(t, censor), (censor < t).astype(int)


def test_rmst_power_and_variance():
    curves = (WEIBULL, _piecewise_sf)
    kw = dict(accrual_time=ACCRUAL, follow_up=FOLLOW_UP, dropout_rate=DROPOUT)
    n = rmst_sample_size(curves, TAU, **kw)
    target = rmst_power(curves, TAU, n, **kw)
    rng = np.random.default_rng(641)
    reps = 4000
    rejected, first = 0, []
    for _ in range(reps):
        x1, c1 = _censor(rng, WEIBULL.random(n[0], random_state=rng))
        x2, c2 = _censor(rng, _piecewise_random(rng, n[1]))
        result = sp.rmst_diff(
            sp.KaplanMeier.fit(x1, c1), sp.KaplanMeier.fit(x2, c2), tau=TAU
        )
        rejected += result["p_value"] <= ALPHA
        first.append(result["rmst_a"])
    print(f"n {n}")
    check_rate(rejected, reps, target, "RMST difference power", POWER_SLACK)
    # The analytic variance of the first group's RMST, against the
    # simulated: sigma^2 / n1 from the power's standard error.
    from surpyval.test_planning.sample_size import _rmst_moments

    _, variance = _rmst_moments(
        WEIBULL.sf, TAU, ACCRUAL, FOLLOW_UP, DROPOUT, "survival[0]"
    )
    simulated = np.var(first, ddof=1) * n[0]
    print(f"RMST variance: analytic {variance:.4f}, simulated {simulated:.4f}")
    assert abs(simulated / variance - 1) < 0.08


# --- Cure ------------------------------------------------------------------

LATENCY = sp.Weibull.from_params([2.0, 1.2])


def _cure_study(rng, n, cured, latency_hr, accrual, follow_up):
    group = np.repeat([0, 1], n)
    fraction = np.where(group == 0, cured[0], cured[1])
    # The first group's latency hazard is latency_hr times the second's:
    # S(t)^hr is a Weibull with scale alpha * hr^(-1 / beta).
    alpha = np.where(group == 0, 2.0 * latency_hr ** (-1 / 1.2), 2.0)
    t = alpha * rng.weibull(1.2, group.size)
    t = np.where(rng.uniform(size=group.size) < fraction, np.inf, t)
    entry = rng.uniform(0, accrual, group.size)
    end = accrual + follow_up - entry
    x, c = np.minimum(t, end), (end < t).astype(int)
    return x, group, c


def _cure_power(seed, cured, latency_hr):
    study = dict(accrual_time=1.0, follow_up=3.0)
    kw = dict(latency_hazard_ratio=latency_hr, **study)
    n = cure_logrank_sample_size(cured, LATENCY, **kw)
    target = cure_logrank_power(cured, LATENCY, n, **kw)
    rng = np.random.default_rng(seed)
    reps = 4000
    rejected = 0
    for _ in range(reps):
        x, group, c = _cure_study(rng, n[0], cured, latency_hr, 1.0, 3.0)
        rejected += sp.logrank(x, group, c=c).p_value <= ALPHA
    print(f"n {n}")
    return rejected, reps, target


def test_cure_fraction_power():
    rejected, reps, target = _cure_power(651, (0.45, 0.30), 1.0)
    check_rate(rejected, reps, target, "cure fraction log-rank", POWER_SLACK)


def test_cure_latency_power():
    rejected, reps, target = _cure_power(652, (0.30, 0.30), 0.7)
    check_rate(rejected, reps, target, "cure latency log-rank", POWER_SLACK)
