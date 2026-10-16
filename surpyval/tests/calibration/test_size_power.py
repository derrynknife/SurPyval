"""Size and power of the hypothesis tests.

A test's *size* is how often it rejects a true null at the 5% level; it
must be 5% (slack 0.01 for the asymptotic tests, 0 for the exact
MIL-HDBK-189C test), or at most 5% for an exact test of a discrete count
(the intervention test's conditional test, whose size is an exact sum the
simulation must match). *Power* is compared with a closed-form value where
one exists: Schoenfeld's formula for the log-rank test, its subdistribution
analogue for Gray's test (Latouche, Porcher and Chevret 2004), the exact
conditional power of MIL-HDBK-189C under a power-law process, a normal
approximation for the Laplace test, and the exact power of the intervention
test (``rate_ratio_power``). The approximations carry a larger slack
(stated per check); the exact values none.

Past failure this is built to catch: Gray's test rejected a true null up
to 89% of the time when the groups were censored differently. The Gray
and log-rank size studies censor the groups differently for that reason.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import chi2, norm

import surpyval as sp
from surpyval.recurrent import CrowAMSAA, intervention_test
from surpyval.recurrent.tests import laplace, mil_hdbk_189c
from surpyval.test_planning import rate_ratio_power
from surpyval.test_planning.rate_ratio import expected_power
from surpyval.tests.calibration._montecarlo import (
    check_rate,
    simulate_intervention,
    simulate_nhpp,
)

ALPHA = 0.05
Z_CRIT = norm.ppf(1 - ALPHA / 2)


# --- log-rank --------------------------------------------------------------


def test_logrank_size_unequal_censoring():
    rng = np.random.default_rng(501)
    reps, n = 20000, 100
    group = np.repeat([0, 1], n)
    c_max = np.repeat([15.0, 40.0], n)  # group 1 censored much later
    rejected = 0
    for _ in range(reps):
        t = 10.0 * rng.weibull(1.5, 2 * n)
        cens = rng.uniform(0, c_max)
        x, c = np.minimum(t, cens), (cens < t).astype(int)
        rejected += sp.logrank(x, group, c=c).p_value <= ALPHA
    check_rate(rejected, reps, ALPHA, "log-rank size")


def test_stratified_logrank_size():
    # The stratum both shifts the hazard (x4) and predicts the group
    # (25% versus 75% in group 1), so it confounds an unstratified test;
    # the stratified test compares within strata and holds its size.
    rng = np.random.default_rng(502)
    reps, n = 10000, 200
    strat_rej = plain_rej = 0
    for _ in range(reps):
        stratum = rng.binomial(1, 0.5, n)
        group = rng.binomial(1, np.where(stratum == 1, 0.75, 0.25))
        rate = np.where(stratum == 1, 4.0, 1.0)
        t = rng.exponential(1 / rate)
        cens = rng.uniform(0, 2.0, n)
        x, c = np.minimum(t, cens), (cens < t).astype(int)
        strat_rej += sp.logrank(x, group, c=c, strata=stratum).p_value <= ALPHA
        plain_rej += sp.logrank(x, group, c=c).p_value <= ALPHA
    check_rate(strat_rej, reps, ALPHA, "stratified log-rank size")
    # Design check: the confounding is real, so ignoring the strata rejects
    # far more often than 5%.
    print("unstratified rejection rate: {:.4f}".format(plain_rej / reps))
    assert plain_rej / reps > 0.3


def test_logrank_power():
    # Hazard ratio 1.5, exponential times, uniform censoring. Schoenfeld:
    # power = Phi(sqrt(d p (1 - p)) |log HR| - z_{alpha/2}), d the events.
    rng = np.random.default_rng(503)
    reps, n, hr = 10000, 150, 1.5
    group = np.repeat([0, 1], n)
    rejected, events = 0, 0.0
    for _ in range(reps):
        t = rng.exponential(1 / np.where(group == 1, hr, 1.0))
        cens = rng.uniform(0, 2.0, 2 * n)
        x, c = np.minimum(t, cens), (cens < t).astype(int)
        events += (1 - c).sum() / reps
        rejected += sp.logrank(x, group, c=c).p_value <= ALPHA
    power = norm.cdf(np.sqrt(events / 4) * np.log(hr) - Z_CRIT)
    print("expected events {:.1f}".format(events))
    check_rate(rejected, reps, power, "log-rank power (Schoenfeld)", 0.03)


# --- Gray's test -----------------------------------------------------------


def _competing(rng, n, hazard_a, hazard_b, c_max):
    t_a = rng.exponential(1 / hazard_a, n)
    t_b = rng.exponential(1 / hazard_b, n)
    cens = rng.uniform(0, c_max, n)
    t = np.minimum(t_a, t_b)
    x = np.minimum(t, cens)
    e = np.where(cens < t, None, np.where(t_a < t_b, "a", "b"))
    return x, e.astype(object)


def test_gray_size_unequal_censoring():
    # Same cause-specific hazards in both groups, so the same cumulative
    # incidence of cause "a"; group 0 is censored far earlier than group 1.
    rng = np.random.default_rng(511)
    reps, n = 20000, 150
    group = np.repeat([0, 1], n)
    rejected = 0
    for _ in range(reps):
        x0, e0 = _competing(rng, n, 0.1, 0.05, 8.0)
        x1, e1 = _competing(rng, n, 0.1, 0.05, 60.0)
        x, e = np.concatenate([x0, x1]), np.concatenate([e0, e1])
        rejected += sp.gray_test(x, e, group, "a").p_value <= ALPHA
    check_rate(rejected, reps, ALPHA, "Gray's test size")


def test_gray_power():
    # Fine and Gray's (1999) simulation model: subdistribution hazards
    # proportional with ratio exp(b) between the groups,
    # F_a(t | g) = 1 - (1 - p (1 - e^{-t}))^{exp(b g)}. The analogue of
    # Schoenfeld's formula counts the cause-"a" events.
    rng = np.random.default_rng(512)
    reps, n, p, b = 10000, 150, 0.4, np.log(1.6)
    group = np.repeat([0, 1], n)
    rejected, events = 0, 0.0
    for _ in range(reps):
        phi = np.exp(b * group)
        p_a = 1 - (1 - p) ** phi
        is_a = rng.uniform(size=2 * n) < p_a
        # Time given cause a: invert F_a(t | g) / p_a.
        u = rng.uniform(size=2 * n) * p_a
        t_a = -np.log(1 - (1 - (1 - u) ** (1 / phi)) / p)
        t_b = rng.exponential(1.0, 2 * n)
        t = np.where(is_a, t_a, t_b)
        cens = rng.uniform(0, 3.0, 2 * n)
        x = np.minimum(t, cens)
        e = np.where(cens < t, None, np.where(is_a, "a", "b")).astype(object)
        events += (e == "a").sum() / reps
        rejected += sp.gray_test(x, e, group, "a").p_value <= ALPHA
    power = norm.cdf(np.sqrt(events / 4) * b - Z_CRIT)
    print("expected cause-a events {:.1f}".format(events))
    check_rate(rejected, reps, power, "Gray's test power (Latouche)", 0.03)


# --- trend tests -----------------------------------------------------------


def _hpp_failure_truncated(rng, systems, events):
    """Each system observed to its ``events``-th event of a unit-rate HPP."""
    times = np.cumsum(rng.exponential(size=(systems, events)), axis=1)
    return times.ravel(), np.repeat(np.arange(systems), events)


@pytest.mark.parametrize("truncation", ["time", "failure", "delayed"])
@pytest.mark.parametrize(
    "test, slack",
    [(laplace, 0.01), (mil_hdbk_189c, 0.0)],
    ids=["laplace", "mil"],
)
def test_trend_test_size(test, slack, truncation):
    # Null: a homogeneous Poisson process. MIL-HDBK-189C is exact under it
    # (chi-squared on 2N or 2(N - 1) degrees of freedom), so no slack.
    # "delayed": each system enters at a random time and is watched for 20
    # (#575), so the windows (s_q, s_q + 20] do not start at 0.
    seeds = {"time": 521, "failure": 522, "delayed": 524}
    rng = np.random.default_rng(seeds[truncation])
    reps = 30000
    rejected = 0
    for _ in range(reps):
        if truncation in ("time", "delayed"):
            m = rng.poisson(20.0, 3)
            s = rng.uniform(0, 50.0, 3) if truncation == "delayed" else 0.0
            s = np.broadcast_to(s, 3)
            x = np.concatenate(
                [
                    np.sort(rng.uniform(s[q], s[q] + 20.0, k))
                    for q, k in enumerate(m)
                ]
            )
            i = np.repeat(np.arange(3), m)
            res = test(x, i=i, T=s + 20.0, tl=s)
        else:
            x, i = _hpp_failure_truncated(rng, 3, 20)
            res = test(x, i=i)
        rejected += res.p_value <= ALPHA
    check_rate(
        rejected,
        reps,
        ALPHA,
        "{} size ({}-truncated)".format(test.__name__, truncation),
        slack=slack,
    )


def test_trend_test_power():
    # Power-law process, shape 1.5, 3 systems to T = 20 (about 60 events).
    # Given N events the times are T U^{1/1.5}, so -log(t / T) is
    # exponential with rate 1.5: the MIL statistic is chi2(2N) / 1.5 exactly
    # and its conditional power has a closed form, averaged over the
    # replicates (a Rao-Blackwellised expected power). The Laplace sum of
    # t / T has mean 1.5 / 2.5 and variance 1.5 / (2.5^2 3.5) per event and
    # is compared with its normal approximation.
    rng = np.random.default_rng(523)
    reps, shape, T, systems = 20000, 1.5, 20.0, 3
    scale = T / 20.0 ** (1 / shape)  # 20 expected events a system
    rej_mil = rej_lap = 0
    pow_mil = pow_lap = 0.0
    for _ in range(reps):
        m = rng.poisson((T / scale) ** shape, systems)
        x = np.concatenate([T * rng.uniform(size=k) ** (1 / shape) for k in m])
        i = np.repeat(np.arange(systems), m)
        rej_mil += mil_hdbk_189c(x, i=i, T=T).p_value <= ALPHA
        rej_lap += laplace(x, i=i, T=T).p_value <= ALPHA
        dof = 2 * m.sum()
        lo, hi = chi2.ppf(ALPHA / 2, dof), chi2.ppf(1 - ALPHA / 2, dof)
        pow_mil += (
            chi2.cdf(shape * lo, dof) + chi2.sf(shape * hi, dof)
        ) / reps
        nn = m.sum()
        mean_u = shape / (shape + 1)
        var_u = shape / ((shape + 1) ** 2 * (shape + 2))
        # U statistic = (sum t/T - N/2) / sqrt(N/12) in T units.
        mu = (nn * mean_u - nn / 2) / np.sqrt(nn / 12)
        sd = np.sqrt(nn * var_u) / np.sqrt(nn / 12)
        pow_lap += (
            norm.sf((Z_CRIT - mu) / sd) + norm.cdf((-Z_CRIT - mu) / sd)
        ) / reps
    check_rate(rej_mil, reps, pow_mil, "MIL-HDBK-189C power (exact)", 0.0)
    check_rate(rej_lap, reps, pow_lap, "Laplace power (normal approx.)", 0.02)


# --- goodness of fit -------------------------------------------------------


def test_cramer_von_mises_size():
    # Data from the model fitted: a power-law process, 5 systems to T = 50.
    # With 39 bootstrap samples the p-value is a multiple of 1/40, and
    # p <= 0.05 and p <= 0.25 have exact size 0.05 and 0.25 for a valid
    # bootstrap, so the rejection rates are compared with both.
    rng = np.random.default_rng(531)
    reps = 300
    p_values = np.empty(reps)
    for r in range(reps):
        x, i, c = simulate_nhpp(
            rng,
            lambda t: (t / 10.0) ** 1.5,
            lambda n: 10.0 * n ** (1 / 1.5),
            systems=5,
            t_end=50.0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = CrowAMSAA.fit(x, i, c=c)
            p_values[r] = model.cramer_von_mises(
                n_boot=39, random_state=r
            ).p_value
    check_rate((p_values <= 0.05).sum(), reps, 0.05, "CvM size at 0.05")
    check_rate((p_values <= 0.25).sum(), reps, 0.25, "CvM size at 0.25")


# --- the intervention test -------------------------------------------------
#
# One system with rate 0.2 observed to 100 and changed at 50 (about 20
# failures), and fleets. The exact conditional test's size and power are
# exact sums (``expected_power``), so the simulations are compared with
# them with no slack; the mid-p test and the power law's likelihood ratio
# are approximate, with slack 0.01 and 0.015.


def _hpp_size_and_power(ratio, method):
    """The analytic two-sided rejection rate of the exact (or mid-p)
    conditional test, of each tail, for the single-system design."""
    midp = method == "mid-p"
    before, after = 0.2 * 50.0, 0.2 * 50.0
    lower = expected_power(before, after, ratio, ALPHA / 2, midp)
    # The upper tail: the periods swapped (``expected_power`` gives the
    # tail in the direction of a decrease for a ratio of 1).
    upper = (
        expected_power(ratio * after, before, 1.0, ALPHA / 2, midp)
        if ratio == 1.0
        else 0.0
    )
    return lower, upper


@pytest.mark.parametrize("method", ["exact", "mid-p"])
def test_intervention_hpp_size(method):
    # No change. The exact test's size is at most 5% (2.9% here: an exact
    # test of a discrete count is conservative), the mid-p test's near 5%
    # (4.6%), and the simulation must match the exact sums.
    rng = np.random.default_rng(541 if method == "exact" else 542)
    reps = 8000
    rejected = 0
    for _ in range(reps):
        x, _ = simulate_intervention(rng, 5.0, 1.0, 1.0, [50.0], [100.0])
        result = intervention_test(x, T=100, intervention=50, method=method)
        rejected += result.p_value < ALPHA
    size = sum(_hpp_size_and_power(1.0, method))
    print("{} analytic size {:.4f}".format(method, size))
    if method == "exact":
        assert size <= ALPHA
    else:
        assert abs(size - ALPHA) < 0.01
    check_rate(rejected, reps, size, method + " intervention size", 0.0)


@pytest.mark.parametrize("method", ["exact", "mid-p"])
def test_intervention_hpp_power(method):
    # The rate halves at the intervention: the share of tests finding a
    # decrease is the exact power of ``rate_ratio_power``.
    rng = np.random.default_rng(543 if method == "exact" else 544)
    reps = 8000
    found = 0
    for _ in range(reps):
        x, _ = simulate_intervention(rng, 5.0, 1.0, 0.5, [50.0], [100.0])
        result = intervention_test(x, T=100, intervention=50, method=method)
        found += result.verdict == "decrease"
    power = rate_ratio_power(0.5, 0.2, 50.0, 50.0, method=method)
    check_rate(found, reps, power, method + " intervention power", 0.0)


def _fleet(n_systems, end):
    """Each system's observation end, as a dict (a system with no failure
    must be named to count its exposure)."""
    return {q: end for q in range(n_systems)}


def test_intervention_per_system_size_fleet():
    # Four systems with rates 0.05 to 0.4, observed to 100, the busiest
    # changed latest, so their failures fall mostly before the change and
    # the quiet systems' exposure mostly after it. Nothing changed. The
    # per-system (stratified) exact test holds its size; pooling the
    # systems, as baseline="common" does, finds a "decrease" far too often.
    rng = np.random.default_rng(545)
    reps = 4000
    scales = 1 / np.array([0.05, 0.1, 0.2, 0.4])
    taus = [20.0, 40.0, 60.0, 80.0]
    T, tau = _fleet(4, 100.0), dict(enumerate(taus))
    per = pooled = 0
    for _ in range(reps):
        x, i = simulate_intervention(rng, scales, 1.0, 1.0, taus, [100.0] * 4)
        result = intervention_test(x, i, T=T, intervention=tau)
        per += result.p_value < ALPHA
        result = intervention_test(
            x, i, T=T, intervention=tau, baseline="common"
        )
        pooled += result.p_value < ALPHA
    check_rate(per, reps, ALPHA, "per-system size, fleet", 0.0, side="upper")
    print("pooled size, fleet: {:.4f}".format(pooled / reps))
    assert pooled / reps > 0.3


def test_intervention_power_law_size():
    # Five systems wearing out (shape 1.5, about 12 failures each to age
    # 50), changed at ages 15 to 35, and nothing changed. The likelihood
    # ratio test holds its size; the HPP comparison takes the wear-out for
    # a worsening.
    rng = np.random.default_rng(546)
    reps = 3000
    scale = 50.0 / 12.0 ** (1 / 1.5)
    taus = [15.0, 20.0, 25.0, 30.0, 35.0]
    T, tau = _fleet(5, 50.0), dict(enumerate(taus))
    power_law = hpp = 0
    for _ in range(reps):
        x, i = simulate_intervention(rng, scale, 1.5, 1.0, taus, [50.0] * 5)
        result = intervention_test(
            x, i, T=T, intervention=tau, model="power-law"
        )
        power_law += result.p_value < ALPHA
        hpp += intervention_test(x, i, T=T, intervention=tau).p_value < ALPHA
    check_rate(power_law, reps, ALPHA, "power-law LR size", 0.015)
    print("HPP size under wear-out: {:.4f}".format(hpp / reps))
    assert hpp / reps > 0.15
