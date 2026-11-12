"""Size and power of the max-combo, trend, permutation and maximally
selected log-rank tests.

A test's *size* is how often it rejects a true null at the 5% level. The
asymptotic tests (max-combo, trend, the maximally selected statistic's
"exact" Gauss p-value) must be within 3 Monte Carlo standard errors plus
the usual 0.01 slack of 5%. The permutation tests are exact: an exact
p-value rejects at most 5% of the time, and a Monte Carlo p-value
``(1 + count) / (1 + B)`` with ``B = 199`` rejects exactly
``floor(0.05 * 200) / 200 = 5%`` of the time for a continuous statistic,
so no slack. Lausen's improved Bonferroni bound may only be conservative.
The naive p-value at the best cut is checked to be invalid (it must
reject a true null far more often than 5%), as the docs say.

Power: under a delayed treatment effect (no difference for the first
5 time units, then a hazard ratio of 0.5) the max-combo test must be more
powerful than the log-rank test; both rates are printed.

Seeds, sample sizes and tolerances were fixed before the first run.
"""

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_rate

ALPHA = 0.05


def _censored(rng, t, upper):
    cens = rng.uniform(0, upper, t.size)
    return np.minimum(t, cens), (cens < t).astype(int)


def test_maxcombo_size_unequal_censoring():
    rng = np.random.default_rng(601)
    reps, n = 3000, 60
    Z = np.repeat([0, 1], n)
    upper = np.repeat([20.0, 30.0], n)
    rejected = 0
    for _ in range(reps):
        t = 10.0 * rng.weibull(1.2, 2 * n)
        x, c = _censored(rng, t, upper)
        rejected += sp.maxcombo_logrank(x, Z, c=c).p_value <= ALPHA
    check_rate(rejected, reps, ALPHA, "max-combo size")


def _delayed_effect(rng, n, delay=5.0, base=0.1, hr=0.5, follow=25.0):
    # Hazard `base` in both groups to `delay`; then `base * hr` in group 1.
    t0 = rng.exponential(1 / base, n)
    t1 = rng.exponential(1 / base, n)
    late = t1 > delay
    t1[late] = delay + rng.exponential(1 / (base * hr), late.sum())
    t = np.concatenate([t0, t1])
    return np.minimum(t, follow), (t > follow).astype(int)


def test_maxcombo_power_beats_logrank_under_a_delayed_effect():
    rng = np.random.default_rng(602)
    reps, n = 1500, 100
    Z = np.repeat([0, 1], n)
    hits_combo = hits_lr = 0
    for _ in range(reps):
        x, c = _delayed_effect(rng, n)
        hits_combo += sp.maxcombo_logrank(x, Z, c=c).p_value <= ALPHA
        hits_lr += sp.logrank(x, Z, c=c).p_value <= ALPHA
    combo, lr = hits_combo / reps, hits_lr / reps
    se = np.sqrt((combo * (1 - combo) + lr * (1 - lr)) / reps)
    print(f"delayed effect power: max-combo {combo:.3f}, log-rank {lr:.3f}")
    assert combo > lr + 2 * se, (combo, lr, se)


def test_trend_size():
    rng = np.random.default_rng(603)
    reps, n = 4000, 50
    Z = np.repeat([0, 1, 2], n)
    rejected = 0
    for _ in range(reps):
        t = 10.0 * rng.weibull(1.5, 3 * n)
        x, c = _censored(rng, t, 25.0)
        rejected += sp.trend_logrank(x, Z, c=c).p_value <= ALPHA
    check_rate(rejected, reps, ALPHA, "log-rank trend size")


def test_permutation_exact_size():
    rng = np.random.default_rng(604)
    reps, n = 2000, 6
    Z = np.repeat([0, 1], n)
    rejected = 0
    for _ in range(reps):
        t = rng.exponential(10.0, 2 * n)
        x, c = _censored(rng, t, 25.0)
        res = sp.permutation_logrank(x, Z, c=c)
        assert res.method == "exact"
        rejected += res.p_value <= ALPHA
    check_rate(
        rejected, reps, ALPHA, "permutation (exact) size", 0.0, side="upper"
    )


def test_permutation_monte_carlo_size():
    rng = np.random.default_rng(605)
    reps, n = 2000, 20
    Z = np.repeat([0, 1, 2], n)
    rejected = 0
    for _ in range(reps):
        t = rng.exponential(10.0, 3 * n)
        x, c = _censored(rng, t, 25.0)
        res = sp.permutation_logrank(
            x, Z, c=c, method="monte-carlo", n_permutations=199,
            random_state=rng,
        )  # fmt: skip
        rejected += res.p_value <= ALPHA
    check_rate(rejected, reps, ALPHA, "permutation (Monte Carlo) size", 0.0)


def test_maxstat_size():
    rng = np.random.default_rng(606)
    reps, N = 2000, 100
    hits = {"exact": 0, "lausen94": 0, "naive": 0}
    for _ in range(reps):
        Z = rng.uniform(0, 10, N)
        t = rng.exponential(10.0, N)
        x, c = _censored(rng, t, 25.0)
        res = sp.maxstat_logrank(x, Z, c=c)
        hits["exact"] += res.p_value <= ALPHA
        hits["naive"] += res.naive_p_value <= ALPHA
        b = sp.maxstat_logrank(x, Z, c=c, method="lausen94")
        hits["lausen94"] += b.p_value <= ALPHA
    check_rate(hits["exact"], reps, ALPHA, "maxstat size (exact Gauss)")
    check_rate(
        hits["lausen94"], reps, ALPHA, "maxstat size (Lausen 94)",
        side="upper",
    )  # fmt: skip
    naive = check_rate(
        hits["naive"], reps, 0.15, "naive p at the best cut (invalid)",
        0.0, side="lower",
    )  # fmt: skip
    print(f"naive p-value at the best cut rejects {naive:.3f} of true nulls")
