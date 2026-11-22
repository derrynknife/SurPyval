"""The No-U-Turn sampler (``surpyval.bayes.engines.nuts``) on targets
whose answer is known.

The rule for every comparison was fixed before the first run, with the
seeds and sizes: a quantity must be within 4 Monte Carlo standard errors
of its exact value, the standard error from the draws' own effective
sample size (the smaller of bulk and tail). For a mean that is
``4 sd / sqrt(ESS)``; for a distribution, the exact CDF at the draws'
``p`` quantile must be within ``4 sqrt(p (1 - p) / ESS)`` of ``p``, at
``p`` = 2.5%, 25%, 50%, 75% and 97.5%.
"""

import math

import numpy as np
import pytest
from scipy.stats import norm

from surpyval.bayes import convergence as cv
from surpyval.bayes.engines import nuts

PROBABILITIES = (0.025, 0.25, 0.5, 0.75, 0.975)


def within_monte_carlo_error(draws, cdf, mean=None, sd=None):
    """Assert the rule of this module for one quantity's draws
    (``(chains, draws)``) against its exact ``cdf`` (and mean)."""
    ess = min(cv.ess_bulk(draws), cv.ess_tail(draws))
    flat = draws.ravel()
    for p in PROBABILITIES:
        gap = abs(float(cdf(np.quantile(flat, p))) - p)
        assert gap <= 4 * math.sqrt(p * (1 - p) / ess), (p, gap, ess)
    if mean is not None:
        gap = abs(flat.mean() - mean)
        assert gap <= 4 * sd / math.sqrt(cv.ess_bulk(draws)), (gap, ess)


class Normal:
    """A multivariate normal's log density and gradient (a class, so it
    pickles for the chains run in other processes)."""

    def __init__(self, mean, cov):
        self.mean = np.asarray(mean, dtype=float)
        self.cov = np.atleast_2d(np.asarray(cov, dtype=float))
        self.precision = np.linalg.inv(self.cov)

    def __call__(self, z):
        d = z - self.mean
        g = -self.precision @ d
        return 0.5 * float(d @ g), g


def funnel(z):
    """Neal's funnel: ``v ~ N(0, 3)``, nine ``x ~ N(0, exp(v / 2))``."""
    v, x = z[0], z[1:]
    value = -0.5 * (v / 3.0) ** 2 - 0.5 * len(x) * v
    value -= 0.5 * float(np.sum(x**2)) * math.exp(-v)
    grad = np.empty_like(z)
    grad[0] = (
        -v / 9.0 - 0.5 * len(x) + 0.5 * float(np.sum(x**2)) * math.exp(-v)
    )
    grad[1:] = -x * math.exp(-v)
    return value, grad


def test_independent_normals_on_very_different_scales():
    mean = np.array([5.0, -2.0, 0.0, 100.0, -1e4])
    sd = np.array([0.01, 0.3, 1.0, 30.0, 1000.0])
    f = Normal(mean, np.diag(sd**2))
    out = nuts.sample(f, np.tile(mean, (4, 1)), random_state=459001)
    D = out["draws"]
    assert D.shape == (4, 1000, 5)
    for j in range(5):
        within_monte_carlo_error(
            D[:, :, j],
            lambda q, j=j: norm.cdf(q, mean[j], sd[j]),
            mean[j],
            sd[j],
        )
    s = cv.summarise(D, list("abcde"), out["divergent"], out["depth"], 10)
    assert cv.check(s, 4, 1000) == []
    # the mass matrix found the scales (variances, within a factor of 2)
    ratio = out["inv_metric"] / sd**2
    assert np.all((ratio > 0.5) & (ratio < 2.0))


def test_correlated_normal():
    sd = np.array([1.0, 2.0, 0.5])
    corr = np.array([[1.0, 0.9, 0.3], [0.9, 1.0, 0.5], [0.3, 0.5, 1.0]])
    cov = corr * np.outer(sd, sd)
    f = Normal(np.zeros(3), cov)
    out = nuts.sample(f, np.zeros((4, 3)), random_state=459002)
    D = out["draws"]
    for j in range(3):
        within_monte_carlo_error(
            D[:, :, j], lambda q, j=j: norm.cdf(q, 0.0, sd[j]), 0.0, sd[j]
        )
    # the narrowest direction, which a diagonal mass matrix does not see
    w, v = np.linalg.eigh(cov)
    proj = D @ v[:, 0]
    within_monte_carlo_error(
        proj, lambda q: norm.cdf(q, 0.0, math.sqrt(w[0])), 0.0, math.sqrt(w[0])
    )
    s = cv.summarise(D, list("abc"), out["divergent"], out["depth"], 10)
    assert cv.check(s, 4, 1000) == []


@pytest.mark.parametrize("eps", [0.4, 1.3])
def test_transitions_are_exact_at_any_step_size(eps):
    """With the step size and mass matrix fixed (no adaptation), the
    transitions leave the target as it is whatever the step: a coarse
    step loses efficiency, not accuracy. Two normals of spread 1 and
    0.6, 20,000 transitions."""
    sd = np.array([1.0, 0.6])
    f = Normal(np.zeros(2), np.diag(sd**2))
    rng = np.random.default_rng(459003)
    z = np.zeros(2)
    logp, g = f(z)
    draws = np.empty((20000, 2))
    accept = np.empty(20000)
    inv_m = np.ones(2)
    for i in range(20000):
        z, logp, g, info = nuts.transition(f, z, logp, g, eps, inv_m, rng)
        draws[i] = z
        accept[i] = info["accept"]
    if eps > 1:
        # the step is coarse enough for the energy error to matter
        assert accept.mean() < 0.9
    for j in range(2):
        within_monte_carlo_error(
            draws[None, :, j],
            lambda q, j=j: norm.cdf(q, 0.0, sd[j]),
            0.0,
            sd[j],
        )


def test_a_funnel_is_reported_as_divergent():
    out = nuts.sample(
        funnel, np.zeros((4, 10)), draws=500, warmup=500, random_state=459004
    )
    names = ["v"] + [f"x{k}" for k in range(9)]
    s = cv.summarise(out["draws"], names, out["divergent"], out["depth"], 10)
    assert s["divergences"] > 0
    assert any("diverged" in p for p in cv.check(s, 4, 500))


def test_seeded_draws_do_not_depend_on_n_jobs():
    f = Normal([1.0, -1.0], [[1.0, 0.5], [0.5, 2.0]])
    kw = dict(draws=60, warmup=60)
    one = nuts.sample(f, np.zeros((3, 2)), random_state=7, n_jobs=1, **kw)
    two = nuts.sample(f, np.zeros((3, 2)), random_state=7, n_jobs=2, **kw)
    again = nuts.sample(f, np.zeros((3, 2)), random_state=7, **kw)
    other = nuts.sample(f, np.zeros((3, 2)), random_state=8, **kw)
    for key in ("draws", "logp", "depth", "step_size", "inv_metric"):
        assert np.array_equal(one[key], two[key]), key
        assert np.array_equal(one[key], again[key]), key
    assert not np.array_equal(one["draws"], other["draws"])
    # chains differ from each other
    assert not np.array_equal(one["draws"][0], one["draws"][1])


def test_the_one_seed_rule():
    f = Normal([0.0], [[1.0]])
    kw = dict(draws=30, warmup=30)
    np.random.seed(11)
    a = nuts.sample(f, np.zeros((2, 1)), **kw)["draws"]
    np.random.seed(11)
    b = nuts.sample(f, np.zeros((2, 1)), **kw)["draws"]
    assert np.array_equal(a, b)
    np.random.seed(11)
    before = np.random.random()
    np.random.seed(11)
    nuts.sample(f, np.zeros((2, 1)), random_state=3, **kw)
    assert np.random.random() == before


@pytest.mark.parametrize("warmup", [20, 37, 100, 150, 151, 400, 1000, 2500])
def test_warm_up_windows_fill_the_middle(warmup):
    start, ends = nuts.windows(warmup)
    assert 0 < start < ends[0] < warmup
    assert ends == sorted(set(ends))
    sizes = np.diff([start - 1] + ends)
    # each window is at least as long as the one before, and the last
    # leaves room for the step size to settle
    assert np.all(np.diff(sizes) >= 0)
    assert ends[-1] < warmup - 1
    if warmup >= 150:
        assert start == 75 and ends[-1] == warmup - 51


def test_no_windows_in_a_very_short_warm_up():
    assert nuts.windows(19) == (19, [])
    assert nuts.windows(0) == (0, [])


def test_a_point_outside_the_density_is_never_kept():
    """A density that is ``-inf`` (or not a number) beyond a wall: the
    chain stays inside, and the trajectories that hit the wall are
    counted as divergent."""

    def walled(z):
        if z[0] > 1.0:
            return float("nan"), np.zeros(1)
        return -0.5 * float(z[0] ** 2), -z

    out = nuts.sample(
        walled, np.zeros((2, 1)), draws=300, warmup=300, random_state=459005
    )
    assert np.all(out["draws"] <= 1.0)
    assert np.all(np.isfinite(out["logp"]))
    assert out["divergent"].sum() > 0


def test_settings_are_checked():
    f = Normal([0.0], [[1.0]])
    with pytest.raises(ValueError, match="draws must be a whole number"):
        nuts.sample(f, np.zeros((2, 1)), draws=2)
    with pytest.raises(ValueError, match="warmup must be a whole number"):
        nuts.sample(f, np.zeros((2, 1)), warmup=-1)
    with pytest.raises(ValueError, match="not finite at the chain's start"):
        nuts.sample(lambda z: (-np.inf, np.zeros(1)), np.zeros((1, 1)))
