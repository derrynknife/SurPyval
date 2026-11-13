"""Coverage of the non-parametric pointwise bounds and simultaneous bands.

Four studies:

- **Pointwise** ``cb`` of Kaplan-Meier, Nelson-Aalen and Fleming-Harrington
  (default log(-log) bounds) at the true quartiles of a censored Weibull
  sample, n = 100; and the Kaplan-Meier's other interval types ('log',
  'logit', 'arcsine', 'peto', 'rothman') on the same design.
- **Competing risks**: the pointwise ``cb`` of ``CompetingRisks``' cumulative
  incidence (Aalen's variance, log(-log) bounds) at two times of a
  simulated wear-out-or-shock sample, n = 200, at 95% and 90%.
- **Simultaneous** ``band`` (Hall-Wellner and equal precision, with their
  defaults: the arcsine-square-root scale, and for the equal precision
  band the times with 0.1 <= a <= 0.9): the whole true survival curve,
  over the range the band covers, must lie inside it. Over the first to
  the last event and on the log(-log) scale, the defaults until v0.22, the
  equal precision band covered about 0.89, at n = 100 and at n = 400,
  most of its misses at the first few event times (#390).
- **Band critical values**, checked against an independent Monte Carlo of
  the limiting Brownian bridge. This is the check that sees a critical
  value 1.5% low (coverage 94.4% for 95%), which no finite-sample coverage
  study of practical size can: it needs a Monte Carlo error of about 0.1%.
  The Monte Carlo here is exact up to the grid's linear boundary, not a
  grid approximation of the supremum: between grid points the bridge is a
  Brownian bridge between the two values it takes there, whose probability
  of touching a straight boundary is known in closed form, so each path
  contributes its probability of staying inside rather than an indicator
  of doing so at the grid points. (The old simulated critical value did the
  latter, and so missed the excursions between grid points.)

Slack: 0.01 for the pointwise bounds and 0.015 for the bands, whose
approximation is asymptotic in the whole curve rather than at one time;
0.0005 for the critical values, whose only approximation is the linear
boundary within a grid step.
"""

import warnings

import numpy as np
import pytest
from scipy.stats import kstwobign

import surpyval as sp
from surpyval.tests.calibration._montecarlo import (
    Z_TOL,
    check_coverage,
    check_rate,
    rate_se,
)
from surpyval.univariate.nonparametric.nonparametric import NonParametric

TRUE = sp.Weibull.from_params([10.0, 1.5])
N = 100
C_MAX = 25.0  # uniform censoring on (0, C_MAX): about 30% censored


def _sample(rng):
    t = TRUE.qf(rng.uniform(size=N))
    cens = rng.uniform(0, C_MAX, N)
    return np.minimum(t, cens), (cens < t).astype(int)


@pytest.mark.parametrize(
    "name", ["KaplanMeier", "NelsonAalen", "FlemingHarrington"]
)
def test_pointwise_coverage(name):
    fitter = getattr(sp, name)
    rng = np.random.default_rng(201)
    t_eval = TRUE.qf(np.array([0.25, 0.5, 0.75]))
    truth = TRUE.sf(t_eval)
    reps = 2000
    lo, hi = np.empty((reps, 3)), np.empty((reps, 3))
    lo1 = np.empty((reps, 3))
    for r in range(reps):
        x, c = _sample(rng)
        model = fitter.fit(x, c=c)
        b = model.cb(t_eval, on="sf")
        lo[r], hi[r] = b[:, 0], b[:, 1]
        lo1[r] = model.cb(t_eval, on="sf", bound="lower")
    check_coverage(lo, hi, truth, 0.95, name + " cb(sf)")
    check_coverage(lo1, np.inf, truth, 0.95, name + " cb(sf, 'lower')")


@pytest.mark.parametrize(
    "bound_type", ["log", "logit", "arcsine", "peto", "rothman"]
)
def test_pointwise_coverage_by_type(bound_type):
    # The other interval types of the Kaplan-Meier's ``cb``, on the same
    # design as the default's above (n = 100, the true quartiles), two-sided
    # and lower at 95%. The same seed for every type: the same samples.
    rng = np.random.default_rng(206)
    t_eval = TRUE.qf(np.array([0.25, 0.5, 0.75]))
    truth = TRUE.sf(t_eval)
    reps = 2000
    lo, hi = np.empty((reps, 3)), np.empty((reps, 3))
    lo1 = np.empty((reps, 3))
    for r in range(reps):
        x, c = _sample(rng)
        model = sp.KaplanMeier.fit(x, c=c)
        b = model.cb(t_eval, on="sf", bound_type=bound_type)
        lo[r], hi[r] = b[:, 0], b[:, 1]
        lo1[r] = model.cb(
            t_eval, on="sf", bound="lower", bound_type=bound_type
        )
    label = f"KaplanMeier cb(sf, bound_type={bound_type!r})"
    if bound_type == "peto":
        # Peto's variance, S^2 (1 - S) / Y, counts the units at risk at t
        # only, fewer than the information behind the estimate once
        # censoring has thinned the risk set, so it overstates the
        # variance late in follow-up and the interval is conservative there
        # (Dorey and Korn, 1987). This check is one-sided (at least nominal
        # less the tolerance). It was made so after the first run, whose
        # two-sided check failed on over-coverage only: the lower bound at
        # the third quartile covered 0.9770 (two-sided interval 0.9565).
        for name, lower, upper in (
            ("", lo, hi),
            (", 'lower'", lo1, np.full_like(lo1, np.inf)),
        ):
            hits = (lower <= truth) & (truth <= upper)
            for k in range(truth.size):
                check_rate(
                    int(hits[:, k].sum()),
                    reps,
                    0.95,
                    f"{label}{name} [{k}]",
                    side="lower",
                )
        return
    check_coverage(lo, hi, truth, 0.95, label)
    check_coverage(lo1, np.inf, truth, 0.95, label + ", 'lower'")


def _band_covers(model, method):
    """Whether the true sf lies inside the band wherever it is defined.

    The band is a step function, constant on ``[x_j, x_{j+1})``. The true
    sf is continuous and decreasing, so over that step it is largest at
    ``x_j`` and smallest just before ``x_{j+1}``: the band holds on the
    whole step when the first is below the upper edge and the second above
    the lower edge.
    """
    xs = model.x
    band = model.band(method=method)
    valid = np.flatnonzero(np.isfinite(band[:, 0]))
    ends = np.append(xs[1:], xs[-1])
    lo, hi = band[valid, 0], band[valid, 1]
    return bool(
        np.all(TRUE.sf(xs[valid]) <= hi) and np.all(TRUE.sf(ends[valid]) >= lo)
    )


@pytest.mark.parametrize("method", ["hall-wellner", "nair"])
def test_band_coverage(method):
    rng = np.random.default_rng(202 if method == "nair" else 203)
    reps = 500
    hits = np.empty(reps, dtype=bool)
    for r in range(reps):
        x, c = _sample(rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hits[r] = _band_covers(sp.KaplanMeier.fit(x, c=c), method)
    check_rate(
        hits.sum(),
        reps,
        0.95,
        f"KaplanMeier band({method}) coverage",
        slack=0.015,
    )


def _inside_probability(c, a_l, a_u, standardized, n_paths, rng):
    """P(|B(a)| <= b(a) for all a in [a_l, a_u]) for a Brownian bridge B,
    with b = c (Hall-Wellner) or c sqrt(a (1 - a)) (equal precision).

    Each path is simulated exactly at the grid points; its contribution is
    the probability that the bridge between consecutive points stays inside
    the (linearly interpolated) boundary, ``(1 - e^{-2(b0 - x)(b1 - y)/h})``
    for the upper edge times the same for the lower edge.
    """
    if standardized:
        # Uniform in log-odds: the boundary bends most near a_l.
        lg = np.linspace(
            np.log(a_l / (1 - a_l)), np.log(a_u / (1 - a_u)), 4001
        )
        grid = 1.0 / (1.0 + np.exp(-lg))
        bound = c * np.sqrt(grid * (1.0 - grid))
    else:
        grid = np.linspace(a_l, a_u, 2001)
        bound = np.full(grid.size, c)
    out = 0.0
    chunk = 50_000
    for start in range(0, n_paths, chunk):
        m = min(chunk, n_paths - start)
        x = rng.normal(0.0, np.sqrt(grid[0] * (1.0 - grid[0])), m)
        w = (np.abs(x) <= bound[0]).astype(float)
        for k in range(1, grid.size):
            a0, a1 = grid[k - 1], grid[k]
            h = a1 - a0
            mean = x * (1.0 - a1) / (1.0 - a0)
            sd = np.sqrt(h * (1.0 - a1) / (1.0 - a0))
            y = mean + sd * rng.standard_normal(m)
            b0, b1 = bound[k - 1], bound[k]
            inside = np.abs(y) <= b1
            with np.errstate(over="ignore", invalid="ignore"):
                up = -np.expm1(-2.0 * (b0 - x) * (b1 - y) / h)
                dn = -np.expm1(-2.0 * (b0 + x) * (b1 + y) / h)
            w *= np.where(inside, up * dn, 0.0)
            x = y
        out += w.sum()
    return out / n_paths


@pytest.mark.parametrize(
    "a_l, a_u, standardized",
    [
        (0.0, 1.0 - 1e-9, False),
        (0.05, 0.6, False),
        (0.05, 0.6, True),
        (0.2, 0.9, True),
    ],
)
def test_band_critical_value(a_l, a_u, standardized):
    rng = np.random.default_rng(204)
    crit = NonParametric._band_critical_value(a_l, a_u, 0.05, standardized)
    if a_l == 0.0 and not standardized:
        # Over the whole range the supremum of |B| is Kolmogorov's.
        assert crit == pytest.approx(kstwobign.ppf(0.95), abs=1e-6)
    n_paths = 200_000
    p = _inside_probability(crit, a_l, a_u, standardized, n_paths, rng)
    tol = Z_TOL * rate_se(0.95, n_paths) + 0.0005
    line = "critical value {:.5f} on [{}, {}] ({}): P(inside) {:.5f}".format(
        crit, a_l, a_u, "nair" if standardized else "hall-wellner", p
    )
    print(line + f" (target 0.95, tolerance +/- {tol:.5f})")
    assert abs(p - 0.95) <= tol, line
    # The design sees the old error: a value 1.5% low is outside tolerance.
    p_low = _inside_probability(
        0.985 * crit, a_l, a_u, standardized, n_paths, rng
    )
    print(f"  at 0.985 x critical value: P(inside) {p_low:.5f}")
    assert p_low < 0.95 - tol


def _true_wear_cif(t):
    """The cumulative incidence of wear (a Weibull(100, 3) latent life)
    with shocks (an exponential latent life, mean 150) competing."""
    from scipy.integrate import quad

    wear = sp.Weibull.from_params([100.0, 3.0])
    return quad(lambda u: wear.df(u) * np.exp(-u / 150.0), 0.0, t)[0]


@pytest.mark.parametrize("alpha_ci", [0.05, 0.1])
def test_competing_risks_cif_coverage(alpha_ci):
    # n = 200 with uniform censoring on (0, 250); at t = 50 and 100 the
    # true CIF is 0.09 and 0.40. (At a time with no event of the cause
    # yet, the bound is [0, 0] and misses, as a Kaplan-Meier's [1, 1]
    # does; past the last time it is NaN.)
    from surpyval.univariate.competing_risks import CompetingRisks

    rng = np.random.default_rng(205)
    t_eval = np.array([50.0, 100.0])
    truth = np.array([_true_wear_cif(t) for t in t_eval])
    reps, n = 2000, 200
    lo, hi = np.empty((reps, 2)), np.empty((reps, 2))
    for r in range(reps):
        t_wear = 100.0 * rng.weibull(3.0, n)
        t_shock = rng.exponential(150.0, n)
        t_cens = rng.uniform(0.0, 250.0, n)
        x = np.minimum.reduce([t_wear, t_shock, t_cens])
        e = np.where(
            x == t_cens, None, np.where(t_wear < t_shock, "wear", "shock")
        ).astype(object)
        b = CompetingRisks.fit(x, e).cb(t_eval, "wear", alpha_ci=alpha_ci)
        lo[r], hi[r] = b[:, 0], b[:, 1]
    label = f"CompetingRisks cb(cif) at {1 - alpha_ci:.0%}"
    check_coverage(lo, hi, truth, 1 - alpha_ci, label)


def _true_wear_rmtl(tau):
    """The time lost to wear up to ``tau`` in the design above: the area
    under its cumulative incidence, ``int_0^tau (tau - u) f(u) e^{-u /
    150} du``."""
    from scipy.integrate import quad

    wear = sp.Weibull.from_params([100.0, 3.0])
    return quad(
        lambda u: (tau - u) * wear.df(u) * np.exp(-u / 150.0),
        0.0,
        tau,
        epsabs=0.0,
        epsrel=1e-12,
    )[0]


def _wear_or_shock(rng, n):
    t_wear = 100.0 * rng.weibull(3.0, n)
    t_shock = rng.exponential(150.0, n)
    t_cens = rng.uniform(0.0, 250.0, n)
    x = np.minimum.reduce([t_wear, t_shock, t_cens])
    e = np.where(
        x == t_cens, None, np.where(t_wear < t_shock, "wear", "shock")
    ).astype(object)
    return x, e


def test_competing_risks_rmtl_coverage():
    # The 95% interval of the restricted mean time lost to wear, at
    # tau = 75 and 110 (true 5.29 and 17.65), n = 200, on the design of
    # the cumulative incidence's study above. (The first run used 125,
    # which some replicates' largest time falls short of, about 0.2% of
    # them: rmtl refuses a tau past the data, and the run stopped there.)
    from surpyval.univariate.competing_risks import CompetingRisks

    rng = np.random.default_rng(207)
    taus = np.array([75.0, 110.0])
    truth = np.array([_true_wear_rmtl(t) for t in taus])
    reps, n = 2000, 200
    lo, hi = np.empty((reps, 2)), np.empty((reps, 2))
    for r in range(reps):
        model = CompetingRisks.fit(*_wear_or_shock(rng, n))
        for k, tau in enumerate(taus):
            res = model.rmtl("wear", tau=tau)
            lo[r, k], hi[r, k] = res["lower"], res["upper"]
    check_coverage(lo, hi, truth, 0.95, "CompetingRisks rmtl")


def test_rmtl_diff_size():
    # Two groups of 150 from the same design: the z-test of no difference
    # in the time lost to wear by tau = 100 should reject 5% of the time.
    from surpyval.univariate.competing_risks import CompetingRisks

    rng = np.random.default_rng(208)
    reps, n = 2000, 150
    rejected = 0
    for _ in range(reps):
        a = CompetingRisks.fit(*_wear_or_shock(rng, n))
        b = CompetingRisks.fit(*_wear_or_shock(rng, n))
        rejected += sp.rmtl_diff(a, b, "wear", tau=100.0)["p_value"] < 0.05
    check_rate(rejected, reps, 0.05, "rmtl_diff size at 5%")


def test_parametric_competing_risks_rmtl_coverage():
    # The parametric fit's delta-method interval (on the logit of rmtl /
    # tau) with a Weibull for each cause, correctly specified (the shock's
    # exponential is a Weibull of shape 1), n = 200, tau = 100 (true
    # 13.37).
    from surpyval.univariate.competing_risks import ParametricCompetingRisks

    rng = np.random.default_rng(209)
    tau = 100.0
    truth = _true_wear_rmtl(tau)
    reps, n = 500, 200
    lo, hi = np.empty(reps), np.empty(reps)
    for r in range(reps):
        x, e = _wear_or_shock(rng, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = ParametricCompetingRisks.fit(x, e, dist=sp.Weibull)
        res = model.rmtl("wear", tau=tau)
        lo[r], hi[r] = res["lower"], res["upper"]
    check_coverage(
        lo[:, None], hi[:, None], truth, 0.95, "ParametricCompetingRisks rmtl"
    )
