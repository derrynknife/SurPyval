"""Coverage of the non-parametric pointwise bounds and simultaneous bands.

Three studies:

- **Pointwise** ``cb`` of Kaplan-Meier, Nelson-Aalen and Fleming-Harrington
  (default log(-log) bounds) at the true quartiles of a censored Weibull
  sample, n = 100.
- **Simultaneous** ``band`` (Hall-Wellner and equal precision): the whole
  true survival curve, over the range the band covers, must lie inside it.
  The equal precision band used to start at the first event and covered
  about 0.89, at n = 100 and at n = 400, almost all of its misses at the
  first few event times; it now starts where a = n var / (1 + n var)
  reaches 0.1 (#390).
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
        "KaplanMeier band({}) coverage".format(method),
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
    print(line + " (target 0.95, tolerance +/- {:.5f})".format(tol))
    assert abs(p - 0.95) <= tol, line
    # The design sees the old error: a value 1.5% low is outside tolerance.
    p_low = _inside_probability(
        0.985 * crit, a_l, a_u, standardized, n_paths, rng
    )
    print("  at 0.985 x critical value: P(inside) {:.5f}".format(p_low))
    assert p_low < 0.95 - tol
