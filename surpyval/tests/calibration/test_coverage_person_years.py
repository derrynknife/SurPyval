"""Coverage of the exact (Garwood) bounds of an exposure table:
``person_years``' cell rates, its rate ratio and its observed to expected
ratio.

The truth: units at two sites, A and B, enter at an age drawn uniformly
on (0, 5) and are followed for a time drawn uniformly on (1, 6), unless
they fail first. The hazard is piecewise constant in attained age,
``lambda_site * m(age)`` with ``m = 1`` below age 3, 2 on (3, 6] and 3
above, and ``lambda_A = 0.05``, ``lambda_B = 0.1``; 60 units per site.
The table is site by attained age band (0, 3], (3, 6], (6, 15], so each
cell's true rate is ``lambda_site * m``; the mean events per cell
are about 2 to 16 (2.4, 9.3 and 5.0 at A; 4.7, 16.3 and 6.9 at
B). The rate ratio of B to A over the whole table is 2, and with the
true rates as the reference rates the observed to expected ratio of the
whole table is 1.

The Garwood interval is exact: its coverage is *at least* the nominal
level for every true rate, and above it where the counts are small (a
discrete count cannot hit 95% exactly). So the check is one-sided: each
coverage must not fall below 95% by more than the suite's tolerance (3
Monte Carlo standard errors + 0.01); the coverages above it are printed
and reported, not failed. 2000 replicates; the seed, the replicates and
the tolerance were fixed before any run.

Result on 13 Nov 2026: cell coverages 0.987, 0.970, 0.975, 0.971, 0.962,
0.964; the rate ratio 0.972; the observed to expected ratio 0.957 (all at
least 0.95, as an exact interval's should be).
"""

import numpy as np

from surpyval import person_years
from surpyval.tests.calibration._montecarlo import Z_TOL, rate_se

SEED = 20261113
REPS = 2000
N = 60
LAMBDA = {"A": 0.05, "B": 0.1}
EDGES = np.array([0.0, 3.0, 6.0, 15.0])
MULT = np.array([1.0, 2.0, 3.0])
NOMINAL = 0.95


def _cumulative(age, lam):
    """The cumulative hazard from age 0 to ``age``."""
    widths = np.clip(age[:, None] - EDGES[None, :-1], 0, np.diff(EDGES))
    return lam * widths @ MULT


def _inverse(h, lam):
    """The age at which the cumulative hazard reaches ``h``."""
    cum = np.concatenate([[0.0], np.cumsum(lam * MULT * np.diff(EDGES))])
    j = np.clip(np.searchsorted(cum, h, side="right") - 1, 0, MULT.size - 1)
    return EDGES[j] + (h - cum[j]) / (lam * MULT[j])


def _sample(rng):
    x, c, site, age0 = [], [], [], []
    for s, lam in LAMBDA.items():
        start = rng.uniform(0, 5, N)
        follow = rng.uniform(1, 6, N)
        h = _cumulative(start, lam) + rng.exponential(size=N)
        life = _inverse(h, lam) - start
        x.append(np.minimum(life, follow))
        c.append((life > follow).astype(int))
        site += [s] * N
        age0.append(start)
    return np.concatenate(x), np.concatenate(c), site, np.concatenate(age0)


def _check(label, covered):
    cov = np.atleast_1d(covered.mean(axis=0))
    tol = Z_TOL * rate_se(NOMINAL, covered.shape[0]) + 0.01
    for k, v in enumerate(cov):
        print(
            "{} [{}]: coverage {:.4f}, nominal {} (at least {:.4f})".format(
                label, k, v, NOMINAL, NOMINAL - tol
            )
        )
    assert np.all(cov >= NOMINAL - tol), (label, cov)
    return cov


def test_garwood_bounds_cover_at_least_the_nominal_level():
    rng = np.random.default_rng(SEED)
    truth = np.array([LAMBDA[s] * m for s in "AB" for m in MULT])
    rates = {
        "site": [s for s in "AB" for _ in MULT],
        "age": [e for _ in "AB" for e in EDGES[:-1]],
        "rate": truth,
    }
    cells, ratio, oe, events = [], [], [], []
    for _ in range(REPS):
        x, c, site, age0 = _sample(rng)
        py = person_years(
            x,
            c,
            factors={"site": site},
            timescales={"age": (age0, EDGES)},
            rates=rates,
        )
        t = py.table
        cells.append((t["rate_lower"] <= truth) & (truth <= t["rate_upper"]))
        events.append(t["events"].to_numpy())
        rr = py.rate_ratio({"site": "B"}, {"site": "A"})
        ratio.append(rr["lower"] <= 2.0 <= rr["upper"])
        total = py.margins().iloc[0]
        oe.append(total["oe_lower"] <= 1.0 <= total["oe_upper"])
    print("mean events per cell:", np.mean(events, axis=0).round(2))
    _check("Garwood rate, site x age cell", np.array(cells, dtype=float))
    _check("rate ratio B / A", np.array(ratio, dtype=float))
    _check("observed / expected", np.array(oe, dtype=float))
