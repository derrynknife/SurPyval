"""Coverage of the cause-specific Cox model fitted by its full likelihood
(``CoxPH``'s route for interval-censored data), with interval censoring
and delayed entry: Wald intervals whose standard errors are the profile
likelihood's curvature.

The design, fixed before any run:

- ``N = 200`` units in each sample, with a binary covariate ``z1``
  (probability 0.5) and a standard normal ``z2``;
- two causes with exponential cause-specific hazards,
  ``0.10 exp(0.5 z1 - 0.3 z2)`` (cause ``a``) and
  ``0.07 exp(-0.4 z1 + 0.3 z2)`` (cause ``b``);
- delayed entry: half the units enter at 0, the other half at a
  uniform(0, 3) time, and a unit is in the sample only if no event has
  happened by its entry (units are drawn until there are ``N``);
- inspections from the entry time, every uniform(0.5, 1.5), until the end
  of follow-up, a uniform(10, 20) time; a failure is known only to lie
  between the inspection before it (or the entry) and the one after it,
  with its cause; a unit still working at its last inspection is right
  censored there. No exact times.

Wald ``param_cb`` at 90% of the four coefficients from ``CoxPH.fit(x, Z,
c=c, e=e, tl=entry)`` with its defaults, 300 replicates, seed 20261116,
and the suite's slack (0.01) and tolerance: each coverage must lie within
``Z_TOL`` binomial standard errors plus 0.01 of 0.90.
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage, process_pool

N = 200
REPS = 300
SEED = 20261116
LEVEL = 0.90
TRUTH = np.array([0.5, -0.3, -0.4, 0.3])  # a: z1, z2; b: z1, z2
RATES = (0.10, 0.07)


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    rows = []
    while len(rows) < N:
        z1 = float(rng.binomial(1, 0.5))
        z2 = float(rng.normal())
        h_a = RATES[0] * np.exp(TRUTH[0] * z1 + TRUTH[1] * z2)
        h_b = RATES[1] * np.exp(TRUTH[2] * z1 + TRUTH[3] * z2)
        t = rng.exponential(1.0 / (h_a + h_b))
        cause = "a" if rng.random() < h_a / (h_a + h_b) else "b"
        entry = 0.0 if rng.random() < 0.5 else rng.uniform(0.0, 3.0)
        end = rng.uniform(10.0, 20.0)
        if t <= entry:
            continue
        visits = entry + np.cumsum(rng.uniform(0.5, 1.5, 40))
        visits = np.r_[entry, visits[visits <= end]]
        if t > visits[-1]:
            rows.append(((visits[-1], visits[-1]), 1, None, entry, z1, z2))
        else:
            k = np.searchsorted(visits, t) - 1
            rows.append(((visits[k], visits[k + 1]), 2, cause, entry, z1, z2))
    x = np.array([row[0] for row in rows])
    c = np.array([row[1] for row in rows])
    e = np.array([row[2] for row in rows], dtype=object)
    tl = np.array([row[3] for row in rows])
    Z = np.array([[row[4], row[5]] for row in rows])
    return x, c, e, tl, Z


def _bounds(r):
    x, c, e, tl, Z = _sample(r)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.CoxPH.fit(x, Z, c=c, e=e, tl=tl)
        bounds = np.array(
            [
                model.param_cb(name, alpha_ci=1 - LEVEL)
                for name in model.parameter_names
            ]
        )
    return bounds[:, 0], bounds[:, 1]


def test_wald_coverage_with_interval_censoring_and_delayed_entry():
    with process_pool(max_workers=min(os.cpu_count() or 1, 4)) as pool:
        out = list(pool.map(_bounds, range(REPS)))
    lower = np.array([o[0] for o in out])
    upper = np.array([o[1] for o in out])
    check_coverage(lower, upper, TRUTH, LEVEL, "CoxPH NPMLE (causes) Wald")
