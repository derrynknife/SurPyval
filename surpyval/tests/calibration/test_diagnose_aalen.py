"""``diagnose()`` of Aalen's additive hazards model on simulated data.

Three designs, each 200 replicates, fitted with ``AalenAdditive.fit`` and
diagnosed with ``n_boot=200`` resampled processes (replicate ``r`` seeds
its test with ``random_state=r``), judged at the 5% level:

- **Null** (seed 28801, n = 300): ``h(t | z) = 0.1 + 0.08 z1 + 0.1 z2``,
  ``z1`` binary and ``z2`` uniform on (0, 1), follow-up ending at a
  uniform time on (2, 14) and at 12. Every effect is constant and every
  hazard positive. The constant-effects rows must reject about 5% of the
  time for each covariate (slack 0.02, as for the model's own test at
  this size), and the falling-hazard test must reject at most 5% of the
  time (slack 0.02).
- **Changing effect** (seed 28802, n = 500): ``z2``'s effect is 0.5 to
  t = 4 and 0.1 after. Its constant-effects row must reject at least half
  the time.
- **Negative fitted hazards** (seed 28803, n = 300): one covariate ``z``
  uniform on (0, 1) with the convex hazard ``0.02 + 0.6 (1 - z)^4``,
  constant in time. The additive model's straight line in ``z`` gives the
  units near ``z = 1`` a negative hazard; the falling-hazard test must
  reject at least half the time.

The designs, seeds and rules were fixed before the first run. Results
(first run of each): sizes 0.035 (z1) and 0.030 (z2); the falling-hazard
test rejects 0.000 under the null (it is conservative by construction:
its null is a zero hazard) and 0.900 with the convex hazard; the power
against z2's changing effect is 0.335, short of its rule (a strict xfail,
see ``XFAIL``).

The falling-hazard check was first a pointwise screen (a fall of more
than 1.96 standard errors of the increment since the running maximum, by
more than 5% of the units); on the null design it warned in 0.735 of
the replicates, so it was replaced by the resampled drawdown test before
this file was committed. The designs, seeds and rules did not change.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_rate

REPS = 200
N_BOOT = 200
XFAIL = (
    "diagnose() of AalenAdditive at n = 500: the power of the "
    "constant-effects test against z2's effect of 0.5 to t = 4 and 0.1 "
    "after is 0.335, short of the 0.5 set before the run. The test's null "
    "is deliberately wide (it multiplies the events, not the martingale "
    "residuals, which over-reject at these sizes), and the price is power; "
    "with 1000 units such an effect is found (test_aalen_additive.py)"
)


def _two_covariates(rng, n, early_z2):
    z1 = rng.binomial(1, 0.5, n).astype(float)
    z2 = rng.uniform(0.0, 1.0, n)
    early = 0.1 + 0.08 * z1 + early_z2 * z2
    late = 0.1 + 0.08 * z1 + 0.1 * z2
    e = rng.exponential(1.0, n)
    t = np.where(e < 4.0 * early, e / early, 4.0 + (e - 4.0 * early) / late)
    end = np.minimum(rng.uniform(2.0, 14.0, n), 12.0)
    return np.minimum(t, end), (t > end).astype(int), np.c_[z1, z2]


def _convex(rng, n):
    z = rng.uniform(0.0, 1.0, n)
    t = rng.exponential(1.0 / (0.02 + 0.6 * (1.0 - z) ** 4))
    end = np.minimum(rng.uniform(2.0, 14.0, n), 12.0)
    return np.minimum(t, end), (t > end).astype(int), z[:, None]


def _diagnose(x, c, Z, r):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.AalenAdditive.fit(x, Z, c=c)
        return model.diagnose(n_boot=N_BOOT, random_state=r)


def _p(d, name):
    table = d.table
    row = table[
        (table["check"] == "constant_effects")
        & (table["covariates"] == (name,))
    ]
    return float(row["p"].iloc[0])


def _flagged(d):
    table = d.table
    row = table[table["check"] == "non_negative_hazard"]
    return float(row["p"].iloc[0]) < 0.05


def test_null_size_and_falls_test():
    rng = np.random.default_rng(28801)
    rejected = np.zeros(2)
    warned = 0
    for r in range(REPS):
        x, c, Z = _two_covariates(rng, 300, 0.1)
        d = _diagnose(x, c, Z, r)
        rejected += [_p(d, "coef_0") < 0.05, _p(d, "coef_1") < 0.05]
        warned += _flagged(d)
    check_rate(rejected[0], REPS, 0.05, "constant z1: size", slack=0.02)
    check_rate(rejected[1], REPS, 0.05, "constant z2: size", slack=0.02)
    check_rate(
        warned,
        REPS,
        0.05,
        "positive hazards: falls test",
        slack=0.02,
        side="upper",
    )


@pytest.mark.xfail(strict=True, reason=XFAIL)
def test_a_changing_effect_is_found():
    rng = np.random.default_rng(28802)
    rejected = 0
    for r in range(REPS):
        x, c, Z = _two_covariates(rng, 500, 0.5)
        rejected += _p(_diagnose(x, c, Z, r), "coef_1") < 0.05
    check_rate(
        rejected,
        REPS,
        0.5,
        "z2 0.5 then 0.1: power",
        slack=0.0,
        side="lower",
    )


def test_negative_fitted_hazards_are_flagged():
    rng = np.random.default_rng(28803)
    warned = 0
    for r in range(REPS):
        x, c, Z = _convex(rng, 300)
        warned += _flagged(_diagnose(x, c, Z, r))
    check_rate(
        warned, REPS, 0.5, "convex hazard: falls test", slack=0.0, side="lower"
    )
