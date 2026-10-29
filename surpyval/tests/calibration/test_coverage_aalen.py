"""Aalen's additive hazards model on data from a true additive model.

The hazard of a unit with a binary ``z1`` and a uniform ``z2`` is

    h(t | z) = 0.1 + 0.08 z1 + 0.1 z2 1(t < 4),

so ``z1``'s effect is constant and ``z2``'s stops at t = 4: the truth is
``B0(t) = 0.1 t``, ``B1(t) = 0.08 t`` and ``B2(t) = 0.1 min(t, 4)``.
Follow-up ends at a uniform time on (2, 14) and at 12. n = 300, 500
replicates, seed 8801.

- **Bias and standard errors** of ``B`` at t = 2, 6 and 10 (``check_bias``
  defaults: the least-squares increments are unbiased where the risk set
  separates the covariates).
- **Coverage** of the pointwise 95% Wald intervals of
  ``cumulative_coefficients_cb`` there. Slack 0.02: the intervals are
  large-sample, and the variance of a sum of least-squares increments
  over small late risk sets is estimated with error.
- **Size and power** of ``constant_effects_test`` (Kolmogorov-Smirnov,
  200 draws each) at the 5% level, on the first 200 replicates: the
  constant ``z1`` must be rejected about 5% of the time (slack 0.02, for
  the resampling approximation), the changing ``z2`` at least half the
  time. This is a strict xfail: the power fell short of the rule set
  before the run (see ``XFAIL``).

The design, seed and rules were fixed before the first run.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import (
    check_bias,
    check_coverage,
    check_rate,
)

N = 300
REPS = 500
TEST_REPS = 200
TIMES = np.array([2.0, 6.0, 10.0])
XFAIL = (
    "constant_effects_test at n = 300: the size here is 0.080 (inside the "
    "rule), but the power against z2's effect stopping at 4 is 0.145, "
    "short of the 0.5 set before the run; in a separate check (seed 1, 100 "
    "replicates of 200 draws) the size was 0.13 to 0.16 at n = 300 and "
    "0.05 at n = 1000, where the power against an effect of 0.4 was 0.91. "
    "The resampled null is right for large samples and too narrow for "
    "moderate ones, most likely from the few units at risk near the last "
    "event time, where B(tau) anchors the statistic"
)


def _truth(t):
    return np.column_stack([0.1 * t, 0.08 * t, 0.1 * np.minimum(t, 4.0)])


def _simulate(rng):
    z1 = rng.binomial(1, 0.5, N).astype(float)
    z2 = rng.uniform(0.0, 1.0, N)
    early = 0.1 + 0.08 * z1 + 0.1 * z2
    late = 0.1 + 0.08 * z1
    e = rng.exponential(1.0, N)
    # Piecewise exponential: the first 4 time units at the early rate.
    t = np.where(e < 4.0 * early, e / early, 4.0 + (e - 4.0 * early) / late)
    end = np.minimum(rng.uniform(2.0, 14.0, N), 12.0)
    return np.minimum(t, end), (t > end).astype(int), np.c_[z1, z2]


def test_aalen_bias_and_coverage():
    rng = np.random.default_rng(8801)
    truth = _truth(TIMES)
    est = np.empty((REPS, TIMES.size, 3))
    se = np.empty_like(est)
    lo = np.empty_like(est)
    hi = np.empty_like(est)
    for r in range(REPS):
        x, c, Z = _simulate(rng)
        model = sp.AalenAdditive.fit(x, Z, c=c)
        est[r] = model.cumulative_coefficients(TIMES).to_numpy()
        se[r] = np.sqrt(model._B_var_at(TIMES))
        band = model.cumulative_coefficients_cb(TIMES)
        lo[r], hi[r] = band[..., 0], band[..., 1]
    for k, t in enumerate(TIMES):
        check_bias(
            est[:, k],
            truth[k],
            f"Aalen B at t = {t:g}",
            standard_errors=se[:, k],
        )
        check_coverage(
            lo[:, k],
            hi[:, k],
            truth[k],
            0.95,
            f"Aalen 95% pointwise interval on B at t = {t:g}",
            slack=0.02,
        )


@pytest.mark.xfail(strict=True, reason=XFAIL)
def test_constant_effects_test_size_and_power():
    # The same data sets as the study above (the same seed and draws).
    rng = np.random.default_rng(8801)
    rejected = np.zeros(2)
    for r in range(TEST_REPS):
        x, c, Z = _simulate(rng)
        model = sp.AalenAdditive.fit(x, Z, c=c)
        p = model.constant_effects_test(n_sim=200, random_state=r)
        rejected += p["KS p"].to_numpy()[1:] < 0.05
    check_rate(rejected[0], TEST_REPS, 0.05, "constant z1: size", slack=0.02)
    check_rate(
        rejected[1],
        TEST_REPS,
        0.5,
        "z2 stops at 4: power",
        slack=0.0,
        side="lower",
    )
