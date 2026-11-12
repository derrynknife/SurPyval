"""Coverage of the landmark supermodel's robust intervals.

The truth: ``N = 300`` units with a covariate ``z1`` (0 or 1, even odds)
and ``z2`` (standard normal), fixed over time, and a Weibull
proportional-hazards life: cumulative baseline hazard ``(t / 5)^1.5``,
coefficients 0.7 and -0.4. Each unit is followed for a uniform (3, 12).
Conditional on surviving to any landmark the hazards stay proportional
with the same coefficients, so the supermodel over the landmarks 0, 1, 2
and 3 with the window 3 and effects linear in the landmark is correctly
specified: the constant effects are 0.7 and -0.4, their slopes 0, and a
pooled baseline's term 0. A unit is at risk at up to four landmarks, so
its rows are correlated, which the robust standard errors allow for.

Checked over 400 replicates, for the stratified and the pooled
supermodel: the robust Wald intervals of the four effect coefficients,
and the robust interval (log-log, the default) on the dynamic prediction
``P(T <= 4 | T > 1, z1 = 1, z2 = 0.5)``, whose truth is ``1 - exp(-(H0(4)
- H0(1)) exp(0.7 - 0.2))``. The seed, replicates, sizes and slack (0.01,
the suite's rule) were fixed before any run.
"""

import os
import warnings

import numpy as np

import surpyval as sp
from surpyval.tests.calibration._montecarlo import (
    check_coverage,
    process_pool,
)

N = 300
REPS = 400
SEED = 26111
BETA = np.array([0.7, -0.4])
AT, W = [0, 1, 2, 3], 3.0
S_QUERY, Z_QUERY = 1.0, np.array([1.0, 0.5])
TRUTH_COEF = np.array([0.7, -0.4, 0.0, 0.0])


def _H0(t):
    return (t / 5.0) ** 1.5


def _true_window():
    H = (_H0(S_QUERY + W) - _H0(S_QUERY)) * np.exp(Z_QUERY @ BETA)
    return float(-np.expm1(-H))


def _sample(r):
    rng = np.random.default_rng([SEED, r])
    Z = np.column_stack(
        [rng.integers(0, 2, N).astype(float), rng.standard_normal(N)]
    )
    # H0(T) exp(Z beta) is standard exponential
    T = 5.0 * (rng.exponential(size=N) / np.exp(Z @ BETA)) ** (1 / 1.5)
    end = rng.uniform(3.0, 12.0, N)
    return sp.SurpyvalData(x=np.minimum(T, end), c=(T > end).astype(int), Z=Z)


def _one(r):
    life = _sample(r)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for baseline in ("stratified", "pooled"):
            model = sp.LandmarkCox.fit(
                life, at=AT, horizon=W, baseline=baseline, degree=1
            )
            b = model.params[:4]
            se = model.standard_errors()[:4]
            band = model.cb(S_QUERY + W, Z_QUERY, s=S_QUERY)
            out.append(
                (b - 1.959963984540054 * se, b + 1.959963984540054 * se, band)
            )
    return out


def test_coverage_of_the_robust_intervals():
    with process_pool(max_workers=min(os.cpu_count() or 1, 2)) as pool:
        out = list(pool.map(_one, range(REPS)))
    for k, baseline in enumerate(("stratified", "pooled")):
        lo = np.array([o[k][0] for o in out])
        hi = np.array([o[k][1] for o in out])
        band = np.array([o[k][2] for o in out])
        check_coverage(
            lo, hi, TRUTH_COEF, 0.95, f"LandmarkCox[{baseline}] coefficients"
        )
        check_coverage(
            band[:, :1],
            band[:, 1:],
            _true_window(),
            0.95,
            f"LandmarkCox[{baseline}] P(T <= 4 | T > 1)",
        )
