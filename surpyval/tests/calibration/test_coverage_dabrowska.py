"""Coverage of the Dabrowska estimator's bounds on the joint survival.

The design, fixed before the first run: pairs drawn from a Clayton copula
(theta 2, Kendall's tau 0.5) with Weibull(10, 2) and Weibull(20, 3)
margins, as the copula studies (``test_coverage_copula.py``), each series
right-censored independently by a uniform time (on 0 to 33 and 0 to 60:
27% and 30% of the series censored). At n = 100 and n = 400 pairs, the
95% two-sided ``cb`` of the joint survival at two points, (8, 18) near
the joint median (true sf 0.382) and (4, 10) in the upper part (true sf
0.827), and its 90% two-sided bound (``alpha_ci=0.1``, the
lower bounds' level of the 90% one-sided):

- ``method="wald"`` (the influence-function standard error on the logit
  scale), 400 replications (the band is +- 4.3 points at 95%, +- 5.5
  at 90%);
- ``method="bootstrap"`` (the case bootstrap's percentile interval, 200
  resamples), 200 replications (+- 5.6 points at 95%, +- 7.4 at 90%).

The seeds are 7210 (Wald) and 7211 (bootstrap), plus the sample size.
Slack 0.01 (``_montecarlo``). The truth is the copula's closed-form
joint survival.
"""

import warnings

import numpy as np
import pytest

from surpyval import Weibull
from surpyval import multivariate as mv
from surpyval.tests.calibration._montecarlo import check_coverage

MARGINS = [Weibull.from_params([10.0, 2.0]), Weibull.from_params([20.0, 3.0])]
TRUE = mv.Clayton.from_params([2.0], MARGINS)
POINTS = np.array([[8.0, 18.0], [4.0, 10.0]])
LEVELS = (0.90, 0.95)
LIMITS = [33.0, 60.0]


def _draw(n, rng):
    X = TRUE.random(n, random_state=rng)
    limit = rng.uniform(0, LIMITS, size=(n, 2))
    c = (X > limit).astype(int)
    return np.minimum(X, limit), c


def _study(n, reps, seed, **kw):
    rng = np.random.default_rng(seed + n)
    bounds = {level: np.empty((reps, 2, len(POINTS))) for level in LEVELS}
    for r in range(reps):
        X, c = _draw(n, rng)
        boot = {"random_state": int(rng.integers(2**31))} if kw else {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = mv.Dabrowska.fit(X, c=c)
            for level in LEVELS:
                cb = model.cb(POINTS, alpha_ci=1 - level, **kw, **boot)
                bounds[level][r] = cb.T
    truth = TRUE.sf(POINTS)
    label = f"Dabrowska n={n} {kw.get('method', 'wald')}"
    for level in LEVELS:
        check_coverage(
            bounds[level][:, 0],
            bounds[level][:, 1],
            truth,
            level,
            f"{label} cb(sf) at {level:.0%}",
        )


def test_the_points_are_where_the_design_says():
    assert TRUE.sf(POINTS) == pytest.approx([0.382, 0.827], abs=5e-4)


@pytest.mark.parametrize("n", [100, 400])
def test_dabrowska_wald_coverage(n):
    _study(n, 400, 7210)


@pytest.mark.parametrize("n", [100, 400])
def test_dabrowska_bootstrap_coverage(n):
    _study(n, 200, 7211, method="bootstrap", n_boot=200)
