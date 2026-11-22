"""Coverage of the copula models' Wald bounds (#540).

Every built-in family with a parameter, fitted both ways (``how="IFM"``,
the two-stage fit with the Godambe covariance, and ``how="MLE"``, the
joint fit with the inverse joint Hessian): the 95% ``param_cb`` interval
of each copula parameter, and the 95% ``cb`` interval of the joint
survival at one point, which carries the margins' uncertainty too. Data
are drawn from the family with Weibull(10, 2) and Weibull(20, 3) margins,
n = 200, 200 replications (the verification of #291 ran the same number);
once more for the Clayton with 27% and 30% of the series right-censored.
The draws use the families' samplers, which ``multivariate/
test_sampling.py`` checks against independent references.

Slack 0.01 (``_montecarlo``): with 200 replications the band is the
nominal 95% +- 5.6 points; the coverages the studies reach are printed
(``-rP``).

With ``KaplanMeier`` margins the fit is the maximum pseudo-likelihood
estimate from the ranks. On complete data its ``param_cb`` is the Wald
interval from Genest, Ghoudi and Rivest's (1995) rank-based covariance:
its 90% and 95% intervals are studied for three families at n = 100 and
n = 400, 500 replications each (the bands: 90% +- 5.0 points, 95% +- 3.9).
On censored data (27% and 30% of the series, as above) the bound is the
case bootstrap's percentile interval (200 refits): the Clayton's
``param_cb`` and ``cb`` of the joint survival at n = 100, 200
replications. The seeds, sizes and replication counts were fixed before
any of these were run.
"""

import warnings

import numpy as np
import pytest

from surpyval import KaplanMeier, Weibull
from surpyval import multivariate as mv
from surpyval.tests.calibration._montecarlo import check_coverage

N = 200
REPS = 200
MARGINS = [Weibull.from_params([10.0, 2.0]), Weibull.from_params([20.0, 3.0])]
# A point near the joint median: sf about 0.4 to 0.5
POINT = np.array([[8.0, 18.0]])

# family, true parameters, seed
FAMILIES = {
    "Clayton": ([2.0], 540),
    "Gumbel": ([2.0], 541),
    "Frank": ([5.0], 542),
    "Gaussian": ([0.6], 543),
    "Joe": ([2.0], 544),
    "AMH": ([0.6], 545),
    "StudentT": ([0.6, 5.0], 546),
}


def _study(family, truth, seed, how, censored=False):
    copula = getattr(mv, family)
    true_model = copula.from_params(truth, MARGINS)
    rng = np.random.default_rng(seed)
    k = len(truth)
    lo, hi = np.empty((REPS, k)), np.empty((REPS, k))
    sf_lo, sf_hi = np.empty(REPS), np.empty(REPS)
    for r in range(REPS):
        X = true_model.random(N, random_state=rng)
        fit_kw = {}
        if censored:
            # Independent uniform censoring: 27% and 30% of the series
            limit = rng.uniform(0, [33.0, 60.0], size=(N, 2))
            fit_kw["c"] = (X > limit).astype(int)
            X = np.minimum(X, limit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = copula.fit(
                X, margins=[Weibull, Weibull], how=how, **fit_kw
            )
            bounds = [model.param_cb(name) for name in model.parameter_names]
            sf_lo[r], sf_hi[r] = model.cb(POINT)[0]
        lo[r], hi[r] = np.array(bounds).T
    label = f"{family} ({how}{', censored' if censored else ''})"
    check_coverage(lo, hi, np.asarray(truth), 0.95, label + " param_cb")
    sf = true_model.sf(POINT)[0]
    check_coverage(sf_lo, sf_hi, sf, 0.95, label + " cb(sf)")


@pytest.mark.parametrize("how", ["IFM", "MLE"])
@pytest.mark.parametrize("family", list(FAMILIES))
def test_copula_bound_coverage(family, how):
    truth, seed = FAMILIES[family]
    _study(family, truth, seed, how)


@pytest.mark.parametrize("how", ["IFM", "MLE"])
def test_copula_bound_coverage_censored(how):
    _study("Clayton", [2.0], 547, how, censored=True)


# -- non-parametric margins -----------------------------------------------
RANK_REPS = 500
# family, true parameter, seed (the sample size is added to it)
RANK_FAMILIES = {
    "Clayton": (2.0, 5200),
    "Gumbel": (2.0, 5300),
    "Frank": (5.0, 5400),
}
LEVELS = (0.90, 0.95)


@pytest.mark.parametrize("n", [100, 400])
@pytest.mark.parametrize("family", list(RANK_FAMILIES))
def test_rank_based_bound_coverage(family, n):
    truth, seed = RANK_FAMILIES[family]
    copula = getattr(mv, family)
    true_model = copula.from_params([truth], MARGINS)
    rng = np.random.default_rng(seed + n)
    bounds = {level: np.empty((RANK_REPS, 2)) for level in LEVELS}
    for r in range(RANK_REPS):
        X = true_model.random(n, random_state=rng)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = copula.fit(X, margins=[KaplanMeier, KaplanMeier])
            for level in LEVELS:
                bounds[level][r] = model.param_cb("theta", alpha_ci=1 - level)
    for level in LEVELS:
        check_coverage(
            bounds[level][:, 0],
            bounds[level][:, 1],
            truth,
            level,
            f"{family} (KaplanMeier margins, n={n}) param_cb",
        )


def test_case_bootstrap_coverage_censored():
    n, n_boot, truth = 100, 200, 2.0
    true_model = mv.Clayton.from_params([truth], MARGINS)
    rng = np.random.default_rng(5500)
    theta = {level: np.empty((REPS, 2)) for level in LEVELS}
    sf = {level: np.empty((REPS, 2)) for level in LEVELS}
    for r in range(REPS):
        X = true_model.random(n, random_state=rng)
        limit = rng.uniform(0, [33.0, 60.0], size=(n, 2))
        c = (X > limit).astype(int)
        X = np.minimum(X, limit)
        # One seed per replication: every bound of it is from one set of
        # refits
        seed = int(rng.integers(2**31))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = mv.Clayton.fit(X, c=c, margins=[KaplanMeier, KaplanMeier])
            for level in LEVELS:
                kw = {
                    "alpha_ci": 1 - level,
                    "method": "bootstrap",
                    "n_boot": n_boot,
                    "random_state": seed,
                }
                theta[level][r] = model.param_cb("theta", **kw)
                sf[level][r] = model.cb(POINT, **kw)[0]
    label = f"Clayton (KaplanMeier margins, censored, n={n}) bootstrap"
    for level in LEVELS:
        check_coverage(
            theta[level][:, 0],
            theta[level][:, 1],
            truth,
            level,
            label + " param_cb",
        )
        check_coverage(
            sf[level][:, 0],
            sf[level][:, 1],
            true_model.sf(POINT)[0],
            level,
            label + " cb(sf)",
        )
