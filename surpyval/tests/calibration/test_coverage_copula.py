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
"""

import warnings

import numpy as np
import pytest

from surpyval import Weibull
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
