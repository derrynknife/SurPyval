"""Coverage of the Wald bounds of the copulas of three series.

The Gaussian copula of three series, unstructured and exchangeable, and
the Student-t copula, fitted by ``how="IFM"`` (the Godambe covariance):
the 95% ``param_cb`` interval of each copula parameter and the 95% ``cb``
interval of the joint survival at one point. Data are drawn from the
copula with Weibull(10, 2), Weibull(20, 3) and Weibull(30, 1.5) margins,
n = 200, 200 replications, as the bivariate study
(``test_coverage_copula.py``); once more for the unstructured Gaussian
with each series right-censored at its 70% point (so that rows have one,
two and three censored series).

The design, the seeds and the slack (0.01, ``_montecarlo``: with 200
replications the band is the nominal 95% +- 5.6 points) were fixed before
any run; the coverages the studies reach are printed (``-rP``).

Study to run at consolidation: it was committed without having been run
(the package-wide calibration run before the release runs it first).
"""

import warnings

import numpy as np
import pytest

from surpyval import Weibull
from surpyval import multivariate as mv
from surpyval.tests.calibration._montecarlo import check_coverage

N = 200
REPS = 200
MARGINS = [
    Weibull.from_params([10.0, 2.0]),
    Weibull.from_params([20.0, 3.0]),
    Weibull.from_params([30.0, 1.5]),
]
POINT = np.array([[8.0, 18.0, 22.0]])

# family, structure, true parameters, seed
STUDIES = {
    "Gaussian": ("Gaussian", "unstructured", [0.5, 0.2, 0.4], 4651),
    "Gaussian exchangeable": ("Gaussian", "exchangeable", [0.4], 4652),
    "StudentT": ("StudentT", "unstructured", [0.5, 0.2, 0.4, 5.0], 4653),
    "Gaussian censored": ("Gaussian", "unstructured", [0.5, 0.2, 0.4], 4654),
}


@pytest.mark.parametrize("study", list(STUDIES))
def test_copula_of_three_series_bound_coverage(study):
    family, structure, truth, seed = STUDIES[study]
    copula = getattr(mv, family)
    true_model = copula.from_params(truth, MARGINS, structure=structure)
    rng = np.random.default_rng(seed)
    k = len(truth)
    lo, hi = np.empty((REPS, k)), np.empty((REPS, k))
    sf_lo, sf_hi = np.empty(REPS), np.empty(REPS)
    limit = np.array([m.qf(0.7) for m in MARGINS])
    for r in range(REPS):
        X = true_model.random(N, random_state=rng)
        fit_kw = {}
        if study.endswith("censored"):
            fit_kw["c"] = (X > limit).astype(int)
            X = np.minimum(X, limit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = copula.fit(
                X, margins=[Weibull] * 3, structure=structure, **fit_kw
            )
            bounds = [model.param_cb(name) for name in model.parameter_names]
            sf_lo[r], sf_hi[r] = model.cb(POINT)[0]
        lo[r], hi[r] = np.array(bounds).T
    check_coverage(lo, hi, np.asarray(truth), 0.95, study + " param_cb")
    sf = true_model.sf(POINT)[0]
    check_coverage(sf_lo, sf_hi, sf, 0.95, study + " cb(sf)")
