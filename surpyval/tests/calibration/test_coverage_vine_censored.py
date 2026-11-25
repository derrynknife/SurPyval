"""Coverage of the Wald bounds of a vine copula fitted to rows censored in
its ends.

A D-vine of three series in the order 0, 1, 2 with a Clayton (theta = 2)
and a Gaussian (rho = 0.5) pair-copula in its first tree and a Frank
(theta = 3) in its second, Weibull(10, 1.5), Weibull(20, 2) and
Weibull(30, 2.5) margins, fitted by ``how="IFM"`` (the Godambe
covariance) with the families and order given: the 95% ``param_cb``
interval of each pair-copula parameter. The ends (series 0 and 2) are
right-censored at their 70% points, so rows have none, one or both ends
censored; n = 300, 200 replications.

The design, the seed and the slack (0.01, ``_montecarlo``) were fixed
before any run; the coverages are printed (``-rP``).
"""

import warnings

import numpy as np

from surpyval import Weibull
from surpyval.multivariate import Clayton, Frank, Gaussian, Vine
from surpyval.tests.calibration._montecarlo import check_coverage

N = 300
REPS = 200
MARGINS = [
    Weibull.from_params([10.0, 1.5]),
    Weibull.from_params([20.0, 2.0]),
    Weibull.from_params([30.0, 2.5]),
]
FAMILIES = [[Clayton, Gaussian], [Frank]]
TRUTH = np.array([2.0, 0.5, 3.0])


def test_vine_censored_in_its_ends_bound_coverage():
    truth = Vine.from_params(TRUTH, MARGINS, families=FAMILIES)
    rng = np.random.default_rng(4711)
    limit = np.array(
        [MARGINS[0].qf(0.7), np.inf, MARGINS[2].qf(0.7)], dtype=float
    )
    lo, hi = np.empty((REPS, 3)), np.empty((REPS, 3))
    for r in range(REPS):
        X = truth.random(N, random_state=rng)
        c = (X > limit).astype(int)
        X = np.minimum(X, limit)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Vine.fit(
                X,
                c=c,
                margins=[Weibull] * 3,
                order=[0, 1, 2],
                families=FAMILIES,
            )
            bounds = [model.param_cb(name) for name in model.parameter_names]
        lo[r], hi[r] = np.array(bounds).T
    check_coverage(lo, hi, TRUTH, 0.95, "censored vine param_cb")
