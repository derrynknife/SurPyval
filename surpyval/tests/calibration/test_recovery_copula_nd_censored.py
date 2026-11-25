"""A Gaussian copula of four series fitted to rows censored in every
series: the lattice rule of four or more censored coordinates in a fit.

Four Weibull series (scales 10, 20, 30 and 40, shape 2) joined by a
Gaussian copula with correlations (0.5, 0.3, 0.2, 0.4, 0.3, 0.5), n =
150, each row right-censored at its own time (a uniform(0.4, 1.6) multiple
of (10, 20, 30, 40)), so that 22 rows have all four series censored and
another 79 one to three. The fitted correlations are within three
standard errors of the truth (measured: 0.488, 0.194, 0.274, 0.337,
0.331, 0.426 with standard errors 0.067 to 0.088).

A calibration study (``--run-calibration``) because the fit and its
Godambe covariance take about ten minutes: the 22 rows' four-dimensional
integrals are evaluated a few thousand times. The design and seeds were
fixed before the run.
"""

import numpy as np

import surpyval as sp
from surpyval.multivariate import Gaussian


def test_the_fit_takes_rows_with_every_series_censored():
    margins = [sp.Weibull.from_params([10.0 * (d + 1), 2.0]) for d in range(4)]
    rho = np.array([0.5, 0.3, 0.2, 0.4, 0.3, 0.5])
    truth = Gaussian.from_params(rho, margins)
    X = truth.random(150, random_state=4721)
    limit = np.random.default_rng(4722).uniform(0.4, 1.6, (150, 1)) * (
        np.array([10.0, 20.0, 30.0, 40.0])
    )
    c = (X > limit).astype(int)
    X = np.minimum(X, limit)
    assert np.sum(c.sum(axis=1) == 4) == 22
    model = Gaussian.fit(X, c=c, margins=[sp.Weibull] * 4)
    se = model.standard_errors()
    assert np.all(np.abs(model.params - rho) < 3 * se), (model.params, se)
