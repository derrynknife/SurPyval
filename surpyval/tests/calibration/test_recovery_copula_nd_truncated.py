"""A Gaussian copula of three series fitted to late entrants (left
truncation in every series): the truncated likelihood of more than two
series in a fit.

Three Weibull series (scales 10, 20 and 30, shape 2) joined by a Gaussian
copula with correlations (0.6, 0.3, 0.5); 1,200 units each enter
observation in each series at a uniform age up to that series' scale, and
are seen only if they outlive their entry in every series (582 are). The
fitted correlations are within three standard errors of the truth
(measured: 0.633, 0.273, 0.382 with standard errors 0.031, 0.047, 0.051;
a fit that ignores the truncation gives 0.591, 0.241, 0.362).

A calibration study (``--run-calibration``) because the fit takes about
seven minutes: every row's window is a three-dimensional rectangle. The
design and seeds were fixed before the run.
"""

import numpy as np

import surpyval as sp
from surpyval.multivariate import Gaussian


def test_the_fit_recovers_the_correlations_of_late_entrants():
    margins = [sp.Weibull.from_params([10.0 * (d + 1), 2.0]) for d in range(3)]
    rho = [0.6, 0.3, 0.5]
    truth = Gaussian.from_params(rho, margins)
    rng = np.random.default_rng(4734)
    X = truth.random(1200, random_state=4735)
    entry = rng.uniform(0.0, 1.0, X.shape) * np.array([10.0, 20.0, 30.0])
    keep = np.all(X > entry, axis=1)
    X, entry = X[keep], entry[keep]
    assert 250 < X.shape[0] < 700
    t = np.stack([entry, np.full(X.shape, np.inf)], axis=-1)
    model = Gaussian.fit(X, t=t, margins=[sp.Weibull] * 3)
    se = model.standard_errors()
    assert np.all(np.abs(model.params - rho) < 3 * se), (model.params, se)
