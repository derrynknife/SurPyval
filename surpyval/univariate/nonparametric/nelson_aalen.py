import numpy as np

from surpyval.univariate.nonparametric.nonparametric_fitter import (
    NonParametricFitter,
)


def nelson_aalen_variance(r, d):
    """
    Aalen's (Poisson) estimate of the variance of the Nelson-Aalen
    cumulative hazard estimator:

    Var(H) = sum(d / r**2)

    Recommended by Klein (1991) for its small sample performance.

    Klein, J. P. (1991), "Small sample moments of some estimators of
    the variance of the Kaplan-Meier and Nelson-Aalen estimators",
    Scandinavian Journal of Statistics, 18(4), 333-340.
    """
    with np.errstate(all="ignore"):
        var = d / r**2
        var = np.where(np.isfinite(var), var, np.nan)
        return np.cumsum(var)


def nelson_aalen(r, d):
    H = np.cumsum(d / r)
    H[np.isnan(H)] = np.inf
    R = np.exp(-H)
    return R


class NelsonAalen_(NonParametricFitter):
    r"""
    Nelson-Aalen estimator class. Returns a `NonParametric`
    object from method :code:`fit()` Calculates the Non-Parametric
    estimate of the survival function using:

    .. math::
        R(x) = e^{-\sum_{i:x_{i} \leq x}^{} \frac{d_{i} }{r_{i}}}

    The variance of the cumulative hazard used for confidence bounds is
    estimated with Aalen's (Poisson) estimator, as recommended by
    Klein (1991):

    .. math::
        \widehat{Var}(H(x)) = \sum_{i:x_{i} \leq x} \frac{d_{i}}{r_{i}^{2}}

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import NelsonAalen
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = NelsonAalen.fit(x)
    >>> model.R
    array([0.81873075, 0.63762815, 0.45688054, 0.27711205, 0.10194383])
    """

    def __init__(self):
        self.how = "Nelson-Aalen"


NelsonAalen = NelsonAalen_()
