import numpy as np

from surpyval.univariate.nonparametric.nonparametric_fitter import (
    NonParametricFitter,
)


def fh_h(r_i, d_i):
    out = 0
    while d_i > 1:
        out += 1.0 / r_i
        r_i -= 1
        d_i -= 1
    out += d_i / r_i
    return out


def fh_var_h(r_i, d_i):
    # Variance increment with the same tie-splitting as fh_h, i.e.
    # each of the d tied events contributes 1/r**2 with a risk set
    # that shrinks by one for each event.
    out = 0
    while d_i > 1:
        out += 1.0 / r_i**2
        r_i -= 1
        d_i -= 1
    out += d_i / r_i**2
    return out


def fleming_harrington_variance(r, d):
    """
    Variance of the Fleming-Harrington cumulative hazard estimator
    using the same tie correction as the estimator itself:

    Var(H) = sum(sum(1 / (r - i)**2 for i in 0 ... d - 1))

    This is the variance used by R's ``survfit`` with ``ctype=2`` and
    reduces to the Nelson-Aalen (Aalen/Poisson) variance, sum(d / r**2),
    when there are no tied events.
    """
    with np.errstate(all="ignore"):
        var = np.array([fh_var_h(r_i, d_i) for r_i, d_i in zip(r, d)])
        var = np.where(np.isfinite(var), var, np.nan)
        return np.cumsum(var)


def fleming_harrington(r, d):
    Y = np.array([fh_h(r_i, d_i) for r_i, d_i in zip(r, d)])
    H = Y.cumsum()
    H[np.isnan(H)] = np.inf
    R = np.exp(-H)
    return R


class FlemingHarrington_(NonParametricFitter):
    r"""
    Fleming-Harrington estimation of survival distribution.
    Returns a `NonParametric` object from method :code:`fit()`
    calculates the Non-Parametric estimate of the survival function using:

    .. math::

        R = e^{-\sum_{i:x_{i} \leq x} \sum_{i=0}^{d_x-1} \frac{1}{r_x - i}}

    See 'NonParametric section for detailed estimate of how H is computed.'

    The variance of the cumulative hazard used for confidence bounds is
    estimated with the same tie correction as the estimator itself
    (as used by R's ``survfit`` with ``ctype=2``):

    .. math::
        \widehat{Var}(H(x)) = \sum_{i:x_{i} \leq x}
            \sum_{j=0}^{d_i-1} \frac{1}{\left ( r_i - j \right )^{2}}

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import FlemingHarrington
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = FlemingHarrington.fit(x)
    >>> model.R
    array([0.81873075, 0.63762815, 0.45688054, 0.27711205, 0.10194383])
    """

    def __init__(self):
        self.how = "Fleming-Harrington"


FlemingHarrington = FlemingHarrington_()
