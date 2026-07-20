import numpy as np
from scipy.special import digamma, polygamma

from surpyval.univariate.nonparametric.nonparametric_fitter import (
    NonParametricFitter,
)

# The tie-splitting ladder is evaluated with an exact term-by-term sum
# for ordinary tie counts, and in closed form (digamma / trigamma
# harmonic sums) beyond this, so the cost is O(1) in the event count.
# The Turnbull EM feeds these functions *fractional expected* counts
# which, under heavy truncation, can grow without bound between EM
# iterations -- a per-event Python loop then never returns (this hung
# ReadTheDocs builds), while the closed form stays instant.
_MAX_TIE_LOOP = 64


def _ladder_steps(r_i, d_i):
    """Number of whole 1/r terms in the tie ladder, or -1 if the
    ladder exhausts the risk set (the hazard diverges)."""
    if np.isnan(d_i) or d_i <= 1:
        return 0
    if not np.isfinite(d_i):
        return -1
    full = int(np.ceil(d_i)) - 1
    if full >= r_i:
        return -1
    return full


def fh_h(r_i, d_i):
    # sum(1 / (r - i) for i in 0 ... ceil(d) - 2) + (d - full) / (r - full):
    # each of the d tied events sees a risk set that shrinks by one,
    # with the fractional remainder of d contributing pro rata.
    full = _ladder_steps(r_i, d_i)
    if full < 0:
        return np.inf
    if full <= _MAX_TIE_LOOP:
        out = 0.0
        for _ in range(full):
            out += 1.0 / r_i
            r_i -= 1.0
        return out + (d_i - full) / r_i
    out = float(digamma(r_i + 1.0) - digamma(r_i - full + 1.0))
    return out + (d_i - full) / (r_i - full)


def fh_var_h(r_i, d_i):
    # Variance increment with the same tie-splitting as fh_h, i.e.
    # each of the d tied events contributes 1/r**2 with a risk set
    # that shrinks by one for each event.
    full = _ladder_steps(r_i, d_i)
    if full < 0:
        return np.inf
    if full <= _MAX_TIE_LOOP:
        out = 0.0
        for _ in range(full):
            out += 1.0 / r_i**2
            r_i -= 1.0
        return out + (d_i - full) / r_i**2
    out = float(polygamma(1, r_i - full + 1.0) - polygamma(1, r_i + 1.0))
    return out + (d_i - full) / (r_i - full) ** 2


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
