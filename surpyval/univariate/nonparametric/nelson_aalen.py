import numpy as np
import numpy.typing as npt

from surpyval.univariate.nonparametric.fleming_harrington import (
    _snap,
    _snap_array,
)
from surpyval.univariate.nonparametric.nonparametric_fitter import (
    NonParametricFitter,
)


def nelson_aalen_variance(r: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """
    Aalen's (Poisson) estimate of the variance of the Nelson-Aalen
    cumulative hazard estimator:

    Var(H) = sum(d / r**2)

    Recommended by Klein (1991) for its small sample performance.

    Klein, J. P. (1991), "Small sample moments of some estimators of
    the variance of the Kaplan-Meier and Nelson-Aalen estimators",
    Scandinavian Journal of Statistics, 18(4), 333-340.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.nonparametric import nelson_aalen_variance
    >>> r = np.array([10, 8, 5])
    >>> d = np.array([2, 1, 3])
    >>> nelson_aalen_variance(r, d).round(4)
    array([0.02  , 0.0356, 0.1556])
    """
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    with np.errstate(all="ignore"):
        var = d / r**2
        # A proportion failing that is 0 up to round-off (a Turnbull EM
        # expected count of ~1e-15) is no event, as in Greenwood's formula;
        # left in, it gave a variance where the estimate is still 1.
        q = np.array([_snap(v) for v in d / r])
        var = np.where(q == 0, 0.0, var)
        var = np.where(np.isfinite(var), var, np.nan)
        return np.cumsum(var)


def nelson_aalen(r: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    r"""
    Nelson-Aalen estimate of the survival function from the number at
    risk and the number of events at each time:

    .. math::
        R(x_i) = e^{-\sum_{j \leq i} \frac{d_{j}}{r_{j}}}

    This is the low-level function behind :code:`NelsonAalen.fit()`,
    which builds ``r`` and ``d`` from the data (see
    :code:`surpyval.xcnt_to_xrd`) and wraps the result in a
    ``NonParametric`` model; use that unless you already have the
    counts.

    Parameters
    ----------
    r : ndarray
        Number of items at risk just before each distinct event time,
        in time order.
    d : ndarray
        Number of events at each of those times. May be fractional (the
        Turnbull EM passes expected counts).

    Returns
    -------
    R : ndarray
        The survival estimate just after each time, the same length as
        ``r``. Unlike the Kaplan-Meier estimate it stays above zero when
        the last items at risk all fail; a step with no one at risk and
        no events (0 / 0) takes it to zero.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.nonparametric import nelson_aalen
    >>> r = np.array([10, 8, 5])
    >>> d = np.array([2, 1, 3])
    >>> nelson_aalen(r, d).round(4)
    array([0.8187, 0.7225, 0.3965])
    """
    # The Turnbull EM hands over expected counts carrying round-off. Past
    # the last event the risk set is 0 in exact arithmetic but came out as
    # 9e-16 on alternate iterations: 0 / 9e-16 = 0 kept the survival up
    # while 0 / 0 dropped it to zero, so the EM flipped between the two and
    # never converged. Counts that are whole numbers up to round-off are
    # taken as those whole numbers, as the Fleming-Harrington does.
    r = _snap_array(r)
    d = _snap_array(d)
    with np.errstate(divide="ignore", invalid="ignore"):
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

    def __init__(self) -> None:
        self.how = "Nelson-Aalen"


NelsonAalen = NelsonAalen_()
