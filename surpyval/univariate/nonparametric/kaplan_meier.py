import numpy as np
import numpy.typing as npt

from surpyval.univariate.nonparametric.fleming_harrington import _snap
from surpyval.univariate.nonparametric.nonparametric_fitter import (
    NonParametricFitter,
)


def greenwood_variance(r: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """
    Greenwood's formula for the variance of the cumulative hazard
    (equivalently, of -log(R)) of the Kaplan-Meier estimator:

    Var(H) = sum(d / (r * (r - d)))

    Where d == r (i.e. the survival function reaches zero) the variance
    is undefined and NaN is returned at, and after, that point.
    """
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    with np.errstate(all="ignore"):
        var = d / (r * (r - d))
        # The Turnbull EM hands over expected counts, often fractional and
        # carrying round-off: at the last value r - d came out as ~1e-15
        # (of either sign) instead of 0, so that one term was ~1e14 and the
        # bounds there meaningless, and a zero count came out as ~1e-16
        # where the estimate is still 1. Judge the proportion failing,
        # d / r, rather than the difference: snapped to a whole number up
        # to round-off, it is exactly 1 for the undefined d == r case and
        # exactly 0 for no events, whatever the scale of the counts.
        q = np.array([_snap(v) for v in d / r])
        var = np.where(q == 1, np.nan, np.where(q == 0, 0.0, var))
        var = np.where(np.isfinite(var), var, np.nan)
        return np.cumsum(var)


def kaplan_meier(r: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    # d cannot exceed r, so a negative factor is round-off in the Turnbull
    # EM's expected counts (d = r + 4e-15 at the last value); left in, it
    # made the survival there -2e-16 rather than 0.
    factor = np.maximum(1 - (d / r), 0.0)
    R = factor.copy()
    R[np.isnan(R)] = 0
    old_err_state = np.seterr(under="raise")

    try:
        R = np.cumprod(R)
    except FloatingPointError:
        R = np.cumsum(np.log(factor))
        R = np.exp(R)

    np.seterr(**old_err_state)
    return R


class KaplanMeier_(NonParametricFitter):
    r"""
    Kaplan-Meier estimator class. Calculates the Non-Parametric
    estimate of the survival function using:

    .. math::
        R(x) = \prod_{i:x_{i} \leq x}^{}
            \left ( 1 - \frac{d_{i} }{r_{i}}  \right )

    The variance of the cumulative hazard used for confidence bounds is
    estimated with Greenwood's formula:

    .. math::
        \widehat{Var}(H(x)) = \sum_{i:x_{i} \leq x}
            \frac{d_{i}}{r_{i} \left ( r_{i} - d_{i} \right )}

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import KaplanMeier
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = KaplanMeier.fit(x)
    >>> model.R
    array([0.8, 0.6, 0.4, 0.2, 0. ])
    """

    def __init__(self) -> None:
        self.how = "Kaplan-Meier"


KaplanMeier = KaplanMeier_()
