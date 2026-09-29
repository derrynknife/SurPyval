import numpy as np
import numpy.typing as npt
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


def _snap(v: float) -> float:
    """``v`` rounded to the nearest integer when it is one up to round-off.

    The Turnbull EM hands these functions expected counts that are whole
    numbers plus round-off (``1 + 2e-16``). ``ceil`` of such a count adds a
    ladder step with a risk set of about ``1e-16``, so the hazard of a
    single death doubled (and ``d = r = 3`` gave 2.83 instead of 1.83).
    """
    nearest = float(np.round(v))
    if abs(v - nearest) <= 1e-9 * max(1.0, abs(nearest)):
        return nearest
    return float(v)


def _snap_array(v: npt.ArrayLike) -> npt.NDArray:
    """``_snap`` applied elementwise, vectorised for use inside the EM."""
    v = np.asarray(v, dtype=float)
    nearest = np.round(v)
    with np.errstate(invalid="ignore"):
        close = np.abs(v - nearest) <= 1e-9 * np.maximum(1.0, np.abs(nearest))
    return np.where(close, nearest, v)


def _check_at_risk(
    r: npt.ArrayLike, d: npt.ArrayLike
) -> tuple[npt.NDArray, npt.NDArray]:
    """``r`` and ``d`` as float arrays, refused where a step has events but
    no one at risk: the proportion failing, ``d / r``, is then undefined.
    (A step with neither, 0 / 0, is valid: nothing happens there.)"""
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    bad = (d > 0) & ~(r > 0)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise ValueError(
            "A step has events but no one at risk (d = {} with r = {} at "
            "index {}); the number at risk must be positive wherever "
            "there are events.".format(d[i], r[i], i)
        )
    return r, d


def _ladder_steps(r_i: float, d_i: float) -> int:
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


def fh_h(r_i: float, d_i: float) -> float:
    # sum(1 / (r - i) for i in 0 ... ceil(d) - 2) + (d - full) / (r - full):
    # each of the d tied events sees a risk set that shrinks by one,
    # with the fractional remainder of d contributing pro rata.
    r_i, d_i = _snap(r_i), _snap(d_i)
    if d_i == 0:
        return 0.0  # no deaths, no hazard (whatever the risk set)
    if not r_i > 0:
        # Deaths with no one at risk (only round-off in the Turnbull EM
        # reaches here; the public functions refuse it): used to raise
        # ZeroDivisionError.
        return np.inf
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


def fh_var_h(r_i: float, d_i: float) -> float:
    # Variance increment with the same tie-splitting as fh_h, i.e.
    # each of the d tied events contributes 1/r**2 with a risk set
    # that shrinks by one for each event.
    r_i, d_i = _snap(r_i), _snap(d_i)
    if d_i == 0:
        return 0.0  # no deaths, no hazard (whatever the risk set)
    if not r_i > 0:
        # Deaths with no one at risk (only round-off in the Turnbull EM
        # reaches here; the public functions refuse it): used to raise
        # ZeroDivisionError.
        return np.inf
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


def fleming_harrington_variance(r: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    """
    Variance of the Fleming-Harrington cumulative hazard estimator
    using the same tie correction as the estimator itself:

    Var(H) = sum(sum(1 / (r - i)**2 for i in 0 ... d - 1))

    This is the variance used by R's ``survfit`` with ``ctype=2`` and
    reduces to the Nelson-Aalen (Aalen/Poisson) variance, sum(d / r**2),
    when there are no tied events.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.nonparametric import (
    ...     fleming_harrington_variance,
    ... )
    >>> r = np.array([10, 8, 5])
    >>> d = np.array([2, 1, 3])
    >>> fleming_harrington_variance(r, d).round(4)
    array([0.0223, 0.038 , 0.2516])
    """
    with np.errstate(all="ignore"):
        var = np.array([fh_var_h(r_i, d_i) for r_i, d_i in zip(r, d)])
        var = np.where(np.isfinite(var), var, np.nan)
        return np.cumsum(var)


def fleming_harrington(r: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    r"""
    Fleming-Harrington estimate of the survival function from the number
    at risk and the number of events at each time. It is the Nelson-Aalen
    estimate with ties counted one after another, each of the ``d``
    events at a time removing one item from the risk set before the next:

    .. math::
        R(x_i) = e^{-\sum_{j \leq i} \sum_{k=0}^{d_j-1}
            \frac{1}{r_j - k}}

    A fractional count (from the Turnbull EM) contributes its remainder
    pro rata. With no ties this is the Nelson-Aalen estimate.

    This is the low-level function behind :code:`FlemingHarrington.fit()`,
    which builds ``r`` and ``d`` from the data (see
    :code:`surpyval.xcnt_to_xrd`) and wraps the result in a
    ``NonParametric`` model; use that unless you already have the
    counts.

    Parameters
    ----------
    r : array_like
        Number of items at risk just before each distinct event time,
        in time order.
    d : array_like
        Number of events at each of those times. May be fractional.

    Returns
    -------
    R : ndarray
        The survival estimate just after each time, the same length as
        ``r``. Like the Nelson-Aalen estimate it stays above zero, even
        when all the items at risk fail at once (``fleming_harrington([3],
        [3])`` is ``exp(-(1/3 + 1/2 + 1))``, 0.1599). A step with no
        events leaves it unchanged, even with no one at risk (0 / 0).

    Raises
    ------
    ValueError
        If a step has events but no one at risk (``d > 0`` where ``r``
        is not positive).

    Examples
    --------
    The ties at the first and last times make the estimate lower than
    the Nelson-Aalen one (0.8187, 0.7225, 0.3965):

    >>> import numpy as np
    >>> from surpyval.univariate.nonparametric import fleming_harrington
    >>> r = np.array([10, 8, 5])
    >>> d = np.array([2, 1, 3])
    >>> fleming_harrington(r, d).round(4)
    array([0.8097, 0.7145, 0.3265])
    """
    r, d = _check_at_risk(r, d)
    return _fleming_harrington(r, d)


def _fleming_harrington(r: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
    # ``fleming_harrington`` without the check of its counts, for the
    # Turnbull EM, whose expected counts carry round-off.
    Y = np.array([fh_h(r_i, d_i) for r_i, d_i in zip(r, d)], dtype=float)
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

        R(x) = e^{-\sum_{i:x_{i} \leq x} \sum_{j=0}^{d_i-1}
            \frac{1}{r_i - j}}

    That is, the ``d_i`` deaths tied at ``x_i`` are counted one after
    another, each removing one unit from the risk set before the next
    (a fractional expected count, from the Turnbull EM, contributes its
    remainder pro rata). With no ties this is the Nelson-Aalen estimate.

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

    def __init__(self) -> None:
        self.how = "Fleming-Harrington"


FlemingHarrington = FlemingHarrington_()
