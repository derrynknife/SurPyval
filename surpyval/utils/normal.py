"""
The normal distribution's functions from ``scipy.special``, differentiable
by autograd.

``scipy.stats.norm`` (and autograd's wrapper of it, which the fits used)
spends most of a call checking, broadcasting and placing its arguments in
the generic ``rv_continuous`` machinery: 4 to 7 times the cost of the
maths (#469). Once those checks are done it evaluates exactly what is
here -- ``ndtr``, ``log_ndtr`` and ``ndtri`` of :math:`z = (x - \\mu) /
\\sigma`, and the density as ``exp(-z**2 / 2) / sqrt(2 pi)`` -- so the
values are the same to the last bit, including the tails (``log_ndtr``
is accurate far into them) and ``nan`` for a non-positive or ``nan``
scale. Importing it does not import ``scipy.stats`` (#470).

``ndtr`` and ``log_ndtr`` are autograd primitives with their derivatives
(:math:`\\phi(z)` and :math:`\\phi(z) / \\Phi(z)`, the latter from the
logs so that it stays finite where both underflow), so the
likelihoods built from these functions keep their exact gradients and
Hessians. ``ppf`` is not differentiable, as ``scipy.stats.norm.ppf`` was
not.

Examples
--------
>>> import numpy as np
>>> from surpyval.utils import normal
>>> normal.sf(np.array([0.0, 1.0, 30.0]))
array([5.00000000e-001, 1.58655254e-001, 4.90671393e-198])
>>> normal.logsf(40.0)
np.float64(-804.6084420137539)
>>> normal.ppf(0.975, 10.0, 2.0)
np.float64(13.919927969080108)
"""

from typing import Any

import autograd.numpy as np
from autograd.extend import defvjp, primitive
from scipy import special

# scipy.stats' constants, so that the density is the same to the last bit
_NORM_PDF_C = float(np.sqrt(2 * np.pi))
_NORM_PDF_LOGC = float(np.log(_NORM_PDF_C))


def _std_logpdf(z: Any) -> Any:
    # z**2 overflows to inf beyond |z| ~ 1e154, where the log density is
    # -inf and the density 0, as they should be: no raw numpy warning
    # (principle 22), which scipy.stats let through.
    with np.errstate(over="ignore"):
        return -(z**2) / 2.0 - _NORM_PDF_LOGC


def _std_pdf(z: Any) -> Any:
    with np.errstate(over="ignore"):
        return np.exp(-(z**2) / 2.0) / _NORM_PDF_C


ndtr = primitive(special.ndtr)
log_ndtr = primitive(special.log_ndtr)
defvjp(ndtr, lambda ans, z: lambda g: g * _std_pdf(z))
defvjp(
    log_ndtr,
    lambda ans, z: lambda g: g * np.exp(_std_logpdf(z) - log_ndtr(z)),
)


def _scale(scale: Any) -> Any:
    """``scale``, ``nan`` where it is not positive (scipy's bad value)."""
    if np.all(np.asarray(scale > 0)):
        return scale
    return np.where(scale > 0, scale, np.nan)


def pdf(x: Any, loc: Any = 0.0, scale: Any = 1.0) -> Any:
    """The density at ``x``."""
    scale = _scale(scale)
    return _std_pdf((x - loc) / scale) / scale


def logpdf(x: Any, loc: Any = 0.0, scale: Any = 1.0) -> Any:
    """The log of the density at ``x``."""
    scale = _scale(scale)
    return _std_logpdf((x - loc) / scale) - np.log(scale)


def cdf(x: Any, loc: Any = 0.0, scale: Any = 1.0) -> Any:
    """:math:`\\Phi((x - \\mu) / \\sigma)`."""
    return ndtr((x - loc) / _scale(scale))


def sf(x: Any, loc: Any = 0.0, scale: Any = 1.0) -> Any:
    """:math:`1 - \\Phi((x - \\mu) / \\sigma)`, as ``ndtr(-z)``: exact in
    the upper tail, where ``1 - cdf`` is 0."""
    return ndtr(-((x - loc) / _scale(scale)))


def logcdf(x: Any, loc: Any = 0.0, scale: Any = 1.0) -> Any:
    """The log of :func:`cdf`, finite where it underflows."""
    return log_ndtr((x - loc) / _scale(scale))


def logsf(x: Any, loc: Any = 0.0, scale: Any = 1.0) -> Any:
    """The log of :func:`sf`, finite where it underflows."""
    return log_ndtr(-((x - loc) / _scale(scale)))


def ppf(q: Any, loc: Any = 0.0, scale: Any = 1.0) -> Any:
    """The quantile at probability ``q``: ``-inf`` at 0, ``inf`` at 1,
    ``nan`` outside [0, 1] and for a non-positive scale."""
    return special.ndtri(q) * _scale(scale) + loc


def isf(q: Any, loc: Any = 0.0, scale: Any = 1.0) -> Any:
    """The quantile at survival probability ``q``."""
    return -special.ndtri(q) * _scale(scale) + loc
