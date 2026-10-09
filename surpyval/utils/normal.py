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
logs so that it stays finite where both underflow, and from a continued
fraction far below 0, where its derivatives are differences of nearly
equal numbers), so the likelihoods built from these functions keep their
exact gradients and Hessians however far into the tails (#710). ``ppf``
is not differentiable, as ``scipy.stats.norm.ppf`` was not.

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

from typing import Any, Callable

import autograd.numpy as np
import numpy as onp
from autograd.extend import defjvp, defvjp, primitive
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


#: Below this ``z`` the ratio phi / Phi and its derivatives are taken from
#: a continued fraction (:func:`_tail`).
_TAIL_CUT = -5.0
#: Terms of the continued fraction: converged to rounding from ``z = -5``
#: (20 are within 1e-13 there, and 10 from z = -12).
_TAIL_TERMS = 30


def _tail(t: Any) -> "tuple[Any, Any]":
    """``(gap, c)`` at ``t = -z > 5``: Laplace's continued fraction for
    Mills' ratio gives ``phi(z) / Phi(z) = t + gap`` with ``gap = 1 / (t
    + c)`` and ``c = 2 / (t + 3 / (t + ...))``, each without the
    cancellation of taking them as differences."""
    acc = t.copy()
    for k in range(_TAIL_TERMS, 2, -1):
        acc = t + k / acc
    c = 2.0 / acc
    return 1.0 / (t + c), c


def _by_tail(z: Any, near: Callable, far: Callable) -> Any:
    """``near(z)`` where ``z >= _TAIL_CUT`` (or is not finite), and
    ``far(-z)`` below it, elementwise."""
    z = onp.asarray(z, dtype=float)
    tail = z < _TAIL_CUT
    if not tail.any():
        return near(z)
    tail &= onp.isfinite(z)
    out = onp.empty_like(z)
    with onp.errstate(all="ignore"):
        out[~tail] = near(z[~tail])
        out[tail] = far(-z[tail])
    return out[()] if out.ndim == 0 else out


def _ratio_near(z: Any) -> Any:
    # From the logs, so that it stays finite where both underflow
    return onp.exp(_std_logpdf(z) - special.log_ndtr(z))


def ratio_raw(z: Any) -> Any:
    """``r = phi(z) / Phi(z)``, ``log_ndtr``'s derivative."""
    return _by_tail(z, _ratio_near, lambda t: t + _tail(t)[0])


def gap_raw(z: Any) -> Any:
    """``z + r``: ``-r'/r``, about ``-1 / z`` far below 0."""
    return _by_tail(z, lambda z: z + _ratio_near(z), lambda t: _tail(t)[0])


def _gap_slope_raw(z: Any) -> Any:
    """``1 - r (z + r)``, the derivative of :func:`gap_raw`: ``gap (c -
    gap)`` in the tail, where it is about ``1 / z^2``."""

    def far(t: Any) -> Any:
        gap, c = _tail(t)
        return gap * (c - gap)

    return _by_tail(z, lambda z: 1.0 - _ratio_near(z) * gap_raw(z), far)


ndtr = primitive(special.ndtr)
log_ndtr = primitive(special.log_ndtr)
# The derivatives of log_ndtr, each from its own primitive. Taken from the
# ratio as differences, the ratio's own rounding (``|z|`` times that of its
# log) swamped them far below 0: at z = -6e5 the first was 3e-5 out and the
# second, -1 to 1e-12, came out as +-7e6, and a LogNormal regression with
# sigma at 2e-7 had a gradient of 0.04 and a Hessian of 3e10 that were
# rounding (#710).
_ratio = primitive(ratio_raw)
_gap = primitive(gap_raw)
_gap_slope = primitive(_gap_slope_raw)
defvjp(ndtr, lambda ans, z: lambda g: g * _std_pdf(z))
defvjp(log_ndtr, lambda ans, z: lambda g: g * _ratio(z))
# The same first derivatives in forward mode, which the Wald confidence
# bounds take their Jacobian in (one pass per parameter)
defjvp(ndtr, lambda g, ans, z: g * _std_pdf(z))
defjvp(log_ndtr, lambda g, ans, z: g * _ratio(z))
defvjp(_ratio, lambda ans, z: lambda g: -g * ans * _gap(z))
defvjp(_gap, lambda ans, z: lambda g: g * _gap_slope(z))
defvjp(
    _gap_slope,
    lambda ans, z: lambda g: g * _ratio(z) * (_gap(z) ** 2 - ans),
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
