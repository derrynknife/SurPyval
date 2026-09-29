"""
Numerically stable pieces the continuous distributions are built from.

A distribution's functions lose their accuracy in the tails when they are
formed from a probability that has already rounded to 1, to 0 or to inf:
``-log(sf)`` is -0.0 where ``sf`` rounds to 1, ``log(ff)`` is -inf once
``ff`` underflows, and ``df / sf`` is 0 / 0 once both do (#410, #442,
#443, #444). The helpers here compute the complements and logs directly
(principle 9).

Every ``np.where`` below evaluates each branch only on arguments that
branch is exact and finite on, so that neither the values nor autograd's
gradients of the branch not taken can be NaN: the fits differentiate
these functions.
"""

from typing import Any, Callable

from autograd.scipy.stats import norm

from surpyval import np
from surpyval.univariate.parametric.parametric_fitter import Boxable, Numeric

# ln 2: log(1 - e^-r) is taken as log1p(-e^-r) above it and as
# log(-expm1(-r)) below it, each exact on its own side (Maechler, 2012).
_LN2 = float(np.log(2.0))
# Below this log r, log(1 - e^-r) = log r - r / 2 to double precision (the
# next term is r^2 / 24), and it stays finite after r itself underflows.
_LOG_SMALL = -20.0
# Above this z the normal hazard is its asymptotic series (normal_hazard).
_Z_MILLS = 100.0
_TINY = float(np.finfo(float).tiny)
_HUGE = float(np.finfo(float).max)


def log_ratio(x: Numeric, scale: Boxable) -> Boxable:
    r"""
    :math:`\ln(x / s)` for :math:`x, s > 0`: the log of the rounded ratio
    where that is a normal double (one rounding, however close
    :math:`x` is to :math:`s`), and :math:`\ln x - \ln s` where the ratio
    under- or overflows.
    """
    with np.errstate(over="ignore", under="ignore"):
        ratio = x / scale
    normal = (ratio > _TINY) & (ratio < _HUGE)
    if np.all(normal):
        return np.log(ratio)
    # autograd's ``where`` does not unbroadcast its gradient, so the
    # parameter it selects must have the full shape
    x_in = np.where(normal, x, scale + np.zeros_like(ratio))
    x_out = np.where(normal, 1.0, x)
    return np.where(
        normal, np.log(x_in / scale), np.log(x_out) - np.log(scale)
    )


def log1mexp(r: Boxable, log_r: Boxable) -> tuple[Boxable, Boxable]:
    r"""
    :math:`\log(1 - e^{-r})` for :math:`r \geq 0`, and
    :math:`\log((1 - e^{-r}) / r)`, from :math:`r` and :math:`\log r`.

    ``log_r`` carries :math:`r` where it underflows: a Weibull's
    :math:`\log F` at :math:`(x/\alpha)^{\beta} = 10^{-400}` is -921, not
    the -inf of ``log(-expm1(-r))``. At :math:`r = 0` (``log_r`` -inf)
    both are -inf and 0.
    """
    small = log_r < _LOG_SMALL
    if not np.any(small):
        # the common case, without the branches it does not need
        above = r > _LN2
        if np.all(above):
            mid = np.log1p(-np.exp(-r))
        elif not np.any(above):
            mid = np.log(-np.expm1(-r))
        else:
            mid = np.where(
                above,
                np.log1p(-np.exp(-np.where(above, r, 1.0))),
                np.log(-np.expm1(-np.where(above, 0.5, r))),
            )
        return mid, mid - log_r
    r_mid = np.where(small, 1.0, r)
    mid = np.where(
        r_mid > _LN2,
        np.log1p(-np.exp(-np.where(r_mid > _LN2, r_mid, 1.0))),
        np.log(-np.expm1(-np.where(r_mid > _LN2, 0.5, r_mid))),
    )
    log_r_mid = np.where(small, 0.0, log_r)
    log_r_small = np.where(small, log_r, _LOG_SMALL)
    half_r = np.exp(log_r_small) / 2.0
    value = np.where(small, log_r_small - half_r, mid)
    ratio = np.where(small, -half_r, mid - log_r_mid)
    return value, ratio


def softplus(z: Boxable) -> Boxable:
    r"""
    :math:`\log(1 + e^{z})`, without overflow for large :math:`z` and
    without rounding to 0 for very negative :math:`z` (where it is
    :math:`e^{z}`). ``-softplus(-z)`` is the log of the logistic function
    :math:`1 / (1 + e^{-z})`.
    """
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def normal_hazard(z: Boxable) -> Boxable:
    r"""
    The standard normal hazard :math:`\phi(z) / (1 - \Phi(z))`.

    The quotient of the density and survival function is 0 / 0 once both
    underflow (#444), and the difference of their logs loses
    :math:`z^2 \epsilon` to cancellation, so above :math:`z = 100` it is
    the asymptotic series of the inverse Mills ratio,
    :math:`z / (1 - z^{-2} + 3 z^{-4} - 15 z^{-6} + 105 z^{-8})`,
    whose next term (:math:`945 z^{-10}`) is below double precision
    there.
    """
    big = z > _Z_MILLS
    z_small = np.where(big, 0.0, z)
    z_big = np.where(big, z, _Z_MILLS)
    small = np.exp(norm.logpdf(z_small) - norm.logsf(z_small))
    w = (1.0 / z_big) ** 2
    series = 1.0 - w * (1.0 - 3.0 * w * (1.0 - 5.0 * w * (1.0 - 7.0 * w)))
    return np.where(big, z_big / series, small)


def on_support(
    x: Numeric,
    inside: Boxable,
    at_edge: Boxable | Callable[[], Boxable],
    edge: float = 0.0,
) -> Boxable:
    """
    ``inside`` for x above ``edge``, ``at_edge`` at the edge itself, and
    NaN below it and for a NaN x. ``at_edge`` may be a function of no
    arguments, called only when some x is at or below the edge.

    ``inside`` must be computed with the points at and below the edge
    replaced by a point inside, so that no branch sees, say, a log of 0.
    The public functions are guarded below the support already
    (``parametric_fitter._support_guarded``); the NaN here only keeps an
    internal call honest.
    """
    out: Any
    if np.all(x > edge):
        out = inside
    else:
        # autograd's ``where`` does not unbroadcast its gradient, so a
        # parameter-dependent ``at_edge`` must have the full shape.
        if callable(at_edge):
            at_edge = at_edge()
        at_edge = at_edge + np.zeros_like(inside)
        out = np.where(x > edge, inside, np.where(x == edge, at_edge, np.nan))
    # a scalar in, a scalar out (a 0-d where is an array)
    return out[()] if hasattr(out, "shape") else out


def positive_or_one(x: Numeric) -> Numeric:
    """``x`` with the points at or below 0 (and NaN) replaced by 1, a
    stand-in inside the support for ``on_support`` to overwrite."""
    if np.all(x > 0):
        return x
    return np.where(x > 0, x, 1.0)


def power_at_zero(power: Boxable, scale: Boxable) -> tuple[Boxable, Boxable]:
    r"""
    The limit at :math:`x = 0` of :math:`c\,x^{p}` and of its log, for a
    density or hazard that behaves like that there: :math:`(\infty,
    \infty)` for :math:`p < 0`, :math:`(c, \ln c)` for :math:`p = 0` and
    :math:`(0, -\infty)` for :math:`p > 0`. ``scale`` is :math:`\ln c`.

    The formulas themselves are :math:`0 \cdot \log 0` = NaN there at
    :math:`p = 0` (a Weibull with shape 1, #444).
    """
    log_value = np.where(
        power < 0, np.inf, np.where(power == 0, scale, -np.inf)
    )
    value = np.where(
        power < 0,
        np.inf,
        np.where(power == 0, np.exp(np.where(power == 0, scale, 0.0)), 0.0),
    )
    return value, log_value
