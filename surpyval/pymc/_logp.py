"""The log density, log survival and log CDF of each supported family
as pytensor expressions, in SurPyval's parameterisation.

Each family is a ``Family`` of three functions ``f(pt, x, *params)``
taking the ``pytensor.tensor`` module, the times (a numpy array, never
traced) and the parameters (tensors, broadcastable against ``x``). They
are written in the stable forms SurPyval's own ``log_df``, ``log_sf`` and
``log_ff`` use: ``log1mexp`` for ``log(1 - exp(-H))``, ``softplus`` for
the logistic families and the scaled complementary error function for the
normal tails. Where a formula has two branches, each branch is evaluated
only at arguments it is finite on (a stand-in elsewhere), so that the
branch ``switch`` discards cannot put a NaN into a gradient.

pytensor is imported by the caller, never here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_LOG2 = math.log(2.0)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Family:
    """The three log functions of one family, each ``f(pt, x, *params)``."""

    log_df: Callable[..., Any]
    log_sf: Callable[..., Any]
    log_ff: Callable[..., Any]


def log1mexp(pt: Any, d: Any) -> Any:
    """``log(1 - exp(d))`` for ``d <= 0``, exact at both ends (Maechler,
    2012): ``log(-expm1(d))`` above ``-log 2`` and ``log1p(-exp(d))``
    below it, each fed only the arguments of its own branch."""
    near = d > -_LOG2
    d_near = pt.switch(near, d, -1.0)
    d_far = pt.switch(near, -1.0, d)
    return pt.switch(near, pt.log(-pt.expm1(d_near)), pt.log1p(-pt.exp(d_far)))


def _log_ndtr(pt: Any, z: Any) -> Any:
    """``log Phi(z)``, the standard normal log CDF, accurate in both
    tails: through ``erfcx`` below -1 (where ``Phi`` underflows) and
    ``log1p(-erfc(z / sqrt 2) / 2)`` above it."""
    low = z < -1.0
    z_low = pt.switch(low, z, -1.0)
    z_high = pt.switch(low, 0.0, z)
    lower = pt.log(pt.erfcx(-z_low / _SQRT2) / 2.0) - z_low**2 / 2.0
    upper = pt.log1p(-pt.erfc(z_high / _SQRT2) / 2.0)
    return pt.switch(low, lower, upper)


def _normal_log_pdf(pt: Any, z: Any, sigma: Any) -> Any:
    return -(z**2) / 2.0 - pt.log(sigma) - _HALF_LOG_2PI


# Weibull: alpha (scale), beta (shape); H = (x / alpha) ** beta
def _weibull_h(pt: Any, x: Any, alpha: Any, beta: Any) -> Any:
    return pt.exp(beta * (pt.log(x) - pt.log(alpha)))


WEIBULL = Family(
    log_df=lambda pt, x, alpha, beta: (
        pt.log(beta)
        - pt.log(alpha)
        + (beta - 1.0) * (pt.log(x) - pt.log(alpha))
        - _weibull_h(pt, x, alpha, beta)
    ),
    log_sf=lambda pt, x, alpha, beta: -_weibull_h(pt, x, alpha, beta),
    log_ff=lambda pt, x, alpha, beta: log1mexp(
        pt, -_weibull_h(pt, x, alpha, beta)
    ),
)

# Exponential: failure_rate
EXPONENTIAL = Family(
    log_df=lambda pt, x, rate: pt.log(rate) - rate * x,
    log_sf=lambda pt, x, rate: -rate * x,
    log_ff=lambda pt, x, rate: log1mexp(pt, -rate * x),
)

# Rayleigh: sigma; H = x ** 2 / (2 sigma ** 2)
RAYLEIGH = Family(
    log_df=lambda pt, x, sigma: (
        pt.log(x) - 2.0 * pt.log(sigma) - x**2 / (2.0 * sigma**2)
    ),
    log_sf=lambda pt, x, sigma: -(x**2) / (2.0 * sigma**2),
    log_ff=lambda pt, x, sigma: log1mexp(pt, -(x**2) / (2.0 * sigma**2)),
)

# Normal: mu, sigma
NORMAL = Family(
    log_df=lambda pt, x, mu, sigma: _normal_log_pdf(
        pt, (x - mu) / sigma, sigma
    ),
    log_sf=lambda pt, x, mu, sigma: _log_ndtr(pt, -(x - mu) / sigma),
    log_ff=lambda pt, x, mu, sigma: _log_ndtr(pt, (x - mu) / sigma),
)

# LogNormal: mu, sigma of log(x)
LOGNORMAL = Family(
    log_df=lambda pt, x, mu, sigma: (
        _normal_log_pdf(pt, (pt.log(x) - mu) / sigma, sigma) - pt.log(x)
    ),
    log_sf=lambda pt, x, mu, sigma: _log_ndtr(pt, -(pt.log(x) - mu) / sigma),
    log_ff=lambda pt, x, mu, sigma: _log_ndtr(pt, (pt.log(x) - mu) / sigma),
)


# Gamma: alpha (shape), beta (rate)
def _gamma_q(pt: Any, a: Any, y: Any) -> Any:
    """The regularised upper incomplete gamma ``Q(a, y)``, its value from
    ``gammaincc`` (exact in the upper tail) and its derivative from
    ``-P(a, y)``'s: pytensor's derivative of ``gammaincc`` in ``a`` is
    off by about 0.2% where ``P`` is near 1, that of ``gammainc`` is
    not, and ``dQ = -dP``."""
    from importlib import import_module

    zero_grad = import_module("pytensor.gradient").zero_grad
    p = pt.gammainc(a, y)
    return zero_grad(pt.gammaincc(a, y)) + zero_grad(p) - p


GAMMA = Family(
    log_df=lambda pt, x, alpha, beta: (
        alpha * pt.log(beta)
        + (alpha - 1.0) * pt.log(x)
        - beta * x
        - pt.gammaln(alpha)
    ),
    log_sf=lambda pt, x, alpha, beta: pt.log(_gamma_q(pt, alpha, beta * x)),
    log_ff=lambda pt, x, alpha, beta: pt.log(pt.gammainc(alpha, beta * x)),
)


# LogLogistic: alpha (scale), beta (shape); z = beta * log(x / alpha)
def _loglogistic_z(pt: Any, x: Any, alpha: Any, beta: Any) -> Any:
    return beta * (pt.log(x) - pt.log(alpha))


LOGLOGISTIC = Family(
    log_df=lambda pt, x, alpha, beta: (
        pt.log(beta)
        - pt.log(x)
        - pt.softplus(_loglogistic_z(pt, x, alpha, beta))
        - pt.softplus(-_loglogistic_z(pt, x, alpha, beta))
    ),
    log_sf=lambda pt, x, alpha, beta: -pt.softplus(
        _loglogistic_z(pt, x, alpha, beta)
    ),
    log_ff=lambda pt, x, alpha, beta: -pt.softplus(
        -_loglogistic_z(pt, x, alpha, beta)
    ),
)

# Gumbel (smallest extreme value): mu, sigma; H = exp((x - mu) / sigma)
GUMBEL = Family(
    log_df=lambda pt, x, mu, sigma: (
        (x - mu) / sigma - pt.exp((x - mu) / sigma) - pt.log(sigma)
    ),
    log_sf=lambda pt, x, mu, sigma: -pt.exp((x - mu) / sigma),
    log_ff=lambda pt, x, mu, sigma: log1mexp(pt, -pt.exp((x - mu) / sigma)),
)

# Logistic: mu, sigma
LOGISTIC = Family(
    log_df=lambda pt, x, mu, sigma: (
        -pt.log(sigma)
        - pt.softplus((x - mu) / sigma)
        - pt.softplus(-(x - mu) / sigma)
    ),
    log_sf=lambda pt, x, mu, sigma: -pt.softplus((x - mu) / sigma),
    log_ff=lambda pt, x, mu, sigma: -pt.softplus(-(x - mu) / sigma),
)

# By SurPyval's distribution name (``dist.name``).
FAMILIES: dict[str, Family] = {
    "Weibull": WEIBULL,
    "Exponential": EXPONENTIAL,
    "LogNormal": LOGNORMAL,
    "Gamma": GAMMA,
    "LogLogistic": LOGLOGISTIC,
    "Normal": NORMAL,
    "Gumbel": GUMBEL,
    "Logistic": LOGISTIC,
    "Rayleigh": RAYLEIGH,
}
