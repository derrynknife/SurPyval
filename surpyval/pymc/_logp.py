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
    """The three log functions of one family, each ``f(pt, x, *params)``,
    and ``centre(pt, *params)``: a point inside the support where ``F``
    and ``S`` are both strictly between 0 and 1 (and their logs have
    finite derivatives) for any parameters, the stand-in at which a
    window's bound is evaluated where the bound itself must not be."""

    log_df: Callable[..., Any]
    log_sf: Callable[..., Any]
    log_ff: Callable[..., Any]
    centre: Callable[..., Any]


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
    centre=lambda pt, alpha, beta: alpha,
)

# Exponential: failure_rate
EXPONENTIAL = Family(
    log_df=lambda pt, x, rate: pt.log(rate) - rate * x,
    log_sf=lambda pt, x, rate: -rate * x,
    log_ff=lambda pt, x, rate: log1mexp(pt, -rate * x),
    centre=lambda pt, rate: 1.0 / rate,
)


# Gompertz: shape a, rate b; H = (b / a) expm1(a x), and b x times the
# series of expm1(y) / y below |a x| = 1e-3, so that it is exact through
# a = 0 (as surpyval's own)
def _gompertz_h(pt: Any, x: Any, a: Any, b: Any) -> Any:
    y = a * x
    small = pt.abs(y) < 1e-3
    ys = pt.switch(small, y, 0.0)
    series = 1.0 + ys * (
        1.0 / 2 + ys * (1.0 / 6 + ys * (1.0 / 24 + ys * (1.0 / 120)))
    )
    a_l = pt.switch(small, 1.0, a)
    x_l = pt.switch(small, 0.0, x)
    return pt.switch(small, b * x * series, b / a_l * pt.expm1(a_l * x_l))


GOMPERTZ = Family(
    log_df=lambda pt, x, a, b: pt.log(b) + a * x - _gompertz_h(pt, x, a, b),
    log_sf=lambda pt, x, a, b: -_gompertz_h(pt, x, a, b),
    log_ff=lambda pt, x, a, b: log1mexp(pt, -_gompertz_h(pt, x, a, b)),
    centre=lambda pt, a, b: 1.0 / b,
)


# InverseGaussian: mu (mean), lam (shape); a, b = sqrt(lam / x) (x / mu -+ 1).
# F = Phi(a) + e^{2 lam / mu} Phi(-b), added in logs; above the mean
# S = phi(a) (M(a) - M(b)), M Mills' ratio, as surpyval's own
def _ig_ab(pt: Any, x: Any, mu: Any, lam: Any) -> Any:
    # x = inf, where both are +inf, from a stand-in (0 * inf is NaN)
    top = pt.isinf(x)
    x_in = pt.switch(top, 1.0, x)
    r = pt.sqrt(lam / x_in)
    a = pt.switch(top, math.inf, r * (x_in / mu - 1.0))
    return a, pt.switch(top, math.inf, r * (x_in / mu + 1.0))


def _ig_log_ff(pt: Any, x: Any, mu: Any, lam: Any) -> Any:
    a, b = _ig_ab(pt, x, mu, lam)
    return pt.logaddexp(_log_ndtr(pt, a), 2.0 * lam / mu + _log_ndtr(pt, -b))


def _ig_log_sf(pt: Any, x: Any, mu: Any, lam: Any) -> Any:
    a, b = _ig_ab(pt, x, mu, lam)
    above = a > 0
    a_up = pt.switch(above, a, 1.0)
    b_up = pt.switch(above, b, 2.0)
    mills = math.sqrt(math.pi / 2.0)
    up = (
        -(a_up**2) / 2.0
        - _HALF_LOG_2PI
        + pt.log(
            mills * pt.erfcx(a_up / _SQRT2) - mills * pt.erfcx(b_up / _SQRT2)
        )
    )
    x_dn = pt.switch(above, 0.5 * mu + 0.0 * x, x)
    down = log1mexp(pt, _ig_log_ff(pt, x_dn, mu, lam))
    return pt.switch(above, up, down)


INVERSE_GAUSSIAN = Family(
    log_df=lambda pt, x, mu, lam: (
        0.5 * (pt.log(lam) - math.log(2.0 * math.pi))
        - 1.5 * pt.log(x)
        - lam * (x - mu) ** 2 / (2.0 * mu**2 * x)
    ),
    log_sf=_ig_log_sf,
    log_ff=_ig_log_ff,
    centre=lambda pt, mu, lam: mu,
)

# Rayleigh: sigma; H = x ** 2 / (2 sigma ** 2)
RAYLEIGH = Family(
    log_df=lambda pt, x, sigma: (
        pt.log(x) - 2.0 * pt.log(sigma) - x**2 / (2.0 * sigma**2)
    ),
    log_sf=lambda pt, x, sigma: -(x**2) / (2.0 * sigma**2),
    log_ff=lambda pt, x, sigma: log1mexp(pt, -(x**2) / (2.0 * sigma**2)),
    centre=lambda pt, sigma: sigma,
)

# Normal: mu, sigma
NORMAL = Family(
    log_df=lambda pt, x, mu, sigma: _normal_log_pdf(
        pt, (x - mu) / sigma, sigma
    ),
    log_sf=lambda pt, x, mu, sigma: _log_ndtr(pt, -(x - mu) / sigma),
    log_ff=lambda pt, x, mu, sigma: _log_ndtr(pt, (x - mu) / sigma),
    centre=lambda pt, mu, sigma: mu,
)

# LogNormal: mu, sigma of log(x)
LOGNORMAL = Family(
    log_df=lambda pt, x, mu, sigma: (
        _normal_log_pdf(pt, (pt.log(x) - mu) / sigma, sigma) - pt.log(x)
    ),
    log_sf=lambda pt, x, mu, sigma: _log_ndtr(pt, -(pt.log(x) - mu) / sigma),
    log_ff=lambda pt, x, mu, sigma: _log_ndtr(pt, (pt.log(x) - mu) / sigma),
    centre=lambda pt, mu, sigma: pt.exp(mu),
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
    centre=lambda pt, alpha, beta: alpha / beta,
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
    centre=lambda pt, alpha, beta: alpha,
)

# Gumbel (smallest extreme value): mu, sigma; H = exp((x - mu) / sigma)
GUMBEL = Family(
    log_df=lambda pt, x, mu, sigma: (
        (x - mu) / sigma - pt.exp((x - mu) / sigma) - pt.log(sigma)
    ),
    log_sf=lambda pt, x, mu, sigma: -pt.exp((x - mu) / sigma),
    log_ff=lambda pt, x, mu, sigma: log1mexp(pt, -pt.exp((x - mu) / sigma)),
    centre=lambda pt, mu, sigma: mu,
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
    centre=lambda pt, mu, sigma: mu,
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
    "Gompertz": GOMPERTZ,
    "InverseGaussian": INVERSE_GAUSSIAN,
}
