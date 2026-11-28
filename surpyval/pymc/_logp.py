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


# BirnbaumSaunders: alpha (shape), beta (scale, the median); F = Phi(xi),
# xi = (x - beta) / (alpha sqrt(x) sqrt(beta)), as surpyval's own; at
# x = inf, where that is inf / inf, from a stand-in
def _bs_xi(pt: Any, x: Any, a: Any, b: Any) -> Any:
    top = pt.isinf(x)
    x_in = pt.switch(top, 1.0, x)
    xi = (x_in - b) / (a * pt.sqrt(x_in) * pt.sqrt(b))
    return pt.switch(top, math.inf, xi)


BIRNBAUM_SAUNDERS = Family(
    log_df=lambda pt, x, a, b: (
        -(_bs_xi(pt, x, a, b) ** 2) / 2.0
        - _HALF_LOG_2PI
        + pt.log(x + b)
        - pt.log(2.0 * a)
        - 0.5 * pt.log(b)
        - 1.5 * pt.log(x)
    ),
    log_sf=lambda pt, x, a, b: _log_ndtr(pt, -_bs_xi(pt, x, a, b)),
    log_ff=lambda pt, x, a, b: _log_ndtr(pt, _bs_xi(pt, x, a, b)),
    centre=lambda pt, a, b: b,
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

# GeneralizedGamma: mu, sigma (of the log time) and Q, Prentice's form;
# w = (ln x - mu) / sigma, q = 1 / Q^2 and the gamma variable u = q e^{Qw}.
# As surpyval's own (``generalized_gamma.py``): near the lognormal (|Q| <
# 0.1, |eta| <= 1) the tails from Temme's uniform expansion, written in
# (w, Q) so it is exact through Q = 0, where the library's incomplete
# gamma functions at shapes of 1e4 and more lose their digits; elsewhere
# (shapes of 100 and less) from the incomplete gamma at shape q, as the
# Gamma family's.
_STIRLING = (
    1.0 / 12.0,
    -1.0 / 360.0,
    1.0 / 1260.0,
    -1.0 / 1680.0,
    1.0 / 1188.0,
    -691.0 / 360360.0,
    1.0 / 156.0,
)
_H_SERIES = tuple(1.0 / math.factorial(n + 2) for n in range(14))


# The stand-ins below (the values a branch sees on the points it does not
# own) are never 0: pytensor's rewrite ``local_mul_switch_sink`` moves
# every product with a ``switch`` that has a zero branch inside it, and
# on these nested sums that took minutes to compile.
def _gg_stirlerr(pt: Any, z: Any) -> Any:
    """``ln Gamma(q) - (q - 1/2) ln q + q - ln sqrt(2 pi)`` at ``q = 1/z``,
    0 at ``z = 0``: Stirling's series for ``q >= 10``."""
    small = z <= 0.1
    zs = pt.switch(small, z, 0.05)
    z2 = zs * zs
    series = _STIRLING[-1]
    for c in reversed(_STIRLING[:-1]):
        series = series * z2 + c
    q = 1.0 / pt.switch(small, 1.0, z)
    direct = pt.gammaln(q) - (q - 0.5) * pt.log(q) + q - _HALF_LOG_2PI
    return pt.switch(small, series * zs, direct)


def _gg_h(pt: Any, y: Any) -> Any:
    """``(e^y - 1 - y) / y^2``, 1/2 at 0."""
    small = pt.abs(y) < 0.1
    ys = pt.switch(small, y, 0.05)
    yl = pt.switch(small, 1.0, y)
    series = _H_SERIES[-1]
    for c in reversed(_H_SERIES[:-1]):
        series = series * ys + c
    return pt.switch(small, series, (pt.expm1(yl) - yl) / yl**2)


def _gg_w(pt: Any, x: Any, mu: Any, sigma: Any) -> Any:
    return (pt.log(x) - mu) / sigma


def _gg_log_df(pt: Any, x: Any, mu: Any, sigma: Any, Q: Any) -> Any:
    w = _gg_w(pt, x, mu, sigma)
    return (
        -_HALF_LOG_2PI
        - _gg_stirlerr(pt, Q * Q)
        - (w * w) * _gg_h(pt, Q * w)
        - pt.log(sigma)
        - pt.log(x)
    )


def _gg_temme_sum(pt: Any, eta: Any, Q: Any) -> Any:
    """``sum_k c_k(eta) Q^(2k)``, as two products with the table of
    coefficients rather than nested Horner sums (a graph of a few nodes
    instead of hundreds, which pytensor compiles in seconds)."""
    from surpyval.univariate.parametric.distributions._temme import TEMME

    rows, cols = TEMME.shape
    powers = [pt.ones_like(eta)]
    for _ in range(cols - 1):
        powers.append(powers[-1] * eta)
    q2 = Q * Q
    q_powers = [pt.ones_like(q2)]
    for _ in range(rows - 1):
        q_powers.append(q_powers[-1] * q2)
    ck = pt.dot(pt.stack(powers, axis=-1), pt.as_tensor(TEMME.T.copy()))
    return pt.sum(ck * pt.stack(q_powers, axis=-1), axis=-1)


def _gg_tails(pt: Any, x: Any, mu: Any, sigma: Any, Q: Any) -> Any:
    """``(ln S, ln F)``."""
    from surpyval.univariate.parametric.distributions._temme import (
        ETA_MAX,
        Q_MAX,
    )

    w = _gg_w(pt, x, mu, sigma)
    y = Q * w
    hy = _gg_h(pt, y)
    root = pt.sqrt(hy)
    eta = y * root * _SQRT2
    temme = (pt.abs(Q) < Q_MAX) & (pt.abs(eta) <= ETA_MAX)
    # Temme's expansion on its own points
    w_t = pt.switch(temme, w, 1.0)
    Q_t = pt.switch(temme, Q, 0.05)
    y_t = Q_t * w_t
    h_t = _gg_h(pt, y_t)
    r_t = pt.sqrt(h_t)
    zeta = w_t * r_t
    upper = zeta >= 0
    sign = pt.switch(upper, 1.0, -1.0)
    bracket = 0.5 * pt.erfcx(pt.abs(zeta)) + sign * Q_t * _gg_temme_sum(
        pt, y_t * r_t * _SQRT2, Q_t
    ) / math.sqrt(2.0 * math.pi)
    small_t = -(w_t * w_t) * h_t + pt.log(bracket)
    big_t = log1mexp(pt, small_t)
    s_t = pt.switch(upper, small_t, big_t)
    f_t = pt.switch(upper, big_t, small_t)
    # the incomplete gamma at shape q elsewhere, as the Gamma family's
    Q_c = pt.switch(temme, 0.5, Q)
    w_c = pt.switch(temme, 1.0, w)
    q = 1.0 / (Q_c * Q_c)
    u = q * pt.exp(Q_c * w_c)
    log_p = pt.log(pt.gammainc(q, u))
    log_q = pt.log(_gamma_q(pt, q, u))
    positive = Q_c > 0
    s_c = pt.switch(positive, log_q, log_p)
    f_c = pt.switch(positive, log_p, log_q)
    return pt.switch(temme, s_t, s_c), pt.switch(temme, f_t, f_c)


GENERALIZED_GAMMA = Family(
    log_df=_gg_log_df,
    log_sf=lambda pt, x, mu, sigma, Q: _gg_tails(pt, x, mu, sigma, Q)[0],
    log_ff=lambda pt, x, mu, sigma, Q: _gg_tails(pt, x, mu, sigma, Q)[1],
    centre=lambda pt, mu, sigma, Q: pt.exp(mu),
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
    "GeneralizedGamma": GENERALIZED_GAMMA,
    "BirnbaumSaunders": BIRNBAUM_SAUNDERS,
}
