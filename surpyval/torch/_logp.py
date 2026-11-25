"""The log density, log survival and log CDF of each supported family
in torch, in SurPyval's parameterisation.

Each family is a ``Family`` of three functions ``f(torch, x, *params)``
taking the ``torch`` module, the times and the parameters (tensors that
broadcast against each other). They mirror ``surpyval/pymc/_logp.py``
term by term (see DECISIONS.md for why the two are not one), in the
stable forms SurPyval's own ``log_df``, ``log_sf`` and ``log_ff`` use:
``log1mexp`` for ``log(1 - exp(-H))``, ``softplus`` for the logistic
families and the scaled complementary error function for the normal
tails. Where a formula has two branches, each branch is evaluated only
at arguments it is finite on (a stand-in elsewhere), so that the branch
``where`` discards cannot put a NaN into a gradient.

torch is imported by the caller, never here.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from typing import Any

_LOG2 = math.log(2.0)
_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)
_SQRT2 = math.sqrt(2.0)


@dataclass(frozen=True)
class Family:
    """The three log functions of one family, each
    ``f(torch, x, *params)``."""

    log_df: Callable[..., Any]
    log_sf: Callable[..., Any]
    log_ff: Callable[..., Any]


def log1mexp(torch: Any, d: Any) -> Any:
    """``log(1 - exp(d))`` for ``d <= 0``, exact at both ends (Maechler,
    2012): ``log(-expm1(d))`` above ``-log 2`` and ``log1p(-exp(d))``
    below it, each fed only the arguments of its own branch."""
    near = d > -_LOG2
    d_near = torch.where(near, d, -1.0)
    d_far = torch.where(near, -1.0, d)
    return torch.where(
        near, torch.log(-torch.expm1(d_near)), torch.log1p(-torch.exp(d_far))
    )


def softplus(torch: Any, z: Any) -> Any:
    """``log(1 + exp(z))``, exact in both tails (``-logsigmoid(-z)``;
    ``torch.nn.functional.softplus`` returns ``z`` itself above 20, which
    is 2e-9 off there)."""
    return -torch.nn.functional.logsigmoid(-z)


def _log_ndtr(torch: Any, z: Any) -> Any:
    """``log Phi(z)``, the standard normal log CDF, accurate in both
    tails: through ``erfcx`` below -1 (where ``Phi`` underflows) and
    ``log1p(-erfc(z / sqrt 2) / 2)`` above it."""
    low = z < -1.0
    z_low = torch.where(low, z, -1.0)
    z_high = torch.where(low, 0.0, z)
    lower = torch.log(torch.special.erfcx(-z_low / _SQRT2) / 2.0) - (
        z_low**2 / 2.0
    )
    upper = torch.log1p(-torch.special.erfc(z_high / _SQRT2) / 2.0)
    return torch.where(low, lower, upper)


def _normal_log_pdf(torch: Any, z: Any, sigma: Any) -> Any:
    return -(z**2) / 2.0 - torch.log(sigma) - _HALF_LOG_2PI


# Weibull: alpha (scale), beta (shape); H = (x / alpha) ** beta
def _weibull_h(torch: Any, x: Any, alpha: Any, beta: Any) -> Any:
    return torch.exp(beta * (torch.log(x) - torch.log(alpha)))


WEIBULL = Family(
    log_df=lambda torch, x, alpha, beta: (
        torch.log(beta)
        - torch.log(alpha)
        + (beta - 1.0) * (torch.log(x) - torch.log(alpha))
        - _weibull_h(torch, x, alpha, beta)
    ),
    log_sf=lambda torch, x, alpha, beta: -_weibull_h(torch, x, alpha, beta),
    log_ff=lambda torch, x, alpha, beta: log1mexp(
        torch, -_weibull_h(torch, x, alpha, beta)
    ),
)

# Exponential: failure_rate
EXPONENTIAL = Family(
    log_df=lambda torch, x, rate: torch.log(rate) - rate * x,
    log_sf=lambda torch, x, rate: -rate * x,
    log_ff=lambda torch, x, rate: log1mexp(torch, -rate * x),
)


# Gompertz: shape a, rate b; H = (b / a) expm1(a x), and b x times the
# series of expm1(y) / y below |a x| = 1e-3 (as surpyval's own)
def _gompertz_h(torch: Any, x: Any, a: Any, b: Any) -> Any:
    y = a * x
    small = torch.abs(y) < 1e-3
    ys = torch.where(small, y, 0.0)
    series = 1.0 + ys * (
        1.0 / 2 + ys * (1.0 / 6 + ys * (1.0 / 24 + ys * (1.0 / 120)))
    )
    a_l = torch.where(small, 1.0, a)
    x_l = torch.where(small, 0.0, x)
    return torch.where(small, b * x * series, b / a_l * torch.expm1(a_l * x_l))


GOMPERTZ = Family(
    log_df=lambda torch, x, a, b: (
        torch.log(b) + a * x - _gompertz_h(torch, x, a, b)
    ),
    log_sf=lambda torch, x, a, b: -_gompertz_h(torch, x, a, b),
    log_ff=lambda torch, x, a, b: log1mexp(
        torch, -_gompertz_h(torch, x, a, b)
    ),
)

# Rayleigh: sigma; H = x ** 2 / (2 sigma ** 2)
RAYLEIGH = Family(
    log_df=lambda torch, x, sigma: (
        torch.log(x) - 2.0 * torch.log(sigma) - x**2 / (2.0 * sigma**2)
    ),
    log_sf=lambda torch, x, sigma: -(x**2) / (2.0 * sigma**2),
    log_ff=lambda torch, x, sigma: log1mexp(torch, -(x**2) / (2.0 * sigma**2)),
)

# Normal: mu, sigma
NORMAL = Family(
    log_df=lambda torch, x, mu, sigma: _normal_log_pdf(
        torch, (x - mu) / sigma, sigma
    ),
    log_sf=lambda torch, x, mu, sigma: _log_ndtr(torch, -(x - mu) / sigma),
    log_ff=lambda torch, x, mu, sigma: _log_ndtr(torch, (x - mu) / sigma),
)

# LogNormal: mu, sigma of log(x)
LOGNORMAL = Family(
    log_df=lambda torch, x, mu, sigma: (
        _normal_log_pdf(torch, (torch.log(x) - mu) / sigma, sigma)
        - torch.log(x)
    ),
    log_sf=lambda torch, x, mu, sigma: _log_ndtr(
        torch, -(torch.log(x) - mu) / sigma
    ),
    log_ff=lambda torch, x, mu, sigma: _log_ndtr(
        torch, (torch.log(x) - mu) / sigma
    ),
)


# Gamma: alpha (shape), beta (rate)
#
# torch's regularised incomplete gamma functions have no derivative in
# the shape, and for a shape above 20 with y within 0.3 of it (relative)
# they are accurate to about 1e-9 only (measured against mpmath with
# torch 2.15: their uniform asymptotic expansion). So P and Q come from
# torch outside that region, with the shape's derivative from a
# difference (``_d_shape``), and from a quadrature of the density inside
# it (``_near_mode``), which autograd differentiates.
_NEAR = (19.5, 0.33)  # the region, a little wider than torch's (20, 0.3)
_NODES = 96
_DROP = 38.0  # the density falls by exp(-38) over the range integrated


def _d_shape(torch: Any, a: Any, y: Any, p: Any, q: Any) -> Any:
    """``dP(a, y) / da`` by differences. ``P`` is smooth in ``a``, so a
    central difference of the log of the smaller of ``P`` and ``Q``
    (which keeps its relative accuracy in that tail), at three steps with
    Richardson's extrapolation (error of order ``h ** 6``), gives it to
    about 1e-11 relative. Where the smaller of the two underflows to 0,
    so does its derivative."""
    lower = p <= q
    # The step: 1% of the scale P varies on, a for a small shape and
    # sqrt(a) for a large one.
    h = 0.01 * a / torch.sqrt(1.0 + a)

    def slope(step: Any) -> Any:
        up = torch.where(
            lower,
            torch.special.gammainc(a + step, y),
            torch.special.gammaincc(a + step, y),
        )
        down = torch.where(
            lower,
            torch.special.gammainc(a - step, y),
            torch.special.gammaincc(a - step, y),
        )
        return (torch.log(up) - torch.log(down)) / (2.0 * step)

    d1, d2, d4 = slope(h), slope(h / 2.0), slope(h / 4.0)
    r1, r2 = (4.0 * d2 - d1) / 3.0, (4.0 * d4 - d2) / 3.0
    d_log = (16.0 * r2 - r1) / 15.0
    small = torch.where(lower, p, q)
    d_small = torch.where(small > 0, small * d_log, 0.0)
    return torch.where(lower, d_small, -d_small)


@cache
def _incomplete_gamma(torch: Any) -> Any:
    """The autograd function ``(a, y) -> (P(a, y), Q(a, y))``: each
    value from torch's own function (exact in its tail), and both
    derivatives from ``P``'s (``dQ = -dP``): the density in ``y``,
    :func:`_d_shape` in ``a``."""

    class IncompleteGamma(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, a: Any, y: Any) -> Any:
            p = torch.special.gammainc(a, y)
            q = torch.special.gammaincc(a, y)
            ctx.save_for_backward(a, y, p, q)
            return p, q

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(ctx: Any, grad_p: Any, grad_q: Any) -> Any:
            a, y, p, q = ctx.saved_tensors
            grad = grad_p - grad_q
            grad_a = grad_y = None
            if ctx.needs_input_grad[0]:
                grad_a = grad * _d_shape(torch, a, y, p, q)
            if ctx.needs_input_grad[1]:
                grad_y = grad * torch.exp(
                    (a - 1.0) * torch.log(y) - y - torch.lgamma(a)
                )
            return grad_a, grad_y

    return IncompleteGamma


@cache
def _gauss_legendre() -> tuple:
    from numpy.polynomial.legendre import leggauss

    return leggauss(_NODES)


def _near_mode(torch: Any, a: Any, y: Any) -> Any:
    """``(P, Q)`` for a shape above 19.5 and ``y`` within a third of it:
    the smaller of the two as the integral of the density from ``y``
    away from the mode, over the range on which the density falls by
    ``exp(-_DROP)`` (bounded from its log-concavity), by Gauss-Legendre
    quadrature. The density is written about ``t = a``, with Stirling's
    series for ``log Gamma(a)``, so that nothing of the size of ``a``
    cancels."""
    above = y >= a
    k = 2.0 * _DROP / (a - 1.0)
    # log g(y + s) - log g(y) <= -rate * s - (a - 1) u^2 / (2 (1 + u))
    # above the mode (u = s / y), and <= -rate * s - (a - 1) u^2 / 2
    # below it: the range is the shorter of the two each bound gives.
    rate = torch.abs(1.0 - (a - 1.0) / y)
    u = torch.where(above, (k + torch.sqrt(k * k + 4.0 * k)) / 2.0, k.sqrt())
    span = torch.minimum(_DROP / rate, u * y)
    span = torch.where(above, span, torch.minimum(span, y))
    nodes, weights = (
        torch.as_tensor(v, dtype=a.dtype, device=a.device)
        for v in _gauss_legendre()
    )
    half = (span / 2.0).unsqueeze(-1)
    middle = torch.where(above, y + span / 2.0, y - span / 2.0).unsqueeze(-1)
    a_ = a.unsqueeze(-1)
    d = middle + half * nodes - a_  # t - a at each node
    inv = 1.0 / a
    stirling = inv * (
        1.0 / 12.0
        - inv**2 * (1.0 / 360.0 - inv**2 * (1.0 / 1260.0 - inv**2 / 1680.0))
    )
    log_g = (
        (a_ - 1.0) * torch.log1p(d / a_)
        - d
        - (0.5 * torch.log(a) + _HALF_LOG_2PI + stirling).unsqueeze(-1)
    )
    tail = (half * weights * torch.exp(log_g)).sum(-1)
    return torch.where(above, 1.0 - tail, tail), torch.where(
        above, tail, 1.0 - tail
    )


def _gamma_pq(torch: Any, a: Any, y: Any) -> Any:
    """The regularised incomplete gamma functions ``(P(a, y), Q(a, y))``
    with their derivatives in both arguments."""
    a, y = torch.broadcast_tensors(a, y)
    near = (a > _NEAR[0]) & (torch.abs(y - a) < _NEAR[1] * a)
    if not bool(near.any()):
        return _incomplete_gamma(torch).apply(a, y)
    if bool(near.all()):
        return _near_mode(torch, a, y)
    # Each rule on its own rows.
    p_near, q_near = _near_mode(torch, a[near], y[near])
    p_far, q_far = _incomplete_gamma(torch).apply(a[~near], y[~near])
    p = torch.zeros_like(y).masked_scatter(near, p_near)
    q = torch.zeros_like(y).masked_scatter(near, q_near)
    return p.masked_scatter(~near, p_far), q.masked_scatter(~near, q_far)


GAMMA = Family(
    log_df=lambda torch, x, alpha, beta: (
        alpha * torch.log(beta)
        + (alpha - 1.0) * torch.log(x)
        - beta * x
        - torch.lgamma(alpha)
    ),
    log_sf=lambda torch, x, alpha, beta: torch.log(
        _gamma_pq(torch, alpha, beta * x)[1]
    ),
    log_ff=lambda torch, x, alpha, beta: torch.log(
        _gamma_pq(torch, alpha, beta * x)[0]
    ),
)


# LogLogistic: alpha (scale), beta (shape); z = beta * log(x / alpha)
def _loglogistic_z(torch: Any, x: Any, alpha: Any, beta: Any) -> Any:
    return beta * (torch.log(x) - torch.log(alpha))


LOGLOGISTIC = Family(
    log_df=lambda torch, x, alpha, beta: (
        torch.log(beta)
        - torch.log(x)
        - softplus(torch, _loglogistic_z(torch, x, alpha, beta))
        - softplus(torch, -_loglogistic_z(torch, x, alpha, beta))
    ),
    log_sf=lambda torch, x, alpha, beta: -softplus(
        torch, _loglogistic_z(torch, x, alpha, beta)
    ),
    log_ff=lambda torch, x, alpha, beta: -softplus(
        torch, -_loglogistic_z(torch, x, alpha, beta)
    ),
)

# Gumbel (smallest extreme value): mu, sigma; H = exp((x - mu) / sigma)
GUMBEL = Family(
    log_df=lambda torch, x, mu, sigma: (
        (x - mu) / sigma - torch.exp((x - mu) / sigma) - torch.log(sigma)
    ),
    log_sf=lambda torch, x, mu, sigma: -torch.exp((x - mu) / sigma),
    log_ff=lambda torch, x, mu, sigma: log1mexp(
        torch, -torch.exp((x - mu) / sigma)
    ),
)

# Logistic: mu, sigma
LOGISTIC = Family(
    log_df=lambda torch, x, mu, sigma: (
        -torch.log(sigma)
        - softplus(torch, (x - mu) / sigma)
        - softplus(torch, -(x - mu) / sigma)
    ),
    log_sf=lambda torch, x, mu, sigma: -softplus(torch, (x - mu) / sigma),
    log_ff=lambda torch, x, mu, sigma: -softplus(torch, -(x - mu) / sigma),
)

# By SurPyval's distribution name (``dist.name``), in the order of
# ``surpyval.pymc.SUPPORTED``.
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
}
