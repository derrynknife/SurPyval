"""Hand-written log-likelihood gradients for the common families.

Autograd's per-operation cost dominates a gradient at one point, so for
the Weibull, Exponential, Normal, LogNormal and Gamma, without an offset,
limited-failure or zero-inflated form, the likelihood and its gradient in
the parameters are written out here in plain NumPy. Each family gives, at
times ``x`` (inside the support), ``log f``, ``log S`` or ``log F`` and
their derivatives in the parameters; :class:`FastLikelihood` assembles
them into the same sum, term for term, as the public likelihood
(:class:`surpyval.bayes.likelihood.LogLikelihood`), whose value it equals
to rounding.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.special import digamma, gammaln, log_ndtr

from surpyval.bayes.likelihood import _log1mexp
from surpyval.univariate.parametric.distributions._stable import (
    log1mexp,
)

_LOG_SQRT_2PI = float(np.log(np.sqrt(2 * np.pi)))

Pieces = tuple[npt.NDArray, npt.NDArray]
#: A block of times prepared once: the times and their logs.
Prepared = tuple[npt.NDArray, npt.NDArray]


def prepare(x: Any) -> Prepared:
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return x, np.log(x)


def _log1mexp_pair(r: npt.NDArray, log_r: npt.NDArray) -> Pieces:
    value, ratio = log1mexp(r, log_r)
    return np.asarray(value), np.asarray(ratio)


def _weibull(p: Prepared, kind: str, alpha: float, beta: float) -> Pieces:
    _, log_x = p
    log_a = math.log(alpha)
    u = log_x - log_a
    t = np.exp(beta * u)
    if kind == "f":
        v = (math.log(beta) - log_a - u) + beta * u - t
        g = np.stack([(t - 1.0) * (beta / alpha), 1.0 / beta + u * (1.0 - t)])
    elif kind == "S":
        v = -t
        g = np.stack([t * (beta / alpha), -t * u])
    else:
        v, ratio = _log1mexp_pair(t, beta * u)
        # d log F = d log t * t / expm1(t), and t / expm1(t) is
        # exp(-t - ratio), 1 at t = 0 and 0 as t grows
        w = np.exp(-t - ratio)
        g = np.stack([w * (-beta / alpha), u * w])
    return v, g


def _exponential(p: Prepared, kind: str, rate: float) -> Pieces:
    x, log_x = p
    if kind == "f":
        return math.log(rate) - rate * x, (1.0 / rate - x)[None, :]
    if kind == "S":
        return -rate * x, -x[None, :]
    r = rate * x
    v, ratio = _log1mexp_pair(r, math.log(rate) + log_x)
    return v, (np.exp(-r - ratio) / rate)[None, :]


def _hazard(
    z: npt.NDArray, log_phi: npt.NDArray, log_s: npt.NDArray
) -> npt.NDArray:
    """``phi(z) / S(z)`` from the logs, with the inverse Mills ratio's
    series above ``z = 100`` (where the difference of logs loses its
    digits), as ``_stable.normal_hazard`` has it."""
    big = z > 100.0
    if not np.any(big):
        return np.exp(log_phi - log_s)
    w = (1.0 / np.where(big, z, 100.0)) ** 2
    series = 1.0 - w * (1.0 - 3.0 * w * (1.0 - 5.0 * w * (1.0 - 7.0 * w)))
    return np.where(big, z / series, np.exp(log_phi - log_s))


def _normal_on(
    y: npt.NDArray, kind: str, mu: float, sigma: float, jac: Any
) -> Pieces:
    z = (y - mu) / sigma
    log_phi = -(z**2) / 2.0 - _LOG_SQRT_2PI
    if kind == "f":
        v = log_phi - math.log(sigma) - jac
        g = np.stack([z / sigma, (z * z - 1.0) / sigma])
    elif kind == "S":
        v = log_ndtr(-z)
        h = _hazard(z, log_phi, v)
        g = np.stack([h / sigma, z * h / sigma])
    else:
        v = log_ndtr(z)
        h = _hazard(-z, log_phi, v)
        g = np.stack([-h / sigma, -z * h / sigma])
    return v, g


def _normal(p: Prepared, kind: str, mu: float, sigma: float) -> Pieces:
    return _normal_on(p[0], kind, mu, sigma, 0.0)


def _lognormal(p: Prepared, kind: str, mu: float, sigma: float) -> Pieces:
    return _normal_on(p[1], kind, mu, sigma, p[1])


def _gamma(p: Prepared, kind: str, alpha: float, beta: float) -> Pieces:
    from surpyval import Gamma
    from surpyval.utils.autograd_gamma_compat import log_pq_da

    x, log_x = p
    log_b = math.log(beta)
    log_f = alpha * log_b - gammaln(alpha) + (alpha - 1) * log_x - beta * x
    if kind == "f":
        g = np.stack(
            [log_b - digamma(alpha) + log_x, np.full_like(x, alpha / beta) - x]
        )
        return log_f, g
    fn = Gamma.log_sf if kind == "S" else Gamma.log_ff
    v = np.asarray(fn(x, alpha, beta), dtype=float)
    sign = -1.0 if kind == "S" else 1.0
    d_beta = sign * (x / beta) * np.exp(log_f - v)
    d_alpha = np.asarray(
        log_pq_da(1 if kind == "S" else 0)(alpha, beta * x), dtype=float
    )
    bad = ~np.isfinite(d_alpha)
    if np.any(bad):
        # where the series and continued fraction did not settle: central
        # differences of the family's own function
        h = 1e-5 * max(1.0, alpha)
        xb = x[bad]
        d_alpha[bad] = (
            np.asarray(fn(xb, alpha + h, beta))
            - np.asarray(fn(xb, alpha - h, beta))
        ) / (2 * h)
    return v, np.stack([d_alpha, d_beta])


FAMILIES: dict[str, Callable[..., Pieces]] = {
    "Weibull": _weibull,
    "Exponential": _exponential,
    "Normal": _normal,
    "LogNormal": _lognormal,
    "Gamma": _gamma,
}


def fast_family(dist: Any) -> Callable[..., Pieces] | None:
    """The hand-written pieces of ``dist``, or None."""
    import surpyval

    name = getattr(dist, "name", None)
    if name in FAMILIES and getattr(surpyval, name, None) is dist:
        return FAMILIES[name]
    return None


def _take(p: Prepared, mask: npt.NDArray) -> Prepared:
    return p[0][mask], p[1][mask]


class _Win:
    def __init__(self, xl: Any, xr: Any, n: Any, lo: float) -> None:
        xl = np.asarray(xl, dtype=float)
        xr = np.asarray(xr, dtype=float)
        n = np.asarray(n, dtype=float)
        with np.errstate(invalid="ignore"):
            lo_eval = np.isfinite(xl) & (xl > lo)
        hi = np.isfinite(xr)
        both = lo_eval & hi
        self.b_l, self.b_r = prepare(xl[both]), prepare(xr[both])
        self.b_n = n[both]
        lower = lo_eval & ~hi
        self.l_x, self.l_n = prepare(xl[lower]), n[lower]
        upper = ~lo_eval & hi
        self.u_x, self.u_n = prepare(xr[upper]), n[upper]


class FastLikelihood:
    """``theta -> (log-likelihood, gradient in theta)`` for a family in
    :data:`FAMILIES`, with no offset, limited-failure or zero-inflated
    form, on data that lie strictly inside the support."""

    def __init__(
        self, family: Callable[..., Pieces], data: Any, k: int, lo: float
    ) -> None:
        self.family = family
        self.k = k
        self.blocks = [
            (prepare(x), np.asarray(n, dtype=float), kind)
            for x, n, kind in (
                (data.x_o, data.n_o, "f"),
                (data.x_r, data.n_r, "S"),
                (data.x_l, data.n_l, "F"),
            )
            if len(x)
        ]
        self.interval = _Win(data.x_il, data.x_ir, data.n_i, lo)
        self.trunc = _Win(data.tl_unique, data.tr_unique, data.n_t_unique, lo)

    @classmethod
    def build(cls, dist: Any, data: Any) -> FastLikelihood | None:
        """The fast form for ``dist`` on ``data``, or None where it does not
        apply (another family, or a time at or below the support's lower
        edge or not finite, which the families' own functions handle)."""
        family = fast_family(dist)
        if family is None:
            return None
        lo = float(dist.support[0])
        for arr in (data.x_o, data.x_r, data.x_l):
            arr = np.asarray(arr, dtype=float)
            if not np.all(np.isfinite(arr)):
                return None
            if np.isfinite(lo) and np.any(arr <= lo):
                return None
        return cls(family, data, dist.k, lo if np.isfinite(lo) else -np.inf)

    def _block(
        self, x: Prepared, n: npt.NDArray, kind: str, P: Any
    ) -> tuple[float, npt.NDArray]:
        v, g = self.family(x, kind, *P)
        return float(v @ n), g @ n

    def _windows(self, w: _Win, P: Any) -> tuple[float, npt.NDArray]:
        f = self.family
        val = 0.0
        grad = np.zeros(self.k)
        if w.b_n.size:
            lsl, dlsl = f(w.b_l, "S", *P)
            tail = lsl < np.log(0.5)
            if np.any(tail):
                n = w.b_n[tail]
                a, da = lsl[tail], dlsl[:, tail]
                b, db = f(_take(w.b_r, tail), "S", *P)
                v = a + _log1mexp(np.maximum(a - b, 0.0))
                r = np.exp(b - a)
                d = (da - r * db) / (1.0 - r)
                val += float(np.sum(v * n))
                grad += d @ n
            body = ~tail
            if np.any(body):
                n = w.b_n[body]
                a, da = f(_take(w.b_l, body), "F", *P)
                b, db = f(_take(w.b_r, body), "F", *P)
                v = b + _log1mexp(np.maximum(b - a, 0.0))
                q = np.exp(a - b)
                d = (db - q * da) / (1.0 - q)
                val += float(np.sum(v * n))
                grad += d @ n
        if w.l_n.size:
            lsl, dlsl = f(w.l_x, "S", *P)
            tail = lsl < np.log(0.5)
            if np.any(tail):
                val += float(np.sum(lsl[tail] * w.l_n[tail]))
                grad += dlsl[:, tail] @ w.l_n[tail]
            body = ~tail
            if np.any(body):
                n = w.l_n[body]
                a, da = f(_take(w.l_x, body), "F", *P)
                v = _log1mexp(np.maximum(-a, 0.0))
                d = -da / np.expm1(-a)
                val += float(np.sum(v * n))
                grad += d @ n
        if w.u_n.size:
            v, g = self._block(w.u_x, w.u_n, "F", P)
            val += v
            grad += g
        return val, grad

    def __call__(self, P: Any) -> tuple[float, npt.NDArray]:
        val = 0.0
        grad = np.zeros(self.k)
        for x, n, kind in self.blocks:
            v, g = self._block(x, n, kind, P)
            val += v
            grad += g
        v, g = self._windows(self.interval, P)
        val += v
        grad += g
        v, g = self._windows(self.trunc, P)
        return val - v, grad - g
