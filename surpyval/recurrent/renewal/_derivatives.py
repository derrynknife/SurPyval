"""Hand-written derivatives for the renewal likelihoods' search (#728).

The renewal fits search on the likelihood's value and gradient together
(``value_and_grad`` of each family's likelihood). autograd gives the
exact gradient of any likelihood written for it (#710), but on these
models' small data one gradient cost 8 to 30 likelihoods, its per-op
overhead dominating, and a gradient search was slower than Nelder-Mead.
Here are the derivatives of the few functions the likelihoods are made
of, in plain numpy, for the lifetimes and baseline intensities whose
derivatives have a closed form: each returns a function's value with
its derivatives in its argument and in each parameter, and the
likelihoods chain them with their own recursions' derivatives.

``lifetime_derivatives(dist)`` and ``intensity_derivatives(baseline)``
return ``None`` for any other model, whose fits keep their Nelder-Mead
search.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from autograd import value_and_grad
from scipy.special import log_ndtr

from surpyval.recurrent.parametric.crow_amsaa import CrowAMSAA
from surpyval.recurrent.parametric.duane import Duane
from surpyval.univariate.parametric.distributions.lognormal import LogNormal
from surpyval.univariate.parametric.distributions.weibull import Weibull
from surpyval.utils.normal import gap_raw, ratio_raw

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)


class WeibullDerivatives:
    """The Weibull's ``log S``, ``log f`` and hazard, each as ``(value,
    d/dt, [d/dalpha, d/dbeta])``, for ``t >= 0`` (0 only for ``log
    S``)."""

    @staticmethod
    def _parts(t: np.ndarray, alpha: float, beta: float) -> tuple:
        z = t / alpha
        log_z = np.log(z)
        H = z**beta
        # H log z -> 0 as z -> 0
        H_log_z = np.where(z > 0, H * log_z, 0.0)
        # The hazard, by its own formula: its limit at t = 0 (0, 1 /
        # alpha or inf) where beta H / t would be 0 / 0
        h = (beta / alpha) * z ** (beta - 1)
        return log_z, H, H_log_z, h

    def log_sf(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        _, H, H_log_z, h = self._parts(t, alpha, beta)
        return -H, -h, [beta * H / alpha, -H_log_z]

    def log_end(self, t: np.ndarray, params: Any, failed: np.ndarray) -> tuple:
        """``log f`` where ``failed`` and ``log S`` elsewhere, at ``t >
        0``."""
        alpha, beta = params
        log_z, H, H_log_z, h = self._parts(t, alpha, beta)
        log_df = np.log(beta) - np.log(alpha) + (beta - 1) * log_z - H
        value = np.where(failed, log_df, -H)
        d_t = np.where(failed, (beta - 1) / t - h, -h)
        d_alpha = np.where(failed, beta * (H - 1) / alpha, beta * H / alpha)
        d_beta = np.where(failed, 1 / beta + log_z - H_log_z, -H_log_z)
        return value, d_t, [d_alpha, d_beta]

    def hf(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        log_z, _, _, h = self._parts(t, alpha, beta)
        return (
            h,
            h * (beta - 1) / t,
            [-beta * h / alpha, h * (1 / beta + log_z)],
        )


class LogNormalDerivatives:
    """The LogNormal's ``log S``, ``log f`` and hazard, as
    ``WeibullDerivatives`` gives the Weibull's, for ``t > 0``."""

    @staticmethod
    def _parts(t: np.ndarray, mu: float, sigma: float) -> tuple:
        z = (np.log(t) - mu) / sigma
        log_sf = log_ndtr(-z)
        # phi(z) / Phi(-z), the normal hazard at z, and its excess over z
        # (the slope of its log), each accurate far into the tail
        # (``surpyval.utils.normal``, as autograd's derivatives are)
        mills = ratio_raw(-z)
        excess = gap_raw(-z)
        return z, log_sf, mills, excess

    def log_sf(self, t: np.ndarray, params: Any) -> tuple:
        mu, sigma = params
        z, log_sf, mills, _ = self._parts(t, mu, sigma)
        return log_sf, -mills / (sigma * t), [mills / sigma, mills * z / sigma]

    def log_end(self, t: np.ndarray, params: Any, failed: np.ndarray) -> tuple:
        mu, sigma = params
        z, log_sf, mills, _ = self._parts(t, mu, sigma)
        log_t = np.log(t)
        log_df = -log_t - np.log(sigma) - _LOG_SQRT_2PI - 0.5 * z * z
        value = np.where(failed, log_df, log_sf)
        d_t = np.where(failed, -(1 + z / sigma) / t, -mills / (sigma * t))
        d_mu = np.where(failed, z / sigma, mills / sigma)
        d_sigma = np.where(failed, (z * z - 1) / sigma, mills * z / sigma)
        return value, d_t, [d_mu, d_sigma]

    def hf(self, t: np.ndarray, params: Any) -> tuple:
        mu, sigma = params
        z, _, mills, excess = self._parts(t, mu, sigma)
        h = mills / (sigma * t)
        # d log h / dz = mills - z
        return (
            h,
            h * (excess / sigma - 1) / t,
            [-h * excess / sigma, -h * (excess * z + 1) / sigma],
        )


class CrowAMSAADerivatives:
    """The power law's cumulative intensity ``(t / alpha)**beta`` and
    intensity, each as ``(value, [d/dalpha, d/dbeta])``, for ``t >= 0``
    (0 only for the cumulative intensity)."""

    def cif(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        z = t / alpha
        value = z**beta
        H_log_z = np.where(z > 0, value * np.log(np.where(z > 0, z, 1.0)), 0.0)
        return value, [-beta * value / alpha, H_log_z]

    def iif(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        value = (beta / alpha**beta) * t ** (beta - 1)
        return value, [
            -beta * value / alpha,
            value * (1 / beta + np.log(t / alpha)),
        ]


class DuaneDerivatives:
    """Duane's cumulative intensity ``b t**alpha`` and intensity, as
    ``CrowAMSAADerivatives`` gives the power law's."""

    def cif(self, t: np.ndarray, params: Any) -> tuple:
        a, b = params
        power = t**a
        log_t = np.log(np.where(t > 0, t, 1.0))
        return b * power, [b * power * log_t, power]

    def iif(self, t: np.ndarray, params: Any) -> tuple:
        a, b = params
        value = a * b * t ** (a - 1.0)
        return value, [value * (1 / a + np.log(t)), a * t ** (a - 1.0)]


def lifetime_derivatives(dist: Any) -> "Any | None":
    """The hand-written derivatives of the lifetime ``dist``, or ``None``
    where there are none (exactly the package's own Weibull and
    LogNormal: a subclass may define its functions differently)."""
    if type(dist) is type(Weibull):
        return WeibullDerivatives()
    if type(dist) is type(LogNormal):
        return LogNormalDerivatives()
    return None


def intensity_derivatives(baseline: Any) -> "Any | None":
    """The hand-written derivatives of the baseline intensity
    ``baseline``, or ``None`` where there are none."""
    if type(baseline) is type(CrowAMSAA):
        return CrowAMSAADerivatives()
    if type(baseline) is type(Duane):
        return DuaneDerivatives()
    return None


def negated(found: "tuple | None", neg_ll: Any, params: np.ndarray) -> tuple:
    """``(neg_ll(params), its gradient)`` from a likelihood's hand-written
    ``found = (ll, d ll / d restoration, d ll / d params)``; where that is
    ``None`` (a point outside the hand-written terms' domain), from
    autograd, with a gradient of nan where autograd has none either."""
    if found is not None:
        ll, d_r, d_p = found
        return -ll, -np.concatenate([[d_r], d_p])
    params = np.asarray(params, dtype=float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            value, grad = value_and_grad(neg_ll)(params)
            return float(value), np.asarray(grad, dtype=float)
        except Exception:
            return float(neg_ll(params)), np.full(params.size, np.nan)
