"""Hand-written derivatives for the renewal likelihoods' search (#728).

The renewal fits search on the likelihood's value and gradient together
(``value_and_grad`` of each family's likelihood). autograd gives the
exact gradient of any likelihood written for it (#710), but on these
models' small data one gradient cost 8 to 30 likelihoods, its per-op
overhead dominating, and a gradient search was slower than Nelder-Mead.
Here are the derivatives of the few functions the likelihoods are made
of, in plain numpy: each returns a function's value with its
derivatives in its argument and in each parameter, and the likelihoods
chain them with their own recursions' derivatives. They have a closed
form for every lifetime and baseline here but the Gamma's log S in its
shape, which is taken from differences of its log and from its
asymptotic series (``GammaDerivatives``, #746). The ExpoWeibull's,
Normal's, Gumbel's and Logistic's followed (#760).

``lifetime_derivatives(dist)`` and ``intensity_derivatives(baseline)``
return ``None`` for any other model, whose fits keep their Nelder-Mead
search.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from autograd import value_and_grad
from scipy.special import digamma, expit, gammaln, log_ndtr

from surpyval.recurrent.parametric.cox_lewis import CoxLewis
from surpyval.recurrent.parametric.crow_amsaa import CrowAMSAA
from surpyval.recurrent.parametric.duane import Duane
from surpyval.recurrent.parametric.hpp import HPP
from surpyval.univariate.parametric.distributions.expo_weibull import (
    ExpoWeibull,
    log_forms,
)
from surpyval.univariate.parametric.distributions.exponential import (
    Exponential,
)
from surpyval.univariate.parametric.distributions.gamma import Gamma
from surpyval.univariate.parametric.distributions.gumbel import Gumbel
from surpyval.univariate.parametric.distributions.logistic import Logistic
from surpyval.univariate.parametric.distributions.loglogistic import (
    LogLogistic,
)
from surpyval.univariate.parametric.distributions.lognormal import LogNormal
from surpyval.univariate.parametric.distributions.normal import Normal
from surpyval.univariate.parametric.distributions.rayleigh import Rayleigh
from surpyval.univariate.parametric.distributions.weibull import Weibull
from surpyval.utils.autograd_gamma_compat import gammainccln
from surpyval.utils.normal import gap_raw, ratio_raw

_LOG_SQRT_2PI = 0.5 * np.log(2.0 * np.pi)

#: The Gamma's shape derivatives: the relative step in the shape of the
#: central differences, and the standard-Gamma times past which (and
#: past ``_GAMMA_TAIL_SHAPES`` times the shape) the asymptotic series is
#: summed instead (``GammaDerivatives``).
_GAMMA_STEP = 1e-3
_GAMMA_TAIL = 50.0
_GAMMA_TAIL_SHAPES = 20.0

#: The ExpoWeibull's ``(t / alpha)**beta`` past which its hazard takes
#: ``-log(1 - e**-u) / e**-u`` as 1 (``ExpoWeibull.hf``'s own threshold)
_EW_T_LARGE = 40.0


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


class ExponentialDerivatives:
    """The Exponential's ``log S``, ``log f`` and hazard, as
    ``WeibullDerivatives`` gives the Weibull's."""

    def log_sf(self, t: np.ndarray, params: Any) -> tuple:
        (rate,) = params
        return -rate * t, np.full(t.shape, -rate), [-t]

    def log_end(self, t: np.ndarray, params: Any, failed: np.ndarray) -> tuple:
        (rate,) = params
        value = np.where(failed, np.log(rate) - rate * t, -rate * t)
        d_rate = np.where(failed, 1 / rate - t, -t)
        return value, np.full(t.shape, -rate), [d_rate]

    def hf(self, t: np.ndarray, params: Any) -> tuple:
        (rate,) = params
        return np.full(t.shape, rate), np.zeros(t.shape), [np.ones(t.shape)]


class RayleighDerivatives:
    """The Rayleigh's ``log S``, ``log f`` and hazard, as
    ``WeibullDerivatives`` gives the Weibull's, for ``t > 0``."""

    def log_sf(self, t: np.ndarray, params: Any) -> tuple:
        (sigma,) = params
        return (
            -0.5 * (t / sigma) ** 2,
            -t / sigma**2,
            [t * t / sigma**3],
        )

    def log_end(self, t: np.ndarray, params: Any, failed: np.ndarray) -> tuple:
        (sigma,) = params
        log_sf = -0.5 * (t / sigma) ** 2
        log_df = np.log(t) - 2 * np.log(sigma) + log_sf
        value = np.where(failed, log_df, log_sf)
        d_t = np.where(failed, 1 / t, 0.0) - t / sigma**2
        d_sigma = np.where(failed, -2 / sigma, 0.0) + t * t / sigma**3
        return value, d_t, [d_sigma]

    def hf(self, t: np.ndarray, params: Any) -> tuple:
        (sigma,) = params
        return (
            t / sigma**2,
            np.full(t.shape, 1 / sigma**2),
            [-2 * t / sigma**3],
        )


class LogLogisticDerivatives:
    """The LogLogistic's ``log S``, ``log f`` and hazard, as
    ``WeibullDerivatives`` gives the Weibull's, for ``t > 0``: each a
    function of ``z = beta log(t / alpha)``. The values are the
    distribution's own."""

    @staticmethod
    def _parts(t: np.ndarray, alpha: float, beta: float) -> tuple:
        log_ratio = np.log(t) - np.log(alpha)
        z = beta * log_ratio
        # dz / dt, dz / dalpha and dz / dbeta, with F and S at z
        return beta / t, -beta / alpha, log_ratio, expit(z), expit(-z)

    def log_sf(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        z_t, z_alpha, z_beta, F, _ = self._parts(t, alpha, beta)
        value = LogLogistic.log_sf(t, alpha, beta)
        return value, -F * z_t, [-F * z_alpha, -F * z_beta]

    def log_end(self, t: np.ndarray, params: Any, failed: np.ndarray) -> tuple:
        alpha, beta = params
        z_t, z_alpha, z_beta, F, S = self._parts(t, alpha, beta)
        value = np.where(
            failed,
            LogLogistic.log_df(t, alpha, beta),
            LogLogistic.log_sf(t, alpha, beta),
        )
        # d log f / dz = S - F, d log S / dz = -F
        slope = np.where(failed, S - F, -F)
        d_t = slope * z_t - np.where(failed, 1 / t, 0.0)
        d_beta = slope * z_beta + np.where(failed, 1 / beta, 0.0)
        return value, d_t, [slope * z_alpha, d_beta]

    def hf(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        z_t, z_alpha, z_beta, _, S = self._parts(t, alpha, beta)
        h = LogLogistic.hf(t, alpha, beta)
        # h = (beta / t) F: d log h / dz = S
        return (
            h,
            h * (S * z_t - 1 / t),
            [h * S * z_alpha, h * (S * z_beta + 1 / beta)],
        )


class ExpoWeibullDerivatives:
    """The ExpoWeibull's ``log S``, ``log f`` and hazard, as
    ``WeibullDerivatives`` gives the Weibull's, for ``t > 0``.

    Each is a function of ``l = log u`` with ``u = (t / alpha)**beta``
    (and of ``mu``), taken from the distribution's own log forms
    (``log_forms``, which keep their digits in both tails), with ``dl /
    dt = beta / t``, ``dl / dalpha = -beta / alpha`` and ``dl / dbeta =
    log(t / alpha)``. The values are the distribution's own. With ``g =
    1 - e**-u`` and ``w = d log g / dl = u e**-u / g``:

    - ``d log S / dl = -mu w F / S`` and ``d log S / dmu = F log g / S``
      (as ``ExpoWeibull``'s own ``_partials``);
    - ``d log f / dl = (mu - 1) w + 1 - u`` and ``d log f / dmu = 1 /
      mu + log g``;
    - ``d log h / dl = (mu - 1) w + 1 + u (mu w F / (u S) - 1)`` and
      ``d log h / dmu = (1 - r / (1 - e**-r)) / mu`` with ``r = -log
      F``, each from ``log f - log S`` with the two terms that cancel in
      the right tail (``u`` and ``mu w F / S``, ``1 / mu`` and ``log g /
      S``) taken together, by ``expm1`` of the log of their ratio.
    """

    @staticmethod
    def _parts(t: np.ndarray, alpha: float, beta: float, mu: float) -> dict:
        p = log_forms(t, alpha, beta, mu, xp=np)
        u = p["t"]
        large = u > _EW_T_LARGE
        p["log_u"] = beta * (p["log_x"] - np.log(alpha))
        # dl / dt, dl / dalpha, dl / dbeta
        p["l_t"] = beta / t
        p["l_alpha"] = -beta / alpha
        p["l_beta"] = p["log_x"] - np.log(alpha)
        p["w"] = np.exp(-p["ratio_g"] - u)
        # log(-log g / e**-u), 0 to double precision in the right tail
        p["log_q"] = np.where(
            large, 0.0, p["log_nl"] + np.where(large, 0.0, u)
        )
        return p

    @staticmethod
    def _chain(
        p: dict, t: np.ndarray, beta: float, d_l: Any, d_mu: Any, own: Any
    ) -> tuple:
        """``(d/dt, [d/dalpha, d/dbeta, d/dmu])`` from ``d/dl`` and
        ``d/dmu``, with the slopes of the ``beta / t`` of ``f`` and ``h``
        where ``own`` (1 there, 0 elsewhere)."""
        d_t = d_l * p["l_t"] - own / t
        d_beta = d_l * p["l_beta"] + own / beta
        return d_t, [d_l * p["l_alpha"], d_beta, d_mu]

    @staticmethod
    def _log_sf_slopes(p: dict, mu: float) -> tuple:
        log_f_over_s = p["log_ff"] - p["ratio_r"]
        # log(-log g) + u first: they cancel exactly in the right tail
        # (as ``ExpoWeibull``'s own ``_partials`` takes them)
        log_nl_u = -p["log_nl"] - p["t"]
        d_l = -np.exp(log_f_over_s - p["ratio_g"] + log_nl_u)
        return d_l, np.exp(log_f_over_s) / mu

    def log_sf(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta, mu = params
        p = self._parts(t, alpha, beta, mu)
        d_l, d_mu = self._log_sf_slopes(p, mu)
        d_t, d_p = self._chain(p, t, beta, d_l, d_mu, 0.0)
        return p["log_sf"], d_t, d_p

    def log_end(self, t: np.ndarray, params: Any, failed: np.ndarray) -> tuple:
        alpha, beta, mu = params
        p = self._parts(t, alpha, beta, mu)
        # log f as ExpoWeibull.log_df takes it
        log_df = (
            np.log(beta)
            + np.log(mu)
            - p["log_x"]
            + mu * p["log_g"]
            - p["ratio_g"]
            - p["t"]
        )
        sf_l, sf_mu = self._log_sf_slopes(p, mu)
        df_l = (mu - 1.0) * p["w"] + 1.0 - p["t"]
        df_mu = 1.0 / mu + p["log_g"]
        d_t, d_p = self._chain(
            p,
            t,
            beta,
            np.where(failed, df_l, sf_l),
            np.where(failed, df_mu, sf_mu),
            np.where(failed, 1.0, 0.0),
        )
        return np.where(failed, log_df, p["log_sf"]), d_t, d_p

    def hf(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta, mu = params
        p = self._parts(t, alpha, beta, mu)
        u, log_g, ratio_r = p["t"], p["log_g"], p["ratio_r"]
        # h as ExpoWeibull.hf takes it
        log_hf = (
            np.log(beta)
            - p["log_x"]
            + mu * log_g
            - p["ratio_g"]
            - ratio_r
            - p["log_q"]
        )
        h = np.exp(log_hf)
        # log(mu w F / (u S)), the ratio of the two terms that cancel
        ratio = (mu - 1.0) * log_g - ratio_r - p["log_q"]
        excess = np.where(
            ratio < 1.0,
            u * np.expm1(np.minimum(ratio, 1.0)),
            np.exp(p["log_u"] + np.maximum(ratio, 1.0)) - u,
        )
        d_l = (mu - 1.0) * p["w"] + 1.0 + excess
        d_mu = -np.expm1(-ratio_r) / mu
        d_t, d_p = self._chain(p, t, beta, d_l, d_mu, 1.0)
        return h, h * d_t, [h * d for d in d_p]


class _LocationScaleDerivatives:
    """The ``log S``, ``log f`` and hazard of a location-scale lifetime
    on the whole line, as ``WeibullDerivatives`` gives the Weibull's:
    each a function of ``z = (t - mu) / sigma``, from the standard
    distribution's slopes (``_slopes``). The values are the
    distribution's own (``dist``)."""

    #: A lifetime on the whole line: its log S at 0 is not 0, and an age
    #: of 0 is inside the terms' domain (``VirtualAgeLikelihood``).
    real_line = True
    dist: Any = None

    @staticmethod
    def _slopes(z: np.ndarray) -> tuple:
        """``d log S / dz``, ``d log f / dz`` and ``d log h / dz`` of the
        standard distribution."""
        raise NotImplementedError

    def log_sf(self, t: np.ndarray, params: Any) -> tuple:
        mu, sigma = params
        z = (t - mu) / sigma
        slope = self._slopes(z)[0] / sigma
        value = self.dist.log_sf(t, mu, sigma)
        return value, slope, [-slope, -z * slope]

    def log_end(self, t: np.ndarray, params: Any, failed: np.ndarray) -> tuple:
        mu, sigma = params
        z = (t - mu) / sigma
        sf_z, df_z, _ = self._slopes(z)
        slope = np.where(failed, df_z, sf_z) / sigma
        value = np.where(
            failed,
            self.dist.log_df(t, mu, sigma),
            self.dist.log_sf(t, mu, sigma),
        )
        d_sigma = -z * slope - np.where(failed, 1.0 / sigma, 0.0)
        return value, slope, [-slope, d_sigma]

    def hf(self, t: np.ndarray, params: Any) -> tuple:
        mu, sigma = params
        z = (t - mu) / sigma
        h = self.dist.hf(t, mu, sigma)
        slope = h * self._slopes(z)[2] / sigma
        return h, slope, [-slope, -z * slope - h / sigma]


class NormalDerivatives(_LocationScaleDerivatives):
    """The Normal's (``_LocationScaleDerivatives``)."""

    dist = Normal

    @staticmethod
    def _slopes(z: np.ndarray) -> tuple:
        # The normal hazard at z, phi(z) / Phi(-z), and its excess over z
        # (the slope of its log), accurate far into the tail
        # (``LogNormalDerivatives``)
        mills = ratio_raw(-z)
        return -mills, -z, gap_raw(-z)


class GumbelDerivatives(_LocationScaleDerivatives):
    """The Gumbel's (``_LocationScaleDerivatives``): ``log S = -e**z``,
    ``log f = z - e**z``, ``h = e**z / sigma``."""

    dist = Gumbel

    @staticmethod
    def _slopes(z: np.ndarray) -> tuple:
        H = np.exp(z)
        return -H, 1.0 - H, np.ones_like(z)


class LogisticDerivatives(_LocationScaleDerivatives):
    """The Logistic's (``_LocationScaleDerivatives``): ``d log S / dz =
    -F``, ``d log f / dz = S - F``, ``h = F / sigma``."""

    dist = Logistic

    @staticmethod
    def _slopes(z: np.ndarray) -> tuple:
        F, S = expit(z), expit(-z)
        return -F, S - F, S


def _upper_gamma_series(a: float, y: np.ndarray) -> tuple:
    """The asymptotic series of the upper incomplete gamma, ``Q(a, y) =
    y**(a - 1) e**-y S / Gamma(a)`` with ``S = sum_k u_k``, ``u_0 = 1``
    and ``u_k = u_{k-1} (a - k) / y``, for ``y`` far above ``a``
    (``GammaDerivatives``): ``(S - 1, dS / da, -y dS / dy)``. Each term's
    derivative in ``a`` is taken by its own recursion, which stays exact
    where a term is 0 (a whole ``a``)."""
    u = np.ones_like(y)
    du = np.zeros_like(y)
    s, ds, ks = np.zeros_like(y), np.zeros_like(y), np.zeros_like(y)
    for k in range(1, 60):
        u, du = u * (a - k) / y, (du * (a - k) + u) / y
        s += u
        ds += du
        ks += k * u
        if np.all(np.abs(u) * k + np.abs(du) <= 1e-17 * np.abs(ds)):
            break
    return s, ds, ks


class GammaDerivatives:
    """The Gamma's ``log S``, ``log f`` and hazard, as
    ``WeibullDerivatives`` gives the Weibull's, for ``t > 0``.

    ``log S = log Q(alpha, y)`` at ``y = beta t``, the regularised upper
    incomplete gamma, whose derivative in its shape has no closed form.
    It is taken from the central differences (four points, to the fourth
    order) of ``log(-log Q)`` in ``alpha``, with a step of ``1e-3`` of
    it: that log is smooth in ``alpha`` where ``Q`` is near 1 (``log Q ~
    -y**alpha / Gamma(alpha + 1)``, which a difference of ``log Q``
    itself takes only to 1e-6 there), and the differences are good to
    1e-11 against mpmath from ``Q`` near 1 to ``y`` of 50. Above that
    (and 20 times ``alpha``) the differences lose digits to the size of
    ``log Q`` (``-y``), and the derivatives come from its asymptotic
    series instead (``_upper_gamma_series``), to 1e-15. There the
    hazard's ratio to ``beta``, ``1 / S``, is the series too: ``f / S``
    loses its excess over 1 to rounding (by a ``y`` of 1e9, 1e-8 of it),
    and a Kijima-II ``q`` of 3 ages an item far past that.
    """

    @staticmethod
    def _parts(t: np.ndarray, alpha: float, beta: float) -> tuple:
        """``log Q`` with its slope in ``alpha``, and the log of the
        hazard's ratio to ``beta``, ``log r``, with its slopes in ``y``
        and in ``alpha``."""
        y = beta * t
        log_y = np.log(y)
        tail = (y > _GAMMA_TAIL) & (y > _GAMMA_TAIL_SHAPES * alpha)
        body = np.flatnonzero(~tail)
        h = _GAMMA_STEP * alpha
        shapes = np.repeat(
            [alpha, alpha - 2 * h, alpha - h, alpha + h, alpha + 2 * h],
            [y.size] + [body.size] * 4,
        )
        logs = np.asarray(
            gammainccln(shapes, np.concatenate([y, np.tile(y[body], 4)])),
            dtype=float,
        )
        log_q = logs[: y.size]
        # d log Q / d alpha = log Q * d log(-log Q) / d alpha; 0 where Q
        # rounds to 1 at a point (the slope is below 1e-300 there)
        g = np.log(-logs[y.size :].reshape(4, -1))
        slope = log_q[body] * (8 * (g[2] - g[1]) - (g[3] - g[0])) / (12 * h)
        d_alpha = np.zeros(y.size)
        d_alpha[body] = np.where(np.isfinite(slope), slope, 0.0)
        log_density = (alpha - 1) * log_y - y - gammaln(alpha)
        log_r = log_density - log_q
        r = np.exp(log_r)
        r_dy = (alpha - 1) / y - 1 + r
        r_dalpha = log_y - digamma(alpha) - d_alpha
        if np.any(tail):
            s, ds, ks = _upper_gamma_series(alpha, y[tail])
            d_alpha[tail] = log_y[tail] - digamma(alpha) + ds / (1 + s)
            r[tail] = 1 / (1 + s)
            r_dy[tail] = ks / (y[tail] * (1 + s))
            r_dalpha[tail] = -ds / (1 + s)
        return log_q, d_alpha, r, r_dy, r_dalpha, tail

    def log_sf(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        log_q, d_alpha, r, *_ = self._parts(t, alpha, beta)
        return log_q, -beta * r, [d_alpha, -t * r]

    def log_end(self, t: np.ndarray, params: Any, failed: np.ndarray) -> tuple:
        alpha, beta = params
        # log f as Gamma.log_df takes it
        log_t = np.log(t)
        log_scale = alpha * np.log(beta) - gammaln(alpha)
        value = log_scale + (alpha - 1) * log_t - beta * t
        d_t = (alpha - 1) / t - beta
        d_alpha = np.log(beta) - digamma(alpha) + log_t
        d_beta = alpha / beta - t
        censored = np.flatnonzero(~failed)
        if censored.size:
            value, d_t, d_alpha, d_beta = (
                np.array(v, dtype=float) for v in (value, d_t, d_alpha, d_beta)
            )
            tc = t[censored]
            log_q, q_alpha, r, *_ = self._parts(tc, alpha, beta)
            value[censored] = log_q
            d_t[censored] = -beta * r
            d_alpha[censored] = q_alpha
            d_beta[censored] = -tc * r
        return value, d_t, [d_alpha, d_beta]

    def hf(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        log_q, _, r, r_dy, r_dalpha, tail = self._parts(t, alpha, beta)
        # f / S as Gamma.hf takes it; in the tail beta r, the same but
        # where f / S has lost its excess over beta to rounding (f and S
        # are each of size e**-y)
        log_scale = alpha * np.log(beta) - gammaln(alpha)
        h = np.exp(log_scale + (alpha - 1) * np.log(t) - beta * t - log_q)
        h = np.where(tail, beta * r, h)
        return (
            h,
            h * beta * r_dy,
            [h * r_dalpha, h * (1 / beta + t * r_dy)],
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


class HPPDerivatives:
    """The homogeneous Poisson process's cumulative intensity ``rate t``
    and intensity ``rate``, as ``CrowAMSAADerivatives`` gives the power
    law's."""

    def cif(self, t: np.ndarray, params: Any) -> tuple:
        (rate,) = params
        t = np.asarray(t, dtype=float)
        return rate * t, [t]

    def iif(self, t: np.ndarray, params: Any) -> tuple:
        (rate,) = params
        ones = np.where(np.isnan(t), np.nan, 1.0)
        return ones * rate, [ones]


class CoxLewisDerivatives:
    """The Cox-Lewis cumulative intensity ``e**alpha (e**(beta t) - 1) /
    beta`` and intensity ``e**(alpha + beta t)``, as
    ``CrowAMSAADerivatives`` gives the power law's."""

    #: ``(k - 1) / k!`` for ``k = 2, 3, ...``: the series of the slope of
    #: ``(e**(beta t) - 1) / beta`` in ``beta``, over ``t**2``, in powers
    #: of ``beta t`` (to 1e-20 below ``|beta t|`` of 1/2)
    _SERIES = tuple(
        (k - 1) / float(np.prod(np.arange(1, k + 1))) for k in range(2, 20)
    )

    def cif(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        t = np.asarray(t, dtype=float)
        scale = np.exp(alpha)
        u = beta * t
        small = np.abs(u) < 0.5
        if beta == 0:
            over = t
        else:
            over = np.expm1(u) / beta
        # (t e**u - over) / beta, the slope in beta, loses digits to
        # cancellation as u -> 0: there by its series
        series = np.zeros_like(u)
        for coeff in reversed(self._SERIES):
            series = series * u + coeff
        with np.errstate(all="ignore"):
            direct = (t * np.exp(u) - over) / np.where(small, 1.0, beta)
        slope = np.where(small, t * t * series, direct)
        value = scale * over
        return value, [value, scale * slope]

    def iif(self, t: np.ndarray, params: Any) -> tuple:
        alpha, beta = params
        value = np.exp(alpha + beta * t)
        return value, [value, t * value]

    @staticmethod
    def search_floor(t: np.ndarray) -> list:
        """The unit of each parameter's search (``GradientSearch``):
        ``alpha`` is a log, of unit 1, and ``beta`` a rate per unit
        time, of unit one over the longest time ``t`` (BFGS's steps of 1
        in ``beta`` overflowed ``e**(beta t)`` on data in thousands of
        hours, and the search stopped where it started)."""
        longest = float(np.max(np.abs(t), initial=0.0))
        return [1.0, min(1.0, 1.0 / longest) if longest > 0 else 1.0]


def lifetime_derivatives(dist: Any) -> "Any | None":
    """The hand-written derivatives of the lifetime ``dist``, or ``None``
    where there are none (exactly the package's own Weibull, LogNormal,
    Gamma, LogLogistic, Exponential, Rayleigh, ExpoWeibull, Normal, Gumbel
    and Logistic: a subclass may define its functions differently)."""
    for life, terms in (
        (Weibull, WeibullDerivatives),
        (LogNormal, LogNormalDerivatives),
        (Gamma, GammaDerivatives),
        (LogLogistic, LogLogisticDerivatives),
        (Exponential, ExponentialDerivatives),
        (Rayleigh, RayleighDerivatives),
        (ExpoWeibull, ExpoWeibullDerivatives),
        (Normal, NormalDerivatives),
        (Gumbel, GumbelDerivatives),
        (Logistic, LogisticDerivatives),
    ):
        if type(dist) is type(life):
            return terms()
    return None


def intensity_derivatives(baseline: Any) -> "Any | None":
    """The hand-written derivatives of the baseline intensity
    ``baseline``, or ``None`` where there are none."""
    if type(baseline) is type(CrowAMSAA):
        return CrowAMSAADerivatives()
    if type(baseline) is type(Duane):
        return DuaneDerivatives()
    if type(baseline) is type(CoxLewis):
        return CoxLewisDerivatives()
    if type(baseline) is type(HPP):
        return HPPDerivatives()
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
