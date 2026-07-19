"""Stochastic-process degradation models: Wiener and Gamma processes.

Unlike the general-path (pseudo-failure-time) approach in
:mod:`surpyval.degradation.degradation_analysis`, these model the degradation
*increments* directly as a stochastic process. Each has a first-passage
failure-time distribution derived analytically from the process, so the life
distribution comes straight from the fitted process rather than via noisy
pseudo failure times, and irregular measurement spacing is handled naturally
(every increment carries its own ``dt``).

Two complementary processes are provided, chosen by the physics of the
degradation:

* :class:`WienerProcess` -- Brownian motion with drift, ``W(t) = mu*t +
  sigma*B(t)``. Increments are Gaussian and may go up or down, so it suits
  *non-monotone* / noisy degradation signals (sensor drift, measurement noise).
  Its first passage to a threshold is a closed-form **Inverse Gaussian** law.
* :class:`GammaProcess` -- a sum of independent non-negative increments, so the
  path is *strictly monotone increasing*. It suits genuinely irreversible
  damage (wear, corrosion, crack growth, fatigue). Its first-passage
  distribution comes from the (regularised) incomplete gamma function.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaincc, gammaln
from scipy.stats import norm

__all__ = [
    "WienerProcess",
    "WienerProcessModel",
    "GammaProcess",
    "GammaProcessModel",
    "ProcessRUL",
]


def _increments(x, y, i):
    """
    Validate degradation data and return the pooled per-unit increments.

    Returns ``(dt, dy)`` arrays of the time and degradation increments between
    consecutive (time-ordered) measurements within each unit, pooled across
    units. Requires strictly increasing times within a unit.
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    y = np.atleast_1d(np.asarray(y, dtype=float))
    i = np.atleast_1d(np.asarray(i))
    if x.ndim != 1 or y.ndim != 1 or i.ndim != 1:
        raise ValueError("x, y, and i must be one dimensional")
    if not (len(x) == len(y) == len(i)):
        raise ValueError("x, y, and i must have the same length")
    if len(x) == 0:
        raise ValueError("x, y, and i must not be empty")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("x and y must contain only finite values")

    dts, dys = [], []
    for unit in np.unique(i):
        mask = i == unit
        xu, yu = x[mask], y[mask]
        order = np.argsort(xu)
        xu, yu = xu[order], yu[order]
        if len(xu) < 2:
            continue
        dt = np.diff(xu)
        dy = np.diff(yu)
        if np.any(dt <= 0):
            raise ValueError(
                "unit {!r} has repeated or non-increasing measurement "
                "times; times must be strictly increasing within a "
                "unit".format(unit)
            )
        dts.append(dt)
        dys.append(dy)

    if not dts:
        raise ValueError(
            "no unit has two or more measurements; at least one increment "
            "is required to fit a process model"
        )
    return np.concatenate(dts), np.concatenate(dys)


class ProcessRUL:
    """Remaining-useful-life summary from a fitted process model.

    Attributes
    ----------
    rul : float
        Median remaining useful life from the current state.
    rul_interval : tuple of float
        Equal-tailed ``1 - alpha_ci`` interval for the remaining life.
    prob_already_failed : float
        Probability the unit has already crossed the threshold (i.e. the
        current degradation is at or beyond it).
    """

    def __init__(self, rul, rul_interval, prob_already_failed, alpha_ci):
        self.rul = rul
        self.rul_interval = rul_interval
        self.prob_already_failed = prob_already_failed
        self.alpha_ci = alpha_ci

    def __repr__(self):
        lo, hi = self.rul_interval
        return (
            "ProcessRUL(rul={:.4g}, interval=({:.4g}, {:.4g}), "
            "prob_already_failed={:.4g})".format(
                self.rul, lo, hi, self.prob_already_failed
            )
        )


# --------------------------------------------------------------------------
# Wiener process
# --------------------------------------------------------------------------


class WienerProcessModel:
    """
    A fitted Wiener-process degradation model, ``W(t) = mu*t + sigma*B(t)``.

    The first passage of the process to the failure ``threshold`` (from an
    assumed degradation of zero at ``t = 0``) is Inverse-Gaussian distributed
    with mean ``threshold / mu`` and shape ``threshold**2 / sigma**2``, and the
    failure-time methods below evaluate that distribution. A positive drift
    ``mu`` is required for a proper (non-defective) life distribution.

    Parameters
    ----------
    mu : float
        Fitted drift (mean degradation rate).
    sigma : float
        Fitted diffusion (volatility) coefficient.
    threshold : float
        The degradation level defining failure.
    """

    def __init__(self, mu, sigma, threshold):
        self.mu = float(mu)
        self.sigma = float(sigma)
        self.threshold = float(threshold)
        self.params = np.array([self.mu, self.sigma])
        self.param_names = ["mu", "sigma"]

    def _ig(self, distance):
        # Inverse-Gaussian (mean nu, shape lam) parameters for first passage
        # over ``distance`` at drift mu / diffusion sigma.
        nu = distance / self.mu
        lam = distance**2 / self.sigma**2
        return nu, lam

    def _ig_cdf(self, t, distance):
        t = np.asarray(t, dtype=float)
        nu, lam = self._ig(distance)
        out = np.zeros_like(t, dtype=float)
        pos = t > 0
        tp = t[pos]
        root = np.sqrt(lam / tp)
        cdf = norm.cdf(root * (tp / nu - 1.0)) + np.exp(
            2.0 * lam / nu
        ) * norm.cdf(-root * (tp / nu + 1.0))
        out[pos] = cdf
        return out

    def ff(self, t):
        """Failure (CDF) of the first-passage time to the threshold."""
        scalar = np.isscalar(t)
        res = self._ig_cdf(np.atleast_1d(t), self.threshold)
        return float(res[0]) if scalar else res

    def sf(self, t):
        """Survival function of the first-passage time."""
        res = 1.0 - self.ff(np.atleast_1d(t))
        return float(res[0]) if np.isscalar(t) else res

    def df(self, t):
        """Density of the first-passage (Inverse-Gaussian) time."""
        scalar = np.isscalar(t)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        nu, lam = self._ig(self.threshold)
        out = np.zeros_like(t)
        pos = t > 0
        tp = t[pos]
        out[pos] = np.sqrt(lam / (2.0 * np.pi * tp**3)) * np.exp(
            -lam * (tp - nu) ** 2 / (2.0 * nu**2 * tp)
        )
        return float(out[0]) if scalar else out

    def hf(self, t):
        """Hazard function of the first-passage time."""
        return self.df(t) / self.sf(t)

    def Hf(self, t):
        """Cumulative hazard of the first-passage time."""
        return -np.log(self.sf(t))

    def mean(self):
        """Mean time to failure (``threshold / mu``)."""
        return self.threshold / self.mu

    def qf(self, p):
        """Quantile (inverse CDF) of the first-passage time."""
        p = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.array([self._quantile(pi, self.threshold) for pi in p])
        return float(out[0]) if out.shape == (1,) else out

    def _quantile(self, p, distance):
        if not (0.0 < p < 1.0):
            if p <= 0.0:
                return 0.0
            return np.inf
        nu = distance / self.mu
        # bracket around the mean and expand until it contains the quantile
        hi = nu
        while self._ig_cdf(np.array([hi]), distance)[0] < p:
            hi *= 2.0
            if hi > 1e12:
                return np.inf
        return brentq(
            lambda t: self._ig_cdf(np.array([t]), distance)[0] - p,
            1e-12,
            hi,
        )

    def random(self, size, random_state=None):
        """Draw first-passage (failure) times from the fitted model."""
        rng = np.random.default_rng(random_state)
        nu, lam = self._ig(self.threshold)
        return rng.wald(nu, lam, size=size)

    def predict_rul(self, current_degradation, alpha_ci=0.05):
        """
        Remaining useful life given the current degradation level.

        Because Wiener increments are independent, the remaining first passage
        over the residual distance ``threshold - current_degradation`` is again
        Inverse-Gaussian; its median and interval are returned.

        Parameters
        ----------
        current_degradation : float
            The unit's current degradation level.
        alpha_ci : float, optional
            Tail probability of the returned interval. Default ``0.05``.
        """
        distance = self.threshold - float(current_degradation)
        if distance <= 0:
            return ProcessRUL(0.0, (0.0, 0.0), 1.0, alpha_ci)
        med = self._quantile(0.5, distance)
        lo = self._quantile(alpha_ci / 2.0, distance)
        hi = self._quantile(1.0 - alpha_ci / 2.0, distance)
        return ProcessRUL(med, (lo, hi), 0.0, alpha_ci)

    def __repr__(self):
        return (
            "Wiener Process Degradation Model\n"
            "================================\n"
            "Drift (mu)          : {:.6g}\n"
            "Diffusion (sigma)   : {:.6g}\n"
            "Threshold           : {:.6g}\n"
            "Mean time to failure: {:.6g}".format(
                self.mu, self.sigma, self.threshold, self.mean()
            )
        )


class WienerProcess:
    """Fitter for the Wiener-process degradation model (see
    :class:`WienerProcessModel`)."""

    @classmethod
    def fit(cls, x, y, i, threshold):
        """
        Fit a Wiener-process degradation model by maximum likelihood.

        Parameters
        ----------
        x : array_like
            Measurement times.
        y : array_like
            Degradation measurements.
        i : array_like
            Unit identifier for each measurement.
        threshold : float
            The degradation level defining failure.

        Returns
        -------
        WienerProcessModel
        """
        dt, dy = _increments(x, y, i)
        # Increments dy | dt ~ Normal(mu*dt, sigma**2 * dt), independent.
        # Closed-form MLE:
        mu = dy.sum() / dt.sum()
        sigma2 = np.mean((dy - mu * dt) ** 2 / dt)
        sigma = np.sqrt(sigma2)
        if mu <= 0:
            raise ValueError(
                "fitted drift mu = {:.4g} is not positive, so the process "
                "does not reliably reach the threshold and the first-passage "
                "life distribution is defective. Check the sign of the "
                "degradation / threshold, or use a monotone model.".format(mu)
            )
        return WienerProcessModel(mu, sigma, threshold)


# --------------------------------------------------------------------------
# Gamma process
# --------------------------------------------------------------------------


class GammaProcessModel:
    """
    A fitted Gamma-process degradation model with stationary independent
    increments: over an interval ``dt`` the degradation increment is
    ``Gamma(shape = alpha * dt, rate = beta)``. The path is monotone
    increasing.

    Because the path is monotone, the first passage to the failure
    ``threshold`` has failure CDF ``P(W(t) >= threshold)``, evaluated with the
    regularised upper incomplete gamma function.

    Parameters
    ----------
    alpha : float
        Fitted shape rate (shape accrues as ``alpha * t``).
    beta : float
        Fitted rate parameter of the increments.
    threshold : float
        The degradation level defining failure.
    """

    def __init__(self, alpha, beta, threshold):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.threshold = float(threshold)
        self.params = np.array([self.alpha, self.beta])
        self.param_names = ["alpha", "beta"]

    def _ff_distance(self, t, distance):
        # P(T <= t) = P(W(t) >= distance) with W(t) ~ Gamma(alpha t, beta).
        t = np.asarray(t, dtype=float)
        out = np.zeros_like(t, dtype=float)
        pos = t > 0
        out[pos] = gammaincc(self.alpha * t[pos], self.beta * distance)
        return out

    def ff(self, t):
        """Failure (CDF) of the first-passage time to the threshold."""
        scalar = np.isscalar(t)
        res = self._ff_distance(np.atleast_1d(t), self.threshold)
        return float(res[0]) if scalar else res

    def sf(self, t):
        """Survival function of the first-passage time."""
        scalar = np.isscalar(t)
        res = 1.0 - self._ff_distance(np.atleast_1d(t), self.threshold)
        return float(res[0]) if scalar else res

    def df(self, t):
        """Density of the first-passage time (numeric derivative of ``ff``)."""
        scalar = np.isscalar(t)
        t = np.atleast_1d(np.asarray(t, dtype=float))
        h = 1e-6
        out = np.zeros_like(t)
        pos = t > 0
        tp = t[pos]
        step = np.maximum(h, tp * h)
        f_hi = self._ff_distance(tp + step, self.threshold)
        f_lo = self._ff_distance(np.maximum(tp - step, 1e-12), self.threshold)
        out[pos] = (f_hi - f_lo) / ((tp + step) - np.maximum(tp - step, 1e-12))
        out = np.clip(out, 0.0, None)
        return float(out[0]) if scalar else out

    def hf(self, t):
        """Hazard function of the first-passage time."""
        return self.df(t) / self.sf(t)

    def Hf(self, t):
        """Cumulative hazard of the first-passage time."""
        return -np.log(self.sf(t))

    def mean(self):
        """Mean time to failure, ``integral of the survival function``."""
        val, _ = quad(
            lambda t: self._sf_distance_scalar(t, self.threshold),
            0.0,
            np.inf,
            limit=200,
        )
        return val

    def _sf_distance_scalar(self, t, distance):
        if t <= 0:
            return 1.0
        return 1.0 - gammaincc(self.alpha * t, self.beta * distance)

    def qf(self, p):
        """Quantile (inverse CDF) of the first-passage time."""
        p = np.atleast_1d(np.asarray(p, dtype=float))
        out = np.array([self._quantile(pi, self.threshold) for pi in p])
        return float(out[0]) if out.shape == (1,) else out

    def _quantile(self, p, distance):
        if not (0.0 < p < 1.0):
            return 0.0 if p <= 0.0 else np.inf
        # rough starting scale from the mean increment rate
        rate = self.alpha / self.beta  # mean degradation per unit time
        hi = max(distance / rate, 1.0)
        while self._ff_distance(np.array([hi]), distance)[0] < p:
            hi *= 2.0
            if hi > 1e12:
                return np.inf
        return brentq(
            lambda t: self._ff_distance(np.array([t]), distance)[0] - p,
            1e-12,
            hi,
        )

    def random(self, size, random_state=None):
        """Draw first-passage (failure) times via inverse-CDF sampling."""
        rng = np.random.default_rng(random_state)
        u = rng.uniform(size=size)
        return np.array([self._quantile(ui, self.threshold) for ui in u])

    def predict_rul(self, current_degradation, alpha_ci=0.05):
        """
        Remaining useful life given the current degradation level. The
        residual distance ``threshold - current_degradation`` is crossed by a
        fresh gamma process, so its first-passage median and interval are
        returned.
        """
        distance = self.threshold - float(current_degradation)
        if distance <= 0:
            return ProcessRUL(0.0, (0.0, 0.0), 1.0, alpha_ci)
        med = self._quantile(0.5, distance)
        lo = self._quantile(alpha_ci / 2.0, distance)
        hi = self._quantile(1.0 - alpha_ci / 2.0, distance)
        return ProcessRUL(med, (lo, hi), 0.0, alpha_ci)

    def __repr__(self):
        return (
            "Gamma Process Degradation Model\n"
            "===============================\n"
            "Shape rate (alpha)  : {:.6g}\n"
            "Rate (beta)         : {:.6g}\n"
            "Threshold           : {:.6g}\n"
            "Mean time to failure: {:.6g}".format(
                self.alpha, self.beta, self.threshold, self.mean()
            )
        )


class GammaProcess:
    """Fitter for the Gamma-process degradation model (see
    :class:`GammaProcessModel`)."""

    @classmethod
    def fit(cls, x, y, i, threshold):
        """
        Fit a Gamma-process degradation model by maximum likelihood.

        The degradation must be monotone increasing (all increments
        non-negative); an increment that decreases raises an error pointing to
        the Wiener model for non-monotone signals.

        Parameters
        ----------
        x : array_like
            Measurement times.
        y : array_like
            Degradation measurements.
        i : array_like
            Unit identifier for each measurement.
        threshold : float
            The degradation level defining failure.

        Returns
        -------
        GammaProcessModel
        """
        dt, dy = _increments(x, y, i)
        if np.any(dy < 0):
            raise ValueError(
                "the degradation decreases over at least one interval, but a "
                "Gamma process is monotone increasing. Use WienerProcess for "
                "non-monotone / noisy signals."
            )
        # Guard against exact-zero increments (log 0) by nudging.
        dy = np.where(dy <= 0, 1e-12, dy)

        sum_dt = dt.sum()
        sum_dy = dy.sum()
        log_dy = np.log(dy)

        def neg_ll(alpha):
            # Profile out beta: d/dbeta -> beta = alpha * sum_dt / sum_dy.
            beta = alpha * sum_dt / sum_dy
            k = alpha * dt
            ll = np.sum(
                k * np.log(beta) + (k - 1.0) * log_dy - beta * dy - gammaln(k)
            )
            return -ll

        res = minimize_scalar(neg_ll, bounds=(1e-6, 1e6), method="bounded")
        alpha = float(res.x)
        beta = alpha * sum_dt / sum_dy
        return GammaProcessModel(alpha, beta, threshold)
