import functools
from typing import Callable

import numpy as np

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .nhpp_fitter import NHPPFitter


@singleton_fitter
class CrowAMSAA(NHPPFitter):
    """
    The Crow-AMSAA (power-law) non-homogeneous Poisson process, with
    cumulative intensity and intensity

    .. math::
        \\Lambda(t) = \\left(\\frac{t}{\\alpha}\\right)^{\\beta}, \\qquad
        \\lambda(t) = \\frac{\\beta}{\\alpha^{\\beta}} t^{\\beta - 1}.

    ``beta`` above 1 is a rising event rate (deterioration), below 1 a
    falling one (reliability growth) and 1 the HPP with rate
    ``1 / alpha``; ``alpha`` is the time by which one event is expected.
    ``CrowAMSAA`` is an instance of this class; ``fit`` and ``from_params``
    return a ``ParametricRecurrenceModel``.

    Examples
    --------

    >>> from surpyval import Exponential
    >>> from surpyval.recurrent import CrowAMSAA
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1e-3).cumsum()
    >>> model = CrowAMSAA.fit(x)
    >>> print(model)
    Parametric Recurrence SurPyval Model
    ==================================
    Process             : Crow-AMSAA
    Fitted by           : MLE
    Parameters          :
         alpha: 913.8466210685444
          beta: 1.4781707110680866
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([4.20072057e-05, 1.17030084e-04, 2.13103439e-04, 3.26040266e-04,
           4.53440995e-04, 5.93696079e-04])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([6.20938211e-05, 8.64952211e-05, 1.05001087e-04, 1.20485793e-04,
           1.34052640e-04, 1.46264026e-04])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([ 913.84662107, 1460.57434899, 1921.54912724, 2334.39329941,
           2714.78099355, 3071.15581638])
    """

    def __init__(self) -> None:
        self.name = "Crow-AMSAA"
        self.parameter_names = ["alpha", "beta"]
        self.has_scale = True
        # beta > 0: the intensity beta / alpha**beta * x**(beta - 1) and
        # log(beta) are undefined below zero (a negative beta was allowed and
        # the optimiser could stop there, with a decreasing MCF).
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return (x / alpha) ** beta

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return (beta / alpha**beta) * (x ** (beta - 1))

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return np.log(beta) - beta * np.log(alpha) + (beta - 1) * np.log(x)

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return alpha * (N ** (1.0 / beta))


# -- Crow's (1982) exact bounds on the demonstrated MTBF (#578) ------------
#
# For a power-law process the instantaneous MTBF at the end of the test,
# M(T) = 1 / iif(T), has the MLE M_hat = T / (N beta_hat) (k systems all
# observed on (0, T]: k T / (N beta_hat)). Crow (1982) gives exact bounds
# M_hat * L <= M(T) <= M_hat * U, with coefficients that depend only on N
# and the level, tabulated in MIL-HDBK-189C. They are computed here.
#
# Failure terminated (one system, observed to its N-th failure t_N):
# M_hat / M = Lambda(t_N) * beta / (N beta_hat) = Z G / N^2, where
# Z = Lambda(t_N) ~ Gamma(N) and, given t_N, G = beta * sum_{i<N}
# log(t_N / t_i) ~ Gamma(N - 1) independently. So L = 1 / r_{1 - a} and
# U = 1 / r_a, with r_p the p-quantile of Z G / N^2.
#
# Time terminated at T: N is random, so there is no pivot. With
# S = sum log(T / t_i) and psi = beta * E[N] = T / M (k T / M for k
# systems), P(N = n | S) is proportional to z^n / (n! (n - 1)!), z =
# psi S, free of beta; normalised by sqrt(z) I_1(2 sqrt(z)). As
# z = n^2 M_hat / M, inverting this conditional test of psi gives
# L = n^2 / z_U with P(N <= n | z_U) = a, and U = n^2 / z_L with
# P(N >= n | z_L) = a (U is infinite for n = 1).


def _log_conditional_terms(n: int, z: float) -> np.ndarray:
    """The logs of P(N = k | z), k = 1..n, for the time-terminated test."""
    from scipy.special import gammaln, ive

    k = np.arange(1, n + 1)
    root = 2.0 * np.sqrt(z)
    log_norm = 0.5 * np.log(z) + np.log(ive(1, root)) + root
    return k * np.log(z) - gammaln(k + 1) - gammaln(k) - log_norm


def _conditional_cdf(n: int, log_z: float) -> float:
    """P(N <= n | z) for the time-terminated test (z = exp(log_z))."""
    if n < 1:
        return 0.0
    from scipy.special import logsumexp

    terms = _log_conditional_terms(n, float(np.exp(log_z)))
    return float(min(1.0, np.exp(logsumexp(terms))))


def _solve_log(
    f: "Callable[[float], float]", target: float, centre: float
) -> float:
    """The root in ``u`` of ``f(u) = target`` for ``f`` decreasing in
    ``u``, bracketed outward from ``centre``."""
    from scipy.optimize import brentq

    lo, hi = centre - 2.0, centre + 2.0
    while f(lo) < target:
        lo -= 2.0
    while f(hi) > target:
        hi += 2.0
    return brentq(lambda u: f(u) - target, lo, hi, xtol=1e-14, rtol=1e-14)


@functools.lru_cache(maxsize=1024)
def crow_time_terminated_coefficients(
    n: int, alpha: float
) -> "tuple[float, float]":
    """Crow's (1982) coefficients ``(L, U)`` for the demonstrated MTBF of
    a time-terminated test with ``n`` failures: ``M_hat * L`` is a lower
    and ``M_hat * U`` an upper ``1 - alpha`` confidence bound."""
    centre = 2.0 * np.log(n)  # the conditional mode of N is about sqrt(z)
    log_z_upper = _solve_log(lambda u: _conditional_cdf(n, u), alpha, centre)
    lower = n**2 / np.exp(log_z_upper)
    if n == 1:
        return float(lower), np.inf
    log_z_lower = _solve_log(
        lambda u: _conditional_cdf(n - 1, u), 1.0 - alpha, centre
    )
    return float(lower), float(n**2 / np.exp(log_z_lower))


def _product_cdf(n: int, r: float) -> float:
    """P(Z G / n^2 <= r), Z ~ Gamma(n), G ~ Gamma(n - 1) independent."""
    from scipy.integrate import quad
    from scipy.special import gammainc
    from scipy.stats import gamma

    a, b = gamma.ppf([1e-15, 1.0 - 1e-15], n - 1)
    value, _ = quad(
        lambda g: gammainc(n, r * n**2 / g) * gamma.pdf(g, n - 1),
        a,
        b,
        epsabs=1e-13,
        epsrel=1e-12,
        limit=200,
    )
    return float(min(1.0, max(0.0, value)))


@functools.lru_cache(maxsize=1024)
def crow_failure_terminated_coefficients(
    n: int, alpha: float
) -> "tuple[float, float]":
    """Crow's (1982) coefficients ``(L, U)`` for the demonstrated MTBF of
    a failure-terminated test (one system, observed to its ``n``-th
    failure, ``n >= 2``)."""

    # P(r <= R) falls in log r; the coefficients are the reciprocals of
    # the ratio's quantiles.
    def sf(u: float) -> float:
        return 1.0 - _product_cdf(n, float(np.exp(u)))

    log_r_hi = _solve_log(sf, alpha, 0.0)
    log_r_lo = _solve_log(sf, 1.0 - alpha, 0.0)
    return float(np.exp(-log_r_hi)), float(np.exp(-log_r_lo))
