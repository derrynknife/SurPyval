import functools
from typing import TYPE_CHECKING, Callable, Iterable

import numpy as np
from numpy.typing import ArrayLike

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .nhpp_fitter import NHPPFitter

if TYPE_CHECKING:
    from surpyval.utils.recurrent_event_data import RecurrentEventData

    from .growth_projection import GrowthProjection


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
         alpha: 913.8467364753063
          beta: 1.4781708312933042
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([4.20071634e-05, 1.17029976e-04, 2.13103252e-04, 3.26039992e-04,
           4.53440627e-04, 5.93695609e-04])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([6.20937637e-05, 8.64951483e-05, 1.05001004e-04, 1.20485702e-04,
           1.34052542e-04, 1.46263922e-04])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([ 913.84673648, 1460.57447773, 1921.54925376, 2334.39341615,
           2714.78109598, 3071.15590144])
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

    def _closed_form_mle(
        self, data: "RecurrentEventData"
    ) -> "np.ndarray | None":
        """The closed-form MLE (MIL-HDBK-189C, Crow 1974) where every
        item is observed from 0 to a common end ``T``, closed by a ``c=1``
        row, by its ``tr`` or by its last failure (time- or
        failure-terminated), with exact failures only:

        .. math::
            \\hat\\beta = \\frac{N}{\\sum_{q, i} \\ln(T / t_{qi})},
            \\qquad
            \\hat\\alpha = T \\left(\\frac{k}{N}\\right)^{1 /
            \\hat\\beta}

        for ``N`` failures over ``k`` items. The search agreed with it to
        only about 1e-5 (#665), where a handbook check reads four or five
        figures. ``None`` for any other data (delayed entry, censored
        counts, unequal ends), which are searched."""
        x = np.asarray(data.x, dtype=float)
        c = np.asarray(data.c)
        if not np.all((c == 0) | (c == 1)):
            return None
        if x.ndim == 2:
            # Exact rows given as [t, t] pairs, the same data as 1-D
            if not np.all(x[:, 0] == x[:, 1]):
                return None
            x = x[:, 1]
        tl = np.asarray(data.tl, dtype=float)
        if np.any(np.isfinite(tl) & (tl != 0)):
            return None
        _, ends = data.item_observation_windows()
        T = float(ends[0])
        if not (np.isfinite(T) and T > 0 and np.all(ends == T)):
            return None
        failures = c == 0
        times = x[failures]
        counts = np.asarray(data.n, dtype=float)[failures]
        N = float(counts.sum())
        if N < 1 or np.any(times <= 0):
            return None
        log_sum = float(np.sum(counts * np.log(T / times)))
        if not log_sum > 0:
            return None
        beta = N / log_sum
        alpha = T * (len(ends) / N) ** (1.0 / beta)
        return np.array([alpha, beta])

    def projection(
        self,
        x: ArrayLike,
        modes: ArrayLike,
        fef: dict,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        bc: "Iterable | None" = None,
    ) -> "GrowthProjection":
        """
        Project the MTBF of a reliability growth test once the fixes
        delayed to its end are in: the AMSAA-Crow projection model
        (MIL-HDBK-189C, section 6.2; Crow 1983) and, with fixes made during
        the test, Crow's (2004) extended model.

        Each failure carries the label of its failure mode (``modes``),
        and each mode is one of three kinds:

        - a **BD mode**, whose fix is delayed to the end of the test: a key
          of ``fef``, whose value is the mode's fix-effectiveness factor,
          the fraction of its intensity the fix removes (in [0, 1]);
        - a **BC mode**, fixed during the test: listed in ``bc``;
        - an **A mode**, not to be fixed: every other mode.

        The projected intensity of one system after the delayed fixes is

        .. math::
            r_P = \\lambda_{CA} - \\frac{N_{BD}}{kT}
                  + \\sum_{i=1}^{K} (1 - d_i) \\frac{N_i}{kT}
                  + \\bar d\\, h(T),

        for ``k`` systems each tested to ``T``, with ``N_i`` failures of BD
        mode ``i`` (``K`` BD modes seen, ``N_BD`` failures in all), FEFs
        ``d_i`` with mean :math:`\\bar d`, and :math:`\\lambda_{CA}` the
        intensity the test demonstrates: ``N / (kT)`` when there are no BC
        modes (the system did not change during the test), the Crow-AMSAA
        intensity at ``T`` fitted to every failure when there are, with the
        bias-corrected shape: :math:`\\bar\\beta_{CA} N / (kT)`,
        :math:`\\bar\\beta_{CA} = (N - 1) / N \\cdot \\hat\\beta_{CA}`.
        :math:`h(T) = K \\bar\\beta / (kT)` is the rate at which new BD
        modes were still being found, from the power-law fit to the BD
        modes' first occurrences ``t_i`` with the unbiased shape
        :math:`\\bar\\beta = (K - 1) / \\sum_i \\ln(T / t_i)`; it allows for
        the modes not seen yet, which the fixes do not reach. The growth
        potential is the same without that term: every BD mode found and
        fixed with these factors.

        Parameters
        ----------
        x : array_like
            The failure times and the end of each system's test, as
            :meth:`fit` takes them.
        modes : array_like
            The failure mode of each row of ``x``: any hashable label (a
            string, a number). Every failure needs one; the end-of-test
            (``c=1``) rows' labels are ignored (``None`` will do).
        fef : dict
            The BD modes, each mapped to its fix-effectiveness factor.
        i : array_like, optional
            The system each row belongs to; one system by default.
        c : array_like, optional
            0 a failure, 1 the end of a system's test. Every system must
            be tested from 0 to the same time ``T``, given as its ``c=1``
            row (a time-terminated test).
        bc : iterable, optional
            The BC modes, fixed during the test.

        Returns
        -------
        GrowthProjection
            The demonstrated, projected and growth-potential intensities
            and MTBFs (of one system), the BD modes' table, and the
            Crow-AMSAA fit (``model``).

        Raises
        ------
        ValueError
            If a failure has no mode label, a mode is in both ``fef`` and
            ``bc``, a classified mode has no failures, a factor is outside
            [0, 1], the test is not time-terminated, or there are BC
            modes and fewer than 2 failures.

        Notes
        -----
        Several systems are taken to be tested side by side, so system
        time ``t`` is ``k t`` of total test time; a mode's first
        occurrence is its earliest over the systems.

        References
        ----------
        Crow, L. H. (1983), "Reliability growth projection from delayed
        fixes", Proceedings of the Annual Reliability and Maintainability
        Symposium, 84-89.

        Crow, L. H. (2004), "An extended reliability growth model for
        managing and assessing corrective actions", Proceedings of the
        Annual Reliability and Maintainability Symposium, 73-80.

        MIL-HDBK-189C (2011), "Reliability Growth Management", section 6.

        MIL-HDBK-00189A (2009), "Reliability Growth Management", section
        7.5 (the Crow extended model and its test-fix-find-test example).

        ReliaSoft, "Crow Extended", Reliability Growth and Repairable
        System Analysis Reference (ReliaWiki, RGA chapter 9): the same
        projection, :math:`\\hat\\lambda_P = \\hat\\lambda_{CA} -
        \\hat\\lambda_{BD} + \\sum (1 - d_i) N_i / T + \\bar d\\,
        \\hat h(T \\mid BD)`, and growth potential without the last term;
        with BC modes the demonstrated intensity is "the instantaneous
        failure intensity based on all of the data" (the Crow-AMSAA model
        fitted to the A, BC and BD failures), without them ``N / T``
        (#710). The handbook's test-fix-find-test example (MIL-HDBK-00189A
        section 7.5; ReliaWiki's Crow Extended examples), 56 failures to
        T = 400 with 14 BC and 16 BD modes, is reproduced: shape 0.9103
        (bias-corrected; the MLE is 0.9268), demonstrated MTBF 7.84708,
        BD modes' shape 0.7472, projected MTBF 11.29418 (#730).

        Examples
        --------
        One prototype tested to 400 hours. Modes ``a1`` and ``a2`` will not
        be fixed; ``b1`` to ``b4`` will be, after the test, with the
        effectiveness factors given:

        >>> from surpyval.recurrent import CrowAMSAA
        >>> x = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
        >>> modes = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1",
        ...          "a1", "b3", None]
        >>> c = [0] * 11 + [1]
        >>> fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6}
        >>> result = CrowAMSAA.projection(x, modes, fef, c=c)
        >>> result
        Reliability growth projection (AMSAA-Crow)
        ==========================================
        Test                : 1 system to T = 400
        Failures            : 3 A, 0 BC, 8 BD (4 BD modes)
        Mean FEF            : 0.7125
        New BD modes        : beta = 0.4454, h(T) = 0.004454
                                 intensity         MTBF
        Demonstrated        :       0.0275        36.36
        Projected           :      0.01592         62.8
        Growth potential    :      0.01275        78.43

        The 11 failures in 400 hours demonstrate an MTBF of 36.4 hours;
        the delayed fixes are projected to raise it to 62.8, short of the
        78.4 they would reach if every BD mode had been found.
        """
        from .growth_projection import growth_projection

        return growth_projection(self, x, modes, fef, i=i, c=c, bc=bc)


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
