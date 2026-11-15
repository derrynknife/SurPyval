r"""Pooling event counts from several sources: :func:`pooled_rate`, the
two-stage (hierarchical) estimate of how an event rate, or an event
probability per trial, varies between sources, with the average across
them, the range for a new source like them and each source's own
estimate pulled towards the others.

Each source gives a count of events and its exposure: the events in a
total time at risk (a rate, ``kind="rate"``: gamma-Poisson), or the
events in a number of trials or demands (a probability, ``kind=
"probability"``: beta-binomial). The sources are treated as a sample from
a population of sources, each with its own true rate, rather than as one
rate measured several times.

It imports no plotting library, and scipy only when called.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from surpyval.utils.validation import alpha_ci_error

if TYPE_CHECKING:
    import pandas as pd

__all__ = ["pooled_rate", "PooledRate"]

#: The fewest sources a population estimate is given for, by default.
MIN_SOURCES = 4

# The search range of the population's shape (the gamma's shape, the
# beta's first parameter): an MLE at the top is "no spread between the
# sources" (an infinite shape).
_LOG_SHAPE_MIN = np.log(1e-4)
_LOG_SHAPE_MAX = np.log(1e8)
# The Bayes grid of the shape, and its size (points per axis).
_GRID_SHAPE = (np.log(1e-3), np.log(1e6))
_GRID_N = 161


# ---------------------------------------------------------------------------
# The two families. ``loc`` is the population mean on its working scale
# (the log of the mean rate; the logit of the mean probability) and ``u``
# the log of the shape (gamma shape alpha; beta parameter a).
# ---------------------------------------------------------------------------


class _Poisson:
    kind = "rate"
    population_name = "Gamma"
    marginal = "negative binomial (gamma-Poisson)"

    @staticmethod
    def to_loc(value: Any) -> Any:
        return np.log(value)

    @staticmethod
    def from_loc(loc: Any) -> Any:
        return np.exp(loc)

    @staticmethod
    def params(loc: Any, u: Any) -> tuple[Any, Any]:
        """(shape, rate) of the gamma: mean ``exp(loc)``, shape
        ``exp(u)``."""
        a = np.exp(u)
        return a, a / np.exp(loc)

    @staticmethod
    def loglik(x: Any, T: Any, loc: Any, u: Any) -> Any:
        """The log-likelihood of each source, the rate integrated out:
        negative binomial with mean ``T m`` and shape ``a``, written so
        that a large shape is accurate."""
        from scipy.special import betaln, gammaln

        a = np.exp(u)
        Tm = T * np.exp(loc)
        lpoch = np.where(
            x > 0,
            gammaln(np.where(x > 0, x, 1.0))
            - betaln(a, np.where(x > 0, x, 1.0)),
            0.0,
        )
        return (
            lpoch
            - gammaln(x + 1)
            - a * np.log1p(Tm / a)
            + x * (np.log(Tm) - np.log(a + Tm))
        )

    @staticmethod
    def own(x: Any, T: Any) -> tuple[Any, Any]:
        """The Jeffreys posterior of a source on its own: Gamma(x + 1/2,
        T), as (shape, rate)."""
        return x + 0.5, T

    @staticmethod
    def conditional(a: Any, b: Any, x: Any, T: Any) -> tuple[Any, Any]:
        """A source's posterior given the population: Gamma(a + x,
        b + T)."""
        return a + x, b + T

    @staticmethod
    def cdf(a: Any, b: Any, value: Any) -> Any:
        """The cdf of Gamma(a, rate b) at ``value``; a point mass at
        ``a / b`` where ``a`` is infinite (``b`` holds the mean there)."""
        from scipy.special import gammainc

        inf = ~np.isfinite(a)
        safe_a = np.where(inf, 1.0, a)
        safe_b = np.where(inf, 1.0, b)
        return np.where(
            inf, (value >= b).astype(float), gammainc(safe_a, safe_b * value)
        )

    @staticmethod
    def ppf(a: Any, b: Any, p: Any) -> Any:
        from scipy.special import gammaincinv

        inf = ~np.isfinite(a)
        safe_a = np.where(inf, 1.0, a)
        safe_b = np.where(inf, 1.0, b)
        return np.where(inf, b, gammaincinv(safe_a, p) / safe_b)

    @staticmethod
    def mean(a: Any, b: Any) -> Any:
        return np.where(
            np.isfinite(a), a / np.where(np.isfinite(a), b, 1.0), b
        )

    @staticmethod
    def var(a: Any, b: Any) -> Any:
        return np.where(
            np.isfinite(a), a / np.where(np.isfinite(a), b, 1.0) ** 2, 0.0
        )

    @staticmethod
    def simulate(rng: np.random.Generator, a: float, b: float, T: Any) -> Any:
        if np.isfinite(a):
            lam = rng.gamma(a, 1.0 / b, size=np.shape(T))
        else:
            lam = np.full(np.shape(T), b)
        return rng.poisson(lam * T).astype(float)

    @staticmethod
    def chi2(x: Any, T: Any) -> tuple[float, int]:
        """Pearson's chi-square test that every source has one rate:
        ``sum (x_i - T_i r)^2 / (T_i r)`` with ``r = sum x / sum T``, on
        ``k - 1`` degrees of freedom."""
        e = T * x.sum() / T.sum()
        return float(np.sum((x - e) ** 2 / e)), x.size - 1


class _Binomial:
    kind = "probability"
    population_name = "Beta"
    marginal = "beta-binomial"

    @staticmethod
    def to_loc(value: Any) -> Any:
        return np.log(value) - np.log1p(-value)

    @staticmethod
    def from_loc(loc: Any) -> Any:
        from scipy.special import expit

        return expit(loc)

    @staticmethod
    def params(loc: Any, u: Any) -> tuple[Any, Any]:
        """(a, b) of the beta: mean ``expit(loc)``, ``a = exp(u)``."""
        a = np.exp(u)
        return a, a * np.exp(-loc)

    @staticmethod
    def loglik(x: Any, n: Any, loc: Any, u: Any) -> Any:
        from scipy.special import betaln, gammaln

        a = np.exp(u)
        b = a * np.exp(-loc)
        return (
            betaln(x + a, n - x + b)
            - betaln(a, b)
            + gammaln(n + 1)
            - gammaln(x + 1)
            - gammaln(n - x + 1)
        )

    @staticmethod
    def own(x: Any, n: Any) -> tuple[Any, Any]:
        """Jeffreys: Beta(x + 1/2, n - x + 1/2)."""
        return x + 0.5, n - x + 0.5

    @staticmethod
    def conditional(a: Any, b: Any, x: Any, n: Any) -> tuple[Any, Any]:
        return a + x, b + n - x

    @staticmethod
    def cdf(a: Any, b: Any, value: Any) -> Any:
        """The cdf of Beta(a, b); a point mass where ``a`` is infinite
        (``b`` holds the mean there)."""
        from scipy.special import betainc

        inf = ~np.isfinite(a)
        safe_a = np.where(inf, 1.0, a)
        safe_b = np.where(inf, 1.0, b)
        return np.where(
            inf, (value >= b).astype(float), betainc(safe_a, safe_b, value)
        )

    @staticmethod
    def ppf(a: Any, b: Any, p: Any) -> Any:
        from scipy.special import betaincinv

        inf = ~np.isfinite(a)
        safe_a = np.where(inf, 1.0, a)
        safe_b = np.where(inf, 1.0, b)
        return np.where(inf, b, betaincinv(safe_a, safe_b, p))

    @staticmethod
    def mean(a: Any, b: Any) -> Any:
        fin = np.isfinite(a)
        return np.where(fin, a / np.where(fin, a + b, 1.0), b)

    @staticmethod
    def var(a: Any, b: Any) -> Any:
        fin = np.isfinite(a)
        s = np.where(fin, a + b, 1.0)
        return np.where(fin, a * b / (s**2 * (s + 1)), 0.0)

    @staticmethod
    def simulate(rng: np.random.Generator, a: float, b: float, n: Any) -> Any:
        if np.isfinite(a):
            p = rng.beta(a, b, size=np.shape(n))
        else:
            p = np.full(np.shape(n), b)
        return rng.binomial(n.astype(np.int64), p).astype(float)

    @staticmethod
    def chi2(x: Any, n: Any) -> tuple[float, int]:
        """Pearson's chi-square test of the 2 x k table (events and
        non-events by source) that every source has one probability."""
        p = x.sum() / n.sum()
        e1, e0 = n * p, n * (1 - p)
        stat = np.sum((x - e1) ** 2 / e1) + np.sum(((n - x) - e0) ** 2 / e0)
        return float(stat), x.size - 1


_FAMILIES: dict[str, Any] = {"rate": _Poisson, "probability": _Binomial}


# ---------------------------------------------------------------------------
# Mixtures of gammas or betas (a posterior or bootstrap set of populations)
# ---------------------------------------------------------------------------


def _mixture_summary(
    fam: Any, a: Any, b: Any, w: Any, probs: tuple[float, ...]
) -> tuple[float, list[float]]:
    """The mean and the quantiles ``probs`` of the mixture ``sum_k w_k
    F(a_k, b_k)``. A mixture quantile lies between its components'
    quantiles at the same probability, which brackets the search."""
    from scipy.optimize import brentq

    a = np.asarray(a, float)
    b = np.asarray(b, float)
    w = np.asarray(w, float) / np.sum(w)
    mean = float(np.sum(w * fam.mean(a, b)))
    out = []
    for p in probs:
        q = fam.ppf(a, b, p)
        lo, hi = float(np.min(q)), float(np.max(q))
        if not lo < hi:
            out.append(lo)
            continue
        if fam is _Poisson:
            if lo <= 0:
                lo = (
                    min(hi * 1e-300, float(np.min(q[q > 0])))
                    if np.any(q > 0)
                    else 0.0
                )
            if lo <= 0:
                out.append(0.0)
                continue
            to, back = np.log, np.exp
        else:
            lo = max(lo, 1e-300)
            hi = min(hi, 1 - 1e-16)
            to, back = fam.to_loc, fam.from_loc

        def f(t: float, p: float = p) -> float:
            return float(np.sum(w * fam.cdf(a, b, back(t)))) - p

        t_lo, t_hi = float(to(lo)), float(to(hi))
        f_lo, f_hi = f(t_lo), f(t_hi)
        if f_lo >= 0:
            out.append(float(back(t_lo)))
        elif f_hi <= 0:
            out.append(float(back(t_hi)))
        else:
            out.append(
                float(back(brentq(f, t_lo, t_hi, xtol=1e-10, rtol=1e-12)))
            )
    return mean, out


def _matched(
    fam: Any, median: float, upper: float, p_upper: float
) -> tuple[float, float]:
    """The gamma or beta with this median and this upper quantile (the
    stored summary of a mixture: its upper side, which matters most, is
    kept)."""
    from scipy.optimize import brentq

    target = np.log(upper / median)

    def gap(u: float) -> float:
        a = np.exp(u)
        if fam is _Poisson:
            return float(
                np.log(fam.ppf(a, 1.0, p_upper) / fam.ppf(a, 1.0, 0.5))
                - target
            )
        # beta: for a given a, find b with the median, then compare
        b = _beta_b_for_median(a, median)
        return float(np.log(fam.ppf(a, b, p_upper) / median) - target)

    lo, hi = np.log(1e-3), np.log(1e7)
    g_lo, g_hi = gap(lo), gap(hi)
    if g_lo * g_hi > 0:
        u = lo if abs(g_lo) < abs(g_hi) else hi
    else:
        u = brentq(gap, lo, hi, xtol=1e-10)
    a = float(np.exp(u))
    if fam is _Poisson:
        return a, float(fam.ppf(a, 1.0, 0.5) / median)
    return a, float(_beta_b_for_median(a, median))


def _beta_b_for_median(a: float, median: float) -> float:
    from scipy.optimize import brentq
    from scipy.special import betaincinv

    def g(v: float) -> float:
        return float(np.log(betaincinv(a, np.exp(v), 0.5)) - np.log(median))

    return float(np.exp(brentq(g, np.log(1e-8), np.log(1e14), xtol=1e-12)))


# ---------------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------------


def _total_loglik(fam: Any, x: Any, T: Any, loc: Any, u: Any) -> Any:
    """The log-likelihood summed over the sources; ``loc`` and ``u``
    broadcast (a grid), the sources run along a new last axis."""
    loc = np.asarray(loc, float)[..., None]
    u = np.asarray(u, float)[..., None]
    return np.sum(fam.loglik(x, T, loc, u), axis=-1)


def _loc_range(fam: Any, x: Any, T: Any) -> tuple[float, float]:
    """The working-scale range the sources' own estimates span, widened
    by 6 either way (beyond which no population mean has weight)."""
    if fam is _Poisson:
        own = (x + 0.5) / T
    else:
        own = (x + 0.5) / (T + 1.0)
    lo, hi = float(np.min(fam.to_loc(own))), float(np.max(fam.to_loc(own)))
    lo, hi = lo - 6.0, hi + 6.0
    if fam is _Binomial:
        lo, hi = max(lo, -40.0), min(hi, 40.0)
    return lo, hi


def _mle(fam: Any, x: Any, T: Any) -> tuple[float, float, float]:
    """The marginal maximum likelihood (loc, u, loglik). A shape at the
    top of its range is returned as ``u = inf``: the likelihood rises to
    the end, where every source has one rate (no spread)."""
    from scipy.optimize import minimize

    lo, hi = _loc_range(fam, x, T)
    locs = np.linspace(lo, hi, 61)
    us = np.linspace(_LOG_SHAPE_MIN, _LOG_SHAPE_MAX, 41)
    L, U = np.meshgrid(locs, us, indexing="ij")
    ll = _total_loglik(fam, x, T, L, U)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)
    res = minimize(
        lambda p: -float(_total_loglik(fam, x, T, p[0], p[1])),
        x0=[L[i, j], U[i, j]],
        method="L-BFGS-B",
        bounds=[(lo, hi), (_LOG_SHAPE_MIN, _LOG_SHAPE_MAX)],
        options={"ftol": 1e-15, "gtol": 1e-10},
    )
    loc, u = float(res.x[0]), float(res.x[1])
    best = -float(res.fun)
    # the homogeneous end: one rate for all, its own closed-form MLE
    loc_h = float(fam.to_loc(x.sum() / T.sum()))
    ll_h = float(_total_loglik(fam, x, T, loc_h, _LOG_SHAPE_MAX))
    if u > _LOG_SHAPE_MAX - 1e-3 or ll_h >= best - 1e-9:
        return loc_h, np.inf, ll_h
    return loc, u, best


def _profile(fam: Any, x: Any, T: Any, loc: float) -> float:
    """The profile log-likelihood of the population mean (its working
    scale ``loc``), the shape maximised out (to its homogeneous end)."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        lambda u: -float(_total_loglik(fam, x, T, loc, u)),
        bounds=(_LOG_SHAPE_MIN, _LOG_SHAPE_MAX),
        method="bounded",
        options={"xatol": 1e-6},
    )
    end = float(_total_loglik(fam, x, T, loc, _LOG_SHAPE_MAX))
    return max(-float(res.fun), end)


def _profile_interval(
    fam: Any, x: Any, T: Any, loc_hat: float, ll_hat: float, alpha_ci: float
) -> tuple[float, float]:
    """The likelihood-ratio (profile) interval of the population mean."""
    from scipy.optimize import brentq
    from scipy.stats import chi2

    crit = float(chi2.ppf(1 - alpha_ci, 1)) / 2.0
    lo, hi = _loc_range(fam, x, T)

    def g(loc: float) -> float:
        return ll_hat - _profile(fam, x, T, loc) - crit

    ends = []
    for edge in (lo - 20.0, hi + 20.0):
        step = 0.25 if edge > loc_hat else -0.25
        a, b = loc_hat, loc_hat + step
        while g(b) < 0 and abs(b - loc_hat) < abs(edge - loc_hat):
            a, b = b, loc_hat + 1.6 * (b - loc_hat)
        if g(b) < 0:
            ends.append(-np.inf if step < 0 else np.inf)
        else:
            ends.append(brentq(g, min(a, b), max(a, b), xtol=1e-10))
    lower = 0.0 if ends[0] == -np.inf else float(fam.from_loc(ends[0]))
    if ends[1] == np.inf:
        upper = np.inf if fam is _Poisson else 1.0
    else:
        upper = float(fam.from_loc(ends[1]))
    return lower, upper


def _bayes_grid(
    fam: Any, x: Any, T: Any, log_prior: Any = None
) -> tuple[Any, Any, Any]:
    """The posterior of (loc, u) on a grid: the cells' loc and u and
    their normalised weights. Two passes: one over the whole range, then
    a finer one over the cells that carry weight."""
    if log_prior is None:
        log_prior = _default_log_prior
    lo, hi = _loc_range(fam, x, T)
    ulo, uhi = _GRID_SHAPE
    for _ in range(2):
        locs = np.linspace(lo, hi, _GRID_N)
        us = np.linspace(ulo, uhi, _GRID_N)
        L, U = np.meshgrid(locs, us, indexing="ij")
        lp = _total_loglik(fam, x, T, L, U) + log_prior(fam, L, U)
        lp = np.where(np.isfinite(lp), lp, -np.inf)
        w = np.exp(lp - lp.max())
        rows = np.flatnonzero(w.max(axis=1) > 1e-14)
        cols = np.flatnonzero(w.max(axis=0) > 1e-14)
        dl, du = locs[1] - locs[0], us[1] - us[0]
        lo, hi = (
            locs[max(rows[0] - 2, 0)] - dl,
            locs[min(rows[-1] + 2, _GRID_N - 1)] + dl,
        )
        ulo = max(us[max(cols[0] - 2, 0)] - du, _GRID_SHAPE[0])
        uhi = min(us[min(cols[-1] + 2, _GRID_N - 1)] + du, _GRID_SHAPE[1])
    w = (w / w.sum()).ravel()
    # the cells holding all but 1e-7 of the posterior
    order = np.argsort(w)[::-1]
    n_keep = int(np.searchsorted(np.cumsum(w[order]), 1.0 - 1e-7)) + 1
    keep = order[:n_keep]
    return L.ravel()[keep], U.ravel()[keep], w[keep] / w[keep].sum()


def _default_log_prior(fam: Any, loc: Any, u: Any) -> Any:
    """Flat on the population mean's working scale (log rate, logit
    probability); ``p(a) = 1 / (1 + a)^2`` on the shape, written on
    ``u = log a`` (``a / (1 + a)^2``)."""
    return u - 2.0 * np.logaddexp(0.0, u)


def _bootstrap_fits(
    fam: Any, x: Any, T: Any, loc: float, u: float, n_boot: int, rng: Any
) -> tuple[Any, Any]:
    """Parametric bootstrap of the population: ``n_boot`` sets of counts
    from the fitted population, each refitted. Returns the replicates'
    (loc, u); a set with no events (or, for a probability, all events)
    has no maximum and is drawn again."""
    a, b = (
        fam.params(loc, u) if np.isfinite(u) else (np.inf, fam.from_loc(loc))
    )
    locs: list[float] = []
    us: list[float] = []
    tries = 0
    while len(locs) < n_boot and tries < 20 * n_boot:
        tries += 1
        xs = fam.simulate(rng, a, b, T)
        if xs.sum() == 0 or (fam is _Binomial and np.all(xs == T)):
            continue
        lb, ub, _ = _mle(fam, xs, T)
        locs.append(lb)
        us.append(ub)
    return np.array(locs), np.array(us)


# ---------------------------------------------------------------------------
# The public function and result
# ---------------------------------------------------------------------------


def _as_counts(name: str, value: Any) -> npt.NDArray:
    arr = np.atleast_1d(np.asarray(value, dtype=float))
    if arr.ndim != 1:
        raise ValueError(
            "'{}' must be one number per source (a 1-d list); got shape "
            "{}.".format(name, arr.shape)
        )
    if not np.all(np.isfinite(arr)):
        raise ValueError(
            "'{}' must be finite numbers, one per source; got {}.".format(
                name, arr.tolist()
            )
        )
    return arr


def pooled_rate(
    events: npt.ArrayLike,
    exposure: npt.ArrayLike,
    kind: Literal["rate", "probability"] = "rate",
    *,
    labels: Any = None,
    method: Literal["Bayes", "MLE"] = "Bayes",
    alpha_ci: float = 0.05,
    min_sources: int = MIN_SOURCES,
    n_boot: int = 200,
    random_state: Any = None,
) -> PooledRate:
    r"""
    Pool event counts from several sources that may each have their own
    true rate: the two-stage (hierarchical) estimate of how the rate
    varies between sources.

    Each source (a site, a hospital, a study, a fleet, a batch) gives the
    number of events it saw and its exposure: the total time at risk
    (``kind="rate"``: events per unit of time, distance or person-years),
    or the number of trials (``kind="probability"``: events per trial,
    such as failures to start per demand or adverse events per dose).
    Adding everything into one pot assumes every source has the same
    rate, and reports a precision nobody has when they differ. This
    treats the sources as a sample from a population of sources instead,
    estimates how their true rates are spread, and gives:

    - the **population** (between-source) distribution: a gamma for rates,
      a beta for probabilities;
    - the **average across sources** (the population mean) with its
      interval: how well the middle of the spread is known;
    - the range for **a new source like these** (the predictive
      interval): where the rate of one more source, about which nothing
      else is known, probably lies. It is wider, and is the one to use as
      a starting belief for a new source;
    - each **source's own** estimate and interval, alone and *shrunk*
      (pulled towards the others: a source with little exposure is
      pulled a long way, one with a lot hardly at all);
    - a **heterogeneity** test and measures: do the sources differ by
      more than chance?

    Parameters
    ----------
    events : array like
        The number of events at each source (0 is allowed: a source with
        no events in a long exposure is evidence of a low rate).
    exposure : array like
        Each source's exposure: the total time at risk for
        ``kind="rate"`` (any unit; the rates are per that unit), or the
        number of trials (demands) for ``kind="probability"``.
    kind : {"rate", "probability"}, optional
        ``"rate"`` (default): events in a time at risk, Poisson within a
        source, gamma between sources. ``"probability"``: events in a
        number of trials, binomial within a source, beta between sources.
    labels : sequence, optional
        A name for each source; default ``"source 1"``, ``"source 2"``,
        ...
    method : {"Bayes", "MLE"}, optional
        How the population's two parameters are estimated.

        - ``"Bayes"`` (default): their posterior on a grid under a weak
          prior (flat on the log of the mean rate, or the logit of the
          mean probability; ``1 / (1 + a)^2`` on the shape ``a``). Every
          interval averages over that posterior, so it carries the
          uncertainty in how spread the sources are. Exact up to the
          grid, deterministic, and calibrated for small numbers of
          sources.
        - ``"MLE"``: marginal maximum likelihood (the negative binomial
          or beta-binomial likelihood, the empirical-Bayes estimate). The
          average's interval is the profile-likelihood interval; the new
          source's interval and the shrunk intervals average over a
          parametric bootstrap of the population (``n_boot`` refits).
    alpha_ci : float, optional
        Every interval is two-sided ``1 - alpha_ci``. Default 0.05 (95%);
        ``alpha_ci=0.1`` gives the 5th to 95th percentiles usual for
        generic rate data.
    min_sources : int, optional
        The fewest sources a population is estimated from. Default 4:
        with two or three, the spread between sources cannot be told
        apart from the sources themselves, and the range for a new source
        is very wide or wrong. With fewer, the result gives the sources
        side by side (each with its own interval), the one-pot estimate
        and the heterogeneity test, and no population, with a warning.
    n_boot : int, optional
        The bootstrap refits for ``method="MLE"``. Default 200.
    random_state : None, int or numpy.random.Generator, optional
        The seed of the bootstrap (``method="MLE"`` only).

    Returns
    -------
    PooledRate

    Notes
    -----
    For rates, source :math:`i` has :math:`x_i` events in exposure
    :math:`T_i`, :math:`x_i \sim \text{Poisson}(\lambda_i T_i)`, and the
    sources' rates are a sample :math:`\lambda_i \sim \text{Gamma}(a,
    b)` (shape :math:`a`, rate :math:`b`, mean :math:`a / b`). With
    :math:`\lambda_i` integrated out, :math:`x_i` is negative binomial,
    and the likelihood of :math:`(a, b)` is the product over the sources.
    For probabilities, :math:`x_i \sim \text{Binomial}(n_i, p_i)`,
    :math:`p_i \sim \text{Beta}(a, b)` and :math:`x_i` is beta-binomial.

    Given the population, a source's own rate has the conjugate
    posterior :math:`\text{Gamma}(a + x_i, b + T_i)` (or
    :math:`\text{Beta}(a + x_i, b + n_i - x_i)`): the shrunk estimate.
    The new source's distribution is the population itself. Under
    ``"Bayes"`` each is averaged over the posterior of :math:`(a, b)`;
    under ``"MLE"`` over the bootstrap refits (the point values are the
    plug-in ones at the MLE). This is the "two-stage" or hierarchical
    Bayes treatment of generic data in probabilistic risk assessment
    (NUREG/CR-6823, chapter 8); its empirical-Bayes form fixes
    :math:`(a, b)` at their MLE.

    The heterogeneity test is Pearson's chi-square test that every
    source has one rate (``k - 1`` degrees of freedom; NUREG/CR-6823,
    section 6.2.3); it is approximate when the expected counts are small.
    :math:`I^2 = \max(0, (Q - \text{df}) / Q)` is the share of the
    variation beyond chance (Higgins and Thompson 2002).

    A source's own interval is the Jeffreys interval
    (:math:`\text{Gamma}(x + 1/2, T)`, :math:`\text{Beta}(x + 1/2, n - x
    + 1/2)`), which is defined with no events.

    References
    ----------
    Atwood, C. L., LaChance, J. L., Martz, H. F., Anderson, D. J.,
    Englehardt, M., Whitehead, D. and Wheeler, T. (2003), *Handbook of
    Parameter Estimation for Probabilistic Risk Assessment*, NUREG/CR-6823,
    US Nuclear Regulatory Commission, chapters 6 and 8.

    Gaver, D. P. and O'Muircheartaigh, I. G. (1987), "Robust empirical
    Bayes analyses of event rates", Technometrics 29, 1-15.

    Gelfand, A. E. and Smith, A. F. M. (1990), "Sampling-based approaches
    to calculating marginal densities", JASA 85, 398-409.

    Higgins, J. P. T. and Thompson, S. G. (2002), "Quantifying
    heterogeneity in a meta-analysis", Statistics in Medicine 21,
    1539-1558.

    Examples
    --------
    Six sources of failures to start, out of a number of demands:

    >>> from surpyval import pooled_rate
    >>> pooled = pooled_rate(
    ...     [227, 137, 25, 3, 42, 6],
    ...     [410593, 49000, 2100, 260, 3700, 2400],
    ...     kind="probability",
    ...     alpha_ci=0.1,
    ... )
    >>> p = pooled.predictive
    >>> print(round(p["median"], 4), round(p["lower"], 5))
    0.0046 0.00017
    >>> print(round(p["upper"], 3))
    0.026
    """
    if kind not in _FAMILIES:
        raise ValueError(
            "'kind' must be 'rate' (events in a time at risk) or "
            "'probability' (events in a number of trials); got {!r}.".format(
                kind
            )
        )
    if method not in ("Bayes", "MLE"):
        raise ValueError(f"'method' must be 'Bayes' or 'MLE'; got {method!r}.")
    if not (isinstance(alpha_ci, (int, float)) and 0 < alpha_ci < 1):
        raise alpha_ci_error(alpha_ci)
    if (
        isinstance(min_sources, bool)
        or not isinstance(min_sources, (int, np.integer))
        or min_sources < 2
    ):
        raise ValueError(
            "'min_sources' must be a whole number, 2 or more; got "
            "{!r}.".format(min_sources)
        )
    if (
        isinstance(n_boot, bool)
        or not isinstance(n_boot, (int, np.integer))
        or n_boot < 1
    ):
        raise ValueError(
            "'n_boot' must be a whole number, 1 or more; got {!r}.".format(
                n_boot
            )
        )
    fam = _FAMILIES[kind]
    x = _as_counts("events", events)
    T = _as_counts("exposure", exposure)
    if x.size != T.size:
        raise ValueError(
            "'events' and 'exposure' must have one value per source; got "
            "{} and {}.".format(x.size, T.size)
        )
    if x.size == 0:
        raise ValueError(
            "'events' and 'exposure' are empty: give one value per source."
        )
    if np.any(x < 0):
        raise ValueError(f"'events' must not be negative; got {x.tolist()}.")
    if np.any(T <= 0):
        raise ValueError(
            "'exposure' must be more than 0 for every source; got {}. Leave "
            "out a source with no exposure.".format(T.tolist())
        )
    if kind == "probability":
        if np.any(x > T):
            raise ValueError(
                "With kind='probability', 'events' cannot exceed 'exposure' "
                "(the number of trials); got events {} and trials {}.".format(
                    x.tolist(), T.tolist()
                )
            )
        if np.any(x != np.round(x)) or np.any(T != np.round(T)):
            raise ValueError(
                "With kind='probability', 'events' and 'exposure' (the "
                "number of trials) must be whole numbers; got {} and "
                "{}.".format(x.tolist(), T.tolist())
            )
    if labels is None:
        names = [f"source {i + 1}" for i in range(x.size)]
    else:
        names = [str(v) for v in labels]
        if len(names) != x.size:
            raise ValueError(
                "'labels' must have one name per source ({}); got {}.".format(
                    x.size, len(names)
                )
            )
    return PooledRate._fit(
        fam,
        x,
        T,
        names,
        method,
        float(alpha_ci),
        int(min_sources),
        int(n_boot),
        random_state,
    )


def _interval(a: Any, b: Any, fam: Any, alpha_ci: float) -> tuple[Any, Any]:
    return fam.ppf(a, b, alpha_ci / 2), fam.ppf(a, b, 1 - alpha_ci / 2)


class PooledRate:
    """What :func:`pooled_rate` estimated from several sources' counts.

    Attributes
    ----------
    kind : str
        ``"rate"`` (events per unit of exposure) or ``"probability"``
        (events per trial).
    method : str
        ``"Bayes"`` or ``"MLE"``.
    alpha_ci : float
        Every interval is two-sided ``1 - alpha_ci``.
    labels, events, exposure :
        The sources, as given.
    pooled : bool
        Whether a population was estimated. ``False`` with fewer than
        ``min_sources`` sources, or with no events at any source (and for
        a probability, with every trial an event), when :attr:`message`
        says why; the per-source results, :attr:`common` and the
        heterogeneity test are still given.
    message : str
        Why there is no population (``""`` when there is one).
    population : ParametricDistribution or None
        The between-source distribution, as a SurPyval ``Gamma`` (shape,
        rate) or ``Beta`` (a, b). Under ``"MLE"`` it is the MLE. Under
        ``"Bayes"`` the population averaged over the posterior is not
        exactly a gamma or beta; this is the gamma or beta with the same
        median and upper ``1 - alpha_ci / 2`` quantile as
        :attr:`predictive` (its upper side, which matters most, is
        kept). It updates with a new source's own counts by conjugacy
        (see :meth:`update` for the exact update).
    population_params : dict or None
        The population's parameters: ``{"shape", "rate", "mean"}`` for a
        gamma, ``{"a", "b", "mean"}`` for a beta, and ``"cv"``, its
        coefficient of variation (0 when there is no spread). The MLE's
        shape is ``inf`` when the likelihood is largest with no spread.
    average : dict or None
        The population mean (the average across sources): ``estimate``
        (the MLE, or under ``"Bayes"`` the posterior median), ``lower``,
        ``upper``.
    predictive : dict or None
        A new source like these: ``mean``, ``median``, ``lower``,
        ``upper`` of the population averaged over the uncertainty in its
        parameters.
    common : dict
        Everything in one pot (assumes one rate for all): ``estimate``
        (``sum events / sum exposure``), ``lower``, ``upper`` (Jeffreys).
    heterogeneity : dict
        ``chi2``, ``df`` and ``p_value`` of Pearson's test that every
        source has one rate, and ``I2``; ``None`` values with one source.

    Examples
    --------
    >>> from surpyval import pooled_rate
    >>> pooled = pooled_rate([2, 5, 9, 4, 0], [1000, 1200, 900, 1500, 800])
    >>> pooled.pooled, pooled.n_sources
    (True, 5)
    >>> sorted(pooled.predictive)
    ['lower', 'mean', 'median', 'upper']
    """

    def __init__(self) -> None:
        self.kind: str = "rate"
        self.method: str = "Bayes"
        self.alpha_ci: float = 0.05
        self.min_sources: int = MIN_SOURCES
        self.labels: list[str] = []
        self.events: npt.NDArray = np.array([])
        self.exposure: npt.NDArray = np.array([])
        self.pooled: bool = False
        self.message: str = ""
        self.population_params: dict | None = None
        self.average: dict | None = None
        self.predictive: dict | None = None
        self.common: dict = {}
        self.heterogeneity: dict = {}
        self._sources: dict = {}
        # the population's parameter sets the intervals average over
        # (posterior cells or bootstrap refits) and their weights
        self._mix: dict | None = None

    # -- fitting ---------------------------------------------------------

    @classmethod
    def _fit(
        cls,
        fam: Any,
        x: Any,
        T: Any,
        names: list[str],
        method: str,
        alpha_ci: float,
        min_sources: int,
        n_boot: int,
        random_state: Any,
        log_prior: Any = None,
    ) -> PooledRate:
        from scipy.stats import chi2 as chi2_dist

        out = cls()
        out.kind = fam.kind
        out.method = method
        out.alpha_ci = alpha_ci
        out.min_sources = min_sources
        out.labels = names
        out.events = x
        out.exposure = T
        k = x.size
        lo_p, hi_p = alpha_ci / 2, 1 - alpha_ci / 2

        # each source on its own (Jeffreys) and everything in one pot
        oa, ob = fam.own(x, T)
        olo, ohi = _interval(oa, ob, fam, alpha_ci)
        sources = {
            "events": x,
            "exposure": T,
            "own": x / T,
            "own_lower": olo,
            "own_upper": ohi,
        }
        ca, cb = fam.own(x.sum(), T.sum())
        clo, chi = _interval(ca, cb, fam, alpha_ci)
        out.common = {
            "estimate": float(x.sum() / T.sum()),
            "lower": float(clo),
            "upper": float(chi),
        }
        if k >= 2:
            if fam is _Binomial and (x.sum() == 0 or x.sum() == T.sum()):
                stat, df, pv = 0.0, k - 1, 1.0
            elif x.sum() == 0:
                stat, df, pv = 0.0, k - 1, 1.0
            else:
                stat, df = fam.chi2(x, T)
                pv = float(chi2_dist.sf(stat, df))
            out.heterogeneity = {
                "chi2": stat,
                "df": df,
                "p_value": pv,
                "I2": float(max(0.0, (stat - df) / stat)) if stat > 0 else 0.0,
            }
        else:
            out.heterogeneity = {
                "chi2": None,
                "df": None,
                "p_value": None,
                "I2": None,
            }

        what = "rate" if fam is _Poisson else "probability"
        if k < min_sources:
            out.message = (
                "{} source{} given; a spread between sources needs at least "
                "{} (min_sources). With so few, how different another source "
                "would be cannot be estimated, so no population or range for "
                "a new source is given: compare the sources side by side "
                "(their own intervals){}.".format(
                    k,
                    "" if k == 1 else "s",
                    min_sources,
                    "" if k == 1 else " and see the heterogeneity test",
                )
            )
        elif x.sum() == 0:
            out.message = (
                "No source had an event, so the spread of the {}s between "
                "sources cannot be estimated; use each source's own "
                "(Jeffreys) interval, or the one-pot estimate if the sources "
                "are alike.".format(what)
            )
        elif fam is _Binomial and x.sum() == T.sum():
            out.message = (
                "Every trial at every source was an event, so the spread of "
                "the probabilities between sources cannot be estimated."
            )
        if out.message:
            sources.update(
                shrunk=np.full(k, np.nan),
                shrunk_lower=np.full(k, np.nan),
                shrunk_upper=np.full(k, np.nan),
            )
            out._sources = sources
            warnings.warn(
                "pooled_rate: " + out.message, UserWarning, stacklevel=3
            )
            return out

        out.pooled = True
        point_a: float | None = None
        point_b: float | None = None
        if method == "Bayes":
            L, U, w = _bayes_grid(fam, x, T, log_prior)
            a, b = fam.params(L, U)
            locs = L
            # the population mean: its posterior median and interval
            # (each grid row's weight spread evenly over its width)
            rows, inv = np.unique(L, return_inverse=True)
            mass = np.bincount(inv, weights=w)
            half = (np.min(np.diff(rows)) if rows.size > 1 else 1e-9) / 2
            qs = np.interp(
                [0.5, lo_p, hi_p],
                np.concatenate([[0.0], np.cumsum(mass)]),
                np.concatenate([[rows[0] - half], rows + half]),
            )
            est, alo, ahi = (float(v) for v in fam.from_loc(qs))
            out.average = {"estimate": est, "lower": alo, "upper": ahi}
        else:
            loc, u, ll_hat = _mle(fam, x, T)
            alo, ahi = _profile_interval(fam, x, T, loc, ll_hat, alpha_ci)
            out.average = {
                "estimate": float(fam.from_loc(loc)),
                "lower": alo,
                "upper": ahi,
            }
            from surpyval.utils.rng import as_generator

            rng = as_generator(random_state)
            bl, bu = _bootstrap_fits(fam, x, T, loc, u, n_boot, rng)
            a, b = _params_inf(fam, bl, bu)
            locs = bl
            w = np.full(a.size, 1.0 / a.size)
            pa_arr, pb_arr = _params_inf(fam, np.array([loc]), np.array([u]))
            point_a, point_b = float(pa_arr[0]), float(pb_arr[0])
        out._mix = {"loc": locs, "a": a, "b": b, "w": w}

        mean, (med, plo, phi) = _mixture_summary(
            fam, a, b, w, (0.5, lo_p, hi_p)
        )
        out.predictive = {
            "mean": mean,
            "median": med,
            "lower": plo,
            "upper": phi,
        }

        # the shrunk estimate of each source
        sh, shl, shu = np.empty(k), np.empty(k), np.empty(k)
        for i in range(k):
            ca_i, cb_i = fam.conditional(a, b, x[i], T[i])
            ca_i = np.where(np.isfinite(a), ca_i, np.inf)
            cb_i = np.where(np.isfinite(a), cb_i, b)
            m_i, (q1, q2) = _mixture_summary(fam, ca_i, cb_i, w, (lo_p, hi_p))
            if point_a is not None and point_b is not None:
                if np.isfinite(point_a):
                    pa, pb = fam.conditional(point_a, point_b, x[i], T[i])
                    m_i = float(fam.mean(pa, pb))
                else:
                    m_i = point_b
            sh[i], shl[i], shu[i] = m_i, q1, q2
        sources.update(shrunk=sh, shrunk_lower=shl, shrunk_upper=shu)
        out._sources = sources

        # the population as one distribution
        if point_a is not None and point_b is not None:
            pa_, pb_ = point_a, point_b
        else:
            pa_, pb_ = _matched(fam, med, phi, hi_p)
        out.population_params = _population_params(fam, pa_, pb_)
        return out

    # -- results ---------------------------------------------------------

    @property
    def n_sources(self) -> int:
        """The number of sources."""
        return int(self.events.size)

    @property
    def population(self) -> Any:
        """The population (between-source) distribution as a SurPyval
        ``Gamma`` or ``Beta`` (``None`` without a population, or for an
        MLE with no spread, whose population is the single value
        ``population_params["mean"]``)."""
        if self.population_params is None:
            return None
        import surpyval

        pp = self.population_params
        if self.kind == "rate":
            if not np.isfinite(pp["shape"]):
                return None
            return surpyval.Gamma.from_params([pp["shape"], pp["rate"]])
        if not np.isfinite(pp["a"]):
            return None
        return surpyval.Beta.from_params([pp["a"], pp["b"]])

    @property
    def sources(self) -> pd.DataFrame:
        """One row per source: ``events``, ``exposure``, ``own`` (events
        / exposure) with its Jeffreys interval ``own_lower`` and
        ``own_upper``, and ``shrunk`` (the posterior mean given the
        population) with ``shrunk_lower`` and ``shrunk_upper`` (``nan``
        without a population)."""
        import pandas as pd

        return pd.DataFrame(
            {k: np.asarray(v, float) for k, v in self._sources.items()},
            index=pd.Index(self.labels, name="source"),
        )

    def update(self, events: float, exposure: float) -> dict:
        """A new source's estimate from its own counts and the
        population (the second stage): its posterior given these sources
        and its own ``events`` in ``exposure``.

        Parameters
        ----------
        events : float
            The new source's events.
        exposure : float
            Its time at risk (``kind="rate"``) or trials
            (``kind="probability"``).

        Returns
        -------
        dict
            ``mean``, ``median``, ``lower`` and ``upper`` (two-sided
            ``1 - alpha_ci``), averaged over the population's uncertainty
            as :attr:`predictive` is.

        Examples
        --------
        >>> from surpyval import pooled_rate
        >>> pooled = pooled_rate([2, 5, 9, 4, 0], [1000, 1200, 900, 1500, 800])
        >>> new = pooled.update(1, 400)
        >>> new["lower"] < new["mean"] < new["upper"]
        True
        """
        if not self.pooled or self._mix is None:
            raise ValueError(
                f"There is no population to update: {self.message}"
            )
        fam = _FAMILIES[self.kind]
        x = float(events)
        T = float(exposure)
        if not (np.isfinite(x) and np.isfinite(T)) or x < 0 or T <= 0:
            raise ValueError(
                "'events' must be 0 or more and 'exposure' more than 0; got "
                "{} and {}.".format(events, exposure)
            )
        if self.kind == "probability" and x > T:
            raise ValueError(
                "'events' cannot exceed 'exposure' (the trials); got {} and "
                "{}.".format(events, exposure)
            )
        a, b, w = self._mix["a"], self._mix["b"], self._mix["w"]
        ca, cb = fam.conditional(a, b, x, T)
        ca = np.where(np.isfinite(a), ca, np.inf)
        cb = np.where(np.isfinite(a), cb, b)
        if np.all(np.isfinite(a)):
            # weight each population by how likely it made these counts
            lw = fam.loglik(x, T, *_loc_u(fam, a, b))
            w = w * np.exp(lw - lw.max())
        lo_p, hi_p = self.alpha_ci / 2, 1 - self.alpha_ci / 2
        mean, (med, lo, hi) = _mixture_summary(
            fam, ca, cb, w, (0.5, lo_p, hi_p)
        )
        return {"mean": mean, "median": med, "lower": lo, "upper": hi}

    @property
    def assumptions(self) -> list[str]:
        """The assumptions behind the numbers, in words."""
        unit = "rate" if self.kind == "rate" else "probability per trial"
        within = (
            "events at a source follow a Poisson process with a constant "
            "rate over its exposure"
            if self.kind == "rate"
            else "the trials at a source are independent, with one "
            "probability"
        )
        between = (
            "the sources' true {}s are a sample from one {} distribution, "
            "independently of their exposure".format(
                unit, "gamma" if self.kind == "rate" else "beta"
            )
        )
        out = [
            "Within a source: " + within + ".",
            "Between sources: " + between + "; a new source is like these.",
            "The sources are independent (no source repeats another's data).",
        ]
        if self.method == "Bayes":
            out.append(
                "Prior on the population: flat on the log of the mean rate "
                "(the logit of the mean probability) and 1 / (1 + a)^2 on "
                "the shape a; intervals average over its posterior."
            )
        else:
            out.append(
                "Population fitted by marginal maximum likelihood; the "
                "average's interval is the profile-likelihood interval, the "
                "others average over a parametric bootstrap. With few "
                "sources these intervals are too narrow (in a simulation "
                "with 6 sources, a nominal 90% range for a new source "
                "covered 72% of the time, against 88% under method="
                "'Bayes')."
            )
        return out

    @property
    def method_description(self) -> str:
        """The method, in one sentence."""
        fam = _FAMILIES[self.kind]
        how = (
            "posterior on a grid under a weak prior"
            if self.method == "Bayes"
            else "marginal maximum likelihood with profile-likelihood and "
            "parametric-bootstrap intervals"
        )
        return "Two-stage hierarchical {} model ({} marginal), {}.".format(
            "gamma-Poisson" if fam is _Poisson else "beta-binomial",
            fam.marginal,
            how,
        )

    def __repr__(self) -> str:
        pct = f"{100 * (1 - self.alpha_ci):g}%"
        what = "rate" if self.kind == "rate" else "probability"
        lines = [
            "Pooled {} from {} source{} ({}, {} intervals)".format(
                what,
                self.n_sources,
                "" if self.n_sources == 1 else "s",
                self.method,
                pct,
            )
        ]
        c = self.common
        lines.append(
            "  One pot (assumes one {} for all): {:.4g} "
            "({:.4g} to {:.4g})".format(
                what, c["estimate"], c["lower"], c["upper"]
            )
        )
        if self.pooled and self.average and self.predictive:
            av, pr = self.average, self.predictive
            lines.append(
                "  Average across sources: {:.4g} ({:.4g} to {:.4g})".format(
                    av["estimate"], av["lower"], av["upper"]
                )
            )
            lines.append(
                "  A new source like these: median {:.4g}, mean {:.4g} "
                "({:.4g} to {:.4g})".format(
                    pr["median"], pr["mean"], pr["lower"], pr["upper"]
                )
            )
        else:
            lines.append("  No population: " + self.message)
        h = self.heterogeneity
        if h.get("chi2") is not None:
            lines.append(
                "  Heterogeneity: chi-square {:.4g} on {} df, p = {:.3g}, "
                "I2 = {:.0%}".format(h["chi2"], h["df"], h["p_value"], h["I2"])
            )
        return "\n".join(lines)

    # -- saving ------------------------------------------------------------

    def to_dict(self) -> dict:
        """The result as a dict of native Python values, strict JSON (a
        non-finite number is written as ``None`` and recorded under
        ``"non_finite"``). It holds the population's parameter sets
        (posterior cells or bootstrap refits), so :meth:`update` works
        on the restored result.

        Examples
        --------
        >>> import json
        >>> from surpyval import pooled_rate, PooledRate
        >>> pooled = pooled_rate([2, 5, 9, 4, 0], [1000, 1200, 900, 1500, 800])
        >>> text = json.dumps(pooled.to_dict())
        >>> back = PooledRate.from_dict(json.loads(text))
        >>> back.predictive == pooled.predictive
        True
        """
        from surpyval.serialisation import encode_non_finite, to_native

        out = {
            "result": type(self).__name__,
            "schema": 1,
            "kind": self.kind,
            "method": self.method,
            "alpha_ci": float(self.alpha_ci),
            "min_sources": int(self.min_sources),
            "labels": list(self.labels),
            "pooled": bool(self.pooled),
            "message": self.message,
            "population_params": to_native(self.population_params),
            "average": to_native(self.average),
            "predictive": to_native(self.predictive),
            "common": to_native(self.common),
            "heterogeneity": to_native(self.heterogeneity),
            "sources": {
                k: to_native(np.asarray(v, float))
                for k, v in self._sources.items()
            },
            "mix": (
                None
                if self._mix is None
                else {
                    k: to_native(np.asarray(v, float))
                    for k, v in self._mix.items()
                }
            ),
        }
        return encode_non_finite(out)

    @classmethod
    def from_dict(cls, result_dict: dict) -> PooledRate:
        """The result :meth:`to_dict` wrote.

        Parameters
        ----------
        result_dict : dict
            A dictionary from :meth:`to_dict`.

        Returns
        -------
        PooledRate
        """
        from surpyval.serialisation import decode_non_finite

        if (
            not isinstance(result_dict, dict)
            or result_dict.get("result") != cls.__name__
        ):
            raise ValueError(
                "'result_dict' is not a PooledRate dictionary (from "
                "PooledRate.to_dict()); its 'result' is {!r}.".format(
                    result_dict.get("result")
                    if isinstance(result_dict, dict)
                    else None
                )
            )
        d = decode_non_finite(result_dict)
        out = cls()
        out.kind = d["kind"]
        out.method = d["method"]
        out.alpha_ci = float(d["alpha_ci"])
        out.min_sources = int(d["min_sources"])
        out.labels = list(d["labels"])
        out.pooled = bool(d["pooled"])
        out.message = d["message"]
        out.population_params = d["population_params"]
        out.average = d["average"]
        out.predictive = d["predictive"]
        out.common = d["common"]
        out.heterogeneity = d["heterogeneity"]
        out._sources = {
            k: np.asarray(v, float) for k, v in d["sources"].items()
        }
        out.events = out._sources["events"]
        out.exposure = out._sources["exposure"]
        out._mix = (
            None
            if d["mix"] is None
            else {k: np.asarray(v, float) for k, v in d["mix"].items()}
        )
        return out


def _params_inf(fam: Any, locs: Any, us: Any) -> tuple[Any, Any]:
    """(a, b) for each (loc, u); ``a = inf`` and ``b`` the mean where
    ``u`` is infinite (no spread)."""
    fin = np.isfinite(us)
    a, b = fam.params(locs, np.where(fin, us, 0.0))
    return np.where(fin, a, np.inf), np.where(fin, b, fam.from_loc(locs))


def _loc_u(fam: Any, a: Any, b: Any) -> tuple[Any, Any]:
    """(loc, u) of finite (a, b)."""
    mean = fam.mean(a, b)
    return fam.to_loc(mean), np.log(a)


def _population_params(fam: Any, a: float, b: float) -> dict:
    if fam is _Poisson:
        if np.isfinite(a):
            return {
                "shape": a,
                "rate": b,
                "mean": a / b,
                "cv": float(1 / np.sqrt(a)),
            }
        return {"shape": np.inf, "rate": np.inf, "mean": b, "cv": 0.0}
    if np.isfinite(a):
        mean = a / (a + b)
        sd = np.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        return {"a": a, "b": b, "mean": mean, "cv": float(sd / mean)}
    return {"a": np.inf, "b": np.inf, "mean": b, "cv": 0.0}
