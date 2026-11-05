"""Log-rank and Wilcoxon-type tests of interval-censored data.

The score tests of Finkelstein (1986), Sun (1996) and Fay (1999), built on
the Turnbull NPMLE of the pooled data, in the permutation form of Fay and
Shaw (2010) (R's ``interval::ictest``).
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils import xcnt_handler

#: The ``scores`` values of :func:`interval_logrank`, and R's names for them.
SCORES = ("log-rank", "finkelstein", "wilcoxon")
_SCORE_ALIASES = {
    "logrank1": "log-rank",
    "logrank2": "finkelstein",
    "wmw": "wilcoxon",
}
#: The ``method`` values of :func:`interval_logrank`.
METHODS = ("asymptotic", "permutation")


class IntervalLogRankResult:
    """
    Result of a log-rank or Wilcoxon-type test of interval-censored data,
    from :func:`interval_logrank`.

    Attributes
    ----------
    statistic : float
        The chi-squared statistic: the score sums of the groups compared
        with their permutation covariance.
    dof : int
        The degrees of freedom, the number of groups less one; 0 when
        there is nothing to compare (every unit has the same score).
    p_value : float
        The p-value: from the chi-squared distribution with ``method``
        "asymptotic", or the share of the random relabellings whose
        statistic is at least the observed one with "permutation".
    scores : str
        The scores: "log-rank" (Sun's), "finkelstein" or "wilcoxon".
    method : str
        How the p-value was found: "asymptotic" or "permutation".
    U : dict
        The sum of the units' scores in each group, keyed by group label.
        A positive sum means the group failed earlier than the pooled
        data suggest (more events than expected).
    n_permutations : int or None
        The number of random relabellings behind a permutation p-value;
        None for the asymptotic one.
    npmle_converged : bool
        Whether the Turnbull NPMLE the scores come from converged.

    Examples
    --------
    >>> from surpyval import interval_logrank
    >>> xl = [0, 2, 4, 1, 5, 6, 8, 7]
    >>> xr = [3, 5, 6, 4, 9, 10, 12, 11]
    >>> Z = [0, 0, 0, 0, 1, 1, 1, 1]
    >>> res = interval_logrank(xl=xl, xr=xr, Z=Z)
    >>> res
    Interval-Censored Log-Rank Test
    ===============================
    Scores           : log-rank
    p-value method   : asymptotic
    Statistic        : 6.89447
    DoF              : 1
    p-value          : 0.00864627
    >>> res.to_dict()["dof"]
    1
    """

    def __init__(
        self,
        statistic: float,
        dof: int,
        p_value: float,
        scores: str,
        method: str,
        U: dict,
        n_permutations: int | None,
        npmle_converged: bool,
    ) -> None:
        self.statistic = statistic
        self.dof = dof
        self.p_value = p_value
        self.scores = scores
        self.method = method
        self.U = U
        self.n_permutations = n_permutations
        self.npmle_converged = npmle_converged

    def __repr__(self) -> str:
        header = "Interval-Censored Log-Rank Test"
        if self.scores == "wilcoxon":
            header = "Interval-Censored Wilcoxon Test"
        out = (
            header
            + "\n"
            + "=" * len(header)
            + f"\nScores           : {self.scores}"
            + f"\np-value method   : {self.method}"
        )
        if self.n_permutations is not None:
            out += f"\nPermutations     : {self.n_permutations}"
        out += (
            f"\nStatistic        : {self.statistic:.6g}"
            + f"\nDoF              : {self.dof}"
            + f"\np-value          : {self.p_value:.6g}"
        )
        return out

    def to_dict(self) -> dict:
        """The result as strict JSON: the settings, the statistic, its
        degrees of freedom and p-value, and each group's score sum (a
        non-finite number is ``None``)."""

        def finite(v: Any) -> Any:
            v = float(v)
            return v if np.isfinite(v) else None

        return {
            "test": "interval-censored log-rank",
            "scores": self.scores,
            "method": self.method,
            "statistic": finite(self.statistic),
            "dof": int(self.dof),
            "p_value": finite(self.p_value),
            "U": [
                {"group": _plain(g), "score_sum": finite(u)}
                for g, u in self.U.items()
            ],
            "n_permutations": self.n_permutations,
            "npmle_converged": bool(self.npmle_converged),
        }


def _plain(v: Any) -> Any:
    # A numpy scalar label as the Python value JSON can write.
    return v.item() if isinstance(v, np.generic) else v


def _pooled_scores(
    xl: npt.NDArray,
    xr: npt.NDArray,
    n: npt.NDArray,
    scores: str,
    tol: float,
    max_iter: int,
) -> tuple[npt.NDArray, bool]:
    """Each row's score from the Turnbull NPMLE of the pooled data, and
    whether the NPMLE converged.

    With ``S`` the NPMLE's survival at the ends of the innermost intervals
    and a row known to fail in ``(a, b]``, the score is the average over
    the row's interval, under the NPMLE, of the scores of the innermost
    intervals inside it. It telescopes to

        (psi(S(a)) - psi(S(b))) / (S(a) - S(b)),

    with ``psi(s) = s log s`` for Finkelstein's scores and ``s**2 - s``
    for the Wilcoxon's (so ``S(a) + S(b) - 1``); Sun's log-rank scores use
    ``S log S~`` instead, ``S~`` the exponential of minus the cumulative
    hazard, ``sum p_j / S(before j)`` (Fay and Shaw 2010, section 3).
    """
    from .turnbull import (
        _bounds,
        _censoring_supports,
        _Cliques,
        _coverage,
        _emicm,
        _Ranges,
    )

    c = np.where(xl == xr, 0, 2)
    t = np.column_stack([np.full(xl.size, -np.inf), np.full(xl.size, np.inf)])
    bounds = _bounds(np.column_stack([xl, xr]), c, t)
    exact_times = np.unique(xl[c == 0])
    M = bounds.size
    lo, hi, *_ = _censoring_supports(bounds, xl, xr, exact_times)
    no_rows = np.empty(0, dtype=np.int64)
    ranges = _Ranges(lo, hi, n, no_rows, no_rows, no_rows)
    identifiable = _coverage(lo, hi, M) > 0
    em = _emicm(ranges, M, identifiable, tol, max_iter)

    # The NPMLE's mass on each innermost interval (clique), and the
    # survival at the clique ends: S[j] before clique j, S[K] = 0.
    cl = _Cliques(lo, hi, n, M)
    q = np.bincount(cl.clique, weights=em.p[cl.pieces], minlength=cl.K)
    q = q / q.sum()
    S = np.clip(1.0 - np.concatenate([[0.0], np.cumsum(q)]), 0.0, 1.0)
    S[-1] = 0.0
    a, b = cl.lo, cl.hi + 1
    if scores == "wilcoxon":
        return S[a] + S[b] - 1.0, em.converged
    with np.errstate(divide="ignore", invalid="ignore"):
        if scores == "finkelstein":
            psi = np.where(S > 0, S * np.log(np.where(S > 0, S, 1.0)), 0.0)
        else:
            hazard = np.where(S[:-1] > 0, q / S[:-1], 0.0)
            cumulative = np.concatenate([[0.0], np.cumsum(hazard)])
            psi = -S * cumulative
    return (psi[a] - psi[b]) / (S[a] - S[b]), em.converged


def _chisq(
    d: npt.NDArray, units: npt.NDArray, g: npt.NDArray, k: int
) -> float:
    """The k-sample permutation chi-squared statistic of the centred
    scores ``d`` (``ksample.pclt`` of R's ``perm``)."""
    N = units.sum()
    ss = np.sum(units * d * d)
    sizes = np.bincount(g, weights=units, minlength=k)
    sums = np.bincount(g, weights=units * d, minlength=k)
    return float((N - 1.0) / ss * np.sum(sums**2 / sizes))


def interval_logrank(
    x: npt.ArrayLike | None = None,
    Z: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    xl: npt.ArrayLike | None = None,
    xr: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | float | None = None,
    tr: npt.ArrayLike | float | None = None,
    scores: str = "log-rank",
    method: str = "asymptotic",
    n_permutations: int = 9999,
    random_state: Any = None,
    tol: float = 1e-12,
    max_iter: int = 10000,
) -> IntervalLogRankResult:
    r"""
    The k-sample log-rank or Wilcoxon-type test for the equality of
    survival distributions of interval-censored data.

    When failures are found at inspections, each unit's failure time is
    only known to lie between its last good and its first failed
    inspection. :func:`~surpyval.logrank` needs exact or right censored
    times; this test takes any mix of exact, right, left and
    interval-censored rows. It first estimates one survival curve from all
    the groups together (the Turnbull NPMLE). Each unit then gets a score
    from that curve: high when it failed early, low when it lasted long,
    averaged over the interval it is known to have failed in. If the
    groups share one survival distribution, their scores are
    exchangeable, so each group's score sum is compared with how much it
    would vary were the group labels shuffled at random.

    Parameters
    ----------
    x : array like, optional
        The times: one column, with the censoring flags ``c``, or two
        columns, each row the interval ``(xl, xr]`` a failure lies in.
        Leave it out to give ``xl`` and ``xr`` instead.
    Z : array like
        The group label of each row. Any hashable values other than NaN or
        None can be used; at least two groups are needed.
    c : array like, optional
        The censoring flags: 0 observed, 1 right censored, -1 left censored
        and 2 interval censored (with a two-column ``x``). If not given,
        every one-column time is observed and every two-column row whose
        ends differ is an interval.
    n : array like, optional
        The number of units each row stands for. If not given, each row is
        one unit.
    xl : array like, optional
        The lower end of each row's interval, with ``xr`` in place of
        ``x``: ``xl == xr`` is an observed time, ``xr`` infinite is right
        censored at ``xl``, and ``xl`` 0 (for positive times) or minus
        infinity is left censored at ``xr``.
    xr : array like, optional
        The upper end of each row's interval, with ``xl``.
    tl : array like or scalar, optional
        Left truncation is not supported: the permutation argument needs
        every unit to be exchangeable, which delayed entry breaks. A
        finite ``tl`` is refused; for exact and right censored data with
        delayed entry use :func:`~surpyval.logrank` with ``tl``.
    tr : array like or scalar, optional
        Right truncation is not supported either; a finite ``tr`` is
        refused.
    scores : str, optional
        The score each unit gets from the pooled survival curve. One of:

        - "log-rank" (the default): Sun's (1996) log-rank scores, which
          for exact and right censored data are the usual log-rank scores
          (1 minus the Nelson-Aalen cumulative hazard at an event, minus
          the cumulative hazard at a censoring time). Most powerful when
          the hazards are proportional. R's ``scores = "logrank1"``.
        - "finkelstein": Finkelstein's (1986) log-rank scores, the score
          test of a proportional hazards model of the grouped data. R's
          ``"logrank2"``.
        - "wilcoxon": the Wilcoxon-Mann-Whitney scores ``S(a) + S(b) - 1``
          (Peto and Peto 1972; Fay 1999), which give more weight to early
          differences and are most powerful under proportional odds. R's
          ``"wmw"``.

        R's names are accepted too.
    method : str, optional
        How the p-value is found:

        - "asymptotic" (the default): the chi-squared distribution with
          k - 1 degrees of freedom (the permutational central limit
          theorem; R's ``method = "pclt"``).
        - "permutation": the share of ``n_permutations`` random
          relabellings of the units (the scores held fixed) whose
          statistic is at least the observed one, counting the observed
          labelling once: ``(1 + count) / (1 + n_permutations)``, as R's
          ``method = "exact.mc"``. Use it for small samples.
    n_permutations : int, optional
        The number of random relabellings with ``method="permutation"``.
        Defaults to 9999. The p-value's Monte Carlo standard error is
        about ``sqrt(p (1 - p) / n_permutations)``.
    random_state : None, int or numpy.random.Generator, optional
        The seed of the relabellings, as everywhere in SurPyval: None
        draws from numpy's global random number generator, an int or a
        Generator gives a stream of its own.
    tol : float, optional
        The convergence tolerance of the pooled Turnbull NPMLE (the
        largest relative gap in its Karush-Kuhn-Tucker conditions).
        Defaults to 1e-12.
    max_iter : int, optional
        The most EM-ICM iterations for the NPMLE. Defaults to 10000.

    Returns
    -------
    IntervalLogRankResult
        With the chi-squared ``statistic``, the degrees of freedom
        ``dof``, the ``p_value`` and each group's score sum ``U``.

    Raises
    ------
    ValueError
        If there are fewer than two groups, ``Z`` is missing, has missing
        labels or the wrong length, the data are truncated, ``scores`` or
        ``method`` is unknown, or ``n_permutations`` is not a positive
        whole number.

    Warns
    -----
    UserWarning
        If the pooled NPMLE did not converge in ``max_iter`` iterations.

    Notes
    -----
    With ``S`` the pooled NPMLE of the survival function and a unit known
    to fail in :math:`(a_i, b_i]`, its score is

    .. math::

        c_i = \frac{\psi(S(a_i)) - \psi(S(b_i))}{S(a_i) - S(b_i)},

    with :math:`\psi(s) = s \log s` for Finkelstein's scores and
    :math:`\psi(s) = s^2 - s` for the Wilcoxon's
    (:math:`c_i = S(a_i) + S(b_i) - 1`). Sun's scores use
    :math:`\psi = S \log \tilde S`, :math:`\tilde S` the exponential of
    minus the cumulative hazard of the NPMLE's masses. With :math:`N`
    units, group sizes :math:`N_g` and :math:`\bar d_g` the mean of the
    centred scores :math:`d_i = c_i - \bar c` in group :math:`g`,

    .. math::

        \chi^2 = \frac{N - 1}{\sum_i d_i^2} \sum_g N_g \bar d_g^2,

    the statistic of a permutation test of the scores (Fay and Shaw 2010).
    For two groups it is the square of R's ``Z``.

    Examples
    --------
    Breast cosmesis after radiotherapy alone or with chemotherapy (the
    first 10 of each group of Finkelstein and Wolfe's 1985 data), months
    to the first sign of breast retraction, found at visits:

    >>> import numpy as np
    >>> from surpyval import interval_logrank
    >>> inf = np.inf
    >>> xl = [45, 6, 0, 46, 46, 7, 17, 7, 37, 0,
    ...       8, 0, 24, 17, 17, 24, 16, 13, 11, 16]
    >>> xr = [inf, 10, 7, inf, inf, 16, inf, 14, 44, 8,
    ...       12, 22, 31, 27, 23, 30, 24, inf, 13, 20]
    >>> Z = ["radiotherapy"] * 10 + ["with chemotherapy"] * 10
    >>> res = interval_logrank(xl=xl, xr=xr, Z=Z)
    >>> print(round(res.statistic, 3), round(res.p_value, 4))
    0.675 0.4115

    The Wilcoxon-type scores, and the p-value from 9999 relabellings:

    >>> res = interval_logrank(xl=xl, xr=xr, Z=Z, scores="wilcoxon")
    >>> print(round(res.statistic, 3), round(res.p_value, 4))
    0.002 0.968
    >>> res = interval_logrank(
    ...     xl=xl, xr=xr, Z=Z, method="permutation", random_state=1
    ... )
    >>> print(round(res.p_value, 2))
    0.43

    Ten units a group show no clear difference; the full 94 units do
    (log-rank p-value 0.0076; see
    :doc:`Non-Parametric SurPyval Modelling`).

    References
    ----------
    Finkelstein, D. M. (1986), "A proportional hazards model for
    interval-censored failure time data", Biometrics 42, 845-854.

    Sun, J. (1996), "A non-parametric test for interval-censored failure
    time data with application to AIDS studies", Statistics in Medicine
    15, 1387-1395.

    Fay, M. P. (1999), "Comparing several score tests for interval
    censored data", Statistics in Medicine 18, 273-285.

    Fay, M. P. and Shaw, P. A. (2010), "Exact and asymptotic weighted
    logrank tests for interval censored data: the interval R package",
    Journal of Statistical Software 36(2), 1-34.
    """
    from scipy.stats import chi2

    key = str(scores).lower()
    key = _SCORE_ALIASES.get(key, key)
    if key not in SCORES:
        raise ValueError(
            "'scores' must be one of {} (or R's 'logrank1', 'logrank2', "
            "'wmw'); got {!r}".format(list(SCORES), scores)
        )
    scores = key
    if method not in METHODS:
        raise ValueError(
            "'method' must be one of {}; got {!r}".format(
                list(METHODS), method
            )
        )
    if method == "permutation":
        if (
            isinstance(n_permutations, bool)
            or not float(n_permutations).is_integer()
            or n_permutations < 1
        ):
            raise ValueError(
                "'n_permutations' must be a positive whole number, the "
                "number of random relabellings; got {!r}".format(
                    n_permutations
                )
            )
        n_permutations = int(n_permutations)
    if Z is None:
        raise ValueError(
            "'Z', the group label of each row, is needed: "
            "interval_logrank(x, Z, ...) or interval_logrank(xl=..., "
            "xr=..., Z=...)"
        )
    for name, bound in (("tl", tl), ("tr", tr)):
        if bound is not None and np.isfinite(np.asarray(bound, float)).any():
            raise ValueError(
                "interval_logrank does not take truncated data ('{}'): the "
                "permutation test needs every unit to be exchangeable, "
                "which delayed entry or right truncation breaks. For exact "
                "and right censored data with delayed entry, use "
                "surpyval.logrank(x, Z, c, tl=...).".format(name)
            )

    xd, cd, nd, _ = xcnt_handler(
        x=x, c=c, n=n, xl=xl, xr=xr, group_and_sort=False
    )
    rows = cd.shape[0]
    Z = np.asarray(Z)
    if Z.ndim != 1:
        raise ValueError("'Z' must be a 1D array of group labels")
    if len(Z) != rows:
        raise ValueError(
            "'Z' must have a label for each observation; got {} labels "
            "for {} observations".format(len(Z), rows)
        )
    import pandas as pd

    if pd.isna(Z).any():
        raise ValueError(
            "'Z' has missing (NaN or None) group labels; drop those rows "
            "or give them a label"
        )
    groups, g = np.unique(Z, return_inverse=True)
    k = groups.size
    if k < 2:
        raise ValueError("Test requires at least two groups")

    if xd.ndim == 1:
        lower = xd.astype(float).copy()
        upper = xd.astype(float).copy()
    else:
        lower = xd[:, 0].astype(float)
        upper = xd[:, 1].astype(float)
    lower[cd == -1] = -np.inf
    upper[cd == 1] = np.inf
    units = nd.astype(float)

    cc, converged = _pooled_scores(lower, upper, units, scores, tol, max_iter)
    if not converged:
        warnings.warn(
            "interval_logrank: the Turnbull NPMLE of the pooled data did not "
            "converge to tol={} in max_iter={} iterations, so the scores "
            "are approximate; raise 'max_iter'.".format(tol, max_iter),
            stacklevel=2,
        )
    U = {
        _plain(groups[j]): float(np.sum((units * cc)[g == j]))
        for j in range(k)
    }
    d = cc - np.sum(units * cc) / units.sum()
    # No spread in the scores: every unit tells the same story (all
    # right censored before any failure, say), so there is nothing to
    # compare, as logrank's "no event" result.
    if np.sum(units * d * d) <= 1e-14 * max(1.0, np.sum(units * cc * cc)):
        return IntervalLogRankResult(
            0.0,
            0,
            1.0,
            scores,
            method,
            U,
            n_permutations if method == "permutation" else None,
            converged,
        )
    statistic = float(_chisq(d, units, g, k))
    dof = k - 1
    if method == "asymptotic":
        p_value = float(chi2.sf(statistic, dof))
        return IntervalLogRankResult(
            statistic, dof, p_value, scores, method, U, None, converged
        )

    from surpyval.utils.rng import as_generator

    rng = as_generator(random_state)
    # One entry per unit, so a row of count n is relabelled as n units.
    whole = units.astype(np.int64)
    if np.all(whole == units):
        d_units = np.repeat(d, whole)
        g_units = np.repeat(g, whole)
        ones = np.ones(d_units.size)
    else:
        d_units, g_units, ones = d, g, units
    # Each relabelling's statistic, compared with the observed one less a
    # rounding allowance (R compares to 12 significant digits).
    allowance = 1e-12 * max(1.0, abs(statistic))
    count = 0
    for _ in range(n_permutations):
        relabelled = rng.permutation(g_units)
        if _chisq(d_units, ones, relabelled, k) >= statistic - allowance:
            count += 1
    p_value = (1.0 + count) / (1.0 + n_permutations)
    return IntervalLogRankResult(
        statistic,
        dof,
        p_value,
        scores,
        method,
        U,
        n_permutations,
        converged,
    )
