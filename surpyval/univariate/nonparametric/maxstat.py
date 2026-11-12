"""Maximally selected log-rank statistics: the cut-point of a continuous
covariate that best separates survival, with a p-value that allows for
the search."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from .logrank import _NOT_RIGHT_CENSORED, WEIGHTINGS, _weighting_label
from .logrank_permutation import logrank_scores

#: The ``method`` values of :func:`maxstat_logrank`.
METHODS = ("exact", "lausen94", "lausen92", "permutation")


def _finite(v: Any) -> Any:
    v = float(v)
    return v if np.isfinite(v) else None


class MaxStatResult:
    """
    Result of the maximally selected log-rank statistic, from
    :func:`maxstat_logrank`.

    Attributes
    ----------
    cutpoint : float
        The estimated cut-point: the units with ``Z <= cutpoint`` form one
        group and the rest the other.
    statistic : float
        The largest standardised log-rank statistic ``|z|`` over the
        candidate cuts (R's ``M``).
    z : float
        The signed standardised statistic at the cut-point, for the group
        ``Z <= cutpoint``: positive when that group had more events than
        expected (a higher hazard).
    p_value : float
        The p-value of ``statistic``, corrected for having searched the
        candidate cuts (by ``method``).
    p_value_se : float
        The Monte Carlo standard error of a permutation p-value; 0 for
        the others.
    naive_p_value : float
        The p-value at the chosen cut as if it had been fixed in advance,
        ``2 * norm.sf(statistic)``. It is **not** a valid p-value: it
        ignores the search, and rejects a true null far more often than
        its level. It is reported only to show the size of the
        correction.
    method : str
        How ``p_value`` was found.
    cuts : numpy array
        The candidate cut-points.
    statistics : numpy array
        The standardised statistic ``|z|`` at each candidate cut.
    n_below, n_above : int
        The units with ``Z <= cutpoint`` and with ``Z > cutpoint``.
    minprop, maxprop : float
        The smallest and largest share of the units below a candidate
        cut.
    n_permutations : int or None
        The number of random relabellings of a permutation p-value; None
        otherwise.
    weighting : str
        The weighting of the event times.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import maxstat_logrank
    >>> rng = np.random.default_rng(1)
    >>> Z = rng.uniform(0, 10, 60)
    >>> x = rng.exponential(np.where(Z > 6, 1.0, 4.0))
    >>> res = maxstat_logrank(x, Z)
    >>> res.to_dict()["method"]
    'exact'
    """

    def __init__(
        self,
        cutpoint: float,
        statistic: float,
        z: float,
        p_value: float,
        p_value_se: float,
        naive_p_value: float,
        method: str,
        cuts: npt.NDArray,
        statistics: npt.NDArray,
        n_below: int,
        n_above: int,
        minprop: float,
        maxprop: float,
        n_permutations: int | None,
        weighting: str,
    ) -> None:
        self.cutpoint = cutpoint
        self.statistic = statistic
        self.z = z
        self.p_value = p_value
        self.p_value_se = p_value_se
        self.naive_p_value = naive_p_value
        self.method = method
        self.cuts = cuts
        self.statistics = statistics
        self.n_below = n_below
        self.n_above = n_above
        self.minprop = minprop
        self.maxprop = maxprop
        self.n_permutations = n_permutations
        self.weighting = weighting

    def __repr__(self) -> str:
        header = "Maximally Selected Log-Rank Statistic"
        out = (
            header
            + "\n"
            + "=" * len(header)
            + f"\nWeighting        : {self.weighting}"
            + f"\np-value method   : {self.method}"
            + f"\nCandidate cuts   : {self.cuts.size}"
            + f" (minprop {self.minprop:g}, maxprop {self.maxprop:g})"
            + f"\nCut-point        : {self.cutpoint:.6g}"
            + f" ({self.n_below} at or below, {self.n_above} above)"
            + f"\nStatistic        : {self.statistic:.6g}"
            + f"\np-value          : {self.p_value:.6g}"
        )
        if self.method == "permutation":
            out += f"\nMonte Carlo s.e. : {self.p_value_se:.2g}"
        out += (
            f"\nNaive p-value    : {self.naive_p_value:.6g}"
            + " (ignores the search; not valid)"
        )
        return out

    def to_dict(self) -> dict:
        """The result as strict JSON: the settings, the cut-point, the
        statistic and p-values, and every candidate cut with its statistic
        (a non-finite number is ``None``)."""
        return {
            "test": "maximally selected log-rank",
            "weighting": self.weighting,
            "method": self.method,
            "minprop": float(self.minprop),
            "maxprop": float(self.maxprop),
            "cutpoint": _finite(self.cutpoint),
            "statistic": _finite(self.statistic),
            "z": _finite(self.z),
            "p_value": _finite(self.p_value),
            "p_value_se": _finite(self.p_value_se),
            "naive_p_value": _finite(self.naive_p_value),
            "n_below": int(self.n_below),
            "n_above": int(self.n_above),
            "n_permutations": self.n_permutations,
            "cuts": [
                {"cut": _finite(v), "statistic": _finite(s)}
                for v, s in zip(self.cuts, self.statistics)
            ],
        }


def _lausen92(b: float, minprop: float, maxprop: float) -> float:
    """Lausen and Schumacher's (1992) approximation, from the Brownian
    bridge, of ``P(M >= b)`` (R's ``maxstat:::pLausen92``)."""
    from scipy.stats import norm

    if b < 1:
        return 1.0
    db = norm.pdf(b)
    ratio = maxprop * (1 - minprop) / ((1 - maxprop) * minprop)
    p = 4 * db / b + db * (b - 1 / b) * np.log(ratio)
    return float(min(max(p, 0.0), 1.0))


def _lausen94(b: float, N: float, m: npt.NDArray) -> float:
    """The improved Bonferroni bound of Lausen, Sauerbrei and Schumacher
    (1994), Worsley's (1982), over the candidate cuts that leave ``m``
    units below (R's ``maxstat:::pLausen94``)."""
    from scipy.stats import norm

    D = 0.0
    if m.size > 1:
        m1, m2 = m[:-1], m[1:]
        t = np.sqrt(1 - m1 * (N - m2) / ((N - m1) * m2))
        D = float(
            np.sum(
                1
                / np.pi
                * np.exp(-(b**2) / 2)
                * (t - (b**2 / 4 - 1) * t**3 / 6)
            )
        )
    return float(min(max(2 * norm.sf(b) + D, 0.0), 1.0))


def maxstat_logrank(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    minprop: float = 0.1,
    maxprop: float = 0.9,
    method: str = "exact",
    weighting: str = "log-rank",
    rho: float = 0,
    gamma: float = 0,
    n_permutations: int = 9999,
    random_state: Any = None,
    tl: npt.ArrayLike | float | None = None,
    w: npt.ArrayLike | float | None = None,
) -> MaxStatResult:
    r"""
    The maximally selected log-rank statistic: the cut-point of a
    continuous covariate that best separates survival, with a p-value
    corrected for having searched for it.

    Every distinct value of the covariate ``Z`` between the ``minprop``
    and ``maxprop`` quantiles is tried as a cut-point, splitting the units
    into ``Z <= cut`` and ``Z > cut``; each split is scored by its
    standardised two-group log-rank statistic, and the cut with the
    largest is reported (Lausen and Schumacher, 1992).

    **Read this before using it.** The p-value of an ordinary log-rank
    test *at the best cut* (``naive_p_value``) is not valid: with dozens
    of cuts tried, some split will look significant by chance alone, and
    at the 5% level the naive test rejects a true null hypothesis 30% to
    40% of the time or more. ``p_value`` allows for the search and is the
    one to report. Even so, turning a continuous covariate into two
    groups throws information away -- every unit on one side of the cut
    is treated as alike, however far from it -- and the cut found is
    unstable and tends not to replicate in new data. To learn whether and
    how the covariate affects survival, fit a regression on the covariate
    itself (:class:`~surpyval.CoxPH`, with a spline or other smooth term
    if the effect may not be linear). Use a cut-point only when a
    two-way decision rule is genuinely needed, and check it in new data.

    Parameters
    ----------
    x, c, n
        As for :func:`~surpyval.logrank`: the times, the censoring flags
        (0 observed, 1 right censored) and the counts (whole numbers: each
        unit is one row of the permutation argument).
    Z : array like
        The continuous (or ordered) covariate, one finite number per row.
    minprop, maxprop : float, optional
        The smallest and largest share of the units that may fall at or
        below a cut; defaults 0.1 and 0.9 (R's ``maxstat``). A cut leaving
        ``m`` units below is a candidate when ``max(1, floor(N minprop))
        <= m <= floor(N maxprop)``.
    method : str, optional
        How the p-value allows for the search:

        - "exact" (the default): the probability that the largest of the
          standardised statistics reaches the observed one, under their
          joint normal distribution (Hothorn and Lausen, 2003; R's
          ``pmethod = "exactGauss"``), computed by deterministic
          quadrature of the chain the statistics form.
        - "lausen94": Lausen, Sauerbrei and Schumacher's (1994) improved
          Bonferroni bound (R's ``"Lau94"``), an upper bound, so a little
          conservative.
        - "lausen92": Lausen and Schumacher's (1992) approximation from the
          Brownian bridge (R's ``"Lau92"``), which uses only ``minprop``
          and ``maxprop``; it is for large samples and is conservative
          when few distinct cuts are possible.
        - "permutation": the share of ``n_permutations`` random
          relabellings of the units whose largest statistic is at least
          the observed one, ``(1 + count) / (1 + n_permutations)`` (R's
          ``"condMC"``, which counts ``count / B``), with its Monte Carlo
          standard error.
    weighting, rho, gamma
        The weighting of the event times in the log-rank scores, as for
        :func:`~surpyval.logrank`; the default is the log-rank test.
    n_permutations : int, optional
        The number of random relabellings with ``method="permutation"``.
        Defaults to 9999.
    random_state : None, int or numpy.random.Generator, optional
        The seed of the relabellings, as everywhere in SurPyval.
    tl, w : None
        Refused: delayed entry and sampling weights make the units not
        exchangeable, which every p-value here relies on.

    Returns
    -------
    MaxStatResult
        With the ``cutpoint``, the ``statistic``, the corrected
        ``p_value``, the (invalid) ``naive_p_value`` for comparison, and
        each candidate cut with its statistic.

    Raises
    ------
    ValueError
        If ``Z`` is not one finite number per row, no candidate cut lies
        between ``minprop`` and ``maxprop``, the proportions are not
        ``0 <= minprop < maxprop <= 1``, ``method`` or ``weighting`` is
        unknown, a count is not a whole number, the data are not exact or
        right censored, or ``tl`` or ``w`` is given.

    Notes
    -----
    Each unit gets its log-rank score :math:`a_i` from the pooled data
    (:math:`\delta_i - \hat\Lambda(x_i)`, :math:`\hat\Lambda` the
    Nelson-Aalen estimate). For the cut leaving the :math:`m` units with
    the smallest covariate values below it, :math:`S_m = \sum_{i \le m}
    a_i` has, over the relabellings of the units, mean
    :math:`\frac{m}{N} \sum a_i` and variance :math:`\frac{m (N - m)}{N^2
    (N - 1)} \left(N \sum a_i^2 - (\sum a_i)^2\right)`, and the statistic
    is :math:`M = \max_m |S_m - E S_m| / \sqrt{V S_m}`. The standardised
    statistics at successive cuts are, asymptotically, a Gauss-Markov
    chain with correlation :math:`\sqrt{m_1 (N - m_2) / (m_2 (N -
    m_1))}`, so the "exact" p-value is a sum over the cuts of the
    probability that the chain first reaches :math:`M` there, carried
    forward by Gauss-Legendre quadrature. The scores are R's
    ``coin::logrank_trafo``; R's ``maxstat.test(..., smethod =
    "LogRank")`` uses ``exactRankTests::cscores``, which treats tied
    times as if they were ordered, so with tied times its statistic
    differs slightly (they agree without ties).

    Examples
    --------
    Patients with advanced lung cancer (R's ``survival::lung``) and their
    age:

    >>> from surpyval import maxstat_logrank
    >>> from surpyval.datasets import load_lung
    >>> df = load_lung()
    >>> res = maxstat_logrank(df["time"], df["age"], c=1 - df["status"])
    >>> print(res.cutpoint, round(res.statistic, 3))
    70.0 2.014
    >>> print(round(res.naive_p_value, 3), round(res.p_value, 3))
    0.044 0.291

    At the best cut (70 years) a fixed-cut log-rank test would claim
    p = 0.044, "significant" at 5%, but with the search over the 24
    candidate cuts allowed for there is no real evidence of a threshold
    in age (p = 0.29; R's ``maxstat`` gives 0.291).

    References
    ----------
    Lausen, B. and Schumacher, M. (1992), "Maximally selected rank
    statistics", Biometrics 48, 73-85.

    Lausen, B., Sauerbrei, W. and Schumacher, M. (1994), "Classification
    and regression trees (CART) used for the exploration of prognostic
    factors measured on different scales", in Computational Statistics,
    Physica, 483-496.

    Hothorn, T. and Lausen, B. (2003), "On the exact distribution of
    maximally selected rank statistics", Computational Statistics and
    Data Analysis 43, 121-137.

    Altman, D. G., Lausen, B., Sauerbrei, W. and Schumacher, M. (1994),
    "Dangers of using 'optimal' cutpoints in the evaluation of
    prognostic factors", Journal of the National Cancer Institute 86,
    829-835.
    """
    from scipy.stats import norm

    from surpyval.ml.forest.conditional_inference import max_chain_sf

    if tl is not None and np.isfinite(np.asarray(tl, dtype=float)).any():
        raise ValueError(
            "maxstat_logrank does not take delayed entry 'tl': its p-values "
            "need every unit to be exchangeable, which delayed entry "
            "breaks. Fit surpyval.CoxPH with tl on the covariate instead."
        )
    if w is not None:
        raise ValueError(
            "maxstat_logrank does not take sampling weights 'w': weighted "
            "units are not exchangeable. Fit surpyval.CoxPH with weights "
            "on the covariate instead."
        )
    if method not in METHODS:
        raise ValueError(
            "'method' must be one of {}; got {!r}".format(
                list(METHODS), method
            )
        )
    if weighting not in WEIGHTINGS:
        raise ValueError(f"'weighting' must be in {list(WEIGHTINGS)}")
    try:
        minprop = float(minprop)
        maxprop = float(maxprop)
    except (TypeError, ValueError):
        raise ValueError(
            "'minprop' and 'maxprop' must be numbers between 0 and 1"
        ) from None
    if not (0.0 <= minprop < maxprop <= 1.0):
        raise ValueError(
            "'minprop' and 'maxprop' must satisfy 0 <= minprop < maxprop "
            "<= 1; got {} and {}".format(minprop, maxprop)
        )
    if method == "permutation" and (
        isinstance(n_permutations, bool)
        or not float(n_permutations).is_integer()
        or n_permutations < 1
    ):
        raise ValueError(
            "'n_permutations' must be a positive whole number; got "
            "{!r}".format(n_permutations)
        )

    x = np.atleast_1d(np.asarray(x, dtype=float))
    if x.ndim != 1:
        raise ValueError(_NOT_RIGHT_CENSORED)
    try:
        Z = np.atleast_1d(np.asarray(Z, dtype=float))
    except (TypeError, ValueError):
        raise ValueError(
            "'Z' must be a numeric covariate, one number per row; for "
            "group labels use surpyval.logrank"
        ) from None
    if Z.shape != x.shape:
        raise ValueError(
            "'Z' must have one value for each observation; got {} for {} "
            "observations".format(Z.size, x.size)
        )
    if not np.all(np.isfinite(Z)):
        raise ValueError(
            "'Z' has missing or infinite values; drop those rows first"
        )
    c_arr = np.zeros(x.shape) if c is None else np.atleast_1d(c).astype(float)
    if c_arr.shape != x.shape:
        raise ValueError("'c' must have one entry for each observation")
    if ((c_arr != 0) & (c_arr != 1)).any():
        raise ValueError(_NOT_RIGHT_CENSORED)
    if n is not None:
        units = np.atleast_1d(np.asarray(n, dtype=float))
        if units.shape != x.shape:
            raise ValueError("'n' must have one entry for each observation")
        if np.any(units < 0) or not np.all(units == np.round(units)):
            raise ValueError(
                "'n' must hold whole numbers of units; each unit is "
                "relabelled on its own"
            )
        reps = units.astype(np.int64)
        x, Z, c_arr = (np.repeat(a, reps) for a in (x, Z, c_arr))

    N = x.size
    a = logrank_scores(x, c_arr, weighting, rho, gamma)
    order = np.argsort(Z, kind="stable")
    z_sorted = Z[order]
    a_sorted = a[order]
    # A cut after each last occurrence of a value: m units at or below.
    m_all = np.flatnonzero(z_sorted[:-1] != z_sorted[1:]) + 1
    lo = max(1, int(np.floor(N * minprop)))
    hi = int(np.floor(N * maxprop))
    m = m_all[(m_all >= lo) & (m_all <= hi)]
    if m.size == 0:
        raise ValueError(
            "No cut of 'Z' leaves between minprop={} and maxprop={} of the "
            "{} units below it; widen the range or check that 'Z' "
            "varies".format(minprop, maxprop, N)
        )
    total = a.sum()
    spread = N * np.sum(a**2) - total**2
    label = _weighting_label(weighting, rho, gamma)
    cuts = z_sorted[m - 1]
    if not spread > 1e-12 * max(1.0, N * np.sum(a**2)):
        # Every unit has the same score (no event, say): nothing to find.
        zeros = np.zeros(m.size)
        return MaxStatResult(
            float(cuts[0]), 0.0, 0.0, 1.0, 0.0, 1.0, method, cuts, zeros,
            int(m[0]), int(N - m[0]), minprop, maxprop,
            int(n_permutations) if method == "permutation" else None,
            label,
        )  # fmt: skip
    E = m / N * total
    V = m * (N - m) / (N**2 * (N - 1)) * spread
    signed = (np.cumsum(a_sorted)[m - 1] - E) / np.sqrt(V)
    stats = np.abs(signed)
    best = int(np.argmax(stats))
    M = float(stats[best])
    naive = float(min(1.0, 2 * norm.sf(M)))
    se = 0.0
    n_perm = None
    if method == "exact":
        m1, m2 = m[:-1].astype(float), m[1:].astype(float)
        rho_chain = np.sqrt(m1 * (N - m2) / (m2 * (N - m1)))
        p_value = float(max_chain_sf(M * M, rho_chain, 1))
    elif method == "lausen94":
        p_value = _lausen94(M, float(N), m.astype(float))
    elif method == "lausen92":
        p_value = _lausen92(M, minprop, maxprop)
    else:
        from surpyval.utils.rng import as_generator

        rng = as_generator(random_state)
        n_perm = int(n_permutations)
        allowance = 1e-9 * max(1.0, M)
        count = 0
        chunk = max(1, min(n_perm, 2_000_000 // max(1, N)))
        done = 0
        sd = np.sqrt(V)
        while done < n_perm:
            B = min(chunk, n_perm - done)
            shuffled = rng.permuted(np.tile(a_sorted, (B, 1)), axis=1)
            S = np.cumsum(shuffled, axis=1)[:, m - 1]
            Mb = np.max(np.abs((S - E) / sd), axis=1)
            count += int(np.sum(Mb >= M - allowance))
            done += B
        p_value = (1.0 + count) / (1.0 + n_perm)
        se = float(np.sqrt(p_value * (1.0 - p_value) / n_perm))
    return MaxStatResult(
        float(cuts[best]),
        M,
        float(signed[best]),
        p_value,
        se,
        naive,
        method,
        cuts,
        stats,
        int(m[best]),
        int(N - m[best]),
        minprop,
        maxprop,
        n_perm,
        label,
    )
