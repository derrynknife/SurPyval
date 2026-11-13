"""The permutation log-rank test: an exact or Monte Carlo p-value from
relabelling the units, for small samples."""

from __future__ import annotations

from itertools import combinations
from math import comb, prod
from typing import Any

import numpy as np
import numpy.typing as npt

from .logrank import (
    _NOT_RIGHT_CENSORED,
    _checked_inputs,
    _plain,
    _risk_table,
    _time_weights,
    _weighting_label,
)

#: The ``method`` values of :func:`permutation_logrank`.
METHODS = ("auto", "exact", "monte-carlo")


def _finite(v: Any) -> Any:
    v = float(v)
    return v if np.isfinite(v) else None


class PermutationLogRankResult:
    """
    Result of the permutation log-rank test, from
    :func:`permutation_logrank`.

    Attributes
    ----------
    statistic : float
        The chi-squared statistic: the groups' (weighted) observed minus
        expected events compared with their covariance over the
        relabellings of the units.
    dof : int
        The number of groups less one; 0 when there is nothing to compare.
    z : float or None
        For two groups, the standardised observed-minus-expected events of
        the first group (in sorted label order); positive when it had more
        events than expected. None for more groups.
    p_value : float
        The permutation p-value: the share of the relabellings whose
        statistic is at least the observed one (exact), or
        ``(1 + count) / (1 + n_permutations)`` of random ones (Monte
        Carlo).
    p_value_se : float
        The Monte Carlo standard error of ``p_value``,
        ``sqrt(p (1 - p) / n_permutations)``; 0 for an exact p-value.
    p_value_asymptotic : float
        The p-value of the same statistic from the chi-squared
        distribution, for comparison.
    method : str
        "exact" or "monte-carlo".
    n_permutations : int
        The number of relabellings behind the p-value: all of them
        (exact) or the number drawn (Monte Carlo).
    U : dict
        Each group's (weighted) observed minus expected events, keyed by
        group label.
    weighting : str
        The weighting of the event times.
    strata : int or None
        The number of strata (units are relabelled within each); None for
        an unstratified test.

    Examples
    --------
    >>> from surpyval import permutation_logrank
    >>> res = permutation_logrank([3, 5, 7, 2, 3, 4], [0, 0, 0, 1, 1, 1])
    >>> res.method, res.n_permutations
    ('exact', 20)
    """

    def __init__(
        self,
        statistic: float,
        dof: int,
        z: float | None,
        p_value: float,
        p_value_se: float,
        p_value_asymptotic: float,
        method: str,
        n_permutations: int,
        U: dict,
        weighting: str,
        strata: int | None = None,
    ) -> None:
        self.statistic = statistic
        self.dof = dof
        self.z = z
        self.p_value = p_value
        self.p_value_se = p_value_se
        self.p_value_asymptotic = p_value_asymptotic
        self.method = method
        self.n_permutations = n_permutations
        self.U = U
        self.weighting = weighting
        self.strata = strata

    def __repr__(self) -> str:
        header = "Permutation Log-Rank Test"
        if self.strata:
            header = "Stratified " + header
        out = (
            header
            + "\n"
            + "=" * len(header)
            + f"\nWeighting        : {self.weighting}"
            + f"\np-value method   : {self.method}"
            + f"\nPermutations     : {self.n_permutations}"
        )
        if self.strata is not None:
            out += f"\nStrata           : {self.strata}"
        out += (
            f"\nStatistic        : {self.statistic:.6g}"
            + f"\nDoF              : {self.dof}"
            + f"\np-value          : {self.p_value:.6g}"
        )
        if self.method == "monte-carlo":
            out += f"\nMonte Carlo s.e. : {self.p_value_se:.2g}"
        out += f"\nAsymptotic p     : {self.p_value_asymptotic:.6g}"
        return out

    def to_dict(self) -> dict:
        """The result as strict JSON: the settings, the statistic, its
        degrees of freedom, the p-values and each group's observed minus
        expected events (a non-finite number is ``None``)."""
        return {
            "test": "permutation log-rank",
            "weighting": self.weighting,
            "method": self.method,
            "n_permutations": int(self.n_permutations),
            "strata": self.strata,
            "statistic": _finite(self.statistic),
            "dof": int(self.dof),
            "z": None if self.z is None else _finite(self.z),
            "p_value": _finite(self.p_value),
            "p_value_se": _finite(self.p_value_se),
            "p_value_asymptotic": _finite(self.p_value_asymptotic),
            "U": [
                {"group": _plain(g), "observed_minus_expected": _finite(u)}
                for g, u in self.U.items()
            ],
        }


def logrank_scores(
    x: npt.NDArray,
    c: npt.NDArray,
    weighting: str = "log-rank",
    rho: float = 0.0,
    gamma: float = 0.0,
) -> npt.NDArray:
    """Each unit's (weighted) log-rank score from the pooled data of one
    stratum: ``a_i = delta_i w(x_i) - sum_{t_j <= x_i} w(t_j) d_j / r_j``.

    A group's sum of the scores is its (weighted) observed minus expected
    events of :func:`~surpyval.logrank`, whatever the labels, so the
    permutation distribution of the group sums is that of the test's
    numerator. ``x`` and ``c`` have one entry per unit."""
    one = np.zeros(1, dtype=int)
    times, r_gt, d_gt = _risk_table(
        x, np.zeros(x.size, dtype=int), c, None, one
    )
    if times.size == 0:
        return np.zeros(x.size)
    r_t, d_t = r_gt[0], d_gt[0]
    w = _time_weights(weighting, rho, gamma, r_t, d_t)
    hazard = np.concatenate([[0.0], np.cumsum(w * d_t / r_t)])
    upto = np.searchsorted(times, x, "right")
    event = c == 0
    jump = np.zeros(x.size)
    jump[event] = w[np.searchsorted(times, x[event])]
    return jump - hazard[upto]


def _assignments(size: int, counts: list[int]) -> npt.NDArray:
    """Every way to give ``size`` units the group indices ``0..k-1`` with
    ``counts[g]`` units in group ``g``, as rows of a matrix."""
    rows: list[npt.NDArray] = []
    label = np.empty(size, dtype=np.int64)

    def fill(free: tuple, g: int) -> None:
        if g == len(counts) - 1:
            label[list(free)] = g
            rows.append(label.copy())
            return
        for chosen in combinations(free, counts[g]):
            label[list(chosen)] = g
            rest = tuple(i for i in free if i not in set(chosen))
            fill(rest, g + 1)

    fill(tuple(range(size)), 0)
    return np.array(rows)


def permutation_logrank(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    weighting: str = "log-rank",
    rho: float = 0,
    gamma: float = 0,
    strata: npt.ArrayLike | None = None,
    t0: float | None = None,
    method: str = "auto",
    n_permutations: int = 9999,
    max_exact: int = 100000,
    random_state: Any = None,
    tl: npt.ArrayLike | float | None = None,
    w: npt.ArrayLike | float | None = None,
) -> PermutationLogRankResult:
    r"""
    The k-sample (weighted) log-rank test with a permutation p-value:
    exact for small samples, Monte Carlo otherwise.

    The p-value of :func:`~surpyval.logrank` comes from the chi-squared
    distribution, which is an approximation that can be poor with a
    handful of units or events. This test gets the p-value instead from
    the data themselves: if the groups truly share one survival
    distribution (and are censored alike), the group labels are just
    tags, and every way of shuffling them among the units is as likely as
    the one observed. The p-value is the share of the shuffles that give
    a statistic at least as large as the observed one. With few units all
    the shuffles are counted (an exact p-value); with more, a random
    sample of them (a Monte Carlo p-value, with its standard error).

    **When it is valid, and when it is not.** The permutation argument
    needs the units to be exchangeable under the null hypothesis: the
    same survival distribution *and the same censoring distribution* in
    every group. When the groups are censored differently -- one followed
    for longer, or one losing more units to follow-up -- shuffling the
    labels also shuffles the censoring pattern, and the test can reject a
    true null too often, especially with unequal group sizes (Heinze,
    Gnant and Schemper, 2003; the same caveat applies to the
    permutation tests of Heller and Venkatraman, 1996). Then use
    :func:`~surpyval.logrank`, whose asymptotic p-value does not assume
    equal censoring, or, with few events, report both. The same need for
    exchangeability is why delayed entry (``tl``) and sampling weights
    (``w``) are refused.

    Parameters
    ----------
    x, Z, c, n
        As for :func:`~surpyval.logrank`: the times, the group labels, the
        censoring flags (0 observed, 1 right censored) and the counts. A
        count must be a whole number: each unit is relabelled on its own.
    weighting, rho, gamma
        The weighting of the event times, as for
        :func:`~surpyval.logrank`; the weights come from the pooled data,
        so they are the same for every relabelling.
    strata : array like, optional
        Stratum labels: the units are relabelled within each stratum only,
        and the scores and statistic are those of the stratified test.
    t0 : scalar, optional
        Compare the groups up to ``t0`` only, as for
        :func:`~surpyval.logrank`.
    method : str, optional
        "auto" (the default): exact when the number of distinct
        relabellings is at most ``max_exact``, Monte Carlo otherwise.
        "exact": count every relabelling (refused above ``max_exact``).
        "monte-carlo": ``n_permutations`` random relabellings.
    n_permutations : int, optional
        The number of random relabellings of a Monte Carlo p-value.
        Defaults to 9999; the p-value's standard error is about
        ``sqrt(p (1 - p) / n_permutations)`` (0.002 at p = 0.05).
    max_exact : int, optional
        The most relabellings an exact p-value enumerates. Defaults to
        100000 (two groups of 9 units have 48620).
    random_state : None, int or numpy.random.Generator, optional
        The seed of the random relabellings, as everywhere in SurPyval:
        None draws from numpy's global random number generator, an int or
        a Generator gives a stream of its own.
    tl : None
        Refused: delayed entry makes the units not exchangeable. Use
        :func:`~surpyval.logrank` with ``tl``.
    w : None
        Refused: sampling weights make the units not exchangeable. Use
        :func:`~surpyval.logrank` with ``w``.

    Returns
    -------
    PermutationLogRankResult
        With the ``statistic``, ``dof``, the permutation ``p_value``, its
        Monte Carlo standard error ``p_value_se``, the asymptotic p-value
        of the same statistic and each group's observed minus expected
        events ``U``.

    Raises
    ------
    ValueError
        If ``tl`` or ``w`` is given, ``method`` is unknown, an exact
        p-value would need more than ``max_exact`` relabellings, a count
        is not a whole number, ``n_permutations`` or ``max_exact`` is not
        a positive whole number, or for any reason
        :func:`~surpyval.logrank` would raise.

    Notes
    -----
    Each unit gets the score :math:`a_i = \delta_i w(x_i) - \sum_{t_j \le
    x_i} w(t_j) d_j / r_j` from the pooled data of its stratum, so that a
    group's sum of scores :math:`U_g` is its (weighted) observed minus
    expected events. Over the relabellings :math:`U` has mean
    :math:`N_g \bar a` and covariance :math:`\frac{\sum_i (a_i - \bar
    a)^2}{N - 1} (N_g \delta_{gh} - N_g N_h / N)` (summed over the
    strata), and the statistic is the quadratic form of :math:`U` minus
    its mean in the inverse of that covariance (any k - 1 groups). It
    differs a little from :func:`~surpyval.logrank`'s, whose variance is
    the hypergeometric one, and it is R's ``coin::logrank_test`` (with its
    default ``ties.method = "mid-ranks"``) whose ``distribution = "exact"``
    and ``approximate(nresample = B)`` give the exact and Monte Carlo
    p-values; ``coin`` counts ``count / B`` where this test counts
    ``(1 + count) / (1 + B)``, which keeps the Monte Carlo test's size at
    most its level.
    With ``strata`` the scores are formed within each stratum, as the
    stratified log-rank test forms its risk sets; coin's ``logrank_test(y
    ~ x | block)`` forms them from the pooled data and permutes only
    within the blocks, so it matches this test given the scores per block
    (``independence_test`` with ``trafo(..., block = )``).

    Examples
    --------
    The AML maintenance data (R's ``survival::aml``): 11 maintained and
    12 not, weeks to relapse.

    >>> from surpyval import logrank, permutation_logrank
    >>> x = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161,
    ...      5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
    >>> c = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
    ...      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    >>> Z = [1] * 11 + [2] * 12
    >>> res = permutation_logrank(x, Z, c=c, method="exact",
    ...                           max_exact=2000000)
    >>> print(res.n_permutations, round(res.p_value, 4))
    1352078 0.0647

    All 1,352,078 ways of choosing which 11 of the 23 patients were
    maintained are counted; R's ``coin::logrank_test(...,
    distribution = "exact")`` gives the same 0.0647. The censoring is
    similar in the two groups (4 and 1 units censored), so the exact
    p-value can be trusted; here it is close to the chi-squared one:

    >>> round(logrank(x, Z, c=c).p_value, 4)
    0.0653

    References
    ----------
    Heller, G. and Venkatraman, E. S. (1996), "Resampling procedures to
    compare two survival distributions in the presence of right-censored
    data", Biometrics 52, 1204-1213.

    Heinze, G., Gnant, M. and Schemper, M. (2003), "Exact log-rank tests
    for unequal follow-up", Biometrics 59, 1151-1157.

    Hothorn, T., Hornik, K., van de Wiel, M. A. and Zeileis, A. (2006), "A
    Lego system for conditional inference", The American Statistician 60,
    257-263.
    """
    from scipy.stats import chi2

    _check_options(tl, w, method, n_permutations, max_exact)
    n_permutations = int(n_permutations)
    max_exact = int(max_exact)
    blocks, groups, n_strata = _blocks(
        x, Z, c, n, weighting, rho, gamma, strata, t0
    )
    k = groups.size

    U = np.zeros(k)
    mean = np.zeros(k)
    cov = np.zeros((k, k))
    for a, gs in blocks:
        N_s = a.size
        U += np.bincount(gs, weights=a, minlength=k)
        if N_s < 2:
            continue
        sizes = np.bincount(gs, minlength=k).astype(float)
        mean += sizes * a.mean()
        spread = np.sum((a - a.mean()) ** 2) / (N_s - 1)
        cov += spread * (np.diag(sizes) - np.outer(sizes, sizes) / N_s)

    label = _weighting_label(weighting, rho, gamma)
    U_dict = {_plain(gr): float(u) for gr, u in zip(groups, U)}
    eigen = np.linalg.eigvalsh(cov)
    rank = int(np.sum(eigen > 1e-10 * max(float(eigen.max()), 1e-300)))
    if rank == 0:
        # Every unit has the same score: nothing to compare.
        return PermutationLogRankResult(
            0.0, 0, None if k > 2 else 0.0, 1.0, 0.0, 1.0,
            "exact", 1, U_dict, label, strata=n_strata,
        )  # fmt: skip
    inverse = np.linalg.pinv(cov)

    def statistic_of(T: npt.NDArray) -> npt.NDArray:
        d = T - mean
        return np.einsum("...i,ij,...j->...", d, inverse, d)

    observed = float(statistic_of(U))
    z = None
    if k == 2:
        z = float((U[0] - mean[0]) / np.sqrt(cov[0, 0]))
    p_asymptotic = float(chi2.sf(observed, rank))
    # Each relabelling's statistic is compared with the observed one less
    # a rounding allowance, so ties with it count.
    threshold = observed - 1e-9 * max(1.0, abs(observed))

    total = prod(
        prod(comb(int(sz[j:].sum()), int(sz[j])) for j in range(k))
        for sz in (np.bincount(gs, minlength=k) for _, gs in blocks)
    )
    if method == "exact" and total > max_exact:
        raise ValueError(
            "An exact p-value needs all {} relabellings, more than "
            "max_exact={}; raise 'max_exact' (the time and memory grow "
            "with it) or use method='monte-carlo'".format(total, max_exact)
        )
    if method == "exact" or (method == "auto" and total <= max_exact):
        count, seen = _exact_count(blocks, k, statistic_of, threshold)
        return PermutationLogRankResult(
            observed, rank, z, count / seen, 0.0, p_asymptotic, "exact",
            seen, U_dict, label, strata=n_strata,
        )  # fmt: skip

    from surpyval.utils.rng import as_generator

    rng = as_generator(random_state)
    count = 0
    size = sum(a.size for a, _ in blocks)
    chunk = max(1, min(n_permutations, 2_000_000 // max(1, size)))
    done = 0
    while done < n_permutations:
        B = min(chunk, n_permutations - done)
        T = np.zeros((B, k))
        for a, gs in blocks:
            T += _group_sums(a, rng.permuted(np.tile(gs, (B, 1)), axis=1), k)
        count += int(np.sum(statistic_of(T) >= threshold))
        done += B
    p_value = (1.0 + count) / (1.0 + n_permutations)
    se = float(np.sqrt(p_value * (1.0 - p_value) / n_permutations))
    return PermutationLogRankResult(
        observed, rank, z, p_value, se, p_asymptotic, "monte-carlo",
        n_permutations, U_dict, label, strata=n_strata,
    )  # fmt: skip


def _check_options(
    tl: Any, w: Any, method: str, n_permutations: Any, max_exact: Any
) -> None:
    """The refusals of :func:`permutation_logrank` that need no data."""
    if tl is not None and np.isfinite(np.asarray(tl, dtype=float)).any():
        raise ValueError(
            "permutation_logrank does not take delayed entry 'tl': the "
            "permutation test needs every unit to be exchangeable, which "
            "delayed entry breaks. Use surpyval.logrank(x, Z, c, tl=...)."
        )
    if w is not None:
        raise ValueError(
            "permutation_logrank does not take sampling weights 'w': "
            "weighted units are not exchangeable. Use "
            "surpyval.logrank(x, Z, c, w=...)."
        )
    if method not in METHODS:
        raise ValueError(
            "'method' must be one of {}; got {!r}".format(
                list(METHODS), method
            )
        )
    for name, value in (
        ("n_permutations", n_permutations),
        ("max_exact", max_exact),
    ):
        if (
            isinstance(value, bool)
            or not float(value).is_integer()
            or value < 1
        ):
            raise ValueError(
                "'{}' must be a positive whole number; got {!r}".format(
                    name, value
                )
            )


def _blocks(
    x: Any,
    Z: Any,
    c: Any,
    n: Any,
    weighting: str,
    rho: float,
    gamma: float,
    strata: Any,
    t0: Any,
) -> tuple[list, npt.NDArray, int | None]:
    """The checked data as one ``(scores, group index)`` pair of arrays
    per stratum, one entry per unit; the sorted group labels; and the
    number of strata (None if unstratified)."""
    x, Z, c_arr, n_arr, _, strata, _, groups = _checked_inputs(
        x, Z, c, n, weighting, strata, None, t0, None, True, 4
    )
    c_arr = np.zeros(x.shape) if c_arr is None else c_arr
    if ((c_arr != 0) & (c_arr != 1)).any():
        raise ValueError(_NOT_RIGHT_CENSORED)
    if n_arr is not None:
        units = np.asarray(n_arr, dtype=float)
        if np.any(units < 0) or not np.all(units == np.round(units)):
            raise ValueError(
                "'n' must hold whole numbers of units for a permutation "
                "test, each unit relabelled on its own"
            )
        reps = units.astype(np.int64)
        x, Z, c_arr = (np.repeat(a, reps) for a in (x, Z, c_arr))
        if strata is not None:
            strata = np.repeat(strata, reps)
    g = np.searchsorted(groups, Z)
    x = np.asarray(x, dtype=float)
    labels = [None] if strata is None else list(np.unique(strata))
    blocks = []
    for s in labels:
        rows = np.arange(x.size) if s is None else np.flatnonzero(strata == s)
        a = logrank_scores(x[rows], c_arr[rows], weighting, rho, gamma)
        blocks.append((a, g[rows]))
    return blocks, groups, None if strata is None else len(labels)


def _group_sums(a: npt.NDArray, labels: npt.NDArray, k: int) -> npt.NDArray:
    """Each relabelling's group sums of the scores ``a``: ``labels`` is
    ``(B, N)``, the result ``(B, k)``."""
    return np.stack([(labels == j) @ a for j in range(k)], axis=-1)


def _exact_count(
    blocks: list, k: int, statistic_of: Any, threshold: float
) -> tuple[int, int]:
    """The number of relabellings whose statistic reaches ``threshold``,
    and the number of relabellings, counting every one of them."""
    if len(blocks) == 1 and k == 2:
        # Two groups: the first group's members run over every subset of
        # its size, streamed in chunks to bound the memory.
        a, gs = blocks[0]
        first = int(np.sum(gs == 0))
        subsets = combinations(range(a.size), first)
        count = seen = 0
        while True:
            flat = np.fromiter(
                (i for _, s_ in zip(range(65536), subsets) for i in s_),
                dtype=np.int64,
            )
            if flat.size == 0 and first > 0:
                break
            idx = flat.reshape(-1, first) if first else flat.reshape(1, 0)
            T0 = a[idx].sum(axis=1)
            T = np.column_stack([T0, a.sum() - T0])
            count += int(np.sum(statistic_of(T) >= threshold))
            seen += T.shape[0]
            if first == 0 or idx.shape[0] < 65536:
                break
        return count, seen
    # Each block's group sums over all its relabellings, combined across
    # the strata.
    T = np.zeros((1, k))
    for a, gs in blocks:
        sz = np.bincount(gs, minlength=k)
        block = _group_sums(a, _assignments(a.size, [int(v) for v in sz]), k)
        T = (T[:, None, :] + block[None, :, :]).reshape(-1, k)
    return int(np.sum(statistic_of(T) >= threshold)), int(T.shape[0])
