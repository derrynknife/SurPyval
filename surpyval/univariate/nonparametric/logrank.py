from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval.univariate.nonparametric.kaplan_meier import kaplan_meier
from surpyval.utils import xcnt_handler
from surpyval.utils.linalg import safe_quadform

if TYPE_CHECKING:
    import pandas as pd


class LogRankResult:
    """
    Result of a (weighted) log-rank test.

    Attributes
    ----------

    statistic : float
        The chi-squared test statistic.
    dof : int
        The degrees of freedom: the number of groups with a positive
        expected number of events, minus one (R's ``survdiff``
        convention).
    p_value : float
        The p-value of the test.
    weighting : str
        The weighting used for the test; for the Fleming-Harrington
        weights it includes ``rho`` and ``gamma``, e.g.
        ``'fleming-harrington(rho=1, gamma=0)'``.
    strata : int or None
        The number of strata of a stratified test; None for an
        unstratified one.

    Examples
    --------
    :code:`logrank` returns one; here comparing two groups of five:

    >>> import numpy as np
    >>> from surpyval import logrank
    >>> x = np.array([3, 5, 7, 8, 10, 2, 3, 4, 4, 6])
    >>> Z = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    >>> result = logrank(x, Z)
    >>> result
    Log-Rank Test
    =============
    Weighting        : log-rank
    Statistic        : 3.69204
    DoF              : 1
    p-value          : 0.0546728
    >>> round(result.p_value, 4)
    0.0547
    """

    def __init__(
        self,
        statistic: float,
        dof: int,
        p_value: float,
        weighting: str,
        strata: int | None = None,
    ) -> None:
        self.statistic = statistic
        self.dof = dof
        self.p_value = p_value
        self.weighting = weighting
        self.strata = strata

    def __repr__(self) -> str:
        header = "Stratified Log-Rank Test" if self.strata else "Log-Rank Test"
        out = (
            header
            + "\n"
            + "=" * len(header)
            + f"\nWeighting        : {self.weighting}"
        )
        if self.strata is not None:
            out += f"\nStrata           : {self.strata}"
        out += (
            f"\nStatistic        : {self.statistic:.6g}"
            + f"\nDoF              : {self.dof}"
            + f"\np-value          : {self.p_value:.6g}"
        )
        return out


def _logrank_z_v(
    x: npt.NDArray,
    Z: npt.NDArray,
    c: npt.NDArray | None,
    n: npt.NDArray | None,
    groups: npt.NDArray,
    weighting: str,
    rho: float,
    gamma: float,
    tl: npt.NDArray | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """Per-stratum (or whole-sample) weighted log-rank ``(z, V, E)``.

    Returns the length-``k`` vector of weighted observed-minus-expected
    counts, its ``k x k`` covariance and the length-``k`` vector of
    (unweighted) expected event counts, using the fixed group order
    ``groups`` so contributions from different strata are aligned and can be
    summed. A group absent from this stratum simply contributes zeros.

    With ``tl`` (entry times) a row is at risk at ``t`` from its entry,
    ``tl < t <= x``, the risk sets of ``CoxPH`` (``cox_at_risk_mask``)
    and of ``KaplanMeier`` with ``tl``.
    """
    k = groups.size
    x_g, c_g, n_g, tl_g = [], [], [], []
    for g in groups:
        mask = Z == g
        x_i = np.atleast_1d(x)[mask]
        c_i = None if c is None else np.atleast_1d(c)[mask]
        n_i = None if n is None else np.atleast_1d(n)[mask]
        tl_i = None if tl is None else np.atleast_1d(tl)[mask]
        if x_i.size == 0:
            x_g.append(np.array([]))
            c_g.append(np.array([]))
            n_g.append(np.array([]))
            tl_g.append(np.array([]))
            continue
        x_i, c_i, n_i, t_i = xcnt_handler(x=x_i, c=c_i, n=n_i, tl=tl_i)
        if ((c_i != 0) & (c_i != 1)).any():
            raise ValueError(
                "Log-rank test can only be used with observed and "
                + "right censored data"
            )
        x_g.append(x_i)
        c_g.append(c_i)
        n_g.append(n_i)
        tl_g.append(t_i[:, 0])

    event_pool = [x_i[c_i == 0] for x_i, c_i in zip(x_g, c_g) if x_i.size > 0]
    event_times = (
        np.unique(np.concatenate(event_pool)) if event_pool else np.array([])
    )
    m = event_times.size
    if m == 0:
        return np.zeros(k), np.zeros((k, k)), np.zeros(k)

    # Counts at risk and events per group and event time. These were
    # dense ``x_i[:, None] >= event_times`` arrays, O(n * m) in time and
    # memory: 13 s and 2 GB for 3e4 rows, and out of memory at 1e5 (#515).
    # The at-risk count is a suffix sum of the counts over the sorted
    # times, read off with ``searchsorted``, and the events are binned
    # with ``bincount``. The counts are whole numbers, so both sums are
    # exact and the arrays are identical to the dense ones.
    r_gt = np.zeros((k, m))
    d_gt = np.zeros((k, m))
    for j, (x_i, c_i, n_i, tl_i) in enumerate(zip(x_g, c_g, n_g, tl_g)):
        if x_i.size == 0:
            continue
        order = np.argsort(x_i, kind="stable")
        x_sorted = x_i[order]
        # at_or_after[i]: the count of the rows from sorted position i on.
        at_or_after = np.concatenate([np.cumsum(n_i[order][::-1])[::-1], [0]])
        r_gt[j] = at_or_after[np.searchsorted(x_sorted, event_times, "left")]
        if tl is not None:
            # Less the rows not yet entered, tl >= t (each of which has
            # x > tl >= t, so it was counted above).
            order = np.argsort(tl_i, kind="stable")
            tl_sorted = tl_i[order]
            entered_later = np.concatenate(
                [np.cumsum(n_i[order][::-1])[::-1], [0]]
            )
            r_gt[j] -= entered_later[
                np.searchsorted(tl_sorted, event_times, "left")
            ]
        events = c_i == 0
        d_gt[j] = np.bincount(
            np.searchsorted(event_times, x_i[events]),
            weights=n_i[events],
            minlength=m,
        )

    r_t = r_gt.sum(axis=0)
    d_t = d_gt.sum(axis=0)

    if weighting == "log-rank":
        w = np.ones(m)
    elif weighting == "gehan":
        w = r_t
    elif weighting == "tarone-ware":
        w = np.sqrt(r_t)
    elif weighting in ("peto-peto", "modified-peto-peto"):
        # Peto and Peto's survival estimate at each event time, the event
        # included: prod over the event times up to t of 1 - d / (r + 1)
        # (Klein and Moeschberger, 2003, section 7.3), formed within the
        # stratum. Andersen et al.'s modification scales it by r / (r + 1).
        w = np.cumprod(1.0 - d_t / (r_t + 1.0))
        if weighting == "modified-peto-peto":
            w = w * r_t / (r_t + 1.0)
    else:
        # Pooled left-continuous Kaplan-Meier, i.e. the value of the
        # estimate just prior to each event time. Weights are formed
        # within the stratum (standard for a stratified weighted test).
        S = kaplan_meier(r_t, d_t)
        S_prev = np.hstack([[1.0], S[:-1]])
        w = S_prev**rho * (1 - S_prev) ** gamma

    with np.errstate(all="ignore"):
        expected = np.where(r_t > 0, d_t * r_gt / r_t, 0.0)
    z = (w * (d_gt - expected)).sum(axis=1)

    with np.errstate(all="ignore"):
        hyper = np.where(r_t > 1, d_t * (r_t - d_t) / (r_t - 1), 0.0)
        prop = np.where(r_t > 0, r_gt / r_t, 0.0)
    # V[a, b] = sum_t w**2 * hyper * prop[a] * (delta_ab - prop[b]), one row
    # at a time: the same products in the same order as the double loop
    # over (a, b) this replaced, so V is bit-identical, with O(k * m)
    # memory.
    weighted = w**2 * hyper * prop
    V = np.empty((k, k))
    for a in range(k):
        delta = 0.0 - prop
        delta[a] = 1.0 - prop[a]
        V[a] = (weighted[a] * delta).sum(axis=1)

    return z, V, expected.sum(axis=1)


def logrank(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    weighting: str = "log-rank",
    rho: float = 0,
    gamma: float = 0,
    strata: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | float | None = None,
    t0: float | None = None,
) -> LogRankResult:
    r"""
    The k-sample (weighted) log-rank test for the equality of survival
    distributions of right censored data.

    At each distinct event time the observed number of events in each
    group is compared with the number expected under the null
    hypothesis that all groups share the same survival distribution.
    The weighted sums of these differences form a chi-squared statistic
    with k - 1 degrees of freedom.

    Parameters
    ----------

    x : array like
        Array of observations of the random variables.
    Z : array like
        Array of group labels for each observation. Any hashable values
        other than NaN or None can be used. The test has one degree of
        freedom fewer than the number of groups with a positive expected
        number of events (as R's ``survdiff``): a group that is never at
        risk at an event time carries no information and is left out. If
        fewer than two groups have expected events there is nothing to
        compare, and the statistic is 0 with ``dof`` 0 and a p-value of 1.
    c : array like, optional
        Array of censoring flags. 0 is observed and 1 is right
        censored. Left or interval censored data cannot be used with
        the log-rank test. If not provided assumes all values are
        observed.
    n : array like, optional
        Array of counts for each x. If :code:`None` assumes each
        observation is 1.
    weighting : str, optional
        The weighting to use at each event time. One of:

        - "log-rank": weight 1 (the standard log-rank test; sensitive
          to proportional hazards alternatives),
        - "gehan": weight r (a.k.a. Gehan-Breslow-Wilcoxon; emphasises
          early differences),
        - "tarone-ware": weight sqrt(r),
        - "fleming-harrington": weight S(t-)**rho * (1 - S(t-))**gamma
          where S is the pooled Kaplan-Meier estimate; rho > 0
          emphasises early differences and gamma > 0 late differences.
        - "peto-peto": weight S~(t), Peto and Peto's estimate of the
          pooled survival at the event time, the product of
          1 - d / (r + 1) over the event times up to and including t
          (Klein and Moeschberger, 2003, section 7.3). Like Gehan's it
          emphasises early differences, but it depends less on the
          pattern of censoring than Gehan's does.
        - "modified-peto-peto": weight S~(t) * r / (r + 1) (Andersen et
          al., 1982).

        Defaults to "log-rank".
    rho, gamma : scalar, optional
        The parameters of the Fleming-Harrington weighting. Only used
        when weighting is "fleming-harrington". Defaults to 0, 0 (which
        is identical to the log-rank weighting).
    strata : array like, optional
        Array of stratum labels, one per observation. When supplied the
        test is *stratified*: the observed-minus-expected numerators and
        their variances are accumulated *within* each stratum (risk sets
        never cross a stratum boundary) and summed before forming the
        statistic. This removes a nuisance factor -- one whose baseline
        hazard differs across strata -- from the comparison, so groups are
        only ever compared against others in the same stratum. The degrees
        of freedom are counted as for an unstratified test, from the
        expected events summed over the strata. NaN or None labels are
        refused.
    tl : array like or scalar, optional
        The entry (left truncation) time of each observation, or one
        time for all of them. A unit is at risk from its entry, at the
        event times ``t`` with ``tl < t <= x`` (the risk sets of
        ``KaplanMeier`` and ``CoxPH`` with ``tl``), so a unit that came
        under observation part-way through its life is not counted
        before it was seen. Without ``tl`` every unit is at risk from
        time 0. The test is then the score test of a ``CoxPH`` fit of
        the group with ``tl`` (exactly so without tied event times; R's
        ``survdiff`` takes right censored data only).
    t0 : scalar, optional
        Compare the groups up to ``t0`` only: an event after ``t0`` is
        treated as a unit still running at ``t0`` (right censored there),
        and a unit that entered at or after ``t0`` (``tl >= t0``) is left
        out, so only the event times up to and including ``t0`` count.
        Use it when the question is about a fixed follow-up window, or
        when the curves are known to cross later. Without ``t0`` the
        whole follow-up is compared.

    Returns
    -------

    result : LogRankResult
        Object with the chi-squared ``statistic``, the degrees of
        freedom ``dof``, the ``p_value``, the ``weighting`` and the number
        of ``strata``.

    Raises
    ------

    ValueError
        If there are fewer than two groups, ``Z``, ``c``, ``n`` or
        ``strata`` does not have one entry per value, a group or stratum
        label is missing (NaN or None), the weighting is unknown, the
        data have left or interval censoring, or ``t0`` is not a number.

    Examples
    --------
    >>> from surpyval import logrank
    >>> x = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161,
    ...      5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
    >>> c = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1,
    ...      0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    >>> Z = [1] * 11 + [2] * 12
    >>> res = logrank(x, Z, c=c)
    >>> print(round(res.statistic, 2), round(res.p_value, 4))
    3.4 0.0653

    These are the AML maintenance data (``aml`` in R's ``survival``
    package), for which R's ``survdiff`` also gives 3.4 on 1 degree of
    freedom. The Gehan-Breslow weights put more weight on the early
    failures:

    >>> res = logrank(x, Z, c=c, weighting="gehan")
    >>> print(round(res.statistic, 3), round(res.p_value, 4))
    2.723 0.0989

    With delayed entry: suppose the first four units of each group came
    under observation at time 6 (``tl``), so they are not at risk at the
    failures at 5:

    >>> tl = [6, 6, 6, 6] + [0] * 7 + [0, 0, 6, 6, 6, 6] + [0] * 6
    >>> res = logrank(x, Z, c=c, tl=tl)
    >>> print(round(res.statistic, 3), round(res.p_value, 4))
    3.366 0.0666

    The Peto-Peto weights, and the comparison over the first 30 weeks
    only (``t0``):

    >>> res = logrank(x, Z, c=c, weighting="peto-peto")
    >>> print(round(res.statistic, 3), round(res.p_value, 4))
    2.708 0.0998
    >>> res = logrank(x, Z, c=c, t0=30)
    >>> print(round(res.statistic, 3), round(res.p_value, 4))
    2.13 0.1444

    To compare every pair of several groups, use
    :func:`~surpyval.pairwise_logrank`.

    References
    ----------

    Klein, J. P. and Moeschberger, M. L. (2003), "Survival Analysis:
    Techniques for Censored and Truncated Data", 2nd ed., Chapter 7.

    Peto, R. and Peto, J. (1972), "Asymptotically efficient rank invariant
    test procedures", Journal of the Royal Statistical Society A 135,
    185-207.

    Andersen, P. K., Borgan, O., Gill, R. D. and Keiding, N. (1982),
    "Linear nonparametric tests for comparison of counting processes, with
    applications to censored survival data", International Statistical
    Review 50, 219-258.
    """
    return _logrank(
        x, Z, c, n, weighting, rho, gamma, strata, tl, t0, warn_singletons=True
    )


#: The ``weighting`` values of :func:`logrank`.
WEIGHTINGS = (
    "log-rank",
    "gehan",
    "tarone-ware",
    "fleming-harrington",
    "peto-peto",
    "modified-peto-peto",
)


def _cut_at(
    t0: float,
    x: npt.NDArray,
    Z: npt.NDArray,
    c: npt.NDArray | None,
    n: npt.NDArray | None,
    tl: npt.NDArray | None,
    strata: npt.ArrayLike | None,
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray | None,
    npt.NDArray | None,
    npt.NDArray | None,
]:
    """The data as they would be had follow-up stopped at ``t0``: an
    observation past ``t0`` is right censored at ``t0``, and a row that
    entered at or after it (``tl >= t0``, never at risk by then) is
    dropped, with its group label and stratum."""
    try:
        t0 = float(t0)
    except (TypeError, ValueError):
        raise ValueError(
            "'t0' must be a number, the time to compare the groups up to; "
            "got {!r}".format(t0)
        ) from None
    if np.isnan(t0):
        raise ValueError(
            "'t0' must be a number, the time to compare the groups up to; "
            "got nan. Leave it out to compare the whole follow-up."
        )
    if strata is not None:
        strata = np.asarray(strata)
        if len(strata) != len(x):
            raise ValueError("'strata' must have a label for each observation")
    c = np.zeros(x.shape) if c is None else np.asarray(c)
    if ((c != 0) & (c != 1)).any():
        raise ValueError(
            "Log-rank test can only be used with observed and "
            + "right censored data"
        )
    x = np.asarray(x, dtype=float)
    after = x > t0
    x = np.where(after, t0, x)
    c = np.where(after, 1, c)
    keep = np.ones(x.shape, dtype=bool) if tl is None else tl < t0
    return (
        x[keep],
        Z[keep],
        c[keep],
        None if n is None else n[keep],
        None if tl is None else tl[keep],
        None if strata is None else strata[keep],
    )


def _logrank(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    weighting: str,
    rho: float,
    gamma: float,
    strata: npt.ArrayLike | None,
    tl: npt.ArrayLike | float | None,
    t0: float | None,
    warn_singletons: bool,
) -> LogRankResult:
    """:func:`logrank`, with its warning of singleton groups optional (the
    pairwise tests warn once, of the whole data)."""
    from scipy.stats import chi2

    weightings = list(WEIGHTINGS)
    if weighting not in weightings:
        raise ValueError(f"'weighting' must be in {weightings}")

    Z = np.asarray(Z)
    if Z.ndim != 1:
        raise ValueError("'Z' must be a 1D array of group labels")
    if len(Z) != len(np.atleast_1d(x)):
        raise ValueError("'Z' must have a label for each observation")
    # A missing label is not a group. NaN != NaN, so a NaN label used to
    # become an extra group whose rows every ``Z == g`` mask then missed:
    # the rows were dropped and the degrees of freedom went up by one.
    import pandas as pd

    if pd.isna(Z).any():
        raise ValueError(
            "'Z' has missing (NaN or None) group labels; drop those rows "
            "or give them a label"
        )

    groups = np.unique(Z)
    if groups.size < 2:
        raise ValueError("Test requires at least two groups")

    x = np.atleast_1d(x)
    c_arr = None if c is None else np.atleast_1d(c)
    n_arr = None if n is None else np.atleast_1d(n)
    tl_arr = None
    if tl is not None:
        tl_arr = np.asarray(tl, dtype=float)
        if tl_arr.ndim == 0:
            tl_arr = np.full(x.shape[0], float(tl_arr))
    # Checked here: a short ``c`` or ``n`` used to fail as an IndexError
    # from the boolean group mask.
    for name, arr in (("c", c_arr), ("n", n_arr), ("tl", tl_arr)):
        if arr is not None and arr.shape != x.shape:
            raise ValueError(
                "'{}' must have one entry for each observation; got {} "
                "for {} observations".format(name, arr.size, x.size)
            )

    k = groups.size
    # Groups are categories. When most of them hold a single unit the
    # labels are almost certainly a continuous variable, or an argument in
    # the wrong slot: logrank(x, x * 1.3) ran a 50-group test with 49
    # degrees of freedom on 50 observations, in silence (#485).
    units = np.ones(x.shape) if n_arr is None else n_arr.astype(float)
    sizes = np.array([units[Z == g].sum() for g in groups])
    singletons = int(np.sum(sizes <= 1))
    if warn_singletons and singletons > k / 2:
        warnings.warn(
            "logrank: {} of the {} groups in `Z` have a single member ({} "
            "observations in all), so `Z` looks like a continuous "
            "variable, or another argument in its place (the "
            "signature is logrank(x, Z, c, n)). `Z` should hold group "
            "labels; to test a continuous covariate, fit CoxPH.".format(
                singletons, k, int(sizes.sum())
            ),
            stacklevel=3,
        )
    if t0 is not None:
        x, Z, c_arr, n_arr, tl_arr, strata = _cut_at(
            t0, x, Z, c_arr, n_arr, tl_arr, strata
        )
    n_strata = None
    if strata is None:
        z, V, E = _logrank_z_v(
            x, Z, c_arr, n_arr, groups, weighting, rho, gamma, tl_arr
        )
    else:
        strata = np.asarray(strata)
        if len(strata) != len(x):
            raise ValueError("'strata' must have a label for each observation")
        # As for Z: every ``strata == s`` mask missed a NaN stratum, so its
        # rows silently dropped out of the test.
        if pd.isna(strata).any():
            raise ValueError(
                "'strata' has missing (NaN or None) labels; drop those rows "
                "or give them a stratum"
            )
        z = np.zeros(k)
        V = np.zeros((k, k))
        E = np.zeros(k)
        unique_strata = np.unique(strata)
        n_strata = int(unique_strata.size)
        for s in unique_strata:
            mask = strata == s
            z_s, V_s, E_s = _logrank_z_v(
                x[mask],
                Z[mask],
                None if c_arr is None else c_arr[mask],
                None if n_arr is None else n_arr[mask],
                groups,
                weighting,
                rho,
                gamma,
                None if tl_arr is None else tl_arr[mask],
            )
            z += z_s
            V += V_s
            E += E_s

    # Only groups expected to have events carry information, as in R's
    # ``survdiff``: a group never at risk at an event time has zero
    # observed-minus-expected and zero variance, and counting it (as the
    # number of labels did) added a degree of freedom for nothing and
    # deflated the p-value. The covariance matrix of the rest is singular
    # (rows sum to zero), so the last of them is dropped.
    informative = E > 0
    z_r = z[informative][:-1]
    V_r = V[np.ix_(informative, informative)][:-1, :-1]
    dof = int(informative.sum()) - 1
    if dof < 1:
        # No event anywhere, or every risk set holds one group only:
        # there is nothing to compare. R reports a zero statistic; the
        # p-value stays 1 (no evidence against the null), as it was.
        statistic = 0.0
        p_value = 1.0
    else:
        statistic = safe_quadform(V_r, z_r)
        p_value = float(chi2.sf(statistic, dof))

    if weighting == "fleming-harrington":
        weighting = f"fleming-harrington(rho={rho}, gamma={gamma})"

    return LogRankResult(statistic, dof, p_value, weighting, strata=n_strata)


#: The ``adjust`` values of :func:`pairwise_logrank`.
ADJUSTMENTS = ("holm", "bonferroni", "bh", "none")


def adjust_p_values(p: npt.ArrayLike, method: str = "holm") -> npt.NDArray:
    """
    P-values adjusted for testing several hypotheses at once, as R's
    ``p.adjust``.

    Parameters
    ----------
    p : array like
        The p-values of the separate tests.
    method : str, optional
        One of:

        - "holm" (the default): Holm's step-down method, which controls
          the chance of any false rejection (the family-wise error rate)
          and always rejects at least what Bonferroni's does;
        - "bonferroni": each p-value times the number of tests;
        - "bh": Benjamini and Hochberg's method, which controls the
          expected share of false rejections among the rejections (the
          false discovery rate), a weaker promise that rejects more;
        - "none": the p-values as they are.

        Each adjusted p-value is capped at 1. A NaN p-value stays NaN and
        is not counted as a test.

    Returns
    -------
    numpy array
        The adjusted p-values, in the order given.

    Examples
    --------
    >>> from surpyval.univariate.nonparametric.logrank import (
    ...     adjust_p_values,
    ... )
    >>> p = [0.01, 0.02, 0.03, 0.04, 0.05]
    >>> adjust_p_values(p, "holm").round(4)
    array([0.05, 0.08, 0.09, 0.09, 0.09])
    >>> adjust_p_values(p, "bh").round(4)
    array([0.05, 0.05, 0.05, 0.05, 0.05])

    References
    ----------
    Holm, S. (1979), "A simple sequentially rejective multiple test
    procedure", Scandinavian Journal of Statistics 6, 65-70.

    Benjamini, Y. and Hochberg, Y. (1995), "Controlling the false discovery
    rate: a practical and powerful approach to multiple testing", Journal
    of the Royal Statistical Society B 57, 289-300.
    """
    method = "none" if method is None else str(method).lower()
    if method not in ADJUSTMENTS:
        raise ValueError(
            "'adjust' must be one of {}; got {!r}".format(
                list(ADJUSTMENTS), method
            )
        )
    p = np.asarray(p, dtype=float)
    out = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = q.size
    if m == 0 or method == "none":
        out[ok] = q
        return out
    if method == "bonferroni":
        out[ok] = np.minimum(1.0, q * m)
        return out
    order = np.argsort(q, kind="stable")
    ranked = q[order]
    if method == "holm":
        adjusted = np.maximum.accumulate(ranked * (m - np.arange(m)))
    else:
        steps = ranked * m / np.arange(1, m + 1)
        adjusted = np.minimum.accumulate(steps[::-1])[::-1]
    adjusted = np.minimum(1.0, adjusted)
    result = np.empty(m)
    result[order] = adjusted
    out[ok] = result
    return out


class PairwiseLogRankResult:
    """
    The log-rank tests of every pair of groups, from
    :func:`pairwise_logrank`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per pair, with columns ``group_a``, ``group_b``,
        ``statistic`` (the chi-squared statistic of the two-group test),
        ``dof`` (1, or 0 where the pair has nothing to compare),
        ``p_value`` (the test's own) and ``p_adjusted`` (adjusted for the
        number of pairs by ``adjust``).
    adjust : str
        The multiplicity adjustment.
    weighting : str
        The weighting of each test.
    strata : int or None
        The number of strata of stratified tests; None otherwise.

    Examples
    --------
    >>> from surpyval import pairwise_logrank
    >>> x = [3, 5, 7, 8, 10, 2, 3, 4, 4, 6, 6, 9, 11, 12, 14]
    >>> Z = ["a"] * 5 + ["b"] * 5 + ["c"] * 5
    >>> res = pairwise_logrank(x, Z)
    >>> res.table[["group_a", "group_b", "p_adjusted"]].round(4)
      group_a group_b  p_adjusted
    0       a       b      0.0942
    1       a       c      0.0942
    2       b       c      0.0150
    """

    def __init__(
        self,
        table: pd.DataFrame,
        adjust: str,
        weighting: str,
        strata: int | None,
    ) -> None:
        self.table = table
        self.adjust = adjust
        self.weighting = weighting
        self.strata = strata

    def __repr__(self) -> str:
        header = "Pairwise Log-Rank Tests"
        if self.strata:
            header = "Pairwise Stratified Log-Rank Tests"
        out = (
            header
            + "\n"
            + "=" * len(header)
            + f"\nWeighting        : {self.weighting}"
            + f"\nAdjustment       : {self.adjust}"
        )
        if self.strata is not None:
            out += f"\nStrata           : {self.strata}"
        return out + "\n" + self.table.to_string(index=False)

    def to_dict(self) -> dict:
        """The result as strict JSON: the settings and one record per pair
        (a non-finite statistic or p-value is ``None``)."""

        def finite(v: Any) -> Any:
            v = float(v)
            return v if np.isfinite(v) else None

        rows = []
        for row in self.table.itertuples(index=False):
            rows.append(
                {
                    "group_a": _plain(row.group_a),
                    "group_b": _plain(row.group_b),
                    "statistic": finite(row.statistic),
                    "dof": int(row.dof),
                    "p_value": finite(row.p_value),
                    "p_adjusted": finite(row.p_adjusted),
                }
            )
        return {
            "test": "pairwise log-rank",
            "weighting": self.weighting,
            "adjust": self.adjust,
            "strata": self.strata,
            "pairs": rows,
        }


def _plain(v: Any) -> Any:
    # A numpy scalar label as the Python value JSON can write.
    return v.item() if isinstance(v, np.generic) else v


def pairwise_logrank(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    weighting: str = "log-rank",
    rho: float = 0,
    gamma: float = 0,
    strata: npt.ArrayLike | None = None,
    tl: npt.ArrayLike | float | None = None,
    t0: float | None = None,
    adjust: str = "holm",
) -> PairwiseLogRankResult:
    """
    The (weighted) log-rank test of every pair of groups, in one table,
    with the p-values adjusted for the number of pairs.

    The k-group :func:`logrank` test says whether any group differs;
    this says which. Each pair is tested on its own two groups' data
    alone (the rows of the other groups are left out), exactly as
    :func:`logrank` would test them, so each row of the table is what
    ``logrank`` gives on that pair. With k groups there are
    k (k - 1) / 2 tests, and the more tests are run the more likely one
    of them rejects by chance; the adjusted p-value, ``p_adjusted``,
    allows for that. Compare it, not ``p_value``, with the significance
    level.

    Parameters
    ----------
    x, Z, c, n, weighting, rho, gamma, strata, tl, t0
        As for :func:`logrank`: the times, the group labels, the censoring
        flags and counts, the weighting and its Fleming-Harrington
        parameters, the strata of a stratified test, the entry times and
        the end of the comparison window. ``Z`` needs at least two
        groups.
    adjust : str, optional
        The multiplicity adjustment of the p-values, as R's ``p.adjust``
        (see :func:`~surpyval.univariate.nonparametric.logrank.\
adjust_p_values`): "holm" (the default; controls the chance of any
        false rejection), "bonferroni", "bh" (Benjamini-Hochberg; controls
        the false discovery rate) or "none".

    Returns
    -------
    PairwiseLogRankResult
        With the ``table`` of the pairs (``group_a``, ``group_b``,
        ``statistic``, ``dof``, ``p_value``, ``p_adjusted``), in the
        sorted order of the group labels.

    Raises
    ------
    ValueError
        As :func:`logrank` does, and for an ``adjust`` not listed above.

    Examples
    --------
    Patients with advanced lung cancer by their ECOG performance score
    (0, 1 or 2; the one patient scored 3 is left out):

    >>> from surpyval import pairwise_logrank
    >>> from surpyval.datasets import load_lung
    >>> df = load_lung().dropna(subset=["ph.ecog"])
    >>> df = df[df["ph.ecog"] < 3]
    >>> res = pairwise_logrank(
    ...     df["time"], df["ph.ecog"].astype(int), c=1 - df["status"]
    ... )
    >>> res.table.round(4)
       group_a  group_b  statistic  dof  p_value  p_adjusted
    0        0        1     3.4568    1   0.0630      0.0630
    1        0        2    16.6823    1   0.0000      0.0001
    2        1        2     8.4334    1   0.0037      0.0074

    Scores 0 and 1 are not distinguished at the 5% level once the three
    comparisons are allowed for; score 2 differs from each.

    References
    ----------
    Klein, J. P. and Moeschberger, M. L. (2003), "Survival Analysis:
    Techniques for Censored and Truncated Data", 2nd ed., Chapter 7.

    Holm, S. (1979), "A simple sequentially rejective multiple test
    procedure", Scandinavian Journal of Statistics 6, 65-70.
    """
    import pandas as pd

    adjust = "none" if adjust is None else str(adjust).lower()
    if adjust not in ADJUSTMENTS:
        raise ValueError(
            "'adjust' must be one of {}; got {!r}".format(
                list(ADJUSTMENTS), adjust
            )
        )
    if weighting not in WEIGHTINGS:
        raise ValueError(f"'weighting' must be in {list(WEIGHTINGS)}")
    Z_arr = np.asarray(Z)
    x_arr = np.atleast_1d(np.asarray(x))
    if Z_arr.ndim != 1:
        raise ValueError("'Z' must be a 1D array of group labels")
    if len(Z_arr) != len(x_arr):
        raise ValueError("'Z' must have a label for each observation")
    if pd.isna(Z_arr).any():
        raise ValueError(
            "'Z' has missing (NaN or None) group labels; drop those rows "
            "or give them a label"
        )
    groups = np.unique(Z_arr)
    if groups.size < 2:
        raise ValueError("Test requires at least two groups")

    def rows_of(arr: Any, mask: npt.NDArray) -> Any:
        if arr is None:
            return None
        arr = np.asarray(arr)
        if arr.ndim == 0:
            return arr
        if arr.shape[0] != mask.shape[0]:
            # Left for ``logrank`` to refuse, with its own message.
            return arr
        return arr[mask]

    # The whole data are checked (and warned of) once, by the k-group
    # test; its result is not needed.
    whole = _logrank(
        x_arr, Z_arr, c, n, weighting, rho, gamma, strata, tl, t0, True
    )
    records = []
    for i, a in enumerate(groups):
        for b in groups[i + 1 :]:
            mask = (Z_arr == a) | (Z_arr == b)
            res = _logrank(
                x_arr[mask],
                Z_arr[mask],
                rows_of(c, mask),
                rows_of(n, mask),
                weighting,
                rho,
                gamma,
                rows_of(strata, mask),
                rows_of(tl, mask),
                t0,
                warn_singletons=False,
            )
            records.append(
                {
                    "group_a": a,
                    "group_b": b,
                    "statistic": float(res.statistic),
                    "dof": int(res.dof),
                    "p_value": float(res.p_value),
                }
            )
    table = pd.DataFrame.from_records(records)
    tested = table["dof"].to_numpy() > 0
    p = np.where(tested, table["p_value"].to_numpy(), np.nan)
    adjusted = adjust_p_values(p, adjust)
    # A pair with nothing to compare keeps the p-value of 1 its test
    # gives, and is not counted among the tests.
    table["p_adjusted"] = np.where(tested, adjusted, 1.0)
    return PairwiseLogRankResult(
        table, adjust, whole.weighting, strata=whole.strata
    )
