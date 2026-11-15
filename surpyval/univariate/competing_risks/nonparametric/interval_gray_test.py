"""Gray's test for interval-censored competing-risks data, by multiple
imputation.

Each interval-censored failure is given an exact time (and, where it was
not recorded, a cause) drawn from the non-parametric estimate of the
pooled data, refitted to a bootstrap sample each time so the imputations
carry the estimate's own uncertainty; Gray's (1988) test is computed on
each completed data set and the results are combined by Rubin's rules for
a multivariate statistic (Li, Raghunathan and Rubin 1991). The approach is
the multiple-imputation route of Pan (2000) and Huang, Lee and Yu (2008)
for interval-censored survival data, applied to Gray's statistic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2, f

from surpyval.univariate.competing_risks.labels import (
    label_mask,
    ordered_labels,
)
from surpyval.univariate.competing_risks.nonparametric import _ic_npmle
from surpyval.univariate.competing_risks.nonparametric._ic_data import (
    read_rows,
)
from surpyval.univariate.competing_risks.nonparametric.gray_test import (
    _score_and_variance,
)
from surpyval.utils import is_missing_event


class IntervalGrayTestResult:
    """
    The result of :func:`interval_gray_test`.

    Attributes
    ----------
    statistic : float
        The combined chi-squared statistic, ``U' T^-1 U`` with ``U`` the
        mean over the imputations of Gray's score and ``T`` its total
        variance (within plus between imputations). With no
        interval-censored row it is Gray's statistic.
    df : int
        The degrees of freedom, the number of groups less one.
    df_denominator : float
        The denominator degrees of freedom of the F reference
        distribution of ``statistic / df`` (Li, Raghunathan and Rubin
        1991); infinite (the chi-squared distribution) when the
        imputations do not vary.
    p_value : float
        The p-value; a small one is evidence the groups' cumulative
        incidences of ``cause`` differ.
    cause : Any
        The cause compared.
    groups : list
        The groups, in order.
    n_imputations : int
        The number of imputed data sets.
    relative_increase : float
        The relative increase in variance due to the interval censoring,
        ``r`` of Rubin's rules (0 without interval-censored rows).
    """

    def __init__(
        self,
        statistic: float,
        df: int,
        df_denominator: float,
        p_value: float,
        cause: Any,
        groups: list,
        n_imputations: int,
        relative_increase: float,
    ) -> None:
        self.statistic = statistic
        self.df = df
        self.df_denominator = df_denominator
        self.p_value = p_value
        self.cause = cause
        self.groups = groups
        self.n_imputations = n_imputations
        self.relative_increase = relative_increase

    def __repr__(self) -> str:
        header = "Gray's Test, Interval-Censored (multiple imputation)"
        return (
            header
            + "\n"
            + "=" * len(header)
            + f"\nCause            : {self.cause!r}"
            + f"\nGroups           : {self.groups!r}"
            + f"\nImputations      : {self.n_imputations}"
            + f"\nStatistic        : {self.statistic:.6g}"
            + f"\nDoF              : {self.df}"
            + f"\nDenominator DoF  : {self.df_denominator:.6g}"
            + f"\np-value          : {self.p_value:.6g}"
        )

    def to_dict(self) -> dict:
        """The result as strict JSON (a non-finite number is ``None``)."""
        from surpyval.serialisation import to_native

        def finite(v: Any) -> Any:
            v = float(v)
            return v if np.isfinite(v) else None

        return {
            "test": "interval-censored Gray",
            "cause": to_native(self.cause),
            "groups": [to_native(g) for g in self.groups],
            "statistic": finite(self.statistic),
            "df": int(self.df),
            "df_denominator": finite(self.df_denominator),
            "p_value": finite(self.p_value),
            "n_imputations": int(self.n_imputations),
            "relative_increase": finite(self.relative_increase),
        }


def _groups(group: Any, N: int) -> tuple[npt.NDArray, list]:
    group_seq: Any = group
    g_list = list(group_seq) if np.ndim(group) else [group]
    if len(g_list) != N:
        raise ValueError(
            f"group must have one label per row: got {len(g_list)} for "
            f"{N} rows."
        )
    g_arr = np.empty(N, dtype=object)
    for i, g in enumerate(g_list):
        g_arr[i] = g
    if any(is_missing_event(g) for g in g_arr):
        raise ValueError(
            "group has a missing label (None / NaN); every row needs a "
            "group."
        )
    groups = [
        g.item() if isinstance(g, np.generic) else g
        for g in ordered_labels(g_arr)
    ]
    if len(groups) < 2:
        raise ValueError("Gray's test needs at least two groups.")
    index = {g: i for i, g in enumerate(groups)}
    return np.array([index[g] for g in g_arr], dtype=int), groups


def _gray_parts(
    time: npt.NDArray,
    cause: npt.NDArray,
    count: npt.NDArray,
    gi: npt.NDArray,
    K: int,
    k: int,
    rho: float,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Gray's score and variance of exact / right-censored rows (cause -1
    censored), as :func:`~surpyval.gray_test` computes them."""
    times, t_idx = np.unique(time, return_inverse=True)
    T = times.size
    counts = np.zeros((K, T))
    np.add.at(counts, (gi, t_idx), count)
    is_cause = cause == k
    is_comp = (cause >= 0) & ~is_cause
    d1 = np.zeros((K, T))
    np.add.at(d1, (gi[is_cause], t_idx[is_cause]), count[is_cause])
    d2 = np.zeros((K, T))
    np.add.at(d2, (gi[is_comp], t_idx[is_comp]), count[is_comp])
    Y = counts[:, ::-1].cumsum(axis=1)[:, ::-1]
    return _score_and_variance(Y, d1, d2, rho)


def interval_gray_test(
    x: npt.ArrayLike | None = None,
    e: npt.ArrayLike | None = None,
    group: npt.ArrayLike | None = None,
    event: Any = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    xl: npt.ArrayLike | None = None,
    xr: npt.ArrayLike | None = None,
    rho: float = 0.0,
    masked: str = "refuse",
    n_imputations: int = 20,
    random_state: Any = None,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> IntervalGrayTestResult:
    r"""
    Gray's k-sample test comparing the cumulative incidence of one cause
    across groups, for interval-censored data.

    :func:`~surpyval.gray_test` needs each failure's time. Here a failure
    may be known only to lie in a window (found at an inspection), and its
    cause may be missing (``masked="include"``). Each such failure is
    given an exact time and cause, drawn from the non-parametric estimate
    (:class:`~surpyval.IntervalCompetingRisks`) of all the groups together;
    Gray's test is computed on the completed data, and this is repeated
    ``n_imputations`` times and the results combined (multiple
    imputation). With no interval-censored row nothing is imputed and the
    result is Gray's test exactly.

    Parameters
    ----------
    x : array_like, optional
        The times: one column with the flags ``c``, or two columns, each
        row the window ``[xl, xr]`` the failure lies in. Leave it out to
        give ``xl`` and ``xr``.
    e : array_like
        The cause of each failure; a missing value marks a right-censored
        row (or, with ``masked="include"``, a failure of unrecorded
        cause).
    group : array_like
        The group of each row (two or more groups, no missing labels).
    event : scalar
        The cause whose cumulative incidence is compared.
    c : array_like, optional
        Censoring flags: 0 exact, 1 right censored, -1 left censored, 2
        interval censored. Derived as by
        :meth:`IntervalCompetingRisks.fit` if omitted.
    n : array_like, optional
        Counts, whole numbers (each unit is imputed on its own). Default 1.
    xl, xr : array_like, optional
        In place of ``x``, each row's lower and upper time.
    rho : float, optional
        Gray's weight parameter, as for :func:`~surpyval.gray_test`
        (``0``, the default, is the standard test).
    masked : str, optional
        ``"refuse"`` (default) or ``"include"``, as for
        :meth:`IntervalCompetingRisks.fit`.
    n_imputations : int, optional
        The number of imputed data sets, at least 2. Defaults to 20.
    random_state : None, int or numpy.random.Generator, optional
        The seed of the bootstrap samples and the imputations.
    tol, max_iter : optional
        The convergence settings of the non-parametric estimates, as for
        :meth:`IntervalCompetingRisks.fit`.

    Returns
    -------
    IntervalGrayTestResult
        The combined ``statistic``, ``df``, ``df_denominator`` and
        ``p_value``, with ``to_dict``.

    Notes
    -----
    For imputation ``m``, the units are resampled with replacement and the
    pooled non-parametric estimate is refitted to the sample (the
    approximate Bayesian bootstrap, which makes the imputations carry the
    estimate's uncertainty); each interval-censored unit is then given one
    of the intervals of its window in proportion to the refitted masses of
    its cause (of every cause if its cause was not recorded), and the time
    at that interval's upper end, the point at which the estimate's
    cumulative incidence rises. If the refitted estimate has no mass in a
    unit's window, the full-data estimate is used for that unit.
    Right-censored and exact rows are kept as they are.

    Gray's score vector :math:`U_m` and variance :math:`V_m` of the first
    :math:`k` groups (:math:`k` the number of groups less one) are combined
    by Rubin's rules: :math:`\bar U` and :math:`\bar V` their means,
    :math:`B` the covariance of the :math:`U_m` over the imputations,
    :math:`r = (1 + 1/M) \operatorname{tr}(B \bar V^{-1}) / k` and the
    statistic :math:`\bar U' \bar V^{-1} \bar U / (1 + r)`, whose
    :math:`1/k` is referred to an F distribution with :math:`k` and
    :math:`\nu` degrees of freedom, :math:`\nu` as given by Li,
    Raghunathan and Rubin (1991). Without interval-censored rows
    :math:`B = 0`, :math:`r = 0` and it is Gray's chi-squared test.

    Imputing from the pooled estimate is imputing under the null
    hypothesis, as the score tests of interval-censored survival data
    (``interval_logrank``) take their scores from the pooled estimate: it
    keeps the test's size when the groups are inspected on different
    schedules, at some cost in power, since an imputed failure's place in
    its window carries no group difference. The test assumes the
    inspection times are independent of the failure process, and that
    the groups share the cumulative incidences of the other causes
    closely enough for pooled imputation (exactly so under the null
    hypothesis that every cause's incidence is the same). Its size was
    checked by simulation (see the guide).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import interval_gray_test
    >>> rng = np.random.default_rng(0)
    >>> group = rng.binomial(1, 0.5, 200)
    >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * group)))
    >>> t_b = rng.exponential(1 / 0.05, 200)
    >>> t = np.minimum(t_a, t_b)
    >>> cause = np.where(t_a < t_b, "a", "b")
    >>> # Inspected every 2 time units up to 20:
    >>> lo = np.floor(t / 2) * 2
    >>> seen = lo < 20
    >>> x = np.column_stack([np.where(seen, lo, 20),
    ...                      np.where(seen, lo + 2, np.inf)])
    >>> e = np.where(seen, cause, None)
    >>> res = interval_gray_test(x, e, group, event="a", random_state=1)
    >>> bool(res.p_value < 0.01)
    True

    References
    ----------
    Gray, R. J. (1988), "A class of K-sample tests for comparing the
    cumulative incidence of a competing risk", Annals of Statistics 16,
    1141-1154.

    Li, K. H., Raghunathan, T. E. and Rubin, D. B. (1991), "Large-sample
    significance levels from multiply imputed data using moment-based
    statistics and an F reference distribution", Journal of the American
    Statistical Association 86, 1065-1073.

    Pan, W. (2000), "A multiple imputation approach to Cox regression with
    interval-censored data", Biometrics 56, 199-203.

    Huang, J., Lee, C. and Yu, Q. (2008), "A generalized log-rank test for
    interval-censored failure time data via multiple imputation",
    Statistics in Medicine 27, 3217-3226.

    Hudgens, M. G., Satten, G. A. and Longini, I. M. (2001),
    "Nonparametric maximum likelihood estimation for competing risks
    survival data subject to interval censoring and truncation",
    Biometrics 57, 74-80.
    """
    from surpyval.utils.rng import as_generator

    if e is None:
        raise ValueError("e, the cause of each failure, is needed.")
    if group is None:
        raise ValueError("group, the group of each row, is needed.")
    if (
        isinstance(n_imputations, bool)
        or not float(n_imputations).is_integer()
        or n_imputations < 2
    ):
        raise ValueError(
            "n_imputations must be a whole number of at least 2; got "
            f"{n_imputations!r}"
        )
    M = int(n_imputations)
    if not np.isfinite(rho):
        raise ValueError("rho must be a finite number.")
    rows = read_rows(x, e, c, n, xl, xr, masked, "interval_gray_test")
    N = rows.xl.size
    gi, groups = _groups(group, N)
    K = len(groups)
    causes_arr = np.empty(len(rows.causes), dtype=object)
    for i, k in enumerate(rows.causes):
        causes_arr[i] = k
    hit = label_mask(causes_arr, event)
    if not hit.any():
        raise ValueError(
            f"No failures of cause {event!r} to compare; the causes are "
            f"{rows.causes}."
        )
    k_event = int(np.flatnonzero(hit)[0])
    units = rows.n
    if not np.all(units == np.round(units)):
        raise ValueError(
            "n must be whole numbers here: each unit's failure time is "
            "imputed on its own."
        )
    units = units.astype(np.int64)

    exact = rows.xl == rows.xr
    right = np.isinf(rows.xr) & (rows.xr > 0)
    known = rows.cause >= 0
    keep = (exact & known) | right
    impute = ~keep
    rng = as_generator(random_state)

    if not impute.any():
        U, V = _gray_parts(
            np.where(right, rows.xl, rows.xr),
            rows.cause,
            units.astype(float),
            gi,
            K,
            k_event,
            float(rho),
        )
        from surpyval.utils.linalg import safe_quadform

        stat = safe_quadform(V[:-1, :-1], U[:-1])
        return IntervalGrayTestResult(
            stat,
            K - 1,
            np.inf,
            float(chi2.sf(stat, K - 1)),
            event,
            groups,
            M,
            0.0,
        )

    s = _ic_npmle.structure(
        rows.xl, rows.xr, rows.cause, rows.n, len(rows.causes)
    )
    full = _ic_npmle.fit_masses(s, tol=tol, max_iter=max_iter).p
    A = s.A.tocsr()
    imp_rows = np.flatnonzero(impute)
    base_time = np.where(right, rows.xl, rows.xr)[keep]
    base_cause = rows.cause[keep]
    base_count = units[keep].astype(float)
    base_group = gi[keep]

    total = float(s.n.sum())
    share = s.n / total
    Us, Vs = [], []
    for _ in range(M):
        counts = rng.multinomial(int(round(total)), share).astype(float)
        p = _ic_npmle.fit_masses(s, n=counts, tol=tol, max_iter=max_iter).p
        t_new, c_new, g_new = [], [], []
        for i in imp_rows.tolist():
            u = int(s.row[i])
            cols = A.indices[A.indptr[u] : A.indptr[u + 1]]
            w = p[cols]
            if w.sum() <= 0:
                w = full[cols]
            draws = rng.multinomial(units[i], w / w.sum())
            hit = np.flatnonzero(draws)
            for j in hit:
                t_new.append(np.repeat(s.upper[cols[j]], draws[j]))
                c_new.append(np.repeat(s.cause[cols[j]], draws[j]))
                g_new.append(np.repeat(gi[i], draws[j]))
        time = np.concatenate([base_time] + t_new)
        cause = np.concatenate([base_cause] + c_new)
        grp = np.concatenate([base_group] + g_new)
        count = np.concatenate(
            [base_count, np.ones(time.size - base_time.size)]
        )
        U, V = _gray_parts(time, cause, count, grp, K, k_event, float(rho))
        Us.append(U[:-1])
        Vs.append(V[:-1, :-1])

    Ubar = np.mean(Us, axis=0)
    Vbar = np.mean(Vs, axis=0)
    B = np.atleast_2d(np.cov(np.array(Us).T, ddof=1))
    q = K - 1
    Vinv = np.linalg.pinv(Vbar)
    r = float((1.0 + 1.0 / M) * np.trace(B @ Vinv) / q)
    stat = float(Ubar @ Vinv @ Ubar) / (1.0 + r)
    if r <= 1e-12:
        return IntervalGrayTestResult(
            stat, q, np.inf, float(chi2.sf(stat, q)), event, groups, M, r
        )
    t = q * (M - 1)
    if t > 4:
        nu = 4.0 + (t - 4.0) * (1.0 + (1.0 - 2.0 / t) / r) ** 2
    else:
        nu = t * (1.0 + 1.0 / q) * (1.0 + 1.0 / r) ** 2 / 2.0
    p_value = float(f.sf(stat / q, q, nu))
    return IntervalGrayTestResult(
        stat, q, float(nu), p_value, event, groups, M, r
    )
