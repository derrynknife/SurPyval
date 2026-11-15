"""Gray's test for interval-censored competing-risks data, by multiple
imputation: what ``gray_test`` runs when the data have left- or
interval-censored rows (or masked causes). Private; reached only through
``gray_test``.

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
    IMPUTED,
    GrayTestResult,
    _score_and_variance,
)
from surpyval.utils import is_missing_event


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


def imputed_gray_test(
    x: npt.ArrayLike | None,
    e: npt.ArrayLike | None,
    group: npt.ArrayLike | None,
    event: Any,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    xl: npt.ArrayLike | None,
    xr: npt.ArrayLike | None,
    rho: float,
    masked: str,
    n_imputations: int,
    random_state: Any,
    tol: float,
    max_iter: int,
) -> GrayTestResult:
    """Gray's test of windowed rows by multiple imputation; see
    :func:`~surpyval.gray_test` (its Notes describe the method)."""
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
    rows = read_rows(x, e, c, n, xl, xr, masked, "gray_test")
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
    exact = rows.xl == rows.xr
    right = np.isinf(rows.xr) & (rows.xr > 0)
    known = rows.cause >= 0
    keep = (exact & known) | right
    impute = ~keep
    rng = as_generator(random_state)

    if not impute.any():
        # Nothing to impute: Gray's test of the times as they are.
        U, V = _gray_parts(
            np.where(right, rows.xl, rows.xr),
            rows.cause,
            rows.n,
            gi,
            K,
            k_event,
            float(rho),
        )
        from surpyval.utils.linalg import safe_quadform

        stat = safe_quadform(V[:-1, :-1], U[:-1])
        return GrayTestResult(
            statistic=stat,
            df=K - 1,
            p_value=float(chi2.sf(stat, K - 1)),
            cause=event,
            groups=groups,
        )
    units = rows.n
    if not np.all(units == np.round(units)):
        raise ValueError(
            "n must be whole numbers when rows are left or interval "
            "censored: each unit's failure time is imputed on its own. "
            "Give each unit its own row, or round the counts."
        )
    units = units.astype(np.int64)

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
        return GrayTestResult(
            statistic=stat,
            df=q,
            p_value=float(chi2.sf(stat, q)),
            cause=event,
            groups=groups,
            method=IMPUTED,
            n_imputations=M,
            df_denominator=np.inf,
            relative_increase=r,
        )
    t = q * (M - 1)
    if t > 4:
        nu = 4.0 + (t - 4.0) * (1.0 + (1.0 - 2.0 / t) / r) ** 2
    else:
        nu = t * (1.0 + 1.0 / q) * (1.0 + 1.0 / r) ** 2 / 2.0
    p_value = float(f.sf(stat / q, q, nu))
    return GrayTestResult(
        statistic=stat,
        df=q,
        p_value=p_value,
        cause=event,
        groups=groups,
        method=IMPUTED,
        n_imputations=M,
        df_denominator=float(nu),
        relative_increase=r,
    )
