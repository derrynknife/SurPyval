"""Gray's test for comparing cumulative incidence functions (#216).

Gray's (1988) k-sample test compares the **cumulative incidence functions**
of a given cause across groups -- the subdistribution analogue of the
log-rank test. It differs from a cause-specific log-rank in the risk set: a
subject who has already failed from a *competing* cause is kept in the
cause-``k`` subdistribution risk set (with an inverse-probability-of-censoring
weight), because for the subdistribution such a subject simply never fails
from cause ``k``.

References
----------
- Gray (1988), "A class of K-sample tests for comparing the cumulative
  incidence of a competing risk", Annals of Statistics 16(3).
- Fine & Gray (1999) for the subdistribution risk set / IPCW weighting.
"""

from typing import Any, NamedTuple

import numpy as np
from scipy.stats import chi2


class GrayTestResult(NamedTuple):
    statistic: float
    df: int
    p_value: float
    cause: Any
    groups: list


def _censoring_survival(x: np.ndarray, censored: np.ndarray, n: np.ndarray):
    """Kaplan-Meier estimate of the censoring-time survival ``G(t)`` on the
    pooled sample (treating censorings as the events), returned as the unique
    times and the right-continuous ``G`` at those times."""
    times = np.unique(x)
    G = np.ones(times.size)
    surv = 1.0
    for i, t in enumerate(times):
        at_risk = n[x >= t].sum()
        cens_here = n[(x == t) & censored].sum()
        if at_risk > 0:
            surv *= 1.0 - cens_here / at_risk
        G[i] = surv
    return times, G


def _G_at(times: np.ndarray, G: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Right-continuous ``G`` evaluated at query times ``q`` (1 before the
    first event time)."""
    idx = np.searchsorted(times, q, side="right") - 1
    out = np.where(idx < 0, 1.0, G[np.clip(idx, 0, times.size - 1)])
    return out


def gray_test(
    x,
    e,
    group,
    cause,
    c=None,
    n=None,
    rho: float = 0.0,
) -> GrayTestResult:
    """
    Gray's k-sample test comparing the cumulative incidence of one cause
    across groups.

    Parameters
    ----------
    x : array_like
        Event/censoring times.
    e : array_like
        Cause label per observation; ``None`` (or matching ``c == 1``) marks a
        censored observation.
    group : array_like
        Group label per observation (two or more groups).
    cause : scalar
        The cause whose cumulative incidence is compared across groups.
    c : array_like, optional
        Censoring flag (``0`` event, ``1`` censored). If omitted it is inferred
        from ``e`` being ``None``.
    n : array_like, optional
        Count weight per observation (default 1).
    rho : float, optional
        Weight-family parameter: the per-time weight is ``(1 - F(t^-))**rho``
        with ``F`` the pooled cumulative incidence of ``cause``. ``0``
        (the default) is the standard Gray test.

    Returns
    -------
    GrayTestResult
        ``(statistic, df, p_value, cause, groups)``; ``df`` is
        ``n_groups - 1`` and a small ``p_value`` is evidence the groups'
        cumulative incidence functions differ.
    """
    x = np.asarray(x, dtype=float)
    group = np.asarray(group)
    N = x.size
    n = np.ones(N) if n is None else np.asarray(n, dtype=float)

    e_arr = np.asarray(e, dtype=object)
    if c is None:
        censored = np.array([ei is None for ei in e_arr])
    else:
        censored = np.asarray(c).astype(int) == 1

    is_cause = (~censored) & (e_arr == cause)
    is_competing = (~censored) & (~is_cause)

    groups = list(np.unique(group))
    G_count = len(groups)
    if G_count < 2:
        raise ValueError("Gray's test needs at least two groups.")

    # Censoring-distribution survival for the IPCW subdistribution weights.
    ctimes, Gsurv = _censoring_survival(x, censored, n)
    G_at_x = _G_at(ctimes, Gsurv, x)  # G at each subject's own time

    # Pooled cumulative incidence of the cause, for the (optional) rho weight.
    cause_times = np.unique(x[is_cause])
    if cause_times.size == 0:
        raise ValueError(f"No events of cause {cause!r} to compare.")

    # Pooled CIF at each cause failure time for the weight (Aalen-Johansen).
    all_event = ~censored
    F_pooled = _pooled_cif(x, all_event, is_cause, n, cause_times)
    weights = (1.0 - np.concatenate([[0.0], F_pooled[:-1]])) ** rho

    G_at_tau = _G_at(ctimes, Gsurv, cause_times)

    grp_idx = {g: i for i, g in enumerate(groups)}
    gi = np.array([grp_idx[g] for g in group])

    U = np.zeros(G_count)
    V = np.zeros((G_count, G_count))

    for m, tau in enumerate(cause_times):
        # Subdistribution at-risk weight per subject at tau:
        #  1 if still at risk (x >= tau);
        #  G(tau)/G(x_j) if already failed from a competing cause;
        #  0 if it failed from this cause or was censored before tau.
        w = np.where(x >= tau, 1.0, 0.0)
        comp_before = is_competing & (x < tau)
        with np.errstate(divide="ignore", invalid="ignore"):
            ipcw = np.where(G_at_x > 0, G_at_tau[m] / G_at_x, 0.0)
        w = np.where(comp_before, ipcw, w)
        wn = w * n

        # Risk set per group and cause-k events per group at tau.
        Rg = np.zeros(G_count)
        np.add.at(Rg, gi, wn)
        R = Rg.sum()
        event_here = is_cause & (x == tau)
        dg = np.zeros(G_count)
        np.add.at(dg, gi[event_here], n[event_here])
        d = dg.sum()
        if R <= 0 or d <= 0:
            continue

        wt = weights[m]
        # Observed minus expected cause-k events per group.
        U += wt * (dg - d * Rg / R)
        # Multivariate hypergeometric variance with the weighted risk sets.
        if R > 1:
            var_scale = wt**2 * d * (R - d) / (R - 1)
        else:
            var_scale = 0.0
        p = Rg / R
        V += var_scale * (np.diag(p) - np.outer(p, p))

    # Drop the last group for a full-rank (G-1) quadratic form.
    U_r = U[:-1]
    V_r = V[:-1, :-1]
    try:
        stat = float(U_r @ np.linalg.solve(V_r, U_r))
    except np.linalg.LinAlgError:
        stat = float(U_r @ np.linalg.pinv(V_r) @ U_r)
    df = G_count - 1
    return GrayTestResult(
        statistic=stat,
        df=df,
        p_value=float(chi2.sf(stat, df=df)),
        cause=cause,
        groups=groups,
    )


def _pooled_cif(x, all_event, is_cause, n, cause_times) -> np.ndarray:
    """Aalen-Johansen cumulative incidence of the cause on the pooled sample,
    evaluated at ``cause_times``."""
    times = np.unique(x)
    surv = 1.0
    cif = 0.0
    out = {}
    for t in times:
        at_risk = n[x >= t].sum()
        d_all = n[all_event & (x == t)].sum()
        d_k = n[is_cause & (x == t)].sum()
        if at_risk > 0:
            cif += surv * d_k / at_risk
            surv *= 1.0 - d_all / at_risk
        out[t] = cif
    return np.array([out[t] for t in cause_times])
