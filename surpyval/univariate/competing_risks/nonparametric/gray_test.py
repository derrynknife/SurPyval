"""Gray's test for comparing cumulative incidence functions (#216).

Gray's (1988) k-sample test compares the **cumulative incidence functions**
of a given cause across groups -- the subdistribution analogue of the
log-rank test. It differs from a cause-specific log-rank in the risk set: a
subject who has already failed from a *competing* cause is kept in the
cause-``k`` subdistribution risk set, because for the subdistribution such a
subject simply never fails from cause ``k``.

Each group's subdistribution risk set is estimated from that group's own
data, ``R_g(t) = Y_g(t) (1 - F_g(t-)) / S_g(t-)``, with ``Y_g`` the number
at risk, ``F_g`` the group's Aalen-Johansen cumulative incidence of the
cause and ``S_g`` its all-cause Kaplan-Meier survival. ``Y_g / S_g`` is the
group's size times its censoring survival, so every group carries its own
censoring distribution: the test stays calibrated when the groups are
censored differently.

References
----------
- Gray (1988), "A class of K-sample tests for comparing the cumulative
  incidence of a competing risk", Annals of Statistics 16(3).
- Fine & Gray (1999) for the subdistribution risk set.
"""

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.stats import chi2

from surpyval.univariate.competing_risks.aalen_johansen import (
    aalen_johansen_iif,
)
from surpyval.univariate.competing_risks.labels import (
    label_mask,
    ordered_labels,
)
from surpyval.utils import (
    check_c_and_e,
    is_missing_event,
    resolve_cr_censoring,
)
from surpyval.utils.linalg import safe_quadform


class GrayTestResult(NamedTuple):
    statistic: float
    df: int
    p_value: float
    cause: Any
    groups: list


def gray_test(
    x: npt.ArrayLike,
    e: npt.ArrayLike,
    group: npt.ArrayLike,
    cause: Any,
    c: "npt.ArrayLike | None" = None,
    n: "npt.ArrayLike | None" = None,
    rho: float = 0.0,
) -> GrayTestResult:
    """
    Gray's k-sample test comparing the cumulative incidence of one cause
    across groups.

    Parameters
    ----------
    x : array_like
        Event/censoring times (finite).
    e : array_like
        Cause label per observation; a missing value (``None``, ``NaN`` or
        pandas ``NA``) marks a right-censored observation, as for the
        competing-risks model classes.
    group : array_like
        Group label per observation (two or more groups, no missing
        labels). Labels of different types (``0`` and ``"a"``) may be
        mixed.
    cause : scalar
        The cause whose cumulative incidence is compared across groups.
    c : array_like, optional
        Censoring flag (``0`` event, ``1`` right-censored; left and interval
        censoring are not supported). If omitted it is derived from ``e``
        (a missing cause is censored). If given, every row with ``c == 1``
        must have a missing cause and every other row a cause, otherwise a
        ``ValueError`` is raised.
    n : array_like, optional
        Count weight per observation (default 1), each positive.
    rho : float, optional
        Weight-family parameter: the per-time weight is ``(1 - F(t-))**rho``
        with ``F`` the pooled cumulative incidence of ``cause``. ``0``
        (the default) is the standard Gray test.

    Returns
    -------
    GrayTestResult
        ``(statistic, df, p_value, cause, groups)``; ``df`` is
        ``n_groups - 1`` and a small ``p_value`` is evidence the groups'
        cumulative incidence functions differ.

    Notes
    -----
    This is Gray's (1988) construction. Group ``g``'s subdistribution risk
    set is ``R_g(t) = Y_g(t) (1 - F_g(t-)) / S_g(t-)``, from the group's own
    at-risk count ``Y_g``, Aalen-Johansen incidence ``F_g`` of the cause and
    all-cause Kaplan-Meier survival ``S_g``; since ``Y_g / S_g`` estimates
    the group size times the group's censoring survival, each group's
    censoring is estimated separately, and the groups may be censored
    differently. The score is the weighted observed-minus-expected count
    ``sum_t w(t) (d_g(t) - R_g(t) d(t) / R(t))``, with ``d`` the failures
    from the cause and ``R = sum_g R_g``.

    Its variance is Gray's asymptotic estimate: the score is linearised in
    each group's counting-process martingales, of the cause *and* of the
    competing causes (the latter enter through ``F_g`` and ``S_g`` in the
    risk set), and the martingale variances are estimated by the observed
    failure counts. It is not the hypergeometric (log-rank) variance, which
    ignores the variability of the estimated risk sets. The pooled
    incidence in the variance and in the ``rho`` weight is Gray's
    ``F^0(t) = 1 - prod_{s <= t} (1 - d(s) / R(s))``. R's ``cmprsk``
    implements the same test; its numerical conventions at tied times may
    differ slightly.

    Examples
    --------
    Group 1 has twice group 0's hazard of cause ``a``:

    >>> import numpy as np
    >>> from surpyval import gray_test
    >>> rng = np.random.default_rng(0)
    >>> group = rng.binomial(1, 0.5, 200)
    >>> t_a = rng.exponential(1 / (0.1 * np.exp(0.7 * group)))
    >>> t_b = rng.exponential(1 / 0.05, 200)
    >>> t_c = rng.uniform(0, 20, 200)  # censoring times
    >>> x = np.minimum(np.minimum(t_a, t_b), t_c).round(3)
    >>> first = np.where(t_a < t_b, "a", "b")
    >>> e = np.where(t_c < np.minimum(t_a, t_b), None, first)
    >>> res = gray_test(x, e, group, cause="a")
    >>> round(res.statistic, 3), res.df
    (19.962, 1)
    >>> bool(res.p_value < 0.001)
    True
    """
    x_arr, n_arr, gi, groups, is_cause, is_competing, rho = _validate(
        x, e, group, cause, c, n, rho
    )
    K = len(groups)

    # Per-group counts on the pooled grid of distinct times, shape (K, T).
    times, t_idx = np.unique(x_arr, return_inverse=True)
    T = times.size
    counts = np.zeros((K, T))
    np.add.at(counts, (gi, t_idx), n_arr)
    d1 = np.zeros((K, T))
    np.add.at(d1, (gi[is_cause], t_idx[is_cause]), n_arr[is_cause])
    d2 = np.zeros((K, T))
    np.add.at(d2, (gi[is_competing], t_idx[is_competing]), n_arr[is_competing])
    # Everyone with x >= t is at risk at t.
    Y = counts[:, ::-1].cumsum(axis=1)[:, ::-1]
    at_risk = Y > 0
    Y_safe = np.where(at_risk, Y, 1.0)

    # Each group's all-cause Kaplan-Meier survival and Aalen-Johansen
    # incidences (of the cause, F1, and of the competing causes, F2).
    S = np.cumprod(1.0 - np.where(at_risk, (d1 + d2) / Y_safe, 0.0), axis=1)
    S_minus = _left_limit(S, 1.0)
    F1 = np.array(
        [aalen_johansen_iif(S[g], d1[g] / Y_safe[g]) for g in range(K)]
    ).cumsum(axis=1)
    F2 = np.array(
        [aalen_johansen_iif(S[g], d2[g] / Y_safe[g]) for g in range(K)]
    ).cumsum(axis=1)
    F1_minus = _left_limit(F1, 0.0)

    # Gray's subdistribution risk sets. S(t-) > 0 wherever anyone is still
    # at risk, so the ratio is only formed there.
    R = np.where(
        at_risk,
        Y * (1.0 - F1_minus) / np.where(at_risk, S_minus, 1.0),
        0.0,
    )
    R_tot = R.sum(axis=0)
    d_tot = d1.sum(axis=0)
    pos = R_tot > 0
    R_tot_safe = np.where(pos, R_tot, 1.0)

    # Gray's pooled subdistribution hazard increments and incidence F^0.
    dGamma = np.where(pos, d_tot / R_tot_safe, 0.0)
    F0 = 1.0 - np.cumprod(1.0 - dGamma)
    F0_minus = _left_limit(F0, 0.0)
    L = (1.0 - F0_minus) ** rho

    # Score: weighted observed minus expected failures from the cause.
    U = (L * (d1 - R * d_tot / R_tot_safe) * pos).sum(axis=1)

    # Variance. With c[t, k, r] = L (delta_kr R_k - R_k R_r / R), the score
    # is z_k = sum_r int c_kr dGamma_r, and linearising group r's estimated
    # subdistribution hazard Gamma_r in its martingales M1 (the cause) and
    # M2 (the competing causes) gives z_k ~ sum_r int A_kr dM1_r +
    # int B_kr dM2_r with
    #   A_kr(u) = c_kr(u) / R_r(u) - F2_r(u) Q_kr(u) / Y_r(u),
    #   B_kr(u) = -(1 - F^0(u)) Q_kr(u) / Y_r(u),
    #   Q_kr(u) = int_(u, inf) c_kr(t) dGamma(t) / (1 - F^0(t-)).
    # The competing causes enter because a group's risk set R_r depends on
    # its incidence and survival estimates. The martingale variances are
    # estimated by the observed failure counts d1 and d2.
    p = np.where(pos, R / R_tot_safe, 0.0)  # (K, T)
    eye = np.eye(K)
    c_tkr = L[:, None, None] * (
        R.T[:, :, None] * (eye[None, :, :] - p.T[:, None, :])
    )
    alive = 1.0 - F0_minus
    q = np.where(alive > 0, dGamma / np.where(alive > 0, alive, 1.0), 0.0)
    # Q at u sums the strictly later times: a reversed cumulative sum,
    # shifted by one.
    later = (c_tkr * q[:, None, None])[::-1].cumsum(axis=0)[::-1]
    Q = np.concatenate([later[1:], np.zeros((1, K, K))], axis=0)
    R_safe = np.where(R > 0, R, 1.0)
    A = np.where(
        (R > 0).T[:, None, :],
        c_tkr / R_safe.T[:, None, :] - (F2 / Y_safe).T[:, None, :] * Q,
        0.0,
    )
    B = np.where(
        at_risk.T[:, None, :],
        -((1.0 - F0)[:, None] / Y_safe.T)[:, None, :] * Q,
        0.0,
    )
    V = np.einsum("tkr,tlr,rt->kl", A, A, d1) + np.einsum(
        "tkr,tlr,rt->kl", B, B, d2
    )

    # Drop the last group for a full-rank (G-1) quadratic form.
    stat = safe_quadform(V[:-1, :-1], U[:-1])
    df = K - 1
    return GrayTestResult(
        statistic=stat,
        df=df,
        p_value=float(chi2.sf(stat, df=df)),
        cause=cause,
        groups=groups,
    )


def _left_limit(values: npt.NDArray, start: float) -> npt.NDArray:
    """The left limit ``f(t-)`` of a step function stored at the grid
    times along the last axis: shifted by one, with ``start`` first."""
    first = np.full(values.shape[:-1] + (1,), start)
    return np.concatenate([first, values[..., :-1]], axis=-1)


def _validate(
    x: npt.ArrayLike,
    e: npt.ArrayLike,
    group: npt.ArrayLike,
    cause: Any,
    c: "npt.ArrayLike | None",
    n: "npt.ArrayLike | None",
    rho: float,
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    list,
    npt.NDArray,
    npt.NDArray,
    float,
]:
    """Check the inputs; returns ``(x, n, group_index, groups, is_cause,
    is_competing, rho)``. A silently accepted bad input used to give a
    wrong statistic (a ``c = -1`` row counted as a failure, a NaN time
    dropped out of every risk set), so each is refused here."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("x must be a one-dimensional array of times.")
    N = x.size
    if N == 0:
        raise ValueError("Gray's test needs data; x is empty.")
    if not np.all(np.isfinite(x)):
        raise ValueError("x must be finite (no NaN or infinite times).")

    # The same missing-cause rule as every other competing-risks class: a
    # missing cause (None, NaN or pandas NA) is a censored row, and a given
    # ``c`` must agree with it.
    e_arr, c_arr = resolve_cr_censoring(e, c)
    if e_arr.shape != (N,):
        raise ValueError("e must have one cause per time in x.")
    c_arr = np.asarray(c_arr)
    if c_arr.shape != (N,):
        raise ValueError("c must have one censoring flag per time in x.")
    if not np.isin(c_arr, (0, 1)).all():
        raise ValueError(
            "c must be 0 (a failure) or 1 (right-censored); left and "
            "interval censoring are not supported by Gray's test."
        )
    check_c_and_e(c_arr, e_arr)
    censored = c_arr.astype(int) == 1

    if n is None:
        n_arr = np.ones(N)
    else:
        n_arr = np.asarray(n, dtype=float)
        if n_arr.shape != (N,):
            raise ValueError("n must have one count per time in x.")
        if not (np.all(np.isfinite(n_arr)) and np.all(n_arr > 0)):
            raise ValueError("n must be finite and positive.")

    g_arr = np.empty(N, dtype=object)
    # One label per row whatever it is (``np.asarray`` would split a tuple
    # label into a column).
    group_seq: Any = group
    g_list = list(group_seq) if np.ndim(group) else [group]
    if len(g_list) != N:
        raise ValueError("group must have one label per time in x.")
    for i, g in enumerate(g_list):
        g_arr[i] = g
    if any(is_missing_event(g) for g in g_arr):
        raise ValueError(
            "group has a missing label (None / NaN); every observation "
            "needs a group."
        )

    if not np.isfinite(rho):
        raise ValueError("rho must be a finite number.")

    is_cause = (~censored) & label_mask(e_arr, cause)
    is_competing = (~censored) & (~is_cause)
    if not is_cause.any():
        raise ValueError(f"No events of cause {cause!r} to compare.")

    groups = ordered_labels(g_arr)
    if len(groups) < 2:
        raise ValueError("Gray's test needs at least two groups.")
    grp_idx = {g: i for i, g in enumerate(groups)}
    gi = np.array([grp_idx[g] for g in g_arr], dtype=int)
    return x, n_arr, gi, groups, is_cause, is_competing, float(rho)
