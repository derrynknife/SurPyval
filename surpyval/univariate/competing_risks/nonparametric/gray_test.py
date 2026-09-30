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
    event: Any,
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
    event : scalar
        The cause (a label in ``e``) whose cumulative incidence is compared
        across groups. The result keeps it as ``cause``.
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
        with ``F`` the pooled cumulative incidence of ``event``. ``0``
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

    Its variance is Gray's asymptotic estimate, computed as R's
    ``cmprsk::cuminc`` computes it, and the statistic agrees with
    ``cuminc``'s to rounding, ties included (#380). The score is linearised
    in each group's counting-process martingales, of the cause *and* of
    the competing causes (the latter enter through ``F_g`` and ``S_g`` in
    the risk set), under the null hypothesis: the martingale variances are
    the failures each group is expected to have from the cause, and those
    it had from the competing causes, each with a correction for tied
    failures. It is not the hypergeometric (log-rank) variance, which
    ignores the variability of the estimated risk sets. The pooled
    incidence ``F^0`` in the ``rho`` weight and the variance steps by
    ``d(t) / sum_g Y_g(t) / S_g(t-)``, the failures over the whole
    sample's censoring-weighted size. ``cmprsk`` forms ``R_g`` the same
    way: ``Y_g(t)`` counts the rows censored at ``t`` and ``S_g`` is taken
    just before ``t``, so ``Y_g(t) / S_g(t-)`` is the group size times its
    censoring survival just before ``t`` -- a censoring tied with a failure
    counts after it, as in ``FineGray``'s weights. Counts ``n`` enter as
    frequency weights (``cmprsk`` has none).

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
    >>> res = gray_test(x, e, group, event="a")
    >>> round(res.statistic, 3), res.df
    (20.113, 1)
    >>> bool(res.p_value < 0.001)
    True
    """
    x_arr, n_arr, gi, groups, is_cause, is_competing, rho = _validate(
        x, e, group, event, c, n, rho
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

    U, V = _score_and_variance(Y, d1, d2, rho)
    # Drop the last group for a full-rank (G-1) quadratic form.
    stat = safe_quadform(V[:-1, :-1], U[:-1])
    df = K - 1
    return GrayTestResult(
        statistic=stat,
        df=df,
        p_value=float(chi2.sf(stat, df=df)),
        cause=event,
        groups=groups,
    )


def _score_and_variance(
    Y: npt.NDArray, d1: npt.NDArray, d2: npt.NDArray, rho: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """Gray's score and its variance, as R's ``cmprsk`` computes them.

    ``Y``, ``d1`` and ``d2`` are, per group (rows) and distinct time
    (columns), the number at risk and the failures from the cause and from
    the competing causes. Each step is ``cmprsk``'s ``crst`` routine (read
    from its compiled code), vectorised over the times; the two agree to
    rounding (#380).
    """
    K = Y.shape[0]
    at_risk = Y > 0
    Y_safe = np.where(at_risk, Y, 1.0)
    # Each group's all-cause Kaplan-Meier survival (S, and S_minus just
    # before each time) and Aalen-Johansen incidence of the cause (F1).
    S = np.cumprod(1.0 - np.where(at_risk, (d1 + d2) / Y_safe, 0.0), axis=1)
    S_minus = _left_limit(S, 1.0)
    S_minus_safe = np.where(at_risk, S_minus, 1.0)
    F1_minus = _left_limit(
        np.cumsum(S_minus * np.where(at_risk, d1 / Y_safe, 0.0), axis=1), 0.0
    )

    # Gray's subdistribution risk sets R_g = Y_g (1 - F1_g(t-)) / S_g(t-).
    # Y_g / S_g(t-) (``size``) is the group size times its censoring
    # survival just before t.
    size = np.where(at_risk, Y / S_minus_safe, 0.0)
    R = size * (1.0 - F1_minus)
    R_tot, size_tot = R.sum(axis=0), size.sum(axis=0)
    d_tot = d1.sum(axis=0)
    size_safe = np.where(size_tot > 0, size_tot, 1.0)
    # The pooled incidence of the cause, F0, steps by d / sum_g Y_g / S_g:
    # the failures over the censoring-weighted size of the whole sample.
    dF0 = np.where(size_tot > 0, d_tot / size_safe, 0.0)
    F0 = np.cumsum(dF0)
    F0_minus = _left_limit(F0, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        L = (1.0 - F0_minus) ** rho

    # Score: weighted observed minus expected failures from the cause.
    pos = R_tot > 0
    R_tot_safe = np.where(pos, R_tot, 1.0)
    U = (L * (d1 - R * d_tot / R_tot_safe) * pos).sum(axis=1)

    # Variance. The score is linearised in each group's counting-process
    # martingales, of the cause and of the competing causes (these enter
    # through the estimated risk sets), under the null hypothesis: every
    # group's incidence is F0 and its expected failures from the cause at a
    # time are its share of d. With share size_r / size, at each time
    #   a_kr = W size_k (delta_kr - size_r / size),
    #   Q_kr(u) = sum_{t > u} a_kr(t) dF0(t) / (1 - F0(t-)),
    #   A_kr = a_kr + (1 - (1 - F0) / S_r) Q_kr,  B_kr = (1 - F0) / S_r Q_kr,
    # and V_kl = sum_u sum_r A_kr A_lr w1_r + B_kr B_lr w2_r, the weights
    # being the variances of the martingale steps: w1_r = d dF0 / size_r
    # and w2_r = d2_r (S_r(u-) / Y_r)^2, each with a correction for tied
    # failures.
    share = size / size_safe
    eye = np.eye(K)
    both = at_risk.T[:, :, None] & at_risk.T[:, None, :]
    a = np.where(
        both,
        (L[:, None] * size.T)[:, :, None] * (eye[None] - share.T[:, None, :]),
        0.0,
    )
    alive = 1.0 - F0_minus
    ok = (size_tot > 0) & (alive != 0)
    q = np.where(ok, dF0 / np.where(ok, alive, 1.0), 0.0)
    C = np.cumsum(a * q[:, None, None], axis=0)
    Q = C[-1][None] - C
    S_safe = np.where(S > 0, S, 1.0)
    alive_now = (1.0 - F0)[None, :] / S_safe
    tied1 = d_tot[None, :] > 1
    spread = size_tot[None, :] * S_minus - 1.0
    tie1 = np.where(
        tied1,
        1.0 - (d_tot[None, :] - 1.0) / np.where(tied1, spread, 1.0),
        1.0,
    )
    w1 = np.where(
        at_risk & (d_tot > 0)[None, :],
        dF0[None, :] * tie1 / np.where(at_risk, size, 1.0),
        0.0,
    )
    A = a + np.where(S > 0, 1.0 - alive_now, 1.0).T[:, None, :] * Q
    tied2 = d2 > 1
    tie2 = np.where(
        tied2, 1.0 - (d2 - 1.0) / np.where(tied2, Y - 1.0, 1.0), 1.0
    )
    w2 = np.where((S > 0) & (d2 > 0), d2 * tie2 * (S_minus / Y_safe) ** 2, 0.0)
    B = np.where(S > 0, alive_now, 0.0).T[:, None, :] * Q
    V = np.einsum("tkr,tlr,rt->kl", A, A, w1) + np.einsum(
        "tkr,tlr,rt->kl", B, B, w2
    )
    return U, V


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
