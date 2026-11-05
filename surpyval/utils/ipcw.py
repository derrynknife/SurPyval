"""
The inverse-probability-of-censoring-weighting (IPCW) toolkit: the
Kaplan-Meier estimate of the *censoring* distribution and the step
lookups used to evaluate it (right-continuous, ``step_at``, and its left
limit, ``step_left_limit``).

Three modules -- Gray's test, Fine-Gray regression and the prediction
metrics -- each carried their own copy of both functions. The copies had
already started to drift: the metrics copy silently ignored count
weights, consistent with its callers but a trap for the next reuse. This
module is the single, weighted implementation; passing no ``n`` is the
unweighted case.

The bodies are transplants from the competing-risks copies, kept
bit-identical so consolidating changed no fitted numbers. The callers do
differ in one convention, how an event and a censoring at the same time
are ordered, which :func:`censoring_survival` selects with ``ties``:
Fine-Gray regression uses the default (``cmprsk::crr``'s), the
prediction metrics use ``"events_first"`` (``prodlim``'s and
scikit-survival's reverse Kaplan-Meier).
"""

import numpy as np
import numpy.typing as npt

from surpyval.utils.validation import check_option


def censoring_survival(
    x: npt.NDArray,
    censored: npt.NDArray,
    n: "npt.NDArray | None" = None,
    ties: str = "censoring_first",
) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Kaplan-Meier estimate of the censoring survival ``G(t) = P(C > t)``.

    The roles are reversed relative to an ordinary survival fit:
    right-censored rows (``censored`` true) are the "events" for the
    censoring distribution and observed events are treated as censored.
    Returns the sorted unique times and the right-continuous ``G``
    evaluated at each.

    Parameters
    ----------
    x : ndarray
        Observed times.
    censored : ndarray of bool
        True where the observation was right-censored.
    n : ndarray, optional
        Count weight per observation; default 1 each.
    ties : {"censoring_first", "events_first"}, optional
        Which risk set a censoring at ``t`` is measured against when an
        event also falls at ``t``.

        * ``"censoring_first"`` (default): everyone with ``x >= t``,
          including the events at ``t`` -- they count as still at risk of
          being censored at ``t``. This is the convention of
          ``cmprsk::crr``, which Fine-Gray regression reproduces.
        * ``"events_first"``: everyone with ``x > t`` plus those censored
          at ``t``. An event at ``t`` is taken to precede a censoring at
          ``t`` (the convention the data themselves follow: a tie is
          recorded as an event), so it is no longer at risk of censoring.
          This is the reverse Kaplan-Meier of ``prodlim`` (``reverse =
          TRUE``) and scikit-survival, used by the prediction metrics in
          :mod:`surpyval.metrics.validation`.

        The two agree unless an event and a censoring share a time.

    Examples
    --------
    An event and a censoring both at ``t = 2``:

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 2.0, 3.0])
    >>> censored = np.array([False, False, True, False])
    >>> censoring_survival(x, censored)[1]  # 1 - 1/3 at t = 2
    array([1.        , 0.66666667, 0.66666667])
    >>> censoring_survival(x, censored, ties="events_first")[1]  # 1 - 1/2
    array([1. , 0.5, 0.5])
    """
    check_option("ties", ties, ("censoring_first", "events_first"))
    x = np.asarray(x, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    if n is None:
        n = np.ones(x.size)
    # A nan time is at risk at no time (every comparison with it is false).
    n = np.where(np.isnan(x), 0.0, np.asarray(n, dtype=float))
    # The counts at each distinct time, and those at or after it as a
    # suffix sum: O(N log N), where a sum over the rows at each time was
    # O(N x times), a minute at 1e5 rows (#517). Exact for integer counts,
    # so G is unchanged.
    times, inv = np.unique(x, return_inverse=True)
    cens_here = np.bincount(inv, np.where(censored, n, 0.0), times.size)
    at_or_after = np.cumsum(np.bincount(inv, n, times.size)[::-1])[::-1]
    if ties == "events_first":
        after = np.append(at_or_after[1:], 0.0)
        at_risk = after + cens_here
    else:
        at_risk = at_or_after
    # A time nobody is at risk at leaves G as it is.
    factor = 1.0 - np.divide(
        cens_here, at_risk, out=np.zeros(times.size), where=at_risk > 0
    )
    return times, np.cumprod(factor)


def step_at(
    times: npt.NDArray,
    values: npt.NDArray,
    query: npt.ArrayLike,
    before: float,
) -> npt.NDArray:
    """
    Right-continuous step function: the value carried by the largest
    ``times`` entry ``<= query``; ``before`` is returned where ``query``
    precedes the first time. For a survival curve ``before`` is 1, for a
    cumulative hazard it is 0.
    """
    idx = np.searchsorted(times, query, side="right") - 1
    return np.where(idx < 0, before, values[np.clip(idx, 0, values.size - 1)])


def step_left_limit(
    times: npt.NDArray,
    values: npt.NDArray,
    query: npt.ArrayLike,
    before: float,
) -> npt.NDArray:
    """
    The left limit ``f(query-)`` of the step function :func:`step_at`
    evaluates: the value carried by the largest ``times`` entry strictly
    ``< query``; ``before`` is returned where no time precedes ``query``.

    It differs from :func:`step_at` only where ``query`` is one of
    ``times``: there it is the value before that step rather than after
    it. Fine-Gray regression evaluates the censoring survival this way,
    ``G(t-)``, so that censorings at a time do not yet count against an
    event at the same time (see
    :mod:`surpyval.univariate.competing_risks.regression.fine_gray`).

    Examples
    --------
    >>> import numpy as np
    >>> times, values = np.array([1.0, 2.0]), np.array([0.8, 0.5])
    >>> step_at(times, values, [1.0, 1.5, 2.0], before=1.0)
    array([0.8, 0.8, 0.5])
    >>> step_left_limit(times, values, [1.0, 1.5, 2.0], before=1.0)
    array([1. , 0.8, 0.8])
    """
    idx = np.searchsorted(times, query, side="left") - 1
    return np.where(idx < 0, before, values[np.clip(idx, 0, values.size - 1)])


def ipcw_censoring_influence(
    x: npt.NDArray,
    censored: npt.NDArray,
    coef: npt.NDArray,
    at: npt.NDArray,
    strict: npt.NDArray,
) -> npt.NDArray:
    r"""
    The censoring part of the influence function of an IPCW sum.

    An IPCW statistic ``sum_j coef_j / G(u_j)`` (scaled by the caller)
    moves with the Kaplan-Meier estimate ``G`` of the censoring
    distribution. Its influence through ``G`` at subject ``k`` is

    .. math::
        \sum_j \mathrm{coef}_j\, f_k(u_j), \qquad
        f_k(u) = \int_0^u \frac{dM^C_k(s)}{y(s)},

    where ``M^C_k`` is subject ``k``'s censoring martingale, ``y(s)`` the
    proportion still under observation at ``s`` and ``dM^C_k(s) =
    dN^C_k(s) - I(x_k \ge s)\, d\Lambda_C(s)``: the influence of
    ``-log G(u)``, so of ``1 / G(u)`` divided by ``1 / G(u)``. ``u_j`` is
    ``at[j]``, or its left limit ``at[j]-`` where ``strict[j]`` (the
    weight of an event, ``1 / G(x_j-)``).

    The at-risk counts and hazard are those of ``riskRegression``'s
    Kaplan-Meier censoring term (everyone with ``x >= s`` at risk at
    ``s``), so the standard errors agree with ``Score(se.fit = TRUE)``.
    O(N log N).

    Parameters
    ----------
    x : ndarray
        Observed times.
    censored : ndarray of bool
        True where the observation was right-censored.
    coef : ndarray
        The coefficient of each term ``j`` (0 for a row that does not
        enter the statistic).
    at : ndarray
        The time ``u_j`` at which term ``j``'s weight evaluates ``G``.
    strict : ndarray of bool
        True where the weight is ``1 / G(u_j-)`` rather than
        ``1 / G(u_j)``.

    Returns
    -------
    influence : ndarray
        One value per subject ``k``.

    Examples
    --------
    One weight ``1 / G(2)`` with censorings at 1 and 3 (four subjects, so
    the compensator to 2 is ``M(2) = 4 * (1/4) / 4 = 1/4``): the subject
    censored at 1 raises it by its own jump ``4 / 4`` less ``M(1) = 1/4``,
    the others lower it by ``M`` up to 2:

    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0])
    >>> censored = np.array([True, False, True, False])
    >>> coef = np.array([0.0, 0.0, 0.0, 1.0])
    >>> ipcw_censoring_influence(x, censored, coef, np.full(4, 2.0),
    ...                          np.zeros(4, bool))
    array([ 0.75, -0.25, -0.25, -0.25])
    """
    x = np.asarray(x, dtype=float)
    censored = np.asarray(censored, dtype=bool)
    coef = np.asarray(coef, dtype=float)
    at = np.asarray(at, dtype=float)
    strict = np.asarray(strict, dtype=bool)
    n = x.size
    times, inv = np.unique(x, return_inverse=True)
    at_risk = np.cumsum(np.bincount(inv, minlength=times.size)[::-1])[::-1]
    cens_here = np.bincount(inv, censored.astype(float), times.size)
    # M(s) = sum_{s' <= s} n dLambda_C(s') / y(s'), y = at_risk / n.
    big_m = np.cumsum(n * cens_here / at_risk.astype(float) ** 2)
    m_x = big_m[inv]
    # M at each term's time: M(u) or its left limit M(u-).
    m_u = np.where(
        strict,
        step_left_limit(times, big_m, at, before=0.0),
        step_at(times, big_m, at, before=0.0),
    )
    jump = np.where(censored, n / at_risk[inv].astype(float), 0.0)

    # Subject k is "before" term j when x_k < u_j (strict) or x_k <= u_j:
    # its own censoring jump counts and its compensator stops at x_k.
    # Otherwise the compensator runs to u_j (or u_j-).
    out = np.zeros(n)
    for flag in (True, False):
        sel = (strict == flag) & (coef != 0)
        if not sel.any():
            continue
        order = np.argsort(at[sel], kind="stable")
        u_sorted = at[sel][order]
        c_sorted = coef[sel][order]
        cm_sorted = (coef[sel] * m_u[sel])[order]
        # Terms with u_j > x_k (strict) or u_j >= x_k: x_k is before them.
        if flag:
            first = np.searchsorted(u_sorted, x, side="right")
        else:
            first = np.searchsorted(u_sorted, x, side="left")
        c_suffix = np.append(np.cumsum(c_sorted[::-1])[::-1], 0.0)
        cm_prefix = np.concatenate(([0.0], np.cumsum(cm_sorted)))
        before_sum = c_suffix[first]
        out += (jump - m_x) * before_sum - cm_prefix[first]
    return out
