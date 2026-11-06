"""Shared Aalen-Johansen incidence increment (#299).

The S(t-) weighting that turns cause-specific hazard increments into
cumulative-incidence increments was implemented three times (the
nonparametric CIF, the CR-PH ``cif`` and Gray's pooled CIF), and one of
those copies went wrong (#278). It now lives here once.
"""

import numpy as np
import numpy.typing as npt


def aalen_johansen_iif(
    S: npt.NDArray, hazard_increments: npt.NDArray
) -> npt.NDArray:
    """Instantaneous incidence increments.

    The cause-specific hazard increment at ``t_i`` acts on the
    population still alive just *before* ``t_i``, so each increment is
    weighted by ``S(t_i-)`` — the survival after the previous event
    time — not ``S(t_i)`` (#253). ``S`` and the last axis of
    ``hazard_increments`` must be aligned on the same event-time grid;
    the cumulative incidence is the cumulative sum of the result.

    Parameters
    ----------
    S : array_like
        The all-cause (Kaplan-Meier) survival at each event time.
    hazard_increments : array_like
        The cause-specific hazard increments ``d_j / r`` at the same
        times, one row per cause (or a single 1-D row).

    Returns
    -------
    numpy array
        ``hazard_increments * S(t-)``, the same shape as
        ``hazard_increments``.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.univariate.competing_risks.aalen_johansen import (
    ...     aalen_johansen_iif,
    ... )
    >>> S = np.array([0.9, 0.8, 0.6])
    >>> aalen_johansen_iif(S, np.array([0.1, 0.0, 0.25]))
    array([0.1, 0. , 0.2])
    """
    S = np.asarray(S, dtype=float)
    S_prev = np.concatenate([[1.0], S[:-1]])
    return np.asarray(hazard_increments) * S_prev


def aalen_johansen_variance(
    r: npt.ArrayLike, d: npt.ArrayLike, d_k: npt.ArrayLike
) -> npt.NDArray:
    r"""Variance of the Aalen-Johansen cumulative incidence of one cause.

    Aalen's (1978) estimate of the asymptotic variance, as R's
    ``cmprsk::cuminc`` reports it (``var``; it is the same number). With
    :math:`\hat{S}` the all-cause Kaplan-Meier survival, :math:`\hat{F}`
    the cause's cumulative incidence and :math:`\hat{F}_j = \hat{F}(x_j)`,

    .. math::
        \widehat{Var}\,\hat{F}(t) = \sum_{x_j \le t}
            \frac{\hat{S}(x_{j-1})^2}{r_j^2}\left[
            c(d_{k,j})\, d_{k,j}
            \left(1 + \frac{\hat{F}_j - \hat{F}(t)}{\hat{S}(x_j)}\right)^2
            + c(d_j - d_{k,j})\,(d_j - d_{k,j})
            \left(\frac{\hat{F}_j - \hat{F}(t)}{\hat{S}(x_j)}\right)^2
            \right],

    where :math:`c(m) = (r_j - m) / (r_j - 1)` for :math:`m > 1` tied
    events (1 otherwise) corrects for ties. Where :math:`\hat{S}(x_j) = 0`
    (the last event, with no one left), cmprsk's convention is kept: the
    other causes' term is left out and the cause's is
    :math:`\hat{S}(x_{j-1})^2 c\, d_{k,j} / r_j^2`.

    Parameters
    ----------
    r : array_like
        The number at risk at each distinct time.
    d : array_like
        The number of events of any cause at each time.
    d_k : array_like
        The number of events of the cause at each time.

    Returns
    -------
    numpy array
        The variance of the cause's cumulative incidence at each time; 0
        before its first event.

    References
    ----------
    Aalen, O. (1978), "Nonparametric estimation of partial transition
    probabilities in multiple decrement models", *The Annals of
    Statistics*, 6(3), 534-545.

    Examples
    --------
    Three units, failing from causes a, b and a:

    >>> import numpy as np
    >>> from surpyval.univariate.competing_risks.aalen_johansen import (
    ...     aalen_johansen_variance,
    ... )
    >>> r, d, d_a = np.array([3, 2, 1]), np.array([1, 1, 1]), [1, 0, 1]
    >>> aalen_johansen_variance(r, d, d_a).round(4)
    array([0.1111, 0.1111, 0.25  ])
    """
    r = np.asarray(r, dtype=float)
    d = np.asarray(d, dtype=float)
    d_k = np.asarray(d_k, dtype=float)
    d_o = d - d_k
    with np.errstate(divide="ignore", invalid="ignore"):
        # Censoring-only times (d = 0) leave the survival as it was.
        S = np.cumprod(np.where(d > 0, 1.0 - d / r, 1.0))
        a = np.where(S > 0, 1.0 / S, 0.0)

        def ties(m: npt.NDArray) -> npt.NDArray:
            return np.where(m > 1, (r - m) / np.maximum(r - 1.0, 1.0), 1.0)

        S_prev = np.concatenate([[1.0], S[:-1]])
        w_k = np.where(d_k > 0, ties(d_k) * d_k * S_prev**2 / r**2, 0.0)
        w_o = np.where(
            (d_o > 0) & (S > 0), ties(d_o) * d_o * S_prev**2 / r**2, 0.0
        )
    F = aalen_johansen_iif(S, np.where(d_k > 0, d_k / np.maximum(r, 1), 0.0))
    F = F.cumsum()
    # sum_j w_j (b_j - F(t) a_j)^2, expanded so that it is three running
    # sums: b_j is 1 + F_j a_j for the cause's events and F_j a_j for the
    # others'.
    b_k = 1.0 + F * a
    b_o = F * a
    c0 = np.cumsum(w_k * b_k**2 + w_o * b_o**2)
    c1 = np.cumsum((w_k * b_k + w_o * b_o) * a)
    c2 = np.cumsum((w_k + w_o) * a**2)
    var = c0 - 2.0 * F * c1 + F**2 * c2
    # Zero before the cause's first event (and never negative from the
    # round-off of the expansion).
    return np.where(F > 0, np.maximum(var, 0.0), 0.0)


def weighted_aalen_johansen(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    n: npt.ArrayLike,
    w: npt.ArrayLike,
    cause: npt.ArrayLike,
    n_causes: int,
    t: npt.ArrayLike | None = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    r"""The Aalen-Johansen cumulative incidences of weighted data and
    their robust (infinitesimal jackknife) variances.

    Each unit counts ``n * w`` in the numbers at risk and of events, and
    the variance of cause :math:`k`'s cumulative incidence at :math:`t` is
    :math:`\sum_i n_i w_i^2 U_{ik}(t)^2`, with :math:`U_{ik}(t)` the
    derivative of the estimate in unit :math:`i`'s weight (its influence),

    .. math::
        U_{ik}(t) = \sum_{t_j \le t} \hat{S}(t_{j-1}) \left[
            \frac{dN_{ik}(t_j) - Y_i(t_j)\, h_{jk}}{r_j}
            + h_{jk}\, V_i(t_{j-1}) \right],

    where :math:`h_{jk} = d_{jk} / r_j` is the cause's hazard increment,
    :math:`Y_i(t_j)` is 1 while unit :math:`i` is at risk (from its entry,
    :math:`tl_i < t_j`, to its exit, :math:`t_j \le x_i`), :math:`dN_{ik}`
    is 1 at its failure from cause :math:`k`, and :math:`V_i` is the unit's
    influence on the log of the all-cause Kaplan-Meier survival
    (``_weighted.WeightedLadder``). This is R's ``survfit(Surv(x,
    factor(e)) ~ 1, weights = w)$std.err`` squared, and with entry times
    that of ``survfit(Surv(tl, x, factor(e)) ~ 1, weights = w)``.

    Parameters
    ----------
    x : array_like
        The units' times.
    c : array_like
        Their censoring flags: 0 a failure, 1 right censored.
    n : array_like
        Their counts.
    w : array_like
        Their weights.
    cause : array_like
        The index of each unit's cause (anything for a censored unit).
    n_causes : int
        The number of causes.
    t : array_like, optional
        The ``(N, 2)`` truncation bounds of the units: the first column is
        each unit's entry time (left truncation), the second must be
        infinite. ``None`` (the default) is no truncation.

    Returns
    -------
    tuple of numpy arrays
        ``(x, r, d, d_e, var)``: the distinct times, the weighted numbers
        at risk and of events (all causes, and one row per cause) at each,
        and the robust variance of each cause's cumulative incidence (one
        row per cause).

    Examples
    --------
    Two units fail from causes 0 and 1; a weight of 2 counts as two units
    in the estimate:

    >>> from surpyval.univariate.competing_risks.aalen_johansen import (
    ...     weighted_aalen_johansen,
    ... )
    >>> x, r, d, d_e, var = weighted_aalen_johansen(
    ...     [1, 2, 3], [0, 0, 1], [1, 1, 1], [2, 1, 1], [0, 1, 0], 2
    ... )
    >>> r, d_e[0]
    (array([4., 2., 1.]), array([2., 0., 0.]))
    """
    from surpyval.univariate.nonparametric._weighted import WeightedLadder

    x = np.asarray(x, dtype=float)
    c = np.asarray(c)
    n = np.asarray(n, dtype=float)
    w = np.asarray(w, dtype=float)
    cause = np.asarray(cause, dtype=np.intp)
    bounds = (
        np.column_stack([np.full(x.size, -np.inf), np.full(x.size, np.inf)])
        if t is None
        else np.asarray(t, dtype=float)
    )
    ladder = WeightedLadder(x, c, n, bounds, w)
    xs, r, d = ladder.x, ladder.r, ladder.d
    m = xs.size
    died = c == 0
    d_e = np.zeros((n_causes, m))
    np.add.at(d_e, (cause[died], ladder.exit[died]), (n * w)[died])
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.cumprod(np.where(d > 0, 1.0 - d / r, 1.0))
        h_e = np.where(d_e > 0, d_e / r, 0.0)
    S_prev = np.concatenate([[1.0], S[:-1]])
    alive = S_prev > 0
    var = np.zeros((n_causes, m))
    factor = n * w**2
    cols = np.arange(m)[None, :]
    for rows, V in ladder.influence_blocks("Kaplan-Meier"):
        V_prev = np.column_stack([np.zeros(rows.size), V[:, :-1]])
        e = ladder.exit[rows][:, None]
        # At risk from the first time after its entry to its exit.
        at_risk = (cols >= ladder.entry[rows][:, None]) & (cols <= e)
        failed = died[rows][:, None] & (cols == e)
        for k in range(n_causes):
            jumps = failed & (cause[rows][:, None] == k)
            with np.errstate(all="ignore"):
                term = (jumps - at_risk * h_e[k]) / r + h_e[k] * V_prev
                term = np.where(alive, S_prev * term, 0.0)
            U = np.cumsum(term, axis=1)
            var[k] += np.sum(factor[rows][:, None] * U**2, axis=0)
    return xs, r, d, d_e, var
