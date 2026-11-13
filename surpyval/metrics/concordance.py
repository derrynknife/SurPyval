"""Harrell's and Uno's concordance indices, in O(n log n) (#512).

The index compares every usable pair of subjects: the one that failed
first should carry the higher risk score. Counting the pairs one by one
costs O(n^2) (about 5 minutes at 50,000 subjects); here the subjects are
sorted by time and the pairs are counted with a merge sort over the ranked
scores, as lifelines and scikit-survival do, in O(n log^2 n) array
operations. The tie conventions, Therneau's (the default, as R's
``survival::concordance`` and lifelines) or Harrell's original, are those
of the pairwise definitions (see :func:`concordance_index`), which the
tests keep as the oracle.

:func:`concordance_estimate` adds Uno's inverse-probability-of-censoring
weighted index, truncation at a horizon ``tau``, and the
infinitesimal-jackknife standard error R's ``survival::concordance``
reports;
:func:`compare_concordance` compares two scores on the same subjects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.utils import validate_1d as _as_1d
from surpyval.utils.ipcw import censoring_survival, step_left_limit
from surpyval.utils.validation import check_option

__all__ = [
    "ConcordanceComparison",
    "ConcordanceResult",
    "compare_concordance",
    "concordance_estimate",
    "concordance_index",
]

#: The conventions for two events at the same time (``ties=``).
TIES = ("therneau", "harrell")

#: How the pairs are weighted (``weighting=``).
WEIGHTINGS = ("harrell", "uno")

#: The scales of the confidence interval (``ci_method=``).
CI_METHODS = ("normal", "logit")


def _interval(
    estimate: float, se: float, alpha: float, ci_method: str
) -> tuple[float, float]:
    """The ``1 - alpha`` interval of an index on its own scale
    (``"normal"``) or the log-odds scale (``"logit"``, by the delta
    method: ``se / (C (1 - C))``), which keeps it inside (0, 1)."""
    z = norm.ppf(1 - alpha / 2)
    if ci_method == "normal":
        return estimate - z * se, estimate + z * se
    if not 0 < estimate < 1:
        # A perfect (or perfectly wrong) ranking, or nan: no log-odds
        # scale, and the jackknife spread is 0 there.
        return estimate, estimate
    centre = np.log(estimate / (1 - estimate))
    half = z * se / (estimate * (1 - estimate))
    lower, upper = 1 / (1 + np.exp(-(centre + np.array([-half, half]))))
    return float(lower), float(upper)


# math.isclose's default relative tolerance, which the pairwise definition
# used together with ``tie_tol``.
_REL_TOL = 1e-9


def _close(a: npt.NDArray, b: npt.NDArray, tol: float) -> npt.NDArray:
    """``math.isclose(a, b, abs_tol=tol)``, elementwise: equal infinities
    are close, and an infinity is close to nothing else (its relative
    bound would be infinite)."""
    with np.errstate(invalid="ignore"):
        bound = np.maximum(_REL_TOL * np.maximum(np.abs(a), np.abs(b)), tol)
        finite = np.isfinite(a) & np.isfinite(b)
        return (a == b) | (finite & (np.abs(a - b) <= bound))


def _close_ranks(u: npt.NDArray, tol: float) -> tuple[npt.NDArray, ...]:
    """For each distinct score ``u[k]`` (``u`` sorted), the lowest and the
    highest rank whose score is within the tie tolerance of it.

    Closeness is not transitive, so it is not a partition of the scores;
    but for a fixed ``u[k]`` the scores close to it form a run of ranks
    around ``k``, whose ends are found by a search and a few steps.
    """
    m = u.size
    k = np.arange(m)
    reach = np.maximum(tol, _REL_TOL * np.abs(u)) * (1 + 4 * _REL_TOL)
    with np.errstate(invalid="ignore", over="ignore"):
        hi = np.searchsorted(u, u + reach, side="right") - 1
        lo = np.searchsorted(u, u - reach, side="left")
    hi = np.clip(hi, k, m - 1)
    lo = np.clip(lo, 0, k)
    # Step the ends until they are exactly the last close ranks.
    while True:
        out = (hi < m - 1) & _close(u, u[np.minimum(hi + 1, m - 1)], tol)
        back = (hi > k) & ~_close(u, u[hi], tol)
        if not (out.any() or back.any()):
            break
        hi = hi + out - back
    while True:
        out = (lo > 0) & _close(u, u[np.maximum(lo - 1, 0)], tol)
        back = (lo < k) & ~_close(u, u[lo], tol)
        if not (out.any() or back.any()):
            break
        lo = lo - out + back
    return lo, hi


def _count_after(keys: npt.NDArray, limits: npt.NDArray) -> npt.NDArray:
    """For each position ``i``, the number of positions ``j > i`` with
    ``keys[j] <= limits[i]`` (non-negative integer keys).

    A bottom-up merge sort: at each level every pair ``i < j`` that falls
    in the left and right halves of one block is counted once, by a search
    of the block's right half, sorted, for the limit of ``i``.
    """
    n = keys.size
    counts = np.zeros(n, dtype=np.int64)
    span = int(max(keys.max(initial=0), limits.max(initial=0))) + 2
    pos = np.arange(n)
    width = 1
    while width < n:
        block = pos // (2 * width)
        right = (pos // width) % 2 == 1
        # the right halves' keys, sorted within each block
        comp = np.sort(block[right] * span + keys[right])
        left = ~right
        b = block[left] * span
        counts[left] += np.searchsorted(
            comp, b + limits[left], side="right"
        ) - np.searchsorted(comp, b - 1, side="right")
        width *= 2
    return counts


def concordance_index(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    risk: npt.ArrayLike,
    tie_tol: float = 1e-8,
    ties: str = "therneau",
) -> float:
    r"""Harrell's concordance index (C) of risk scores against right-censored
    outcomes.

    A pair of subjects is *usable* when the one with the earlier time had
    the event; it is *concordant* when that subject also has the higher
    risk. C is the proportion of usable pairs that are concordant: 1 is a
    perfect ranking, 0.5 is chance. It measures discrimination only (a
    model can rank perfectly and still be miscalibrated; see
    :func:`brier_score`).

    Parameters
    ----------
    x : array_like
        Observed times.
    c : array_like
        Censoring flags: ``0`` event, ``1`` right censored. Left and
        interval censoring are not supported.
    risk : array_like
        Risk scores, *higher meaning an earlier event*: a hazard ratio or
        linear predictor, a cumulative hazard, ``1 - sf`` at a horizon. To
        score predicted times or survival probabilities (higher meaning a
        later event), pass their negative.
    tie_tol : float, optional
        Two scores within ``tie_tol`` of each other (or within a relative
        ``1e-9``, as :func:`math.isclose`) are tied. Default ``1e-8``.
    ties : {"therneau", "harrell"}, optional
        How a pair of events at the same time counts. ``"therneau"`` (the
        default, as R's ``survival::concordance`` and lifelines): it is not
        usable, since neither subject outlived the other. ``"harrell"``
        (Harrell's original definition): it is usable, and counts 1 if the
        scores are tied, else 0.5. Every other pair is treated the same
        way by both (see Notes).

    Returns
    -------
    float
        The concordance index; ``nan`` if a time is missing.

    Raises
    ------
    ValueError
        If the arrays differ in length, a score is missing (NaN), a flag
        is not 0 or 1, ``ties`` is not one of the conventions, or no pair
        is usable (every event is tied with, or later than, every other
        subject's time).

    Notes
    -----
    Each pair is scored by the pairwise definition this function
    reproduces exactly (#276):

    - times ``x_i < x_j`` with an event at ``x_i`` (whatever ``c_j``):
      1 if ``risk_i > risk_j``, 0.5 if the scores are tied, else 0;
    - equal times, both events: not usable under ``ties="therneau"``;
      under ``ties="harrell"`` usable, 1 if the scores are tied, else 0.5;
    - equal times, one event and one censored: usable (the censored
      subject outlived the event), 1 if the event has the higher score,
      0.5 on a tie, else 0;
    - equal times, both censored, or an earlier censored time: not usable.

    The default is R's (``survival::concordance``, Therneau) and
    lifelines' (``lifelines.utils.concordance_index``, which scores
    predicted *times*, so its value for ``-risk`` is this one). The two
    conventions agree on data without tied event times.

    The pairs are counted in :math:`O(n \log^2 n)` array operations, not
    one by one: 50,000 subjects take a fraction of a second.

    Examples
    --------
    >>> from surpyval.metrics import concordance_index
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0]
    >>> c = [0, 0, 1, 0, 0]
    >>> risk = [0.9, 0.5, 0.7, 0.6, 0.2]
    >>> concordance_index(x, c, risk)
    0.75

    Two deaths at the same time are a pair only under Harrell's
    convention:

    >>> x = [1.0, 1.0, 2.0, 3.0]
    >>> c = [0, 0, 0, 1]
    >>> risk = [0.9, 0.5, 0.7, 0.2]
    >>> concordance_index(x, c, risk)
    0.8
    >>> concordance_index(x, c, risk, ties="harrell")
    0.75
    """
    counts = concordance_counts(x, c, risk, tie_tol, ties)
    if counts is None:
        return float("nan")
    concordant, usable = counts
    if usable == 0:
        raise ValueError(
            "No usable pairs: the concordance index needs an event that "
            "is earlier than another subject's time, or at the time of a "
            "censored one"
            + (
                " or of another event"
                if ties == "harrell"
                else " (two events at one time are not a pair under "
                "ties='therneau')"
            )
        )
    return float(concordant / usable)


def concordance_counts(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    risk: npt.ArrayLike,
    tie_tol: float = 1e-8,
    ties: str = "therneau",
) -> tuple[float, float] | None:
    """The concordant (a tied score counting half) and usable pairs that
    :func:`concordance_index` divides, ``None`` if a time is missing:
    for pooling the pairs of several groups, as a stratified model's
    concordance pools its strata's (R's ``concordance`` of a stratified
    ``coxph``). The arguments are :func:`concordance_index`'s.

    Examples
    --------
    >>> from surpyval.metrics.concordance import concordance_counts
    >>> concordance_counts([1.0, 2.0, 3.0], [0, 0, 1], [0.9, 0.5, 0.7])
    (2.0, 3.0)
    """
    check_option("ties", ties, TIES)
    x_arr = _as_1d(x, "x")
    c_arr = _as_1d(c, "c")
    s = np.asarray(risk, dtype=float).ravel()
    if not (x_arr.size == c_arr.size == s.size):
        raise ValueError(
            "'x', 'c' and 'risk' must have the same length: got "
            f"{x_arr.size}, {c_arr.size} and {s.size}"
        )
    if np.isnan(s).any():
        raise ValueError(
            "'risk' has {} missing (NaN) score(s) of {}: the concordance "
            "index ranks every subject, so drop those subjects (from 'x' "
            "and 'c' too) or predict their scores first".format(
                int(np.isnan(s).sum()), s.size
            )
        )
    if np.isnan(x_arr).any():
        return None
    if not np.isin(c_arr, (0, 1)).all():
        raise ValueError(
            "'c' must be 0 (event) or 1 (right censored); the concordance "
            "index is not defined for left or interval censoring"
        )
    n = x_arr.size
    event = c_arr == 0

    # Ranks of the scores and, for each, the ranks of the scores tied
    # with it.
    u, rank = np.unique(s, return_inverse=True)
    lo_u, hi_u = _close_ranks(u, tie_tol)
    lo, hi = lo_u[rank], hi_u[rank]

    # Time groups (exact ties), and the subjects after each group.
    times, group, size = np.unique(
        x_arr, return_inverse=True, return_counts=True
    )
    later = n - np.cumsum(size)[group]

    # Pairs at different times, the event first: its score above the
    # later subject's counts 1, tied 0.5. Ordered by (time, score) a later
    # subject at the same time never has a lower score, so it is not
    # counted among the lower ones; ordered by (time, -score) every one
    # after it at the same time has a score at most its own, and is
    # subtracted from those at most its highest tie.
    order = np.lexsort((rank, group))
    below = np.empty(n, dtype=np.int64)
    below[order] = _count_after(rank[order], rank[order] - 1)
    order = np.lexsort((-rank, group))
    upto = np.empty(n, dtype=np.int64)
    same_after = np.cumsum(size)[group[order]] - 1 - np.arange(n)
    upto[order] = _count_after(rank[order], hi[order]) - same_after
    tied_above = upto - below
    concordant = np.sum(below[event]) + 0.5 * np.sum(tied_above[event])
    usable = float(np.sum(later[event]))

    # Pairs at the same time, both events (Harrell's convention only;
    # Therneau's leaves them out): 1 if tied, else 0.5. In the events
    # sorted by (time, score) a tied pair is counted once, from its member
    # that comes first: the later one's score is at least its own.
    span = u.size + 2
    g_ev = group[event] * span
    n_events = np.bincount(group[event], minlength=times.size)
    if ties == "harrell":
        ev_keys = g_ev + rank[event]
        position = np.empty(ev_keys.size, dtype=np.int64)
        position[np.argsort(ev_keys, kind="stable")] = np.arange(ev_keys.size)
        ahead = np.searchsorted(np.sort(ev_keys), g_ev + hi[event], "right")
        tied_pairs = np.sum(ahead - position - 1)
        both = float(np.sum(n_events * (n_events - 1) // 2))
        concordant += 0.5 * both + 0.5 * tied_pairs
        usable += both
    # one event, one censored: 1 if the event's score is above the
    # censored one's and not tied, 0.5 if tied, else 0
    cens = ~event
    cen_key = np.sort(group[cens] * span + rank[cens])
    start = np.searchsorted(cen_key, g_ev, side="left")
    lower = np.searchsorted(cen_key, g_ev + lo[event], side="left")
    upper = np.searchsorted(cen_key, g_ev + hi[event], side="right")
    concordant += np.sum(lower - start) + 0.5 * np.sum(upper - lower)
    n_cens = np.bincount(group[cens], minlength=times.size)
    usable += float(np.sum(n_events * n_cens))
    return float(concordant), usable


def _sum_after(
    keys: npt.NDArray, limits: npt.NDArray, weights: npt.NDArray
) -> npt.NDArray:
    """For each position ``i`` and each column ``k`` of ``limits``, the sum
    of ``weights[j]`` over the positions ``j > i`` with
    ``keys[j] <= limits[i, k]`` (non-negative integer keys, limits at
    least ``-1``).

    :func:`_count_after` with weights and several limits at once: the
    same bottom-up merge sort, with each block's right half sorted and its
    weights summed cumulatively, so a search gives the sum below a limit.
    """
    n = keys.size
    lim = limits.reshape(n, -1)
    out = np.zeros(lim.shape)
    span = int(max(keys.max(initial=0), lim.max(initial=0))) + 2
    pos = np.arange(n)
    width = 1
    while width < n:
        block = pos // (2 * width)
        right = (pos // width) % 2 == 1
        key_r = block[right] * span + keys[right]
        order = np.argsort(key_r, kind="stable")
        comp = key_r[order]
        cum = np.concatenate(([0.0], np.cumsum(weights[right][order])))
        left = ~right
        b = (block[left] * span)[:, None]
        top = np.searchsorted(comp, b + lim[left], side="right")
        bottom = np.searchsorted(comp, b - 1, side="right")
        out[left] += cum[top] - cum[bottom]
        width *= 2
    return out


def _pair_sums(
    x: npt.NDArray,
    event: npt.NDArray,
    s: npt.NDArray,
    wt: npt.NDArray,
    tie_tol: float,
    ties: str,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Each subject's share of the weighted pairs: ``N_i``, the sum over
    the usable pairs it belongs to of the pair's weight times its score
    (1 concordant, 0.5 tied, 0 discordant), and ``M_i``, the sum of those
    pairs' weights. A pair's weight is ``wt`` of its earlier subject (an
    event). Every pair is in two subjects' sums, so ``N_i.sum() / 2`` and
    ``M_i.sum() / 2`` are the index's numerator and denominator.

    The pairs and their scores are :func:`concordance_counts`'s.
    """
    n = x.size
    u, rank = np.unique(s, return_inverse=True)
    rank = rank.ravel()
    lo_u, hi_u = _close_ranks(u, tie_tol)
    lo, hi = lo_u[rank], hi_u[rank]
    m = u.size
    times, group = np.unique(x, return_inverse=True)
    group = group.ravel()
    N = np.zeros(n)
    M = np.zeros(n)
    ev = np.flatnonzero(event)

    # Pairs at different times. The earlier subject (an event) scores
    # 1 against a later one with a lower rank, 0.5 against one whose rank
    # is from its own up to its highest tie, else 0 (see the Notes of
    # concordance_index on scores within the tolerance).
    # As the earlier subject: every subject is a point (weight 1) and each
    # event a query, ordered by time with a time's points ahead of its
    # queries, so the points after a query are those at later times.
    keys = np.concatenate((rank, rank[ev]))
    is_query = np.concatenate((np.zeros(n, bool), np.ones(ev.size, bool)))
    t_key = np.concatenate((group, group[ev]))
    order = np.lexsort((is_query, t_key))
    limits = np.full((n + ev.size, 3), -1)
    limits[n:] = np.column_stack((rank[ev] - 1, hi[ev], np.full(ev.size, m)))
    weights = (~is_query).astype(float)
    sums = np.empty((n + ev.size, 3))
    sums[order] = _sum_after(keys[order], limits[order], weights[order])
    below, upto, later = sums[n:].T
    N[ev] += wt[ev] * (below + 0.5 * (upto - below))
    M[ev] += wt[ev] * later
    # As the later subject: each event is a point weighted by its wt and
    # every subject a query, ordered by decreasing time, a time's points
    # ahead of its queries. The earlier event scores 1 when its rank is
    # above the query's, 0.5 from the query's lowest tie up to its rank.
    keys = np.concatenate((rank[ev], rank))
    is_query = np.concatenate((np.zeros(ev.size, bool), np.ones(n, bool)))
    t_key = np.concatenate((group[ev], group))
    order = np.lexsort((is_query, -t_key))
    limits = np.full((n + ev.size, 3), -1)
    limits[ev.size :] = np.column_stack((rank, lo - 1, np.full(n, m)))
    weights = np.concatenate((wt[ev], np.zeros(n)))
    sums = np.empty((n + ev.size, 3))
    sums[order] = _sum_after(keys[order], limits[order], weights[order])
    upto, below_lo, total = sums[ev.size :].T
    N += (total - upto) + 0.5 * (upto - below_lo)
    M += total

    # Pairs at the same time, an event and a censored subject: usable,
    # 1 if the event's score is the higher and not tied, 0.5 if tied.
    span = m + 2
    g_wt = np.zeros(times.size)
    g_wt[group[ev]] = wt[ev]
    n_ev = np.bincount(group[ev], minlength=times.size)
    cens = np.flatnonzero(~event)
    n_cens = np.bincount(group[cens], minlength=times.size)
    cen_key = np.sort(group[cens] * span + rank[cens])
    ev_key = np.sort(group[ev] * span + rank[ev])
    g = group[ev] * span
    start = np.searchsorted(cen_key, g, side="left")
    low = np.searchsorted(cen_key, g + lo[ev], side="left")
    high = np.searchsorted(cen_key, g + hi[ev], side="right")
    N[ev] += wt[ev] * ((low - start) + 0.5 * (high - low))
    M[ev] += wt[ev] * n_cens[group[ev]]
    g = group[cens] * span
    end = np.searchsorted(ev_key, g + span - 1, side="right")
    low = np.searchsorted(ev_key, g + lo[cens], side="left")
    high = np.searchsorted(ev_key, g + hi[cens], side="right")
    w_c = g_wt[group[cens]]
    N[cens] += w_c * ((end - high) + 0.5 * (high - low))
    M[cens] += w_c * n_ev[group[cens]]

    # Pairs at the same time, both events: usable only under Harrell's
    # convention, 1 if the scores are tied, else 0.5.
    if ties == "harrell":
        g = group[ev] * span
        low = np.searchsorted(ev_key, g + lo[ev], side="left")
        high = np.searchsorted(ev_key, g + hi[ev], side="right")
        others = n_ev[group[ev]] - 1
        tied = high - low - 1
        N[ev] += wt[ev] * (0.5 * others + 0.5 * tied)
        M[ev] += wt[ev] * others
    return N, M


def _check_risk(s: npt.NDArray, n: int, name: str) -> None:
    if s.size != n:
        raise ValueError(
            f"'x', 'c' and '{name}' must have the same length: got {n}, "
            f"{n} and {s.size}"
        )
    if np.isnan(s).any():
        raise ValueError(
            "'{}' has {} missing (NaN) score(s) of {}: the concordance "
            "index ranks every subject, so drop those subjects (from 'x' "
            "and 'c' too) or predict their scores first".format(
                name, int(np.isnan(s).sum()), s.size
            )
        )


def _prepare(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    weighting: str,
    tau: float | None,
    ties: str,
    alpha: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray] | None:
    """Validated times, event flags and each subject's pair weight (that
    of the pairs in which it is the earlier subject); ``None`` if a time
    is missing."""
    check_option("ties", ties, TIES)
    check_option("weighting", weighting, WEIGHTINGS)
    if not 0 < alpha < 1:
        raise ValueError(
            f"'alpha' must be between 0 and 1 (0.05 for a 95% interval); "
            f"got {alpha}"
        )
    if tau is not None and not tau > 0:
        raise ValueError(
            f"'tau' must be a positive time (the horizon of the pairs "
            f"counted) or None for no horizon; got {tau}"
        )
    x_arr = _as_1d(x, "x")
    c_arr = _as_1d(c, "c")
    if x_arr.size != c_arr.size:
        raise ValueError(
            "'x' and 'c' must have the same length: got "
            f"{x_arr.size} and {c_arr.size}"
        )
    if not np.isin(c_arr, (0, 1)).all():
        raise ValueError(
            "'c' must be 0 (event) or 1 (right censored); the concordance "
            "index is not defined for left or interval censoring"
        )
    if np.isnan(x_arr).any():
        return None
    event = c_arr == 0
    if weighting == "uno":
        # 1 / G(x-)^2, G the Kaplan-Meier estimate of the censoring
        # distribution with an event ahead of a censoring at its time
        # (R's timewt = "n/G2"). G(x-) > 0 at every event.
        times, g = censoring_survival(x_arr, ~event, ties="events_first")
        wt = 1.0 / step_left_limit(times, g, x_arr, before=1.0) ** 2
    else:
        wt = np.ones(x_arr.size)
    wt = np.where(event, wt, 0.0)
    if tau is not None:
        wt = np.where(x_arr <= tau, wt, 0.0)
    return x_arr, event, wt


def _no_pairs(ties: str, tau: float | None) -> ValueError:
    return ValueError(
        "No usable pairs: the concordance index needs an event that "
        "is earlier than another subject's time, or at the time of a "
        "censored one"
        + (
            " or of another event"
            if ties == "harrell"
            else " (two events at one time are not a pair under "
            "ties='therneau')"
        )
        + (
            f", at or before tau = {tau:g}; raise 'tau' or pass None"
            if tau is not None
            else ""
        )
    )


def _influence(
    x: npt.NDArray,
    event: npt.NDArray,
    s: npt.NDArray,
    wt: npt.NDArray,
    tie_tol: float,
    ties: str,
) -> tuple[float, float, float, npt.NDArray]:
    """The index, its numerator and denominator, and each subject's
    infinitesimal-jackknife influence ``(N_i - C M_i) / M``."""
    N, M = _pair_sums(x, event, s, wt, tie_tol, ties)
    num, den = 0.5 * N.sum(), 0.5 * M.sum()
    if not den > 0:
        return np.nan, num, den, N
    estimate = num / den
    return estimate, num, den, (N - estimate * M) / den


def _finite(v: Any) -> float | None:
    v = float(v)
    return v if np.isfinite(v) else None


class ConcordanceResult:
    r"""A concordance index with its standard error and confidence
    interval, from :func:`concordance_estimate`.

    Attributes
    ----------
    estimate : float
        The concordance index: the weighted proportion of usable pairs
        that the scores rank correctly (a tied score counting half).
    se : float
        Its infinitesimal-jackknife standard error (R's
        ``sqrt(concordance(...)$var)``).
    ci : tuple of float
        The ``1 - alpha`` confidence interval: ``estimate -/+ z se``
        (``ci_method="normal"``), or that interval on the log-odds scale
        mapped back (``"logit"``), which stays inside (0, 1).
    alpha : float
        The interval's significance level.
    ci_method : str
        ``"normal"`` or ``"logit"``.
    weighting : str
        ``"harrell"`` (every pair weighs 1) or ``"uno"`` (each pair
        weighted by ``1 / G(t-)^2`` at its earlier time ``t``).
    tau : float or None
        The horizon: only pairs whose earlier time is at most ``tau``
        count; ``None`` for none.
    ties : str
        The convention for two events at the same time.
    n : int
        The number of subjects.
    concordant : float
        The weighted number of concordant pairs, a tied score counting
        half (the numerator).
    pairs : float
        The weighted number of usable pairs (the denominator).
    influence : ndarray
        Each subject's influence on the index, ``(N_i - C M_i) / M`` (R's
        ``dfbeta``); the variance is the sum of their squares.

    Examples
    --------
    >>> from surpyval.metrics import concordance_estimate
    >>> res = concordance_estimate([1, 2, 3, 4, 5], [0, 0, 1, 0, 0],
    ...                   [0.9, 0.5, 0.7, 0.6, 0.2])
    >>> round(res.estimate, 4), round(res.se, 4)
    (0.75, 0.2119)
    """

    def __init__(
        self,
        estimate: float,
        se: float,
        alpha: float,
        weighting: str,
        tau: float | None,
        ties: str,
        n: int,
        concordant: float,
        pairs: float,
        influence: npt.NDArray,
        ci_method: str = "normal",
    ) -> None:
        self.estimate = float(estimate)
        self.se = float(se)
        self.alpha = alpha
        self.ci_method = ci_method
        self.ci = _interval(self.estimate, self.se, alpha, ci_method)
        self.weighting = weighting
        self.tau = tau
        self.ties = ties
        self.n = n
        self.concordant = float(concordant)
        self.pairs = float(pairs)
        self.influence = influence

    def __float__(self) -> float:
        return self.estimate

    def __repr__(self) -> str:
        name = {"harrell": "Harrell's C", "uno": "Uno's C"}[self.weighting]
        level = round(100 * (1 - self.alpha), 6)
        out = (
            name
            + "\n"
            + "=" * len(name)
            + f"\nC                : {self.estimate:.6g}"
            + f"\nStd. error       : {self.se:.6g}"
            + "\n{:<17}: ".format(f"{level:g}% CI")
            + f"({self.ci[0]:.6g}, "
            + f"{self.ci[1]:.6g})"
            + "\nHorizon tau      : "
            + ("none" if self.tau is None else f"{self.tau:g}")
            + f"\nSubjects         : {self.n}"
        )
        return out

    def to_dict(self) -> dict:
        """The result as strict JSON (a non-finite number is ``None``);
        the per-subject influence is left out."""
        return {
            "metric": "concordance",
            "weighting": self.weighting,
            "estimate": _finite(self.estimate),
            "se": _finite(self.se),
            "ci": [_finite(self.ci[0]), _finite(self.ci[1])],
            "alpha": self.alpha,
            "ci_method": self.ci_method,
            "tau": None if self.tau is None else float(self.tau),
            "ties": self.ties,
            "n": self.n,
            "concordant": _finite(self.concordant),
            "pairs": _finite(self.pairs),
        }


def concordance_estimate(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    risk: npt.ArrayLike,
    weighting: str = "harrell",
    tau: float | None = None,
    alpha: float = 0.05,
    tie_tol: float = 1e-8,
    ties: str = "therneau",
    ci_method: str = "normal",
) -> ConcordanceResult:
    r"""Harrell's or Uno's concordance index, with its standard error and
    confidence interval.

    Harrell's C (the default) is :func:`concordance_index`: the share of
    usable pairs -- the subject with the earlier time had the event --
    that the scores rank correctly. It depends on the censoring: with
    heavy censoring the usable pairs are mostly early ones, so C
    describes the early ranking more than the late. Uno's C weights each
    pair by ``1 / G(t-)^2``, where ``G`` is the Kaplan-Meier estimate of
    the probability of being still uncensored and ``t`` the pair's earlier
    time, so the late pairs that censoring thins out count for as many as
    they would have without it; the estimate then does not depend on the
    censoring distribution (when censoring is independent of the scores).
    ``tau`` keeps only the pairs whose earlier time is at most ``tau``:
    Uno's C needs a horizon at which ``G`` is still well away from 0.

    The standard error is the infinitesimal jackknife: how much C moves
    when one subject's weight changes, ``d_i = (N_i - C M_i) / M``, where
    ``N_i`` and ``M_i`` are the subject's share of the weighted concordant
    and usable pairs and ``M`` all usable pairs; the variance is
    :math:`\sum_i d_i^2`. The censoring weights are held fixed, as R
    does. The interval is the normal one, ``C -/+ z se``.

    .. math::
        \hat C = \frac{\sum_{i,j} \delta_i\, w(t_i)\, I(t_i < t_j,\,
            t_i \le \tau)\, \big(I(r_i > r_j) + \tfrac12 I(r_i = r_j)\big)}
            {\sum_{i,j} \delta_i\, w(t_i)\, I(t_i < t_j,\, t_i \le \tau)},
        \qquad w(t) = 1 \text{ (Harrell) or } \hat G(t-)^{-2}
        \text{ (Uno).}

    Parameters
    ----------
    x : array_like
        Observed times.
    c : array_like
        Censoring flags: ``0`` event, ``1`` right censored.
    risk : array_like
        Risk scores, higher meaning an earlier event (see
        :func:`concordance_index`).
    weighting : {"harrell", "uno"}, optional
        ``"harrell"`` (default): every usable pair weighs 1. ``"uno"``:
        each weighs ``1 / G(t-)^2`` (R's ``timewt = "n/G2"``).
    tau : float, optional
        Count only the pairs whose earlier time (the event) is at most
        ``tau`` (R's ``ymax``). Default ``None``, every pair.
    alpha : float, optional
        Significance level of the interval. Default ``0.05``.
    tie_tol : float, optional
        Two scores within ``tie_tol`` are tied, as in
        :func:`concordance_index`. Default ``1e-8``.
    ties : {"therneau", "harrell"}, optional
        How two events at the same time count, as in
        :func:`concordance_index`. Default ``"therneau"`` (R's).
    ci_method : {"normal", "logit"}, optional
        The scale of the interval: ``"normal"`` (default), ``C -/+ z se``;
        ``"logit"``, ``logit(C) -/+ z se / (C (1 - C))`` mapped back.

    Returns
    -------
    ConcordanceResult
        The estimate, ``se``, ``ci``, the weighted pair counts and each
        subject's influence. A missing time gives a ``nan`` estimate and
        standard error.

    Raises
    ------
    ValueError
        If the arrays differ in length, a score is missing, a flag is not
        0 or 1, an option is not one of its choices, ``tau`` is not
        positive, ``alpha`` is not in (0, 1), or no pair is usable.

    Notes
    -----
    This reproduces R's ``survival::concordance(Surv(x, 1 - c) ~ risk,
    reverse = TRUE, timewt = "n" or "n/G2", ymax = tau)``: its estimate
    and ``sqrt(var)``. For Uno's weighting the censoring distribution is
    estimated from the same data, an event taken to precede a censoring
    at its time, as R and Uno et al. (2011) do. The estimate is Uno's
    :math:`C_\tau` (``survC1::Est.Cval``) on data without tied times; its
    standard error is R's jackknife, not Uno's perturbation-resampling
    one. Scores tied with each other count half, as in R; two events at
    one time are not a pair by default, as in R.

    The weighted pairs are counted with sorts and searches, in
    :math:`O(n \log^2 n)` array operations.

    References
    ----------
    Uno, H., Cai, T., Pencina, M. J., D'Agostino, R. B. and Wei, L. J.
    (2011). On the C-statistics for evaluating overall adequacy of risk
    prediction procedures with censored survival data. *Statistics in
    Medicine*, 30(10), 1105-1117.

    Therneau, T. M. and Atkinson, E. (2023). Concordance. Vignette of the
    R package ``survival``.

    Examples
    --------
    >>> from surpyval.metrics import concordance_estimate
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> c = [0, 1, 0, 1, 0, 0]
    >>> risk = [0.9, 0.5, 0.7, 0.6, 0.2, 0.4]
    >>> res = concordance_estimate(x, c, risk)
    >>> round(res.estimate, 4), round(res.se, 4)
    (0.8889, 0.1341)
    >>> res = concordance_estimate(x, c, risk, weighting="uno", tau=5)
    >>> round(res.estimate, 4), round(res.se, 4)
    (0.7337, 0.2602)
    """
    check_option("ci_method", ci_method, CI_METHODS)
    prepared = _prepare(x, c, weighting, tau, ties, alpha)
    s = np.asarray(risk, dtype=float).ravel()
    n = _as_1d(x, "x").size
    _check_risk(s, n, "risk")
    if prepared is None:
        nan = float("nan")
        return ConcordanceResult(
            nan,
            nan,
            alpha,
            weighting,
            tau,
            ties,
            n,
            nan,
            nan,
            np.full(n, nan),
            ci_method,
        )
    x_arr, event, wt = prepared
    estimate, num, den, d = _influence(x_arr, event, s, wt, tie_tol, ties)
    if not den > 0:
        raise _no_pairs(ties, tau)
    se = float(np.sqrt(np.sum(d**2)))
    return ConcordanceResult(
        estimate, se, alpha, weighting, tau, ties, n, num, den, d, ci_method
    )


class ConcordanceComparison:
    r"""Two scores' concordance indices on the same subjects and the test
    of their difference, from :func:`compare_concordance`.

    Attributes
    ----------
    estimates : ndarray
        The two indices, ``[C_a, C_b]``.
    cov : ndarray
        Their 2 x 2 infinitesimal-jackknife covariance (R's
        ``concordance(fit_a, fit_b)$var``).
    difference : float
        ``C_a - C_b``.
    se : float
        The standard error of the difference.
    ci : tuple of float
        The ``1 - alpha`` normal confidence interval of the difference.
    statistic : float
        ``difference / se``, standard normal when the indices are equal.
    p_value : float
        Its two-sided p-value.
    alpha : float
        The interval's significance level.
    weighting : str
        ``"harrell"`` or ``"uno"``.
    tau : float or None
        The horizon of the pairs counted.
    ties : str
        The convention for two events at the same time.
    n : int
        The number of subjects.

    Examples
    --------
    >>> from surpyval.metrics import compare_concordance
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> c = [0, 1, 0, 1, 0, 0]
    >>> res = compare_concordance(x, c, [0.9, 0.5, 0.7, 0.6, 0.2, 0.4],
    ...                           [0.9, 0.5, 0.2, 0.6, 0.7, 0.4])
    >>> res.estimates.round(4)
    array([0.8889, 0.6667])
    >>> round(res.difference, 4), round(res.se, 4)
    (0.2222, 0.2927)
    """

    def __init__(
        self,
        a: ConcordanceResult,
        b: ConcordanceResult,
        alpha: float,
    ) -> None:
        self.estimates = np.array([a.estimate, b.estimate])
        d = np.column_stack((a.influence, b.influence))
        self.cov = d.T @ d
        self.difference = float(a.estimate - b.estimate)
        var = self.cov[0, 0] + self.cov[1, 1] - 2 * self.cov[0, 1]
        self.se = float(np.sqrt(max(var, 0.0)))
        z = norm.ppf(1 - alpha / 2)
        self.ci = (
            self.difference - z * self.se,
            self.difference + z * self.se,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            self.statistic = float(np.divide(self.difference, self.se))
        self.p_value = float(2 * norm.sf(abs(self.statistic)))
        self.alpha = alpha
        self.weighting = a.weighting
        self.tau = a.tau
        self.ties = a.ties
        self.n = a.n

    def __repr__(self) -> str:
        name = {"harrell": "Harrell's C", "uno": "Uno's C"}[self.weighting]
        header = f"Comparison of two {name}"
        level = round(100 * (1 - self.alpha), 6)
        return (
            header
            + "\n"
            + "=" * len(header)
            + f"\nC (a, b)         : {self.estimates[0]:.6g}, "
            + f"{self.estimates[1]:.6g}"
            + f"\nDifference       : {self.difference:.6g}"
            + f"\nStd. error       : {self.se:.6g}"
            + "\n{:<17}: ".format(f"{level:g}% CI")
            + f"({self.ci[0]:.6g}, "
            + f"{self.ci[1]:.6g})"
            + f"\nz                : {self.statistic:.6g}"
            + f"\np-value          : {self.p_value:.6g}"
        )

    def to_dict(self) -> dict:
        """The result as strict JSON (a non-finite number is ``None``)."""
        return {
            "metric": "concordance comparison",
            "weighting": self.weighting,
            "estimates": [_finite(v) for v in self.estimates],
            "cov": [[_finite(v) for v in row] for row in self.cov],
            "difference": _finite(self.difference),
            "se": _finite(self.se),
            "ci": [_finite(self.ci[0]), _finite(self.ci[1])],
            "statistic": _finite(self.statistic),
            "p_value": _finite(self.p_value),
            "alpha": self.alpha,
            "tau": None if self.tau is None else float(self.tau),
            "ties": self.ties,
            "n": self.n,
        }


def compare_concordance(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    risk_a: npt.ArrayLike,
    risk_b: npt.ArrayLike,
    weighting: str = "harrell",
    tau: float | None = None,
    alpha: float = 0.05,
    tie_tol: float = 1e-8,
    ties: str = "therneau",
) -> ConcordanceComparison:
    r"""Compare the concordance of two risk scores on the same subjects.

    The two indices share their subjects, so they are correlated, and the
    standard error of their difference is not that of two independent
    estimates: it comes from the subjects' influences on both,
    :math:`\operatorname{var}(C_a - C_b) = \sum_i (d_{a,i} - d_{b,i})^2`
    (see :func:`concordance_estimate`). The test is ``z = (C_a - C_b) / se``
    against the standard normal, as R's
    ``concordance(fit_a, fit_b)`` with the contrast ``c(1, -1)``.

    Parameters
    ----------
    x : array_like
        Observed times.
    c : array_like
        Censoring flags: ``0`` event, ``1`` right censored.
    risk_a : array_like
        The first model's risk scores, higher meaning an earlier event.
    risk_b : array_like
        The second model's risk scores, on the same subjects.
    weighting : {"harrell", "uno"}, optional
        The pair weights, as in :func:`concordance_estimate`. Default
        ``"harrell"``.
    tau : float, optional
        The horizon of the pairs counted, as in :func:`concordance_estimate`.
    alpha : float, optional
        Significance level of the interval. Default ``0.05``.
    tie_tol : float, optional
        Scores within this of each other are tied. Default ``1e-8``.
    ties : {"therneau", "harrell"}, optional
        How two events at the same time count. Default ``"therneau"``.

    Returns
    -------
    ConcordanceComparison
        The two indices, their covariance, the difference, its standard
        error, interval, z statistic and p-value.

    Raises
    ------
    ValueError
        As :func:`concordance_estimate`, for either score; or if a time is
        missing.

    Examples
    --------
    >>> from surpyval.metrics import compare_concordance
    >>> x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    >>> c = [0, 1, 0, 1, 0, 0]
    >>> res = compare_concordance(x, c, [0.9, 0.5, 0.7, 0.6, 0.2, 0.4],
    ...                           [0.9, 0.5, 0.2, 0.6, 0.7, 0.4])
    >>> round(res.statistic, 4), round(res.p_value, 4)
    (0.7593, 0.4477)
    """
    if np.isnan(_as_1d(x, "x")).any():
        raise ValueError(
            "'x' has missing (NaN) times: drop those subjects (from 'c', "
            "'risk_a' and 'risk_b' too) to compare the scores"
        )
    a = concordance_estimate(
        x, c, risk_a, weighting, tau, alpha, tie_tol, ties
    )
    b = concordance_estimate(
        x, c, risk_b, weighting, tau, alpha, tie_tol, ties
    )
    return ConcordanceComparison(a, b, alpha)
