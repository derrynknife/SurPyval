"""Harrell's concordance index, in O(n log n) (#512).

The index compares every usable pair of subjects: the one that failed
first should carry the higher risk score. Counting the pairs one by one
costs O(n^2) (about 5 minutes at 50,000 subjects); here the subjects are
sorted by time and the pairs are counted with a merge sort over the ranked
scores, as lifelines and scikit-survival do, in O(n log^2 n) array
operations. The tie conventions are those of the pairwise definition
(see :func:`concordance_index`), which the tests keep as the oracle.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from surpyval.utils import validate_1d as _as_1d

__all__ = ["concordance_index"]

# math.isclose's default relative tolerance, which the pairwise definition
# used together with ``tie_tol``.
_REL_TOL = 1e-9


def _close(a: npt.NDArray, b: npt.NDArray, tol: float) -> npt.NDArray:
    """``math.isclose(a, b, abs_tol=tol)``, elementwise."""
    with np.errstate(invalid="ignore"):
        bound = np.maximum(_REL_TOL * np.maximum(np.abs(a), np.abs(b)), tol)
        return (a == b) | (np.abs(a - b) <= bound)


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

    Returns
    -------
    float
        The concordance index; ``nan`` if a time or a score is missing.

    Raises
    ------
    ValueError
        If the arrays differ in length, a flag is not 0 or 1, or no pair
        is usable (every event is tied with, or later than, every other
        subject's time).

    Notes
    -----
    The ties follow Harrell (the pairwise definition this function
    reproduces exactly, #276):

    - times ``x_i < x_j`` with an event at ``x_i`` (whatever ``c_j``):
      1 if ``risk_i > risk_j``, 0.5 if the scores are tied, else 0;
    - equal times, both events: usable, 1 if the scores are tied, else 0.5;
    - equal times, one event and one censored: usable (the censored
      subject outlived the event), 1 if the event has the higher score,
      0.5 on a tie, else 0;
    - equal times, both censored, or an earlier censored time: not usable.

    lifelines (``lifelines.utils.concordance_index``) scores predicted
    *times*, so its value for ``-risk`` is this one; it counts tied times
    as usable only when exactly one is an event, so the two differ on data
    with tied event times.

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
    """
    x_arr = _as_1d(x, "x")
    c_arr = _as_1d(c, "c")
    s = np.asarray(risk, dtype=float).ravel()
    if not (x_arr.size == c_arr.size == s.size):
        raise ValueError(
            "'x', 'c' and 'risk' must have the same length: got "
            f"{x_arr.size}, {c_arr.size} and {s.size}"
        )
    if np.isnan(x_arr).any() or np.isnan(s).any():
        return float("nan")
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

    # Pairs at the same time, both events: 1 if tied, else 0.5. In the
    # events sorted by (time, score) a tied pair is counted once, from its
    # member that comes first: the later one's score is at least its own.
    span = u.size + 2
    g_ev = group[event] * span
    ev_keys = g_ev + rank[event]
    position = np.empty(ev_keys.size, dtype=np.int64)
    position[np.argsort(ev_keys, kind="stable")] = np.arange(ev_keys.size)
    ahead = np.searchsorted(np.sort(ev_keys), g_ev + hi[event], "right")
    tied_pairs = np.sum(ahead - position - 1)
    n_events = np.bincount(group[event], minlength=times.size)
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

    if usable == 0:
        raise ValueError(
            "No usable pairs: the concordance index needs an event that "
            "is earlier than another subject's time (or tied with it)"
        )
    return float(concordant / usable)
