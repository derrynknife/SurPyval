import warnings
from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from surpyval.univariate.nonparametric.nonparametric_fitter import (
    NonParametricFitter,
)

from ._turnbull_npmle import (
    DOES_NOT_EXIST,
    EXISTS,
    NOT_UNIQUE,
    UNDETERMINED,
    npmle_existence,
)
from .fleming_harrington import fleming_harrington as fh
from .kaplan_meier import kaplan_meier as km
from .nelson_aalen import nelson_aalen as na

# The estimators that can be applied to the Turnbull ladder. Checked up
# front: an unknown name used to fall through to Fleming-Harrington in the
# EM and only fail later, as a bare KeyError, when the variance was looked
# up by that name.
TURNBULL_ESTIMATORS: dict[str, Callable[..., npt.NDArray]] = {
    "Fleming-Harrington": fh,
    "Nelson-Aalen": na,
    "Kaplan-Meier": km,
}


def check_turnbull_estimator(estimator: str) -> None:
    """Raise a ``ValueError`` if ``estimator`` is not a Turnbull option."""
    if estimator not in TURNBULL_ESTIMATORS:
        raise ValueError(
            "'turnbull_estimator' must be one of {}; got {!r}".format(
                ", ".join(repr(k) for k in TURNBULL_ESTIMATORS), estimator
            )
        )


def _innermost(
    lo: npt.NDArray, hi: npt.NDArray, M: int
) -> npt.NDArray[np.bool_]:
    """Pieces inside Turnbull's innermost intervals.

    An innermost interval is a run of pieces ``[a, b]`` that starts at
    some observation's support start (``a`` is a ``lo``) and ends at some
    support end (``b`` is a ``hi``) with no other start in ``(a, b]`` and
    no other end in ``[a, b)``. Without truncation every other piece is
    dominated -- each support containing it also contains an innermost
    interval -- so the NPMLE puts no mass there (Turnbull 1976).
    """
    valid = lo <= hi
    is_lo = np.zeros(M, dtype=bool)
    is_lo[lo[valid]] = True
    is_hi = np.zeros(M, dtype=bool)
    is_hi[hi[valid]] = True
    idx = np.arange(M)
    # The latest start at or before each piece, and the latest end
    # strictly before it (-1 where there is none).
    last_lo = np.maximum.accumulate(np.where(is_lo, idx, -1))
    last_hi = np.maximum.accumulate(np.where(is_hi, idx, -1))
    prev_hi = np.concatenate([[-1], last_hi[:-1]])
    ends = is_hi & (last_lo >= 0) & (prev_hi < last_lo)
    mark = np.zeros(M + 1)
    np.add.at(mark, last_lo[ends], 1.0)
    np.add.at(mark, idx[ends] + 1, -1.0)
    return np.cumsum(mark[:M]) > 0


def turnbull(
    x: npt.ArrayLike,
    c: npt.ArrayLike,
    n: npt.ArrayLike,
    t: npt.ArrayLike,
    estimator: str = "Fleming-Harrington",
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> dict:
    """
    Turnbull NPMLE via the EM (self-consistency) algorithm.

    Every observation's support -- the set of Turnbull interval endpoints
    its event could have occurred at -- is a *contiguous* run of indices
    into the sorted ``bounds`` array, as is its truncation (observation)
    window. The E-step therefore never needs an (N x M) matrix: the
    per-observation support probabilities are range sums of ``p``
    (prefix sums), and the per-interval expected event counts are sums of
    per-observation weights over ranges (difference arrays). Each
    iteration is O(N + M) in both time and memory.
    """
    if max_iter < 1:
        raise ValueError(f"max_iter must be at least 1; got {max_iter}")
    check_turnbull_estimator(estimator)
    # Taken as arrays before anything indexes or slices them. The
    # signature accepts array-like because callers pass lists, but the
    # body below is written against arrays throughout.
    x = np.asarray(x)
    c = np.asarray(c)
    n = np.asarray(n)
    t = np.asarray(t)
    any_truncated = np.isfinite(t).any()
    # Find all unique bounding points
    bounds = np.unique(np.concatenate([np.unique(x), np.unique(t)]))
    # Add the times at which there was an observation again since
    # the failure occurs in a 0 bound e.g. in the [1, 1] "interval".
    exact_times = np.unique(x[c == 0])
    bounds = np.sort(np.concatenate([bounds, exact_times]))

    if x.ndim == 1:
        x_new = np.empty(shape=(x.shape[0], 2))
        x_new[:, 0] = x
        x_new[:, 1] = x
        x = x_new

    # Unpack x array
    xl = x[:, 0].astype(float)
    xr = x[:, 1].astype(float)

    # Unpack t array
    tl = t[:, 0]
    tr = t[:, 1]

    # If there are left and right censored observations,
    # convert them to interval censored observations
    xl[c == -1] = -np.inf
    xr[c == 1] = np.inf

    # Find the count of intervals (M) and unique observation windows (N)
    M = bounds.size
    N = xl.size

    # Each observation's support is the contiguous index range [lo, hi] of
    # the bound points its event may sit on:
    # - an exactly observed event sits on the zero-width "interval" at the
    #   first copy of its (duplicated) time;
    # - a right-censored event may sit on any bound strictly after the
    #   censoring time;
    # - an interval-censored event (including left censored, whose interval
    #   is (-inf, xr]) may sit on any bound in (xl, xr]: the zero-width
    #   exact interval at xl is excluded when xl is also an exactly
    #   observed time (the event is known to be after xl), and the one at
    #   xr is *included* -- the standard (l, r] convention (Turnbull 1976),
    #   under which an interval whose right endpoint coincides with an
    #   exact event time may have failed at that time (#272).
    exact = xl == xr
    right = ~exact & np.isinf(xr)
    interval = ~exact & ~right

    lo = np.empty(N, dtype=np.int64)
    hi = np.empty(N, dtype=np.int64)
    lo[exact] = np.searchsorted(bounds, xl[exact], side="left")
    hi[exact] = lo[exact]
    lo[right] = np.searchsorted(bounds, xl[right], side="right")
    hi[right] = M - 1
    lo[interval] = np.searchsorted(
        bounds, xl[interval], side="left"
    ) + np.isin(xl[interval], exact_times)
    hi[interval] = (
        np.searchsorted(bounds, xr[interval], side="left")
        - 1
        + np.isin(xr[interval], exact_times)
    )

    # Exact + right-censored (possibly weighted) untruncated data -- in
    # either 1-D or degenerate-interval form -- is the regime where
    # Turnbull reduces exactly to Kaplan-Meier; keep equivalent 1-D
    # inputs so the variance can use the *observed* count ladder rather
    # than the EM's expected-count ladder, which redistributes censored
    # mass as fractional later events and silently understates the
    # variance (#260, #273).
    km_reducible = (not any_truncated) and not interval.any()
    if km_reducible:
        x_raw = xl.copy()
        c_raw = np.where(exact, 0, 1)
        n_raw = np.asarray(n).copy()
        t_raw = np.asarray(t).copy()

    # Each observation's truncation window is likewise a contiguous index
    # range [w_lo, w_hi]: the bound points at which an event was observable
    # -- strictly after its left truncation time and at or before its right
    # truncation time.
    #
    # Index j stands for the half-open interval ``(bounds[j], bounds[j+1]]``,
    # so an event placed there is already strictly after ``bounds[j]``. The
    # first admissible index is thus the *last* bound equal to ``tl``: that
    # interval is ``(tl, next]``, which respects the strict (entry, exit]
    # convention, while the interval starting one index earlier is the
    # zero-width ``(tl, tl]`` that an exact event time duplicated into
    # ``bounds`` creates -- an event at exactly the entry time, which the
    # convention excludes (#260).
    #
    # ``side="right" - 1`` lands there exactly, because every finite
    # truncation time is itself in ``bounds``. Neither endpoint of the
    # search alone will do: ``side="left"`` keeps the zero-width interval
    # and readmits an event at the entry time, while ``side="right"``
    # discards ``(tl, next]`` as well, one interval too many.
    #
    # That over-exclusion is what made left censoring under truncation
    # fail. A left-censored event lies in ``(-inf, xr]``, which under an
    # entry at ``tl`` is the single interval ``(tl, xr]`` -- often the only
    # one such a row has. Dropping it left the row with an empty support,
    # so a *vacuous* entry time, one below every observation and excluding
    # nobody, turned a working fit into a raise or drove the EM to the
    # degenerate all-zero end of the ladder (#308).
    if any_truncated:
        w_lo_all = np.where(
            np.isfinite(tl),
            np.maximum(np.searchsorted(bounds, tl, side="right") - 1, 0),
            0,
        )
        w_hi_all = np.where(
            np.isfinite(tr),
            np.searchsorted(bounds, tr, side="right") - 1,
            M - 1,
        )
        # An observation's event provably lies inside its own truncation
        # window (it was observed), so its support is the *intersection*
        # of its censoring support with the window. Without this, mass
        # from left-censored or entry-spanning interval observations is
        # redistributed below the entry time -- where the event cannot
        # be -- and the EM can wander to the degenerate all-zero fixed
        # point on perfectly valid data (#273).
        lo = np.maximum(lo, w_lo_all)
        hi = np.minimum(hi, w_hi_all)
        if (lo > hi).any():
            raise ValueError(
                "An observation's censoring interval does not intersect "
                "its own truncation window, so it has zero probability of "
                "being observed as recorded; check the x, c and t inputs."
            )
        truncated = np.isfinite(tl) | np.isfinite(tr)
        w_lo, w_hi = w_lo_all[truncated], w_hi_all[truncated]
        n_truncated = n[truncated]
        # Whether the likelihood has a maximum at all, and whether it is
        # unique, is a property of the data alone; see _turnbull_npmle.
        npmle, npmle_piece, npmle_reason = npmle_existence(
            lo, hi, w_lo_all, w_hi_all, n, M
        )
    else:
        # No denominators: the likelihood is continuous on the compact
        # simplex and attains its maximum.
        npmle, npmle_piece, npmle_reason = EXISTS, -1, ""

    # The identifiable support: a bound may carry probability mass only if it
    # lies inside at least one observation's support ``[lo, hi]``. Mass placed
    # elsewhere is non-identifiable, and under truncation the ghost step
    # otherwise migrates it below every entry window into a degenerate,
    # all-zero-survival fixed point (issue #203). Restricting the expected
    # counts to this region each iteration keeps the EM in the identifiable
    # part of the parameter space.
    cover = np.zeros(M + 1)
    np.add.at(cover, lo, 1.0)
    np.add.at(cover, np.minimum(hi + 1, M), -1.0)
    identifiable = np.cumsum(cover[:M]) > 0

    # Intervals where extra mass never lowers the likelihood.
    #
    # Every contribution is a ratio, P(support) / P(window). Raising one
    # interval's mass lifts the numerator and the denominator of every
    # observation whose support contains it -- a gain wherever the support
    # is smaller than the window -- and only the denominator of one whose
    # window contains it but whose support does not: the only cost. An
    # interval that no observation pays for in that way, but that some
    # observation gains from, is reported with the share of the fitted
    # mass on it as ``exploitable_mass``.
    #
    # It is a diagnostic, not the verdict. Such an interval makes the
    # NPMLE fail to exist only if the gain is forced at every admissible
    # fit (see ``_turnbull_npmle``), and the share depends on how far the
    # EM got: over 240 simulated left-truncated samples whose NPMLE does
    # not exist it ranged from 0.11 to 0.99 at the default ``max_iter``.
    # The warning used to fire when it passed 0.9 (or the EM did not
    # converge); it now follows ``npmle``.
    exploitable = np.zeros(M, dtype=bool)
    if any_truncated:

        def coverage(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
            # How many of the index ranges [a, b] contain each interval.
            count = np.zeros(M + 1)
            np.add.at(count, a, 1.0)
            np.add.at(count, np.minimum(b + 1, M), -1.0)
            return np.cumsum(count[:M])

        # The supports are already inside the windows, so an interval with
        # as many windows as supports over it is paid for by no one.
        unpaid = coverage(w_lo_all, w_hi_all) == coverage(lo, hi)
        gains = (lo > w_lo_all) | (hi < w_hi_all)
        exploitable = unpaid & (coverage(lo[gains], hi[gains]) > 0)

    d = np.zeros(M)
    if any_truncated and identifiable.any():
        p = identifiable / identifiable.sum()
    else:
        # Without truncation the NPMLE has no mass off the innermost
        # intervals, but the self-consistency EM only drains the mass it
        # starts with there sublinearly: a residue of ~1e-8 expected
        # failures stayed on such pieces, leaving the estimate at 1 - 1e-9
        # and its log(-log) bounds at [0, 1]. Starting on the innermost
        # intervals keeps that mass exactly zero (p = 0 stays 0 under the
        # update) and converges in fewer iterations. Only for the
        # Kaplan-Meier update, which is the NPMLE: the Nelson-Aalen and
        # Fleming-Harrington iterations are not likelihood steps and do
        # settle with real mass on those pieces. Under truncation the
        # dominance argument fails (moving mass changes each window's
        # probability), so the identifiable start above is kept.
        support = _innermost(lo, hi, M)
        if any_truncated or estimator != "Kaplan-Meier" or not support.any():
            support = np.ones(M, dtype=bool)
        p = support / support.sum()

    func = TURNBULL_ESTIMATORS[estimator]

    old_err_state = np.seterr(all="ignore")

    converged = False
    degenerate = False
    r = np.zeros(M)
    R = np.ones(M)
    for iters in range(1, max_iter + 1):
        # Prefix sums of p turn every range sum into two lookups.
        cumulative = np.concatenate([[0.0], np.cumsum(p)])

        # E-step, observed events: each observation distributes its n
        # events over its support in proportion to p, i.e. it adds
        # n * p_j / P(support) to every interval j in [lo, hi]. Summing
        # the weights n / P(support) over observations via a difference
        # array gives all M totals in one cumsum.
        support_p = cumulative[hi + 1] - cumulative[lo]
        # A row whose support carries no mass (or is empty) contributes
        # nothing, rather than propagating inf/nan through the totals.
        weight = np.where(support_p > 0, n / support_p, 0.0)
        delta = np.zeros(M + 1)
        np.add.at(delta, lo, weight)
        np.add.at(delta, hi + 1, -weight)
        d_observed = p * np.cumsum(delta[:M])

        # E-step, ghosts: a truncated observation was only observable
        # because its event fell inside its window, so for every one seen,
        # unseen "ghost" events fell outside it at rate p_j / P(window).
        # Add n / P(window) everywhere, subtract it back over the window.
        # A scalar in the untruncated branch, where it broadcasts over
        # the per-interval counts without allocating an array of zeros.
        d_ghosts: npt.NDArray | float
        if any_truncated:
            window_p = cumulative[w_hi + 1] - cumulative[w_lo]
            ghost_weight = np.where(window_p > 0, n_truncated / window_p, 0.0)
            delta = np.zeros(M + 1)
            delta[0] = ghost_weight.sum()
            np.add.at(delta, w_lo, -ghost_weight)
            np.add.at(delta, w_hi + 1, ghost_weight)
            d_ghosts = p * np.cumsum(delta[:M])
        else:
            d_ghosts = 0.0

        # Deaths/Failures/Events
        d = d_ghosts + d_observed
        if any_truncated:
            # Confine the expected counts to the identifiable region.
            d = np.where(identifiable, d, 0.0)
        # total observed and unobserved failures.
        total_events = d.sum()
        # Risk set, i.e the number of items at risk at immediately before x
        r = total_events - d.cumsum() + d
        # Iterate with the Kaplan-Meier self-consistency update (``p`` ∝
        # ``d``), the canonical Turnbull M-step. The requested hazard-form
        # estimator (Fleming-Harrington / Nelson-Aalen) sets ``R = exp(-H)``,
        # which does *not* satisfy ``p`` ∝ ``d`` -- iterating with it biases
        # every step and leaves truncated fits reporting tol-level
        # non-convergence (issue #203). It is applied only to the converged
        # ladder below. Untruncated fits keep their historical behaviour.
        update = km if any_truncated else func
        R = update(r, d)
        # Calculate the probability mass in each interval
        p_new = np.abs(np.diff(np.hstack([[1], R])))
        # A non-finite update, or (under truncation) a total collapse of mass,
        # is a degenerate fixed point -- not convergence. The old ``nanmax``
        # check silently accepted these.
        if not np.all(np.isfinite(p_new)) or (
            any_truncated and p_new.sum() <= 0
        ):
            degenerate = True
            break
        if any_truncated:
            p_new = p_new / p_new.sum()
        if np.max(np.abs(p_new - p)) < tol:
            p = p_new
            converged = True
            break
        p = p_new

    # Report the requested hazard-form estimator on the converged ladder.
    R = func(r, d)

    # A converged fit whose survival has entirely collapsed (all mass forced
    # to the boundary, so S(x) ~ 0 across the whole observed range) is the
    # non-identifiable degenerate state, not a real estimate.
    if not degenerate and R.size > 2:
        reported = R[:-2]
        if any_truncated:
            # Only inspect the identifiable region: positions before the
            # earliest entry time are pinned at 1.0 and previously masked
            # every partial collapse from the detector (#260).
            min_tl = (
                np.min(tl[np.isfinite(tl)])
                if np.isfinite(tl).any()
                else -np.inf
            )
            idx0 = int(
                np.searchsorted(
                    bounds[: reported.shape[0]], min_tl, side="right"
                )
            )
            inspect = reported[idx0:] if idx0 < reported.shape[0] else reported
        else:
            inspect = reported
        if not np.all(np.isfinite(reported)) or (
            any_truncated and inspect.size > 0 and np.nanmax(inspect) < 1e-8
        ):
            degenerate = True

    # The warnings follow the structural verdict computed before the EM
    # (#327). It replaces a cut-off on ``exploitable_mass`` (warn above 0.9)
    # that was tuned on simulated samples: the verdict needs no constant,
    # and it also catches fits that only *look* settled.
    # The time the verdict rests on: the end of the gap's piece under left
    # truncation (the data do not link the probability before and after
    # it), the start under right truncation (the mirror image).
    at = np.nan
    if npmle_piece >= 0:
        at = bounds[npmle_piece + (0 if npmle_reason == "right" else 1)]
    where = " (at t = {:.4g})".format(at) if np.isfinite(at) else ""
    if degenerate:
        warnings.warn(
            "The Turnbull EM reached a degenerate, non-identifiable fixed "
            "point: all probability mass migrated outside the observable "
            "region (typically below the earliest entry time under heavy "
            "left truncation), so the survival estimate has collapsed. The "
            "result is unreliable -- more data or a narrower truncation range "
            "is needed."
        )
    elif npmle == DOES_NOT_EXIST:
        if npmle_reason == "right":
            cause = (
                "It typically comes from early failures observable only up "
                "to times below the later failures (right truncation), so "
                "nothing links the two; overlapping observation windows "
                "remove it."
            )
        elif npmle_reason == "left":
            cause = (
                "It typically comes from a left- or interval-censored "
                "observation (or an early failure) below the other "
                "observations' entry times; entering every unit at a common "
                "time removes it."
            )
        else:
            cause = (
                "It comes from truncation windows that link the "
                "observations in one direction only: some could have seen "
                "the others' failures, but not the reverse."
            )
        warnings.warn(
            "The Turnbull estimate is not identifiable from this data: the "
            "NPMLE does not exist (`npmle` is {!r}), so the result is "
            "unreliable. The likelihood has no maximum, only a supremum on "
            "the boundary{}, where some observations' truncation windows "
            "carry no probability at all (under left truncation: the "
            "survival drops to zero before their entry). Their "
            "contributions are conditional on their windows, so that costs "
            "them nothing, and the EM climbs towards the boundary instead "
            "of settling; raising `max_iter` will not help. {}".format(
                npmle, where, cause
            )
        )
    elif npmle == NOT_UNIQUE:
        if npmle_reason == "left":
            reason = (
                "no unit that had entered by t = {:.4g} is known to have "
                "survived past it and further units enter only later, so "
                "how much probability lies before and after it is not "
                "determined".format(at)
            )
        elif npmle_reason == "right":
            reason = (
                "no unit still observable at t = {:.4g} is known to have "
                "failed before it and the others are observable only up "
                "to earlier times, so how much probability lies before and "
                "after it is not determined".format(at)
            )
        else:
            reason = (
                "the observations split into groups whose truncation "
                "windows do not overlap, so how much probability each "
                "group's range carries is not determined"
            )
        warnings.warn(
            "The Turnbull estimate is not unique for this data (`npmle` is "
            "{!r}): {}. The likelihood is flat in that direction, and the "
            "returned curve is one of many that fit equally well.".format(
                npmle, reason
            )
        )
    elif not converged:
        hint = ""
        if npmle == UNDETERMINED:
            # Two-sided windows with censoring, where the structure alone
            # does not decide whether the maximum is attained.
            hint = (
                " The data's structure does not guarantee that the NPMLE "
                "exists (`npmle` is {!r}): if a larger `max_iter` does not "
                "help, the maximum may be on the boundary.".format(npmle)
            )
        warnings.warn(
            "The Turnbull EM did not converge to within `tol` ({}) in "
            "`max_iter` ({}) iterations; the estimate may be "
            "inaccurate.{}".format(tol, max_iter, hint)
        )

    if any_truncated:
        # Variance ladder from *observed* counts only. The estimation
        # ladder above includes the ghost events -- they are what make the
        # estimate correct under truncation -- but ghosts are not data, and
        # a risk set inflated by them understates the variance. Exactly
        # observed items count one event at their atom and leave the risk
        # set there; right-censored items count no event anywhere and
        # leave the risk set at their censoring time (the previous ladder
        # redistributed their mass as fractional later events and kept
        # them at risk via a conditional tail probability -- the same
        # anti-conservative mechanism #260 removed for untruncated data,
        # and at the last event the near-equal r and d floats produced
        # huge *negative* Greenwood increments, #273). Only genuinely
        # interval/left-censored items, whose event position is unknown,
        # keep the probabilistic redistribution over their support. Every
        # item is at risk only while the bound lies inside its own
        # observation window, so delayed entry removes it from the early
        # risk sets exactly as in the Kaplan-Meier delayed-entry risk
        # set -- to which this ladder reduces for exact + right-censored
        # left-truncated data.
        cumulative = np.concatenate([[0.0], np.cumsum(p)])
        support_p = cumulative[hi + 1] - cumulative[lo]

        # Events: hard counts at exact atoms; interval rows redistribute.
        d_var = np.zeros(M)
        np.add.at(d_var, lo[exact], n[exact])
        weight_int = np.where(interval & (support_p > 0), n / support_p, 0.0)
        delta = np.zeros(M + 1)
        np.add.at(delta, lo, weight_int)
        np.add.at(delta, hi + 1, -weight_int)
        d_var += p * np.cumsum(delta[:M])

        const = np.zeros(M + 1)
        coeff = np.zeros(M + 1)
        # Exact rows: at risk through their event atom, within the window.
        a1 = w_lo_all
        b1 = np.minimum(lo, w_hi_all)
        ok = exact & (a1 <= b1)
        np.add.at(const, a1[ok], n[ok])
        np.add.at(const, b1[ok] + 1, -n[ok])
        # Right-censored rows: at risk through their censoring time (the
        # last bound before their support starts), within the window.
        b1r = np.minimum(lo - 1, w_hi_all)
        ok = right & (a1 <= b1r)
        np.add.at(const, a1[ok], n[ok])
        np.add.at(const, b1r[ok] + 1, -n[ok])
        # Interval rows: probability 1 before the support, conditional
        # tail (cum[hi+1] - cum[j]) / P(support) inside it, 0 after --
        # all within the window. The j-dependent part is cum[j] times a
        # range-added weight, keeping the ladder O(N + M).
        ok = interval & (a1 <= b1)
        np.add.at(const, a1[ok], n[ok])
        np.add.at(const, b1[ok] + 1, -n[ok])
        a2 = np.maximum(lo + 1, w_lo_all)
        b2 = np.minimum(hi, w_hi_all)
        ok = interval & (a2 <= b2) & (support_p > 0)
        tail_const = np.where(
            support_p > 0, n * cumulative[hi + 1] / support_p, 0.0
        )
        np.add.at(const, a2[ok], tail_const[ok])
        np.add.at(const, b2[ok] + 1, -tail_const[ok])
        weight = np.where(support_p > 0, n / support_p, 0.0)
        np.add.at(coeff, a2[ok], weight[ok])
        np.add.at(coeff, b2[ok] + 1, -weight[ok])

        r_var = np.cumsum(const[:M]) - np.cumsum(coeff[:M]) * cumulative[:M]

    # Heterogeneous by design: arrays, the estimator name, and the
    # convergence flags all go out in the one dictionary.
    #
    # Ladder index j is the piece ``(bounds[j], bounds[j+1]]``, so the
    # survival after it, ``R[j]``, is reported at its *right* end,
    # ``bounds[j+1]`` -- hence ``x = bounds[1:-1]`` with ``R[:-2]``. The
    # counts that produce ``R[j]`` must go out on the same index: slicing
    # them ``[1:-1]`` instead paired each x with the *next* piece's
    # failures, so the variance had already stepped where the estimate had
    # not yet dropped, and ``cb()`` on interval-censored data gave bounds
    # like [0, 1] where the survival estimate was still 1.
    out: dict[str, Any] = {}
    out["x"] = bounds[1:-1]
    out["r"] = r[:-2]
    out["d"] = d[:-2]
    if any_truncated:
        out["var_r"] = r_var[:-2]
        out["var_d"] = d_var[:-2]
    elif km_reducible:
        # Variance from the observed counts (the Greenwood ladder): the
        # estimation ladder redistributes each right-censored observation
        # as fractional expected events at later times, inflating the
        # information and giving silently narrower intervals (#260). Only
        # positions with observed events contribute to the variance sum.
        from surpyval.utils import xcnt_to_xrd

        xg, rg, dg = xcnt_to_xrd(x_raw, c_raw, n_raw, t_raw)
        ladder_x = bounds[1:-1]
        var_d = np.zeros(ladder_x.shape[0])
        var_r = np.ones(ladder_x.shape[0])
        pos = np.searchsorted(xg, ladder_x)
        ok = pos < xg.shape[0]
        ok[ok] = np.isclose(xg[pos[ok]], ladder_x[ok])
        var_r[ok] = rg[pos[ok]]
        # Exact times appear twice on the bounds ladder (the zero-width
        # [x, x] interval trick); credit each event count once so the
        # cumulative variance steps once per event time, and on the
        # *second* copy -- where the estimate drops -- since the first
        # carries the survival just before the event.
        last = np.ones(ladder_x.shape[0], dtype=bool)
        last[:-1] = ladder_x[:-1] != ladder_x[1:]
        take = ok & last
        var_d[take] = dg[pos[take]]
        out["var_r"] = var_r
        out["var_d"] = var_d
    out["R"] = R[0:-2]
    out["F"] = 1 - R[0:-2]
    out["R_upper"] = R[0:-2]
    out["R_lower"] = R[1:-1]
    out["bounds"] = bounds
    out["model"] = "Turnbull"
    out["turnbull_estimator"] = estimator
    out["iters"] = iters
    out["converged"] = converged
    out["degenerate"] = degenerate
    # How much of the fitted mass landed where the likelihood can be
    # inflated for free (see ``exploitable`` above). Reported so a caller
    # can judge a fit that sits largely on that free direction; data
    # without such pieces report exactly zero.
    out["exploitable_mass"] = (
        float(p[exploitable].sum()) if exploitable.any() else 0.0
    )
    out["npmle"] = npmle

    np.seterr(**old_err_state)

    return out


class Turnbull_(NonParametricFitter):
    r"""
    Turnbull estimator class. Returns a `NonParametric` object from method
    :code:`fit()`. Calculates the Non-Parametric estimate of the survival
    function using the Turnbull NPMLE.

    The EM iterates until the largest change in any interval's probability
    mass falls below ``tol`` or ``max_iter`` iterations have run (with a
    warning in the latter case); both can be passed to :code:`fit()`.

    Besides the attributes every non-parametric model has, a Turnbull
    model carries:

    - ``bounds``: the endpoints of the Turnbull pieces, including
      :math:`\pm\infty`; ``x`` is ``bounds[1:-1]``, so an exactly observed
      time appears twice, and ``d[k]`` is the expected number of failures
      in the piece ending at ``x[k]``;
    - ``R_upper`` and ``R_lower``: the survival at the start and end of
      each piece, the range any curve through it could take;
    - ``turnbull_estimator``, ``converged`` and ``iters``;
    - ``degenerate``: True if the estimate collapsed (with a warning);
    - ``npmle``: whether the likelihood has a maximum, decided from the
      data before the EM runs (#327). ``"exists"``: it does (always the
      case without truncation). ``"not unique"``: it does, but the data
      leave the probability on one side of some time (or of some group of
      observations whose truncation windows do not overlap the others')
      free, so many curves fit equally well; a warning says so. ``"does
      not exist"``: the likelihood only approaches its supremum as the
      survival drops to zero before some observations' entry, the EM
      cannot settle and a warning says the estimate is not identifiable.
      ``"undetermined"``: two-sided truncation windows with censoring,
      where the structure alone does not decide (the EM's convergence is
      then the guide). The derivation, and what "exists" does not rule
      out, are in ``surpyval/univariate/nonparametric/_turnbull_npmle.py``;
    - ``exploitable_mass``: the share of the fitted mass in pieces where
      mass never lowers the likelihood -- some observation could have
      failed there, and every observation whose truncation window covers
      the piece could too. A diagnostic: it is zero when no such piece
      exists, but its size depends on how far the EM got, so the warnings
      follow ``npmle`` instead.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import Turnbull
    >>> x = np.array([[1, 5], [2, 3], [3, 6], [1, 8], [9, 10]])
    >>> model = Turnbull.fit(x)
    >>> model.R
    array([1.        , 1.        , 0.63472351, 0.29479882, 0.2631432 ,
           0.2631432 , 0.2631432 , 0.09680497])
    """

    def __init__(self) -> None:
        self.how = "Turnbull"

    def _fit(
        self,
        x: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike,
        t: npt.ArrayLike,
        turnbull_estimator: str,
        tol: float,
        max_iter: int,
    ) -> dict:
        return turnbull(
            x, c, n, t, turnbull_estimator, tol=tol, max_iter=max_iter
        )


Turnbull = Turnbull_()
