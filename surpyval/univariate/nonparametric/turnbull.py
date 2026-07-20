import warnings

import numpy as np

from surpyval.univariate.nonparametric.nonparametric_fitter import (
    NonParametricFitter,
)

from .fleming_harrington import fleming_harrington as fh
from .kaplan_meier import kaplan_meier as km
from .nelson_aalen import nelson_aalen as na


def turnbull(
    x, c, n, t, estimator="Fleming-Harrington", tol=1e-10, max_iter=1000
):
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
    #   is (-inf, xr)) may sit on any bound in [xl, xr), *except* the
    #   zero-width exact interval at xl when xl is also an exactly observed
    #   time -- the event is known to be after xl.
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
    hi[interval] = np.searchsorted(bounds, xr[interval], side="left") - 1

    # Each observation's truncation window is likewise a contiguous index
    # range [w_lo, w_hi]: the bound points at which an event was observable
    # -- strictly after its left truncation time and at or before its right
    # truncation time.
    if any_truncated:
        w_lo = np.where(
            np.isfinite(tl), np.searchsorted(bounds, tl, side="right"), 0
        )
        w_hi = np.where(
            np.isfinite(tr),
            np.searchsorted(bounds, tr, side="right") - 1,
            M - 1,
        )
        truncated = np.isfinite(tl) | np.isfinite(tr)
        w_lo, w_hi = w_lo[truncated], w_hi[truncated]
        n_truncated = n[truncated]

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

    d = np.zeros(M)
    if any_truncated and identifiable.any():
        p = identifiable / identifiable.sum()
    else:
        p = np.ones(M) / M

    if estimator == "Kaplan-Meier":
        func = km
    elif estimator == "Nelson-Aalen":
        func = na
    else:
        func = fh

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
        if not np.all(np.isfinite(reported)) or (
            any_truncated and np.nanmax(reported) < 1e-8
        ):
            degenerate = True

    if degenerate:
        warnings.warn(
            "The Turnbull EM reached a degenerate, non-identifiable fixed "
            "point: all probability mass migrated outside the observable "
            "region (typically below the earliest entry time under heavy "
            "left truncation), so the survival estimate has collapsed. The "
            "result is unreliable -- more data or a narrower truncation range "
            "is needed."
        )
    elif not converged:
        warnings.warn(
            "The Turnbull EM did not converge to within `tol` ({}) in "
            "`max_iter` ({}) iterations; the estimate may be "
            "inaccurate.".format(tol, max_iter)
        )

    if any_truncated:
        # Variance ladder from *observed* information only. The estimation
        # ladder above includes the ghost events -- they are what make the
        # estimate correct under truncation -- but ghosts are not data, and
        # a risk set inflated by them understates the variance. Instead,
        # each observed item contributes its conditional probability of
        # still being event-free at each bound, and only while that bound
        # lies inside its observation window (so delayed entry removes it
        # from the early risk sets, exactly as in the Kaplan-Meier
        # delayed-entry risk set, to which this reduces for exactly
        # observed left-truncated data):
        #
        #   r_var[j] = sum_i n_i * 1[w_lo_i <= j <= w_hi_i] * P_i(T >= j)
        #
        # with P_i(T >= j) equal to 1 before the item's support, the
        # conditional tail (cum[hi+1] - cum[j]) / P(support) inside it, and
        # 0 after it. Piece one and the constant part of piece two are
        # range-adds; the j-dependent part is cum[j] times a range-added
        # weight -- all still O(N + M). For untruncated data this ladder
        # equals the estimation ladder, so it is only computed (and only
        # used for the variance) when truncation is present.
        cumulative = np.concatenate([[0.0], np.cumsum(p)])
        support_p = cumulative[hi + 1] - cumulative[lo]
        weight = np.where(support_p > 0, n / support_p, 0.0)
        delta = np.zeros(M + 1)
        np.add.at(delta, lo, weight)
        np.add.at(delta, hi + 1, -weight)
        d_var = p * np.cumsum(delta[:M])

        w_lo_all = np.where(
            np.isfinite(tl), np.searchsorted(bounds, tl, side="right"), 0
        )
        w_hi_all = np.where(
            np.isfinite(tr),
            np.searchsorted(bounds, tr, side="right") - 1,
            M - 1,
        )

        const = np.zeros(M + 1)
        coeff = np.zeros(M + 1)
        # Before the support (probability 1), within the window.
        a1 = w_lo_all
        b1 = np.minimum(lo, w_hi_all)
        ok = a1 <= b1
        np.add.at(const, a1[ok], n[ok])
        np.add.at(const, b1[ok] + 1, -n[ok])
        # Within the support (conditional tail), within the window.
        a2 = np.maximum(lo + 1, w_lo_all)
        b2 = np.minimum(hi, w_hi_all)
        ok = (a2 <= b2) & (support_p > 0)
        tail_const = np.where(
            support_p > 0, n * cumulative[hi + 1] / support_p, 0.0
        )
        np.add.at(const, a2[ok], tail_const[ok])
        np.add.at(const, b2[ok] + 1, -tail_const[ok])
        np.add.at(coeff, a2[ok], weight[ok])
        np.add.at(coeff, b2[ok] + 1, -weight[ok])

        r_var = np.cumsum(const[:M]) - np.cumsum(coeff[:M]) * cumulative[:M]

    out = {}
    out["x"] = bounds[1:-1]
    out["r"] = r[1:-1]
    out["d"] = d[1:-1]
    if any_truncated:
        out["var_r"] = r_var[1:-1]
        out["var_d"] = d_var[1:-1]
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

    def __init__(self):
        self.how = "Turnbull"

    def _fit(self, x, c, n, t, turnbull_estimator, tol, max_iter):
        return turnbull(
            x, c, n, t, turnbull_estimator, tol=tol, max_iter=max_iter
        )


Turnbull = Turnbull_()
