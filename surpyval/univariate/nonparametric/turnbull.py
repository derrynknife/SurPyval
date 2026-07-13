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

    d = np.zeros(M)
    p = np.ones(M) / M

    if estimator == "Kaplan-Meier":
        func = km
    elif estimator == "Nelson-Aalen":
        func = na
    else:
        func = fh

    old_err_state = np.seterr(all="ignore")

    converged = False
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
        # total observed and unobserved failures.
        total_events = d.sum()
        # Risk set, i.e the number of items at risk at immediately before x
        r = total_events - d.cumsum() + d
        # Find the survival function values (R) using the deaths and risk set
        # The 'official' way to do it, which is equivalent to using KM,
        # is to do p = (nu + mu).sum(axis=0)/(nu + mu).sum()
        R = func(r, d)
        # Calculate the probability mass in each interval
        p_new = np.abs(np.diff(np.hstack([[1], R])))
        if np.nanmax(np.abs(p_new - p)) < tol:
            p = p_new
            converged = True
            break
        p = p_new

    if not converged:
        warnings.warn(
            "The Turnbull EM did not converge to within `tol` ({}) in "
            "`max_iter` ({}) iterations; the estimate may be "
            "inaccurate.".format(tol, max_iter)
        )

    out = {}
    out["x"] = bounds[1:-1]
    out["r"] = r[1:-1]
    out["d"] = d[1:-1]
    out["R"] = R[0:-2]
    out["F"] = 1 - R[0:-2]
    out["R_upper"] = R[0:-2]
    out["R_lower"] = R[1:-1]
    out["bounds"] = bounds
    out["model"] = "Turnbull"
    out["turnbull_estimator"] = estimator
    out["iters"] = iters
    out["converged"] = converged

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
