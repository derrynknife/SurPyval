r"""Turnbull-score (log-rank score) split for left- and interval-censored
and right-truncated data.

The risk-set log-rank split (``log_rank_split``) needs every observation
to enter and leave a risk set at known times, which left- and
interval-censored observations do not, and which right-truncated ones
(seen only because they failed in time) do not either. The log-rank
test itself does not
need that: it is the score test of a proportional hazards effect, and the
score of an observation whose event lies in :math:`(L, R]` is

.. math::

    c = \frac{S(L)\log S(L) - S(R)\log S(R)}{S(L) - S(R)},

with :math:`S` the pooled survival function and :math:`0 \log 0 = 0`
(Peto and Peto, 1972; Finkelstein, 1986). Its limits are the familiar
scores: :math:`c = \log S(L)` for a row right censored at :math:`L`
(:math:`R \to \infty`), :math:`c = 1 + \log S(t)` for an event observed
at :math:`t` (:math:`L \to R`), and
:math:`c = -S(R)\log S(R) / (1 - S(R))` for a row left censored at
:math:`R` (:math:`L \to 0`).

At each node the pooled Turnbull NPMLE is fitted once, and :math:`S` is
taken as :math:`e^{-H}`, with :math:`H` the Nelson-Aalen cumulative hazard
of the NPMLE's expected risk sets and events. :math:`H` is the NPMLE's own
hazard, so this is the NPMLE's survival up to the ties correction, but it
is positive everywhere (a Kaplan-Meier type estimate reaches zero at a
last observed failure, where the score :math:`1 + \log S` is minus
infinity), and on right-censored data, where the NPMLE is the Kaplan-Meier
estimate and its risk sets and events are the observed ones, :math:`H` is
exactly the Nelson-Aalen estimate. The scores are then the classic
log-rank (Savage) scores :math:`\delta_i - H(x_i)`, and their sum over a
child is exactly the log-rank statistic's numerator
:math:`O - E`.

A candidate split is scored by the standardised sum of the scores in the
left child, a two-sample linear rank statistic, with its permutation
variance:

.. math::

    \frac{\left|\sum_{i \in L} c_i - n_L \bar c\right|}
    {\sqrt{\frac{n_L n_R}{n(n - 1)} \sum_i (c_i - \bar c)^2}},

counting a row with count :math:`n_i` as :math:`n_i` units. The
permutation variance takes the place of the log-rank's hypergeometric
variance; the two agree asymptotically, so on right-censored data the
split is asymptotically equivalent to the log-rank split. The scores are
computed once per node, so each candidate costs O(1) after a sort per
feature.

**Truncation.** A row observable only if its event falls in a window
:math:`(t_l, t_r]` contributes the conditional likelihood
:math:`P(L < T \leq R) / P(t_l < T \leq t_r)`, and its log-rank score is
the score of that contribution: the derivative at :math:`\theta = 1` of
its logarithm along the proportional hazards path :math:`S^\theta`, as
the scores above are of the untruncated likelihood. With
:math:`g(a, b) = [S(a)\log S(a) - S(b)\log S(b)] / [S(a) - S(b)]`,

.. math::

    c = g(\max(L, t_l), \min(R, t_r)) - g(t_l, t_r):

the score of the event, which lies in the window, less that of the
window. An untruncated row has :math:`g(-\infty, \infty) = 0` and keeps
its score. Under left truncation alone the window term is
:math:`\log S(t_l)`, so the scores are the martingale residuals
:math:`\delta_i - [H(x_i) - H(t_{l,i})]` of the delayed-entry
Nelson-Aalen estimate on observed and right-censored data (Fu and
Simonoff, 2017), and their sum over a child is again the log-rank
numerator :math:`O - E` with delayed-entry risk sets. :math:`S` comes from
the same pooled Turnbull estimate, fitted with the truncation (Turnbull,
1976), whose expected risk sets include the units the windows hid; its
hazard is the NPMLE's under any truncation.

The window term is what makes the scores right. A score of a conditional
likelihood has mean zero given the row's window, so under no effect the
scores are unrelated to the covariates even when the windows depend on
them (a covariate that only delays entry, say); the event term alone
does not have that property and would split on such a covariate. The
alternative the test has power against is proportional hazards in the
distribution the data identify: under right truncation that is the
distribution conditional on failing before the largest truncation time,
which is what the NPMLE estimates (it puts no probability after it).

References
----------
Finkelstein, D.M., 1986. A proportional hazards model for
interval-censored failure time data. *Biometrics*, 42(4), pp.845-854.

Peto, R. and Peto, J., 1972. Asymptotically efficient rank invariant test
procedures. *Journal of the Royal Statistical Society, Series A*, 135(2),
pp.185-207.

Fay, M.P. and Shaw, P.A., 2010. Exact and asymptotic weighted logrank
tests for interval censored data: the interval R package. *Journal of
Statistical Software*, 36(2), pp.1-34.

Turnbull, B.W., 1976. The empirical distribution function with
arbitrarily grouped, censored and truncated data. *Journal of the Royal
Statistical Society, Series B*, 38(3), pp.290-295.

Fu, W. and Simonoff, J.S., 2017. Survival trees for left-truncated and
right-censored data, with application to time-varying covariate data.
*Biostatistics*, 18(2), pp.352-369.
"""

import warnings
from collections.abc import Callable, Iterable

import numpy as np
from numpy.typing import NDArray

from surpyval.univariate.nonparametric.turnbull import turnbull
from surpyval.utils.surpyval_data import SurpyvalData


def pooled_cumulative_hazard(data: SurpyvalData) -> tuple[NDArray, NDArray]:
    """The ladder of the pooled Turnbull NPMLE of ``data``: its times and
    the Nelson-Aalen cumulative hazard of its expected risk sets and
    events at each (the hazard just after the time)."""
    with warnings.catch_warnings():
        # The EM converges slowly where the NPMLE is not unique; the
        # scores only rank candidate splits, and a nearly converged
        # estimate ranks them the same.
        warnings.simplefilter("ignore", UserWarning)
        out = turnbull(
            data.x, data.c, data.n, data.t, estimator="Kaplan-Meier"
        )
    r = np.asarray(out["r"], dtype=float)
    d = np.asarray(out["d"], dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        steps = np.where(r > 0, d / r, 0.0)
    return np.asarray(out["x"], dtype=float), np.cumsum(steps)


def _interval_scores(
    H_at: Callable[[NDArray], NDArray],
    lo: NDArray,
    hi: NDArray,
    exact: NDArray,
) -> NDArray:
    r""":math:`g(a, b) = [S(a)\log S(a) - S(b)\log S(b)] / [S(a) - S(b)]`
    for each interval :math:`(a, b]`, with :math:`S = e^{-H}`; the limit
    :math:`1 + \log S(a) = 1 - H(a)` where ``exact`` (an event observed at
    ``a``) or the interval carries no probability."""
    H_lo, H_hi = H_at(lo), H_at(hi)
    S_lo, S_hi = np.exp(-H_lo), np.exp(-H_hi)
    # S log S = -H exp(-H), 0 where S = 0 (H infinite).
    with np.errstate(invalid="ignore"):
        g_lo = np.where(np.isinf(H_lo), 0.0, -H_lo * S_lo)
        g_hi = np.where(np.isinf(H_hi), 0.0, -H_hi * S_hi)
    width = S_lo - S_hi
    # An exact event, or an interval the estimate puts no probability in,
    # takes the limit: the derivative of S log S, 1 + log S = 1 - H.
    point = exact | (width <= 1e-12 * np.maximum(S_lo, 1e-300))
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(point, 1.0 - H_lo, (g_lo - g_hi) / width)


def log_rank_scores(data: SurpyvalData) -> NDArray:
    r"""The log-rank score of each row of ``data`` under its pooled
    Turnbull estimate (see the module docstring):
    :math:`g(L, R) = [S(L)\log S(L) - S(R)\log S(R)] / [S(L) - S(R)]` for
    an event in :math:`(L, R]`, with :math:`S = e^{-H}`, less
    :math:`g(t_l, t_r)` for a row truncated to :math:`(t_l, t_r]`, the
    event's interval then being its intersection with the window."""
    times, H = pooled_cumulative_hazard(data)

    def H_at(q: NDArray) -> NDArray:
        # The step function: H just after q; 0 before the first time and
        # infinite at +inf (S = 0 there).
        idx = np.searchsorted(times, q, side="right") - 1
        out = np.where(idx >= 0, H[np.maximum(idx, 0)], 0.0)
        return np.where(np.isposinf(q), np.inf, out)

    x = np.asarray(data.x, dtype=float)
    c = np.asarray(data.c)
    tl = np.asarray(data.t[:, 0], dtype=float)
    tr = np.asarray(data.t[:, 1], dtype=float)
    lo = x if x.ndim == 1 else x[:, 0]
    hi = x if x.ndim == 1 else x[:, -1]
    # The event's interval, within the window it was observable in: a row
    # left censored at R with entry at t_l failed in (t_l, R], one right
    # censored at L and truncated at t_r in (L, t_r].
    lo = np.maximum(np.where(c == -1, -np.inf, lo), tl)
    hi = np.minimum(np.where(c == 1, np.inf, hi), tr)
    scores = _interval_scores(H_at, lo, hi, c == 0)
    truncated = np.isfinite(tl) | np.isfinite(tr)
    if truncated.any():
        # Less the score of the window: g(-inf, inf) = 0 untruncated.
        window = _interval_scores(
            H_at, tl[truncated], tr[truncated], np.zeros(truncated.sum(), bool)
        )
        scores[truncated] -= window
    return scores


def turnbull_score(u: int, v: float, data: SurpyvalData, Z: NDArray) -> float:
    """The standardised score statistic of one split, feature ``u`` at
    most ``v`` against the rest (see the module docstring); the
    statistic :func:`turnbull_score_split` maximises. NaN if a child is
    empty or the scores are all equal."""
    scores = log_rank_scores(data)
    n = np.asarray(data.n, dtype=float)
    N = n.sum()
    left = Z[:, u] <= v
    n_left = n[left].sum()
    c_bar = np.sum(n * scores) / N
    variance = (
        n_left
        * (N - n_left)
        / (N * (N - 1.0))
        * np.sum(n * (scores - c_bar) ** 2)
    )
    if not variance > 0:
        return float("nan")
    return float(
        abs(np.sum(n[left] * scores[left]) - n_left * c_bar)
        / np.sqrt(variance)
    )


def turnbull_score_split(
    data: SurpyvalData,
    Z: NDArray,
    min_leaf_samples: int,
    min_leaf_failures: int,
    feature_indices_in: Iterable[int],
) -> tuple[int, float]:
    """
    The feature index and value of the split with the largest
    standardised log-rank score statistic (see the module docstring).

    Parameters
    ----------
    data : SurpyvalData
        Survival data (x, c, n, t): any censoring and truncation.
    Z : NDArray
        Covariate matrix, of shape (n_samples, n_features).
    min_leaf_samples : int
        Minimum number of rows each child must have.
    min_leaf_failures : int
        Minimum ``n``-weighted number of rows that are not right censored
        each child must have.
    feature_indices_in : Iterable[int]
        Indices of the features to consider for the split.

    Returns
    -------
    tuple[int, float]
        The feature index and value of the best split: the left child
        has the rows with feature at most the value. ``(-1, -inf)`` if no
        split satisfies the constraints or the scores carry no
        information (all equal).
    """
    best_u, best_v = -1, -float("inf")
    scores = log_rank_scores(data)
    n = np.asarray(data.n, dtype=float)
    N = n.sum()
    if N < 2:
        return best_u, best_v
    c_bar = np.sum(n * scores) / N
    spread = np.sum(n * (scores - c_bar) ** 2)
    if not np.isfinite(spread) or spread <= 0:
        return best_u, best_v
    # Failures n-weighted, as every split counts them (#193)
    event = (np.asarray(data.c) != 1) * n
    n_rows = scores.size
    n_events = event.sum()

    best = -float("inf")
    for u in feature_indices_in:
        order = np.argsort(Z[:, u], kind="stable")
        z = Z[order, u]
        # Cut after each last occurrence of a value: the left child is
        # every row with the feature at most that value.
        cut = np.flatnonzero(z[:-1] != z[1:])
        if cut.size == 0:
            continue
        rows_left = cut + 1.0
        n_left = np.cumsum(n[order])[cut]
        sum_left = np.cumsum((n * scores)[order])[cut]
        events_left = np.cumsum(event[order])[cut]
        ok = (
            (rows_left >= min_leaf_samples)
            & (n_rows - rows_left >= min_leaf_samples)
            & (events_left >= min_leaf_failures)
            & (n_events - events_left >= min_leaf_failures)
        )
        if not ok.any():
            continue
        n_right = N - n_left
        variance = n_left * n_right / (N * (N - 1.0)) * spread
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic = np.abs(sum_left - n_left * c_bar) / np.sqrt(variance)
        statistic = np.where(ok & (variance > 0), statistic, -np.inf)
        k = int(np.argmax(statistic))
        if statistic[k] > best:
            best = float(statistic[k])
            best_u, best_v = int(u), float(z[cut[k]])
    return best_u, best_v
