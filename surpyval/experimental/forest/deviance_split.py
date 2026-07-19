r"""Full-likelihood (exponential deviance) split for survival trees.

The risk-set log-rank split (see ``log_rank_split``) is defined only for
data that can be expressed in the xrd (risk set / death count) format:
observed and right-censored observations, optionally with delayed entry
(left truncation). Left-censored, interval-censored and right-truncated
observations carry their event information as *probabilities of intervals*
rather than as points entering and leaving a risk set, so no risk-set
statistic exists for them.

The likelihood does not have that limitation. This module implements the
classic parametric alternative -- the exponential log-likelihood
(deviance) split of Davis & Anderson (1989), scored with SurPyval's full
likelihood so that every observation type contributes exactly:

- observed ``x``:            :math:`\log \lambda - \lambda x`
- right censored at ``x``:   :math:`-\lambda x`
- left censored at ``x``:    :math:`\log(1 - e^{-\lambda x})`
- interval ``(x_l, x_r]``:   :math:`\log(e^{-\lambda x_l} - e^{-\lambda x_r})`
- truncated to ``(t_l, t_r]``: minus
  :math:`\log(S(t_l) - S(t_r))`

A candidate split is scored by the joint maximised log-likelihood of its
two children; since the parent's log-likelihood is constant within a node,
maximising :math:`\ell_L^* + \ell_R^*` maximises the deviance gain
:math:`2(\ell_L^* + \ell_R^* - \ell_{parent}^*)`. For purely observed /
right-censored data this rule and the log-rank rule pick very similar
splits; the deviance rule simply remains defined for everything else.

References
----------
Davis, R.B. and Anderson, J.R., 1989. Exponential survival trees.
*Statistics in Medicine*, 8(8), pp.947-961.
"""

from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar

from surpyval.utils.surpyval_data import SurpyvalData

# Cap on candidate split values per feature: above this, candidates are
# thinned to quantiles so a node's split search stays O(cap * n).
_MAX_SPLIT_CANDIDATES = 64

# Value used for non-finite likelihoods so the bounded optimiser can keep
# working (it cannot handle inf/nan).
_HUGE = 1e300


def needs_full_likelihood_split(data: SurpyvalData) -> bool:
    """
    True when ``data`` cannot be expressed in the xrd format -- i.e. it
    contains left censoring, interval censoring, or right truncation --
    so the risk-set log-rank split is undefined and the full-likelihood
    (deviance) split must be used.
    """
    return bool(
        np.isfinite(data.t[:, 1]).any()
        or (data.c == 2).any()
        or (data.c == -1).any()
    )


def _exp_neg_ll_parts(data: SurpyvalData):
    """
    Pre-extract the arrays the exponential negative log-likelihood needs,
    so the optimiser's objective is pure vectorised arithmetic.
    """
    a_trunc = np.maximum(data.x_tl, 0.0)  # exponential support starts at 0
    return (
        data.x_o,
        data.n_o,
        data.x_r,
        data.n_r,
        data.x_l,
        data.n_l,
        data.x_il,
        data.x_ir,
        data.n_i,
        a_trunc,
        data.x_tr,
        data.n_t,
    )


def _exp_neg_ll(theta: float, parts) -> float:
    """
    Negative log-likelihood of an exponential distribution with rate
    ``lambda = exp(theta)`` under the full data model (all censoring
    types plus truncation). Returns a large finite value where the
    likelihood is degenerate so bounded optimisation stays stable.
    """
    (
        x_o,
        n_o,
        x_r,
        n_r,
        x_l,
        n_l,
        x_il,
        x_ir,
        n_i,
        t_a,
        t_r,
        n_t,
    ) = parts
    lam = np.exp(theta)
    ll = 0.0
    with np.errstate(all="ignore"):
        if x_o.size:
            ll += np.sum(n_o * (theta - lam * x_o))
        if x_r.size:
            ll += -lam * np.sum(n_r * x_r)
        if x_l.size:
            # F(x) = 1 - exp(-lam x)
            ll += np.sum(n_l * np.log(-np.expm1(-lam * x_l)))
        if x_il.size:
            # S(xl) - S(xr) = exp(-lam xl)(1 - exp(-lam (xr - xl)))
            ll += np.sum(
                n_i * (-lam * x_il + np.log(-np.expm1(-lam * (x_ir - x_il))))
            )
        if t_a.size:
            # subtract log(S(tl) - S(tr)) per truncated observation
            finite_tr = np.isfinite(t_r)
            log_denom = -lam * t_a
            if finite_tr.any():
                width = np.where(finite_tr, t_r - t_a, np.inf)
                log_denom = log_denom + np.where(
                    finite_tr, np.log(-np.expm1(-lam * width)), 0.0
                )
            ll -= np.sum(n_t * log_denom)
    if not np.isfinite(ll):
        return _HUGE
    return -float(ll)


def _exp_theta0(data: SurpyvalData) -> "float | None":
    """
    Crude ``log(lambda)`` scale for centring the search window:
    event-weight over exposure, or ``None`` when the data carries no
    event information.
    """
    exposure = 0.0
    events = 0.0
    if data.x_o.size:
        exposure += float(np.sum(data.n_o * data.x_o))
        events += float(np.sum(data.n_o))
    if data.x_r.size:
        exposure += float(np.sum(data.n_r * data.x_r))
    if data.x_l.size:
        exposure += float(np.sum(data.n_l * data.x_l)) / 2.0
        events += float(np.sum(data.n_l))
    if data.x_il.size:
        exposure += float(np.sum(data.n_i * (data.x_il + data.x_ir))) / 2.0
        events += float(np.sum(data.n_i))
    if exposure <= 0 or events <= 0:
        return None
    return float(np.log(events / exposure))


def _exp_max_ll(
    data: SurpyvalData, bounds: "tuple[float, float] | None" = None
) -> float:
    """
    Maximised exponential log-likelihood of ``data`` under the full data
    model, via a bounded scalar search over ``log(lambda)``.

    ``bounds`` fixes the search window. Within a node, every candidate
    child must be scored over the SAME window as the parent: for
    degenerate data directions (e.g. a likelihood whose supremum sits at
    ``lambda -> 0``) the attained value depends on where the window
    ends, so per-subset windows would make the child and parent scores
    incomparable. A common window guarantees, by additivity of the
    log-likelihood over a partition, that a split never scores below
    its parent.
    """
    parts = _exp_neg_ll_parts(data)

    if bounds is None:
        theta0 = _exp_theta0(data)
        if theta0 is None:
            return -_HUGE
        bounds = (theta0 - 15.0, theta0 + 15.0)

    res = minimize_scalar(
        _exp_neg_ll,
        args=(parts,),
        bounds=bounds,
        method="bounded",
        options={"xatol": 1e-6},
    )
    # The bounded method can converge slightly inside a boundary
    # supremum; take the better of the optimiser's answer and the
    # window edges.
    best = min(
        float(res.fun),
        _exp_neg_ll(bounds[0], parts),
        _exp_neg_ll(bounds[1], parts),
    )
    return -best


def _candidate_values(Z_u: NDArray) -> NDArray:
    """Unique candidate split values, quantile-thinned for large nodes."""
    values = np.unique(Z_u)
    if values.size > _MAX_SPLIT_CANDIDATES:
        quantiles = np.linspace(0, 1, _MAX_SPLIT_CANDIDATES)
        values = np.unique(np.quantile(Z_u, quantiles))
    return values


def deviance_split(
    data: SurpyvalData,
    Z: NDArray,
    min_leaf_samples: int,
    min_leaf_failures: int,
    feature_indices_in: Iterable[int],
) -> tuple[int, float]:
    """
    Best ``(feature index, value)`` split by the exponential deviance
    criterion under the full likelihood.

    Mirrors ``log_rank_split``'s contract: candidates leaving either
    child with fewer than ``min_leaf_samples`` observations or fewer
    than ``min_leaf_failures`` event-informative observations (any
    observation that is not purely right censored, weighted by ``n``)
    are discarded, and ``(-1, -inf)`` is returned when no candidate
    survives.

    Parameters
    ----------
    data : SurpyvalData
        Survival data in the full xcnt data model: observed, left, right
        and interval censoring, with optional left and/or right
        truncation.
    Z : NDArray
        Covariate matrix, shape ``(n_samples, n_features)``.
    min_leaf_samples : int
        Minimum number of samples each child must keep.
    min_leaf_failures : int
        Minimum ``n``-weighted count of event-informative observations
        (``c != 1``) each child must keep.
    feature_indices_in : Iterable[int]
        Indices of the features considered for the split.

    Returns
    -------
    tuple[int, float]
        The best feature index and split value, or ``(-1, -inf)`` if no
        valid split exists.
    """
    best_score = -np.inf
    best_u = -1
    best_v = -float("inf")

    event_weight = data.n * (data.c != 1)

    # All candidates -- and both children of each -- are scored over the
    # parent's search window, so their maximised log-likelihoods are
    # directly comparable (see _exp_max_ll).
    theta0 = _exp_theta0(data)
    if theta0 is None:
        return best_u, best_v
    bounds = (theta0 - 15.0, theta0 + 15.0)
    parent_ll = _exp_max_ll(data, bounds)

    for u in feature_indices_in:
        Z_u = Z[:, u]
        for v in _candidate_values(Z_u):
            mask = Z_u <= v
            n_left = int(mask.sum())
            if n_left < min_leaf_samples:
                continue
            if (mask.size - n_left) < min_leaf_samples:
                continue
            if event_weight[mask].sum() < min_leaf_failures:
                continue
            if event_weight[~mask].sum() < min_leaf_failures:
                continue

            score = _exp_max_ll(data[mask], bounds) + _exp_max_ll(
                data[~mask], bounds
            )
            if score > best_score:
                best_score = score
                best_u = int(u)
                best_v = float(v)

    # A split that does not improve on the parent's log-likelihood by a
    # meaningful margin carries no information (for fully degenerate
    # data every partition scores exactly the parent value); stop
    # rather than split arbitrarily.
    if best_u != -1 and best_score <= parent_ll + 1e-6:
        return -1, -float("inf")

    return best_u, best_v
