# This code was created for and sponsored by Cartiga (www.cartiga.com).
# Cartiga makes no representations or warranties in connection with the code
# and waives any and all liability in connection therewith. Your use of the
# code constitutes acceptance of these terms.

# Copyright 2022 Cartiga LLC


from copy import copy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.ma as ma
import numpy.typing as npt
from numpy.linalg import inv, pinv
from scipy.optimize import minimize, root
from scipy.stats import norm

if TYPE_CHECKING:
    import pandas as pd

from surpyval.univariate.nonparametric import (
    FlemingHarrington,
    KaplanMeier,
    NelsonAalen,
    Turnbull,
)
from surpyval.utils import validate_coxph, validate_coxph_df_inputs

from ..semi_parametric_regression_model import SemiParametricRegressionModel
from .tvc import handle_tvc, handle_tvc_timeline

nonparametric_dists = {
    "Nelson-Aalen": NelsonAalen,
    "Kaplan-Meier": KaplanMeier,
    "Fleming-Harrington": FlemingHarrington,
    "Turnbull": Turnbull,
}


class _GroupBy:
    """Pure-NumPy grouped aggregation, replacing numpy_indexed.group_by.

    The multi-dimensional sum used to be ``np.add.at``, which is an
    unbuffered scatter with no fast path: at n=50 000 with ten covariates
    it was 6.3s of a 16.9s Efron fit, called about ten times per
    ``jac_hess`` on arrays of shape ``(n, p, p)`` (#329).

    Sorting once here turns each of those into ``np.add.reduceat``, a
    C-level segmented reduction. Every unique key has at least one member
    by construction, so the group starts strictly increase and
    ``reduceat`` is well defined.

    Two cases skip work entirely. When the keys already arrive grouped --
    the common case for start-stop (time-varying covariate) data -- there
    is nothing to permute. And when every key is distinct there is nothing
    to *add*: the sum is the permutation and no reduction is needed at
    all. That second case is continuous event times, where ``reduceat``
    would otherwise be asked for fifty thousand one-element segments and
    pay the per-segment overhead on every one of them.

    One consequence to know about: when both shortcuts apply at once --
    keys already grouped *and* all distinct -- ``sum`` returns the input
    array itself rather than a copy, because the sum of one-element groups
    in their existing order is the input. Treat the result as read-only.
    Every caller in this module builds its argument as a fresh temporary
    and only ever rebinds the result, never mutates it in place.
    """

    def __init__(self, keys: npt.NDArray) -> None:
        self.unique, self._inv = np.unique(keys, return_inverse=True)
        self._inv = np.asarray(self._inv).ravel()
        self._n = len(self.unique)

        counts = np.bincount(self._inv, minlength=self._n)
        self._starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        self._all_distinct = self._n == len(self._inv)

        already_grouped = bool(np.all(np.diff(self._inv) >= 0))
        self._order = (
            None if already_grouped else np.argsort(self._inv, kind="stable")
        )

    def sum(self, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        # ``asarray`` rather than ``astype``: no copy when the caller
        # already handed over float64, which it almost always does.
        values = np.asarray(values, dtype=float)
        if values.ndim == 1:
            result = np.bincount(self._inv, weights=values, minlength=self._n)
        else:
            ordered = values if self._order is None else values[self._order]
            result = (
                ordered
                if self._all_distinct
                else np.add.reduceat(ordered, self._starts, axis=0)
            )
        return self.unique, result

    def max(self, values: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        values = np.asarray(values)
        result = np.full(self._n, -np.inf)
        np.maximum.at(result, self._inv, values)
        return self.unique, result


def _efron_tie_weights(n_d: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """``c = j / d`` for every tied death, with a mask marking the entries
    that only exist to square off the ragged ``j < d`` ranges.

    The count ``d`` can be fractional -- ``n`` is a weight, not necessarily
    an integer -- and the loop this replaces ran ``range(int(d))`` while
    dividing by the unrounded ``d``. The mask therefore truncates and the
    weights do not; getting that backwards would silently change the
    Efron correction for weighted data.

    Shared by the log-likelihood denominator and the hessian so the two
    agree on the ragged-edge convention by construction.
    """
    counts = n_d.astype(int)
    j = np.arange(int(counts.max()) if counts.size else 0)
    valid = j[None, :] < counts[:, None]
    weights = np.where(valid, j[None, :] / n_d[:, None], 0.0)
    return weights, valid


def efron_log_denominator(
    n_d: npt.NDArray, Ri: npt.NDArray, Di: npt.NDArray
) -> npt.NDArray:
    """Per event time, ``sum_j log(R - (j/d) D)`` over the ``d`` tied deaths.

    Where at most one death occurs, ``j`` only ever takes the value 0, so
    ``c = 0`` and this collapses to ``log(R)`` — Breslow's denominator. On
    continuous data that is every event time, which is why the Efron and
    Breslow fits agree digit for digit there; splitting the two cases out
    means that agreement no longer costs a Python loop (#329).
    """
    out = np.zeros(len(n_d))
    R = np.asarray(Ri).reshape(len(n_d))
    D = np.asarray(Di).reshape(len(n_d))

    active = n_d >= 1
    if not active.any():
        return out

    weights, valid = _efron_tie_weights(n_d[active])
    v = R[active][:, None] - weights * D[active][:, None]
    out[active] = np.where(valid, np.log(v), 0.0).sum(axis=1)
    return out


# @njit
def efron_jac(
    n_d: npt.NDArray,
    Ri: npt.NDArray,
    ZRi: npt.NDArray,
    Di: npt.NDArray,
    ZDi: npt.NDArray,
    masked_array: npt.NDArray,
) -> npt.NDArray:
    # Vectorised implementation of term two of the efron ll
    # jacobian.

    # This implementation runs the risk of large memory usage
    # given the 3D array that is created.

    r = masked_array / n_d

    denom = Ri - Di * r
    denom = np.expand_dims(denom, axis=-1)

    r = np.expand_dims(r, axis=-1)
    numer = np.expand_dims(ZDi, axis=1) * r
    numer = np.expand_dims(ZRi, axis=1) - numer

    out = numer / denom
    out = out.sum(axis=1)
    return out


def efron_hess(
    n_d: npt.NDArray,
    Ri: npt.NDArray,
    ZRi: npt.NDArray,
    Z2Ri: npt.NDArray,
    Di: npt.NDArray,
    ZDi: npt.NDArray,
    Z2Di: npt.NDArray,
) -> npt.NDArray:
    # Per-event-time contribution to the observed information (the Hessian of
    # the negative Efron partial log-likelihood). For each of the ``n_d[i]``
    # tied deaths the Efron correction shrinks the risk set by ``c * D``:
    #
    #     sum_j (Z2R - c Z2D) / (R - c D) - a a' / (R - c D)^2,
    #
    # with ``a = ZR - c ZD``. The second term is the *outer* product ``a a'``
    # (a p x p matrix), which is where this previously went wrong -- an inner
    # product collapses it to a scalar and silently corrupts the off-diagonal
    # information for any model with more than one covariate.
    #
    # This used to be a Python double loop, 4.8s of a 16.9s fit at n=50 000
    # with ten covariates, plus 370 000 calls to ``np.outer`` (#329).
    #
    # The sum over ``j`` factors out of the p x p part entirely, which is
    # what makes the vectorised form cheap rather than merely loop-free.
    # Only ``c`` depends on ``j``, so with ``u = 1 / (R - c D)``:
    #
    #     sum_j (Z2R - c Z2D) u  =  (sum u) Z2R - (sum c u) Z2D
    #
    # and, expanding ``a a' = ZR ZR' - c (ZR ZD' + ZD ZR') + c^2 ZD ZD'``,
    #
    #     sum_j a a' u^2 = (sum u^2) ZR ZR'
    #                    - (sum c u^2) (ZR ZD' + ZD ZR')
    #                    + (sum c^2 u^2) ZD ZD'.
    #
    # The five sums are scalars per event time, so the ragged ``j`` axis
    # never has to carry a p x p payload: it costs O(times x ties) instead
    # of O(times x ties x p^2). Untied times fall out of the same formula
    # with a single j = 0 term and c = 0, so there is no separate branch --
    # on continuous data the ragged axis is one element wide.
    m = len(n_d)
    p = ZRi.shape[1]
    out = np.zeros((m, p, p))

    active = n_d >= 1
    if not active.any():
        return out

    weights, valid = _efron_tie_weights(n_d[active])
    R = np.asarray(Ri).reshape(m)[active][:, None]
    D = np.asarray(Di).reshape(m)[active][:, None]

    u = np.where(valid, 1.0 / (R - weights * D), 0.0)
    u2 = u**2

    s_u = u.sum(axis=1)[:, None, None]
    s_cu = (weights * u).sum(axis=1)[:, None, None]
    s_u2 = u2.sum(axis=1)[:, None, None]
    s_cu2 = (weights * u2).sum(axis=1)[:, None, None]
    s_c2u2 = (weights**2 * u2).sum(axis=1)[:, None, None]

    ZR = ZRi[active]
    ZD = ZDi[active]
    RR = ZR[:, :, None] * ZR[:, None, :]
    RD = ZR[:, :, None] * ZD[:, None, :]
    DD = ZD[:, :, None] * ZD[:, None, :]

    out[active] = (
        s_u * Z2Ri[active]
        - s_cu * Z2Di[active]
        - s_u2 * RR
        + s_cu2 * (RD + RD.transpose(0, 2, 1))
        - s_c2u2 * DD
    )
    return out


def _sort_by_event_time(
    x: npt.NDArray,
    Z: npt.NDArray,
    c: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Put the rows in event-time order before building the closures.

    Nothing in the partial likelihood depends on the order of the rows --
    every quantity is aggregated to unique event times first -- but
    ``_GroupBy`` gets to skip its permutation when the keys already arrive
    grouped. One reordering of ``Z`` here replaces a gather of an
    ``(n, p, p)`` array on every ``jac_hess`` call, roughly ten of them per
    root-finding iteration (#329).

    The caller keeps the unsorted arrays: ``fit`` stores those on the model
    for the residual and diagnostic code, and the closures only ever hand
    back beta-shaped or unique-time-shaped results.
    """
    order = np.argsort(x, kind="stable")
    return x[order], Z[order], c[order], n[order], tl[order]


def at_risk_beta_Z(
    arr: npt.NDArray, n: npt.NDArray, gb_x: "_GroupBy"
) -> npt.NDArray:
    R = gb_x.sum(n * arr)[1]
    # Get the reverse cumulative sum
    return R[::-1].cumsum(axis=0)[::-1]


def not_yet_entered(pos: npt.NDArray, mass_by_tl: npt.NDArray) -> npt.NDArray:
    """Per unique event time, the total ``mass_by_tl`` of observations whose
    entry (left-truncation) time is at or after that event time — the amount
    to subtract from the reverse-cumulative at-risk sums so that a subject
    only enters the risk set strictly after its ``tl``.

    ``pos`` is ``searchsorted(unique_tl, unique_x, side="left")``. This is an
    exact suffix-sum gather and is valid for *signed* quantities (the
    Z-weighted score and information sums), unlike the previous scatter +
    ``minimum.accumulate`` forward fill, which is only a forward fill for
    positive non-increasing sequences and silently corrupted the gradient and
    Hessian of every delayed-entry / start-stop fit containing a negative
    covariate value (#250).
    """
    suffix = mass_by_tl[::-1].cumsum(axis=0)[::-1]
    pad = np.zeros((1,) + suffix.shape[1:])
    return np.concatenate([suffix, pad], axis=0)[pos]


def _sub(a: "npt.ArrayLike | None", mask: npt.NDArray) -> "npt.NDArray | None":
    """Index ``a`` by ``mask``, passing ``None`` through unchanged."""
    if a is None:
        return None
    return np.asarray(a)[mask]


def _kp_tie_term(
    eta: npt.NDArray, Z: npt.NDArray, d: int, derivs: bool = True
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """``log e_d`` of the risk-set scores ``exp(eta)``, with its gradient and
    Hessian in ``beta`` (``eta = Z @ beta`` over the risk set).

    ``e_d`` is the ``d``-th elementary symmetric polynomial -- the sum, over
    every ``d``-subset of the risk set, of the product of its scores -- which
    is the denominator of the Kalbfleisch-Prentice (discrete) tie term. It is
    evaluated by the Gail, Lubin & Rubinstein (1981) recursion over the risk
    set that R's ``coxph(ties="exact")`` also uses: with ``B_k(j)`` the value
    of ``e_k`` over the first ``j`` members,

        B_k(j) = B_k(j - 1) + r_j B_{k-1}(j - 1),

    and the same recursion differentiated once and twice gives the score and
    information. For fixed ``k`` that is a cumulative sum over ``j``, so the
    Python loop runs ``d`` times over vectorised risk-set arrays, O(d m p^2)
    in all. The same recursion used to run as a scalar autograd trace of
    ``m * d`` Python-level operations, re-traced for every gradient, which
    took minutes on a tie set of a hundred.

    Each row ``k`` is rescaled by its largest entry (the running ``log_scale``
    keeps the value) so ``e_d`` cannot overflow; derivatives are carried in
    the same scale, and only their ratios to ``e_d`` are used.
    """
    m, p = Z.shape
    if d > m - d:
        # e_d(v) = prod(v) * e_{m-d}(1/v): the complement runs fewer
        # iterations, and when every member of the risk set dies (d = m) it
        # runs none at all.
        log_e, g, h = _kp_tie_term(-eta, -Z, m - d, derivs)
        return float(eta.sum()) + log_e, Z.sum(axis=0) + g, h

    shift = float(eta.max()) if m else 0.0
    r = np.exp(eta - shift)
    B = np.ones(m + 1)
    dB = np.zeros((m + 1, p))
    d2B = np.zeros((m + 1, p, p))
    ZZ = Z[:, :, None] * Z[:, None, :] if derivs else None
    log_scale = 0.0
    for _ in range(d):
        Bp = B[:-1]
        B = np.concatenate([[0.0], np.cumsum(r * Bp)])
        if derivs:
            dBp, d2Bp = dB[:-1], d2B[:-1]
            t1 = dBp + Z * Bp[:, None]
            t2 = (
                d2Bp
                + Z[:, :, None] * dBp[:, None, :]
                + dBp[:, :, None] * Z[:, None, :]
                + ZZ * Bp[:, None, None]
            )
            dB = np.concatenate(
                [np.zeros((1, p)), np.cumsum(r[:, None] * t1, axis=0)]
            )
            d2B = np.concatenate(
                [
                    np.zeros((1, p, p)),
                    np.cumsum(r[:, None, None] * t2, axis=0),
                ]
            )
        # The cumulative sums only add non-negative terms, so the last entry
        # is the largest.
        scale = B[-1]
        log_scale += np.log(scale)
        B = B / scale
        if derivs:
            dB = dB / scale
            d2B = d2B / scale

    log_e = log_scale + d * shift
    if not derivs:
        return log_e, np.zeros(p), np.zeros((p, p))
    g = dB[-1] / B[-1]
    return log_e, g, d2B[-1] / B[-1] - np.outer(g, g)


def _weighted_moments(
    eta: npt.NDArray, Z: npt.NDArray
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """``log sum exp(eta)`` with the ``exp(eta)``-weighted mean and
    covariance of the rows of ``Z``."""
    shift = eta.max()
    w = np.exp(eta - shift)
    total = w.sum()
    w = w / total
    mean = w @ Z
    Zc = Z - mean
    cov = (w[:, None] * Zc).T @ Zc
    return float(shift + np.log(total)), mean, cov


# DeLong et al. integrand, integrated over ``w = log t``: points further than
# this many log-units below the mode contribute below 1e-26 relative, and the
# coarse grid used to bracket the mode steps by ``_EXACT_COARSE_STEP``.
_EXACT_DROP = 60.0
_EXACT_COARSE_STEP = 0.1
_EXACT_NODES = 401


def _exact_tie_term(
    eta_d: npt.NDArray,
    Z_d: npt.NDArray,
    eta_w: npt.NDArray,
    Z_w: npt.NDArray,
    derivs: bool = True,
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """``log`` of the exact (average-over-orderings) tie contribution, with
    its gradient and Hessian in ``beta``.

    For ``d`` tied deaths with scores ``a_j = exp(eta_j)`` and the rest of the
    risk set scoring ``W = sum exp(eta_w)``, the sum over the ``d!`` orderings
    of the sequential Cox terms equals (DeLong, Guirguis & So 1994)

        L = int_0^inf prod_j (1 - exp(-a_j t / W)) exp(-t) dt,

    the formula SAS uses for ``TIES=EXACT``. This replaces an O(2^d) subset
    recursion that was capped at twelve ties and still took tens of seconds
    to fit. In ``w = log t`` the log-integrand

        g(w) = sum_j log(1 - exp(-c_j e^w)) - e^w + w,   c_j = a_j / W,

    is concave, so the integrand is a single smooth bump. A coarse grid
    brackets the region within ``_EXACT_DROP`` log-units of its peak and the
    trapezoid rule on a fine grid there -- spectrally accurate for a smooth,
    negligible-at-the-ends integrand -- gives ``L`` to machine precision. The
    score and information follow by differentiating under the integral:
    with ``E`` the expectation over the normalised integrand,

        d log L = E[dg],   d2 log L = E[d2g] + Var[dg].
    """
    d, p = Z_d.shape
    if eta_w.size == 0:
        # Everyone left at risk dies: every ordering's product telescopes
        # and the orderings sum to exactly one.
        return 0.0, np.zeros(p), np.zeros((p, p))

    lse_w, mean_w, cov_w = _weighted_moments(eta_w, Z_w)
    log_c = eta_d - lse_w

    def log_integrand(w: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        x = np.exp(log_c[None, :] + w[:, None])
        with np.errstate(divide="ignore"):
            # log(1 - e^-x); x can underflow to 0 far left of the mode,
            # where the log is -inf and the node simply carries no weight.
            g = np.log(-np.expm1(-x)).sum(axis=1) - np.exp(w) + w
        return g, x

    # The mode lies in (0, log(d + 1)): g' = sum x/(e^x - 1) - e^w + 1 with
    # every summand in (0, 1). g falls at least as fast as w - e^w to the left
    # and as (d + 1)(w - e^w) to the right, so this range contains every point
    # within _EXACT_DROP of the peak.
    coarse = np.arange(
        -(_EXACT_DROP + 2.0), np.log(d + 1.0) + 4.0, _EXACT_COARSE_STEP
    )
    g_coarse = log_integrand(coarse)[0]
    # g is concave, so the points above the threshold form one run; a coarse
    # point below it bounds the region from outside on each side.
    inside = np.flatnonzero(g_coarse >= g_coarse.max() - _EXACT_DROP)
    lo = coarse[max(inside[0] - 1, 0)]
    hi = coarse[min(inside[-1] + 1, coarse.size - 1)]

    nodes = np.linspace(lo, hi, _EXACT_NODES)
    step = nodes[1] - nodes[0]
    g, x = log_integrand(nodes)
    g_max = g.max()
    weight = np.exp(g - g_max)
    total = weight.sum()
    log_L = float(g_max + np.log(step * total))
    if not derivs:
        return log_L, np.zeros(p), np.zeros((p, p))

    # q = x / (e^x - 1) = d/dlog(x) of log(1 - e^-x), written to stay finite
    # for large x; q -> 1 as x -> 0 (a node that underflowed to x = 0).
    one_minus = -np.expm1(-x)
    positive = x > 0
    safe = np.where(positive, one_minus, 1.0)
    q = np.where(positive, x * np.exp(-x) / safe, 1.0)
    # x q'(x), the second log-derivative, is q (1 - x / (1 - e^-x)).
    xq = np.where(positive, q * (1.0 - x / safe), 0.0)

    Zc = Z_d - mean_w
    dg = q @ Zc
    d2g = (
        np.einsum("nd,dp,dq->npq", xq, Zc, Zc)
        - q.sum(axis=1)[:, None, None] * cov_w
    )
    prob = weight / total
    mean_dg = prob @ dg
    hess = (
        np.einsum("n,npq->pq", prob, d2g)
        + np.einsum("n,np,nq->pq", prob, dg, dg)
        - np.outer(mean_dg, mean_dg)
    )
    return log_L, mean_dg, hess


def _solve_beta_and_p_values(
    neg_ll: Callable,
    jac: Callable,
    beta_init: npt.NDArray,
    tol: float,
) -> tuple[Any, npt.NDArray]:
    """Root-find the score (with BFGS fallback) and compute Wald p-values
    from the observed information; shared by ``fit`` and
    ``_fit_stratified`` so the most-patched block in this file exists
    exactly once."""
    # Have found that root finding is faster than minimization. ``jac``
    # returns (score, hessian), hence ``jac=True``.
    res = root(jac, beta_init, jac=True, tol=tol)

    # MINPACK's hybr root-finder can stall on delayed-entry data with
    # staggered risk sets (e.g. the start-stop representation used for
    # time-varying covariates) even though the partial log-likelihood is
    # well behaved there. Fall back to a direct minimisation of the
    # negative partial log-likelihood whenever root-finding fails to
    # converge or lands at a worse point, so such fits still succeed.
    if not res.success:
        fallback = minimize(
            lambda b: float(neg_ll(b)), beta_init, method="BFGS"
        )
        if float(neg_ll(fallback.x)) < float(neg_ll(res.x)):
            res = fallback

    hessian_matrix = jac(res.x)[1]
    # An exactly singular information matrix raises before the
    # pseudo-inverse fallback can run (#259); route it there.
    try:
        var = np.diag(inv(hessian_matrix))
    except np.linalg.LinAlgError:
        var = np.full(len(np.atleast_1d(res.x)), -1.0)
    # Use the pseudo-inverse if the hessian does not have a diagonal that
    # is all positive.
    if np.any(var <= 0):
        var = np.diag(pinv(hessian_matrix))
    # A near-singular information matrix (e.g. a degenerate start-stop
    # design with duplicated rows) can still leave a non-positive
    # variance; the resulting standard error is simply unavailable (nan),
    # which is the correct signal, so suppress the sqrt-of-negative
    # warning rather than emit it.
    with np.errstate(invalid="ignore"):
        z_score = res.x / np.sqrt(var)
    p_values = 2 * (1 - norm.cdf(np.abs(z_score)))
    return res, p_values


def _combine_generators(gens: list) -> tuple[Callable, Callable]:
    """Sum per-stratum ``(log_like, jac_hess)`` generators into one.

    The Cox partial likelihood factorises across strata: with a separate
    baseline hazard per stratum and a *shared* coefficient vector, the total
    log-likelihood (and hence its score and observed information) is the sum
    of the per-stratum contributions. Risk sets never cross a stratum
    boundary because each stratum's score/information is built only from its
    own observations.
    """

    def neg_ll(beta: npt.NDArray) -> float:
        return sum(g[0](beta) for g in gens)

    def jac_hess(beta: npt.NDArray) -> tuple:
        jac_total = None
        hess_total = None
        for g in gens:
            j, h = g[1](beta)
            jac_total = j if jac_total is None else jac_total + j
            hess_total = h if hess_total is None else hess_total + h
        return jac_total, hess_total

    return neg_ll, jac_hess


def cox_at_risk_mask(
    x: npt.NDArray, tl: npt.NDArray, tau: float
) -> npt.NDArray:
    """The Cox risk-set convention, in one place (#299): a row is at risk
    at event time ``tau`` once it has entered (``tl < tau`` — strict, so a
    start-stop row is not at risk at its own entry time) and until it
    exits (``x >= tau`` — inclusive, so a row is at risk at its own event
    or censoring time)."""
    return (tl < tau) & (x >= tau)


class CoxPH_:
    """
    The Cox proportional hazards model: a baseline hazard left entirely
    to the data, multiplied by :math:`e^{\\beta' Z}`,

    .. math::
        h(x \\mid Z) = h_0(x)\\, e^{\\beta' Z}.

    The coefficients are estimated from the partial likelihood (with a
    choice of tie handling) and the baseline by the Breslow estimator.
    Supports right censoring, left truncation (delayed entry),
    stratification and time-varying covariates in start-stop form; left-
    and interval-censored data are refused, as the partial likelihood has
    no term for them (use a parametric regression model).
    ``CoxPH`` is an instance of this class; its fit methods return a
    :class:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel`.
    """

    # Best reference I can find that covers all the
    # possibilities for estimating betas
    # http://www-personal.umich.edu/~yili/lect4notes.pdf

    def baseline(
        self,
        beta: npt.NDArray,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        Z: npt.NDArray,
        tl: "npt.NDArray | None" = None,
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        # Breslow baseline hazard. The risk set at each event time ``tau_i``
        # follows ``cox_at_risk_mask`` (entered ``tl < tau_i``, not yet
        # exited ``x >= tau_i``), each row weighted by its count ``n`` and
        # hazard multiplier ``exp(Z'beta)``. Respecting ``tl`` is what makes
        # the baseline correct for left-truncated and time-varying-covariate
        # (start-stop) data. Computed by suffix sums — the risk set is
        # everyone with ``x >= tau_i`` minus the not-yet-entered
        # ``tl >= tau_i`` (valid because ``tl < x`` on every row) — the same
        # subtraction the Efron generator uses, replacing the previous
        # O(K·N) Python loop (#299).

        unique_x = np.unique(x)
        if tl is None:
            tl = np.full(x.shape[0], -np.inf)

        w = n * np.exp(Z @ beta)

        event = c == 0
        d = np.zeros_like(unique_x)
        np.add.at(d, np.searchsorted(unique_x, x[event]), n[event])

        r_exit = np.zeros_like(unique_x)
        np.add.at(r_exit, np.searchsorted(unique_x, x), w)
        r_exit = r_exit[::-1].cumsum()[::-1]

        # Bucket each row at the largest event time <= its entry time; the
        # suffix sum then gives, at each tau_i, the weight not yet entered.
        k = np.searchsorted(unique_x, tl, side="right") - 1
        entered_late = k >= 0
        r_pre = np.zeros_like(unique_x)
        np.add.at(r_pre, k[entered_late], w[entered_late])
        r_pre = r_pre[::-1].cumsum()[::-1]

        return unique_x, r_exit - r_pre, d

    def create_efron_ll_jac_hess(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple[Callable, Callable]:
        # The reference used to compute the jacobian and hessian
        # was https://mathweb.ucsd.edu/~rxu/math284/slect5.pdf
        # Left-truncation is handled by subtracting the pre-entry risk set
        # (``Ri - TRi``) below, so delayed-entry data is fitted correctly.

        x, Z, c, n, tl = _sort_by_event_time(x, Z, c, n, tl)

        # Groupby object for repeated use
        gb_x = _GroupBy(x)
        gb_tl = _GroupBy(tl)
        n_d_x = np.where(c == 0, n, 0)
        n_d = gb_x.sum(n_d_x)[1]
        n_d_x = n_d_x.reshape(-1, 1)
        n = n.reshape(-1, 1)

        max_n = n_d.max()

        x_ = gb_x.unique
        x_tl = gb_tl.unique
        # For each unique event time, how many unique entry times precede it:
        # feeds the not-yet-entered suffix-sum gather below.
        pos = np.searchsorted(x_tl, x_, side="left")

        def log_like(beta: npt.NDArray) -> float:
            beta_z = Z @ beta

            S_d = gb_x.sum(n_d_x * beta_z.reshape(-1, 1))[1].reshape(-1, 1)
            e_beta_z = np.exp(beta_z).reshape(-1, 1)

            x_, Ri = gb_x.sum(n * e_beta_z)

            Ri = Ri[::-1].cumsum(axis=0)[::-1]

            # Subtract the not-yet-entered mass from the risk sums.
            Ri = Ri - not_yet_entered(pos, gb_tl.sum(n * e_beta_z)[1])

            Di = gb_x.sum(n_d_x * e_beta_z)[1]

            efron_denom = efron_log_denominator(n_d, Ri, Di)

            like = S_d.sum() - efron_denom.sum()
            return -like

        S_d = gb_x.sum(n_d_x * Z)[1]

        arr = np.repeat([np.arange(max_n)], len(n_d), axis=0)
        mask = 1 - (arr < n_d.reshape(-1, 1)).astype(int)

        masked_array = ma.array(arr, mask=mask)

        # Z is fixed for the life of the fit, so its outer product is too.
        # It used to be rebuilt inside ``jac_hess`` -- an (n, p, p) einsum
        # on every root-finding iteration, 1.1s of a 16.9s fit (#329).
        Z2 = np.einsum("ij, ik -> ijk", Z, Z)

        def jac_hess(beta: npt.NDArray) -> tuple:
            # This line troubled me for longer than I care
            # to admit. I was using n, but it is only the
            # number of deaths at each point, n_d_x

            # Only call this once.. Yay.
            beta_z = Z @ beta

            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            z_e_beta_z = Z * e_beta_z
            z2_e_beta_z = Z2 * (n * e_beta_z)[:, :, None]

            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)
            ZRi = at_risk_beta_Z(z_e_beta_z, n, gb_x)
            Z2Ri = gb_x.sum(z2_e_beta_z)[1]
            Z2Ri = Z2Ri[::-1].cumsum(axis=0)[::-1]

            # Subtract the not-yet-entered mass from the risk sums. The
            # Z-weighted sums are signed, so this must be the exact gather —
            # see ``not_yet_entered`` (#250).
            Ri = Ri - not_yet_entered(pos, gb_tl.sum(n * e_beta_z)[1])
            ZRi = ZRi - not_yet_entered(pos, gb_tl.sum(n * z_e_beta_z)[1])
            Z2Ri = Z2Ri - not_yet_entered(pos, gb_tl.sum(z2_e_beta_z)[1])

            Di = gb_x.sum(n_d_x * e_beta_z)[1]
            ZDi = gb_x.sum(n_d_x * z_e_beta_z)[1]

            expected_S_d = np.zeros_like(S_d)
            expected_S_d = efron_jac(
                n_d.reshape(-1, 1), Ri, ZRi, Di, ZDi, masked_array
            )

            diff = S_d - expected_S_d
            jacobian = -diff.sum(axis=0)

            # Observed information (Hessian of the negative log-likelihood),
            # accumulated per tied time then summed. Same positive-definite
            # convention as the Breslow branch, so ``inv(hess)`` gives the
            # parameter covariance directly.
            Z2Di = Z2 * (n_d_x * e_beta_z)[:, :, None]
            Z2Di = gb_x.sum(Z2Di)[1]

            hess_matrix = efron_hess(n_d, Ri, ZRi, Z2Ri, Di, ZDi, Z2Di).sum(
                axis=0
            )

            return jacobian, hess_matrix

        return log_like, jac_hess

    def create_breslow_ll_jac_hess(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple[Callable, Callable]:
        # The reference used to compute the jacobian and hessian
        # was https://mathweb.ucsd.edu/~rxu/math284/slect5.pdf
        # Left-truncation is handled by subtracting the pre-entry risk set
        # (``Ri - TRi``) below, so delayed-entry data is fitted correctly.

        x, Z, c, n, tl = _sort_by_event_time(x, Z, c, n, tl)

        gb_x = _GroupBy(x)
        gb_tl = _GroupBy(tl)
        n_d_x = np.where(c == 0, n, 0)
        n_d = gb_x.sum(n_d_x)[1]
        n_d_x = n_d_x.reshape(-1, 1)
        n = n.reshape(-1, 1)

        x_ = gb_x.unique
        x_tl = gb_tl.unique
        # For each unique event time, how many unique entry times precede it:
        # feeds the not-yet-entered suffix-sum gather below.
        pos = np.searchsorted(x_tl, x_, side="left")

        # Create the log_like function for the data
        def log_like(beta: npt.NDArray) -> float:
            beta_z = Z @ beta
            di_beta_z = gb_x.sum(n_d_x * beta_z.reshape(-1, 1))[1].reshape(
                -1, 1
            )
            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)

            # Subtract the not-yet-entered mass from the risk sums.
            Ri = Ri - not_yet_entered(pos, gb_tl.sum(n * e_beta_z)[1])

            Ri = np.log(Ri)
            Ri = n_d.reshape(-1, 1) * Ri

            like = di_beta_z - Ri

            return -like.sum()

        S_d = gb_x.sum(n_d_x.reshape(-1, 1) * Z)[1]

        # Constant for the life of the fit; see the Efron branch (#329).
        Z2 = np.einsum("ij, ik -> ijk", Z, Z)

        def jac_hess(beta: npt.NDArray) -> tuple:
            # Only call this once.. Yay.
            beta_z = Z @ beta

            e_beta_z = np.exp(beta_z).reshape(-1, 1)
            z_e_beta_z = Z * e_beta_z
            z2_e_beta_z = Z2 * (n * e_beta_z)[:, :, None]

            Ri = at_risk_beta_Z(e_beta_z, n, gb_x)
            ZRi = at_risk_beta_Z(z_e_beta_z, n, gb_x)
            Z2Ri = gb_x.sum(z2_e_beta_z)[1]
            Z2Ri = Z2Ri[::-1].cumsum(axis=0)[::-1]

            # Subtract the not-yet-entered mass from the risk sums. The
            # Z-weighted sums are signed, so this must be the exact gather —
            # see ``not_yet_entered`` (#250).
            Ri = Ri - not_yet_entered(pos, gb_tl.sum(n * e_beta_z)[1])
            ZRi = ZRi - not_yet_entered(pos, gb_tl.sum(n * z_e_beta_z)[1])
            Z2Ri = Z2Ri - not_yet_entered(pos, gb_tl.sum(z2_e_beta_z)[1])

            EZ = ZRi / Ri
            EZ = n_d.reshape(-1, 1) * EZ

            jacobian = -(S_d - EZ).sum(axis=0)

            # calc term 1
            term_1 = Z2Ri / Ri[:, :, None]

            # calc term 2
            term_2 = ZRi / Ri
            term_2 = np.einsum("ij, ik-> ijk", term_2, term_2)

            # Compute the Hessian matrix
            hess_matrix = term_1 - term_2
            hess_matrix = np.einsum("ijk,i->ijk", hess_matrix, n_d.flatten())
            hess_matrix = hess_matrix.sum(axis=0)

            return jacobian, hess_matrix

        return log_like, jac_hess

    def _prepare_exact_tie_data(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple:
        """Expand count-weighted rows and pre-compute, per event time, the
        death rows, the (delayed-entry-aware) risk-set rows, and the death
        covariate sum. Shared by the ``exact`` and ``kalbfleisch-prentice``
        generators.

        Counts ``n`` are expanded into individual rows -- a row with a count of
        two events is genuinely two tied deaths -- so the exact and discrete
        tie formulae see the true multiplicity. Non-integer counts have no such
        interpretation and are rejected.
        """
        n = np.asarray(n, dtype=float)
        n_int = np.round(n).astype(int)
        if np.any(n_int < 1) or np.any(np.abs(n - n_int) > 1e-9):
            raise ValueError(
                "The 'exact' and 'kalbfleisch-prentice' tie methods require "
                "integer counts n (each row is expanded to n identical "
                "observations); use 'efron' or 'breslow' for fractional "
                "weights."
            )
        rep = np.repeat(np.arange(len(x)), n_int)
        xe = np.asarray(x, dtype=float)[rep]
        ce = np.asarray(c, dtype=int)[rep]
        tle = np.asarray(tl, dtype=float)[rep]
        Ze = np.asarray(Z, dtype=float)[rep]

        event_times = np.unique(xe[ce == 0])
        death_idx = []
        risk_idx = []
        death_Z_sum = []
        for tau in event_times:
            d_mask = (xe == tau) & (ce == 0)
            r_mask = cox_at_risk_mask(xe, tle, tau)
            death_idx.append(np.where(d_mask)[0])
            risk_idx.append(np.where(r_mask)[0])
            death_Z_sum.append(Ze[d_mask].sum(axis=0))
        return Ze, event_times, death_idx, risk_idx, np.array(death_Z_sum)

    @staticmethod
    def _tie_term_ll_jac_hess(
        n_events: int,
        term: Callable[[npt.NDArray, int, bool], tuple],
    ) -> tuple[Callable, Callable]:
        """Sum per-event-time ``term(eta, i, derivs) -> (log L_i, grad,
        hess)`` contributions into the ``(neg_ll, jac_hess)`` contract used
        by :meth:`fit` (the gradient and Hessian of the *negative*
        log-likelihood, so the Hessian is the observed information)."""

        def total(beta: npt.NDArray, derivs: bool) -> tuple:
            beta = np.asarray(beta, dtype=float)
            p = beta.shape[0]
            ll, score, hess = 0.0, np.zeros(p), np.zeros((p, p))
            for i in range(n_events):
                ll_i, g_i, h_i = term(beta, i, derivs)
                ll += ll_i
                score = score + g_i
                hess = hess + h_i
            return -ll, -score, -hess

        def neg_ll(beta: npt.NDArray) -> float:
            return float(total(beta, False)[0])

        def jac_hess(beta: npt.NDArray) -> tuple:
            _, score, hess = total(beta, True)
            return score, hess

        return neg_ll, jac_hess

    def create_kalbfleisch_prentice_ll_jac_hess(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple[Callable, Callable]:
        """Kalbfleisch-Prentice discrete (conditional-logistic) tie handling.

        Treats tied event times as genuinely discrete: the contribution of a
        tie set ``D`` (``d`` deaths) with risk set ``R`` is

            exp(b' * sum_{j in D} Z_j) / e_d({exp(Z_k'b) : k in R}),

        where ``e_d`` is the ``d``-th elementary symmetric polynomial of the
        risk-set scores -- i.e. the sum over all ``d``-subsets of ``R`` of the
        product of their scores. This is the exact discrete
        proportional-hazards (Cox 1972 discrete model / Kalbfleisch-Prentice)
        likelihood, R's ``ties="exact"``. ``e_d`` and its derivatives come
        from the polynomial recursion in :func:`_kp_tie_term`.
        """
        Ze, event_times, death_idx, risk_idx, S = self._prepare_exact_tie_data(
            x, Z, c, n, tl
        )
        Z_risk = [Ze[r] for r in risk_idx]
        ds = [len(d) for d in death_idx]

        def term(beta: npt.NDArray, i: int, derivs: bool) -> tuple:
            log_e, g, h = _kp_tie_term(
                Z_risk[i] @ beta, Z_risk[i], ds[i], derivs
            )
            return float(S[i] @ beta) - log_e, S[i] - g, -h

        return self._tie_term_ll_jac_hess(len(event_times), term)

    def create_exact_ll_jac_hess(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
    ) -> tuple[Callable, Callable]:
        """Exact (average-over-orderings) partial-likelihood tie handling.

        Appropriate when ties arise from coarse rounding of an underlying
        continuous time. Each tie set is treated as having occurred in an
        unknown order and its contribution is the sequential Cox partial
        likelihood summed over all orderings of the tied deaths, evaluated
        as the DeLong et al. integral (SAS's ``TIES=EXACT``; see
        :func:`_exact_tie_term`). Reduces to Breslow/Efron when there are no
        ties.
        """
        Ze, event_times, death_idx, risk_idx, S = self._prepare_exact_tie_data(
            x, Z, c, n, tl
        )
        Z_death = [Ze[d] for d in death_idx]
        # The rest of the risk set: at risk at the time but not dying there.
        survivors = [np.setdiff1d(r, d) for r, d in zip(risk_idx, death_idx)]
        Z_surv = [Ze[s] for s in survivors]
        Z_risk = [Ze[r] for r in risk_idx]

        def term(beta: npt.NDArray, i: int, derivs: bool) -> tuple:
            if len(death_idx[i]) == 1:
                # A single death needs no ordering: a_j / sum over the risk
                # set, the Breslow term, in closed form.
                lse, mean, cov = _weighted_moments(Z_risk[i] @ beta, Z_risk[i])
                return float(S[i] @ beta) - lse, S[i] - mean, -cov
            return _exact_tie_term(
                Z_death[i] @ beta,
                Z_death[i],
                Z_surv[i] @ beta,
                Z_surv[i],
                derivs,
            )

        return self._tie_term_ll_jac_hess(len(event_times), term)

    def _resolve_func_generator(self, method: str) -> Callable[..., Any]:
        """Map a tie-handling ``method`` name to its likelihood generator."""
        generators: dict[str, Callable[..., Any]] = {
            "efron": self.create_efron_ll_jac_hess,
            "breslow": self.create_breslow_ll_jac_hess,
            "exact": self.create_exact_ll_jac_hess,
            "kalbfleisch-prentice": (
                self.create_kalbfleisch_prentice_ll_jac_hess
            ),
            "kp": self.create_kalbfleisch_prentice_ll_jac_hess,
        }
        if method not in generators:
            raise ValueError(
                "method must be one of {}".format(sorted(generators))
            )
        return generators[method]

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        method: str = "breslow",
        tol: float = 1e-10,
        strata: npt.ArrayLike | None = None,
    ) -> SemiParametricRegressionModel:
        """
        Fits Cox Proportional Hazards model to the provided data.

        Parameters
        ----------

        x: array-like
            The observed times of the events.
        Z: array-like
            The covariates of the model, one row per observation.
        c: array-like, optional
            The censoring indicator. 0 if observed (event),
            1 if right-censored. Defaults to all observed. Left-censored
            (-1) and interval-censored (2) rows raise a ``ValueError``: the
            partial likelihood has no term for them, so fit such data with
            a parametric regression model (e.g. ``WeibullPH``) instead.
        n: array-like, optional
            The number of observations at each time point.
        tl: array-like, optional
            The left-truncation times of the observations.
        method: str, optional
            The method to use for tie handling. One of ``'breslow'``
            (default), ``'efron'``, ``'exact'`` (the average-over-orderings
            exact partial likelihood, for ties from coarse rounding of
            continuous time) or ``'kalbfleisch-prentice'`` (alias ``'kp'`` --
            the exact discrete/conditional-logistic likelihood, for genuinely
            discrete time). Breslow and Efron match what R's ``survival`` and
            lifelines use by default; the two exact methods are only
            meaningfully different under heavy ties and are correspondingly
            more expensive.
        tol: float, optional
            The tolerance for the root finding algorithm.
        strata: array-like, optional
            Stratum label for each observation. When supplied the model is
            *stratified*: a separate baseline hazard is estimated per stratum
            while the coefficients ``beta`` are shared. The partial likelihood
            is summed within strata (risk sets never cross a stratum boundary),
            which is the standard remedy when proportional hazards fails for a
            nuisance covariate that you would rather not model. Prediction
            (``hf``/``Hf``/``sf``/``ff``/``df``) then takes a ``stratum``
            argument to select that stratum's baseline.

        Returns
        -------

        model: SemiParametricRegressionModel
            The fitted model: ``params`` (also ``beta``) are the
            coefficients and ``p_values`` their Wald p-values.

        Examples
        --------
        In the bundled copy of the Rossi recidivism data ``arrest`` is 1
        for a subject still free at week 52, so it is already the
        censoring flag:

        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, df["arrest"].values
        >>> Z = df[["fin", "age", "prio"]].values
        >>> model = CoxPH.fit(x, Z, c=c)
        >>> model.params.round(4)
        array([-0.3464, -0.0669,  0.0965])
        >>> model.p_values.round(4)
        array([0.0686, 0.0013, 0.0004])
        >>> model.sf([20, 52], [1, 25, 3]).round(4)
        array([0.9327, 0.7968])
        """
        func_generator = self._resolve_func_generator(method)

        if strata is not None:
            return self._fit_stratified(
                x, Z, c, n, tl, method, tol, strata, func_generator
            )

        x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, method)

        # Good initial guess assumes no impact
        beta_init = np.zeros(Z.shape[1])

        neg_ll, jac = func_generator(x, Z, c, n, tl)

        res, p_values = _solve_beta_and_p_values(neg_ll, jac, beta_init, tol)

        model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
        model._neg_log_like = neg_ll(res.x)
        model.p_values = p_values
        model.neg_ll = neg_ll
        model.jac = jac
        model.tie_method = method
        model.baseline_method = "breslow"
        model.res = res
        model.beta = copy(res.x)
        model.phi = lambda Z: np.exp(Z @ model.beta)
        model.params = res.x

        # Retain the per-observation training data (before ``baseline``
        # reassigns ``x`` to the unique event times) so the model can compute
        # residuals (Schoenfeld, martingale, ...) and the proportional-
        # hazards test.
        model._fit_data = {
            "x": np.asarray(x, dtype=float),
            "c": np.asarray(c, dtype=int),
            "n": np.asarray(n, dtype=float),
            "Z": np.asarray(Z, dtype=float),
            "tl": np.asarray(tl, dtype=float),
        }

        x, r, d = self.baseline(model.beta, x, c, n, Z, tl)
        model.x = x
        model.r = r
        model.d = d
        model.tl = tl
        model.h0 = d / r
        model.H0 = model.h0.cumsum()

        return model

    def _fit_stratified(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: "npt.ArrayLike | None",
        n: "npt.ArrayLike | None",
        tl: "npt.ArrayLike | None",
        method: str,
        tol: float,
        strata: npt.ArrayLike,
        func_generator: Callable,
    ) -> SemiParametricRegressionModel:
        """Fit a stratified Cox model (shared ``beta``, per-stratum baseline).

        Each stratum is validated and turned into its own partial-likelihood
        generator; the generators are summed (see :func:`_combine_generators`)
        so the score equations are solved once for the shared coefficients.
        A separate Breslow baseline hazard is then estimated within each
        stratum.
        """
        strata = np.asarray(strata)
        if len(strata) != len(np.atleast_1d(x)):
            raise ValueError("'strata' must have a label for each observation")

        labels = np.unique(strata)
        per_stratum = []
        n_params = None
        for s in labels:
            mask = strata == s
            xs, cs, ns_, tls, Zs = validate_coxph(
                _sub(x, mask),
                _sub(c, mask),
                _sub(n, mask),
                _sub(Z, mask),
                _sub(tl, mask),
                method,
            )
            if n_params is None:
                n_params = Zs.shape[1]
            gen = func_generator(xs, Zs, cs, ns_, tls)
            per_stratum.append((s, gen, (xs, cs, ns_, Zs, tls)))

        if n_params is None:
            raise ValueError("no observations to fit")
        gens = [g for _, g, _ in per_stratum]
        neg_ll, jac = _combine_generators(gens)

        beta_init = np.zeros(n_params)
        res, p_values = _solve_beta_and_p_values(neg_ll, jac, beta_init, tol)

        model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
        model._neg_log_like = neg_ll(res.x)
        model.p_values = p_values
        model.neg_ll = neg_ll
        model.jac = jac
        model.tie_method = method
        model.baseline_method = "breslow"
        model.res = res
        model.beta = copy(res.x)
        model.phi = lambda Z: np.exp(Z @ model.beta)
        model.params = res.x
        model.is_stratified = True
        model.strata_labels = list(labels)

        # A separate Breslow baseline per stratum. Prediction selects the
        # stratum's baseline via the ``stratum`` argument to ``hf``/``Hf``/...
        baselines: dict[Any, dict[str, npt.NDArray]] = {}
        for s, _, (xs, cs, ns_, Zs, tls) in per_stratum:
            bx, br, bd = self.baseline(model.beta, xs, cs, ns_, Zs, tls)
            bh0 = bd / br
            baselines[s] = {
                "x": bx,
                "r": br,
                "d": bd,
                "h0": bh0,
                "H0": bh0.cumsum(),
            }
        model.strata_baselines = baselines

        # Expose the first stratum's baseline as the default so generic
        # attribute access (e.g. ``model.x``) still works; correct prediction
        # must pass an explicit ``stratum``.
        first = baselines[labels[0]]
        model.x = first["x"]
        model.r = first["r"]
        model.d = first["d"]
        model.h0 = first["h0"]
        model.H0 = first["H0"]
        model.tl = None

        return model

    def fit_from_df(
        self,
        df: "pd.DataFrame",
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
        method: str = "efron",
        strata_col: str | None = None,
        tl_col: str | None = None,
    ) -> SemiParametricRegressionModel:
        """
        Fits a Cox PH model using a pandas dataframe as the input.

        Parameters
        ----------

        df: pandas.DataFrame
            The dataframe containing the data.
        x_col: str
            The column name of the observed times.
        Z_cols: list, optional
            The column names of the covariates.
        c_col: str, optional
            The column name of the censoring indicator.
        n_col: str, optional
            The column name of the number of observations at each time point.
        formula: str, optional
            The formula to use for the model. If not provided, the column names
            will be used.
        method: str, optional
            The tie-handling method: ``'breslow'``, ``'efron'``, ``'exact'``
            or ``'kalbfleisch-prentice'`` (alias ``'kp'``). See :meth:`fit`.
        strata_col: str, optional
            The column name of the stratum label. When supplied the model is
            fitted stratified (a separate baseline hazard per stratum, shared
            coefficients); see :meth:`fit`.
        tl_col: str, optional
            The column name of the left-truncation (delayed-entry) times,
            passed to :meth:`fit` as ``tl``. A subject enters the risk sets
            only after its entry time.

        Returns
        -------

        model: SemiParametricRegressionModel
            The fitted model.
        """
        x, c, n, tl, strata, Z, form, feature_names, model_spec = (
            validate_coxph_df_inputs(
                df,
                x_col,
                c_col,
                n_col,
                Z_cols,
                formula,
                tl_col=tl_col,
                strata_col=strata_col,
            )
        )

        model = self.fit(x, Z, c, n, tl=tl, method=method, strata=strata)
        model.formula = form
        model.feature_names = feature_names
        model._model_spec = model_spec

        return model

    def fit_tvc(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        method: str = "efron",
        tol: float = 1e-10,
    ) -> SemiParametricRegressionModel:
        """
        Fit a Cox model with time-varying covariates in start-stop format.

        Each row is one observation interval ``(xl, xr]`` of a subject
        (identified by ``i``) on which the covariate row ``Z`` is constant;
        ``c`` is ``0`` (event) only on the interval that ends at the subject's
        event and ``1`` (right-censored) otherwise. The rows are validated (see
        :func:`~surpyval.univariate.regression.proportional_hazards.tvc.
        handle_tvc`) and fitted as delayed-entry observations -- exact for the
        Cox partial likelihood.

        Parameters
        ----------
        i, xl, xr, c, Z : array_like
            The start-stop interval data: subject id, interval entry time
            ``xl``, exit time ``xr``, censoring flag ``c`` (``0`` event at
            ``xr``, ``1`` right-censored -- surpyval's convention), and the
            per-interval covariates.
        n : array_like, optional
            Count weight per interval row.
        method : {'efron', 'breslow'}, optional
            Tie-handling method. Default ``'efron'``.
        tol : float, optional
            Optimiser tolerance.

        Returns
        -------
        SemiParametricRegressionModel
            The fitted model, with ``is_tvc`` set and TVC-aware prediction
            available through :meth:`~surpyval.univariate.regression.
            semi_parametric_regression_model.SemiParametricRegressionModel.
            predict_tvc`.
        """
        x, c, n_arr, tl, Z_arr, ident = handle_tvc(i, xl, xr, c, Z, n)
        model = self.fit(
            x=x, Z=Z_arr, c=c, n=n_arr, tl=tl, method=method, tol=tol
        )
        model.is_tvc = True
        # Subject ids per *internal* (sorted) row, and the permutation from
        # the caller's row order to the internal order: residuals and
        # cluster-robust SEs align with the internal order, so user-supplied
        # per-row labels must be permuted the same way (#259).
        model.tvc_subject_ids = ident
        model.tvc_row_order = np.lexsort(
            (np.asarray(xl, dtype=float), np.asarray(i))
        )
        return model

    def fit_tvc_from_df(
        self,
        df: "pd.DataFrame",
        id_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: str | list[str],
        n_col: str | None = None,
        method: str = "efron",
    ) -> SemiParametricRegressionModel:
        """
        Fit a time-varying-covariate Cox model from a start-stop DataFrame.

        See :meth:`fit_tvc`; ``Z_cols`` names the covariate column(s) and the
        remaining arguments name the id / ``xl`` / ``xr`` / ``c`` columns.
        """
        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        model = self.fit_tvc(
            i=df[id_col].to_numpy(),
            xl=df[xl_col].to_numpy(),
            xr=df[xr_col].to_numpy(),
            c=df[c_col].to_numpy(),
            Z=df[cols].to_numpy(),
            n=None if n_col is None else df[n_col].to_numpy(),
            method=method,
        )
        model.feature_names = cols
        return model

    def fit_tvc_timeline(
        self,
        i: npt.ArrayLike,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        method: str = "efron",
        tol: float = 1e-10,
    ) -> SemiParametricRegressionModel:
        """
        Fit a time-varying-covariate Cox model from a covariate *timeline*.

        This is the timeline / ``xicnt``-style alternative to
        :meth:`fit_tvc`'s explicit ``(start, stop]`` intervals. Each subject's
        rows give its covariate history: a covariate value ``Z`` takes effect
        at time ``x`` and holds until the subject's next row, with the terminal
        event / censoring marked on the last row's ``c``. The timeline is
        expanded to start-stop intervals (see
        :func:`~surpyval.univariate.regression.proportional_hazards.tvc.
        handle_tvc_timeline`) and fitted exactly as :meth:`fit_tvc`, so it
        gives an identical fit to the equivalent start-stop data.

        Parameters
        ----------
        i : array_like
            Subject identifier for each timeline row (the ``xicnt`` item id).
        x : array_like
            The time each row's covariate value takes effect. Strictly
            increasing within a subject; the first is the entry
            (delayed-entry) time, the last is the event / censoring time.
        Z : array_like
            The covariate vector effective from this row's ``x``. The value on
            a subject's terminal (last) row is ignored.
        c : array_like
            Censoring status; only each subject's last row is read (``0``
            event, ``1`` right-censored).
        n : array_like, optional
            Per-subject count weight (read from the terminal row).
        method : {'efron', 'breslow'}, optional
            Tie-handling method. Default ``'efron'``.
        tol : float, optional
            Optimiser tolerance.

        Returns
        -------
        SemiParametricRegressionModel
            The fitted model, with ``is_tvc`` set.
        """
        i_ss, xl, xr, c_ss, Z_ss, n_ss = handle_tvc_timeline(i, x, Z, c, n)
        return self.fit_tvc(
            i=i_ss,
            xl=xl,
            xr=xr,
            c=c_ss,
            Z=Z_ss,
            n=n_ss,
            method=method,
            tol=tol,
        )

    def fit_tvc_timeline_from_df(
        self,
        df: "pd.DataFrame",
        id_col: str,
        time_col: str,
        Z_cols: str | list[str],
        c_col: str,
        n_col: str | None = None,
        method: str = "efron",
    ) -> SemiParametricRegressionModel:
        """
        Fit a timeline TVC Cox model from a DataFrame.

        See :meth:`fit_tvc_timeline`; ``time_col`` names the change-point time
        column, ``Z_cols`` the covariate column(s) and ``c_col`` the terminal
        event / censoring column (``0`` event, ``1`` censored).
        """
        cols = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        model = self.fit_tvc_timeline(
            i=df[id_col].to_numpy(),
            x=df[time_col].to_numpy(),
            Z=df[cols].to_numpy(),
            c=df[c_col].to_numpy(),
            n=None if n_col is None else df[n_col].to_numpy(),
            method=method,
        )
        model.feature_names = cols
        return model


CoxPH = CoxPH_()
