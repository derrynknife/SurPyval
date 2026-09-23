# This code was created for and sponsored by Cartiga (www.cartiga.com).
# Cartiga makes no representations or warranties in connection with the code
# and waives any and all liability in connection therewith. Your use of the
# code constitutes acceptance of these terms.

# Copyright 2022 Cartiga LLC


from copy import copy
from typing import TYPE_CHECKING, Any, Callable

import autograd.numpy as anp
import numpy as np
import numpy.ma as ma
import numpy.typing as npt
from autograd import grad
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


# Cap on the tie-set size for the average-over-orderings exact method. Its
# risk-set recursion is O(2^d) in the number ``d`` of tied deaths at a single
# time, so a large tie set is both slow and a sign the exact-marginal method is
# the wrong tool -- Efron is the intended approximation there.
_EXACT_MAX_TIES = 12


def _elementary_symmetric(v: Any, d: int) -> Any:
    """The ``d``-th elementary symmetric polynomial ``e_d`` of the entries of
    ``v`` -- i.e. the sum, over every ``d``-subset of ``v``, of the product of
    that subset's entries.

    This is exactly the denominator of the Kalbfleisch-Prentice (discrete
    conditional-logistic) tie contribution: summing ``prod exp(Z_j'b)`` over
    the ``j`` in every size-``d`` subset of the risk set. It is computed by the
    standard O(len(v) * d) recursion rather than by enumerating subsets, and is
    written in ``autograd.numpy`` so the score and Hessian differentiate
    through it.
    """
    # e[k] accumulates e_k; start at e_0 = 1, e_{>0} = 0.
    e = [anp.ones_like(v[0])] + [anp.zeros_like(v[0]) for _ in range(d)]
    for vk in v:
        # Update high-to-low so each e[k] uses the previous iteration's e[k-1].
        for k in range(d, 0, -1):
            e[k] = e[k] + vk * e[k - 1]
    return e[d]


def _exact_ordering_logterm(a: Any, risk_sum: Any) -> Any:
    """``log`` of the average-over-orderings exact tie term.

    For ``d`` tied deaths with risk scores ``a`` (``a_j = exp(Z_j'b)``) drawn
    from a risk set whose total score is ``risk_sum``, the exact (continuous)
    partial-likelihood contribution treats the tied deaths as having occurred
    in some unknown order and sums the sequential Cox contribution over all
    ``d!`` orderings:

        T = sum_{orderings} prod_{m=1..d} 1 / (risk_sum - sum of placed a).

    ``T`` is evaluated with an O(2^d) subset recursion ``h`` over the set of
    already-placed deaths (``h[mask] = sum_{j in mask} h[mask - j] /
    (risk_sum - A(mask - j))``), which is exact and far cheaper than the ``d!``
    orderings. The numerator ``prod a_j = exp(b' * sum Z)`` is added separately
    by the caller, so this returns ``log T`` only.
    """
    d = len(a)
    full = (1 << d) - 1
    # A[mask] = sum of a over the death-bits set in mask.
    A = [anp.zeros_like(risk_sum) for _ in range(1 << d)]
    for mask in range(1, 1 << d):
        low = (mask & -mask).bit_length() - 1
        A[mask] = A[mask ^ (1 << low)] + a[low]

    h = [None] * (1 << d)
    h[0] = anp.ones_like(risk_sum)
    for mask in range(1, 1 << d):
        total = anp.zeros_like(risk_sum)
        m = mask
        while m:
            j = (m & -m).bit_length() - 1
            prev = mask ^ (1 << j)
            # risk_sum - A(prev) is strictly positive: prev omits death j, so
            # it is at most (risk set minus one death), leaving >= a_j > 0.
            total = total + h[prev] / (risk_sum - A[prev])
            m ^= 1 << j
        h[mask] = total
    return anp.log(h[full])


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
    def _autograd_ll_jac_hess(neg_ll: Callable) -> tuple[Callable, Callable]:
        """Wrap a scalar ``autograd.numpy`` negative-log-likelihood into the
        ``(neg_ll, jac_hess)`` contract used by :meth:`fit`.

        The score is the reverse-mode automatic gradient (exact and cheap). The
        observed information is obtained by forward finite-differencing that
        gradient rather than by autograd's forward-over-reverse ``hessian``:
        the exact tie term's O(2^d) risk-set recursion builds a large trace,
        and re-differentiating it a second time makes the full Hessian
        prohibitively slow, whereas ``p + 1`` gradient evaluations stay fast.
        The resulting
        information matrix is symmetrised; it is accurate to O(eps) and only
        feeds the Newton step and the standard-error covariance.
        """
        score = grad(neg_ll)
        eps = 1e-6

        def jac_hess(beta: npt.NDArray) -> tuple:
            beta = np.asarray(beta, dtype=float)
            s0 = np.asarray(score(beta))
            p = beta.shape[0]
            hess_matrix = np.zeros((p, p))
            for j in range(p):
                db = beta.copy()
                db[j] += eps
                hess_matrix[:, j] = (np.asarray(score(db)) - s0) / eps
            hess_matrix = 0.5 * (hess_matrix + hess_matrix.T)
            return s0, hess_matrix

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
        likelihood.
        """
        Ze, event_times, death_idx, risk_idx, S = self._prepare_exact_tie_data(
            x, Z, c, n, tl
        )
        ds = [len(d) for d in death_idx]

        def neg_ll(beta: npt.NDArray) -> float:
            r = anp.exp(anp.dot(Ze, beta))
            total = anp.zeros(())
            for i in range(len(event_times)):
                beta_S = anp.dot(S[i], beta)
                e_d = _elementary_symmetric(r[risk_idx[i]], ds[i])
                total = total + beta_S - anp.log(e_d)
            return -total

        return self._autograd_ll_jac_hess(neg_ll)

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
        likelihood averaged over all orderings of the tied deaths (see
        :func:`_exact_ordering_logterm`). Reduces to Breslow/Efron when there
        are no ties.
        """
        Ze, event_times, death_idx, risk_idx, S = self._prepare_exact_tie_data(
            x, Z, c, n, tl
        )
        ds = [len(d) for d in death_idx]
        too_many = [
            event_times[i] for i, d in enumerate(ds) if d > _EXACT_MAX_TIES
        ]
        if too_many:
            raise ValueError(
                "The 'exact' tie method is O(2^d) in the number of tied "
                "deaths d at a single time; {} deaths tie at time {:g} "
                "(limit {}). "
                "Use method='efron' for heavily tied data.".format(
                    max(ds), too_many[0], _EXACT_MAX_TIES
                )
            )

        def neg_ll(beta: npt.NDArray) -> float:
            r = anp.exp(anp.dot(Ze, beta))
            total = anp.zeros(())
            for i in range(len(event_times)):
                beta_S = anp.dot(S[i], beta)
                risk_sum = anp.sum(r[risk_idx[i]])
                log_t = _exact_ordering_logterm(r[death_idx[i]], risk_sum)
                # L_i = exp(b'S) * T, so log L_i = b'S + log T; for a single
                # death T = 1/risk_sum, recovering the Breslow term b'S -
                # log(risk_sum).
                total = total + beta_S + log_t
            return -total

        return self._autograd_ll_jac_hess(neg_ll)

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
            The covariates of the model.
        c: array-like, optional
            The censoring indicator. 0 if observed (event),
            1 if right-censored.
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

        model: SemiParametricProportionalHazardsModel
            The fitted model.
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

        Returns
        -------

        model: SemiParametricProportionalHazardsModel
            The fitted model.
        """
        x, c, n, Z, form, feature_names, model_spec = validate_coxph_df_inputs(
            df, x_col, c_col, n_col, Z_cols, formula
        )

        strata = None if strata_col is None else df[strata_col].to_numpy()
        model = self.fit(x, Z, c, n, method=method, strata=strata)
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
