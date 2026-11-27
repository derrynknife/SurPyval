"""
Rank-based semi-parametric accelerated-failure-time regression.

**In plain words.**
The accelerated-failure-time model says each covariate stretches or
shrinks the time to failure by a fixed factor, and leaves the shape of the
failure-time distribution unspecified. Once the coefficients are right,
the "residual" log times ``log x_i + beta'Z_i`` of all the units come from
one and the same distribution, so no covariate should help to say which
unit's residual is the smaller. The rank estimators choose the coefficients
that make that so: they compare every event with the units whose residual
is still "at risk" when it happens (is no smaller), exactly as a log-rank
test compares groups, and solve for the coefficients at which the
covariates of the units that fail look like those of the units at risk.
They use only the *order* of the residuals, so a few extreme times do not
drag the fit, and censoring is handled as the log-rank test handles it.

Two weightings are offered. **Gehan** weights each event by the number at
risk, which makes the estimating function the gradient of a convex loss,
so every solution is a global one and it is found reliably.
**Log-rank** gives each event
the same weight, which is the more efficient choice when the errors have
an extreme-value (Weibull-like) shape; its estimating function is not
monotone, so the solution is found by iterating Gehan fits (below).

**The model and the estimating functions.**
With :math:`Y_i = \\log x_i` and the textbook slope :math:`\\gamma`
(surpyval reports :math:`\\beta = -\\gamma`, so a positive coefficient
shortens life, as in ``WeibullAFT`` and ``BuckleyJames``),

.. math::
    Y_i = \\gamma' Z_i + \\varepsilon_i ,\\qquad
    e_i(\\gamma) = Y_i - \\gamma' Z_i .

With :math:`\\delta_i` the event flag and :math:`w_i` the counts, the
weighted log-rank estimating function (Tsiatis 1990; Jin, Lin, Wei and
Ying 2003) is

.. math::
    U_\\phi(\\gamma) = \\sum_i w_i \\delta_i \\, \\phi(e_i)
    \\Big\\{ Z_i - \\frac{\\sum_j w_j Z_j 1(e_j \\ge e_i)}
                       {\\sum_j w_j 1(e_j \\ge e_i)} \\Big\\},

with :math:`\\phi = 1` (log-rank) or :math:`\\phi(e_i) = \\sum_j w_j
1(e_j \\ge e_i)` (Gehan). The Gehan function is
:math:`\\sum_i \\sum_j w_i w_j \\delta_i (Z_i - Z_j) 1(e_j \\ge e_i)`, the
gradient of the convex loss

.. math::
    L_G(\\gamma) = \\sum_i \\sum_j w_i w_j \\delta_i \\,
    \\{e_j(\\gamma) - e_i(\\gamma)\\}^+ ,

which is a linear programme.

**Induced smoothing.**
Both functions are step functions of :math:`\\gamma`. Brown and Wang
(2005, 2007) replace :math:`\\gamma` by :math:`\\gamma + \\Gamma^{1/2} N`,
:math:`N` standard normal, and take the expectation: each indicator
:math:`1(e_j \\ge e_i)` becomes :math:`\\Phi(d_{ij} / r_{ij})`, with
:math:`d_{ij} = e_j - e_i` and
:math:`r_{ij}^2 = (Z_i - Z_j)' \\Gamma (Z_i - Z_j)`, and the Gehan loss
becomes :math:`\\sum w_i w_j \\delta_i \\{d_{ij} \\Phi(d_{ij}/r_{ij}) +
r_{ij} \\varphi(d_{ij}/r_{ij})\\}` (the mean of the positive part of a
normal), smooth and still convex. :math:`\\Gamma` is the estimate's own
covariance, so the smoothing is of the order of the estimate's error and
vanishes as the data grow. It starts at the inverse of the covariates'
covariance over the number of observations and is updated to the
sandwich covariance below (Brown and Wang 2007; Chiou, Kang and Yan
2014) until no standard deviation it implies changes by more than 1%;
it is then held, so the estimate is the root of one smooth function.
The estimate reported is the root of the
smoothed function, not of the step function. The two differ by a
fraction of a standard error that shrinks as the data grow: on 100
units, a tenth of one or less for Gehan (against the exact linear
programme) and up to about 0.3 for log-rank (median about 0.15, over
40 seeded data sets). As the smoothing covariance shrinks, the smoothed
root tends to a root of the step function.

The log-rank root is found as Jin, Lin, Wei and Ying (2003) find it: from
the Gehan estimate, refit the weighted Gehan loss with each event's weight
:math:`1 / \\tilde R_i`, the inverse of its (smoothed) number at risk at
the previous estimate, until the estimate stops moving. A fixed point is
a root of the smoothed log-rank function. The unsmoothed log-rank
function has several exact roots, within a small fraction of a standard
error of each other (up to 0.06 of one on the test data); the estimate
reported is the smoothed root this iteration reaches from the Gehan
estimate, and the exact roots near it are within the smoothing gap above
of it.

**Standard errors.**
The sandwich :math:`V = A^{-1} B A^{-\\top}`. :math:`A` is the derivative
of the smoothed estimating function at the estimate (for Gehan the
Hessian of the smoothed loss). :math:`B` is the empirical variance of the
estimating function's influence terms, computed from the unsmoothed
indicators: for Gehan, a U-statistic, the Hajek projection
:math:`g_i = \\sum_j w_j \\{h(i, j) + h(j, i)\\}` with
:math:`h(i, j) = \\delta_i (Z_i - Z_j) 1(e_j \\ge e_i)`; for log-rank,
the martingale residual of the score,
:math:`\\eta_i = \\delta_i \\{Z_i - \\bar Z(e_i)\\} - \\sum_k w_k
\\delta_k 1(e_i \\ge e_k) \\{Z_i - \\bar Z(e_k)\\} / R_k`.

References: Tsiatis, A. A. (1990) Estimating regression parameters using
linear rank tests for censored data. *Annals of Statistics* 18,
354-372. Jin, Z., Lin, D. Y., Wei, L. J. and Ying, Z. (2003) Rank-based
inference for the accelerated failure time model. *Biometrika* 90,
341-353. Brown, B. M. and Wang, Y.-G. (2005) Standard errors and
covariance matrices for smoothed rank estimators. *Biometrika* 92,
149-158; (2007) Induced smoothing for rank regression with censored
survival times. *Statistics in Medicine* 26, 828-836. Chiou, S. H.,
Kang, S. and Yan, J. (2014) Fast accelerated failure time modeling for
case-cohort data. *Statistics and Computing* 24, 559-568.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr
from scipy.stats import norm

if TYPE_CHECKING:
    import pandas as pd

from surpyval.serialisation import require_model_tag, stamp_schema
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils.data_summary import data_summary
from surpyval.utils.dataframe import check_columns
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.linalg import percentile_bounds
from surpyval.utils.rng import as_generator
from surpyval.utils.surpyval_data import accepts_surpyval_data
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

from .._aliasing import expand, warn_aliased
from .._residual_aft import ResidualAFTModel, residual_km
from .._summary import coefficient_table
from ..buckley_james.buckley_james import _aliased
from ..regression_data import (
    FrameDesign,
    canonical_rows,
    restore_covariate_meta,
    semi_parametric_inputs,
    serialise_covariate_meta,
)

#: The rank weightings offered.
RANK_WEIGHTS = ("gehan", "logrank")
#: The bound methods of :meth:`RankAFTModel.param_cb`.
CB_METHODS = ("wald", "bootstrap")
#: The pairs (event rows times all rows) formed at once by the smoothed
#: sums, to bound their memory.
_PAIR_BLOCK = 500_000
_SQRT_2PI = np.sqrt(2.0 * np.pi)


# -- the unsmoothed functions (step functions of gamma) ----------------------


def _at_or_above(e: npt.NDArray, w: npt.NDArray, V: npt.NDArray) -> tuple:
    """For each ``i``: ``sum_j w_j 1(e_j >= e_i)`` and ``sum_j w_j V_j
    1(e_j >= e_i)`` (``V`` one row per observation)."""
    order = np.argsort(e, kind="mergesort")
    es = e[order]
    wv = (w[:, None] * V)[order]
    # Sums from the top: position k holds the sum over sorted rows k...n-1.
    top_w = np.concatenate([np.cumsum(w[order][::-1])[::-1], [0.0]])
    top_v = np.vstack(
        [np.cumsum(wv[::-1], axis=0)[::-1], np.zeros(V.shape[1])]
    )
    k = np.searchsorted(es, e, side="left")
    return top_w[k], top_v[k]


def _at_or_below(e: npt.NDArray, a: npt.NDArray, V: npt.NDArray) -> tuple:
    """For each ``i``: ``sum_k a_k 1(e_k <= e_i)`` and ``sum_k a_k V_k
    1(e_k <= e_i)``."""
    order = np.argsort(e, kind="mergesort")
    es = e[order]
    low_a = np.concatenate([[0.0], np.cumsum(a[order])])
    low_v = np.vstack(
        [np.zeros(V.shape[1]), np.cumsum((a[:, None] * V)[order], axis=0)]
    )
    k = np.searchsorted(es, e, side="right")
    return low_a[k], low_v[k]


def gehan_function(
    gamma: npt.ArrayLike,
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
) -> npt.NDArray:
    """The unsmoothed Gehan estimating function
    :math:`\\sum_i \\sum_j w_i w_j \\delta_i (Z_i - Z_j) 1(e_j \\ge e_i)`
    at the textbook slope ``gamma`` (a step function)."""
    e = Y - Z @ np.asarray(gamma, dtype=float)
    R, S = _at_or_above(e, w, Z)
    return ((w * delta)[:, None] * (Z * R[:, None] - S)).sum(axis=0)


def gehan_loss(
    gamma: npt.ArrayLike,
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
    a: npt.NDArray | None = None,
) -> float:
    """The (unsmoothed) Gehan loss :math:`\\sum_i \\sum_j a_i w_j
    \\delta_i \\{e_j - e_i\\}^+`, ``a`` the events' weights (``w`` when
    ``None``), whose minimiser is the Gehan estimate."""
    a = w if a is None else a
    e = Y - Z @ np.asarray(gamma, dtype=float)
    R, S = _at_or_above(e, w, e[:, None])
    return float(((a * delta) * (S[:, 0] - e * R)).sum())


def logrank_function(
    gamma: npt.ArrayLike,
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
) -> npt.NDArray:
    """The unsmoothed log-rank estimating function
    :math:`\\sum_i w_i \\delta_i \\{Z_i - \\bar Z(e_i)\\}` at ``gamma``."""
    e = Y - Z @ np.asarray(gamma, dtype=float)
    R, S = _at_or_above(e, w, Z)
    return ((w * delta)[:, None] * (Z - S / R[:, None])).sum(axis=0)


def gehan_lp(
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
    a: npt.NDArray | None = None,
) -> npt.NDArray:
    """The exact minimiser of :func:`gehan_loss`, as a linear programme
    (HiGHS): minimise :math:`\\sum a_i w_j \\delta_i u_{ij}` over
    :math:`\\gamma` and :math:`u_{ij} \\ge 0` with
    :math:`u_{ij} \\ge e_j - e_i`. It has a variable per (event, unit)
    pair, so it is for small data: the fit itself solves the smoothed
    problem (see the module docstring)."""
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix, hstack

    a = w if a is None else a
    ev = np.flatnonzero(delta == 1)
    ii = np.repeat(ev, Y.size)
    jj = np.tile(np.arange(Y.size), ev.size)
    keep = ii != jj
    ii, jj = ii[keep], jj[keep]
    m, p = ii.size, Z.shape[1]
    cost = np.concatenate([np.zeros(p), a[ii] * w[jj]])
    # u_ij >= (Y_j - Y_i) - gamma'(Z_j - Z_i), as
    # -(Z_j - Z_i)'gamma - u_ij <= -(Y_j - Y_i).
    A_g = coo_matrix(-(Z[jj] - Z[ii]))
    A_u = coo_matrix((-np.ones(m), (np.arange(m), np.arange(m))), (m, m))
    res = linprog(
        cost,
        A_ub=hstack([A_g, A_u]).tocsr(),
        b_ub=-(Y[jj] - Y[ii]),
        bounds=[(None, None)] * p + [(0, None)] * m,
        method="highs",
    )
    if res.status != 0:  # pragma: no cover - a feasible bounded LP
        raise RuntimeError(f"the Gehan linear programme failed: {res.message}")
    return np.asarray(res.x[:p])


# -- the smoothed sums ------------------------------------------------------


def _smoothed_sums(
    gamma: npt.NDArray,
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
    chol: npt.NDArray,
    parts: str = "all",
) -> dict:
    """The induced-smoothed sums over each event ``i``'s pairs ``j``:

    - ``R``: :math:`\\sum_j w_j \\Phi_{ij}`, the smoothed number at risk;
    - ``U``: :math:`\\sum_j w_j (Z_i - Z_j) \\Phi_{ij}`;
    - ``L``: :math:`\\sum_j w_j E\\{d_{ij} + r_{ij} N\\}^+`;
    - ``H``: :math:`\\sum_j w_j (Z_i - Z_j)(Z_i - Z_j)' \\varphi_{ij} /
      r_{ij}`, the derivative of ``U``;
    - ``dR``: :math:`\\sum_j w_j (Z_i - Z_j) \\varphi_{ij} / r_{ij}`, the
      derivative of ``R``;

    with :math:`\\Phi_{ij} = \\Phi(d_{ij} / r_{ij})`, :math:`d_{ij} = e_j -
    e_i`, :math:`r_{ij} = \\lVert L'(Z_i - Z_j) \\rVert` (``chol`` the
    Cholesky factor :math:`L` of :math:`\\Gamma`). A pair with
    :math:`r_{ij} = 0` (equal covariates) is unsmoothed:
    :math:`1(d_{ij} \\ge 0)`, as the step function counts it.
    ``parts`` is ``"all"``, ``"loss"`` (``L`` only) or ``"risk"`` (``R``
    only)."""
    e = Y - Z @ gamma
    ev = np.flatnonzero(delta == 1)
    n, p = Z.shape
    # The sums over j of products with Z_i - Z_j are formed from matrix
    # products, sum_j g_ij (Z_i - Z_j) = Z_i sum_j g_ij - sum_j g_ij Z_j
    # (and so on for the outer products), on covariates centred to keep
    # the cancellation small.
    Zc = Z - Z.mean(axis=0)
    ZZ = (Zc[:, :, None] * Zc[:, None, :]).reshape(n, p * p)
    ZL = Zc @ chol
    want = {
        "all": ("R", "U", "L", "H", "dR"),
        "loss": ("L",),
        "risk": ("R",),
    }[parts]
    shapes = {"R": (), "U": (p,), "L": (), "H": (p, p), "dR": (p,)}
    out = {k: np.empty((ev.size, *shapes[k])) for k in want}
    step = max(1, _PAIR_BLOCK // max(n, 1))
    for start in range(0, ev.size, step):
        rows = ev[start : start + step]
        sl = slice(start, start + rows.size)
        # r_ij, column by column, so that equal covariates give exactly 0.
        r2 = np.zeros((rows.size, n))
        for k in range(ZL.shape[1]):
            r2 += (ZL[rows, k, None] - ZL[None, :, k]) ** 2
        r = np.sqrt(r2)
        d = e[None, :] - e[rows, None]
        smooth = r > 0
        r_s = np.where(smooth, r, 1.0)
        u = d / r_s
        Zi = Zc[rows]
        if "L" in want or "H" in want or "dR" in want:
            dens = np.exp(-0.5 * u * u) / _SQRT_2PI
        if "R" in want or "U" in want or "L" in want:
            cdf = np.where(smooth, ndtr(u), (d >= 0).astype(float))
        if "L" in want:
            plus = np.where(smooth, d * cdf + r_s * dens, np.maximum(d, 0.0))
            out["L"][sl] = plus @ w
        if "R" in want or "U" in want:
            Phi = cdf * w
            R = Phi.sum(axis=1)
            if "R" in want:
                out["R"][sl] = R
            if "U" in want:
                out["U"][sl] = Zi * R[:, None] - Phi @ Zc
        if "H" in want or "dR" in want:
            g = np.where(smooth, dens / r_s, 0.0) * w
            g0 = g.sum(axis=1)
            g1 = g @ Zc
            out["dR"][sl] = Zi * g0[:, None] - g1
            ZiZi = Zi[:, :, None] * Zi[:, None, :]
            cross = Zi[:, :, None] * g1[:, None, :]
            out["H"][sl] = (
                ZiZi * g0[:, None, None]
                - cross
                - np.swapaxes(cross, 1, 2)
                + (g @ ZZ).reshape(rows.size, p, p)
            )
    return out


def smoothed_gehan(
    gamma: npt.ArrayLike,
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
    Gamma: npt.NDArray,
    a: npt.NDArray | None = None,
) -> tuple[float, npt.NDArray, npt.NDArray]:
    """The induced-smoothed (weighted) Gehan loss, its gradient (the
    smoothed estimating function) and its Hessian at ``gamma``, with
    smoothing covariance ``Gamma`` and event weights ``a`` (``w`` when
    ``None``)."""
    ev = delta == 1
    a_ev = (w if a is None else a)[ev]
    s = _smoothed_sums(
        np.asarray(gamma, dtype=float), Y, delta, Z, w, _chol(Gamma)
    )
    return (
        float(a_ev @ s["L"]),
        a_ev @ s["U"],
        np.einsum("i,ipq->pq", a_ev, s["H"]),
    )


def _chol(Gamma: npt.NDArray) -> npt.NDArray:
    """A factor ``L`` with ``L L' = Gamma`` (``Gamma`` symmetric,
    positive semi-definite)."""
    vals, vecs = np.linalg.eigh(0.5 * (Gamma + Gamma.T))
    return vecs * np.sqrt(np.clip(vals, 0.0, None))


# -- the fit ----------------------------------------------------------------


def _newton(
    gamma: npt.NDArray,
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
    a: npt.NDArray,
    Gamma: npt.NDArray,
    tol: float,
    max_iter: int,
) -> npt.NDArray:
    """Minimise the smoothed weighted Gehan loss (convex) by Newton's
    method with a backtracking line search, from ``gamma``; the steps are
    measured in units of the smoothing standard deviations."""
    scale = np.sqrt(np.clip(np.diag(Gamma), 1e-300, None))
    chol = _chol(Gamma)
    loss, grad, hess = smoothed_gehan(gamma, Y, delta, Z, w, Gamma, a)
    for _ in range(max_iter):
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, grad, rcond=None)[0]
        t = 1.0
        while True:
            trial = gamma - t * step
            l_t = float(
                a[delta == 1]
                @ _smoothed_sums(trial, Y, delta, Z, w, chol, "loss")["L"]
            )
            if l_t <= loss - 1e-4 * t * float(grad @ step) or t < 1e-10:
                break
            t *= 0.5
        moved = np.max(np.abs(t * step) / scale)
        gamma = trial
        if moved < tol:
            break
        loss, grad, hess = smoothed_gehan(gamma, Y, delta, Z, w, Gamma, a)
    return gamma


def _weighted_sandwich(
    gamma: npt.NDArray,
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
    Gamma: npt.NDArray,
    rank_weights: str,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """The sandwich covariance ``A^-1 B A^-T`` of the estimate at
    ``gamma`` (see the module docstring), with ``A`` and ``B``."""
    N = w.sum()
    e = Y - Z @ gamma
    ev = delta == 1
    s = _smoothed_sums(gamma, Y, delta, Z, w, _chol(Gamma))
    w_ev = w[ev]
    R, S = _at_or_above(e, w, Z)
    if rank_weights == "gehan":
        A = np.einsum("i,ipq->pq", w_ev, s["H"]) / N**2
        # Hajek projection of the U-statistic: g_i = sum_j w_j {h(i, j) +
        # h(j, i)}, h(i, j) = delta_i (Z_i - Z_j) 1(e_j >= e_i).
        first = delta[:, None] * (Z * R[:, None] - S)
        C0, C1 = _at_or_below(e, w * delta, Z)
        g = (first + C1 - Z * C0[:, None]) / N
    else:
        Rt = s["R"]
        A = (
            np.einsum("i,ipq->pq", w_ev / Rt, s["H"])
            - np.einsum("i,ip,iq->pq", w_ev / Rt**2, s["U"], s["dR"])
        ) / N
        # The score's martingale residuals.
        Zbar = S / R[:, None]
        k = w * delta / R
        C0, C1 = _at_or_below(e, k, Zbar)
        g = delta[:, None] * (Z - Zbar) - (Z * C0[:, None] - C1)
    g = g - (w[:, None] * g).sum(axis=0) / N
    B = (w[:, None] * g).T @ g / N**2
    A_inv = np.linalg.inv(A)
    V = A_inv @ B @ A_inv.T
    return 0.5 * (V + V.T), A, B


def _start_gamma(Z: npt.NDArray, w: npt.NDArray) -> npt.NDArray:
    """The first smoothing covariance: the inverse of the covariates'
    covariance over the number of observations (a least-squares
    covariance with unit error variance), which rescales with the
    covariates."""
    N = w.sum()
    Zc = Z - (w[:, None] * Z).sum(axis=0) / N
    cov = (w[:, None] * Zc).T @ Zc / N
    return np.linalg.inv(cov) / N


def _fit_gamma(
    Y: npt.NDArray,
    delta: npt.NDArray,
    Z: npt.NDArray,
    w: npt.NDArray,
    rank_weights: str,
    tol: float,
    max_iter: int,
    Gamma: npt.NDArray | None = None,
    update: bool = True,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, int, bool]:
    """The induced-smoothing estimate. Each round refits the smoothed
    (weighted) Gehan loss by Newton's method from the last estimate; for
    log-rank the event weights are then set to ``w / R~`` at the new
    estimate (the first round is the Gehan fit). While ``update``, the
    smoothing covariance is then set to the sandwich covariance there,
    until no standard deviation it implies changes by more than 1%; it is
    then held, so the estimate is the exact root of one smooth function
    (the sandwich's middle is a step function of the estimate, and kept
    updating it made the last digits cycle). The fit stops when a round,
    with the smoothing held, moves the estimate by less than ``tol`` of
    its standard deviation and leaves the weights within ``tol``
    (relative). Returns ``(gamma, covariance, smoothing covariance,
    rounds, converged)``."""
    Gamma = _start_gamma(Z, w) if Gamma is None else Gamma
    gamma = np.zeros(Z.shape[1])
    a = w.copy()
    converged = False
    rounds = 0
    for rounds in range(1, max_iter + 1):
        # For log-rank after the first (Gehan) round, one Newton step per
        # update of the weights: the fixed point is the same (a step is 0
        # only where the weighted gradient is), and a full solve for
        # weights about to change was wasted work.
        steps = 1 if rank_weights == "logrank" and rounds > 1 else 100
        new = _newton(gamma, Y, delta, Z, w, a, Gamma, tol * 1e-2, steps)
        scale = np.sqrt(np.clip(np.diag(Gamma), 1e-300, None))
        moved = np.max(np.abs(new - gamma) / scale)
        gamma = new
        settled = True
        if rank_weights == "logrank":
            Rt = _smoothed_sums(
                gamma, Y, delta, Z, w, _chol(Gamma), parts="risk"
            )["R"]
            a_new = np.zeros_like(w)
            a_new[delta == 1] = w[delta == 1] / Rt
            # The first round, the Gehan fit, is not a fixed point.
            settled = rounds > 1 and np.allclose(a_new, a, rtol=tol, atol=0.0)
            a = a_new
        held = not update
        if update:
            V, _, _ = _weighted_sandwich(
                gamma, Y, delta, Z, w, Gamma, rank_weights
            )
            update = not np.allclose(
                np.sqrt(np.diag(V)), scale, rtol=1e-2, atol=0.0
            )
            Gamma = V
        if held and moved < tol and settled:
            converged = True
            break
    V, _, _ = _weighted_sandwich(gamma, Y, delta, Z, w, Gamma, rank_weights)
    return gamma, V, Gamma, rounds, converged


# -- the model --------------------------------------------------------------


class RankAFTModel(ResidualAFTModel):
    """
    A fitted rank-based (Gehan or log-rank) accelerated-failure-time model.

    Predictions use the Kaplan-Meier estimate of the residuals, as
    ``BuckleyJamesModel``'s do: ``sf(t | Z) = S_eps(log t + beta'Z)``.
    ``coef`` are the covariate coefficients in surpyval's
    accelerated-failure convention: a positive coefficient accelerates
    failure (shortens life). ``covariance()`` is the induced-smoothing
    sandwich covariance, so ``summary()`` and ``param_cb()`` give Wald
    standard errors and intervals without resampling.

    Examples
    --------
    On the Rossi recidivism data, where ``arrest`` is 1 for an arrest (so
    the censoring flag is ``1 - arrest``):

    >>> from surpyval import RankAFT
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> Z = df[["fin", "age", "prio"]].values
    >>> model = RankAFT.fit(x, Z, c=c)
    >>> model.beta.round(4)
    array([-0.2285, -0.0408,  0.0705])
    >>> model.standard_errors().round(4)
    array([0.147 , 0.0154, 0.0194])
    """

    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: Any = None
    #: Covariates of the wrong width are refused by name (#657).
    _CHECKS_WIDTH = True

    #: The covariate coefficients (``params`` and ``coef`` are the same
    #: array), in the accelerated-failure convention.
    beta: npt.NDArray
    params: npt.NDArray
    coef: npt.NDArray
    _se: npt.NDArray
    #: ``"gehan"`` or ``"logrank"``.
    rank_weights: str
    #: The rounds of the fit (Newton refits, each followed by an update of
    #: the smoothing covariance and, for log-rank, the weights), and
    #: whether the estimate settled.
    n_iter: int
    converged: bool
    #: Fitted by estimating equations, not a likelihood: ``"not
    #: applicable"`` (one of ``MAXIMUM_STATES``, surpyval.utils.no_maximum).
    maximum: str = "not applicable"
    _cov: npt.NDArray
    _smoothing: npt.NDArray

    _ALIASED_WHY = (
        "a constant column, which is the intercept the rank estimating "
        "equations cannot see, or a linear combination of the others"
    )
    _NO_LIKELIHOOD_WHY = (
        "the rank estimator solves the (Gehan or log-rank) estimating "
        "equations of the residuals' ranks, with the residual distribution "
        "left unspecified; there is no likelihood to maximise"
    )

    def __init__(
        self,
        beta: npt.ArrayLike,
        cov: npt.NDArray,
        smoothing: npt.NDArray,
        resid: npt.NDArray,
        resid_surv: npt.NDArray,
        rank_weights: str,
        n_iter: int,
        converged: bool,
        data: tuple | None,
    ) -> None:
        self.beta = np.asarray(beta, dtype=float)
        self.params = self.beta
        self.coef = self.beta
        self._cov = np.asarray(cov, dtype=float)
        self._se = np.sqrt(np.diag(self._cov))
        self._smoothing = np.asarray(smoothing, dtype=float)
        self._resid = resid
        self._resid_surv = resid_surv
        self.rank_weights = rank_weights
        self.n_iter = n_iter
        self.converged = converged
        self._data = data  # (Y, delta, Z, w) for the bootstrap

    def standard_errors(self) -> npt.NDArray:
        """The coefficients' standard errors, the square roots of the
        diagonal of :meth:`covariance` (``nan`` for an aliased one)."""
        return self._se.copy()

    def covariance(self) -> npt.NDArray:
        """The induced-smoothing sandwich covariance of the coefficients
        (``nan`` in an aliased coefficient's row and column), in the
        accelerated-failure sign (which leaves it unchanged)."""
        return self._cov.copy()

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted model to a plain, JSON-serialisable dict:
        the coefficients, their covariance, the smoothing covariance, the
        residual survival step arrays the predictions read, and the fit
        data ``(Y, delta, Z, w)`` so a restored model can still run the
        bootstrap of :meth:`param_cb`.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out = {
            "model": "RankAFTModel",
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "covariance": np.asarray(self._cov, dtype=float).tolist(),
            "smoothing": np.asarray(self._smoothing, dtype=float).tolist(),
            "resid": np.asarray(self._resid, dtype=float).tolist(),
            "resid_surv": np.asarray(self._resid_surv, dtype=float).tolist(),
            "rank_weights": self.rank_weights,
            "n_iter": int(self.n_iter),
            "converged": bool(self.converged),
        }
        if self._data is not None:
            Y, delta, Z, w = self._data
            out["data"] = {
                "Y": np.asarray(Y, dtype=float).tolist(),
                "delta": np.asarray(delta, dtype=float).tolist(),
                "Z": np.asarray(Z, dtype=float).tolist(),
                "w": np.asarray(w, dtype=float).tolist(),
            }
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> RankAFTModel:
        """
        Rebuild a rank-based AFT model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(model_dict, "RankAFTModel", "a rank-based AFT model")
        data = None
        if "data" in model_dict:
            d = model_dict["data"]
            data = tuple(
                np.array(d[k], dtype=float) for k in ("Y", "delta", "Z", "w")
            )
        p = len(model_dict["beta"])
        out = cls(
            np.array(model_dict["beta"], dtype=float),
            np.array(model_dict["covariance"], dtype=float).reshape(p, p),
            np.array(model_dict["smoothing"], dtype=float),
            np.array(model_dict["resid"], dtype=float),
            np.array(model_dict["resid_surv"], dtype=float),
            str(model_dict["rank_weights"]),
            int(model_dict["n_iter"]),
            bool(model_dict["converged"]),
            data,
        )
        restore_covariate_meta(out, model_dict)
        return out

    # -- inference -------------------------------------------------------

    def summary(
        self,
        alpha_ci: float = 0.05,
        n_boot: int | None = None,
        random_state: Any = None,
    ) -> pd.DataFrame:
        """
        The coefficient table, in the layout of ``CoxPH``'s
        :meth:`summary`: each coefficient (in the accelerated-failure
        convention) and ``exp(coef)``, the factor by which a unit of the
        covariate shortens the life, with the sandwich standard error, a
        two-sided Wald interval, ``z`` and its p-value. With ``n_boot``
        the intervals are instead the percentile bootstrap intervals of
        :meth:`param_cb` (``n_boot`` refits, seeded by ``random_state``),
        as ``BuckleyJamesModel.summary`` gives them.

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.
        n_boot : int, optional
            The number of bootstrap refits for the intervals; ``None``
            (the default) gives the Wald intervals.
        random_state : None, int or numpy.random.Generator, optional
            The seed of the bootstrap.

        Examples
        --------
        >>> from surpyval import RankAFT
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = RankAFT.fit(x, df[["fin", "age", "prio"]].values, c=c)
        >>> model.summary()["p"].round(4).tolist()
        [0.1201, 0.0078, 0.0003]
        """
        check_alpha_ci(alpha_ci)
        beta = np.asarray(self.beta, dtype=float)
        table = coefficient_table(
            self.parameter_names, beta, self._se, alpha_ci
        )
        if n_boot is not None:
            bounds = self.param_cb(
                None,
                alpha_ci,
                method="bootstrap",
                n_boot=n_boot,
                random_state=random_state,
            )
            level = f"{100 * (1 - alpha_ci):g}%"
            lower, upper = bounds[:, 0], bounds[:, 1]
            table["coef lower " + level] = lower
            table["coef upper " + level] = upper
            table["exp(coef) lower " + level] = np.exp(lower)
            table["exp(coef) upper " + level] = np.exp(upper)
        return table

    def param_cb(
        self,
        name: str | None = None,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
        n_boot: int = 200,
        random_state: Any = None,
    ) -> npt.NDArray:
        """
        Confidence bounds on the coefficients.

        - ``"wald"`` (the default): the estimate plus or minus a normal
          quantile times its sandwich standard error (see
          :meth:`covariance`).
        - ``"bootstrap"``: the percentile bootstrap, as for
          ``BuckleyJamesModel``: the observations (rows expanded by their
          counts) are resampled ``n_boot`` times and refitted.

        A likelihood-ratio bound is refused: the model has no likelihood.

        Parameters
        ----------
        name : str, optional
            A coefficient, one of :attr:`parameter_names`. ``None`` (the
            default) bounds every coefficient.
        alpha_ci : float, optional
            One minus the confidence level. Default 0.05.
        bound : ('two-sided', 'lower', 'upper'), optional
            A one-sided bound puts all of ``alpha_ci`` on one side.
        method : ('wald', 'bootstrap'), optional
            As above. Default ``'wald'``.
        n_boot : int, optional
            The bootstrap's resamples. Default 200.
        random_state : None, int or numpy.random.Generator, optional
            The bootstrap's seed.

        Returns
        -------
        numpy.ndarray
            For every coefficient, an ``(n_coef, 2)`` array of ``[lower,
            upper]`` bounds (``(n_coef,)`` one-sided); for one, its
            ``[lower, upper]`` (or the one bound).

        Examples
        --------
        >>> from surpyval import RankAFT
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = RankAFT.fit(x, df[["fin", "age", "prio"]].values, c=c)
        >>> model.param_cb("coef_2").round(4)
        array([0.0324, 0.1086])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        if method in ("lr", "likelihood", "likelihood-ratio", "profile"):
            raise ValueError(
                "RankAFTModel has no likelihood-ratio bound: the rank "
                "estimator has no likelihood. Use method='wald' (the "
                "sandwich standard errors) or method='bootstrap', or fit a "
                "parametric AFT model (WeibullAFT, LogNormalAFT, ...) for "
                "a profile-likelihood interval."
            )
        check_option("method", method, CB_METHODS)
        names = list(self.parameter_names)
        if name is not None and name not in names:
            raise ValueError(
                f"Unknown coefficient {name!r}; this model's coefficients "
                f"are {names}."
            )
        if method == "bootstrap":
            boot = self._bootstrap_draws(n_boot, random_state)
            out = np.asarray(percentile_bounds(boot, alpha_ci, bound))
        else:
            beta = np.asarray(self.beta, dtype=float)
            if bound == "two-sided":
                q = norm.ppf(1 - alpha_ci / 2)
                out = np.column_stack(
                    [beta - q * self._se, beta + q * self._se]
                )
            else:
                q = norm.ppf(1 - alpha_ci)
                sign = -1.0 if bound == "lower" else 1.0
                out = beta + sign * q * self._se
        return out if name is None else out[names.index(name)]

    def _bootstrap_draws(self, n_boot: int, random_state: Any) -> list:
        """The coefficients of ``n_boot`` refits to resampled data."""
        if self._data is None:
            raise ValueError(
                "The bootstrap needs the fit data, which this model does not "
                "carry; use method='wald'."
            )
        Y, delta, Z, w = self._data
        p = self.beta.size
        kept = np.flatnonzero(~np.isnan(self.beta))
        Z = Z[:, kept]
        rng = as_generator(random_state)
        # As BuckleyJamesModel: the counts are frequency weights, so the
        # observations (rows expanded by their counts) are resampled.
        units = np.repeat(np.arange(Y.shape[0]), np.round(w).astype(int))
        whole = np.allclose(w, np.round(w)) and units.size > 0
        boot = []
        dropped = unsettled = 0
        for _ in range(n_boot):
            if whole:
                idx = units[rng.integers(0, units.size, size=units.size)]
                w_b = np.ones(idx.size)
            else:
                counts = rng.multinomial(Y.shape[0], w / w.sum())
                idx = np.flatnonzero(counts)
                w_b = w[idx] * counts[idx]
            if not np.any(delta[idx] == 1):
                dropped += 1
                continue
            try:
                g, _, _, _, settled = _fit_gamma(
                    Y[idx],
                    delta[idx],
                    Z[idx],
                    w_b,
                    self.rank_weights,
                    1e-4,
                    50,
                )
            except np.linalg.LinAlgError:
                dropped += 1
                continue
            unsettled += not settled
            boot.append(expand(-g, kept, p))
        if dropped or unsettled:
            warnings.warn(
                f"Of {n_boot} bootstrap refits, {dropped} could not be "
                "fitted (no event, or a singular design, in the resample) "
                f"and are left out, and {unsettled} did not settle in 50 "
                "rounds and are kept at their last estimate. The bounds "
                "are from the rest; raise n_boot, or use method='wald'."
            )
        return boot

    def __repr__(self) -> str:
        label = "Gehan" if self.rank_weights == "gehan" else "log-rank"
        lines = [
            "Rank-based AFT SurPyval Model",
            "=============================",
            "Kind                : Semi-Parametric AFT",
            f"Rank weights        : {label}",
            f"Converged           : {self.converged} ({self.n_iter} rounds)",
        ]
        if self._data is not None:
            Y, delta, _, w = self._data
            lines.append(
                "Data                : "
                + data_summary(1 - np.asarray(delta, dtype=int), w, x=Y)
            )
        lines += ["Coefficients (positive => accelerates failure):"]
        for nm, b, s in zip(self.parameter_names, self.beta, self._se):
            lines.append(f"   {nm:>10}  :  {b: .6f}  (se {s:.6f})")
        return "\n".join(lines)


class RankAFT_(FitterRepr):
    """
    The rank-based semi-parametric accelerated failure time estimator
    (Gehan or log-rank weights, induced smoothing): the coefficients at
    which the covariates of the units that fail look like those of the
    units still at risk, on the scale of the residual log times. No
    distribution is assumed for the errors. See
    :mod:`surpyval.univariate.regression.rank_aft.rank_aft` for the
    estimating equations and the standard errors.

    Coefficients are reported with the package's AFT sign: a *positive*
    coefficient shortens life. ``RankAFT`` is an instance of this class;
    its ``fit`` returns a :class:`RankAFTModel`.
    """

    #: The ``repr`` (#614)
    fitter_kind = "semi-parametric accelerated failure time fitter"

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        rank_weights: str = "gehan",
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> RankAFTModel:
        """
        Fit the rank-based AFT model.

        Rows with a missing or infinite covariate are dropped, with a
        warning. A constant column (the intercept, which a rank estimator
        cannot see: adding a constant to every residual leaves their
        order alone), or one that is a linear combination of the others,
        is aliased as in ``BuckleyJames``: its coefficient is ``nan``,
        predictions take it as 0, and one warning names it. The rows are
        fitted sorted by every column, so the fit is the same whatever
        order they are given in.

        Parameters
        ----------
        x : array_like
            Observed (positive) times.
        Z : array_like
            Covariate matrix, one row per observation.
        c : array_like, optional
            Censoring flags: 0 observed, 1 right-censored. Left and interval
            censoring are not supported. Defaults to all observed.
        n : array_like, optional
            Counts per row (frequency weights). Defaults to 1.
        rank_weights : {'gehan', 'logrank'}, optional
            The weight of each event in the estimating function: the number
            at risk (``'gehan'``, the default, a convex problem), or
            one (``'logrank'``, more efficient for Weibull-like errors,
            found by iterating weighted Gehan fits from the Gehan
            estimate).
        tol : float, optional
            The fit stops when a round moves every coefficient by less than
            ``tol`` of its standard error. Default 1e-5.
        max_iter : int, optional
            The most rounds. Default 100.

        Returns
        -------
        RankAFTModel
            The fitted model. A warning is raised if the estimate did not
            settle within ``max_iter`` rounds.

        Examples
        --------
        Log-life falls by 0.5 per unit of the covariate; follow-up ends at
        12:

        >>> import numpy as np
        >>> from surpyval import RankAFT
        >>> rng = np.random.default_rng(2)
        >>> Z = rng.normal(size=(100, 1))
        >>> t = np.exp(2.0 - 0.5 * Z[:, 0] + rng.normal(0, 0.5, 100))
        >>> c = (t > 12).astype(int)
        >>> model = RankAFT.fit(np.minimum(t, 12), Z, c=c)
        >>> model.beta.round(3)
        array([0.446])
        >>> model.param_cb().round(3)
        array([[0.346, 0.546]])
        """
        check_option("rank_weights", rank_weights, RANK_WEIGHTS)
        x_a, c_a, n_a, tl_a, Z_a = semi_parametric_inputs(
            x,
            Z,
            c,
            n,
            censoring=(
                "RankAFT supports only observed (c=0) and right-censored "
                "(c=1) data."
            ),
        )
        order = canonical_rows(x_a, c_a, n_a, tl_a, Z_a)
        x_a, c_a, n_a, Z_a = (a[order] for a in (x_a, c_a, n_a, Z_a))
        if np.any(x_a <= 0):
            raise ValueError(
                "RankAFT models log(time); all times must be positive."
            )
        if not np.any((c_a == 0) & (n_a > 0)):
            raise ValueError(
                "RankAFT needs at least one event (c=0); with every "
                "observation censored there are no ranks to compare."
            )
        p = Z_a.shape[1]
        aliased = _aliased(Z_a, n_a)
        kept = np.setdiff1d(np.arange(p), aliased)
        if aliased.size:
            warn_aliased(
                aliased,
                "they are constant (the intercept, which the rank "
                "estimating equations cannot see) or a linear combination "
                "of the other columns",
            )
        Z_k = Z_a[:, kept]
        Y = np.log(x_a)
        delta = (c_a == 0).astype(float)
        cov = np.full((p, p), np.nan)
        smoothing = np.full((kept.size, kept.size), np.nan)
        if kept.size:
            gamma, V, smoothing, n_iter, converged = _fit_gamma(
                Y, delta, Z_k, n_a, rank_weights, tol, max_iter
            )
            cov[np.ix_(kept, kept)] = V
        else:
            gamma, n_iter, converged = np.zeros(0), 0, True
        if not converged:
            warnings.warn(
                "RankAFT did not settle in {} rounds; returning the last "
                "estimate. Raise max_iter, or loosen tol.".format(max_iter)
            )
        resid, resid_surv, _ = residual_km(Y - Z_k @ gamma, delta, n_a)
        return RankAFTModel(
            expand(-gamma, kept, p),
            cov,
            smoothing,
            resid,
            resid_surv,
            rank_weights,
            n_iter,
            converged,
            (Y, delta, Z_a, n_a),
        )

    @takes_unit
    def fit_from_df(
        self,
        df: Any,
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
        rank_weights: str = "gehan",
        tol: float = 1e-5,
        max_iter: int = 100,
    ) -> RankAFTModel:
        """
        Fit a rank-based AFT model from a pandas DataFrame. See :meth:`fit`
        for the estimator; ``Z_cols`` or ``formula`` selects the covariates.

        Parameters
        ----------
        df : DataFrame
            The data.
        x_col : str
            The column of times.
        Z_cols : str or list of str, optional
            The covariate columns. Give either this or ``formula``.
        c_col, n_col : str, optional
            The censoring-flag and count columns.
        formula : str, optional
            A formula (formulaic syntax) for the covariates.
        rank_weights, tol, max_iter : optional
            As for :meth:`fit`.

        Returns
        -------
        RankAFTModel
            The fitted model, which keeps the covariate names (or formula)
            so it predicts from DataFrame rows.
        """
        from surpyval.utils.dataframe import as_pandas as _as_pandas

        df = _as_pandas(df)
        check_columns(df, x_col=x_col, c_col=c_col, n_col=n_col)
        design = FrameDesign(df, Z_cols, formula, drop_missing=True)
        return design.fit(
            self.fit,
            design.column(x_col),
            design.Z,
            c=design.column(c_col),
            n=design.column(n_col),
            rank_weights=rank_weights,
            tol=tol,
            max_iter=max_iter,
        )


RankAFT = RankAFT_()
