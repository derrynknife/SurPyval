"""Fitting the joint model by maximum likelihood (#891).

The steps, in internal units (:class:`~._params.Scaling`):

1. **Starting values.** The longitudinal part is a linear mixed model,
   fitted alone by REML (the population model of
   :mod:`surpyval.degradation.population`), from each unit's own
   least-squares line. With each unit's path at its best linear unbiased
   prediction, the survival part is fitted as a Weibull proportional
   hazards model with the path as a time-varying covariate: the
   two-stage estimate, which is biased (it takes the predicted paths as
   known) but close.
2. **Joint maximum likelihood.** The random effects are integrated out
   by adaptive Gauss-Hermite quadrature with the nodes held fixed, the
   likelihood maximised by BFGS on autograd's gradient, the nodes moved
   to the new modes, and again, until the estimates stop moving.
3. **Checks.** The answer is verified as a maximum (a zero gradient and a
   positive-definite Hessian, principle 13), a hazard coefficient or the
   association running off to infinity is caught by Newton's test
   (``runaway_coefficients``, #392), and the covariance is the inverse of
   the Hessian, carried to the natural parameters by the delta method.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import jacobian, value_and_grad
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import verify_or_polish
from surpyval.univariate.parametric.fitters.runaway import (
    runaway_coefficients,
    search_derivatives,
)
from surpyval.univariate.regression._aliasing import (
    aliased_columns,
    constant_columns,
)

from ._data import JointData
from ._likelihood import (
    INTERVAL,
    LEFT,
    OBSERVED,
    AllNodes,
    Layout,
    Problem,
    Q,
    basis,
    chol_entries,
    fixed_node_neg_ll,
    legendre,
    log_event,
    make_block,
    place_nodes,
    truncation_block,
    unpack,
)
from ._params import (
    Scaling,
    internal_from_natural,
    kept_positions,
    natural_from_internal,
)

#: The most rounds of node placement, and the largest move of the
#: internal parameters between rounds at which they are taken to have
#: stopped moving.
MAX_ROUNDS = 12
ROUND_TOL = 1e-7


class Prepared(NamedTuple):
    """A fit's problem in internal units, and what it was built from."""

    prob: Problem
    scaling: Scaling
    z_kept: npt.NDArray
    w_kept: npt.NDArray
    reading_t: list
    reading_y: list
    n_exact: int


class FitResult(NamedTuple):
    """What :func:`fit_joint` reached."""

    theta: npt.NDArray
    natural: npt.NDArray
    covariance: "npt.NDArray | None"
    neg_ll: float
    maximum: str
    runaway: list
    nodes: AllNodes
    prepared: Prepared
    start: npt.NDArray


# ---------------------------------------------------------------------------
# Aliased columns
# ---------------------------------------------------------------------------
def kept_columns(M: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    """``(kept, aliased)``: the columns of the unit-level covariates ``M``
    that the data determine, and those they do not (constant, or a linear
    combination of earlier ones: the model has an intercept in both of
    its parts)."""
    p = M.shape[1]
    if p == 0:
        return np.arange(0), np.arange(0)
    constant = constant_columns(M)
    centred = M - M.mean(0)
    aliased = aliased_columns(centred.T @ centred, M.shape[0], constant)
    kept = np.setdiff1d(np.arange(p), aliased)
    return kept, np.asarray(aliased, dtype=int)


# ---------------------------------------------------------------------------
# The problem in internal units
# ---------------------------------------------------------------------------
def _spread(M: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
    if M.shape[1] == 0:
        return np.zeros(0), np.ones(0)
    return M.mean(0), M.std(0)


def prepare(
    data: JointData,
    z_kept: npt.NDArray,
    w_kept: npt.NDArray,
    n_quad: int,
    n_legendre: int,
    centre: "float | None" = None,
) -> Prepared:
    """The fit's :class:`Problem` in internal units (the marker centred at
    ``centre``, by default the mean reading)."""
    values = np.concatenate([y for _, y in data.readings])
    y_s = float(values.std())
    y_c = float(values.mean()) if centre is None else float(centre)
    if not y_s > 0:
        raise ValueError(
            "Every marker reading has the same value: the readings carry "
            "no information on the path, so there is nothing to join to "
            "the failures. Fit a regression such as surpyval.WeibullPH."
        )
    Z, W = data.Z[:, z_kept], data.W[:, w_kept]
    z_m, z_s = _spread(Z)
    w_m, w_s = _spread(W)
    t_s = float(np.max(data.xr))
    sc = Scaling(t_s, y_c, y_s, z_m, z_s, w_m, w_s)
    reading_t = [t / t_s for t, _ in data.readings]
    reading_y = [(y - y_c) / y_s for _, y in data.readings]
    Zh, Wh = (Z - z_m) / z_s, (W - w_m) / w_s
    t1 = np.where(data.c == INTERVAL, data.xl, data.xr) / t_s
    units = make_block(reading_t, reading_y, data.c, t1, data.xr / t_s, Zh, Wh)
    truncated = truncation_block(data.tl / t_s, data.tr / t_s, Zh, Wh)
    layout = Layout(int(Z.shape[1]), int(W.shape[1]))
    prob = Problem(units, truncated, layout, legendre(n_legendre), n_quad)
    n_exact = int(np.sum(data.c == OBSERVED))
    return Prepared(prob, sc, z_kept, w_kept, reading_t, reading_y, n_exact)


def natural_neg_ll(neg_ll: float, prepared: Prepared) -> float:
    """The negative log-likelihood in the data's units: the internal one
    plus the Jacobians of the time scale (each exact failure time is a
    density) and of the marker scale (each reading is)."""
    sc = prepared.scaling
    n_readings = sum(len(t) for t in prepared.reading_t)
    return float(
        neg_ll
        + prepared.n_exact * np.log(sc.t_s)
        + n_readings * np.log(sc.y_s)
    )


# ---------------------------------------------------------------------------
# Starting values
# ---------------------------------------------------------------------------
def _unit_lines(prepared: Prepared) -> tuple[npt.NDArray, float]:
    """Each unit's own least-squares line (units with readings at two or
    more ages): the moments of the lines, and the pooled noise."""
    coefs, resid, df = [], 0.0, 0
    for t, y in zip(prepared.reading_t, prepared.reading_y):
        if np.unique(t).size < Q:
            continue
        X = basis(t)
        cf, *_ = np.linalg.lstsq(X, y, rcond=None)
        coefs.append(cf)
        resid += float(np.sum((y - X @ cf) ** 2))
        df += t.size - Q
    if len(coefs) < 2:
        raise ValueError(
            "The joint model's path has a random intercept and slope for "
            "each unit, which needs at least two units with readings at two "
            f"or more different ages; {len(coefs)} unit(s) have them. Take "
            "more readings, or model the marker another way."
        )
    values = np.concatenate(prepared.reading_y)
    sigma2 = resid / df if df > 0 and resid > 0 else 0.1 * values.var()
    return np.asarray(coefs), float(sigma2)


def _mixed_model(prepared: Prepared) -> tuple[npt.NDArray, Any, float]:
    """The longitudinal part alone, by REML: ``(path, D, sigma2)``."""
    from surpyval.degradation.population import reml_estimate

    coefs, sigma2 = _unit_lines(prepared)
    W1 = prepared.prob.units.W1
    y_list, x_list, a_list = [], [], []
    for u, (t, y) in enumerate(zip(prepared.reading_t, prepared.reading_y)):
        if t.size == 0:
            continue
        X = basis(t)
        y_list.append(y)
        x_list.append(X)
        a_list.append(np.hstack([X * W1[u, j] for j in range(W1.shape[1])]))
    cov = np.cov(coefs.T) if coefs.shape[0] > 1 else np.eye(Q)
    try:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            path, D, s2, _ = reml_estimate(
                y_list, x_list, cov, sigma2, a_mat_list=a_list
            )
    except (np.linalg.LinAlgError, ValueError):
        path = np.zeros(Q * W1.shape[1])
        path[:Q] = coefs.mean(0)
        D, s2 = cov, sigma2
    D = 0.5 * (D + D.T)
    floor = 1e-4 * max(float(np.max(np.diag(D))), s2, 1e-8)
    vals, vecs = np.linalg.eigh(D)
    D = (vecs * np.maximum(vals, floor)) @ vecs.T
    return np.asarray(path, dtype=float), D, float(s2)


def _blups(
    prepared: Prepared, path: npt.NDArray, D: npt.NDArray, sigma2: float
) -> npt.NDArray:
    blk = prepared.prob.units
    P = path.reshape(blk.W1.shape[1], Q).T
    fixed = blk.W1 @ P.T
    info = (
        np.einsum("njq,njr->nqr", blk.BR, blk.BR) / sigma2
        + np.linalg.inv(D)[None]
    )
    r = (blk.Y - np.einsum("njq,nq->nj", blk.BR, fixed)) * blk.M
    rhs = np.einsum("nj,njq->nq", r, blk.BR) / sigma2
    return np.linalg.solve(info, rhs[..., None])[..., 0]


def two_stage_start(prepared: Prepared) -> npt.NDArray:
    """The starting point: the mixed model by REML, then the survival
    part with each unit's path at its prediction (the two-stage
    estimate)."""
    prob = prepared.prob
    layout, blk = prob.layout, prob.units
    path, D, sigma2 = _mixed_model(prepared)
    b_hat = _blups(prepared, path, D, sigma2)
    d = max(int(np.sum(blk.kind != 1)), 1)
    exposure = float(np.sum(blk.t2))
    theta = np.zeros(layout.size)
    theta[0] = np.log(exposure / d)
    theta[layout.path()] = path
    theta[layout.lsig()] = 0.5 * np.log(sigma2)
    theta[layout.chol()] = chol_entries(D)
    surv = layout.survival()
    fixed = blk.W1 @ path.reshape(blk.W1.shape[1], Q)
    U = (fixed + b_hat)[:, None, :]

    def neg(s: Any) -> Any:
        th = anp.concatenate([s, anp.asarray(theta[surv.size :])])
        return -anp.sum(log_event(unpack(th, layout), U, blk, prob.gl))

    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = minimize(
            value_and_grad(neg), theta[surv], jac=True, method="BFGS"
        )
    if np.all(np.isfinite(res.x)) and np.isfinite(res.fun):
        theta[surv] = res.x
    return theta


# ---------------------------------------------------------------------------
# Joint maximum likelihood
# ---------------------------------------------------------------------------
def _search(
    theta: npt.NDArray, prob: Problem, nodes: AllNodes, max_iter: int
) -> Any:
    """BFGS on the fixed-node likelihood (per unit), started from the
    inverse of its Hessian at ``theta`` where that is positive definite,
    so that the first steps are already in the right units."""
    n = float(prob.units.n)

    def f(th: Any) -> Any:
        return fixed_node_neg_ll(th, prob, nodes) / n

    vg = value_and_grad(f)

    def guarded(th: npt.NDArray) -> tuple[float, npt.NDArray]:
        value, grad = vg(th)
        grad = np.asarray(grad, dtype=float)
        if not (np.isfinite(value) and np.all(np.isfinite(grad))):
            return np.inf, np.zeros_like(grad)
        return float(value), grad

    options: dict = {"gtol": 1e-9, "maxiter": max_iter}
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        derivs = search_derivatives(f, theta)
        if derivs is not None:
            H = 0.5 * (derivs[0] + derivs[0].T)
            try:
                inv = np.linalg.inv(H)
                inv = 0.5 * (inv + inv.T)
                np.linalg.cholesky(inv)
                options["hess_inv0"] = inv
            except np.linalg.LinAlgError:
                d = np.abs(np.diag(H))
                if np.all(np.isfinite(d)) and np.all(d > 0):
                    options["hess_inv0"] = np.diag(1.0 / d)
        return minimize(
            guarded, theta, jac=True, method="BFGS", options=options
        )


def fit_joint(
    data: JointData,
    n_quad: int = 7,
    n_legendre: int = 20,
    init: "npt.ArrayLike | None" = None,
    max_iter: int = 1000,
    centre: "float | None" = None,
) -> FitResult:
    """Fit the joint model to ``data`` by maximum likelihood."""
    z_kept, _ = kept_columns(data.Z)
    w_kept, _ = kept_columns(data.W)
    prepared = prepare(data, z_kept, w_kept, n_quad, n_legendre, centre)
    prob = prepared.prob
    if init is None:
        theta = two_stage_start(prepared)
    else:
        # the estimated parameters of a full init (aliased ones left out)
        values = np.asarray(init, dtype=float)
        kept = kept_positions(data.Z.shape[1], data.W.shape[1], z_kept, w_kept)
        if values.size != kept.size:
            values = values[kept]
        theta = internal_from_natural(values, prob.layout, prepared.scaling)
    start = theta.copy()
    nodes = place_nodes(theta, prob)
    stalled = False
    for _ in range(MAX_ROUNDS):
        res = _search(theta, prob, nodes, max_iter)
        if not (np.all(np.isfinite(res.x)) and np.isfinite(res.fun)):
            stalled = True
            break
        move = float(np.max(np.abs(res.x - theta)))
        theta = np.asarray(res.x, dtype=float)
        nodes = place_nodes(theta, prob, nodes)
        if res.nit >= max_iter:
            stalled = True
            break
        if move < ROUND_TOL:
            break
    return _finish(theta, start, nodes, prepared, stalled)


def _finish(
    theta: npt.NDArray,
    start: npt.NDArray,
    nodes: AllNodes,
    prepared: Prepared,
    stalled: bool,
) -> FitResult:
    prob = prepared.prob

    def neg_ll(th: Any) -> Any:
        return fixed_node_neg_ll(th, prob, nodes)

    with np.errstate(all="ignore"):
        value = float(neg_ll(theta))

    class _Res(NamedTuple):
        x: npt.NDArray
        fun: float

    res: Any = _Res(theta, value)
    verified = False
    if not stalled:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res, verified = verify_or_polish(neg_ll, res, float(prob.units.n))
        theta = np.asarray(res.x, dtype=float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        derivs = search_derivatives(neg_ll, theta)
    layout = prob.layout
    coefs = list(range(2, 3 + layout.p))
    runaway: list = []
    if derivs is not None:
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            runaway = runaway_coefficients(
                neg_ll, theta, coefs, start=start, derivatives=derivs
            )
    if runaway:
        maximum = "no finite maximum"
    else:
        maximum = "verified" if verified else "unverified"
    with np.errstate(all="ignore"):
        value = float(neg_ll(theta))
    cov = _covariance(theta, derivs, prepared)
    natural = np.asarray(
        natural_from_internal(theta, layout, prepared.scaling), dtype=float
    )
    return FitResult(
        theta,
        natural,
        cov,
        natural_neg_ll(value, prepared),
        maximum,
        list(runaway),
        nodes,
        prepared,
        start,
    )


def _covariance(
    theta: npt.NDArray,
    derivs: "tuple[npt.NDArray, npt.NDArray] | None",
    prepared: Prepared,
) -> "npt.NDArray | None":
    """The covariance of the natural parameters: the inverse Hessian of
    the negative log-likelihood in the search space, carried by the
    Jacobian of the map to the natural parameters."""
    if derivs is None:
        return None
    H = 0.5 * (derivs[0] + derivs[0].T)
    if not np.all(np.isfinite(H)):
        return None
    try:
        np.linalg.cholesky(H)
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return None
    layout, sc = prepared.prob.layout, prepared.scaling
    J = np.asarray(
        jacobian(lambda th: natural_from_internal(th, layout, sc))(theta),
        dtype=float,
    )
    out = J @ cov @ J.T
    return 0.5 * (out + out.T)


def observed_failures(data: JointData) -> int:
    """The number of units whose failure was observed (exactly, or in a
    left or interval censored window): the sample size of ``bic``."""
    return int(np.sum(np.isin(data.c, (OBSERVED, LEFT, INTERVAL))))
