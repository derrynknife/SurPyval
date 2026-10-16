"""The likelihood of the joint model and its numerics (#891).

The model, for unit ``i`` with readings ``y_ij`` at ages ``t_ij``, is

.. code-block:: text

    y_ij = m_i(t_ij) + e_ij,                 e_ij ~ N(0, sigma^2)
    m_i(t) = basis(t)' (P [1, w_i] + b_i),   b_i ~ N(0, D)
    h_i(t | b_i) = h_0(t) exp(gamma' z_i + a (m_i(t) - c))

with ``basis(t) = [1, t]`` (a straight path with a random intercept and
slope), ``w_i`` the path covariates, ``z_i`` the hazard covariates, ``c``
the marker centre and ``h_0`` the Weibull hazard ``k t^(k-1) alpha^-k``.
The readings and the event share ``b_i``, and the likelihood of a unit is

.. code-block:: text

    L_i = integral p(y_i | b) p(event_i | b) p(b) db

where the event's part is ``h S`` (observed), ``S`` (right censored),
``1 - S`` (left censored) or ``S(l) - S(r)`` (interval censored). A unit
under left or right truncation is divided by the probability of its
window, ``integral P(tl < T <= tr | b) p(b) db``: the unit is in the data
because it survived to ``tl``, so ``p(b)`` stays the distribution of the
random effects at age 0.

Everything here works in the *internal* units of :class:`Scaling`: the
times divided by a time scale, the readings centred and scaled, the
covariates standardised, so that every parameter is of order one. The
search vector ``theta`` is laid out by :class:`Layout`.

Numerics
--------
- The integral over ``b`` is adaptive Gauss-Hermite quadrature: each
  unit's integrand is centred at its mode ``b_hat_i`` and scaled by the
  Cholesky factor of the inverse of its negative Hessian there
  (:func:`unit_modes`, a vectorised damped Newton iteration with the
  analytic derivatives in ``b``). As in R's ``JM``, the nodes are held
  fixed while the parameters are searched, then moved to the new modes,
  until the estimates stop moving; with the nodes fixed the
  log-likelihood is a smooth function of ``theta`` that autograd
  differentiates (:func:`fixed_node_neg_ll`).
- The cumulative hazard has no closed form (the hazard is a Weibull
  times the exponential of a linear function of time), so it is a
  Gauss-Legendre sum in ``u = s^k``, which removes the singularity of
  ``s^(k-1)`` at 0, with the nodes graded towards 0 (:func:`legendre`).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd.scipy.special import logsumexp as alogsumexp
from scipy.special import logsumexp

LOG2PI = float(np.log(2.0 * np.pi))

#: The kinds of a unit's event, as the censoring flag ``c``.
OBSERVED, RIGHT, LEFT, INTERVAL = 0, 1, -1, 2


# ---------------------------------------------------------------------------
# The path: basis functions of age
# ---------------------------------------------------------------------------
def basis(t: Any) -> Any:
    """The path's basis at ages ``t`` (any shape): ``[1, t]`` on a last
    axis of length ``Q``. The only path of v1 is the straight line; a
    spline path (later) is a longer basis, and nothing else changes."""
    return anp.stack([anp.ones(anp.shape(t)), t], axis=-1)


#: The number of basis functions (and random effects) of the path.
Q = 2


def path_values(bas: Any, U: Any) -> Any:
    """The paths of coefficients ``U`` (``(n, K, Q)``) at ages whose basis
    is ``bas`` (``(n, X, Q)``): ``(n, K, X)``. A sum over the few basis
    functions, which is much faster than ``einsum`` and its derivative."""
    out = bas[:, None, :, 0] * U[:, :, None, 0]
    for q in range(1, Q):
        out = out + bas[:, None, :, q] * U[:, :, None, q]
    return out


def basis_time_scale(t_s: float) -> npt.NDArray:
    """``S`` with ``basis(t / t_s) = S basis(t)``, which carries a path's
    coefficients between time units."""
    return np.diag([1.0, 1.0 / t_s])


# ---------------------------------------------------------------------------
# The search vector
# ---------------------------------------------------------------------------
class Layout(NamedTuple):
    """Where each part of the search vector ``theta`` is: ``la`` (the log
    Weibull scale), ``lk`` (the log shape), ``gamma`` (the hazard
    coefficients, ``p`` of them), ``assoc`` (the association), ``path``
    (the path coefficients, ``Q * (1 + pw)`` of them: the matrix ``P``
    column by column), ``lsig`` (the log noise) and ``chol`` (the
    Cholesky factor of ``D``: its log diagonal, then the entries below
    it, row by row)."""

    p: int
    pw: int

    @property
    def n_path(self) -> int:
        return Q * (1 + self.pw)

    @property
    def n_chol(self) -> int:
        return Q * (Q + 1) // 2

    @property
    def size(self) -> int:
        return 4 + self.p + self.n_path + self.n_chol

    def gamma(self) -> slice:
        return slice(2, 2 + self.p)

    def assoc(self) -> int:
        return 2 + self.p

    def path(self) -> slice:
        start = 3 + self.p
        return slice(start, start + self.n_path)

    def lsig(self) -> int:
        return 3 + self.p + self.n_path

    def chol(self) -> slice:
        start = 4 + self.p + self.n_path
        return slice(start, start + self.n_chol)

    def survival(self) -> npt.NDArray:
        """The positions of the survival parameters (scale, shape, hazard
        coefficients and association)."""
        return np.arange(0, 3 + self.p)


_TRIL = np.tril_indices(Q, -1)
# Constant tensors that build the Cholesky factor from its entries, so
# that autograd differentiates through it.
_DIAG_BASIS = np.stack([np.outer(e, e) for e in np.eye(Q)])
_LOW_BASIS = (
    np.stack(
        [
            np.outer(np.eye(Q)[i], np.eye(Q)[j])
            for i, j in zip(_TRIL[0], _TRIL[1])
        ]
    )
    if Q > 1
    else np.zeros((0, Q, Q))
)


def chol_factor(z: Any) -> Any:
    """The lower-triangular Cholesky factor of ``D`` from its entries
    ``z`` (log diagonal, then the lower entries)."""
    L = anp.tensordot(anp.exp(z[:Q]), _DIAG_BASIS, axes=1)
    if Q > 1:
        L = L + anp.tensordot(z[Q:], _LOW_BASIS, axes=1)
    return L


def _forward_solve(z: Any, B: Any) -> Any:
    """``L^-1 b`` for each ``b`` on the last axis of ``B``, ``L`` the
    Cholesky factor of entries ``z``, by forward substitution (finite
    wherever ``L`` is, which an inverse is not)."""
    diag = anp.exp(z[:Q])
    rows: list = []
    k = Q
    for i in range(Q):
        acc = B[..., i]
        for j in range(i):
            acc = acc - z[k + j] * rows[j]
        k += i
        rows.append(acc / diag[i])
    return anp.stack(rows, axis=-1)


def chol_entries(D: npt.NDArray) -> npt.NDArray:
    """The entries of :func:`chol_factor` for a covariance ``D``."""
    L = np.linalg.cholesky(D)
    return np.r_[np.log(np.diag(L)), L[_TRIL]]


class Parts(NamedTuple):
    """The search vector unpacked (see :class:`Layout`)."""

    la: Any
    lk: Any
    gamma: Any
    assoc: Any
    path: Any  # (Q, 1 + pw)
    lsig: Any
    chol: Any


def unpack(theta: Any, layout: Layout) -> Parts:
    pw1 = 1 + layout.pw
    path = anp.reshape(theta[layout.path()], (pw1, Q)).T
    return Parts(
        theta[0],
        theta[1],
        theta[layout.gamma()],
        theta[layout.assoc()],
        path,
        theta[layout.lsig()],
        theta[layout.chol()],
    )


# ---------------------------------------------------------------------------
# The data of a set of units, in internal units
# ---------------------------------------------------------------------------
class Block(NamedTuple):
    """The units of one likelihood term, as padded arrays.

    ``Y``, ``BR`` and ``M`` are the readings (``(n, J)``), their basis
    (``(n, J, Q)``) and which entries are readings; ``nobs`` the number
    of readings of each unit. ``kind`` is each unit's event (the
    censoring flag), ``t1`` its time (the lower end of an interval) and
    ``t2`` the upper end of an interval (``t1`` otherwise). ``Zh`` are
    the hazard covariates and ``W1`` the path design ``[1, w]``.
    ``yy``, ``Xy`` and ``XX`` are each unit's sums ``y'y``, ``X'y`` and
    ``X'X`` (``X`` its readings' basis), by which the readings' sum of
    squares at any path is a quadratic form, without a pass over them.
    """

    Y: npt.NDArray
    BR: npt.NDArray
    M: npt.NDArray
    nobs: npt.NDArray
    kind: npt.NDArray
    t1: npt.NDArray
    t2: npt.NDArray
    Zh: npt.NDArray
    W1: npt.NDArray
    yy: npt.NDArray
    Xy: npt.NDArray
    XX: npt.NDArray

    @property
    def n(self) -> int:
        return int(self.kind.size)


def make_block(
    reading_t: list,
    reading_y: list,
    kind: npt.ArrayLike,
    t1: npt.ArrayLike,
    t2: npt.ArrayLike,
    Zh: npt.ArrayLike,
    W: npt.ArrayLike,
) -> Block:
    """A :class:`Block` from each unit's readings (``reading_t[i]``,
    ``reading_y[i]``) and its event."""
    n = len(reading_t)
    J = max([len(t) for t in reading_t] + [1])
    Y = np.zeros((n, J))
    T = np.zeros((n, J))
    M = np.zeros((n, J))
    for u, (t, y) in enumerate(zip(reading_t, reading_y)):
        k = len(t)
        T[u, :k] = t
        Y[u, :k] = y
        M[u, :k] = 1.0
    Zh = np.asarray(Zh, dtype=float).reshape(n, -1)
    W = np.asarray(W, dtype=float).reshape(n, -1)
    BR = basis(T) * M[..., None]
    return Block(
        Y=Y,
        BR=BR,
        M=M,
        nobs=M.sum(1),
        kind=np.asarray(kind, dtype=int),
        t1=np.asarray(t1, dtype=float),
        t2=np.asarray(t2, dtype=float),
        Zh=Zh,
        W1=np.hstack([np.ones((n, 1)), W]),
        yy=np.sum(Y**2 * M, axis=1),
        Xy=np.einsum("nj,njq->nq", Y * M, BR),
        XX=np.einsum("njq,njr->nqr", BR, BR),
    )


def truncation_block(
    tl: npt.NDArray, tr: npt.NDArray, Zh: npt.NDArray, W: npt.NDArray
) -> "tuple[Block, npt.NDArray] | None":
    """The denominators of the truncated units, as a block of units with
    no readings whose event is their window: right censored at ``tl``
    (``P(T > tl)``), interval censored on ``(tl, tr]``, or left censored
    at ``tr`` (``P(T <= tr)``); and which units they are. ``None`` where
    no unit is truncated."""
    left = tl > 0
    right = np.isfinite(tr)
    which = np.flatnonzero(left | right)
    if which.size == 0:
        return None
    lo, hi = tl[which], tr[which]
    kind = np.where(
        left[which] & right[which],
        INTERVAL,
        np.where(left[which], RIGHT, LEFT),
    )
    t1 = np.where(kind == LEFT, hi, lo)
    t2 = np.where(kind == INTERVAL, hi, t1)
    empty = [np.zeros(0)] * which.size
    block = make_block(empty, empty, kind, t1, t2, Zh[which], W[which])
    return block, which


# ---------------------------------------------------------------------------
# Gauss-Legendre and Gauss-Hermite rules
# ---------------------------------------------------------------------------
#: The power by which the Gauss-Legendre nodes are graded towards 0.
GRADE = 4


def legendre(n: int) -> tuple[npt.NDArray, npt.NDArray]:
    """An ``n``-point rule on (0, 1) for the cumulative hazard, in the
    fraction ``r`` of ``u = s^k`` covered: Gauss-Legendre in ``v`` with
    ``r = v^4``, which grades the nodes towards 0, where the integrand
    behaves like a power of ``u`` (``u^(1/k)`` in the path). Over the
    shapes 0.5 to 6 and paths whose hazard changes by ``e^-5`` to ``e^3``
    over the interval, 15 points are accurate to 1e-7 of the cumulative
    hazard and 20 to 5e-10, where the plain rule (``r = v``) was 1e-2
    off."""
    z, w = np.polynomial.legendre.leggauss(n)
    v = 0.5 * (z + 1.0)
    return v**GRADE, GRADE * v ** (GRADE - 1) * 0.5 * w


def hermite_grid(n: int) -> tuple[npt.NDArray, npt.NDArray]:
    """The ``Q``-dimensional product Gauss-Hermite rule for the weight
    ``exp(-z'z)``: nodes ``(n**Q, Q)`` and log weights."""
    z, w = np.polynomial.hermite.hermgauss(n)
    grids = np.meshgrid(*([z] * Q), indexing="ij")
    weights = np.meshgrid(*([w] * Q), indexing="ij")
    Z = np.stack([g.ravel() for g in grids], axis=-1)
    logw = np.sum([np.log(g.ravel()) for g in weights], axis=0)
    return Z, logw


# ---------------------------------------------------------------------------
# The cumulative hazard
# ---------------------------------------------------------------------------
def _log_cumhaz(
    k: Any,
    lin: Any,
    assoc: Any,
    U: Any,
    lo: npt.NDArray,
    hi: npt.NDArray,
    gl: tuple[npt.NDArray, npt.NDArray],
) -> Any:
    """``log H(lo -> hi | b)``, ``(n, K)``: the log of the integral of
    ``k s^(k-1) exp(lin + assoc m(s))`` over ``(lo, hi]``, ``m(s) =
    basis(s)' U`` with ``U`` the path coefficients ``(n, K, Q)``. ``lin``
    is ``(n,)`` and ``lo < hi`` (``lo`` may be 0)."""
    v, w = gl
    lo_pos = lo > 0
    lo_safe = np.where(lo_pos, lo, 1.0)
    lo_k = anp.where(lo_pos, lo_safe**k, 0.0)
    hi_k = hi**k
    span = hi_k - lo_k  # (n,)
    s = (lo_k[:, None] + span[:, None] * v[None, :]) ** (1.0 / k)  # (n, G)
    m = path_values(basis(s), U)
    log_terms = assoc * m + anp.log(w)[None, None, :]
    return lin[:, None] + anp.log(span)[:, None] + alogsumexp(log_terms, -1)


def _log1mexp(x: Any) -> Any:
    """``log(1 - exp(-x))`` for ``x > 0``."""
    return anp.log(-anp.expm1(-x))


def log_event(
    parts: Parts,
    U: Any,
    blk: Block,
    gl: tuple[npt.NDArray, npt.NDArray],
) -> Any:
    """The log of each unit's event probability (or density) given its
    path coefficients ``U`` (``(n, K, Q)``): ``(n, K)``."""
    k = anp.exp(parts.lk)
    lin = -k * parts.la + anp.dot(blk.Zh, parts.gamma)
    kind = blk.kind
    t1_pos = blk.t1 > 0
    # H(0 -> t1), 0 where t1 is 0 (an interval from 0)
    t1_safe = np.where(t1_pos, blk.t1, 1.0)
    log_H1 = _log_cumhaz(k, lin, parts.assoc, U, np.zeros(blk.n), t1_safe, gl)
    H1 = anp.where(t1_pos[:, None], anp.exp(log_H1), 0.0)
    out = -H1
    observed = (kind == OBSERVED)[:, None]
    if np.any(observed):
        m1 = path_values(basis(t1_safe)[:, None, :], U)[..., 0]
        log_h = (
            parts.lk
            + (k - 1.0) * anp.log(t1_safe)[:, None]
            + lin[:, None]
            + parts.assoc * m1
        )
        out = anp.where(observed, out + log_h, out)
    left = (kind == LEFT)[:, None]
    if np.any(left):
        safe = anp.where(left, H1, 1.0)
        out = anp.where(left, _log1mexp(safe), out)
    interval = kind == INTERVAL
    if np.any(interval):
        t2 = np.where(interval, blk.t2, blk.t1 + 1.0)
        lo = np.where(interval, blk.t1, 0.0)
        hi = np.where(interval, t2, 1.0)
        log_gap = _log_cumhaz(k, lin, parts.assoc, U, lo, hi, gl)
        gap = anp.where(interval[:, None], anp.exp(log_gap), 1.0)
        out = anp.where(interval[:, None], -H1 + _log1mexp(gap), out)
    return out


def _sum_of_squares(U: Any, blk: Block) -> Any:
    """Each unit's sum of squared residuals of its readings about the
    paths of coefficients ``U`` (``(n, K, Q)``): ``y'y - 2 U'X'y +
    U'X'X U``, ``(n, K)``."""
    out = anp.reshape(blk.yy, (-1, 1)) + 0.0 * U[..., 0]
    for q in range(Q):
        out = out - 2.0 * U[..., q] * blk.Xy[:, None, q]
        for r in range(Q):
            out = out + U[..., q] * U[..., r] * blk.XX[:, None, q, r]
    return out


def log_integrand(
    theta: Any,
    B: Any,
    blk: Block,
    layout: Layout,
    gl: tuple[npt.NDArray, npt.NDArray],
) -> Any:
    """``log p(y_i | b) + log p(event_i | b) + log p(b)`` at the random
    effects ``B`` (``(n, K, Q)``, ``K`` nodes per unit): ``(n, K)``."""
    parts = unpack(theta, layout)
    fixed = anp.dot(blk.W1, parts.path.T)  # (n, Q)
    U = fixed[:, None, :] + B
    rss = _sum_of_squares(U, blk)
    sig2 = anp.exp(2.0 * parts.lsig)
    ll_long = (
        -0.5 * blk.nobs[:, None] * (LOG2PI + 2.0 * parts.lsig)
        - 0.5 * rss / sig2
    )
    u = _forward_solve(parts.chol, B)
    ll_b = (
        -0.5 * Q * LOG2PI - anp.sum(parts.chol[:Q]) - 0.5 * anp.sum(u**2, -1)
    )
    return ll_long + ll_b + log_event(parts, U, blk, gl)


# ---------------------------------------------------------------------------
# Each unit's mode in b (numpy, analytic derivatives)
# ---------------------------------------------------------------------------
def _cumhaz_derivatives(
    k: float,
    lin: npt.NDArray,
    assoc: float,
    U: npt.NDArray,
    lo: npt.NDArray,
    hi: npt.NDArray,
    gl: tuple[npt.NDArray, npt.NDArray],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """``H(lo -> hi | b)`` and its gradient and Hessian in ``b`` (``U``
    is ``(n, Q)``): ``(n,)``, ``(n, Q)`` and ``(n, Q, Q)``."""
    v, w = gl
    lo_k = np.where(lo > 0, np.where(lo > 0, lo, 1.0) ** k, 0.0)
    span = hi**k - lo_k
    s = (lo_k[:, None] + span[:, None] * v[None, :]) ** (1.0 / k)
    bs = basis(s)  # (n, G, Q)
    m = np.einsum("ngq,nq->ng", bs, U)
    e = np.exp(lin[:, None] + assoc * m) * (span[:, None] * w[None, :])
    H = e.sum(1)
    grad = assoc * np.einsum("ng,ngq->nq", e, bs)
    hess = assoc**2 * np.einsum("ng,ngq,ngr->nqr", e, bs, bs)
    return H, grad, hess


def _event_derivatives(
    parts: Parts, U: npt.NDArray, blk: Block, gl: Any
) -> tuple[npt.NDArray, npt.NDArray]:
    """The gradient and Hessian in ``b`` of :func:`log_event` at one point
    per unit (``U`` is ``(n, Q)``)."""
    k = float(np.exp(parts.lk))
    a = float(parts.assoc)
    lin = -k * float(parts.la) + blk.Zh @ np.asarray(parts.gamma, float)
    kind = blk.kind
    t1_pos = blk.t1 > 0
    t1_safe = np.where(t1_pos, blk.t1, 1.0)
    H1, g1, h1 = _cumhaz_derivatives(
        k, lin, a, U, np.zeros(blk.n), t1_safe, gl
    )
    H1 = np.where(t1_pos, H1, 0.0)
    g1 = np.where(t1_pos[:, None], g1, 0.0)
    h1 = np.where(t1_pos[:, None, None], h1, 0.0)
    grad = -g1
    hess = -h1
    obs = kind == OBSERVED
    grad = grad + np.where(obs[:, None], a * basis(t1_safe), 0.0)
    left = kind == LEFT
    if np.any(left):
        with np.errstate(over="ignore", divide="ignore"):
            rho = 1.0 / np.expm1(np.where(left, H1, 1.0))
        g_l = rho[:, None] * g1
        h_l = rho[:, None, None] * h1 - (rho * (1 + rho))[
            :, None, None
        ] * np.einsum("nq,nr->nqr", g1, g1)
        grad = np.where(left[:, None], g_l, grad)
        hess = np.where(left[:, None, None], h_l, hess)
    interval = kind == INTERVAL
    if np.any(interval):
        lo = np.where(interval, blk.t1, 0.0)
        hi = np.where(interval, blk.t2, 1.0)
        G, gg, hg = _cumhaz_derivatives(k, lin, a, U, lo, hi, gl)
        with np.errstate(over="ignore", divide="ignore"):
            rho = 1.0 / np.expm1(np.where(interval, G, 1.0))
        g_i = -g1 + rho[:, None] * gg
        h_i = (
            -h1
            + rho[:, None, None] * hg
            - (rho * (1 + rho))[:, None, None]
            * np.einsum("nq,nr->nqr", gg, gg)
        )
        grad = np.where(interval[:, None], g_i, grad)
        hess = np.where(interval[:, None, None], h_i, hess)
    return grad, hess


class Modes(NamedTuple):
    """Each unit's mode ``b`` (``(n, Q)``) and the Cholesky factor ``C``
    (``(n, Q, Q)``) of the inverse of the negative Hessian there."""

    b: npt.NDArray
    C: npt.NDArray


def _value(
    theta: npt.NDArray, b: npt.NDArray, blk: Block, layout: Layout, gl: Any
) -> npt.NDArray:
    with np.errstate(all="ignore"):
        return np.asarray(
            log_integrand(theta, b[:, None, :], blk, layout, gl)[:, 0]
        )


def _gradient_and_information(
    theta: npt.NDArray,
    b: npt.NDArray,
    blk: Block,
    layout: Layout,
    gl: Any,
    gauss: npt.NDArray,
    D_inv: npt.NDArray,
) -> tuple[npt.NDArray, npt.NDArray]:
    """The gradient of the log integrand in ``b`` and its negative
    Hessian (the information)."""
    parts = unpack(theta, layout)
    sig2 = float(np.exp(2.0 * parts.lsig))
    fixed = blk.W1 @ np.asarray(parts.path, float).T
    U = fixed + b
    r = (blk.Y - np.einsum("njq,nq->nj", blk.BR, U)) * blk.M
    grad = np.einsum("nj,njq->nq", r, blk.BR) / sig2 - b @ D_inv
    with np.errstate(all="ignore"):
        g_e, h_e = _event_derivatives(parts, U, blk, gl)
    return grad + g_e, gauss - h_e


def unit_modes(
    theta: npt.NDArray,
    blk: Block,
    layout: Layout,
    gl: Any,
    b_init: "npt.NDArray | None" = None,
    max_iter: int = 100,
) -> Modes:
    """Each unit's mode of its log integrand in ``b``, by damped Newton
    steps (with the information of the Gaussian parts alone where the
    event's makes it indefinite, and step halving so the integrand never
    falls), and the Cholesky factor of the inverse information there."""
    parts = unpack(theta, layout)
    sig2 = float(np.exp(2.0 * parts.lsig))
    L = np.asarray(chol_factor(np.asarray(parts.chol, float)))
    D_inv = np.linalg.inv(L @ L.T)
    # The information of the readings and of the prior: constant in b
    gauss = np.einsum("njq,njr->nqr", blk.BR, blk.BR) / sig2 + D_inv
    b = np.zeros((blk.n, Q)) if b_init is None else np.array(b_init)
    value = _value(theta, b, blk, layout, gl)
    for _ in range(max_iter):
        grad, info = _gradient_and_information(
            theta, b, blk, layout, gl, gauss, D_inv
        )
        ok = np.all(np.isfinite(info), axis=(1, 2))
        ok &= (
            np.linalg.eigvalsh(np.where(ok[:, None, None], info, gauss))[:, 0]
            > 0
        )
        info = np.where(ok[:, None, None], info, gauss)
        step = np.linalg.solve(info, grad[..., None])[..., 0]
        step = np.where(np.isfinite(step), step, 0.0)
        t = np.ones(blk.n)
        for _ in range(40):
            trial = _value(theta, b + t[:, None] * step, blk, layout, gl)
            worse = ~(trial >= value - 1e-12 * np.abs(value))
            if not worse.any():
                break
            t = np.where(worse, 0.5 * t, t)
        else:
            t = np.where(worse, 0.0, t)
            trial = np.where(worse, value, trial)
        move = t[:, None] * step
        b = b + move
        value = trial
        if np.max(np.abs(move), initial=0.0) < 1e-10:
            break
    _, info = _gradient_and_information(
        theta, b, blk, layout, gl, gauss, D_inv
    )
    return Modes(b, _inverse_root(info, gauss))


def _inverse_root(info: npt.NDArray, fallback: npt.NDArray) -> npt.NDArray:
    """The Cholesky factor of the inverse of each information matrix,
    with its eigenvalues floored (and the Gaussian parts' information
    where it is not finite) so it always exists."""
    info = np.where(
        np.all(np.isfinite(info), axis=(1, 2))[:, None, None], info, fallback
    )
    info = 0.5 * (info + np.swapaxes(info, 1, 2))
    vals, vecs = np.linalg.eigh(info)
    floor = 1e-8 * np.max(np.abs(vals), axis=1, keepdims=True) + 1e-300
    vals = np.maximum(vals, floor)
    inv = np.einsum("nij,nj,nkj->nik", vecs, 1.0 / vals, vecs)
    return np.linalg.cholesky(0.5 * (inv + np.swapaxes(inv, 1, 2)))


# ---------------------------------------------------------------------------
# Adaptive Gauss-Hermite quadrature
# ---------------------------------------------------------------------------
class Nodes(NamedTuple):
    """Fixed quadrature nodes ``B`` (``(n, K, Q)``) for a block, their log
    weights (``(K,)``, with the ``exp(z'z)`` factor) and each unit's
    constant ``log(2^(Q/2) |C_i|)``."""

    B: npt.NDArray
    logw: npt.NDArray
    logdet: npt.NDArray


def adaptive_nodes(modes: Modes, n_points: int) -> Nodes:
    """The adaptive Gauss-Hermite nodes centred at the modes."""
    Z, logw = hermite_grid(n_points)
    B = modes.b[:, None, :] + np.sqrt(2.0) * np.einsum(
        "nij,kj->nki", modes.C, Z
    )
    logdet = 0.5 * Q * np.log(2.0) + np.sum(
        np.log(np.diagonal(modes.C, axis1=1, axis2=2)), axis=1
    )
    return Nodes(B, logw + np.sum(Z**2, 1), logdet)


def log_marginal(
    theta: Any, blk: Block, nodes: Nodes, layout: Layout, gl: Any
) -> Any:
    """Each unit's log likelihood, the integral over ``b`` by the fixed
    ``nodes``: ``(n,)``."""
    lf = log_integrand(theta, nodes.B, blk, layout, gl)
    return nodes.logdet + alogsumexp(lf + nodes.logw[None, :], axis=1)


class Problem(NamedTuple):
    """Everything the likelihood needs: the units, the denominators of the
    truncated ones (with which units they are), and the rules."""

    units: Block
    truncated: "tuple[Block, npt.NDArray] | None"
    layout: Layout
    gl: tuple[npt.NDArray, npt.NDArray]
    n_points: int


class AllNodes(NamedTuple):
    units: Nodes
    truncated: "Nodes | None"
    modes: Modes
    truncated_modes: "Modes | None"


def place_nodes(
    theta: npt.NDArray, prob: Problem, last: "AllNodes | None" = None
) -> AllNodes:
    """The adaptive nodes of every term at ``theta`` (warm-started from
    the ``last`` modes)."""
    modes = unit_modes(
        theta,
        prob.units,
        prob.layout,
        prob.gl,
        None if last is None else last.modes.b,
    )
    nodes = adaptive_nodes(modes, prob.n_points)
    t_nodes = t_modes = None
    if prob.truncated is not None:
        blk, _ = prob.truncated
        t_modes = unit_modes(
            theta,
            blk,
            prob.layout,
            prob.gl,
            (
                None
                if last is None or last.truncated_modes is None
                else (last.truncated_modes.b)
            ),
        )
        t_nodes = adaptive_nodes(t_modes, prob.n_points)
    return AllNodes(nodes, t_nodes, modes, t_modes)


def unit_log_likelihoods(theta: Any, prob: Problem, nodes: AllNodes) -> Any:
    """Each unit's log likelihood with the nodes fixed: ``(n,)``."""
    ll = log_marginal(theta, prob.units, nodes.units, prob.layout, prob.gl)
    if prob.truncated is not None and nodes.truncated is not None:
        blk, which = prob.truncated
        denom = log_marginal(theta, blk, nodes.truncated, prob.layout, prob.gl)
        mask = np.zeros((blk.n, prob.units.n))
        mask[np.arange(blk.n), which] = 1.0
        ll = ll - anp.dot(denom, mask)
    return ll


def fixed_node_neg_ll(theta: Any, prob: Problem, nodes: AllNodes) -> Any:
    """The negative log-likelihood with the nodes fixed (a smooth function
    of ``theta`` that autograd differentiates)."""
    return -anp.sum(unit_log_likelihoods(theta, prob, nodes))


def adaptive_neg_ll(
    theta: npt.NDArray, prob: Problem, last: "AllNodes | None" = None
) -> float:
    """The negative log-likelihood with the nodes placed at ``theta``
    itself: the model's likelihood, by adaptive quadrature."""
    nodes = place_nodes(np.asarray(theta, float), prob, last)
    with np.errstate(all="ignore"):
        return float(fixed_node_neg_ll(np.asarray(theta, float), prob, nodes))


def posterior_weights(lf: npt.NDArray, nodes: Nodes) -> npt.NDArray:
    """The normalised quadrature weights of each unit's posterior of
    ``b`` from its log integrand ``lf`` at the nodes (``(n, K)``)."""
    a = lf + nodes.logw[None, :]
    return np.exp(a - logsumexp(a, axis=1, keepdims=True))
