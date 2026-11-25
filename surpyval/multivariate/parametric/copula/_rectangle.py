"""Rectangle probabilities of the multivariate normal and t distributions.

The likelihood of a row with censored coordinates under an elliptical
copula, and the copula's CDF and survival function, are the probability
that a standard multivariate normal (or t) vector with a given correlation
matrix falls in a rectangle :math:`(l_1, h_1] \\times \\dots \\times (l_k,
h_k]`, each side finite or not. :func:`rectangle` gives it, by the number
of coordinates ``k``:

- one: the univariate CDF;
- two: the bivariate CDFs of the bivariate copulas (``elliptical.py``);
- three: the integral, over the first coordinate, of the bivariate
  rectangle of the other two given it, by tanh-sinh quadrature on the
  first coordinate's probability scale, split where the integrand bends
  (where a conditional median passes a limit, and where the other two's
  standardised limits meet). It is a deterministic, smooth function of
  the limits and the correlations, which a likelihood search and its
  finite-difference Hessian need (a randomised integrator gives neither).
  Against R's mvtnorm (``tests/reference/test_rectangle3_mvtnorm.py``)
  it agrees to within the reference's own error (1e-10 for the normal,
  up to 4e-7 for the t), also for correlation matrices whose smallest
  eigenvalue is 7e-4; against a rule ten times finer it is within 4e-10
  (and 5e-6 of the probability, where that is above 1e-8) down to a
  smallest eigenvalue of 1e-4;
- four or more: Genz's (1992) sequential conditioning on all but the last
  coordinate, which is in closed form at each node, integrated on a fixed
  rank-1 lattice of 65,521 points (tent transformed, so the rule is the
  same on every call and the result a smooth function of the limits and
  the correlations for a given order of conditioning). The coordinates
  are conditioned in Genz and Bretz's order (least likely first, given
  the ones before), which makes the rule accurate relative to small
  probabilities and for nearly singular correlation matrices.
  A t vector is the normal one over an independent scale, which is the
  lattice's first coordinate. Against references to 1e-12 (an
  equicorrelated or one-factor correlation reduces the integral to one
  dimension; see ``test_rectangle.py``), the error is below 3e-6 for
  four to ten coordinates, and below 4e-4 of the probability itself for
  the normal (3e-3 for the t, where the probability is below 1e-4).

Every interval is written for the coordinate or its negative, whichever
puts it in the lower tail, so that a small probability is a difference of
small numbers.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as onp
import numpy.typing as npt
from scipy.special import gammainccinv, gammaincinv, ndtr, ndtri, stdtr
from scipy.stats import multivariate_normal

from .elliptical import (
    _SMALL_CDF,
    StudentT,
    _normal_orthant,
    _tanh_sinh,
)

# The outer rule of the three-coordinate integral (a step of 1/12: with
# 1/8 a nearly singular matrix's probability was off by up to 2e-7; with
# 1/12 it is within 4e-10 of a rule ten times finer)
_Z, _ZC, _W = _tanh_sinh(step=1.0 / 12.0)

# The lattice of four or more coordinates: ``_LATTICE_N`` (prime) points
# with the generating vector below, built component by component (the
# fast construction of Nuyens and Cools 2006) for the Korobov space of
# smoothness 2 with product weights ``1 / sqrt(j + 1)``; one component per
# integrated dimension.
_LATTICE_N = 65521
_LATTICE_Z = (
    1, 24876, 5411, 42062, 24076, 38393, 57101,
    26116, 53040, 44926, 30341, 59079, 63116, 52047,
)  # fmt: skip
#: The most coordinates :func:`rectangle` integrates (a t rectangle has
#: one dimension more, its mixing scale).
MAX_AXES = len(_LATTICE_Z) + 1


@functools.cache
def _lattice(dims: int) -> npt.NDArray:
    """The lattice's points in ``dims`` dimensions, shifted off the
    faces of the unit cube and tent transformed."""
    k = onp.arange(_LATTICE_N)[:, None]
    z = onp.asarray(_LATTICE_Z[:dims])[None, :]
    w = ((k * z) % _LATTICE_N + 0.25) / _LATTICE_N
    return 1.0 - onp.abs(2.0 * w - 1.0)


def _cdf(x: Any, nu: float | None) -> Any:
    return ndtr(x) if nu is None else stdtr(nu, x)


def _quantile(w: Any, wc: Any, nu: float | None) -> Any:
    """The quantile at ``w``, from ``w`` or its complement ``wc``,
    whichever is the smaller."""
    if nu is None:
        return onp.where(
            w <= 0.5,
            ndtri(onp.minimum(w, 0.5)),
            -ndtri(onp.minimum(wc, 0.5)),
        )
    return StudentT._quantile(nu, w, wc)


def _lower_tail(lo: Any, hi: Any) -> tuple:
    """Each interval ``(lo, hi]`` of ``Z`` as ``(v2, v1]`` of ``s Z``,
    with ``s = -1`` where the interval's centre is above 0: ``(s, v1,
    v2)``. Its mass is then ``F(v1) - F(v2)`` with both in the lower
    tail."""
    with onp.errstate(invalid="ignore"):
        flip = (lo + hi) > 0
    return (
        onp.where(flip, -1.0, 1.0),
        onp.where(flip, -lo, hi),
        onp.where(flip, -hi, lo),
    )


def _orthant2(x: Any, y: Any, rho: float, nu: float | None) -> Any:
    """``P(X <= x, Y <= y)`` for a standard bivariate normal (``nu`` is
    ``None``) or t pair with correlation ``rho``; either may be
    infinite."""
    out = onp.zeros(onp.shape(x))
    x_top, y_top = x == onp.inf, y == onp.inf
    none = (x == -onp.inf) | (y == -onp.inf) | onp.isnan(x) | onp.isnan(y)
    out = onp.where(x_top & ~none, _cdf(onp.where(none, 0.0, y), nu), out)
    out = onp.where(
        y_top & ~x_top & ~none, _cdf(onp.where(none, 0.0, x), nu), out
    )
    both = ~(x_top | y_top | none)
    if not both.any():
        return out
    a, b = x[both], y[both]
    if nu is None:
        if rho == 0.0:
            value = ndtr(a) * ndtr(b)
        else:
            value = onp.atleast_1d(
                multivariate_normal.cdf(
                    onp.column_stack([a, b]),
                    mean=[0.0, 0.0],
                    cov=[[1.0, rho], [rho, 1.0]],
                    allow_singular=True,
                )
            ).astype(float)
            small = value < _SMALL_CDF
            if small.any():
                value[small] = _normal_orthant(a[small], b[small], rho)
    else:
        value = onp.asarray(
            StudentT._ordered_cdf(stdtr(nu, a), stdtr(nu, b), rho, nu),
            dtype=float,
        )
    out[both] = value
    return out


def _rect1(lo: Any, hi: Any, nu: float | None) -> Any:
    _, v1, v2 = _lower_tail(lo, hi)
    return _cdf(v1, nu) - _cdf(v2, nu)


def _rect2(lo: Any, hi: Any, rho: float, nu: float | None) -> Any:
    """The mass of each row's rectangle (``lo``, ``hi`` of shape ``(m,
    2)``) under a standard bivariate normal or t with correlation
    ``rho``."""
    s, v1, v2 = _lower_tail(lo, hi)
    sign = s[:, 0] * s[:, 1]
    out = onp.zeros(lo.shape[0])
    for which in (1.0, -1.0):
        rows = sign == which
        if not rows.any():
            continue
        r = which * rho
        a1, a2, b1, b2 = v1[rows, 0], v2[rows, 0], v1[rows, 1], v2[rows, 1]
        out[rows] = (
            _orthant2(a1, b1, r, nu)
            - _orthant2(a2, b1, r, nu)
            - _orthant2(a1, b2, r, nu)
            + _orthant2(a2, b2, r, nu)
        )
    return onp.clip(out, 0.0, 1.0)


def _one_minus_square(r: Any) -> Any:
    return (1.0 - onp.abs(r)) * (1.0 + onp.abs(r))


def _rect3(lo: Any, hi: Any, corr: Any, nu: float | None) -> Any:
    """Three coordinates: the first integrated on its probability scale,
    the other two's rectangle given it in closed form."""
    m = lo.shape[0]
    r = corr[1:, 0]
    s = onp.sqrt(_one_minus_square(r))
    rho_c = float(
        onp.clip((corr[1, 2] - r[0] * r[1]) / (s[0] * s[1]), -1.0, 1.0)
    )
    lo0, hi0 = lo[:, 0], hi[:, 0]
    # Where the integrand bends: where a conditional median passes a
    # limit, and where the other two's standardised limits meet (a limit
    # of one equal to a limit of the other, negated if their conditional
    # correlation is negative). Near the meeting points the bivariate
    # rectangle changes over a width of about sqrt(1 - rho_c^2), which a
    # nearly singular correlation matrix makes narrow: without them as
    # panel ends the rule missed up to 1e-5 of the probability.
    limits = onp.concatenate([lo[:, 1:], hi[:, 1:]], axis=1)
    sign = 1.0 if rho_c >= 0.0 else -1.0
    with onp.errstate(divide="ignore", invalid="ignore"):
        kinks = limits / onp.tile(r, 2)[None, :]
        meet = onp.stack(
            [
                (a / s[0] - sign * b / s[1])
                / (r[0] / s[0] - sign * r[1] / s[1])
                for a in (lo[:, 1], hi[:, 1])
                for b in (lo[:, 2], hi[:, 2])
            ],
            axis=1,
        )
    kinks = onp.concatenate([kinks, meet], axis=1)
    kinks = onp.where(onp.isfinite(kinks), kinks, lo0[:, None])
    kinks = onp.clip(kinks, lo0[:, None], hi0[:, None])
    edges = onp.sort(
        onp.concatenate([lo0[:, None], kinks, hi0[:, None]], axis=1), axis=1
    )
    e, ec = _cdf(edges, nu), _cdf(-edges, nu)
    inner_nu = None if nu is None else nu + 1.0
    total = onp.zeros(m)
    for p in range(edges.shape[1] - 1):
        a, b = e[:, p], e[:, p + 1]
        width = onp.where(a > 0.5, ec[:, p] - ec[:, p + 1], b - a)
        act = width > 0
        if not act.any():
            continue
        wd = width[act, None]
        w = onp.clip(a[act, None] + wd * _Z[None, :], 1e-300, None)
        wc = onp.clip(ec[act, p + 1][:, None] + wd * _ZC[None, :], 1e-300, 1)
        z = _quantile(w, wc, nu)
        if nu is None:
            g = onp.ones_like(z)
        else:
            g = onp.hypot(z, onp.sqrt(nu)) / onp.sqrt(nu + 1.0)
        with onp.errstate(invalid="ignore"):
            inner = [
                (
                    (lim[act][:, None, :] - z[:, :, None] * r[None, None, :])
                    / (s[None, None, :] * g[:, :, None])
                ).reshape(-1, 2)
                for lim in (lo[:, 1:], hi[:, 1:])
            ]
        mass = _rect2(inner[0], inner[1], rho_c, inner_nu)
        mass = onp.nan_to_num(mass, nan=0.0).reshape(z.shape)
        total[act] += width[act] * (mass @ _W)
    return onp.clip(total, 0.0, 1.0)


def _smooth_ends(t: Any) -> tuple:
    """``(w, 1 - w, dw / dt)`` for ``w = t^3 (10 - 15 t + 6 t^2)``, a
    polynomial map of ``(0, 1)`` onto itself flat at both ends: it tames
    the t's mixing scale, whose quantile is singular at 0 and 1, for the
    lattice."""

    def psi(v: Any) -> Any:
        return v**3 * (10.0 + v * (6.0 * v - 15.0))

    return psi(t), psi(1.0 - t), 30.0 * (t * (1.0 - t)) ** 2


def _interval_mass(a: Any, b: Any) -> Any:
    """``Phi(b) - Phi(a)``, formed in the lower tail."""
    _, v1, v2 = _lower_tail(a, b)
    with onp.errstate(invalid="ignore"):
        return onp.nan_to_num(ndtr(v1) - ndtr(v2), nan=1.0)


def _order(lo: Any, hi: Any, corr: Any) -> npt.NDArray:
    """Each row's order of conditioning, by Genz and Bretz's (2009,
    section 4.1.3) prioritisation: at each step the coordinate least
    likely to lie in its interval given the ones before it (each at its
    conditional mean in its interval). The rule is then accurate relative
    to small probabilities and for nearly singular correlation matrices,
    where a fixed order can be wrong by orders of magnitude. (The normal
    order serves the t too.)"""
    m, k = lo.shape
    rows = onp.arange(m)
    a, b = lo.copy(), hi.copy()
    C = onp.broadcast_to(corr, (m, k, k)).copy()
    L = onp.zeros((m, k, k))
    y = onp.zeros((m, k))
    perm = onp.tile(onp.arange(k), (m, 1))
    for i in range(k):
        rest = L[:, i:, :i]
        sd = onp.sqrt(
            onp.maximum(
                onp.diagonal(C, axis1=1, axis2=2)[:, i:]
                - onp.sum(rest**2, axis=2),
                1e-300,
            )
        )
        mu = onp.einsum("mjl,ml->mj", rest, y[:, :i])
        with onp.errstate(invalid="ignore"):
            p = _interval_mass((a[:, i:] - mu) / sd, (b[:, i:] - mu) / sd)
        j = i + onp.argmin(p, axis=1)
        for arr in (a, b, perm, y):
            arr[rows, i], arr[rows, j] = arr[rows, j], arr[rows, i].copy()
        for arr in (C, L):
            tmp = arr[rows, i].copy()
            arr[rows, i] = arr[rows, j]
            arr[rows, j] = tmp
        tmp = C[rows, :, i].copy()
        C[rows, :, i] = C[rows, :, j]
        C[rows, :, j] = tmp
        diag = onp.sqrt(
            onp.maximum(C[:, i, i] - onp.sum(L[:, i, :i] ** 2, axis=1), 1e-300)
        )
        L[:, i, i] = diag
        L[:, i + 1 :, i] = (
            C[:, i + 1 :, i]
            - onp.einsum("mjl,ml->mj", L[:, i + 1 :, :i], L[:, i, :i])
        ) / diag[:, None]
        centre = onp.einsum("ml,ml->m", L[:, i, :i], y[:, :i])
        with onp.errstate(invalid="ignore", divide="ignore"):
            alpha = (a[:, i] - centre) / diag
            beta = (b[:, i] - centre) / diag
            dens = onp.where(
                onp.isfinite(alpha), onp.exp(-0.5 * alpha**2), 0.0
            ) - onp.where(onp.isfinite(beta), onp.exp(-0.5 * beta**2), 0.0)
            mass = onp.maximum(_interval_mass(alpha, beta), 1e-300)
            mean = dens / (onp.sqrt(2.0 * onp.pi) * mass)
        # (an interval far in a tail: its end nearest the bulk)
        mean = onp.where(
            onp.isfinite(mean),
            mean,
            onp.where(onp.isfinite(alpha), alpha, beta),
        )
        y[:, i] = onp.clip(mean, -38.0, 38.0)
    return perm


def _rect_lattice(
    lo: Any, hi: Any, corr: Any, nu: float | None, order: Any
) -> Any:
    """Four or more coordinates, by Genz's conditioning on a fixed
    lattice (see the module docstring)."""
    m, k = lo.shape
    dims = k - 1 + (nu is not None)
    if dims > len(_LATTICE_Z):
        raise ValueError(
            f"A joint probability over {k} axes is not computed: the "
            f"rule covers up to {MAX_AXES - (nu is not None)} for this "
            "copula. Ask for the probability of fewer axes at a time (the "
            "model of a subset of the axes has the same pairwise "
            "correlations)."
        )
    nodes = _lattice(dims)
    scale: Any = 1.0
    weight: Any = 1.0
    first = int(nu is not None)
    if nu is not None:
        # The t vector is the normal one over sqrt(chi2_nu / nu); its
        # mixing scale is the lattice's first coordinate (the best
        # spread: it moves every limit at once).
        w0, w0c, weight = _smooth_ends(nodes[:, 0])
        chi2 = 2.0 * onp.where(
            w0 <= 0.5,
            gammaincinv(nu / 2.0, onp.minimum(w0, 0.5)),
            gammainccinv(nu / 2.0, onp.minimum(w0c, 0.5)),
        )
        scale = onp.sqrt(chi2 / nu)[None, :]
    out = onp.empty(m)
    orders, which = onp.unique(order, axis=0, return_inverse=True)
    which = onp.ravel(which)
    # (rows in blocks, to bound the memory of rows x nodes x axes)
    block = max(1, 2_000_000 // (nodes.shape[0] * k))
    for g, perm in enumerate(orders):
        chol = onp.linalg.cholesky(corr[onp.ix_(perm, perm)])
        members = onp.flatnonzero(which == g)
        for start in range(0, members.size, block):
            rows = members[start : start + block]
            a = lo[onp.ix_(rows, perm)]
            b = hi[onp.ix_(rows, perm)]
            y = onp.zeros((rows.size, nodes.shape[0], k - 1))
            f = onp.ones((rows.size, nodes.shape[0]))
            for i in range(k):
                shift = y[:, :, :i] @ chol[i, :i] if i else 0.0
                s, v1, v2 = _lower_tail(
                    (a[:, i, None] * scale - shift) / chol[i, i],
                    (b[:, i, None] * scale - shift) / chol[i, i],
                )
                d, e = ndtr(v2), ndtr(v1)
                f = f * (e - d)
                if i == k - 1:
                    break
                with onp.errstate(divide="ignore"):
                    step = ndtri(d + nodes[None, :, first + i] * (e - d))
                y[:, :, i] = s * onp.clip(step, -38.0, 38.0)
            out[rows] = onp.mean(f * weight, axis=1)
    return onp.clip(out, 0.0, 1.0)


def rectangle(
    lo: npt.ArrayLike,
    hi: npt.ArrayLike,
    corr: npt.ArrayLike,
    nu: float | None = None,
    order: npt.ArrayLike | None = None,
) -> npt.NDArray:
    """The probability that a standard multivariate normal (``nu`` is
    ``None``) or t (``nu`` degrees of freedom) vector with correlation
    matrix ``corr`` lies in each row's rectangle ``(lo, hi]``.

    Parameters
    ----------
    lo, hi : array like, shape (m, k)
        The limits of each row's rectangle; ``-inf`` and ``inf`` for a
        side with none.
    corr : array like, shape (k, k)
        The correlation matrix (unit diagonal, positive definite).
    nu : float, optional
        The degrees of freedom of a t vector.
    order : array like of int, shape (m, k), optional
        For four or more coordinates, the order each row's coordinates
        are conditioned in (a permutation of ``range(k)`` per row); by
        default Genz and Bretz's prioritisation (see :func:`_order`).
        A fixed order makes the result a smooth function of the limits
        and the correlations; the default is smooth except where the
        prioritisation changes, where the result moves by about the
        rule's accuracy.

    Returns
    -------
    numpy array, shape (m,)
    """
    lo = onp.atleast_2d(onp.asarray(lo, dtype=float))
    hi = onp.atleast_2d(onp.asarray(hi, dtype=float))
    corr = onp.asarray(corr, dtype=float)
    m, k = lo.shape
    if k == 0 or m == 0:
        return onp.ones(m)
    given = (
        onp.zeros((m, 0))
        if order is None or k < 4
        else onp.broadcast_to(onp.asarray(order, dtype=int), (m, k))
    )
    # Rows with the same limits (units censored at one time) once
    both, back = onp.unique(
        onp.concatenate([lo, hi, given], axis=1), axis=0, return_inverse=True
    )
    lo, hi = both[:, :k], both[:, k : 2 * k]
    if k >= 4:
        order = (
            _order(lo, hi, corr)
            if order is None
            else both[:, 2 * k :].astype(int)
        )
    if k == 1:
        out = _rect1(lo[:, 0], hi[:, 0], nu)
    elif k == 2:
        out = _rect2(lo, hi, float(corr[0, 1]), nu)
    elif k == 3:
        out = _rect3(lo, hi, corr, nu)
    else:
        out = _rect_lattice(lo, hi, corr, nu, order)
    return onp.asarray(out, dtype=float)[onp.ravel(back)]
