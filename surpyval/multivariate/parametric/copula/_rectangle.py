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
  (where a conditional median passes a limit). It is a deterministic,
  smooth function of the limits and the correlations, which a likelihood
  search and its finite-difference Hessian need (a randomised integrator
  gives neither), accurate to about 1e-9 (see ``test_rectangle.py``);
- four or more: Genz's (1992) sequential conditioning on all but the last
  two coordinates, integrated on a fixed rank-1 lattice (tent
  transformed, so the rule is the same on every call), with the bivariate
  rectangle of the last two in closed form at each node. It is
  deterministic but accurate only to about 1e-4 (1e-5 away from
  near-singular correlation matrices), so it serves the predictions (a
  joint survival) and not the likelihood: a fit refuses rows with four or
  more censored coordinates (see ``elliptical_nd.py``).

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

#: The most coordinates a likelihood's rectangle may have (see the module
#: docstring): beyond it the integral is not accurate enough to fit with.
MAX_LIKELIHOOD_AXES = 3

# The outer rule of the three-coordinate integral
_Z, _ZC, _W = _tanh_sinh(step=1.0 / 8.0)

# The lattice of four or more coordinates: ``_LATTICE_N`` points with the
# generating vector below (built component by component for the Korobov
# space of smoothness 2), one component per integrated dimension.
_LATTICE_N = 4093
_LATTICE_Z = (1, 1210, 1542, 1785, 942, 736)
#: The most coordinates :func:`rectangle` integrates (a t rectangle has
#: one dimension more, its mixing scale).
MAX_AXES = len(_LATTICE_Z) + 2


@functools.lru_cache(maxsize=None)
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
    # Where the integrand bends: where a conditional median passes a limit
    limits = onp.concatenate([lo[:, 1:], hi[:, 1:]], axis=1)
    with onp.errstate(divide="ignore", invalid="ignore"):
        kinks = limits / onp.tile(r, 2)[None, :]
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


def _rect_lattice(lo: Any, hi: Any, corr: Any, nu: float | None) -> Any:
    """Four or more coordinates, by Genz's conditioning on a fixed
    lattice (see the module docstring)."""
    m, k = lo.shape
    outer = k - 2
    dims = outer + (nu is not None)
    if dims > len(_LATTICE_Z):
        raise ValueError(
            f"A joint probability over {k} axes is not computed: the "
            f"rule covers up to {MAX_AXES - (nu is not None)} for this "
            "copula. Ask for the probability of fewer axes at a time (the "
            "model of a subset of the axes has the same pairwise "
            "correlations)."
        )
    nodes = _lattice(dims)
    chol = onp.linalg.cholesky(corr)
    tail = chol[outer:, outer:] @ chol[outer:, outer:].T
    sd = onp.sqrt(onp.diag(tail))
    rho_c = float(onp.clip(tail[0, 1] / (sd[0] * sd[1]), -1.0, 1.0))
    out = onp.empty(m)
    # (rows in blocks, to bound the memory of rows x nodes)
    block = max(1, 400_000 // _LATTICE_N)
    for start in range(0, m, block):
        a = lo[start : start + block, None, :]
        b = hi[start : start + block, None, :]
        if nu is not None:
            w0 = nodes[:, -1]
            chi2 = 2.0 * onp.where(
                w0 <= 0.5,
                gammaincinv(nu / 2.0, onp.minimum(w0, 0.5)),
                gammainccinv(nu / 2.0, onp.minimum(1.0 - w0, 0.5)),
            )
            scale = onp.sqrt(chi2 / nu)[None, :, None]
            a, b = a * scale, b * scale
        rows = a.shape[0]
        y = onp.zeros((rows, _LATTICE_N, outer))
        f = onp.ones((rows, _LATTICE_N))
        for i in range(outer):
            shift = y[:, :, :i] @ chol[i, :i] if i else 0.0
            s, v1, v2 = _lower_tail(
                (a[:, :, i] - shift) / chol[i, i],
                (b[:, :, i] - shift) / chol[i, i],
            )
            d, e = ndtr(v2), ndtr(v1)
            f = f * (e - d)
            with onp.errstate(divide="ignore"):
                step = ndtri(d + nodes[None, :, i] * (e - d))
            y[:, :, i] = s * onp.clip(step, -38.0, 38.0)
        shift = y @ chol[outer:, :outer].T
        inner = [
            ((lim[:, :, outer:] - shift) / sd[None, None, :]).reshape(-1, 2)
            for lim in (a, b)
        ]
        mass = _rect2(inner[0], inner[1], rho_c, None).reshape(f.shape)
        out[start : start + block] = onp.mean(f * mass, axis=1)
    return onp.clip(out, 0.0, 1.0)


def rectangle(
    lo: npt.ArrayLike,
    hi: npt.ArrayLike,
    corr: npt.ArrayLike,
    nu: float | None = None,
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
    # Rows with the same limits (units censored at one time) once
    both, back = onp.unique(
        onp.concatenate([lo, hi], axis=1), axis=0, return_inverse=True
    )
    lo, hi = both[:, :k], both[:, k:]
    if k == 1:
        out = _rect1(lo[:, 0], hi[:, 0], nu)
    elif k == 2:
        out = _rect2(lo, hi, float(corr[0, 1]), nu)
    elif k == 3:
        out = _rect3(lo, hi, corr, nu)
    else:
        out = _rect_lattice(lo, hi, corr, nu)
    return onp.asarray(out, dtype=float)[onp.ravel(back)]
