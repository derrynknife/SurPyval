"""Predictions of a fitted joint model, in the data's own units (#891).

A model's natural parameters are carried to the likelihood's internal
vector in the natural units themselves (``Scaling.identity``), so the
predictions use exactly the functions the fit maximised.

- **A new unit** (no readings): its survival is the average of the
  conditional survival ``S(t | b)`` over the random effects' distribution
  ``N(0, D)``, by Gauss-Hermite quadrature (:func:`marginal`).
- **A unit with readings, still running at ``age``** (dynamic
  prediction): the average of ``S(t | b) / S(age | b)`` over the
  posterior of its random effects given its readings and its survival to
  ``age``, by adaptive Gauss-Hermite quadrature at that posterior's mode
  (:class:`Posterior`).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.special import logsumexp

from ._likelihood import (
    RIGHT,
    Block,
    Layout,
    Parts,
    Q,
    _log_cumhaz,
    adaptive_nodes,
    basis,
    chol_factor,
    hermite_grid,
    legendre,
    log_integrand,
    make_block,
    posterior_weights,
    unit_modes,
    unpack,
)
from ._params import Scaling, internal_from_natural

#: Gauss-Hermite points per dimension for a new unit's survival (over the
#: random effects' distribution) and for a dynamic prediction (over a
#: unit's posterior): more than the fit's, as a prediction is made once.
MARGINAL_POINTS = 24
POSTERIOR_POINTS = 15
#: Query points evaluated together (bounds the memory of the node grid).
CHUNK = 256


class Evaluator(NamedTuple):
    """A model's parameters as the likelihood's vector, in natural units:
    ``values`` are the natural parameters with an aliased coefficient as
    0."""

    theta: npt.NDArray
    parts: Parts
    layout: Layout
    gl: tuple[npt.NDArray, npt.NDArray]

    @classmethod
    def build(
        cls,
        values: npt.ArrayLike,
        p: int,
        pw: int,
        centre: float,
        n_legendre: int,
    ) -> "Evaluator":
        layout = Layout(p, pw)
        theta = internal_from_natural(
            values, layout, Scaling.identity(centre, p, pw)
        )
        return cls(theta, unpack(theta, layout), layout, legendre(n_legendre))

    @property
    def k(self) -> float:
        return float(np.exp(self.parts.lk))

    def fixed(self, W: npt.NDArray) -> npt.NDArray:
        """Each unit's path coefficients at ``b = 0`` (``W`` its path
        covariates): ``(n, Q)``, centred at the marker centre."""
        W1 = np.hstack([np.ones((W.shape[0], 1)), W])
        return W1 @ np.asarray(self.parts.path, dtype=float).T

    def lin(self, Z: npt.NDArray) -> npt.NDArray:
        return -self.k * float(self.parts.la) + Z @ np.asarray(
            self.parts.gamma, dtype=float
        )

    def log_cumhaz(
        self,
        U: npt.NDArray,
        lo: npt.NDArray,
        hi: npt.NDArray,
        Z: npt.NDArray,
    ) -> npt.NDArray:
        """``log H(lo -> hi | b)`` for path coefficients ``U``
        (``(n, K, Q)``): ``-inf`` where ``hi <= lo``."""
        empty = ~(hi > lo)
        hi_safe = np.where(empty, lo + 1.0, hi)
        with np.errstate(all="ignore"):
            out = np.asarray(
                _log_cumhaz(
                    self.k,
                    self.lin(Z),
                    float(self.parts.assoc),
                    U,
                    lo,
                    hi_safe,
                    self.gl,
                )
            )
        return np.where(empty[:, None], -np.inf, out)

    def log_hazard(
        self, U: npt.NDArray, x: npt.NDArray, Z: npt.NDArray
    ) -> npt.NDArray:
        """``log h(x | b)`` (``(n, K)``), with the Weibull's limit at
        ``x = 0``: ``-inf`` for a shape above 1, ``inf`` below it."""
        k = self.k
        pos = x > 0
        x_safe = np.where(pos, x, 1.0)
        m = np.einsum("nq,nkq->nk", basis(x_safe), U)
        with np.errstate(all="ignore"):
            log_t = np.where(
                pos,
                (k - 1.0) * np.log(x_safe),
                -np.inf if k > 1 else (np.inf if k < 1 else 0.0),
            )
        return (
            float(self.parts.lk)
            + log_t[:, None]
            + self.lin(Z)[:, None]
            + float(self.parts.assoc) * m
        )

    def D(self) -> npt.NDArray:
        L = np.asarray(chol_factor(np.asarray(self.parts.chol, float)))
        return L @ L.T


# ---------------------------------------------------------------------------
# A new unit
# ---------------------------------------------------------------------------
def marginal(
    ev: Evaluator, x: npt.NDArray, Z: npt.NDArray, W: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """``(log sf, log df)`` of a new unit with covariates ``Z`` and
    ``W`` (one row per time ``x``), for ``x > 0``."""
    Zg, logw = hermite_grid(MARGINAL_POINTS)
    logw = logw - 0.5 * Q * np.log(np.pi)
    L = np.asarray(chol_factor(np.asarray(ev.parts.chol, float)))
    b = np.sqrt(2.0) * Zg @ L.T  # (K, Q)
    log_sf = np.empty(x.size)
    log_df = np.empty(x.size)
    for start in range(0, x.size, CHUNK):
        sl = slice(start, start + CHUNK)
        xs, Zs, Ws = x[sl], Z[sl], W[sl]
        U = ev.fixed(Ws)[:, None, :] + b[None, :, :]
        with np.errstate(over="ignore"):
            log_S = -np.exp(ev.log_cumhaz(U, np.zeros(xs.size), xs, Zs))
        log_h = ev.log_hazard(U, xs, Zs)
        log_sf[sl] = logsumexp(logw[None, :] + log_S, axis=1)
        with np.errstate(all="ignore"):
            log_df[sl] = logsumexp(logw[None, :] + log_h + log_S, axis=1)
    return log_sf, log_df


def draw_units(
    ev: Evaluator,
    Z: npt.NDArray,
    W: npt.NDArray,
    rng: np.random.Generator,
) -> tuple[npt.NDArray, npt.NDArray]:
    """New units with covariates ``Z`` and ``W`` (a row each): each
    unit's random effects drawn from ``N(0, D)``, then its failure time
    from ``S(t | b)`` by solving ``H(0 -> t | b) = E``, ``E`` exponential,
    by bisection in log time (``inf`` for a unit whose hazard dies out
    before ``E`` is reached). Returns the times and each unit's path
    coefficients, centred at the marker centre (``(n, Q)``)."""
    from .dynamic import _bisect_log_time

    n = Z.shape[0]
    L = np.linalg.cholesky(ev.D())
    b = rng.standard_normal((n, Q)) @ L.T
    E = np.asarray(rng.exponential(size=n), dtype=float)
    U = ev.fixed(W) + b
    zero = np.zeros(n)

    def log_sf(t: npt.NDArray) -> npt.NDArray:
        with np.errstate(over="ignore"):
            return -np.exp(ev.log_cumhaz(U[:, None, :], zero, t, Z)[:, 0])

    T = _bisect_log_time(log_sf, -E, float(np.exp(ev.parts.la)))
    return T, U


def marginal_random(
    ev: Evaluator,
    size: int,
    z: npt.NDArray,
    w: npt.NDArray,
    rng: np.random.Generator,
) -> npt.NDArray:
    """Failure times of ``size`` new units with covariates ``z`` and
    ``w`` (see :func:`draw_units`)."""
    Z = np.repeat(z[None, :], size, axis=0)
    W = np.repeat(w[None, :], size, axis=0)
    return draw_units(ev, Z, W, rng)[0]


# ---------------------------------------------------------------------------
# A unit with readings
# ---------------------------------------------------------------------------
class Posterior(NamedTuple):
    """The posterior of one unit's random effects given its readings and
    its survival to ``age``, as quadrature nodes and weights: ``U`` the
    path coefficients at the nodes (``(K, Q)``), ``weights`` their
    normalised weights."""

    U: npt.NDArray
    weights: npt.NDArray
    z: npt.NDArray
    age: float

    @classmethod
    def build(
        cls,
        ev: Evaluator,
        t: npt.NDArray,
        y: npt.NDArray,
        z: npt.NDArray,
        w: npt.NDArray,
        age: float,
        centre: float,
        n_points: int = POSTERIOR_POINTS,
    ) -> "Posterior":
        blk: Block = make_block(
            [t], [y - centre], [RIGHT], [age], [age], z[None, :], w[None, :]
        )
        with np.errstate(all="ignore"):
            modes = unit_modes(ev.theta, blk, ev.layout, ev.gl)
            nodes = adaptive_nodes(modes, n_points)
            lf = np.asarray(
                log_integrand(ev.theta, nodes.B, blk, ev.layout, ev.gl)
            )
        weights = posterior_weights(lf, nodes)[0]
        U = ev.fixed(w[None, :])[0] + nodes.B[0]
        return cls(U, weights, z, float(age))

    def log_sf(self, ev: Evaluator, x: npt.NDArray) -> npt.NDArray:
        """``log P(T > x | T > age, readings)`` for ``x >= age``."""
        out = np.empty(x.size)
        for start in range(0, x.size, CHUNK):
            xs = x[start : start + CHUNK]
            U = np.broadcast_to(self.U, (xs.size,) + self.U.shape)
            Zs = np.repeat(self.z[None, :], xs.size, axis=0)
            lo = np.full(xs.size, self.age)
            with np.errstate(over="ignore"):
                log_S = -np.exp(ev.log_cumhaz(U, lo, xs, Zs))
            with np.errstate(divide="ignore"):
                out[start : start + CHUNK] = logsumexp(
                    np.log(self.weights)[None, :] + log_S, axis=1
                )
        return out

    def log_df(self, ev: Evaluator, x: npt.NDArray) -> npt.NDArray:
        """The log density of the failure age at ``x >= age``, given the
        readings and survival to ``age``."""
        out = np.empty(x.size)
        for start in range(0, x.size, CHUNK):
            xs = x[start : start + CHUNK]
            U = np.broadcast_to(self.U, (xs.size,) + self.U.shape)
            Zs = np.repeat(self.z[None, :], xs.size, axis=0)
            lo = np.full(xs.size, self.age)
            with np.errstate(over="ignore"):
                log_S = -np.exp(ev.log_cumhaz(U, lo, xs, Zs))
            log_h = ev.log_hazard(np.ascontiguousarray(U), xs, Zs)
            with np.errstate(divide="ignore", invalid="ignore"):
                out[start : start + CHUNK] = logsumexp(
                    np.log(self.weights)[None, :] + log_h + log_S, axis=1
                )
        return out

    def path_moments(
        self, x: npt.NDArray, centre: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """The posterior mean and standard deviation of the unit's true
        marker ``m(x)``."""
        m = basis(x) @ self.U.T + centre  # (n, K)
        mean = m @ self.weights
        var = ((m - mean[:, None]) ** 2) @ self.weights
        return mean, np.sqrt(np.maximum(var, 0.0))
