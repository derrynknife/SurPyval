"""The joint likelihood's integral over each item's varying path
parameters by Laplace's approximation: no grid, so the cost grows with
the square of their number where a grid's grows as a power of it.

At the mode ``b_hat`` of an item's integrand ``exp(g(b))`` (what
:func:`._likelihood.modes` finds),

    log int exp(g(b)) db ~ g(b_hat) + (q / 2) log(2 pi)
                           - (1 / 2) log det(-g''(b_hat)),

exact where ``g`` is quadratic (a path linear in its parameters and no
marker effect) and off by a term of the order of one over the item's
number of readings otherwise (Tierney and Kadane 1986; for joint models
Rizopoulos, Verbeke and Lesaffre 2009).

**The curvature.** ``g''`` is the integrand's own second derivative,
not the Gauss-Newton one the search for the mode uses. It is taken by
central differences of ``g`` on a stencil of ``q^2 + q + 1`` points
about the mode (the mode, a step each way along each direction, and a
step each way along each pair's diagonal), in the coordinates ``z`` in
which the Gauss-Newton curvature is the identity (``b = b_hat + R z``),
so a step is the same share of the integrand's width for every
parameter and every unit of time. That makes the curvature a fixed
linear combination of the integrand at fixed points: the integrand is
evaluated there as on a grid (any path model works, and autograd
differentiates it in the parameters).

**The derivatives.** The approximation at parameters ``u`` is
``phi(z_hat(u), u)``, with ``phi(z, u) = g(z, u) - (1/2) log det
A(z, u)`` and ``A = -g''``, at the mode ``z_hat(u)``, which moves with
``u``. The stencil is placed at the mode of the parameters it was
placed for, and held. For the function it gives to have the
approximation's own value and gradient there:

- the value at the moved mode, to second order in the move, is the
  Newton step's: ``g(0, u) + (1/2) gamma' A^-1 gamma`` with ``gamma``
  the stencil's gradient of ``g`` at its centre (zero where it was
  placed);
- the curvature at the moved mode, to first order, adds ``c' A^-1
  gamma`` with ``c = -(1/2) d log det A / dz`` at the centre (the
  integrand's third derivatives, by central differences of the
  stencil's ``log det`` a step each way, once at each placement). This
  is the implicit-function term: ``d z_hat / du = A^-1 d gamma / du``.

So the gradient of :func:`log_integrals` at the parameters the stencil
was placed for is the gradient of the Laplace approximation there (the
mode's own movement included), which is what a search between
placements and the check of a maximum need. Its second derivatives by
the held stencil leave out second-order terms of the mode's movement;
the fit takes them by central differences of the gradient with the
stencil placed again at each point (``joint_ph._laplace_curvature``).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt

#: The stencil's step, and the step of the differences of its ``log
#: det``, in the coordinates in which the integrand's Gauss-Newton
#: curvature is the identity (so about a hundredth of the integrand's
#: width).
STEP = 0.01
SHIFT = 0.02
#: The most stencil points evaluated at once while placing (memory).
CHUNK = 200_000


@functools.lru_cache(maxsize=None)
def stencil(q: int) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """``(Z, Cg, Ch)``: the stencil's points ``(K, q)`` in units of the
    step (the centre first), and the weights that give the gradient
    ``(q, K)`` and the second derivatives ``(q * q, K)`` of a function
    from its values at the points ``STEP * Z``, by central
    differences."""
    pairs = [(i, j) for i in range(q) for j in range(i)]
    K = 1 + 2 * q + 2 * len(pairs)
    Z = np.zeros((K, q))
    Cg = np.zeros((q, K))
    Ch = np.zeros((q, q, K))
    for i in range(q):
        up, down = 1 + i, 1 + q + i
        Z[up, i], Z[down, i] = 1.0, -1.0
        Cg[i, up], Cg[i, down] = 0.5 / STEP, -0.5 / STEP
        Ch[i, i, [up, down]] = 1.0 / STEP**2
        Ch[i, i, 0] = -2.0 / STEP**2
    for k, (i, j) in enumerate(pairs):
        up, down = 1 + 2 * q + 2 * k, 2 + 2 * q + 2 * k
        Z[up, [i, j]], Z[down, [i, j]] = 1.0, -1.0
        w = np.zeros(K)
        w[[up, down]] = 1.0
        w[[1 + i, 1 + q + i, 1 + j, 1 + q + j]] = -1.0
        w[0] = 2.0
        Ch[i, j] = Ch[j, i] = w / (2.0 * STEP**2)
    return Z, Cg, Ch.reshape(q * q, K)


@dataclass
class LaplaceNodes:
    """An item's stencil, placed: the points ``b`` ``(N, K, q)`` in the
    varying parameters, the mode ``(N, q)`` at their centre, the
    coordinates' factor ``root`` ``(N, q, q)`` (``b = mode + root z``),
    the implicit-function term's ``c`` ``(N, q)`` (module docstring),
    and the curvature ``A`` ``(N, q, q)`` in ``z`` where it was placed."""

    b: npt.NDArray
    mode: npt.NDArray
    root: npt.NDArray
    c: npt.NDArray
    curvature: npt.NDArray

    @property
    def log_volume(self) -> npt.NDArray:
        """``log |det root|`` of each item, the change of variables."""
        return np.sum(
            np.log(np.abs(np.diagonal(self.root, axis1=1, axis2=2))), axis=1
        )

    def normal(self) -> tuple[npt.NDArray, npt.NDArray]:
        """The normal the approximation stands on: its mean (the mode)
        and a factor ``F`` ``(N, q, q)`` of its covariance ``F F'``, the
        inverse of the integrand's curvature at the mode."""
        cov = np.linalg.inv(self.curvature)
        cov = 0.5 * (cov + np.swapaxes(cov, -1, -2))
        return self.mode, self.root @ np.linalg.cholesky(cov)


def _points(
    mode: npt.NDArray, root: npt.NDArray, z: npt.NDArray
) -> npt.NDArray:
    """``mode + root z`` for the points ``z`` ``(K, q)``: ``(N, K, q)``."""
    return mode[:, None, :] + np.einsum("nij,kj->nki", root, z)


def _values(
    theta: dict,
    model: Any,
    arrays: Any,
    b: Any,
    t: npt.NDArray,
    d: npt.NDArray,
    readings: bool,
    upper: npt.NDArray | None,
) -> Any:
    """The log integrand at the points ``b`` ``(N, K, q)``."""
    from ._likelihood import log_integrand

    zero = np.zeros(b.shape[:2])
    return log_integrand(
        theta, model, arrays, (b, zero), t, d, readings, upper
    )


def _curvature(G: npt.NDArray, q: int) -> npt.NDArray:
    """``A = -g''`` ``(N, q, q)`` from the stencil's values ``(N, K)``."""
    _, _, Ch = stencil(q)
    return -(G @ Ch.T).reshape(len(G), q, q)


def _log_det(A: npt.NDArray) -> npt.NDArray:
    """``log det`` of each ``A``, which must be positive definite."""
    try:
        factor = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        raise np.linalg.LinAlgError(
            "the integrand's curvature at an item's mode is not positive "
            "definite"
        ) from None
    return 2.0 * np.sum(np.log(np.diagonal(factor, axis1=1, axis2=2)), axis=1)


def laplace_nodes(
    theta: dict,
    model: Any,
    arrays: Any,
    mode: npt.NDArray,
    root: npt.NDArray,
    t: npt.NDArray,
    d: npt.NDArray,
    readings: bool = True,
    upper: npt.NDArray | None = None,
) -> LaplaceNodes:
    """The stencil of each item placed at its ``mode`` (in the
    coordinates ``root``, both :func:`._likelihood.modes`'), with the
    implicit-function term found at ``theta``. Raises ``LinAlgError``
    where an item's curvature there is not positive definite (the mode
    is not a maximum)."""
    from ._likelihood import numeric_theta

    theta = numeric_theta(theta)
    N, q = mode.shape
    Z, _, _ = stencil(q)
    K = len(Z)
    b = _points(mode, root, STEP * Z)

    def curvature_at(centres: npt.NDArray) -> npt.NDArray:
        """``A`` with the stencil centred at ``centres`` ``(M, q)`` in
        ``z``: ``(N, M, q, q)``."""
        M = len(centres)
        z = (centres[:, None, :] + STEP * Z[None, :, :]).reshape(M * K, q)
        G = np.asarray(
            _values(
                theta,
                model,
                arrays,
                _points(mode, root, z),
                t,
                d,
                readings,
                upper,
            ),
            dtype=float,
        )
        return _curvature(G.reshape(N * M, K), q).reshape(N, M, q, q)

    with np.errstate(all="ignore"):
        A = curvature_at(np.zeros((1, q)))[:, 0]
        if not np.all(np.isfinite(A)):
            raise np.linalg.LinAlgError(
                "the integrand's curvature at an item's mode is not finite"
            )
        _log_det(A)
        # c = -(1/2) d log det A / dz, a direction (two centres) at a time
        # in as many as fit the memory
        c = np.zeros((N, q))
        per = max(1, CHUNK // max(1, 2 * K * N))
        for first in range(0, q, per):
            dims = range(first, min(q, first + per))
            centres = np.zeros((2 * len(dims), q))
            for k, j in enumerate(dims):
                centres[2 * k, j], centres[2 * k + 1, j] = SHIFT, -SHIFT
            shifted = curvature_at(centres)
            if not np.all(np.isfinite(shifted)):
                raise np.linalg.LinAlgError(
                    "the integrand's curvature near an item's mode is not "
                    "finite"
                )
            ld = _log_det(shifted.reshape(-1, q, q)).reshape(N, -1)
            for k, j in enumerate(dims):
                c[:, j] = (
                    -0.5 * (ld[:, 2 * k] - ld[:, 2 * k + 1]) / (2.0 * SHIFT)
                )
    return LaplaceNodes(b=b, mode=mode, root=root, c=c, curvature=A)


def log_integrals(
    theta: dict,
    model: Any,
    arrays: Any,
    nodes: LaplaceNodes,
    t: npt.NDArray,
    d: npt.NDArray,
    readings: bool,
    upper: npt.NDArray | None = None,
) -> Any:
    """Each item's log integral by Laplace's approximation on its placed
    stencil (module docstring): differentiable in ``theta`` by autograd,
    with the approximation's own gradient at the parameters the stencil
    was placed for. ``nan`` where an item's curvature is not positive
    definite."""
    q = nodes.mode.shape[1]
    _, Cg, Ch = stencil(q)
    G = _values(theta, model, arrays, nodes.b, t, d, readings, upper)
    gamma = anp.dot(G, Cg.T)
    second = anp.dot(G, Ch.T)
    # Cholesky factor of A = -g'' and the forward substitutions of gamma
    # and c, written out so autograd differentiates them to any order
    L: list[list[Any]] = [[None] * q for _ in range(q)]
    log_det: Any = 0.0
    for r in range(q):
        for col in range(r + 1):
            s = -second[:, r * q + col]
            for k in range(col):
                s = s - L[r][k] * L[col][k]
            if col == r:
                L[r][r] = anp.sqrt(s)
                log_det = log_det + 2.0 * anp.log(L[r][r])
            else:
                L[r][col] = s / L[col][col]
    y: list[Any] = []
    yc: list[Any] = []
    for r in range(q):
        acc, acc_c = gamma[:, r], nodes.c[:, r]
        for k in range(r):
            acc = acc - L[r][k] * y[k]
            acc_c = acc_c - L[r][k] * yc[k]
        y.append(acc / L[r][r])
        yc.append(acc_c / L[r][r])
    moved: Any = 0.0
    for r in range(q):
        moved = moved + (0.5 * y[r] + yc[r]) * y[r]
    return (
        G[:, 0]
        + moved
        - 0.5 * log_det
        + 0.5 * q * np.log(2.0 * np.pi)
        + nodes.log_volume
    )


def sigma_points(nodes: LaplaceNodes) -> tuple[npt.NDArray, npt.NDArray]:
    """Points ``(N, 2 q, q)`` and weights ``(2 q,)`` whose weighted mean
    and covariance are the approximation's normal's (:meth:`LaplaceNodes.
    normal`): the mode a ``sqrt(q)`` step each way along each column of
    its factor. They give the mean and variance of a function of the
    varying parameters exactly where it is linear in them."""
    mean, factor = nodes.normal()
    q = mean.shape[1]
    steps = np.sqrt(q) * np.concatenate([np.eye(q), -np.eye(q)])
    return _points(mean, factor, steps), np.full(2 * q, 0.5 / q)
