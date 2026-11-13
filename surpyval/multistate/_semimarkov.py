r"""The state occupation probabilities of a semi-Markov ("clock reset")
multi-state model, by numerical convolution of the renewal equations.

In a semi-Markov model the hazard of leaving a state depends on the time
spent in it, :math:`u`, so where a unit is at time :math:`t` depends on
when it made each earlier move. Write :math:`S_g(u) = \exp\{-\sum_k
H_{gk}(u)\}` for the probability of staying in state :math:`g` for
:math:`u`, and :math:`F_{gk}(u) = \int_0^u h_{gk}(v) S_g(v)\, dv` for
the probability of leaving it for :math:`k` within :math:`u`. With
:math:`q_j(t)` the density of the times units enter state :math:`j`, and
:math:`p_0` the states they are in (having just entered them) at time 0,

.. math::
    q_j(t) = \sum_g p_{0g} F'_{gj}(t)
        + \sum_g \int_0^t q_g(v) F'_{gj}(t - v)\, dv,
    \qquad
    p_j(t) = p_{0j} S_j(t) + \int_0^t q_j(v) S_j(t - v)\, dv,

a system of Volterra (renewal-type) equations. It is solved on a uniform
grid of step :math:`\delta`: the entries into each state are counted per
cell :math:`((m - 1)\delta, m\delta]` and taken as spread evenly over
the cell (exact where the entry density is constant on it, second order
in :math:`\delta` otherwise). The weights of the scheme are integrals of
:math:`F_{gk}` and :math:`S_g` over cells, taken by Gauss-Legendre
quadrature, on cells halving towards 0 in the first one, so that a
hazard that is infinite at 0 (a Weibull shape below 1) is integrated
accurately. Without a cycle in the transitions each state's entries are
one fast Fourier transform convolution per transition into it, in an
order where every transition goes forward; with a cycle (recovery) the
entries are built up a move at a time (each further move one
convolution) until the chance of more moves is negligible.

The step is halved until the occupation probabilities at the nodes over
``(0, scale]`` (``scale`` the longest time spent in a state in the
data), from every transient state, change by at most ``tol`` (1e-7);
the finer of the two grids is used. The scheme is second order where the
hazards are smooth; a hazard infinite at 0 makes it converge more slowly
near the start of each stay, and the tolerance is set below the accuracy
stated (about 1e-6 in a probability, 1e-6 of the time in an expected
length of stay; checked against closed forms) to allow for it.
Probabilities at times between the nodes, and the expected time spent in
each state, come from the same sums with the weights integrated to the
time asked for (no interpolation); a time within the first
``MIN_CELLS`` cells, where the entries into a state can be far from even
over a cell, gets a grid of its own of ``MIN_CELLS`` cells ending at
it.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

#: The default accuracy of an occupation probability over the span of the
#: data (the largest absolute change at the grid's nodes when the step is
#: halved).
DEFAULT_TOL = 1e-7
#: The fewest and the most grid cells over the time scale of the data.
MIN_CELLS = 64
MAX_CELLS = 2**17
#: The most moves between states a unit is followed through (with a
#: cycle in the transitions).
MAX_MOVES = 10_000
# The probability of further moves below which they are left out.
_NEGLIGIBLE = 1e-14
# The first cell is integrated on cells halving towards 0, this many.
_LEVELS = 30
_GL_X, _GL_W = np.polynomial.legendre.leggauss(8)
_X = 0.5 * (_GL_X + 1.0)
_W = 0.5 * _GL_W

Hazards = Callable[[npt.NDArray, bool], npt.NDArray]


def _checked(cumhaz: Hazards, u: npt.NDArray, rate: bool) -> npt.NDArray:
    with np.errstate(all="ignore"):
        H = np.asarray(cumhaz(u, rate), dtype=float).reshape(u.size, -1)
    if np.any(np.isnan(H)):
        bad = u[np.any(np.isnan(H), axis=1)]
        raise ValueError(
            "A transition's cumulative hazard is not defined (NaN) at a "
            f"time in the state of {bad[0]:g}, so the probabilities cannot "
            "be found: use distributions defined for every time from 0."
        )
    return H


def is_acyclic(frm: npt.NDArray, to: npt.NDArray, k: int) -> bool:
    """Whether the transitions have no cycle (no unit can return to a
    state it has left)."""
    return topological_order(frm, to, k) is not None


def topological_order(
    frm: npt.NDArray, to: npt.NDArray, k: int
) -> list | None:
    """The states in an order where every transition goes forward, or
    ``None`` where there is a cycle."""
    indeg = np.zeros(k, dtype=int)
    for j in to:
        indeg[j] += 1
    order: list = []
    ready = [g for g in range(k) if indeg[g] == 0]
    while ready:
        g = ready.pop(0)
        order.append(g)
        for f, j in zip(frm, to):
            if f == g:
                indeg[j] -= 1
                if indeg[j] == 0:
                    ready.append(j)
    return order if len(order) == k else None


@dataclass
class Weights:
    """The integrals over each cell ``c`` of a grid of step ``h``
    (``(c h, (c + 1) h]``, ``c = 0, ..., N - 1``) that the scheme uses:
    ``dF`` (``(N, T)``) of each transition's density
    :math:`h_{gk} S_g`, ``dIb`` of that density times the distance to
    the cell's end, ``dJ`` (``(N, K)``) of each state's :math:`S_g` and
    ``dKb`` of :math:`S_g` times the distance to the cell's end."""

    h: float
    dF: npt.NDArray
    dIb: npt.NDArray
    dJ: npt.NDArray
    dKb: npt.NDArray

    @property
    def n(self) -> int:
        return self.dF.shape[0]

    def transfer(self) -> npt.NDArray:
        """``(N, T)``: the probability that a unit entering a state evenly
        over a cell leaves it by each transition ``d`` cells later."""
        out = np.empty_like(self.dF)
        out[0] = self.dIb[0] / self.h
        out[1:] = self.dF[:-1] + (self.dIb[1:] - self.dIb[:-1]) / self.h
        return out


class Model:
    """The hazards of a semi-Markov model, as the solver takes them:
    ``cumhaz(u, rate)`` gives the ``(len(u), T)`` cumulative hazards
    (hazards with ``rate``) of the transitions ``frm[t] -> to[t]`` at the
    times ``u`` spent in their 'from' state."""

    def __init__(
        self, cumhaz: Hazards, frm: npt.NDArray, to: npt.NDArray, k: int
    ):
        self.cumhaz = cumhaz
        self.frm = np.asarray(frm, dtype=int)
        self.to = np.asarray(to, dtype=int)
        self.k = k
        self._from = np.zeros((self.frm.size, k))
        self._from[np.arange(self.frm.size), self.frm] = 1.0
        self.order = topological_order(self.frm, self.to, k)

    def survival(self, u: npt.NDArray) -> npt.NDArray:
        """``(len(u), K)``: the probability of staying in each state for
        ``u``."""
        H = _checked(self.cumhaz, u, False)
        with np.errstate(over="ignore"):
            return np.exp(-np.clip(H, 0.0, None) @ self._from)

    def _parts(self, u: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """The densities ``(len(u), T)`` and survivals ``(len(u), K)`` at
        ``u``."""
        S = self.survival(u)
        rate = _checked(self.cumhaz, u, True)
        Sf = S[:, self.frm]
        with np.errstate(invalid="ignore", over="ignore"):
            f = np.where(Sf > 0, np.clip(rate, 0.0, None) * Sf, 0.0)
        return f, S

    def weights(self, h: float, n: int) -> Weights:
        """The cell integrals of a grid of ``n`` cells of step ``h``."""
        t_count = self.frm.size
        # Cells 1, ..., n - 1: one Gauss-Legendre rule each
        a = h * np.arange(1, n)
        u = (a[:, None] + h * _X[None, :]).ravel()
        f, S = self._parts(u) if u.size else (np.zeros((0, t_count)),) * 2
        if not u.size:
            S = np.zeros((0, self.k))
        w = h * _W
        wb = h * h * _W * (1.0 - _X)
        f = f.reshape(n - 1, _X.size, t_count)
        S = S.reshape(n - 1, _X.size, self.k)
        dF = np.empty((n, t_count))
        dIb = np.empty((n, t_count))
        dJ = np.empty((n, self.k))
        dKb = np.empty((n, self.k))
        dF[1:] = np.einsum("q,cqt->ct", w, f)
        dIb[1:] = np.einsum("q,cqt->ct", wb, f)
        dJ[1:] = np.einsum("q,cqk->ck", w, S)
        dKb[1:] = np.einsum("q,cqk->ck", wb, S)
        dF[0], dIb[0], dJ[0], dKb[0] = self._first(h, h)
        return Weights(h=h, dF=dF, dIb=dIb, dJ=dJ, dKb=dKb)

    def _first(self, length: float, end: float) -> tuple:
        """The integrals over ``(0, length]`` (the distances measured to
        ``end``), on cells halving towards 0, where a hazard can be
        infinite."""
        hi = length * 0.5 ** np.arange(_LEVELS)
        lo = hi * 0.5
        span = hi - lo
        u = (lo[:, None] + span[:, None] * _X[None, :]).ravel()
        f, S = self._parts(u)
        w = (span[:, None] * _W[None, :]).ravel()
        wb = w * (end - u)
        dF = w @ f
        dIb = wb @ f
        dJ = w @ S
        dKb = wb @ S
        # (0, eps]: the share of each transition by its cumulative hazard
        eps = float(lo[-1])
        H = np.clip(_checked(self.cumhaz, np.array([eps]), False)[0], 0, None)
        Hg = H @ self._from
        Hf = Hg[self.frm]
        with np.errstate(invalid="ignore", divide="ignore"):
            share = np.where(Hf > 0, H / Hf, 0.0)
        tiny = share * -np.expm1(-Hf)
        dF = dF + tiny
        dIb = dIb + tiny * (end - 0.5 * eps)
        dJ = dJ + eps
        dKb = dKb + eps * (end - 0.5 * eps)
        return dF, dIb, dJ, dKb

    def partial(
        self, u0: npt.NDArray, length: float
    ) -> tuple[npt.NDArray, npt.NDArray]:
        r"""For each ``u0``: :math:`\int_{u_0}^{u_0 + \ell} S_g(v)\, dv`
        and :math:`\int_{u_0}^{u_0 + \ell} (u_0 + \ell - v) S_g(v)\,
        dv` (``(len(u0), K)`` each), on cells halving towards 0 where
        ``u0`` is 0."""
        out_j = np.zeros((u0.size, self.k))
        out_k = np.zeros((u0.size, self.k))
        if length <= 0 or u0.size == 0:
            return out_j, out_k
        zero = u0 == 0
        if np.any(zero):
            _, _, dJ, dKb = self._first(length, length)
            out_j[zero] = dJ
            out_k[zero] = dKb
        rest = ~zero
        if np.any(rest):
            u = (u0[rest][:, None] + length * _X[None, :]).ravel()
            S = self.survival(u).reshape(-1, _X.size, self.k)
            out_j[rest] = np.einsum("q,cqk->ck", length * _W, S)
            out_k[rest] = np.einsum(
                "q,cqk->ck", length * length * _W * (1.0 - _X), S
            )
        return out_j, out_k


def _fft_conv(a: npt.NDArray, b: npt.NDArray, n: int) -> npt.NDArray:
    """The first ``n`` terms of the convolution of ``a`` (``(m, ...)``)
    with ``b`` (``(l, ...)``, broadcasting with ``a``'s other axes) along
    axis 0."""
    size = 1
    while size < a.shape[0] + b.shape[0]:
        size *= 2
    fa = np.fft.rfft(a, size, axis=0)
    fb = np.fft.rfft(b, size, axis=0)
    return np.fft.irfft(fa * fb, size, axis=0)[:n]


def solve(model: Model, weights: Weights, p0: npt.NDArray) -> npt.NDArray:
    """``(N + 1, B, K)``: the probability of entering each state in each
    cell (row ``m`` for the cell ``((m - 1) h, m h]``, row 0 empty), from
    each starting distribution of the ``(B, K)`` ``p0`` (each unit having
    just entered its state at 0)."""
    p0 = np.atleast_2d(np.asarray(p0, dtype=float))
    n, k = weights.n, model.k
    frm, to = model.frm, model.to
    W = weights.transfer()
    b_count = p0.shape[0]
    # Entries straight from the start
    init = np.zeros((n, b_count, k))
    for t in range(frm.size):
        init[:, :, to[t]] += weights.dF[:, t][:, None] * p0[None, :, frm[t]]
    E = np.zeros((n + 1, b_count, k))
    if model.order is not None:
        # No cycle: each state's entries follow from those of the states
        # before it, by one convolution each.
        for j in model.order:
            total = init[:, :, j].copy()
            for t in np.flatnonzero(to == j).tolist():
                total += _fft_conv(E[1:, :, frm[t]], W[:, t][:, None], n)
            E[1:, :, j] = total
        return E
    # A cycle: add the entries of each further move (a generation) until
    # what is left is below rounding. Each generation is one convolution
    # per transition, and its total is the probability of making that
    # many moves within the grid's reach.
    size = 1
    while size < 2 * n:
        size *= 2
    # (time on the last, contiguous axis: the transforms are faster)
    W_hat = np.fft.rfft(np.ascontiguousarray(W.T), size, axis=-1)  # (T, f)
    generation = np.ascontiguousarray(init.transpose(1, 2, 0))  # (B, K, n)
    total = generation.copy()
    for _ in range(MAX_MOVES):
        g_hat = np.fft.rfft(generation, size, axis=-1)  # (B, K, f)
        n_hat = np.zeros_like(g_hat)
        for t in range(frm.size):
            n_hat[:, to[t]] += g_hat[:, frm[t]] * W_hat[t]
        generation = np.clip(
            np.fft.irfft(n_hat, size, axis=-1)[..., :n], 0, None
        )
        total += generation
        if float(generation.sum(axis=(1, 2)).max()) < _NEGLIGIBLE:
            break
    else:
        from surpyval.utils.warnings import caller_stacklevel

        warnings.warn(
            f"The semi-Markov probabilities count at most {MAX_MOVES} "
            "moves between states, and the chance of more within the time "
            f"asked for is {float(generation.sum(axis=(1, 2)).max()):.3g}: "
            "the units move between states very often for that time; the "
            "probabilities are less accurate.",
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    E[1:] = total.transpose(2, 0, 1)
    return E


@dataclass
class Solution:
    """A solved grid from the starting distributions ``p0`` (``(B,
    K)``)."""

    model: Model
    weights: Weights
    p0: npt.NDArray
    E: npt.NDArray

    def nodes(self, count: int) -> npt.NDArray:
        """``(count + 1, B, K)``: the occupation probabilities at the
        first ``count + 1`` nodes ``0, h, ..., count h``."""
        w = self.weights
        u = w.h * np.arange(count + 1)
        S = np.ones((count + 1, self.model.k))
        if count:
            S[1:] = self.model.survival(u[1:])
        out = self.p0[None, :, :] * S[:, None, :]
        V = w.dJ[:count] / w.h  # (count, K)
        conv = _fft_conv(self.E[1 : count + 1], V[:, None, :], count)
        out[1:] += conv
        return out

    def at(self, u: float) -> tuple[npt.NDArray, npt.NDArray]:
        """``(B, K)`` each: the occupation probabilities at time ``u`` and
        the expected time spent in each state over ``(0, u]``."""
        w = self.weights
        h = w.h
        k = self.model.k
        if u <= 0:
            return self.p0.copy(), np.zeros_like(self.p0)
        n = int(np.floor(u / h))
        theta = u - n * h
        if theta <= 1e-12 * h:
            n, theta = n, 0.0
        elif h - theta <= 1e-12 * h:
            n, theta = n + 1, 0.0
        need = n + (1 if theta > 0 else 0)
        if need > w.n:
            raise ValueError("internal: the grid does not reach the time")
        # node integrals J[c], K[c] at c h
        J = np.concatenate([np.zeros((1, k)), np.cumsum(w.dJ[:need], 0)])
        dK = h * J[:-1] + w.dKb[:need]
        K = np.concatenate([np.zeros((1, k)), np.cumsum(dK, 0)])
        # x_d = (d + theta) h, d = 0..n: J(x_d), K(x_d)
        d = np.arange(n + 1)
        P, R = self.model.partial(d * h, theta)
        Jx = J[: n + 1] + P
        Kx = K[: n + 1] + theta * J[: n + 1] + R
        S_u = self.model.survival(np.array([u]))[0]
        occ = self.p0 * S_u
        los = self.p0 * Jx[n]
        if n:
            # cells m = 1..n, d = n - m
            G = Jx[1:] - Jx[:-1]  # (n, K): over [x_d, x_{d+1}]
            L = Kx[1:] - Kx[:-1]
            Em = self.E[1 : n + 1]  # (n, B, K)
            occ = occ + np.einsum("mbk,mk->bk", Em, G[::-1]) / h
            los = los + np.einsum("mbk,mk->bk", Em, L[::-1]) / h
        if theta > 0:
            occ = occ + self.E[n + 1] * Jx[0] / h
            los = los + self.E[n + 1] * Kx[0] / h
            # Between nodes the entries of the last cell are taken as
            # even over the whole cell, which leaves the probabilities
            # adding up to 1 only to the scheme's accuracy (under 1e-6);
            # they are scaled to add up to 1 (and the lengths of stay to
            # u), as they do at the nodes.
            occ = occ / occ.sum(axis=1, keepdims=True)
            los = los * (u / los.sum(axis=1, keepdims=True))
        return occ, los


def cells_for(h: float, reach: float, scale: float) -> int:
    """The cells of a grid of step ``h`` reaching past ``reach``: the
    cells of ``scale`` times the smallest power of 2 that covers it, so
    the grid does not depend on the other times in a call."""
    base = int(round(scale / h))
    factor = 1
    while factor * scale < reach * (1 + 1e-12):
        factor *= 2
    return base * factor


def step(model: Model, scale: float, tol: float = DEFAULT_TOL) -> float:
    """The grid step: halved from ``scale / MIN_CELLS`` until the
    occupation probabilities at the nodes over ``(0, scale]``, from each
    state that can be left, change by at most ``tol``."""
    k = model.k
    starts = np.eye(k)[np.unique(model.frm)]
    cells = MIN_CELLS
    previous = None
    while True:
        h = scale / cells
        weights = model.weights(h, cells)
        sol = Solution(model, weights, starts, solve(model, weights, starts))
        occ = sol.nodes(cells)
        if previous is not None:
            change = float(np.max(np.abs(occ[::2] - previous)))
            if change <= tol:
                return h
            if 2 * cells > MAX_CELLS:
                from surpyval.utils.warnings import caller_stacklevel

                warnings.warn(
                    "The semi-Markov occupation probabilities did not reach "
                    f"their accuracy ({tol:g}) within {cells} grid cells "
                    f"over the data's time scale: the last halving of the "
                    f"step changed them by up to {change:.3g}. A hazard "
                    "changes very fast for its time scale (a fitted shape "
                    "far from 1, say); the probabilities are less accurate.",
                    UserWarning,
                    stacklevel=caller_stacklevel(),
                )
                return h
        previous = occ
        cells *= 2
