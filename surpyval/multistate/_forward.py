r"""The Kolmogorov forward equations of a Markov multi-state model with
smooth transition hazards, solved by product integration.

With :math:`Q(t)` the matrix of the transition hazards (off the diagonal,
the hazard of each transition; on it, minus the sum of its row), the
transition probabilities solve

.. math::
    \frac{\partial P(s, t)}{\partial t} = P(s, t)\, Q(t), \qquad
    P(s, s) = I.

The solution is the product integral of :math:`Q` over :math:`(s, t]`.
Over a short interval :math:`(a, b]` of length :math:`\delta` it is
approximated by the matrix exponential of the fourth-order Magnus
expansion,

.. math::
    \exp\left\{A(b) - A(a) + \frac{\sqrt{3}}{12} \delta^2
    \left[Q(c_1), Q(c_2)\right]\right\},

:math:`A(b) - A(a)` the matrix of the transitions' cumulative hazard
increments (the integral of :math:`Q`, exactly), :math:`c_1, c_2` the
two Gauss-Legendre points of the interval and :math:`[X, Y] = XY - YX`.
It is exact where the hazards are constant (or proportional to each
other) on the interval, and otherwise in error by
:math:`O(\delta^5)`. Working with cumulative hazards keeps every step
finite where a hazard is not (a Weibull with a shape below 1, at 0),
and the Gauss points are inside the interval.

The grid is adaptive. Each interval is compared with its two halves; an
interval where they differ by more than its share of the tolerance (in
the maximum absolute row sum) is halved, until every interval passes.
The halves' product is the one used, whose error is about a fifteenth
of that difference (the local error falls 32-fold as the interval
halves). As the matrices are (to that accuracy) stochastic, errors do
not grow as they are carried forward, so the error of each row of
:math:`P(s, t)` is below the tolerance.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.linalg import expm

from surpyval.multistate._core import with_diagonal

#: The default accuracy of :math:`P(s, t)`: the largest absolute error of
#: a row (the sum of the absolute errors of its elements).
DEFAULT_TOL = 1e-8
#: The most grid nodes the refinement adds before it stops (with a
#: warning).
MAX_NODES = 2**16
# A cumulative hazard increment above this is a certain transition: the
# probability of staying is exp(-700), below the smallest double.
_CERTAIN = 700.0


def _generators(
    dH: npt.NDArray,
    frm: npt.NDArray,
    to: npt.NDArray,
    k: int,
    cap: float = _CERTAIN,
) -> npt.NDArray:
    """``(m, K, K)``: the increments ``dH`` (``(m, n_transitions)``) as
    the matrices of the transitions, minus their row sums on the
    diagonal."""
    A = np.zeros((dH.shape[0], k, k))
    A[:, frm, to] = np.clip(dH, 0.0, cap)
    return with_diagonal(A)


Hazards = Callable[[npt.NDArray, bool], npt.NDArray]
_GAUSS = np.sqrt(3.0) / 6.0
_MAGNUS = np.sqrt(3.0) / 12.0


def _checked(
    cumhaz: Hazards, t: npt.NDArray, rate: bool = False
) -> npt.NDArray:
    H = np.asarray(cumhaz(t, rate), dtype=float)
    if np.any(np.isnan(H)):
        bad = t[np.any(np.isnan(H), axis=1)]
        raise ValueError(
            "A transition's cumulative hazard is not defined (NaN) at "
            f"t = {bad[0]:g}, so the transition probabilities cannot be "
            "found there: predict from a start 's' inside the support of "
            "every transition's distribution."
        )
    return H


def _step(
    cumhaz: Hazards,
    a: npt.NDArray,
    b: npt.NDArray,
    Ha: npt.NDArray,
    Hb: npt.NDArray,
    frm: npt.NDArray,
    to: npt.NDArray,
    k: int,
) -> npt.NDArray:
    """``(m, K, K)``: the fourth-order Magnus step over each ``(a, b]``."""
    d = b - a
    c = 0.5 * (a + b)
    q1 = _generators(
        _checked(cumhaz, c - _GAUSS * d, True), frm, to, k, np.inf
    )
    q2 = _generators(
        _checked(cumhaz, c + _GAUSS * d, True), frm, to, k, np.inf
    )
    # (P' = P Q multiplies on the right, so the commutator is [Q(c1),
    # Q(c2)], the transpose of the usual Y' = A Y's [A(c2), A(c1)])
    comm = q1 @ q2 - q2 @ q1
    omega = _generators(Hb - Ha, frm, to, k)
    omega = omega + (_MAGNUS * d**2)[:, None, None] * comm
    return expm(omega)


def _halves(
    cumhaz: Hazards,
    nodes: npt.NDArray,
    H: npt.NDArray,
    frm: npt.NDArray,
    to: npt.NDArray,
    k: int,
) -> tuple[npt.NDArray, npt.NDArray]:
    """For each interval between ``nodes``: its step over its two halves,
    and that step's difference from the one-step value (the maximum
    absolute row sum)."""
    a, b = nodes[:-1], nodes[1:]
    mid = 0.5 * (a + b)
    Hm = _checked(cumhaz, mid)
    one = _step(cumhaz, a, b, H[:-1], H[1:], frm, to, k)
    two = _step(cumhaz, a, mid, H[:-1], Hm, frm, to, k) @ _step(
        cumhaz, mid, b, Hm, H[1:], frm, to, k
    )
    diff = np.abs(one - two).sum(axis=2).max(axis=1)
    return two, diff


def _product(steps: npt.NDArray, k: int) -> npt.NDArray:
    """``(m, K, K)``: the running products of ``steps`` (``(m - 1, K,
    K)``), starting from the identity."""
    out = np.empty((steps.shape[0] + 1, k, k))
    P = np.eye(k)
    out[0] = P
    for j in range(steps.shape[0]):
        P = P @ steps[j]
        out[j + 1] = P
    return out


def _refine(
    cumhaz: Hazards,
    nodes: npt.NDArray,
    scale: float,
    tol: float,
    frm: npt.NDArray,
    to: npt.NDArray,
    k: int,
) -> npt.NDArray:
    """``nodes`` with every interval halved until its two halves agree
    with its one step to ``tol * length / scale``. Each interval's fate
    depends only on itself, so a node set holds the same points however
    far it reaches."""
    while True:
        H = _checked(cumhaz, nodes)
        _, diff = _halves(cumhaz, nodes, H, frm, to, k)
        bad = diff > tol * np.diff(nodes) / scale
        if not np.any(bad):
            return nodes
        a, b = nodes[:-1][bad], nodes[1:][bad]
        mid = 0.5 * (a + b)
        # (an interval can halve only so far in floating point)
        ok = (mid > a) & (mid < b)
        if not np.any(ok) or nodes.size + int(ok.sum()) > MAX_NODES:
            from surpyval.utils.warnings import caller_stacklevel

            worst = int(np.argmax(diff))
            warnings.warn(
                "The transition probabilities did not reach their accuracy "
                f"({tol:g} per row) within {MAX_NODES} grid points: the "
                f"largest step error is {float(diff[worst]):.3g}, between "
                f"t = {nodes[worst]:g} and {nodes[worst + 1]:g}. A hazard "
                "changes very fast there (a fitted shape far from 1, say); "
                "the probabilities nearby are less accurate.",
                UserWarning,
                stacklevel=caller_stacklevel(),
            )
            return nodes
        nodes = np.sort(np.concatenate([nodes, mid[ok]]))


@dataclass
class Plan:
    """Where the product integral is taken for :math:`P(s, t)` at some
    times: a grid from ``s`` (``nodes``), and for each time its last
    node at or before it and the nodes of the partial step from there
    (``tails``, ``None`` where the time is a node)."""

    nodes: npt.NDArray
    times: npt.NDArray
    last: npt.NDArray
    tails: list


def plan(
    cumhaz: Hazards,
    s: float,
    times: npt.NDArray,
    scale: float,
    frm: npt.NDArray,
    to: npt.NDArray,
    k: int,
    tol: float = DEFAULT_TOL,
    breaks: npt.ArrayLike = (),
) -> Plan:
    """
    The grid of :math:`P(s, t)` at ``times`` (unique, all at or after
    ``s``).

    ``cumhaz(t, False)`` gives the ``(len(t), n_transitions)`` cumulative
    hazards of the transitions (``frm[h]`` to ``to[h]``) at the times
    ``t``, and ``cumhaz(t, True)`` their hazards.

    The grid does not depend on the other times asked for, so neither
    does a probability: its first segment is ``(s, s + scale]`` and each
    later one doubles the reach (``(s + scale 2^{j-1}, s + scale
    2^j]``), each cut into 32 to start and refined interval by interval
    to ``tol * length / scale``; a time inside an interval is reached by
    a partial step from the node before it, refined the same way. So the
    error of a row of :math:`P(s, t)` is about ``tol`` per ``scale`` of
    time. ``breaks`` are times where a hazard may jump or be infinite
    (the start of a distribution's support), made nodes: the error
    estimate holds only where the hazards are smooth in an interval.
    """
    times = np.asarray(times, dtype=float)
    end = float(times.max()) if times.size else s
    reach = [0.0, 1.0]
    while s + scale * reach[-1] < end:
        reach.append(2.0 * reach[-1])
    start = np.concatenate(
        [
            s + scale * np.linspace(lo, hi, 33)
            for lo, hi in zip(reach[:-1], reach[1:])
        ]
    )
    breaks = np.asarray(breaks, dtype=float).ravel()
    breaks = breaks[(breaks > s) & (breaks < start[-1])]
    nodes = _refine(
        cumhaz,
        np.unique(np.concatenate([start, breaks])),
        scale,
        tol,
        frm,
        to,
        k,
    )
    last = np.searchsorted(nodes, times, side="right") - 1
    tails: list = []
    for t, j in zip(times, last):
        if nodes[j] == t:
            tails.append(None)
        else:
            tails.append(
                _refine(
                    cumhaz, np.array([nodes[j], t]), scale, tol, frm, to, k
                )
            )
    return Plan(nodes=nodes, times=times, last=last, tails=tails)


def evaluate(
    cumhaz: Hazards,
    grid: Plan,
    frm: npt.NDArray,
    to: npt.NDArray,
    k: int,
) -> npt.NDArray:
    """``(len(times), K, K)``: :math:`P(s, t)` at the plan's times, by
    the product integral on its fixed grid. On a fixed grid the result is
    a smooth function of the hazards, so it can be differentiated
    numerically in their parameters."""
    out = np.empty((grid.times.size, k, k))
    if grid.times.size == 0:
        return out
    top = int(grid.last.max()) + 1
    nodes = grid.nodes[:top]
    if nodes.size > 1:
        H = _checked(cumhaz, nodes)
        steps, _ = _halves(cumhaz, nodes, H, frm, to, k)
        P = _product(steps, k)
    else:
        P = np.eye(k)[None]
    for n, (j, tail) in enumerate(zip(grid.last, grid.tails)):
        if tail is None:
            out[n] = P[j]
            continue
        H = _checked(cumhaz, tail)
        steps, _ = _halves(cumhaz, tail, H, frm, to, k)
        out[n] = P[j] @ _chain(steps)
    return out


def _chain(steps: npt.NDArray) -> npt.NDArray:
    out = steps[0]
    for step in steps[1:]:
        out = out @ step
    return out
