r"""The joint NPMLE of the proportional hazards model for interval-censored
data: the engine behind :class:`IntervalCoxPH <.interval_cox.IntervalCoxPH_>`.

The data
--------
Each row's event is known to lie in ``(l, r]`` (``l = r`` for an exact
time, ``r = inf`` for a right-censored row, ``l = -inf`` for a
left-censored one), after an optional entry time ``tl``. The baseline is
left to the data: as in Turnbull's estimator, the likelihood depends on it
only through the probabilities of the pieces between the sorted endpoints,
and only the pieces that some row's support contains can carry mass. The
pieces are SurPyval's Turnbull pieces (``turnbull._bounds``,
``_censoring_supports``, ``_truncation_windows``); without truncation only
Turnbull's innermost intervals carry mass (Turnbull 1976; the argument
holds for any model in which a row's probability rises as the survival at
its left end rises and as that at its right end falls, as here), and with
left truncation every piece some support contains is kept. Runs of kept
pieces with no row's support or window starting or ending inside them are
merged into one *atom*: only the total there matters.

The likelihood
--------------
The baseline is written as cumulative hazard increments ``d_k >= 0`` over
the atoms, :math:`\Lambda_k = \sum_{j < k} d_j`, the survival after atom
``k`` being :math:`S_0 = e^{-\Lambda_{k+1}}`. The last atom's increment is
infinite (every support that contains it ends there, so its mass is all
that is left). Row ``i``, with support atoms ``[a_i, b_i]``, window from
atom ``w_i`` and :math:`r_i = e^{\beta' z_i}`, contributes

.. math::

    \ell_i = -r_i A_i + \log\big(1 - e^{-r_i D_i}\big), \qquad
    A_i = \Lambda_{a_i} - \Lambda_{w_i}, \quad
    D_i = \Lambda_{b_i + 1} - \Lambda_{a_i},

the log of :math:`[S(l)^{r} - S(r)^{r}] / S(t_l)^{r}`; a row whose support
reaches the last atom has :math:`D_i = \infty` and contributes
:math:`-r_i A_i`. For fixed ``beta`` this is concave in ``d`` (the first
term is linear and :math:`\log(1 - e^{-y})` is concave), and for a fixed
baseline it is concave in ``beta``.

The maximisation
----------------
``beta`` and ``d`` are maximised jointly by a projected Newton method
(Bertsekas 1982): Newton steps on the coefficients and the increments not
held at 0, with the full analytic Hessian, a Levenberg shift where it is
not negative definite, and step-halving on the likelihood, the increments
projected onto ``d >= 0``. An increment at 0 whose gradient points below
0 is held there. It stops when the Newton decrement of the free
parameters is below the tolerance and no held increment's gradient is
positive: the Karush-Kuhn-Tucker conditions of the constrained maximum.
The same routine with the coefficients held gives the profile likelihood
:math:`p\ell(\beta) = \max_d \ell(\beta, d)`, a concave maximisation.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import numpy.typing as npt

from surpyval.univariate.nonparametric.turnbull import turnbull_pieces

#: Newton iterations allowed for the joint fit and for a profile point.
MAX_ITER = 500
#: Halvings of a Newton step before the iteration gives up.
MAX_HALVINGS = 50


class Atoms(NamedTuple):
    """The atoms the baseline can put mass on, and each row's place.

    ``left`` and ``right`` are the atoms' time intervals ``(left, right]``
    (``left == right`` for an exact time); ``a``, ``b`` each row's support
    and ``w`` its window start, as atom indices; ``finite`` whether a
    row's support ends before the last atom (else its ``D`` is
    infinite)."""

    left: npt.NDArray
    right: npt.NDArray
    a: npt.NDArray
    b: npt.NDArray
    w: npt.NDArray
    finite: npt.NDArray
    #: Whether the likelihood has a maximum, from the data's structure
    #: alone (``_turnbull_npmle.npmle_existence``; ``"exists"`` without
    #: truncation).
    verdict: str


def find_atoms(
    xl: npt.NDArray,
    xr: npt.NDArray,
    c: npt.NDArray,
    tl: npt.NDArray,
    n: npt.NDArray,
) -> Atoms:
    """The atoms of the rows ``(xl, xr]`` (``xl == xr`` exact, ``xr =
    inf`` right censored, ``xl = -inf`` left censored) with entry times
    ``tl`` (``-inf`` for none), from SurPyval's Turnbull pieces."""
    bounds, lo, hi, w_lo, keep, verdict = turnbull_pieces(xl, xr, c, tl, n)
    M = bounds.size
    is_lo = np.zeros(M, dtype=bool)
    is_lo[lo] = True
    is_lo[w_lo] = True
    is_hi = np.zeros(M, dtype=bool)
    is_hi[hi] = True
    prev_kept = np.concatenate([[False], keep[:-1]])
    prev_hi = np.concatenate([[False], is_hi[:-1]])
    start = keep & (~prev_kept | is_lo | prev_hi)
    atom_of = np.cumsum(start) - 1
    kept = np.flatnonzero(keep)
    first = kept[start[kept]]
    # The last kept piece of each atom: the one before the next atom's
    # first, or the last kept piece.
    stops = np.append(first[1:], M)
    last = np.array(
        [kept[(kept >= f) & (kept < s)].max() for f, s in zip(first, stops)],
        dtype=np.int64,
    )
    left = bounds[first]
    right = np.append(bounds, np.inf)[last + 1]
    # Each row's first atom at or after its support start, and last at or
    # before its support end.
    kept_atom = atom_of[kept]
    a = kept_atom[np.searchsorted(kept, lo, side="left")]
    b = kept_atom[np.searchsorted(kept, hi, side="right") - 1]
    w_pos = np.searchsorted(kept, w_lo, side="left")
    w = np.where(
        w_pos < kept.size, kept_atom[np.minimum(w_pos, kept.size - 1)], 0
    )
    w = np.minimum(w, a)
    m = first.size
    finite = b < m - 1
    return Atoms(left, right, a, b, w, finite, verdict)


def _range_add(
    size: int, lo: npt.NDArray, hi: npt.NDArray, v: npt.ArrayLike
) -> npt.NDArray:
    """``out[j] = sum of v[i] over the rows with lo[i] <= j <= hi[i]``,
    ``v`` one value per row or a ``(rows, k)`` block (``(size, k)`` out);
    an empty range (``hi < lo``) adds nothing."""
    v = np.asarray(v, dtype=float)
    shape = (size + 1,) + v.shape[1:]
    diff = np.zeros(shape)
    ok = hi >= lo
    np.add.at(diff, lo[ok], v[ok])
    np.add.at(diff, hi[ok] + 1, -v[ok])
    return np.cumsum(diff[:size], axis=0)


class Derivatives(NamedTuple):
    value: float
    g_beta: npt.NDArray
    g_d: npt.NDArray
    H_bb: npt.NDArray
    H_bd: npt.NDArray
    H_dd: npt.NDArray


class ICLikelihood:
    """The log-likelihood :math:`\\ell(\\beta, d)` of the module notes, on
    centred covariates ``Z`` and counts ``n``."""

    def __init__(self, atoms: Atoms, Z: npt.NDArray, n: npt.NDArray):
        self.atoms = atoms
        self.Z = np.asarray(Z, dtype=float)
        self.n = np.asarray(n, dtype=float)
        self.m = atoms.left.size
        # The free increments: every atom but the last.
        self.k = self.m - 1

    def _cum(self, d: npt.NDArray) -> npt.NDArray:
        return np.concatenate([[0.0], np.cumsum(d), [np.inf]])

    def _parts(self, beta: npt.NDArray, d: npt.NDArray) -> tuple:
        at = self.atoms
        L = self._cum(d)
        eta = self.Z @ beta
        r = np.exp(eta)
        A = L[at.a] - L[at.w]
        D = np.where(at.finite, L[np.minimum(at.b + 1, self.m)] - L[at.a], 1.0)
        return r, A, D

    def value(self, beta: npt.NDArray, d: npt.NDArray) -> float:
        """The log-likelihood; ``-inf`` where a row's support has no
        probability."""
        r, A, D = self._parts(beta, d)
        y = r * D
        fin = self.atoms.finite
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            term = -r * A + np.where(fin, np.log(-np.expm1(-y)), 0.0)
            out = float(np.sum(self.n * term))
        return out if np.isfinite(out) else -np.inf

    def derivatives(self, beta: npt.NDArray, d: npt.NDArray) -> Derivatives:
        at = self.atoms
        n, Z, k = self.n, self.Z, self.k
        r, A, D = self._parts(beta, d)
        fin = at.finite
        y = np.where(fin, r * D, 1.0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            log_term = np.where(fin, np.log(-np.expm1(-y)), 0.0)
            g1 = np.where(fin, 1.0 / np.expm1(y), 0.0)  # g'(y)
        yg = np.where(fin, y * g1, 0.0)  # y g'(y), 1 as y -> 0
        # y^2 g''(y) = -(y g')(y g' + y)
        y2g2 = np.where(fin, -yg * (yg + y), 0.0)
        value = float(np.sum(n * (-r * A + log_term)))
        # d ell / d eta, and its second derivative
        s_eta = -r * A + yg
        h_eta = -r * A + yg + y2g2
        g_beta = Z.T @ (n * s_eta)
        H_bb = (Z * (n * h_eta)[:, None]).T @ Z
        # The ranges over the free increments (the last atom's is not one)
        a_hi = at.a - 1
        d_lo, d_hi = at.a, np.where(fin, at.b, at.a - 1)
        # d ell / d d_j: -r over A, r g'(y) over D
        coef_A = -n * r
        coef_D = n * r * g1
        g_d = _range_add(k, at.w, a_hi, coef_A) + _range_add(
            k, d_lo, d_hi, coef_D
        )
        # Cross: d^2 ell / d eta d d_j: -r over A, r(g' + y g'') over D,
        # y g'' = y2g2 / y
        with np.errstate(divide="ignore", invalid="ignore"):
            yg2 = np.where(fin, y2g2 / y, 0.0)
        cross_D = n * r * (g1 + yg2)
        H_bd = (
            _range_add(k, at.w, a_hi, coef_A[:, None] * Z)
            + _range_add(k, d_lo, d_hi, cross_D[:, None] * Z)
        ).T
        # d^2 ell / d d_j d d_l: r^2 g''(y) over the pairs in D
        w2 = np.where(fin, n * r * r * yg2 / np.where(fin, y, 1.0), 0.0)
        H_dd = np.zeros((k, k))
        if k:
            C = np.zeros((k, k))
            use = fin & (w2 != 0)
            np.add.at(C, (at.a[use], at.b[use]), w2[use])
            P = np.cumsum(C, axis=0)
            Q = np.cumsum(P[:, ::-1], axis=1)[:, ::-1]
            upper = np.triu(Q)
            H_dd = upper + np.triu(Q, 1).T
        return Derivatives(value, g_beta, g_d, H_bb, H_bd, H_dd)

    def start(self) -> npt.NDArray:
        """Equal masses on the atoms, as increments."""
        m = self.m
        S = 1.0 - np.arange(1, m) / m
        S_prev = 1.0 - np.arange(0, m - 1) / m
        return np.log(S_prev / S)


class Solution(NamedTuple):
    beta: npt.NDArray
    d: npt.NDArray
    value: float
    converged: bool
    n_iter: int
    der: Derivatives


def maximise(
    lik: ICLikelihood,
    beta: npt.NDArray,
    d: npt.NDArray,
    tol: float = 1e-10,
    fix_beta: bool = False,
    max_iter: int = MAX_ITER,
) -> Solution:
    """The constrained maximum of ``lik`` from ``(beta, d)`` by projected
    Newton (see the module notes); with ``fix_beta`` over ``d`` only."""
    p = beta.size
    beta = np.array(beta, dtype=float)
    d = np.maximum(np.array(d, dtype=float), 0.0)
    value = lik.value(beta, d)
    if not np.isfinite(value):
        d = lik.start()
        value = lik.value(beta, d)
    der = lik.derivatives(beta, d)
    for it in range(1, max_iter + 1):
        g = np.concatenate([der.g_beta, der.g_d])
        H = np.block([[der.H_bb, der.H_bd], [der.H_bd.T, der.H_dd]])
        free = np.ones(p + lik.k, dtype=bool)
        if fix_beta:
            free[:p] = False
        held = (d <= 0.0) & (der.g_d <= 0.0)
        free[p:] &= ~held
        idx = np.flatnonzero(free)
        gF = g[idx]
        HF = -H[np.ix_(idx, idx)]
        step = _newton_step(HF, gF)
        decrement = float(gF @ step)
        if decrement < tol:
            return Solution(beta, d, value, True, it - 1, der)
        theta = np.concatenate([beta, d])
        t = 1.0
        accepted = False
        for _ in range(MAX_HALVINGS):
            new = theta.copy()
            new[idx] += t * step
            new[p:] = np.maximum(new[p:], 0.0)
            new_value = lik.value(new[:p], new[p:])
            if new_value >= value + 1e-4 * t * min(decrement, 1.0) or (
                new_value > value and t < 1e-3
            ):
                accepted = True
                break
            t /= 2.0
        if not accepted:
            # No ascent along the Newton direction: try the gradient,
            # projected, before giving up.
            gstep = np.zeros_like(theta)
            gstep[idx] = gF / max(np.max(np.abs(np.diag(HF))), 1.0)
            t = 1.0
            for _ in range(MAX_HALVINGS):
                new = theta + t * gstep
                new[p:] = np.maximum(new[p:], 0.0)
                new_value = lik.value(new[:p], new[p:])
                if new_value > value:
                    accepted = True
                    break
                t /= 2.0
        if not accepted:
            return Solution(beta, d, value, decrement < 1e-6, it, der)
        beta, d, value = new[:p], new[p:], new_value
        der = lik.derivatives(beta, d)
    return Solution(beta, d, value, False, max_iter, der)


def _newton_step(HF: npt.NDArray, gF: npt.NDArray) -> npt.NDArray:
    """Solve ``HF s = gF`` for the ascent step, shifting ``HF`` by a
    multiple of its diagonal scale until it is positive definite."""
    if gF.size == 0:
        return gF
    scale = max(float(np.max(np.abs(np.diag(HF)))), 1e-12)
    mu = 0.0
    eye = np.eye(gF.size)
    for _ in range(60):
        try:
            L = np.linalg.cholesky(HF + mu * eye)
        except np.linalg.LinAlgError:
            mu = max(2.0 * mu, 1e-10 * scale)
            continue
        z = np.linalg.solve(L, gF)
        return np.linalg.solve(L.T, z)
    return gF / scale


def profile(
    lik: ICLikelihood, beta: npt.NDArray, d0: npt.NDArray, tol: float = 1e-12
) -> Solution:
    """The profile likelihood at ``beta``: the maximum over the baseline,
    from the increments ``d0``."""
    return maximise(lik, beta, d0, tol=tol, fix_beta=True)
