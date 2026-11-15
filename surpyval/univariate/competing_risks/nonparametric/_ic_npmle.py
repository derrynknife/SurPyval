"""The non-parametric maximum-likelihood estimate (NPMLE) of the cumulative
incidence functions of interval-censored competing-risks data.

Each row is known to have failed from cause ``k`` somewhere in its window
``(xl, xr]`` (an exact time is the window ``[x, x]``, a left-censored row
``(-inf, xr]``), to have failed in its window from a cause that was not
recorded, or to have been still working at ``xl`` (right censored,
``(xl, inf)``). With ``p_{kj}`` the mass of cause ``k`` on a set ``A_j``,
the likelihood is

    prod_i ( sum_{(k, j) covered by row i} p_{kj} ) ** n_i,

a row of cause ``k`` covering cause ``k``'s sets inside its window, a row
with no recorded cause every cause's, and a right-censored row every
cause's after ``xl`` and the mass left after the last finite set (the
units that had not failed by the end of follow-up). The masses add to 1.

Where the mass may go (Hudgens, Satten and Longini 2001): cause ``k``'s
mass lies on the innermost intervals (Turnbull's "maximal
intersections") of the windows of the rows that can hold it -- those of
cause ``k``, the rows with no recorded cause and the right-censored rows.
Moving mass into such an interval from a set around it raises no row's
probability less than another's, as in Turnbull's (1976) argument, and
the other causes' terms do not involve it. An innermost interval reaching
to infinity is covered by right-censored rows only, so its mass is the
mass left after follow-up, shared by every cause; it is one set,
``beyond``.

The likelihood is concave in the masses (a log of linear functions), so
its maximum value is unique, though the masses at it need not be (with
causes not recorded, how a shared set's mass splits between causes may be
undetermined). The maximum is found by the EM (self-consistency)
algorithm, each step ``p <- p * grad(l)(p) / N``, accelerated by SQUAREM
(Varadhan and Roland 2008), and stopped by the Karush-Kuhn-Tucker gap:
with ``g = grad(l) / N``, every ``g <= 1`` at the maximum, and the
log-likelihood is within ``N log(max g)`` of it, so ``max g - 1 <= tol``
is a certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import sparse

from surpyval.univariate.nonparametric.turnbull import (
    innermost_pieces,
    turnbull_pieces,
)

#: Cause index of a row with no recorded cause (a right-censored row, or a
#: failure whose cause was not recorded).
UNKNOWN = -1


@dataclass
class Structure:
    """The sets the mass may lie on and which rows cover which.

    ``cause``, ``lower`` and ``upper`` describe each atom (cause ``-1`` and
    ``lower = upper = inf`` for ``beyond``); ``A`` is the (unique rows x
    atoms) incidence matrix, ``n`` the unique rows' counts and ``row`` the
    unique row of each input row.
    """

    cause: npt.NDArray
    lower: npt.NDArray
    upper: npt.NDArray
    A: sparse.csr_matrix
    n: npt.NDArray
    row: npt.NDArray
    n_causes: int


def structure(
    xl: npt.NDArray,
    xr: npt.NDArray,
    cause: npt.NDArray,
    n: npt.NDArray,
    n_causes: int,
) -> Structure:
    """The atoms of the NPMLE and the incidence of the rows on them.

    ``xl == xr`` is an exact time; ``xr`` infinite a right-censored row
    (its cause must be ``UNKNOWN``); otherwise the window ``(xl, xr]``.
    """
    xl = np.asarray(xl, dtype=float)
    xr = np.asarray(xr, dtype=float)
    cause = np.asarray(cause, dtype=np.int64)
    n = np.asarray(n, dtype=float)
    N = xl.size
    c_tb = np.where(xl == xr, 0, 2)
    bounds, lo, hi, *_ = turnbull_pieces(xl, xr, c_tb, np.full(N, -np.inf), n)
    M = bounds.size
    # The time at each piece's upper end: piece j is (bounds[j],
    # bounds[j + 1]], the last reaching to infinity.
    piece_upper = np.concatenate([bounds[1:], [np.inf]])

    # Identical rows (support, cause) are one, with their counts summed.
    keys = np.column_stack([lo, hi, cause])
    uniq, row = np.unique(keys, axis=0, return_inverse=True)
    row = np.asarray(row).reshape(-1)
    u_lo, u_hi, u_cause = uniq[:, 0], uniq[:, 1], uniq[:, 2]
    u_n = np.bincount(row, weights=n, minlength=uniq.shape[0])
    right = piece_upper[u_hi] == np.inf
    u_right = right & (u_cause == UNKNOWN)

    a_cause, a_lo, a_hi = [], [], []
    for k in range(n_causes):
        holds = (u_cause == k) | (u_cause == UNKNOWN)
        if not np.any(u_cause == k) and not np.any(
            (u_cause == UNKNOWN) & ~u_right
        ):
            continue
        rows_lo, rows_hi = u_lo[holds], u_hi[holds]
        pieces = np.flatnonzero(innermost_pieces(rows_lo, rows_hi, M))
        if pieces.size == 0:
            continue
        starts = np.zeros(M, dtype=bool)
        starts[rows_lo] = True
        clique = np.cumsum(starts[pieces]) - 1
        # A clique's first piece is a start: drop any leading pieces
        # before the first start (none, by the construction of
        # ``_innermost``, but kept safe).
        keep = clique >= 0
        pieces, clique = pieces[keep], clique[keep]
        first = np.searchsorted(clique, np.unique(clique), side="left")
        last = np.searchsorted(clique, np.unique(clique), side="right") - 1
        c_lo, c_hi = pieces[first], pieces[last]
        finite = piece_upper[c_hi] < np.inf
        a_cause.append(np.full(int(finite.sum()), k))
        a_lo.append(c_lo[finite])
        a_hi.append(c_hi[finite])

    cause_a = np.concatenate(a_cause) if a_cause else np.empty(0, np.int64)
    lo_a = np.concatenate(a_lo) if a_lo else np.empty(0, np.int64)
    hi_a = np.concatenate(a_hi) if a_hi else np.empty(0, np.int64)
    lower = bounds[lo_a].astype(float)
    upper = piece_upper[hi_a].astype(float)

    # Incidence: row i covers atom (k, [a, b]) when it can hold cause k
    # and its support contains the atom's pieces.
    can_hold = (u_cause[:, None] == cause_a[None, :]) | (
        u_cause[:, None] == UNKNOWN
    )
    inside = (u_lo[:, None] <= lo_a[None, :]) & (
        hi_a[None, :] <= u_hi[:, None]
    )
    cover = can_hold & inside
    if np.any(u_right):
        cover = np.column_stack([cover, u_right])
        cause_a = np.concatenate([cause_a, [UNKNOWN]])
        lower = np.concatenate([lower, [np.inf]])
        upper = np.concatenate([upper, [np.inf]])
    A = sparse.csr_matrix(cover.astype(float))
    return Structure(cause_a, lower, upper, A, u_n, row, n_causes)


@dataclass
class EMResult:
    p: npt.NDArray
    loglik: float
    gap: float
    iterations: int
    converged: bool


def _loglik(A: sparse.csr_matrix, n: npt.NDArray, p: npt.NDArray) -> float:
    q = A @ p
    if np.any(q[n > 0] <= 0):
        return -np.inf
    return float(np.sum(n[n > 0] * np.log(q[n > 0])))


def fit_masses(
    s: Structure,
    n: npt.NDArray | None = None,
    tol: float = 1e-9,
    max_iter: int = 10000,
) -> EMResult:
    """The NPMLE's masses on the atoms of ``s`` (with counts ``n`` of the
    unique rows, ``s.n`` by default), by SQUAREM-accelerated EM."""
    A = s.A
    n = s.n if n is None else np.asarray(n, dtype=float)
    N = float(n.sum())
    AT = A.T.tocsr()
    used = n > 0
    # Atoms no row (with a positive count) covers get no mass.
    reach = np.asarray(AT @ used.astype(float)).ravel() > 0
    p = np.where(reach, np.asarray(AT @ n).ravel(), 0.0)
    p = p / p.sum()

    def gradient(p: npt.NDArray) -> npt.NDArray:
        q = A @ p
        w = np.where(used, n / np.where(q > 0, q, 1.0), 0.0)
        return np.asarray(AT @ w).ravel() / N

    def step(p: npt.NDArray) -> npt.NDArray:
        new = p * gradient(p)
        return new / new.sum()

    ll = _loglik(A, n, p)
    converged = False
    gap = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        g = gradient(p)
        gap = float(np.max(g[reach]) - 1.0)
        if gap <= tol:
            converged = True
            break
        p1 = step(p)
        p2 = step(p1)
        r = p1 - p
        v = p2 - p1 - r
        vv = float(v @ v)
        if vv > 0:
            alpha = -np.sqrt(float(r @ r) / vv)
            alpha = min(alpha, -1.0)
            trial = p - 2.0 * alpha * r + alpha * alpha * v
            trial = np.where(reach, np.maximum(trial, 0.0), 0.0)
            if trial.sum() > 0:
                trial = step(trial / trial.sum())
                ll_trial = _loglik(A, n, trial)
            else:
                ll_trial = -np.inf
        else:
            ll_trial = -np.inf
        ll2 = _loglik(A, n, p2)
        if ll_trial >= ll2:
            p, ll = trial, ll_trial
        else:
            p, ll = p2, ll2
    if converged:
        # The EM never sets a mass to 0, only shrinks it geometrically. At
        # the maximum an atom with g < 1 has no mass (the KKT conditions),
        # so the small masses left on such atoms are the iteration's, not
        # the estimate's: they are set to 0.
        g = gradient(p)
        dead = (g < 1.0 - 1e-6) & (p < 1e-6)
        if dead.any() and not dead.all():
            p = np.where(dead, 0.0, p)
            p = p / p.sum()
            ll = _loglik(A, n, p)
            g = gradient(p)
            gap = float(np.max(g[reach]) - 1.0)
    return EMResult(p, float(ll), gap, it, converged)


def cumulative(
    s: Structure, p: npt.NDArray, k: int, x: npt.NDArray
) -> npt.NDArray:
    """Cause ``k``'s cumulative incidence at ``x``: the mass of its atoms
    ending at or before ``x`` (the jump at an interval's upper end, the
    convention of ``Turnbull``)."""
    sel = s.cause == k
    up, mass = s.upper[sel], p[sel]
    order = np.argsort(up, kind="stable")
    up, cum = up[order], np.cumsum(mass[order])
    idx = np.searchsorted(up, x, side="right")
    out = np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)
    return np.where(np.isnan(x), np.nan, out)
