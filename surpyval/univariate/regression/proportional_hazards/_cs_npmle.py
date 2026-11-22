r"""The joint NPMLE of the cause-specific proportional hazards model for
data with any censoring and truncation: the engine behind ``CoxPH``'s
full-likelihood route (interval-, left-censored or right-truncated
rows), with one cause or several.

The model
---------
Cause :math:`k` has the hazard :math:`\lambda_k(t \mid Z) = \lambda_{0k}(t)
e^{\beta_k' Z}`, and the all-cause survival is :math:`S(t \mid Z) =
\exp(-\sum_k \Lambda_k(t \mid Z))`. Each baseline is a step function: an
increment :math:`d_{kj} \ge 0` on each *atom* :math:`j` (a run of
Turnbull pieces of the pooled data) the cause may use. Within an atom the
causes' hazards are taken in proportion, so a unit still event-free at
the atom's start, with probability :math:`S_j`, fails there from cause
:math:`k` with probability

.. math::

    S_j\, \frac{u_{kj}}{\lambda_j} \big(1 - e^{-\lambda_j}\big), \qquad
    u_{kj} = d_{kj} e^{\beta_k' Z},\quad \lambda_j = \sum_k u_{kj},

the step R's ``survfit`` takes for a multi-state ``coxph`` and SurPyval's
``CompetingRisksProportionalHazards`` takes: the causes' incidences sum to
:math:`1 - S` exactly. With one cause it is :math:`S_j (1 - e^{-u_j})`,
the grouped-data step of ``IntervalCoxPH``.

The last atom takes all the mass left (its total increment is
infinite); where several causes may use it, its split between them is
:math:`u_{kL} / \sum_{k'} u_{k'L}`, parameterised by finite weights
:math:`d_{kL}` with one held at 1.

The rows
--------
Every row is a sum of atoms' probabilities over its support ``[a, b]``
(atom indices):

- a failure from cause :math:`k` in :math:`(l, r]` (exact, left- or
  interval-censored) sums cause :math:`k`'s probability on the atoms in
  :math:`(l, r]`;
- a failure whose cause was not recorded, and a right-censored row whose
  window ends at a finite :math:`t_r`, sum every cause's (it telescopes
  to :math:`S(l) - S(r)`);
- a right-censored row with no right truncation gives :math:`S(l)`;
- a truncated row is divided by :math:`S(t_l) - S(t_r)`, the same sum
  over its window.

Where the mass may go
---------------------
Without truncation cause :math:`k` may use, of each innermost interval
of the rows that can hold it (its failures, the rows with no cause and
the right-censored rows; Hudgens, Satten and Longini 2001), the last
piece: where in the interval the failures lay is not known, and the
estimate's curves rise at its end (Turnbull's convention). An innermost
interval reaching to infinity is the units still working at the end of
follow-up, given to no cause. With one cause these are the Turnbull
intervals of the pooled data (``IntervalCoxPH``'s atoms); with no
covariates the likelihood depends only on each interval's total, so it
is Hudgens, Satten and Longini's NPMLE. With truncation every piece some
such row's support covers is kept (Turnbull 1976; Frydman 1994; Alioum
and Commenges 1996): mass outside every support only lowers the
likelihood.

The derivatives
---------------
A row's probability :math:`P` is a sum over atoms, and an increment at
atom :math:`j'` multiplies every later atom's term by
:math:`e^{-u_{k'j'}}`, so for :math:`j' < j''`

.. math::

    \frac{\partial^2 P}{\partial d_{k'j'} \partial d_{k''j''}} =
    -e^{\beta_{k'}' Z} \frac{\partial P}{\partial d_{k''j''}},

which depends on :math:`j'` only through the order: the Hessian over the
increments is assembled from per-atom sums, with dense outer products
only for the :math:`-\nabla P \nabla P' / P^2` term. The coefficients
enter through :math:`u = d e^{\beta' Z}`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import numpy.typing as npt

from surpyval.univariate.nonparametric._turnbull_npmle import (
    EXISTS,
    npmle_existence,
)
from surpyval.univariate.nonparametric.turnbull import (
    innermost_pieces,
    piece_coverage,
    windowed_pieces,
)

#: Newton iterations allowed for the joint fit and for a profile point.
MAX_ITER = 500
#: Halvings of a Newton step before the iteration gives up.
MAX_HALVINGS = 50
#: The epsilon-active set's largest eps (Bertsekas 1982).
EPS_ACTIVE = 1e-8
#: A search has stalled at the arithmetic's precision (and is at the
#: maximum) when its Newton decrement is below ``STALL_DECREMENT`` and its
#: last ``STALL_STEPS`` steps gained less than ``STALL_GAIN`` in all.
STALL_DECREMENT = 1e-8
STALL_STEPS = 5
STALL_GAIN = 1e-10
#: Below this the phi functions are evaluated by their series.
_SERIES = 0.1


# ---------------------------------------------------------------------------
# phi(l) = (1 - exp(-l)) / l and its derivatives
# ---------------------------------------------------------------------------
_N_SERIES = 14
_FACT = np.cumprod(np.arange(1.0, _N_SERIES + 2.0))  # (n + 1)!, n = 0, ...
#: phi(l) = sum_n c0[n] l^n, phi' = sum_n c1[n] l^n, phi'' = sum c2[n] l^n
_C0 = np.array([(-1.0) ** n / _FACT[n] for n in range(_N_SERIES)])
_C1 = np.array(
    [(-1.0) ** (n + 1) * (n + 1) / _FACT[n + 1] for n in range(_N_SERIES - 1)]
)
_C2 = np.array(
    [
        (-1.0) ** n * (n + 2) * (n + 1) / _FACT[n + 2]
        for n in range(_N_SERIES - 2)
    ]
)


def _horner(coef: npt.NDArray, x: npt.NDArray) -> npt.NDArray:
    out = np.full_like(x, coef[-1])
    for c in coef[-2::-1]:
        out = out * x + c
    return out


def _phi(lam: npt.NDArray, order: int = 2) -> tuple[Any, Any, Any]:
    """``phi``, and up to ``order`` of its derivatives (else ``None``), at
    ``lam >= 0`` (finite): the series below ``_SERIES``, the closed forms
    above it."""
    lam = np.asarray(lam, dtype=float)
    flat = lam.ravel()
    small = flat < _SERIES
    ls = flat[small]
    lb = flat[~small]
    out: list[npt.NDArray | None] = []
    with np.errstate(over="ignore", invalid="ignore"):
        E = np.exp(-lb)
        one = -np.expm1(-lb)
        closed = [
            lambda: one / lb,
            lambda: E / lb - one / lb**2,
            lambda: -E / lb - 2.0 * E / lb**2 + 2.0 * one / lb**3,
        ]
        for k, coef in enumerate((_C0, _C1, _C2)):
            if k > order:
                out.append(None)
                continue
            res = np.empty_like(flat)
            res[small] = _horner(coef, ls)
            res[~small] = closed[k]()
            out.append(res.reshape(lam.shape))
    return out[0], out[1], out[2]


# ---------------------------------------------------------------------------
# The structure: atoms, the causes allowed on each, and the terms
# ---------------------------------------------------------------------------
class Structure(NamedTuple):
    """The atoms and the likelihood's terms.

    ``left``/``right``: each atom's time interval ``(left, right]``;
    ``allowed`` (causes x atoms): where each cause may carry hazard;
    ``report_lower`` (causes x atoms): the lower end of the interval each
    allowed (cause, atom) stands for (its innermost interval without
    truncation; the atom's own start with it); ``ref``: the cause whose
    weight on the last atom is held at 1 (-1 for none: the last atom is
    the units not failed by the end, or has a single cause, whose share
    is 1). The terms: ``a``, ``b`` (support, atom indices), ``cause``
    (-1: every cause), ``coef`` (the count, negative for a
    truncation denominator), ``row`` (the data row it belongs to)."""

    left: npt.NDArray
    right: npt.NDArray
    allowed: npt.NDArray
    report_lower: npt.NDArray
    ref: int
    a: npt.NDArray
    b: npt.NDArray
    cause: npt.NDArray
    coef: npt.NDArray
    row: npt.NDArray
    verdict: str
    #: The largest right-truncation time where every row is right
    #: truncated (the mass after it is not identified), else ``inf``.
    tau: float


def build(
    xl: npt.NDArray,
    xr: npt.NDArray,
    cause: npt.NDArray,
    n: npt.NDArray,
    tl: npt.NDArray,
    tr: npt.NDArray,
    n_causes: int,
) -> Structure:
    """The structure of rows ``(xl, xr]`` (``xl == xr`` exact, ``xr =
    inf`` right censored, ``xl = -inf`` left censored) with causes
    ``cause`` (-1: none recorded; a right-censored row has none), counts
    ``n`` and windows ``(tl, tr]`` (``-inf``/``inf`` for none)."""
    xl = np.asarray(xl, dtype=float)
    xr = np.asarray(xr, dtype=float)
    cause = np.asarray(cause, dtype=np.int64)
    n = np.asarray(n, dtype=float)
    tl = np.asarray(tl, dtype=float)
    tr = np.asarray(tr, dtype=float)
    N = xl.size
    bounds, lo, hi, w_lo, w_hi = windowed_pieces(xl, xr, tl, tr)
    M = bounds.size
    piece_upper = np.append(bounds[1:], np.inf)
    truncated = bool(np.isfinite(tl).any() or np.isfinite(tr).any())
    verdict = EXISTS
    if truncated:
        verdict = npmle_existence(lo, hi, w_lo, w_hi, n, M)[0]
    # A row with no cause whose support reaches the end (right censored,
    # no right truncation) holds no cause's hazard: S(l).
    to_end = (cause < 0) & (hi == M - 1)

    allowed_p = np.zeros((n_causes, M), dtype=bool)
    lower_p = np.full((n_causes, M), np.nan)
    for k in range(n_causes):
        holds = (cause == k) | (cause < 0)
        if truncated:
            cover = piece_coverage(lo[holds], hi[holds], M) > 0
            cover &= piece_upper < np.inf
            # Only where a failure of cause k (or with no cause) could be.
            fails = (cause == k) | ((cause < 0) & ~to_end)
            cover &= piece_coverage(lo[fails], hi[fails], M) > 0
            allowed_p[k] = cover
            lower_p[k, cover] = bounds[cover]
            continue
        pieces = np.flatnonzero(innermost_pieces(lo[holds], hi[holds], M))
        if pieces.size == 0:
            continue
        starts = np.zeros(M, dtype=bool)
        starts[lo[holds]] = True
        clique = np.cumsum(starts[pieces]) - 1
        keep = clique >= 0
        pieces, clique = pieces[keep], clique[keep]
        uniq = np.unique(clique)
        first = pieces[np.searchsorted(clique, uniq, side="left")]
        last = pieces[np.searchsorted(clique, uniq, side="right") - 1]
        fin = piece_upper[last] < np.inf
        # Only intervals some failure of cause k (or with no cause) covers
        fails = (cause == k) | ((cause < 0) & ~to_end)
        cov = piece_coverage(lo[fails], hi[fails], M) > 0
        fin &= cov[last]
        allowed_p[k, last[fin]] = True
        lower_p[k, last[fin]] = bounds[first[fin]]

    keep = np.asarray(allowed_p.any(axis=0))
    if np.any(to_end):
        keep[M - 1] = True
    kept = np.flatnonzero(keep)
    if kept.size == 0:
        raise ValueError("No failure to estimate a baseline from.")
    # Merge runs of kept pieces with no support or window starting or
    # ending inside, and the same causes allowed.
    is_lo = np.zeros(M, dtype=bool)
    is_lo[lo] = True
    is_lo[w_lo] = True
    is_hi = np.zeros(M, dtype=bool)
    is_hi[hi] = True
    is_hi[w_hi] = True
    start = np.zeros(M, dtype=bool)
    prev = -2
    for p in kept:
        start[p] = (
            p != prev + 1
            or is_lo[p]
            or is_hi[prev]
            or not np.array_equal(allowed_p[:, p], allowed_p[:, prev])
            or not truncated
        )
        prev = int(p)
    first_of = kept[start[kept]]
    atom_of_kept = np.cumsum(start[kept]) - 1
    m = first_of.size
    last_of = np.array(
        [kept[atom_of_kept == j].max() for j in range(m)], dtype=np.int64
    )
    left = bounds[first_of]
    right = piece_upper[last_of]
    allowed = allowed_p[:, first_of].copy()
    report_lower = np.where(allowed, lower_p[:, first_of], np.nan)
    if truncated:
        report_lower = np.where(allowed, left[None, :], np.nan)

    def first_atom(p: npt.NDArray) -> npt.NDArray:
        pos = np.searchsorted(kept, p, side="left")
        return np.where(
            pos < kept.size,
            atom_of_kept[np.minimum(pos, kept.size - 1)],
            m,
        )

    def last_atom(p: npt.NDArray) -> npt.NDArray:
        pos = np.searchsorted(kept, p, side="right") - 1
        return np.where(pos >= 0, atom_of_kept[np.maximum(pos, 0)], -1)

    a = first_atom(lo)
    b = last_atom(hi)
    b = np.where(to_end, m - 1, b)
    if np.any(a > b):
        raise ValueError(
            "A row's support holds no atom the baseline can put mass on; "
            "check the x, c and truncation inputs."
        )
    L = m - 1
    ref = -1
    on_last = np.flatnonzero(allowed[:, L])
    if right[L] == np.inf:
        allowed[:, L] = False
    elif on_last.size >= 1:
        # The cause with the most failure rows reaching the last atom.
        reach = [float(np.sum(n[(cause == k) & (b == L)])) for k in on_last]
        ref = int(on_last[int(np.argmax(reach))])

    terms_a = [a]
    terms_b = [b]
    terms_cause = [np.where(cause >= 0, cause, -1)]
    terms_coef = [n]
    terms_row = [np.arange(N)]
    trunc_rows = np.flatnonzero(np.isfinite(tl) | np.isfinite(tr))
    if trunc_rows.size:
        wa = first_atom(w_lo[trunc_rows])
        wb = np.where(
            w_hi[trunc_rows] >= M - 1, L, last_atom(w_hi[trunc_rows])
        )
        # A window holding every atom has probability 1 (an entry before
        # the first atom, no right truncation): no term.
        real = (wa > 0) | (wb < L)
        trunc_rows, wa, wb = trunc_rows[real], wa[real], wb[real]
        terms_a.append(wa)
        terms_b.append(wb)
        terms_cause.append(np.full(trunc_rows.size, -1))
        terms_coef.append(-n[trunc_rows])
        terms_row.append(trunc_rows)
    tau = float(np.max(tr)) if np.all(np.isfinite(tr)) else np.inf
    return Structure(
        left,
        right,
        allowed,
        report_lower,
        ref,
        np.concatenate(terms_a).astype(np.int64),
        np.concatenate(terms_b).astype(np.int64),
        np.concatenate(terms_cause).astype(np.int64),
        np.concatenate(terms_coef).astype(float),
        np.concatenate(terms_row).astype(np.int64),
        verdict,
        tau,
    )


# ---------------------------------------------------------------------------
# The likelihood
# ---------------------------------------------------------------------------
class Derivatives(NamedTuple):
    value: float
    grad: npt.NDArray
    hess: npt.NDArray


class Likelihood:
    """The log-likelihood over ``theta = (beta, d_free)``: ``beta`` the
    ``(K, p)`` coefficients on the centred covariates ``Z``, flattened by
    cause, and ``d_free`` the free increments (``free`` true, flattened
    cause by cause)."""

    def __init__(self, s: Structure, Z: npt.NDArray):
        self.s = s
        self.Z = np.asarray(Z, dtype=float)
        self.K, self.m = s.allowed.shape
        self.p = self.Z.shape[1]
        self.free = s.allowed.copy()
        if s.ref >= 0:
            self.free[s.ref, self.m - 1] = False
        self.n_free = int(self.free.sum())
        self.n_beta = self.K * self.p
        T = s.a.size
        j = np.arange(self.m)
        self.in_supp = (j[None, :] >= s.a[:, None]) & (
            j[None, :] <= s.b[:, None]
        )
        self.le_b = j[None, :] <= s.b[:, None]
        self.allc = s.cause < 0
        self.onehot = np.zeros((T, self.K))
        ok = ~self.allc
        self.onehot[np.flatnonzero(ok), s.cause[ok]] = 1.0
        self.Zt = self.Z[s.row]

    # -- parameters ------------------------------------------------------

    def split(self, theta: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        beta = theta[: self.n_beta].reshape(self.K, self.p)
        d = np.zeros((self.K, self.m))
        d[self.free] = theta[self.n_beta :]
        if self.s.ref >= 0:
            d[self.s.ref, self.m - 1] = 1.0
        return beta, d

    def join(self, beta: npt.NDArray, d: npt.NDArray) -> npt.NDArray:
        return np.concatenate([np.ravel(beta), d[self.free]])

    def start(self, beta: npt.NDArray | None = None) -> npt.NDArray:
        """Equal masses on the allowed (cause, atom) pairs, as
        increments."""
        K, m = self.K, self.m
        al = self.s.allowed
        count = int(al[:, : m - 1].sum()) + 1
        d = np.zeros((K, m))
        S = 1.0
        for j in range(m - 1):
            ks = np.flatnonzero(al[:, j])
            if ks.size == 0:
                continue
            pj = ks.size / count
            lam = -np.log1p(-pj / S)
            d[ks, j] = lam / ks.size
            S -= pj
        d[al[:, m - 1], m - 1] = 1.0
        if beta is None:
            beta = np.zeros((K, self.p))
        return self.join(beta, d)

    # -- evaluation ------------------------------------------------------

    def _core(self, beta: npt.NDArray, d: npt.NDArray, order: int = 2) -> dict:
        s = self.s
        T = s.a.size
        m = self.m
        L = m - 1
        rows = np.arange(T)
        with np.errstate(over="ignore", invalid="ignore"):
            r = np.exp(self.Zt @ beta.T)  # (T, K)
        u = r[:, :, None] * d[None, :, :]  # (T, K, m)
        lam = u.sum(axis=1)  # (T, m)
        lam_fin = lam.copy()
        lam_fin[:, L] = 0.0
        Lam = np.concatenate(
            [np.zeros((T, 1)), np.cumsum(lam_fin, axis=1)[:, :-1]], axis=1
        )
        rel = Lam - Lam[rows, s.a][:, None]
        with np.errstate(over="ignore", invalid="ignore"):
            St = np.where(self.in_supp, np.exp(-rel), 0.0)
        ph0, ph1, ph2 = _phi(lam_fin, order)
        E = np.exp(-lam_fin)
        one = -np.expm1(-lam_fin)
        uk = np.einsum("tkm,tk->tm", u, self.onehot)  # W for one-hot
        allc = self.allc[:, None]
        # The atom's own term
        inn = np.where(allc, St * one, St * uk * ph0)
        lamL = lam[:, L]
        with np.errstate(divide="ignore", invalid="ignore"):
            fL = np.where(lamL > 0, uk[:, L] / lamL, 0.0)
        inn[:, L] = np.where(self.allc, St[:, L], St[:, L] * fL)
        P = inn.sum(axis=1)
        rev = np.cumsum(inn[:, ::-1], axis=1)[:, ::-1]
        after = np.concatenate([rev[:, 1:], np.zeros((T, 1))], axis=1)
        after = np.where(
            np.arange(m)[None, :] < s.a[:, None], P[:, None], after
        )
        after = np.where(self.le_b, after, 0.0)
        return dict(
            r=r,
            u=u,
            lam=lam,
            lamL=lamL,
            fL=fL,
            Lam=Lam,
            St=St,
            ph=(ph0, ph1, ph2),
            E=E,
            uk=uk,
            inn=inn,
            P=P,
            after=after,
        )

    def value(self, theta: npt.NDArray) -> float:
        beta, d = self.split(theta)
        if np.any(d < 0):
            return -np.inf
        c = self._core(beta, d, order=0)
        P = c["P"]
        s = self.s
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.log(P) - c["Lam"][np.arange(s.a.size), s.a]
            out = float(np.sum(s.coef * terms))
        return out if np.isfinite(out) else -np.inf

    def derivatives(self, theta: npt.NDArray) -> Derivatives:
        s = self.s
        K, m, p = self.K, self.m, self.p
        L = m - 1
        beta, d = self.split(theta)
        c = self._core(beta, d)
        r, St, after, P = c["r"], c["St"], c["after"], c["P"]
        ph0, ph1, ph2 = c["ph"]
        E, uk = c["E"], c["uk"]
        lamL, fL = c["lamL"], c["fL"]
        T = s.a.size
        rows = np.arange(T)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = float(np.sum(s.coef * (np.log(P) - c["Lam"][rows, s.a])))
        ins = self.in_supp
        allc = self.allc[:, None, None]
        oh = self.onehot[:, :, None]  # (T, K, 1)
        # D1[t, k', j]: d(atom term)/d u_k'j
        D1 = np.where(
            allc,
            (St * E)[:, None, :],
            St[:, None, :] * (oh * ph0[:, None, :] + (uk * ph1)[:, None, :]),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            safe = np.where(lamL > 0, lamL, 1.0)
            D1L = np.where(
                self.allc[:, None],
                0.0,
                St[:, L][:, None]
                * (self.onehot - fL[:, None])
                / safe[:, None],
            )
        D1[:, :, L] = D1L
        D1 = np.where(ins[:, None, :], D1, 0.0)
        g = r[:, :, None] * (D1 - after[:, None, :])  # (T, K, m)
        g = np.where(self.le_b[:, None, :], g, 0.0)
        # D2[t, k', k'', j]
        delta = oh[:, :, None, :] + oh[:, None, :, :]  # (T,K,K,1)
        D2 = np.where(
            self.allc[:, None, None, None],
            -(St * E)[:, None, None, :],
            St[:, None, None, :]
            * (delta * ph1[:, None, None, :] + (uk * ph2)[:, None, None, :]),
        )
        D2L = np.where(
            self.allc[:, None, None],
            0.0,
            -St[:, L][:, None, None]
            * (delta[..., 0] - 2.0 * fL[:, None, None])
            / safe[:, None, None] ** 2,
        )
        D2[..., L] = D2L
        D2 = np.where(ins[:, None, None, :], D2, 0.0)
        Hloc = (
            r[:, :, None, None]
            * r[:, None, :, None]
            * (D2 + after[:, None, None, :])
        )
        Hloc = np.where(self.le_b[:, None, None, :], Hloc, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            w = s.coef / P
            v = s.coef / P**2
        # Gradient
        grad_d = np.einsum("t,tkm->km", w, g)
        h = np.einsum("km,tkm->tk", d, g)  # (T, K)
        Zt = self.Zt
        grad_b = np.einsum("t,tk,tq->kq", w, h, Zt)
        # dd Hessian: -sum r_k' g(k'', j'') above the diagonal
        C = -np.einsum("t,tk,tlm->klm", w, r, g)  # (K', K'', j'')
        Dg = np.einsum("t,tklm->klm", w, Hloc)  # (K', K'', j)
        jj = np.arange(m)
        upper = jj[:, None] < jj[None, :]
        Hfull = np.zeros((K, m, K, m))
        for k1 in range(K):
            for k2 in range(K):
                blk = np.where(upper, C[k1, k2][None, :], 0.0)
                blk = blk + np.where(upper.T, C[k2, k1][:, None], 0.0)
                blk[jj, jj] = Dg[k1, k2]
                Hfull[k1, :, k2, :] = blk
        G = g.reshape(T, K * m)
        Hdd = Hfull.reshape(K * m, K * m) - (G * v[:, None]).T @ G
        # beta-d cross
        Dcum = np.concatenate(
            [np.zeros((K, 1)), np.cumsum(d, axis=1)[:, :-1]], axis=1
        )
        dg = d[None, :, :] * g  # (T, K, m)
        Gtail = np.cumsum(dg[:, :, ::-1], axis=2)[:, :, ::-1]
        Gtail = np.concatenate([Gtail[:, :, 1:], np.zeros((T, K, 1))], axis=2)
        # inner[t, k', k'', j'']
        eye = np.eye(K)
        inner = (
            eye[None, :, :, None] * g[:, None, :, :]
            - r[:, :, None, None] * Dcum[None, :, None, :] * g[:, None, :, :]
            - r[:, None, :, None] * Gtail[:, :, None, :]
            + d[None, :, None, :] * Hloc
        )
        Hbd = np.einsum("t,tq,tkln->kqln", w, Zt, inner) - np.einsum(
            "t,tq,tk,tln->kqln", v, Zt, h, g
        )
        Hbd = Hbd.reshape(K * p, K * m)
        # beta-beta
        # A1[t, k', k''] = sum_j Dcum[k', j] d[k'', j] g[t, k'', j]
        A1 = np.einsum("km,tlm->tkl", Dcum, d[None, :, :] * g)
        X = (
            -r[:, :, None] * A1
            - r[:, None, :] * np.transpose(A1, (0, 2, 1))
            + np.einsum("km,lm,tklm->tkl", d, d, Hloc)
        )
        X = X + eye[None, :, :] * h[:, :, None]
        Hbb = np.einsum("t,tq,ts,tkl->kqls", w, Zt, Zt, X) - np.einsum(
            "t,tq,ts,tk,tl->kqls", v, Zt, Zt, h, h
        )
        Hbb = Hbb.reshape(K * p, K * p)
        free = self.free.ravel()
        Hbd = Hbd[:, free]
        Hdd = Hdd[np.ix_(free, free)]
        grad = np.concatenate([grad_b.ravel(), grad_d.ravel()[free]])
        hess = np.block([[Hbb, Hbd], [Hbd.T, Hdd]])
        return Derivatives(value, grad, hess)

    # -- predictions -----------------------------------------------------

    def masses(
        self, beta: npt.NDArray, d: npt.NDArray, Zc: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """``(mass, S)`` at the centred covariate rows ``Zc``: each
        cause's probability on each atom ``(rows, K, m)`` and the survival
        after each atom ``(rows, m)``."""
        return atom_masses(beta, d, Zc)


def atom_masses(
    beta: npt.NDArray, d: npt.NDArray, Zc: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """Each cause's probability on each atom and the survival after each
    atom, at the centred covariate rows ``Zc``, for coefficients ``beta``
    ``(K, p)`` and increments ``d`` ``(K, m)`` (the last atom's are its
    split's weights)."""
    Zc = np.atleast_2d(np.asarray(Zc, dtype=float))
    with np.errstate(over="ignore", invalid="ignore"):
        r = np.exp(Zc @ beta.T)
    u = r[:, :, None] * d[None, :, :]
    lam = u.sum(axis=1)
    m = d.shape[1]
    L = m - 1
    lam_fin = lam.copy()
    lam_fin[:, L] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        Lam = np.cumsum(lam_fin, axis=1)
        S_after = np.exp(-Lam)
        S_after[:, L] = 0.0
        S_before = np.concatenate(
            [np.ones((Zc.shape[0], 1)), np.exp(-Lam[:, :-1])], axis=1
        )
        drop = S_before - S_after
        share = np.where(lam[:, None, :] > 0, u / lam[:, None, :], 0.0)
    mass = share * drop[:, None, :]
    return mass, S_after


# ---------------------------------------------------------------------------
# The maximisation
# ---------------------------------------------------------------------------
class Solution(NamedTuple):
    theta: npt.NDArray
    value: float
    converged: bool
    n_iter: int
    der: Derivatives


def maximise(
    lik: Likelihood,
    theta: npt.NDArray,
    tol: float = 1e-10,
    fix_beta: bool = False,
    max_iter: int = MAX_ITER,
) -> Solution:
    """The constrained maximum of ``lik`` from ``theta`` by projected
    Newton, as ``_interval_npmle.maximise``: Newton steps on the
    coefficients and the increments not held at 0, a Levenberg shift
    where the Hessian is not negative definite, step-halving, and the
    increments projected onto ``d >= 0``; it stops at the
    Karush-Kuhn-Tucker conditions."""
    from ._interval_npmle import _newton_step

    nb = lik.n_beta
    theta = np.array(theta, dtype=float)
    theta[nb:] = np.maximum(theta[nb:], 0.0)
    value = lik.value(theta)
    if not np.isfinite(value):
        theta = lik.start(theta[:nb].reshape(lik.K, lik.p))
        value = lik.value(theta)
    der = lik.derivatives(theta)
    history: list[float] = []
    for it in range(1, max_iter + 1):
        g = der.grad
        H = der.hess
        free = np.ones(theta.size, dtype=bool)
        if fix_beta:
            free[:nb] = False
        # The epsilon-active set (Bertsekas 1982): an increment within
        # eps of 0 whose gradient points below 0 is held, and set to 0,
        # eps shrinking with the projected gradient, so increments on
        # their way to 0 are not projected one Newton step at a time.
        d, gd = theta[nb:], g[nb:]
        curv = np.maximum(-np.diag(H)[nb:], 1e-12)
        width = float(np.linalg.norm(d - np.maximum(d + gd / curv, 0.0)))
        eps = min(EPS_ACTIVE, width)
        held = (d <= eps) & (gd <= 0.0)
        free[nb:] &= ~held
        idx = np.flatnonzero(free)
        gF = g[idx]
        HF = -H[np.ix_(idx, idx)]
        if not (np.all(np.isfinite(gF)) and np.all(np.isfinite(HF))):
            return Solution(theta, value, False, it, der)
        step = _newton_step(HF, gF)
        decrement = float(gF @ step)
        zeroed = held & (d > 0.0)
        # The held increments above 0 take a projected, diagonally scaled
        # gradient step (to 0 or towards it).
        g_step = gd / curv
        if decrement < tol and not zeroed.any():
            return Solution(theta, value, True, it - 1, der)
        # Stalled at the arithmetic's precision: the active set swaps an
        # increment or two at 0 each step, each gaining a rounding's worth.
        history.append(value)
        if (
            decrement < STALL_DECREMENT
            and len(history) > STALL_STEPS
            and history[-1] - history[-1 - STALL_STEPS] < STALL_GAIN
        ):
            return Solution(theta, value, True, it - 1, der)
        # The step to the first free increment reaching 0, tried after the
        # full (projected) step, before halving.
        moving = idx >= nb
        down = moving & (step < 0)
        t_bound = (
            float(np.min(-theta[idx[down]] / step[down]))
            if down.any()
            else 1.0
        )
        trials = [1.0]
        if 0.0 < t_bound < 1.0:
            trials.append(t_bound)
        t = 1.0
        accepted = False
        for k in range(MAX_HALVINGS):
            t = trials[k] if k < len(trials) else t / 2.0
            new = theta.copy()
            new[idx] += t * step
            new[nb:][zeroed] = d[zeroed] + t * g_step[zeroed]
            new[nb:] = np.maximum(new[nb:], 0.0)
            new_value = lik.value(new)
            if new_value >= value + 1e-4 * t * min(decrement, 1.0) or (
                new_value > value and t < 1e-3
            ):
                accepted = True
                break
        if not accepted:
            gstep = np.zeros_like(theta)
            gstep[idx] = gF / max(np.max(np.abs(np.diag(HF))), 1.0)
            t = 1.0
            for _ in range(MAX_HALVINGS):
                new = theta + t * gstep
                new[nb:] = np.maximum(new[nb:], 0.0)
                new_value = lik.value(new)
                if new_value > value:
                    accepted = True
                    break
                t /= 2.0
        if not accepted:
            return Solution(theta, value, decrement < 1e-6, it, der)
        theta, value = new, new_value
        der = lik.derivatives(theta)
    return Solution(theta, value, False, max_iter, der)


def profile(
    lik: Likelihood, beta: npt.NDArray, theta0: npt.NDArray, tol: float = 1e-12
) -> Solution:
    """The profile likelihood at ``beta`` ``(K, p)``: the maximum over the
    baselines, from the increments of ``theta0``."""
    theta = np.array(theta0, dtype=float)
    theta[: lik.n_beta] = np.ravel(beta)
    return maximise(lik, theta, tol=tol, fix_beta=True)
