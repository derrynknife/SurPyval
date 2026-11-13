"""
The fitting engine of the mixed-effects Cox model: Gaussian random effects
on the log-hazard, fitted as R's ``coxme`` fits them (Ripatti and Palmgren
2000; Therneau, Grambsch and Pankratz 2003).

The hazard of row ``i`` is

.. math::
    h_i(t) = h_0(t) \\exp(X_i \\beta + Z_i b), \\qquad b \\sim N(0, \\Sigma),

with ``X`` the fixed covariates, ``Z`` the random-effects design (one
column per group of a term and effect) and ``Sigma`` block-diagonal, one
block per term. With ``Sigma = L L'`` and ``b = L a`` (``a ~ N(0, I)``,
the parameterisation of ``lme4``, which stays defined where a variance is
0), for given variance parameters:

* ``(beta, a)`` maximise the penalised partial likelihood
  ``PPL = PL(beta, L a) - a'a / 2`` (Newton's method with step-halving;
  ``PL`` is ``CoxPH``'s partial likelihood with the columns ``[X, Z]``);
* the integrated (marginal) partial likelihood is its Laplace
  approximation, with ``beta`` at the joint maximum,

  .. math::
      \\ell_I = PPL(\\hat\\beta, \\hat a)
      - \\tfrac12 \\log\\det(L' I_{bb} L + I),

  ``I_bb`` the partial likelihood's information in ``b``: this is
  ``coxme``'s "Integrated" log-likelihood, and with every variance 0 it is
  the Cox partial likelihood;

and the variance parameters maximise ``ell_I``. A term's covariance is
parameterised by the standard deviations of its effects (each ``>= 0``)
and, for a correlated term, the canonical partial correlations of its
correlation matrix (each in ``(-1, 1)``; for two effects, the correlation
itself).
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.optimize import minimize

from ..proportional_hazards.cox_likelihood import combined_generators
from ..proportional_hazards.cox_ph import CoxPH

# The search of the variance parameters: a standard deviation of a random
# effect on the log-hazard scale up to _SD_MAX (a hazard ratio of e^10
# between groups a standard deviation apart), correlations strictly inside
# (-1, 1).
SD_MAX = 10.0
_CORR_MAX = 1.0 - 1e-8
_SD_START = 0.3
# Newton on the penalised partial likelihood
_NEWTON_MAX_ITER = 200
_NEWTON_MAX_HALVINGS = 40
_NEWTON_TOL = 1e-11
# The central-difference step of the outer gradient
_FD_STEP = 1e-5


class TermDesign:
    """One random-effects term, as the engine uses it: each row's group
    (``inv`` into ``labels``) and the values of its effects (``V``, ``n x
    k``: a column of ones for an intercept, a covariate for a slope)."""

    def __init__(
        self,
        name: str,
        labels: npt.NDArray,
        inv: npt.NDArray,
        V: npt.NDArray,
        effects: "list[str]",
        correlated: bool,
        slope_columns: "list[int]",
        intercept: bool,
    ) -> None:
        self.name = name
        self.labels = labels
        self.inv = np.asarray(inv, dtype=int)
        self.V = np.asarray(V, dtype=float).reshape(self.inv.size, -1)
        self.effects = list(effects)
        self.correlated = bool(correlated) and self.k > 1
        self.slope_columns = list(slope_columns)
        self.intercept = bool(intercept)

    @property
    def G(self) -> int:
        return int(self.labels.shape[0])

    @property
    def k(self) -> int:
        return int(self.V.shape[1])

    @property
    def q(self) -> int:
        return self.G * self.k

    @property
    def n_params(self) -> int:
        k = self.k
        return k + (k * (k - 1) // 2 if self.correlated else 0)

    def design(self) -> npt.NDArray:
        """The term's columns of the random-effects design, effect by
        effect (column ``j G + g`` is effect ``j`` of group ``g``)."""
        n = self.inv.size
        out = np.zeros((n, self.q))
        rows = np.arange(n)
        for j in range(self.k):
            out[rows, j * self.G + self.inv] = self.V[:, j]
        return out

    def factor(self, psi: npt.NDArray) -> npt.NDArray:
        """The ``k x k`` lower-triangular ``L_k`` with ``L_k L_k'`` the
        effects' covariance, from the term's parameters ``psi``."""
        k = self.k
        sd = np.asarray(psi[:k], dtype=float)
        if not self.correlated:
            return np.diag(sd)
        return sd[:, None] * correlation_factor(psi[k:], k)

    def covariance(self, psi: npt.NDArray) -> npt.NDArray:
        L = self.factor(psi)
        return L @ L.T

    def bounds(self) -> "list[tuple[float, float]]":
        out = [(0.0, SD_MAX)] * self.k
        if self.correlated:
            out += [(-_CORR_MAX, _CORR_MAX)] * (self.n_params - self.k)
        return out

    def start(self) -> npt.NDArray:
        return np.r_[
            np.full(self.k, _SD_START), np.zeros(self.n_params - self.k)
        ]

    def parameter_labels(self) -> "list[tuple[str, str]]":
        """``(kind, effect)`` of each parameter: ``("sd", effect)``, then
        ``("corr", "a:b")`` (a canonical partial correlation, the
        correlation itself for two effects)."""
        out = [("sd", e) for e in self.effects]
        if self.correlated:
            for i in range(1, self.k):
                for j in range(i):
                    out.append(
                        (
                            "corr",
                            f"{self.effects[j]}:{self.effects[i]}",
                        )
                    )
        return out


def correlation_factor(z: npt.ArrayLike, k: int) -> npt.NDArray:
    """The Cholesky factor of a ``k x k`` correlation matrix from its
    canonical partial correlations ``z`` (row by row): each row has unit
    length, so any ``z`` in ``(-1, 1)`` gives a valid correlation matrix,
    and for ``k = 2`` the correlation is ``z``."""
    z = np.asarray(z, dtype=float)
    L = np.zeros((k, k))
    L[0, 0] = 1.0
    idx = 0
    for i in range(1, k):
        remaining = 1.0
        for j in range(i):
            L[i, j] = z[idx] * np.sqrt(remaining)
            remaining -= L[i, j] ** 2
            idx += 1
        L[i, i] = np.sqrt(max(remaining, 0.0))
    return L


class Solution:
    """The penalised fit at given variance parameters."""

    def __init__(self, **kw: Any) -> None:
        self.psi: npt.NDArray = kw["psi"]
        self.beta: npt.NDArray = kw["beta"]
        self.a: npt.NDArray = kw["a"]
        self.b: npt.NDArray = kw["b"]
        self.pl: float = kw["pl"]
        self.ppl: float = kw["ppl"]
        self.ilik: float = kw["ilik"]
        self.hessian: npt.NDArray = kw["hessian"]
        self.converged: bool = kw["converged"]


class MixedCoxEngine:
    """The penalised and integrated partial likelihoods of one data set.

    ``X`` are the fixed covariates (centred by the caller), ``strata`` each
    row's stratum code (or ``None``), ``terms`` the random-effects terms.
    The partial likelihood, its score and information come from
    ``CoxPH``'s generators on the columns ``[X, Z]`` (summed over strata),
    so ties, counts and left truncation are those of ``CoxPH``.
    """

    def __init__(
        self,
        x: npt.NDArray,
        X: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
        strata: "npt.NDArray | None",
        terms: "list[TermDesign]",
        tie_method: str,
    ) -> None:
        self.p = X.shape[1]
        self.terms = terms
        self.q = int(sum(t.q for t in terms))
        self.n_psi = int(sum(t.n_params for t in terms))
        Zr = (
            np.column_stack([t.design() for t in terms])
            if terms
            else np.zeros((x.size, 0))
        )
        D = np.column_stack([X, Zr])
        generator = CoxPH._resolve_func_generator(tie_method)
        if strata is None:
            args = [(x, D, c, n, tl)]
        else:
            args = [
                (x[m], D[m], c[m], n[m], tl[m])
                for m in (strata == s for s in np.unique(strata))
            ]
        self.neg_pl, self.jac_hess = combined_generators(generator, args)
        self.last: "Solution | None" = None
        self.not_converged = 0
        self.evaluations: dict[tuple, Solution] = {}

    # -- the variance parameters ---------------------------------------------

    def split(self, psi: npt.NDArray) -> "list[npt.NDArray]":
        out, at = [], 0
        for t in self.terms:
            out.append(np.asarray(psi[at : at + t.n_params], dtype=float))
            at += t.n_params
        return out

    def factor(self, psi: npt.NDArray) -> npt.NDArray:
        """The ``q x q`` ``L`` with ``b = L a``: block-diagonal, each
        term's block ``kron(L_k, I_G)``."""
        L = np.zeros((self.q, self.q))
        at = 0
        for t, part in zip(self.terms, self.split(psi)):
            block = np.kron(t.factor(part), np.eye(t.G))
            L[at : at + t.q, at : at + t.q] = block
            at += t.q
        return L

    def bounds(self) -> "list[tuple[float, float]]":
        return [b for t in self.terms for b in t.bounds()]

    def start(self) -> npt.NDArray:
        if not self.terms:
            return np.zeros(0)
        return np.concatenate([t.start() for t in self.terms])

    # -- the penalised fit ---------------------------------------------------

    def _warm(self, L: npt.NDArray, beta0: npt.NDArray) -> tuple:
        if self.last is None:
            return np.asarray(beta0, dtype=float), np.zeros(self.q)
        beta = self.last.beta.copy()
        b = self.last.b
        a = self.last.a.copy()
        at = 0
        for t in self.terms:
            block = L[at : at + t.q, at : at + t.q]
            diag = np.diag(block)
            if np.all(diag > 1e-8):
                a[at : at + t.q] = np.linalg.solve(block, b[at : at + t.q])
            at += t.q
        return beta, a

    def ppl(
        self, beta: npt.NDArray, a: npt.NDArray, L: npt.NDArray
    ) -> tuple[float, float]:
        """``(PPL, PL)`` at ``(beta, a)``."""
        with np.errstate(all="ignore"):
            pl = -float(self.neg_pl(np.r_[beta, L @ a]))
        return pl - 0.5 * float(a @ a), pl

    def derivatives(
        self, beta: npt.NDArray, a: npt.NDArray, L: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """The gradient of the PPL in ``(beta, a)``, its negative Hessian,
        and the partial likelihood's information in ``(beta, b)``."""
        p = self.p
        with np.errstate(all="ignore"):
            jac, hess = self.jac_hess(np.r_[beta, L @ a])
        jac = np.atleast_1d(np.asarray(jac, dtype=float))
        hess = np.atleast_2d(np.asarray(hess, dtype=float))
        grad = -jac.copy()
        grad[p:] = L.T @ grad[p:] - a
        T = np.zeros((p + self.q, p + self.q))
        T[:p, :p] = np.eye(p)
        T[p:, p:] = L
        H = T.T @ hess @ T
        H[p:, p:] += np.eye(self.q)
        return grad, (H + H.T) / 2, hess

    def solve(
        self, psi: npt.NDArray, beta0: "npt.NDArray | None" = None
    ) -> Solution:
        """The maximum of the PPL at the variance parameters ``psi`` and
        the Laplace-approximated integrated log-likelihood there."""
        psi = np.asarray(psi, dtype=float)
        key = tuple(np.round(psi, 15))
        if key in self.evaluations:
            return self.evaluations[key]
        L = self.factor(psi)
        beta, a = self._warm(L, np.zeros(self.p) if beta0 is None else beta0)
        value, pl = self.ppl(beta, a, L)
        if not np.isfinite(value):
            beta, a = (np.zeros(self.p) if beta0 is None else beta0), np.zeros(
                self.q
            )
            value, pl = self.ppl(beta, a, L)
        converged = False
        for _ in range(_NEWTON_MAX_ITER):
            grad, H, _ = self.derivatives(beta, a, L)
            try:
                step = np.linalg.solve(H, grad)
            except np.linalg.LinAlgError:
                break
            if not np.all(np.isfinite(step)):
                break
            decrement = float(grad @ step)
            if decrement <= _NEWTON_TOL**2 or np.max(np.abs(step)) < 1e-13:
                converged = True
                break
            t = 1.0
            for _ in range(_NEWTON_MAX_HALVINGS):
                nb = beta + t * step[: self.p]
                na = a + t * step[self.p :]
                new, new_pl = self.ppl(nb, na, L)
                if np.isfinite(new) and new >= value - 1e-12 * max(
                    1.0, abs(value)
                ):
                    break
                t /= 2
            else:
                break
            beta, a, value, pl = nb, na, new, new_pl
            if t == 1.0 and decrement <= 1e-20:
                converged = True
                break
        if not converged:
            self.not_converged += 1
        grad, H, _ = self.derivatives(beta, a, L)
        with np.errstate(all="ignore"):
            try:
                chol = np.linalg.cholesky(H[self.p :, self.p :])
                logdet = 2.0 * float(np.sum(np.log(np.diag(chol))))
            except np.linalg.LinAlgError:
                logdet = np.nan
        sol = Solution(
            psi=psi,
            beta=beta,
            a=a,
            b=L @ a,
            pl=pl,
            ppl=value,
            ilik=value - 0.5 * logdet,
            hessian=H,
            converged=converged,
        )
        if np.isfinite(sol.ilik):
            self.last = sol
        if len(self.evaluations) > 2000:
            self.evaluations.clear()
        self.evaluations[key] = sol
        return sol

    def ilik(self, psi: npt.NDArray) -> float:
        return self.solve(psi).ilik

    # -- the variance parameters' maximum -----------------------------------

    def maximise(
        self,
        start: npt.NDArray,
        free: "npt.NDArray | None" = None,
        fixed_values: "npt.NDArray | None" = None,
    ) -> tuple[npt.NDArray, float, bool]:
        """The variance parameters maximising the integrated likelihood,
        from ``start``; with ``free`` (a mask) only those are searched,
        the others held at ``fixed_values``. Returns ``(psi, ell_I,
        success)``."""
        bounds = self.bounds()
        full = np.asarray(start, dtype=float).copy()
        if free is None:
            free = np.ones(full.size, dtype=bool)
        if fixed_values is not None:
            full[~free] = np.asarray(fixed_values, dtype=float)[~free]
        if not free.any():
            return full, self.ilik(full), True
        lo = np.array([b[0] for b in bounds])[free]
        hi = np.array([b[1] for b in bounds])[free]

        def embed(v: npt.NDArray) -> npt.NDArray:
            out = full.copy()
            out[free] = v
            return out

        def objective(v: npt.NDArray) -> float:
            value = self.ilik(embed(np.clip(v, lo, hi)))
            return -value if np.isfinite(value) else 1e300

        def gradient(v: npt.NDArray) -> npt.NDArray:
            v = np.clip(v, lo, hi)
            g = np.zeros(v.size)
            for i in range(v.size):
                h = _FD_STEP * max(1.0, abs(v[i]))
                up, down = v.copy(), v.copy()
                up[i] = min(v[i] + h, hi[i])
                down[i] = max(v[i] - h, lo[i])
                g[i] = (objective(up) - objective(down)) / (up[i] - down[i])
            return g

        x0 = np.clip(full[free], lo, hi)
        res = minimize(
            objective,
            x0,
            jac=gradient,
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options={"ftol": 1e-15, "gtol": 1e-7, "maxiter": 500},
        )
        best = embed(np.clip(res.x, lo, hi))
        best_value = -float(res.fun)
        # A standard deviation at its lower limit is a boundary maximum:
        # snapped to exactly 0 where that is no worse (the maximum in a
        # direction the bounds stop is 0 itself).
        snapped = best.copy()
        sd_mask = np.zeros(full.size, dtype=bool)
        at = 0
        for t in self.terms:
            sd_mask[at : at + t.k] = True
            at += t.n_params
        snapped[sd_mask & free & (snapped < 1e-4)] = 0.0
        if not np.array_equal(snapped, best):
            value = self.ilik(snapped)
            if value >= best_value - 1e-9:
                best, best_value = snapped, value
        return best, best_value, bool(res.success)
