"""
The semi-parametric proportional odds model (Bennett 1983; Murphy, Rossini
and van der Vaart 1997).

The covariates multiply the *survival odds*, as in the parametric
``ProportionalOddsFitter`` (``LogisticPO``, ``WeibullPO``, ...), but the
baseline is left to the data:

.. math::
    \\frac{S(x \\mid Z)}{F(x \\mid Z)} = e^{\\beta' Z}\\,
    \\frac{S_0(x)}{F_0(x)}, \\qquad
    S(x \\mid Z) = \\frac{1}{1 + G_0(x)\\, e^{-\\beta' Z}},

with :math:`G_0 = F_0 / S_0` the baseline *failure* odds, an unspecified
non-decreasing step function that jumps at the event times. A positive
coefficient raises the odds of survival (a longer life), as in the
package's parametric PO models; R's ``timereg::prop.odds`` and
``mets::logitSurv``, and Murphy et al., model the odds of failure, so
their coefficients are the negatives of these.

Estimation
----------
``beta`` and the jumps :math:`g_k` of :math:`G_0` are estimated jointly by
nonparametric maximum likelihood (NPMLE), with the likelihood of Murphy et
al. (1997), in which the jump at an event time takes the place of the
density of the baseline odds:

.. math::
    \\ell(\\beta, g) = \\sum_{i} \\delta_i \\log\\big(g_{k(i)}
    e^{\\eta_i}\\big) - (1 + \\delta_i) \\log\\big(1 + G(x_i)
    e^{\\eta_i}\\big) + \\log\\big(1 + G(t_{l,i}) e^{\\eta_i}\\big),
    \\qquad \\eta_i = -\\beta' Z_i,

the last term for a row with a delayed entry :math:`t_{l,i}`. It is the
marginal likelihood of a Cox model with a unit-exponential (gamma, variance
1) frailty, which is the proportional odds model.

The coefficients maximise the *profile* likelihood
:math:`p\\ell(\\beta) = \\max_g \\ell(\\beta, g)`, by Newton-Raphson with
step-halving on :math:`p\\ell`. For each ``beta`` the baseline is solved
exactly, by Newton-Raphson on the log jumps (falling back to the
self-consistency (EM) step, which always increases the likelihood, where
the Newton step does not). Both Newton steps are exact and cost
:math:`O(m)` for :math:`m` event times however many there are: the
Hessian in the jumps is a diagonal plus a sum of rank-one blocks over
nested risk sets, which a change of variables (the jumps' partial sums)
turns into a tridiagonal matrix. The profile's gradient is the
likelihood's gradient in ``beta`` at the solved baseline (the envelope
theorem), and its Hessian the Schur complement of the full observed
information, so the outer iteration converges quadratically too.

The standard errors are those of the profile likelihood (Murphy and van
der Vaart 2000): the inverse of the negative Hessian of :math:`p\\ell` at
the maximum, which is the ``beta`` block of the inverse observed
information of the full NPMLE.

References
----------
Bennett, S. (1983), "Analysis of survival data by the proportional odds
model", Statistics in Medicine 2, 273-277.

Murphy, S. A., Rossini, A. J. and van der Vaart, A. W. (1997), "Maximum
likelihood estimation in the proportional odds model", Journal of the
American Statistical Association 92, 968-976.

Murphy, S. A. and van der Vaart, A. W. (2000), "On profile likelihood",
Journal of the American Statistical Association 95, 449-465.
"""

from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from scipy.linalg import LinAlgError, solveh_banded
from scipy.special import expit
from scipy.stats import norm

from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils import (
    _caller_stacklevel,
    check_covariate_rows,
    finite_covariate_mask,
    xcnt_handler,
)
from surpyval.utils.data_summary import data_summary
from surpyval.utils.linalg import wald_bound_on_support
from surpyval.utils.no_maximum import warn_no_maximum
from surpyval.utils.shapes import (
    check_paired_rows,
    covariate_rows,
    keeps_query_shape,
)

from .._aliasing import (
    aliased_columns,
    constant_columns,
    covariate_columns,
    expand,
    warn_aliased,
)
from .._concordance import ConcordanceMixin
from .._summary import coefficient_names, coefficient_repr, coefficient_table
from ..proportional_hazards.cox_ph import (
    _baseline_at_origin,
    _covariate_center,
)
from ..regression_data import (
    check_finite_event_times,
    design_matrix_from_df,
    prepare_Z,
    restore_covariate_meta,
    serialise_covariate_meta,
)

if TYPE_CHECKING:
    import pandas as pd

_INNER_MAX_ITER = 500
_OUTER_MAX_ITER = 50
_MAX_HALVINGS = 40


def _rev_cumsum(v: npt.NDArray) -> npt.NDArray:
    """Reverse cumulative sum along the first axis: ``out[k] =
    v[k:].sum(axis=0)``."""
    return np.cumsum(v[::-1], axis=0)[::-1]


class _POLikelihood:
    """The NPMLE log-likelihood of the proportional odds model, with its
    derivatives, for fixed data.

    The parameters are the failure-odds coefficients ``gamma`` (``-beta``;
    the linear predictor is ``eta = Z @ gamma``) and the log jumps ``u``
    of the baseline failure odds at the ``m`` distinct event times. Each
    row gives a term ``a * log(1 + exp(eta) * G(x))`` with ``a = -(1 +
    delta) n`` and, with a delayed entry, one ``n * log(1 + exp(eta) *
    G(tl))``; ``K`` is the number of jumps at or before the term's time.
    """

    def __init__(
        self,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
        Z: npt.NDArray,
    ) -> None:
        event = c == 0
        self.t = np.unique(x[event])
        self.m = self.t.size
        self.d = np.bincount(
            np.searchsorted(self.t, x[event]),
            weights=n[event],
            minlength=self.m,
        )
        delta = event.astype(float)
        # Jumps at or before x (a censored row at an event time has
        # survived it, as in the Kaplan-Meier estimate).
        K_x = np.searchsorted(self.t, x, side="right")
        K_tl = np.searchsorted(self.t, tl, side="right")
        entered = K_tl > 0
        self.K = np.concatenate([K_x, K_tl[entered]])
        self.a = np.concatenate([-(1.0 + delta) * n, n[entered]])
        self.Zt = np.concatenate([Z, Z[entered]])
        # The linear part sum_i n_i delta_i eta_i = gamma @ zsum.
        self.zsum = Z.T @ (n * delta)

    def _terms(self, u: npt.NDArray, gamma: npt.NDArray) -> tuple:
        g = np.exp(u)
        G = np.concatenate([[0.0], np.cumsum(g)])
        eta = self.Zt @ gamma
        with np.errstate(divide="ignore"):
            log_s = eta + np.log(G[self.K])
        log1ps = np.logaddexp(0.0, log_s)
        return g, eta, log_s, log1ps

    def value(self, u: npt.NDArray, gamma: npt.NDArray) -> float:
        """The log-likelihood."""
        _, _, _, log1ps = self._terms(u, gamma)
        return float(self.d @ u + self.zsum @ gamma + self.a @ log1ps)

    def derivatives(self, u: npt.NDArray, gamma: npt.NDArray) -> dict:
        """The log-likelihood with its gradient and the pieces of the
        negative Hessian (see :meth:`solve_uu`)."""
        g, eta, log_s, log1ps = self._terms(u, gamma)
        m = self.m
        p = expit(log_s)  # s / (1 + s)
        q = np.exp(eta - log1ps)  # e^eta / (1 + s)
        aq = self.a * q
        # r[k]: the sum of a q over the terms whose G includes jump k.
        r = _rev_cumsum(np.bincount(self.K, aq, m + 1))[1:]
        grad_u = self.d + g * r
        apq = self.a * p
        grad_g = self.zsum + self.Zt.T @ apq
        # -d2l/du2 = diag(D) - G U diag(w) U' G, U upper triangular ones.
        D = -g * r
        w = np.bincount(self.K, -aq * q, m + 1)[1:]
        # -d2l/du dgamma = -g * (U B), B the per-K sums of a q (1-p) Z.
        B = np.zeros((m + 1, self.Zt.shape[1]))
        np.add.at(B, self.K, (aq * (1.0 - p))[:, None] * self.Zt)
        N_ug = -g[:, None] * _rev_cumsum(B)[1:]
        N_gg = -(self.Zt * (apq * (1.0 - p))[:, None]).T @ self.Zt
        return {
            "value": float(
                self.d @ u + self.zsum @ gamma + self.a @ log1ps
            ),
            "g": g,
            "grad_u": grad_u,
            "grad_gamma": grad_g,
            "D": D,
            "w": w,
            "N_ug": N_ug,
            "N_gg": N_gg,
        }

    @staticmethod
    def solve_uu(der: dict, rhs: npt.NDArray) -> npt.NDArray:
        """Solve ``N_uu x = rhs`` for the negative Hessian in the log
        jumps, ``N_uu = diag(D) - G U diag(w) U' G`` (``G = diag(g)``,
        ``U`` the upper triangular matrix of ones), in O(m).

        ``N_uu = G U T U' G`` with ``T = U^-1 diag(D / g^2) U^-T -
        diag(w)``, which is tridiagonal: ``U^-1`` takes differences of
        neighbours. Raises ``LinAlgError`` where ``T``, and so ``N_uu``,
        is not positive definite.
        """
        g = der["g"]
        e = der["D"] / g**2
        e_next = np.append(e[1:], 0.0)
        diag = e + e_next - der["w"]
        ab = np.zeros((2, g.size))
        ab[0, 1:] = -e[1:]
        ab[1] = diag
        if not (np.all(np.isfinite(ab)) and np.all(np.isfinite(rhs))):
            raise LinAlgError("not finite")
        y = rhs / (g[:, None] if rhs.ndim == 2 else g)
        # U^-1 y: y[k] - y[k + 1].
        y = y - np.concatenate([y[1:], np.zeros_like(y[:1])])
        z = solveh_banded(ab, y, check_finite=False)
        # U^-T z: z[k] - z[k - 1].
        z = z - np.concatenate([np.zeros_like(z[:1]), z[:-1]])
        return z / (g[:, None] if rhs.ndim == 2 else g)

    def schur(self, der: dict) -> npt.NDArray:
        """The negative Hessian of the profile log-likelihood in
        ``gamma``: ``N_gg - N_gu N_uu^-1 N_ug``."""
        X = self.solve_uu(der, der["N_ug"])
        S = der["N_gg"] - der["N_ug"].T @ X
        return 0.5 * (S + S.T)

    def start(self, x: npt.NDArray, n: npt.NDArray) -> npt.NDArray:
        """Log jumps to start from: the Nelson-Aalen increments,
        ``d / (number at risk)`` (the odds and the cumulative hazard
        agree where both are small)."""
        at_risk = np.array([n[x >= t].sum() for t in self.t])
        return np.log(self.d / at_risk)


def _inner(
    lik: _POLikelihood, gamma: npt.NDArray, u: npt.NDArray, tol: float
) -> tuple[npt.NDArray, dict]:
    """The baseline that maximises the likelihood at ``gamma``: Newton's
    method on the log jumps ``u``, with step-halving, from ``u``. Where
    the Newton step is not an ascent direction (the Hessian is not
    negative definite away from the maximum) the self-consistency step is
    taken instead: each jump moved to ``d / (its risk-set sum)``, which
    has the sign of the gradient in every component (it is the EM step of
    the gamma-frailty representation without delayed entry).

    Stops when the Newton decrement ``sqrt(grad' N^-1 grad)`` is at most
    ``tol`` (or has stopped shrinking below ``1e-8``, at rounding)."""
    der = lik.derivatives(u, gamma)
    lam_prev = np.inf
    for _ in range(_INNER_MAX_ITER):
        grad = der["grad_u"]
        step = None
        try:
            step = lik.solve_uu(der, grad)
            lam2 = float(grad @ step)
            if not (np.isfinite(lam2) and lam2 > 0):
                step = None
        except (LinAlgError, ValueError):
            step = None
        if step is None:
            # The self-consistency step, where its denominator is
            # positive; a scaled gradient step elsewhere.
            denom = der["D"]
            with np.errstate(divide="ignore", invalid="ignore"):
                em = np.log(lik.d / denom)
            step = np.where(denom > 0, em, grad / lik.d)
            lam2 = float(np.max(np.abs(grad) / lik.d)) ** 2
            newton = False
        else:
            newton = True
        lam = np.sqrt(lam2)
        if newton and (lam <= tol or (lam <= 1e-8 and lam >= lam_prev / 2)):
            return u, der
        f = der["value"]
        slack = 1e3 * np.finfo(float).eps * max(abs(f), 1.0)
        t = 1.0
        for _ in range(_MAX_HALVINGS):
            new = u + t * step
            f_new = lik.value(new, gamma)
            if np.isfinite(f_new) and f_new >= f - slack:
                break
            t /= 2
        else:
            return u, der
        u = new
        der = lik.derivatives(u, gamma)
        lam_prev = lam if (newton and t == 1.0) else np.inf
    return u, der


def _profile_fit(
    lik: _POLikelihood,
    u: npt.NDArray,
    gamma: npt.NDArray,
    tol: float,
    max_iter: int,
) -> tuple:
    """Maximise the profile log-likelihood in ``gamma`` by Newton-Raphson
    with step-halving, solving the baseline (:func:`_inner`) at each
    ``gamma``. Returns ``(gamma, u, der, S, converged, iterations)``."""
    inner_tol = min(tol, 1e-10) * 1e-2
    u, der = _inner(lik, gamma, u, inner_tol)
    S = lik.schur(der)
    lam_prev = np.inf
    for it in range(1, max_iter + 1):
        grad = der["grad_gamma"]
        try:
            step = np.linalg.solve(S, grad)
        except np.linalg.LinAlgError:
            step = np.full(grad.shape, np.nan)
        lam2 = float(grad @ step)
        if not (np.all(np.isfinite(step)) and np.isfinite(lam2) and lam2 > 0):
            # The profile is not concave here: a gradient step, scaled by
            # the diagonal of the information where it is positive.
            scale = np.where(np.diag(S) > 0, np.diag(S), 1.0)
            step = grad / scale
            lam2 = float(grad @ step)
            newton = False
        else:
            newton = True
        lam = np.sqrt(max(lam2, 0.0))
        if newton and (lam <= tol or (lam <= 1e-8 and lam >= lam_prev / 2)):
            return gamma, u, der, S, True, it - 1
        f = der["value"]
        slack = 1e3 * np.finfo(float).eps * max(abs(f), 1.0)
        t = 1.0
        for _ in range(_MAX_HALVINGS):
            gamma_new = gamma + t * step
            u_new, der_new = _inner(lik, gamma_new, u, inner_tol)
            if np.isfinite(der_new["value"]) and der_new["value"] >= f - slack:
                break
            t /= 2
        else:
            return gamma, u, der, S, False, it
        gamma, u, der = gamma_new, u_new, der_new
        try:
            S = lik.schur(der)
        except (LinAlgError, ValueError):
            S = np.full((gamma.size, gamma.size), np.nan)
        lam_prev = lam if (newton and t == 1.0) else np.inf
    return gamma, u, der, S, False, max_iter
