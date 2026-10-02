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
from surpyval.utils.data_summary import data_summary
from surpyval.utils.linalg import wald_bound_on_support
from surpyval.utils.no_maximum import warn_no_maximum, warn_unverified
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
from .._fit_skeleton import (
    LOG_MAX,
    baseline_at_origin_error,
    covariate_center,
)
from .._kinds import PROPORTIONAL_ODDS
from .._summary import coefficient_names, coefficient_repr, coefficient_table
from ..regression_data import (
    LinearPredictorMixin,
    design_matrix_from_df,
    restore_covariate_meta,
    semi_parametric_inputs,
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
        # A row with no event time in its window (tl, x] has terms that
        # cancel: the likelihood does not depend on it.
        self.informative = K_x > K_tl
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
        w_b = aq * (1.0 - p)
        B = np.column_stack(
            [np.bincount(self.K, w_b * z, m + 1) for z in self.Zt.T]
        )
        N_ug = -g[:, None] * _rev_cumsum(B)[1:]
        N_gg = -(self.Zt * (apq * (1.0 - p))[:, None]).T @ self.Zt
        return {
            "value": float(self.d @ u + self.zsum @ gamma + self.a @ log1ps),
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

    def start(
        self, x: npt.NDArray, n: npt.NDArray, tl: npt.NDArray
    ) -> npt.NDArray:
        """Log jumps to start from: the Nelson-Aalen increments,
        ``d / (number at risk)`` (the odds and the cumulative hazard
        agree where both are small); a row is at risk at ``t`` once it
        has entered (``tl < t``) and until it exits (``x >= t``)."""

        def at_or_above(times: npt.NDArray) -> npt.NDArray:
            order = np.argsort(times)
            tail = _rev_cumsum(np.append(n[order], 0.0))
            return tail[np.searchsorted(times[order], self.t, side="left")]

        at_risk = at_or_above(x) - at_or_above(tl)
        return np.log(self.d / np.maximum(at_risk, self.d))


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
            # Raises where the Hessian is not negative definite; there the
            # decrement can be 0 only by rounding, at the maximum.
            step = lik.solve_uu(der, grad)
            lam2 = max(float(grad @ step), 0.0)
            if not (np.isfinite(lam2) and np.all(np.isfinite(step))):
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
    S = _safe_schur(lik, der)
    lam_prev = np.inf
    for it in range(1, max_iter + 1):
        grad = der["grad_gamma"]
        try:
            # The Newton step where the profile information is positive
            # definite (its Cholesky factor exists).
            L = np.linalg.cholesky(S)
            step = np.linalg.solve(L.T, np.linalg.solve(L, grad))
            newton = bool(np.all(np.isfinite(step)))
        except np.linalg.LinAlgError:
            newton = False
        if not newton:
            # The profile is not concave here: a gradient step, scaled by
            # the diagonal of the information where it is positive.
            diag = np.nan_to_num(np.diag(S), nan=0.0)
            step = grad / np.where(diag > 0, diag, 1.0)
        lam = np.sqrt(max(float(grad @ step), 0.0))
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
        S = _safe_schur(lik, der)
        lam_prev = lam if (newton and t == 1.0) else np.inf
    return gamma, u, der, S, False, max_iter


def _safe_schur(lik: _POLikelihood, der: dict) -> npt.NDArray:
    """:meth:`_POLikelihood.schur`, ``nan`` where the baseline's Hessian
    cannot be solved."""
    try:
        return lik.schur(der)
    except (LinAlgError, ValueError):
        k = der["N_gg"].shape[0]
        return np.full((k, k), np.nan)


def _validate(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: "npt.ArrayLike | None",
    n: "npt.ArrayLike | None",
    tl: "npt.ArrayLike | None",
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    x_arr, c_arr, n_arr, tl_arr, Z_arr = semi_parametric_inputs(
        x,
        Z,
        c,
        n,
        tl,
        censoring=(
            "ProportionalOdds supports only observed (c=0) and "
            "right-censored (c=1) observations, with optional left "
            "truncation (`tl`); its baseline jumps at the event times, so "
            "it has no term for left-censored (c=-1) or interval-censored "
            "(c=2) data. Use a parametric proportional odds model instead, "
            "e.g. PO(LogLogistic).fit(x, Z, c=c) or WeibullPO.fit(x, Z, "
            "c=c), which handle every censoring type."
        ),
        truncation=(
            "ProportionalOdds supports left truncation (delayed entry) "
            "only, given as a one-dimensional `tl`; right or interval "
            "truncation is not available. Use a parametric proportional "
            "odds model (e.g. PO(LogLogistic)) with t=[tl, tr] for such "
            "data."
        ),
    )
    if not np.any(c_arr == 0):
        raise ValueError(
            "ProportionalOdds needs at least one event (c=0); with every "
            "observation censored there is no baseline to estimate."
        )
    return x_arr, c_arr, n_arr, tl_arr, Z_arr


def _po_aliased(
    Z: npt.NDArray, n: npt.NDArray, n_events: float
) -> tuple[npt.NDArray, npt.NDArray]:
    """The columns whose coefficients the likelihood cannot determine
    (#476), and the scale of the information about each coefficient.

    The likelihood depends on ``Z`` only through ``eta = -beta'Z``, and a
    shift of ``eta`` common to every row is absorbed by the baseline odds
    (a constant factor on ``G0``). A direction of ``beta`` along which
    ``Z beta`` is constant over the rows ``Z`` (those the likelihood
    depends on) is therefore not determined: a constant column, or a
    linear combination of the others. These are the null directions of
    the ``n``-weighted Gram matrix of the centred rows, as for the
    parametric models; the profile information at ``beta = 0``, Cox's
    yardstick, need not be positive definite here.

    The scale is the number of events times each column's weighted
    variance, the yardstick of
    :func:`~..proportional_hazards.cox_ph._cox_aliased`."""
    Zc = Z - covariate_center(Z, n)
    gram = (Zc * n[:, None]).T @ Zc
    aliased = aliased_columns(gram, Z.shape[0], constant_columns(Z))
    return aliased, n_events * np.diag(gram) / n.sum()


_LOG_TINY = float(np.log(np.finfo(float).tiny))


def _baseline_at_origin(
    log_g: npt.NDArray, shift: float, lp: npt.NDArray, center: npt.NDArray
) -> npt.NDArray:
    """The log jumps of the baseline odds fitted at the covariate means
    moved to ``Z = 0``: ``log_g + shift``, ``shift = -gamma'center``.
    Refused, with a ``ValueError`` pointing to ``center=True``, where a
    jump there over- or underflows, or the linear predictor ``lp`` of a
    fitted row does (as CoxPH and the parametric fits refuse, #463)."""
    out = log_g + shift
    ok = bool(np.all(np.abs(lp) < LOG_MAX)) and bool(
        np.all((out < LOG_MAX) & (out > _LOG_TINY))
    )
    if not ok:
        # shift = -gamma'center = beta'center, the model's coefficients
        # being beta = -gamma.
        raise baseline_at_origin_error("baseline odds", center, shift, shift)
    return out


class ProportionalOddsModel(
    LinearPredictorMixin, ConcordanceMixin, SerialisableMixin
):
    """
    A fitted semi-parametric proportional odds model, returned by
    :meth:`ProportionalOdds.fit <ProportionalOdds_.fit>` and
    ``fit_from_df``.

    The covariates multiply the survival odds of a baseline whose
    failure odds :math:`G_0 = F_0 / S_0` is a step function:

    .. math::
        S(x \\mid Z) = \\frac{1}{1 + G_0(x)\\, e^{-\\beta' (Z -
        \\text{center})}},

    so ``exp(beta)`` are survival odds ratios, as in the parametric
    ``LogisticPO`` / ``WeibullPO`` models (a positive coefficient means a
    longer life). ``x`` are the distinct observed times, ``g0`` the
    baseline's jumps there (0 at a time with no event) and ``G0`` its
    cumulative value, the baseline failure odds; the baseline is that of
    a unit at ``center``, ``Z = 0`` unless fitted with ``center=True``.
    Before the first time the survival is 1 and after the last it holds
    its last value, as for ``CoxPH``.

    Examples
    --------
    On the Rossi recidivism data (``arrest`` is 1 for an arrest, so ``c =
    1 - arrest``), financial aid raises the odds of staying out of
    prison by about 48 %, and each prior conviction lowers them by
    about 11 %:

    >>> from surpyval import ProportionalOdds
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static()
    >>> x, c = df["week"].values, 1 - df["arrest"].values
    >>> model = ProportionalOdds.fit(x, df[["fin", "age", "prio"]].values, c=c)
    >>> model.beta.round(4)
    array([ 0.391 ,  0.0701, -0.1116])
    >>> model.sf([20, 52], [1, 25, 3]).round(4)
    array([0.9351, 0.7938])
    """

    # Covariate metadata populated by ``fit_from_df``.
    feature_names: list[str] | None = None
    formula: str | None = None
    _model_spec: Any = None

    # Fitted quantities set by ``ProportionalOdds.fit``.
    beta: npt.NDArray
    params: npt.NDArray
    se: npt.NDArray
    cov: npt.NDArray
    p_values: npt.NDArray
    x: npt.NDArray
    d: npt.NDArray
    g0: npt.NDArray
    G0: npt.NDArray
    #: The covariate values the baseline is at: zeros by default, the
    #: ``n``-weighted covariate means for a fit with ``center=True``.
    center: "npt.NDArray | None" = None
    #: The log-likelihood at the maximum (Murphy et al.'s, with the jumps
    #: of the baseline odds in place of its density).
    log_likelihood: float = np.nan
    #: Newton iterations of the profile likelihood.
    n_iter: int = 0
    #: The rows fitted, ``{"x", "c", "n", "Z", "tl"}``, for
    #: ``concordance`` and the printout; not saved.
    _fit_data: "dict | None" = None
    #: The printout's data line of a restored model.
    _data_summary: "str | None" = None
    #: The family (``"Proportional Odds"``) and ``"Semi-Parametric"``,
    #: which the printout shows and ``to_dict`` stores.
    kind: str
    parameterization: str

    def __init__(self) -> None:
        self.kind = PROPORTIONAL_ODDS
        self.parameterization = "Semi-Parametric"

    @property
    def parameter_names(self) -> list[str]:
        """The names of ``params``, entry by entry: ``beta_0``,
        ``beta_1``, ... for the covariate coefficients."""
        return ["beta_{}".format(i) for i in range(len(self.params))]

    _ALIASED_WHY = (
        "a constant column, which the baseline odds absorb, or a linear "
        "combination of the others"
    )

    def phi(self, Z: "npt.ArrayLike | pd.DataFrame") -> npt.NDArray:
        """
        The survival odds multiplier :math:`e^{\\beta' (Z -
        \\text{center})}` of covariates ``Z`` (one row, or one per
        prediction): the survival odds ratio of ``Z`` against a unit at
        ``center``. On covariates far from ``center`` it can overflow to
        ``inf``; the predictions work on the log scale and do not.

        Examples
        --------
        >>> from surpyval import ProportionalOdds
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = ProportionalOdds.fit(x, df[["fin", "prio"]].values, c=c)
        >>> model.phi([[1, 0], [0, 0]]).round(4)
        array([1.5334, 1.    ])
        """
        with np.errstate(over="ignore"):
            return np.exp(self._log_phi(Z))

    def _concordance_risk(self, x: npt.NDArray, Z: Any) -> npt.NDArray:
        # A higher survival odds is a later event: the risk is -beta'Z.
        Z_arr = np.asarray(self._prepare_Z(Z), dtype=float)
        return -self._log_phi(Z_arr.reshape(x.size, -1))

    def _concordance_data(self) -> "tuple | None":
        data = self._fit_data
        if data is None:
            return None
        return data["x"], data["c"], data["n"], data["Z"]

    # -- the baseline at the query times -------------------------------

    def _parts(
        self, x: npt.ArrayLike, Z: "npt.ArrayLike | pd.DataFrame", grid: bool
    ) -> tuple:
        """``(log g, log G, log G_prev, eta)`` broadcast for the query:
        the log jump at the last baseline time at or before ``x``, the
        log baseline odds there and at the time before, and the log
        failure odds multiplier ``eta = -beta'(Z - center)``; paired
        (row ``i`` with ``x[i]``, or one of them single), or on the grid
        ``(len(Z), len(x))``. A missing time gives nan."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        idx = np.searchsorted(self.x, x, side="right") - 1
        top = self.x.size - 1
        with np.errstate(divide="ignore"):
            log_g = np.log(self.g0)
            log_G = np.log(self.G0)
        before = idx < 0
        lg = np.where(before, -np.inf, log_g[np.clip(idx, 0, top)])
        lG = np.where(before, -np.inf, log_G[np.clip(idx, 0, top)])
        lGp = np.where(idx < 1, -np.inf, log_G[np.clip(idx - 1, 0, top)])
        missing = np.isnan(x)
        lg, lG, lGp = (np.where(missing, np.nan, v) for v in (lg, lG, lGp))
        if grid:
            rows = covariate_rows(
                self._prepare_Z(Z), np.asarray(self.beta).shape[0]
            )
            eta = -((rows - self._center()) @ self._coef())[:, None]
            return lg[None, :], lG[None, :], lGp[None, :], eta
        eta = -self._log_phi(Z)
        check_paired_rows(x.size, np.size(eta))
        return lg, lG, lGp, eta

    @keeps_query_shape
    def Hf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        Cumulative hazard :math:`\\log(1 + G_0(x)\\, e^{-\\beta' (Z -
        \\text{center})})` at ``x`` for covariates ``Z`` (one row for every
        ``x``, or one row per ``x``, paired in the order given; a DataFrame
        for a model fitted with ``fit_from_df``). 0 before the first time,
        held after the last. With ``grid=True`` every time is evaluated
        for every row of ``Z``, giving shape ``(len(Z),) + x.shape``.
        """
        _, lG, _, eta = self._parts(x, Z, grid)
        with np.errstate(invalid="ignore"):  # a missing value is nan
            return np.logaddexp(0.0, lG + eta)

    @keeps_query_shape
    def sf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        Survival :math:`1 / (1 + G_0(x)\\, e^{-\\beta' (Z -
        \\text{center})})` at ``x`` for covariates ``Z``; arguments as for
        :meth:`Hf`. A missing time or covariate gives ``nan``.

        Examples
        --------
        >>> from surpyval import ProportionalOdds
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = ProportionalOdds.fit(x, df[["fin", "age"]].values, c=c)
        >>> model.sf([10, 30, 50], [[0, 20], [1, 20]], grid=True).round(3)
        array([[0.948, 0.801, 0.65 ],
               [0.964, 0.856, 0.733]])
        """
        _, lG, _, eta = self._parts(x, Z, grid)
        with np.errstate(invalid="ignore"):
            return expit(-(lG + eta))

    @keeps_query_shape
    def ff(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """Failure probability ``1 - sf`` at ``x`` for covariates ``Z``;
        arguments as for :meth:`Hf`."""
        _, lG, _, eta = self._parts(x, Z, grid)
        with np.errstate(invalid="ignore"):
            return expit(lG + eta)

    @keeps_query_shape
    def hf(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        The jump of the cumulative hazard :meth:`Hf` at the last baseline
        time at or before ``x`` (0 at a time with no event, and before the
        first time): a step size, not a hazard rate, as ``CoxPH``'s
        ``hf`` is. Arguments as for :meth:`Hf`.
        """
        lg, _, lGp, eta = self._parts(x, Z, grid)
        # log(1 + g e / (1 + G_prev e)), e = exp(eta), on the log scale.
        with np.errstate(invalid="ignore"):
            return np.logaddexp(0.0, lg + eta - np.logaddexp(0.0, lGp + eta))

    @keeps_query_shape
    def df(
        self,
        x: npt.ArrayLike,
        Z: "npt.ArrayLike | pd.DataFrame",
        *,
        grid: bool = False,
    ) -> npt.NDArray:
        """
        The probability mass at the last baseline time at or before ``x``,
        ``ff`` there less ``ff`` at the time before (0 at a time with no
        event, and before the first time): the baseline is a step
        function, so this is a jump, not a density. Arguments as for
        :meth:`Hf`.
        """
        lg, lG, lGp, eta = self._parts(x, Z, grid)
        # g e / ((1 + G e)(1 + G_prev e)), e = exp(eta), on the log scale.
        with np.errstate(invalid="ignore"):
            return np.exp(
                lg
                + eta
                - np.logaddexp(0.0, lG + eta)
                - np.logaddexp(0.0, lGp + eta)
            )

    # -- inference -------------------------------------------------------

    def covariance(self) -> npt.NDArray:
        """The covariance of the coefficients: the inverse of the
        negative Hessian of the profile log-likelihood at the maximum
        (``nan`` rows and columns for an aliased coefficient)."""
        return self.cov

    def standard_errors(self) -> npt.NDArray:
        """The coefficients' standard errors, from :meth:`covariance`."""
        return self.se

    def param_cb(
        self,
        name: str,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> npt.NDArray:
        """
        Wald confidence bound(s) on a coefficient, from the profile
        likelihood's information (see :meth:`covariance`).

        Parameters
        ----------
        name : str
            The coefficient, one of :attr:`parameter_names`.
        alpha_ci : float, optional
            Total tail probability of the bound(s). Default 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as ``[lower, upper]``.

        Examples
        --------
        >>> from surpyval import ProportionalOdds
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = ProportionalOdds.fit(x, df[["fin", "prio"]].values, c=c)
        >>> model.param_cb("beta_0").round(4)
        array([-0.0017,  0.8567])
        """
        names = self.parameter_names
        if name not in names:
            raise ValueError(
                "Unknown parameter {!r}; expected one of {}".format(
                    name, names
                )
            )
        idx = names.index(name)
        lower, upper = self._parameter_bounds()[idx]
        return wald_bound_on_support(
            float(self.params[idx]),
            float(self.cov[idx, idx]),
            lower,
            upper,
            alpha_ci,
            bound,
            name=name,
        )

    def _parameter_bounds(self) -> "list[tuple[None, None]]":
        """The support of each parameter: the coefficients are
        unbounded, so their bounds are on the natural scale."""
        return [(None, None)] * len(self.params)

    def summary(self, alpha_ci: float = 0.05) -> "pd.DataFrame":
        """
        The coefficient table: one row per covariate (named by
        ``feature_names`` for a model fitted with ``fit_from_df``), with
        the coefficient, the survival odds ratio ``exp(coef)``, the
        standard error, two-sided ``1 - alpha_ci`` Wald intervals for
        both, the Wald statistic ``z`` and its two-sided p-value. An
        aliased coefficient is ``nan`` throughout.

        Examples
        --------
        >>> from surpyval import ProportionalOdds
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> df["censored"] = 1 - df["arrest"]  # arrest is 1 for an arrest
        >>> model = ProportionalOdds.fit_from_df(
        ...     df, x_col="week", c_col="censored", Z_cols=["fin", "age"]
        ... )
        >>> model.summary()[["coef", "exp(coef)", "se(coef)", "p"]].round(4)
                     coef  exp(coef)  se(coef)       p
        covariate
        fin        0.3896     1.4764    0.2191  0.0754
        age        0.0751     1.0780    0.0227  0.0009
        """
        beta = np.asarray(self.beta, dtype=float)
        names = coefficient_names(self, beta.size)
        se = np.asarray(self.se, dtype=float)
        return coefficient_table(names, beta, se, alpha_ci, p=self.p_values)

    def _data_repr(self) -> str:
        data = self._fit_data
        if not isinstance(data, dict):
            return self._data_summary or ""
        tl = data["tl"]
        if not np.isfinite(tl).any():
            return data_summary(data["c"], data["n"], x=data["x"])
        return data_summary(
            data["c"], data["n"], tl=tl, lower=-np.inf, x=data["x"]
        )

    def __repr__(self) -> str:
        out = (
            "Semi-Parametric Regression SurPyval Model"
            + "\n========================================="
            + "\nType                : Proportional Odds"
            + "\nKind                : NPMLE (Murphy, Rossini & van der "
            "Vaart)" + "\nParameterization    : Semi-Parametric"
        )
        data_line = self._data_repr()
        if data_line:
            out += "\nData                : " + data_line
        if np.any(self._center()):
            out += (
                "\nBaseline at         : the covariate means, Z = {}".format(
                    np.array2string(self._center(), separator=", ")
                )
            )
        out += (
            "\nCoefficients        : exp(coef) is the survival odds ratio; "
            "Wald 95% intervals\n"
        )
        return out + coefficient_repr(self.summary()) + "\n"

    # -- serialisation ---------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise the fitted model to a plain, JSON-serialisable dict: the
        coefficients with their covariance and p-values, and the baseline
        step arrays (the times ``x``, events ``d``, jumps ``g0`` and
        cumulative baseline odds ``G0``), with ``center`` where the
        baseline is at the covariate means. Every prediction round-trips
        exactly; the fitted rows are not stored.

        Examples
        --------
        >>> from surpyval import ProportionalOdds, ProportionalOddsModel
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> model = ProportionalOdds.fit(x, df[["fin", "prio"]].values, c=c)
        >>> restored = ProportionalOddsModel.from_dict(model.to_dict())
        >>> bool(restored.sf(52, [1, 2]) == model.sf(52, [1, 2]))
        True
        """
        out: dict[str, Any] = {
            "model": "ProportionalOddsModel",
            "beta": np.asarray(self.beta, dtype=float).tolist(),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "se": np.asarray(self.se, dtype=float).tolist(),
            "cov": np.asarray(self.cov, dtype=float).tolist(),
            "p_values": np.asarray(self.p_values, dtype=float).tolist(),
            "x": np.asarray(self.x, dtype=float).tolist(),
            "d": np.asarray(self.d, dtype=float).tolist(),
            "g0": np.asarray(self.g0, dtype=float).tolist(),
            "G0": np.asarray(self.G0, dtype=float).tolist(),
            "log_likelihood": float(self.log_likelihood),
            "n_iter": int(self.n_iter),
        }
        if np.any(self._center()):
            out["center"] = self._center().tolist()
        if self._data_repr():
            out["data_summary"] = self._data_repr()
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "ProportionalOddsModel":
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        Examples
        --------
        >>> from surpyval import ProportionalOdds, ProportionalOddsModel
        >>> model = ProportionalOdds.fit([1, 2, 3, 4], [0, 1, 0, 1])
        >>> ProportionalOddsModel.from_dict(model.to_dict()).beta.round(4)
        array([1.1744])
        """
        require_model_tag(
            model_dict,
            "ProportionalOddsModel",
            "a semi-parametric proportional odds model",
        )
        out = cls()
        for key in ("beta", "params", "se", "p_values", "x", "d", "g0"):
            setattr(out, key, np.array(model_dict[key], dtype=float))
        out.G0 = np.array(model_dict["G0"], dtype=float)
        out.cov = np.array(model_dict["cov"], dtype=float).reshape(
            out.beta.size, out.beta.size
        )
        out.center = np.array(
            model_dict.get("center", np.zeros(out.beta.size)), dtype=float
        )
        out.log_likelihood = float(model_dict.get("log_likelihood", np.nan))
        out.n_iter = int(model_dict.get("n_iter", 0))
        out._data_summary = model_dict.get("data_summary")
        restore_covariate_meta(out, model_dict)
        return out


class ProportionalOdds_:
    """
    The semi-parametric proportional odds model: the covariates multiply
    the survival odds of a baseline left to the data,

    .. math::
        \\frac{S(x \\mid Z)}{F(x \\mid Z)} = e^{\\beta' Z}\\,
        \\frac{S_0(x)}{F_0(x)},

    the proportional-odds counterpart of ``CoxPH``, fitted by
    nonparametric maximum likelihood (Murphy, Rossini and van der Vaart
    1997). A positive coefficient raises the odds of survival, as in the
    parametric proportional odds models (``LogisticPO``, ``PO(...)``);
    R's ``timereg::prop.odds`` and ``mets::logitSurv`` model the odds of
    failure, so their coefficients have the opposite sign.
    ``ProportionalOdds`` is an instance of this class; its ``fit``
    returns a :class:`ProportionalOddsModel`.
    """

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        tol: float = 1e-9,
        center: bool = False,
    ) -> ProportionalOddsModel:
        """
        Fit the semi-parametric proportional odds model.

        The coefficients maximise the profile likelihood, the likelihood
        maximised over the baseline odds at each ``beta`` (see the module
        notes), and their standard errors are the profile likelihood's
        (the inverse of its negative Hessian at the maximum), as Murphy,
        Rossini and van der Vaart (1997) justify.

        Parameters
        ----------
        x : array_like
            The observed times. Only their order matters. Tied event times
            share one jump of the baseline, each event contributing its
            full term (Breslow's convention for Cox, which R's ``coxph``
            with ``ties = "breslow"`` and a unit gamma frailty matches);
            under heavy ties the coefficients are attenuated a little
            towards 0, as Breslow's are (by about 5% with 100 events on 30
            distinct times).
        Z : array_like
            The covariates, one row per observation (a 1-D array is one
            covariate). Rows with a missing or infinite covariate are
            dropped, with a warning. A constant column (the baseline odds
            absorb it) or a linear combination of the others is aliased,
            as in ``CoxPH``: its coefficient is ``nan`` and one warning
            names it.
        c : array_like, optional
            The censoring flags: 0 observed, 1 right censored. Defaults to
            all observed. Left (-1) and interval (2) censored rows raise a
            ``ValueError``; fit those with a parametric proportional odds
            model (``PO(LogLogistic)``, ``WeibullPO``, ...).
        n : array_like, optional
            The count of each row. Defaults to 1.
        tl : array_like, optional
            Left-truncation (delayed entry) times: a row is in the risk
            sets only after its entry time, and its likelihood is
            conditional on surviving to it.
        tol : float, optional
            The convergence tolerance: Newton-Raphson on the profile
            likelihood stops once a step is at most ``tol`` standard errors
            long (measured by the profile information).
        center : bool, optional
            ``False`` (the default) reports the baseline (``g0``, ``G0``)
            at ``Z = 0``; ``True`` reports it at the covariate means,
            stored as ``model.center``. The fit runs on centred covariates
            either way, so the coefficients and predictions are the same;
            the default refuses, with a ``ValueError``, covariates so far
            from 0 that the baseline there over- or underflows.

        Returns
        -------
        ProportionalOddsModel
            The fitted model. If a covariate separates the events from the
            survivors (a level with no events, say) the likelihood has no
            finite maximum: the fit warns "No finite maximum" and that
            coefficient is meaningless.

        Examples
        --------
        >>> from surpyval import ProportionalOdds
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> Z = df[["fin", "age", "prio"]].values
        >>> model = ProportionalOdds.fit(x, Z, c=c)
        >>> model.beta.round(4)
        array([ 0.391 ,  0.0701, -0.1116])
        >>> model.se.round(4)
        array([0.2206, 0.0227, 0.0331])
        """
        x, c, n, tl, Z = _validate(x, Z, c, n, tl)
        p = Z.shape[1]
        mean = covariate_center(Z, n)
        Zc = Z - mean
        inner_tol = min(tol, 1e-10) * 1e-2
        lik = _POLikelihood(x, c, n, tl, Zc)
        rows = lik.informative
        aliased, scale = _po_aliased(Z[rows], n[rows], float(lik.d.sum()))
        kept = np.setdiff1d(np.arange(p), aliased)
        if aliased.size:
            warn_aliased(
                aliased,
                "the likelihood does not depend on them beyond a "
                "combination of the other columns (a constant column, "
                "which the baseline odds absorb, or a linear combination "
                "of the others)",
            )
            lik = _POLikelihood(x, c, n, tl, Zc[:, kept])
        gamma = np.zeros(kept.size)
        converged, n_iter = True, 0
        S = np.zeros((0, 0))
        with np.errstate(all="ignore"):
            # The baseline at beta = 0 is the start.
            u, der = _inner(lik, gamma, lik.start(x, n, tl), inner_tol)
            if kept.size:
                gamma, u, der, S, converged, n_iter = _profile_fit(
                    lik, u, gamma, tol, _OUTER_MAX_ITER
                )
        if kept.size:
            _check_maximum(S, scale[kept], kept, converged, n_iter)

        cov_k = _inverse_information(S)
        with np.errstate(invalid="ignore", divide="ignore"):
            se_k = np.sqrt(np.diag(cov_k))
            p_k = 2.0 * norm.sf(np.abs(-gamma / se_k))
        beta = expand(-gamma, kept, p)
        cov = np.full((p, p), np.nan)
        cov[np.ix_(kept, kept)] = cov_k

        # The baseline odds jumps, fitted at the means, moved to Z = 0
        # unless center=True; then laid on every distinct observed time
        # (0 where there is no event), as CoxPH's baseline is.
        gamma_full = np.zeros(p)
        gamma_full[kept] = gamma
        log_g = u
        if center:
            model_center = mean
        else:
            log_g = _baseline_at_origin(
                u, -float(gamma_full @ mean), Z @ gamma_full, mean
            )
            model_center = np.zeros(p)
        times = np.unique(x)
        at = np.searchsorted(times, lik.t)
        g0 = np.zeros(times.size)
        g0[at] = np.exp(log_g)
        d = np.zeros(times.size)
        d[at] = lik.d

        model = ProportionalOddsModel()
        model.beta = beta
        model.params = copy(beta)
        model.se = expand(se_k, kept, p)
        model.cov = cov
        model.p_values = expand(p_k, kept, p)
        model.x = times
        model.d = d
        model.g0 = g0
        model.G0 = np.cumsum(g0)
        model.center = model_center
        model.log_likelihood = float(der["value"])
        model.n_iter = int(n_iter)
        model._fit_data = {"x": x, "c": c, "n": n, "Z": Z, "tl": tl}
        return model

    def fit_from_df(
        self,
        df: "pd.DataFrame",
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
        tl_col: str | None = None,
        tol: float = 1e-9,
        center: bool = False,
    ) -> ProportionalOddsModel:
        """
        Fit the semi-parametric proportional odds model from a pandas
        DataFrame, keeping the covariate names for prediction and the
        coefficient table; see :meth:`fit` for the model.

        Parameters
        ----------
        df : pandas.DataFrame
            The data.
        x_col : str
            The column of the observed times.
        Z_cols : str or list of str, optional
            The covariate column(s). Give this or ``formula``.
        c_col, n_col, tl_col : str, optional
            The columns of the censoring flags, counts and entry times.
        formula : str, optional
            A ``formulaic`` formula for the covariates (``"age + C(site)"``)
            instead of ``Z_cols``.
        tol, center
            As for :meth:`fit`.

        Examples
        --------
        >>> from surpyval import ProportionalOdds
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> df["censored"] = 1 - df["arrest"]
        >>> model = ProportionalOdds.fit_from_df(
        ...     df, x_col="week", c_col="censored", formula="fin + prio"
        ... )
        >>> model.beta.round(4)
        array([ 0.4275, -0.1206])
        """
        Z, feature_names, model_spec = design_matrix_from_df(
            df, Z_cols, formula
        )
        x = df[x_col].values
        c = None if c_col is None else df[c_col].values
        n = None if n_col is None else df[n_col].values
        tl = None if tl_col is None else df[tl_col].values
        with covariate_columns(feature_names, Z, model_spec):
            model = self.fit(x, Z, c=c, n=n, tl=tl, tol=tol, center=center)
        model.feature_names = feature_names
        model.formula = formula
        model._model_spec = model_spec
        return model


def _inverse_information(S: npt.NDArray) -> npt.NDArray:
    """The covariance from the profile information ``S``; the
    pseudo-inverse where the inverse has a non-positive variance (a
    likelihood with no finite maximum), and ``nan`` where that does
    too."""
    k = S.shape[0]
    if k == 0:
        return np.zeros((0, 0))
    if not np.all(np.isfinite(S)):
        return np.full((k, k), np.nan)
    try:
        cov = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(S)
    if np.any(np.diag(cov) <= 0):
        cov = np.linalg.pinv(S)
    bad = ~(np.diag(cov) > 0)
    cov[bad, :] = np.nan
    cov[:, bad] = np.nan
    return cov


def _check_maximum(
    S: npt.NDArray,
    scale: npt.NDArray,
    kept: npt.NDArray,
    converged: bool,
    n_iter: int,
) -> None:
    """One warning when the profile likelihood has no finite maximum or
    the iteration did not converge.

    The criterion is Cox's (``_warn_if_monotone``): where a covariate
    separates the events from the survivors, the likelihood keeps
    increasing as that coefficient grows, and the information for it
    collapses (the survival odds of the separated rows go to 0 or
    infinity, and the likelihood flattens in that direction). A
    coefficient whose profile information ``S`` has fallen below ``1e-8``
    of ``scale``, the number of events times the covariate's variance (of
    the order of the information about it, which on an ordinary fit is a
    fraction of that), is taken to have run away."""
    d = np.diag(np.atleast_2d(S)) if S.size else np.zeros(0)
    d0 = np.asarray(scale, dtype=float)
    collapsed = np.flatnonzero(
        (d0 > 0) & ~(np.nan_to_num(d, nan=0.0) > 1e-8 * d0)
    )
    if collapsed.size:
        warn_no_maximum(
            "the profile likelihood keeps increasing as coefficient(s) {} "
            "of Z grow without bound, so the estimate is infinite (the "
            "covariate separates the events from the survivors, as a "
            "level with no events does)".format(kept[collapsed].tolist()),
            "The reported value, its standard error and its p-value are "
            "meaningless",
            "consider removing or coarsening the covariate",
        )
    elif not converged:
        warn_unverified(
            "The ProportionalOdds fit",
            "the profile likelihood's Newton-Raphson iteration stopped after "
            "{} step(s) without reaching its tolerance".format(n_iter),
            "check the covariates for extreme values, or rescale them",
        )


ProportionalOdds = ProportionalOdds_()
