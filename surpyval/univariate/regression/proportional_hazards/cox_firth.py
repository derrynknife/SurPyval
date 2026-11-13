"""Firth's penalised Cox fit, ``CoxPH.fit_firth`` (Heinze and Schemper
2001).

Where a covariate, or a combination of them, separates the events from
the survivors, the Cox partial likelihood keeps increasing as a
coefficient grows and has no finite maximum (a *monotone likelihood*,
see :mod:`.cox_separation`). Firth's penalty adds half the log
determinant of the information,

.. math::
    \\ell^*(\\beta) = \\ell(\\beta) + \\tfrac{1}{2} \\log |I(\\beta)|,

Jeffreys' invariant prior: the information falls to 0 as a coefficient
runs off, so the penalty pulls it back and the estimate is finite. Where
the data do carry the coefficients the penalty is small, of order one
against a log-likelihood of order the number of events, and removes the
first-order bias of the maximum likelihood estimate (Firth 1993).

The partial likelihood is Breslow's, as R's ``coxphf`` (its only tie
handling). The penalised score is

.. math::
    U^*_r(\\beta) = U_r(\\beta) + \\tfrac{1}{2}
    \\operatorname{tr}\\left[I(\\beta)^{-1}
    \\frac{\\partial I(\\beta)}{\\partial \\beta_r}\\right],

where :math:`\\partial I / \\partial \\beta_r` is, at each event time, the
deaths times the risk set's weighted third central moment of the
covariates with the ``r``-th. The estimate solves :math:`U^* = 0` by
Newton steps on :math:`I(\\beta)` with step-halving (``coxphf``'s
iteration, each step at most 0.5 in the standardised covariates). The
covariance is the inverse of the information of the partial likelihood,
:math:`I(\\hat\\beta)^{-1}`, at the estimate, as ``coxphf`` reports;
the intervals and the tests of each coefficient are by the profile
penalised likelihood (Heinze and Schemper 2001), with the Wald ones from
that covariance beside them.
"""

from __future__ import annotations

from copy import copy
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq
from scipy.stats import chi2, norm

from surpyval.univariate.information_criteria import ic_sample_size
from surpyval.utils import validate_coxph
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.pickling import Rebuilt
from surpyval.utils.validation import check_alpha_ci

from .._fit_skeleton import covariate_center
from ..semi_parametric_regression_model import SemiParametricRegressionModel
from .cox_likelihood import baselines_at_origin, combined_generators

if TYPE_CHECKING:
    from .cox_ph import CoxPH_

#: The tie handling of every Firth fit (``coxphf``'s only one).
FIRTH_TIES = "breslow"

#: The largest Newton step, in each standardised coefficient (coxphf's
#: ``maxstep``).
_MAX_STEP = 0.5

#: The most Newton iterations, and step halvings within one.
_MAX_ITER = 200
_MAX_HALVINGS = 30

#: The cells (event times by rows by p cubed) of one block of the
#: risk-set sums: bounds the memory of the third moments.
_BLOCK_CELLS = 4_000_000


class FirthRefusal(ValueError):
    """An information criterion (AIC, BIC, AICc) asked of a Firth fit,
    which has none to give: a ``ValueError`` saying why and what to use
    instead, as a weighted fit's ``WeightedRefusal``."""


#: Why a Firth fit has no information criteria, and what to use instead.
FIRTH_IC_WHY = (
    "the fit maximises the penalised partial likelihood, which is not a "
    "likelihood to compare models by, and the ordinary partial likelihood "
    "is not at its maximum at the penalised estimate"
)
FIRTH_IC_INSTEAD = (
    "Refit the ordinary model (CoxPH.fit) for AIC and BIC, or test the "
    "covariates with the penalised likelihood-ratio test, "
    "model.likelihood_ratio_test()."
)


def refuse_firth_information_criterion(name: str) -> FirthRefusal:
    """The refusal of the criterion ``name`` (``"aic()"``, ...) of a
    Firth fit: what it is, why, and what to use instead."""
    return FirthRefusal(
        f"{name} is not available for a Firth-penalised Cox fit "
        f"(CoxPH.fit_firth): {FIRTH_IC_WHY}. {FIRTH_IC_INSTEAD}"
    )


class FirthResult(NamedTuple):
    """What a Firth-penalised Cox fit (:meth:`CoxPH.fit_firth
    <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit_firth>`)
    adds to its model, as ``model.firth``: the intervals and tests that
    go with the penalty.

    Attributes
    ----------
    alpha_ci : float
        The intervals' total tail probability.
    profile : bool
        Whether ``lower``, ``upper`` and ``p_values`` are by the profile
        penalised likelihood (the default) or Wald's.
    lower, upper : numpy.ndarray
        The coefficients' ``1 - alpha_ci`` intervals: the profile
        penalised-likelihood ones, or Wald's with ``profile=False``.
    p_values : numpy.ndarray
        The tests of each coefficient being 0: the penalised
        likelihood-ratio test, or Wald's with ``profile=False``.
    chisq : numpy.ndarray
        The penalised likelihood-ratio statistic of each coefficient
        being 0, chi-squared on 1 degree of freedom (``nan`` with
        ``profile=False``).
    wald_lower, wald_upper, wald_p : numpy.ndarray
        The Wald intervals and p-values, from ``covariance()``.
    log_likelihood : float
        The penalised log partial likelihood at the estimate,
        :math:`\\ell(\\hat\\beta) + \\frac12 \\log|I(\\hat\\beta)|`, with
        the information in the covariates' own units.
    log_likelihood_null : float
        The same at every coefficient 0.
    df : int
        The number of coefficients.
    ties : str
        ``"breslow"``.
    """

    alpha_ci: float
    profile: bool
    lower: npt.NDArray
    upper: npt.NDArray
    p_values: npt.NDArray
    chisq: npt.NDArray
    wald_lower: npt.NDArray
    wald_upper: npt.NDArray
    wald_p: npt.NDArray
    log_likelihood: float
    log_likelihood_null: float
    df: int
    ties: str = FIRTH_TIES

    def likelihood_ratio_test(self) -> dict:
        """The penalised likelihood-ratio test of every coefficient being
        0: twice the gain in the penalised log partial likelihood over
        the model with all of them at 0, chi-squared on ``df`` degrees of
        freedom (``coxphf``'s "Likelihood ratio test"). The keys are
        those of the unpenalised model's ``likelihood_ratio_test``."""
        statistic = max(
            2.0 * (self.log_likelihood - self.log_likelihood_null), 0.0
        )
        return {
            "statistic": float(statistic),
            "df": int(self.df),
            "p_value": float(chi2.sf(statistic, self.df)),
            "log_likelihood": float(self.log_likelihood),
            "log_likelihood_null": float(self.log_likelihood_null),
        }

    def to_dict(self) -> dict:
        """The result as JSON (an infinite bound as it is; ``to_dict``
        of the model writes it as strict JSON)."""
        out: dict[str, Any] = {
            "alpha_ci": float(self.alpha_ci),
            "profile": bool(self.profile),
            "df": int(self.df),
            "ties": self.ties,
            "log_likelihood": float(self.log_likelihood),
            "log_likelihood_null": float(self.log_likelihood_null),
        }
        for key in (
            "lower",
            "upper",
            "p_values",
            "chisq",
            "wald_lower",
            "wald_upper",
            "wald_p",
        ):
            out[key] = [float(v) for v in getattr(self, key)]
        return out

    @classmethod
    def from_dict(cls, d: dict) -> FirthResult:
        """The result of a :meth:`to_dict` dictionary."""

        def arr(key: str) -> npt.NDArray:
            return np.array(
                [np.nan if v is None else v for v in d[key]], dtype=float
            )

        return cls(
            float(d["alpha_ci"]),
            bool(d["profile"]),
            arr("lower"),
            arr("upper"),
            arr("p_values"),
            arr("chisq"),
            arr("wald_lower"),
            arr("wald_upper"),
            arr("wald_p"),
            float(d["log_likelihood"]),
            float(d["log_likelihood_null"]),
            int(d["df"]),
            str(d.get("ties", FIRTH_TIES)),
        )

    def line(self) -> str:
        """The printout's line for the fit."""
        return "Firth (Jeffreys prior); {} intervals and tests".format(
            "profile penalised-likelihood" if self.profile else "Wald"
        )

    def ic_line(self) -> str:
        """The printout's line on the information criteria, which a Firth
        fit does not give."""
        return (
            "none: a penalised likelihood does not compare models; "
            "refit with CoxPH.fit for AIC and BIC"
        )


class _BreslowSums:
    """The Breslow partial likelihood of rows ``x``, ``c``, ``n``,
    ``tl`` (delayed entry), covariates ``Z`` and stratum codes, with its
    score, information and the information's derivatives.

    The event groups are the distinct (stratum, event time) pairs; a row
    is in a group's risk set when it is in the stratum and ``tl < t <=
    x`` (as ``cox_at_risk_mask``). The sums over a risk set are formed
    with each group's weights scaled by its largest, so they neither
    over- nor underflow."""

    def __init__(
        self,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        tl: npt.NDArray,
        Z: npt.NDArray,
        strata: npt.NDArray,
    ) -> None:
        self.x, self.n, self.tl, self.Z = x, n, tl, Z
        self.strata = strata
        dead = (c == 0) & (n > 0)
        keys = np.unique(np.column_stack([strata[dead], x[dead]]), axis=0)
        self.group_stratum = keys[:, 0]
        self.group_time = keys[:, 1]
        K, p = keys.shape[0], Z.shape[1]
        self.D = np.zeros(K)
        self.sZ = np.zeros((K, p))
        rows = np.flatnonzero(dead)
        # Each death's group
        lookup = {(s, t): k for k, (s, t) in enumerate(map(tuple, keys))}
        for i in rows:
            k = lookup[(strata[i], x[i])]
            self.D[k] += n[i]
            self.sZ[k] += n[i] * Z[i]
        self.ZZ = np.einsum("ij,ik->ijk", Z, Z).reshape(len(x), p * p)
        self.ZZZ = np.einsum("ij,ik,il->ijkl", Z, Z, Z).reshape(len(x), p**3)
        cells = max(len(x) * max(p, 1) ** 3, 1)
        self.block = max(1, _BLOCK_CELLS // cells)

    def derivatives(
        self, beta: npt.NDArray, third: bool = True
    ) -> tuple[float, npt.NDArray, npt.NDArray, npt.NDArray | None]:
        """``(loglik, score, information, dI)`` at ``beta``; ``dI[a, b,
        r]`` is the derivative of the information's ``(a, b)`` entry in
        ``beta_r`` (``None`` unless ``third``)."""
        Z = self.Z
        p = Z.shape[1]
        eta = Z @ beta
        ll = float(np.sum(self.sZ @ beta))
        U = self.sZ.sum(axis=0)
        info = np.zeros((p, p))
        dI = np.zeros((p, p, p)) if third else None
        for start in range(0, self.D.size, self.block):
            sl = slice(start, start + self.block)
            t = self.group_time[sl]
            at = (
                (self.strata[None, :] == self.group_stratum[sl, None])
                & (self.tl[None, :] < t[:, None])
                & (self.x[None, :] >= t[:, None])
                & (self.n[None, :] > 0)
            )
            shift = np.max(np.where(at, eta[None, :], -np.inf), axis=1)
            with np.errstate(invalid="ignore", over="ignore"):
                W = np.where(
                    at,
                    self.n[None, :] * np.exp(eta[None, :] - shift[:, None]),
                    0,
                )
            S0 = W.sum(axis=1)
            D = self.D[sl]
            m = (W @ Z) / S0[:, None]
            E2 = ((W @ self.ZZ) / S0[:, None]).reshape(-1, p, p)
            V = E2 - np.einsum("kj,kl->kjl", m, m)
            ll -= float(D @ (np.log(S0) + shift))
            U = U - D @ m
            info += np.einsum("k,kjl->jl", D, V)
            if third:
                E3 = ((W @ self.ZZZ) / S0[:, None]).reshape(-1, p, p, p)
                # The third central moment from the raw ones
                T = (
                    E3
                    - np.einsum("ka,kbc->kabc", m, E2)
                    - np.einsum("kb,kac->kabc", m, E2)
                    - np.einsum("kc,kab->kabc", m, E2)
                    + 2 * np.einsum("ka,kb,kc->kabc", m, m, m)
                )
                dI = dI + np.einsum("k,kabc->abc", D, T)
        return ll, U, info, dI


class _Firth:
    """The penalised log partial likelihood of :class:`_BreslowSums`
    and its maximisation, over every coefficient or with some held
    fixed."""

    def __init__(self, sums: _BreslowSums, tol: float) -> None:
        self.sums = sums
        self.tol = tol
        self.p = sums.Z.shape[1]

    def value(self, beta: npt.NDArray) -> float:
        """The penalised log partial likelihood (``-inf`` where the
        information is not positive definite)."""
        ll, _, info, _ = self.sums.derivatives(beta, third=False)
        sign, logdet = np.linalg.slogdet(info)
        if not sign > 0 or not np.isfinite(ll):
            return -np.inf
        return ll + 0.5 * logdet

    def score(self, beta: npt.NDArray) -> tuple[float, npt.NDArray, Any]:
        """The penalised log-likelihood, its score and the information."""
        ll, U, info, dI = self.sums.derivatives(beta)
        assert dI is not None
        sign, logdet = np.linalg.slogdet(info)
        if not sign > 0:
            return -np.inf, np.full(self.p, np.nan), info
        inv = np.linalg.inv(info)
        U_star = U + 0.5 * np.einsum("ab,abr->r", inv, dI)
        return ll + 0.5 * logdet, U_star, info

    def maximise(
        self, start: npt.NDArray, free: npt.NDArray
    ) -> tuple[float, npt.NDArray, bool]:
        """Maximise over the coefficients ``free`` (the others held at
        ``start``): ``(value, beta, converged)``. Newton steps on the
        information, at most ``_MAX_STEP`` in each coefficient, halved
        until the penalised likelihood does not fall."""
        beta = np.array(start, dtype=float)
        value, U, info = self.score(beta)
        if free.size == 0:
            return value, beta, True
        for _ in range(_MAX_ITER):
            sub = info[np.ix_(free, free)]
            step = np.linalg.solve(sub, U[free])
            # The Newton decrement: how far below the maximum, in nats
            if float(U[free] @ step) < self.tol**2 or (
                np.max(np.abs(step)) < self.tol
            ):
                return value, beta, True
            biggest = np.max(np.abs(step))
            if biggest > _MAX_STEP:
                step = step * (_MAX_STEP / biggest)
            for _ in range(_MAX_HALVINGS):
                trial = beta.copy()
                trial[free] += step
                v_trial, U_trial, info_trial = self.score(trial)
                if v_trial >= value - 1e-12 * max(1.0, abs(value)):
                    break
                step = step / 2
            else:
                return value, beta, False
            beta, value, U, info = trial, v_trial, U_trial, info_trial
        return value, beta, False

    def hessian(self, beta: npt.NDArray) -> npt.NDArray:
        """The Hessian of the penalised log-likelihood by central
        differences of its score (the check of the maximum)."""
        h = 1e-5
        H = np.empty((self.p, self.p))
        for r in range(self.p):
            e = np.zeros(self.p)
            e[r] = h
            H[:, r] = (self.score(beta + e)[1] - self.score(beta - e)[1]) / (
                2 * h
            )
        return (H + H.T) / 2


def _profile_bounds(
    firth: _Firth,
    beta_hat: npt.NDArray,
    best: float,
    se: npt.NDArray,
    alpha_ci: float,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, list]:
    """The profile penalised-likelihood bounds and the penalised
    likelihood-ratio statistic of each coefficient being 0 (in the
    standardised coefficients), and the coefficients whose search did not
    converge."""
    p = beta_hat.size
    q = float(chi2.ppf(1 - alpha_ci, 1))
    lower, upper, stat = np.empty(p), np.empty(p), np.empty(p)
    failed: list = []
    for j in range(p):
        free = np.setdiff1d(np.arange(p), [j])
        warm = {"beta": beta_hat.copy()}

        def deviance(b: float) -> float:
            start = warm["beta"].copy()
            start[j] = b
            value, beta, ok = firth.maximise(start, free)
            if not ok:
                failed.append(j)
            if np.isfinite(value):
                warm["beta"] = beta
            return 2.0 * (best - value)

        stat[j] = max(deviance(0.0), 0.0)
        warm["beta"] = beta_hat.copy()
        step0 = max(float(se[j]) if np.isfinite(se[j]) else 1.0, 1e-3)
        for side, out in ((1.0, upper), (-1.0, lower)):
            warm["beta"] = beta_hat.copy()
            inner, step = float(beta_hat[j]), step0 * np.sqrt(q)
            outer = inner + side * step
            found = False
            for _ in range(60):
                if deviance(outer) - q > 0:
                    found = True
                    break
                inner, step = outer, step * 2
                outer = inner + side * step
            if not found:
                out[j] = side * np.inf
                continue
            out[j] = brentq(
                lambda b: deviance(b) - q,
                min(inner, outer),
                max(inner, outer),
                xtol=1e-12,
                rtol=4 * np.finfo(float).eps,
            )
    return lower, upper, stat, sorted(set(failed))


def _validated(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
    strata: npt.ArrayLike | None,
) -> tuple[npt.NDArray | None, list]:
    """The rows, as :func:`cox_ph._stratified_rows` gives them (one
    stratum, labelled ``None``, without ``strata``)."""
    from .cox_ph import _require_an_event, _stratified_rows

    if strata is not None:
        return _stratified_rows(x, Z, c, n, tl, FIRTH_TIES, strata)
    x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, FIRTH_TIES)
    _require_an_event(c, n)
    return None, [(None, x, c, n, tl, Z)]


def _refuse_unidentified(
    Z: npt.NDArray,
    n: npt.NDArray,
    sd: npt.NDArray,
    info0: npt.NDArray,
    n_events: float,
    strata: npt.NDArray | None,
) -> None:
    """Refuse a column the partial likelihood cannot determine: Firth's
    penalty is the log determinant of the information, which has none
    where the information is singular."""
    from .cox_ph import _cox_aliased

    aliased = np.flatnonzero(~(sd > 0))
    if aliased.size == 0:
        aliased = _cox_aliased(info0, Z, n, n_events, strata)
    if aliased.size:
        raise ValueError(
            "CoxPH.fit_firth: the partial likelihood does not depend on "
            "column(s) {} of Z (a constant column -- a Cox model has no "
            "intercept --, one constant within each stratum, or a linear "
            "combination of the others), so their information is 0 and "
            "Firth's penalty, its log determinant, is not defined. Remove "
            "them from Z and refit.".format(aliased.tolist())
        )


def fit_cox_firth(
    fitter: CoxPH_,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
    strata: npt.ArrayLike | None,
    alpha_ci: float,
    profile: bool,
    center: bool,
    tol: float,
) -> SemiParametricRegressionModel:
    """:meth:`CoxPH_.fit_firth`."""
    check_alpha_ci(alpha_ci)
    labels, validated = _validated(x, Z, c, n, tl, strata)
    xs = np.concatenate([v[1] for v in validated]).astype(float)
    cs = np.concatenate([v[2] for v in validated]).astype(int)
    ns = np.concatenate([v[3] for v in validated]).astype(float)
    tls = np.concatenate([v[4] for v in validated]).astype(float)
    Zs = np.vstack([v[5] for v in validated]).astype(float)
    codes = np.concatenate(
        [np.full(len(v[1]), k) for k, v in enumerate(validated)]
    ).astype(float)
    p = Zs.shape[1]
    if p == 0:
        raise ValueError(
            "CoxPH.fit_firth needs at least one covariate column in Z."
        )
    mean = covariate_center(Zs, ns)
    Zc = Zs - mean
    # Fitted on the covariates divided by their (count-weighted) standard
    # deviations, as coxphf: the estimates and tests do not change, and
    # the step limit means the same for every column.
    sd = np.sqrt((ns @ Zc**2) / ns.sum())
    n_events = float(ns[cs == 0].sum())
    unit = np.where(sd > 0, sd, 1.0)
    sums = _BreslowSums(xs, cs, ns, tls, Zc / unit, codes)
    info0 = sums.derivatives(np.zeros(p), third=False)[2]
    _refuse_unidentified(
        Zs,
        ns,
        sd,
        info0 * np.outer(1 / unit, 1 / unit),
        n_events,
        None if strata is None else codes,
    )
    firth = _Firth(sums, tol)
    best, b_std, converged = firth.maximise(np.zeros(p), np.arange(p))
    value_null = firth.value(np.zeros(p))
    _, U_star, info = firth.score(b_std)
    hess = firth.hessian(b_std)
    verified = (
        converged
        and np.all(np.isfinite(U_star))
        and np.all(np.linalg.eigvalsh(-hess) > 0)
    )
    if not verified:
        warn_unverified(
            "Firth's penalised Cox fit",
            None if converged else "its Newton search stopped short",
        )
    cov_std = np.linalg.inv(info)
    se_std = np.sqrt(np.diag(cov_std))
    q = norm.ppf(1 - alpha_ci / 2)
    # In the covariates' own units
    beta = b_std / sd
    covariance = cov_std / np.outer(sd, sd)
    se = se_std / sd
    with np.errstate(divide="ignore"):
        wald_p = 2 * norm.sf(np.abs(beta / se))
    wald_lower, wald_upper = beta - q * se, beta + q * se
    if profile:
        lo, hi, chisq, failed = _profile_bounds(
            firth, b_std, best, se_std, alpha_ci
        )
        if failed:
            warn_unverified(
                "The profile penalised-likelihood search for "
                "coefficient(s) {}".format(failed),
                "a constrained fit stopped short",
                "use the Wald intervals (profile=False)",
            )
        lower, upper = lo / sd, hi / sd
        p_values = chi2.sf(chisq, 1)
    else:
        lower, upper, p_values = wald_lower, wald_upper, wald_p
        chisq = np.full(p, np.nan)
    # The penalised log-likelihood with the information in the
    # covariates' own units: log|I| gains 2 sum(log sd) from the scaling.
    offset = float(np.sum(np.log(sd)))
    result = FirthResult(
        alpha_ci=float(alpha_ci),
        profile=bool(profile),
        lower=np.asarray(lower, dtype=float),
        upper=np.asarray(upper, dtype=float),
        p_values=np.asarray(p_values, dtype=float),
        chisq=np.asarray(chisq, dtype=float),
        wald_lower=wald_lower,
        wald_upper=wald_upper,
        wald_p=wald_p,
        log_likelihood=float(best + offset),
        log_likelihood_null=float(value_null + offset),
        df=p,
    )
    return _model(
        fitter,
        labels,
        validated,
        mean,
        beta,
        se,
        covariance,
        result,
        "verified" if verified else "unverified",
        center,
    )


def _model(
    fitter: CoxPH_,
    labels: npt.NDArray | None,
    validated: list,
    mean: npt.NDArray,
    beta: npt.NDArray,
    se: npt.NDArray,
    covariance: npt.NDArray,
    result: FirthResult,
    maximum: str,
    center: bool,
) -> SemiParametricRegressionModel:
    """The fitted model, as ``CoxPH.fit`` builds it (the Breslow partial
    likelihood, its baselines at the Firth estimate), with ``firth``."""
    from .cox_ph import _baseline_method, _stratified_baselines

    func_generator = fitter._resolve_func_generator(FIRTH_TIES)
    model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
    model.firth = result
    model.p_values = result.p_values.copy()
    model._se = se
    model._covariance = covariance
    model.tie_method = FIRTH_TIES
    model.baseline_method = _baseline_method(FIRTH_TIES)
    model.maximum = maximum
    model.res = None
    model.beta = copy(beta)
    model.params = beta.copy()
    if labels is None:
        _, x, c, n, tl, Z = validated[0]
        Zc = Z - mean
        likelihood_args = (x, Zc, c, n, tl)
        neg_ll, jac = func_generator(*likelihood_args)
        model.neg_ll_of = Rebuilt(
            func_generator, likelihood_args, item=0, built=neg_ll
        )
        model.jac = Rebuilt(func_generator, likelihood_args, item=1, built=jac)
        model._neg_ll = float(neg_ll(beta))
        model._ic_n = ic_sample_size(c, n)
        model._fit_data = {
            "x": np.asarray(x, dtype=float),
            "c": np.asarray(c, dtype=int),
            "n": np.asarray(n, dtype=float),
            "Z": np.asarray(Z, dtype=float),
            "tl": np.asarray(tl, dtype=float),
        }
        x_b, r, d, h0 = fitter.baseline(beta, x, c, n, Zc, tl, FIRTH_TIES)
        moved = False
        if not center:
            [(r, h0)], moved = baselines_at_origin(
                beta, mean, Z, [(r, h0)], maximum
            )
        model.center = np.zeros_like(mean) if moved else mean
        model._fit_center = mean
        model.x = x_b
        model.r = r
        model.d = d
        model.tl = tl
        model.h0 = h0
        model.H0 = h0.cumsum()
        return model
    per_stratum = []
    for s, xs, cs, ns_, tls, Zs in validated:
        Zcs = Zs - mean
        per_stratum.append(
            (s, func_generator(xs, Zcs, cs, ns_, tls), (xs, cs, ns_, Zcs, tls))
        )
    strata_args = (
        func_generator,
        [
            (xs, Zcs, cs, ns_, tls)
            for _, _, (xs, cs, ns_, Zcs, tls) in per_stratum
        ],
    )
    neg_ll, jac = combined_generators(*strata_args)
    model.neg_ll_of = Rebuilt(
        combined_generators, strata_args, item=0, built=neg_ll
    )
    model.jac = Rebuilt(combined_generators, strata_args, item=1, built=jac)
    model._neg_ll = float(neg_ll(beta))
    model._ic_n = ic_sample_size(
        np.concatenate([v[2] for v in validated]),
        np.concatenate([v[3] for v in validated]),
    )
    model.is_stratified = True
    model.strata_labels = list(labels)
    model._fit_center = mean
    _stratified_baselines(
        fitter,
        model,
        beta,
        mean,
        validated,
        per_stratum,
        labels,
        FIRTH_TIES,
        center,
    )
    return model


#: The fix the monotone-likelihood warning of a Cox fit names.
FIRTH_ADVICE = (
    "refit with CoxPH.fit_firth (Firth's penalised partial likelihood), "
    "which keeps the estimates finite and gives profile-likelihood "
    "intervals and tests, or remove or coarsen the covariate"
)
