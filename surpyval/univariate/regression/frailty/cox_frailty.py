"""
The shared gamma frailty model with an unspecified (Cox) baseline (#342).

The hazard of observation ``j`` in group ``g`` is

.. math::
    h(t \\mid Z_j, u_g) = u_g \\, h_0(t) \\, e^{\\beta' Z_j},

with ``h_0`` left to the data, as in a Cox model, and the frailties ``u_g``
gamma of mean 1 and variance ``theta``. For a given ``theta`` the
coefficients, the baseline and the frailties are found by EM over the
frailties (Klein 1992; Nielsen, Gill, Andersen and Sorensen 1992):

* E-step: each group's posterior mean frailty, in closed form,
  ``u_g = (D_g + 1/theta) / (A_g + 1/theta)``, with ``D_g`` its events and
  ``A_g`` the sum over its rows of ``exp(beta'Z) H_0(x)``;
* M-step: a Cox fit with offsets ``log u_g`` (``CoxPH``'s partial
  likelihood and Newton-Raphson), then the Breslow baseline with the
  frailties as weights.

Its fixed point is the maximiser of the penalised partial likelihood that
R's ``coxph(... + frailty(id, dist = "gamma"))`` maximises (Therneau,
Grambsch and Pankratz 2003), with Breslow's or Efron's ties: with Efron's,
``A_g`` is the score of the Efron partial likelihood in the group's offset,
so the fixed point is R's Efron fit too. ``theta`` maximises the profile of
R's integrated ("I-") likelihood,

.. math::
    \\ell(\\theta) = PL(\\hat\\beta, \\hat\\omega)
    + \\sum_g \\Big[\\log\\Gamma(D_g + \\nu) - \\log\\Gamma(\\nu)
    + \\nu \\log \\nu - (D_g + \\nu) \\log(D_g + \\nu)
    + \\nu \\hat\\omega_g + D_g\\Big],

``nu = 1 / theta`` and ``omega_g = log u_g``. With Breslow's ties this is
the marginal likelihood of the model with the frailties integrated out and
the baseline at its nonparametric maximum, up to a constant; at
``theta = 0`` it is the Cox partial likelihood.
"""

import warnings
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from surpyval.serialisation import (
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.utils import _caller_stacklevel
from surpyval.utils.data_summary import data_summary
from surpyval.utils.no_maximum import (
    maximum_entry,
    restored_maximum,
    warn_no_maximum,
    warn_unverified,
)
from surpyval.utils.validation import check_option

from .._aliasing import covariate_columns, expand
from .._fit_skeleton import covariate_center
from ..proportional_hazards.cox_likelihood import (
    baseline_at_origin,
    newton_raphson,
)
from ..proportional_hazards.cox_ph import CoxPH
from ..regression_data import (
    design_matrix_from_df,
    restore_covariate_meta,
    serialise_covariate_meta,
)
from ..semi_parametric_regression_model import SemiParametricRegressionModel
from .frailty_fitter import _log_rising_ratio, grouped_data
from .frailty_model import _SharedFrailty

_TIE_METHODS = ("efron", "breslow")
# The search for theta, on its log: between 1e-6 (no detectable frailty;
# the profile is then flat to rounding) and 100.
_LOG_THETA_BOUNDS = (np.log(1e-6), np.log(100.0))
_EM_TOL = 1e-12
_EM_MAX_ITER = 10000
# The step, in log theta, of the profile's second difference that gives
# theta's standard error.
_CURVATURE_STEP = 0.02


class _CoxFrailtyEM:
    """The EM fit of the gamma-frailty Cox model to one data set.

    ``Z`` are the covariates centred on their means, ``w`` the counts and
    ``inv`` each row's group. Solved states (``beta``, ``log u``) are kept
    by ``theta`` to warm-start the next one.
    """

    def __init__(
        self,
        x: npt.NDArray,
        Z: npt.NDArray,
        c: npt.NDArray,
        w: npt.NDArray,
        inv: npt.NDArray,
        n_groups: int,
        tie_method: str,
    ) -> None:
        self.x, self.Z, self.c, self.w, self.inv = x, Z, c, w, inv
        self.G = n_groups
        self.p = Z.shape[1]
        self.tie_method = tie_method
        self.tl = np.full(x.shape[0], -np.inf)
        self.D = np.bincount(inv, weights=w * (c == 0), minlength=n_groups)
        self.generator = CoxPH._resolve_func_generator(tie_method)
        self.solved: dict[float, tuple[npt.NDArray, npt.NDArray]] = {}
        # The coefficients of the Cox fit without frailty, set by the
        # fitter: where EM starts.
        self.cox_beta = np.zeros(self.p)
        self.not_converged = 0
        self.max_iter = _EM_MAX_ITER

    # -- the M-step: CoxPH with offsets -----------------------------------

    def partial_likelihood(
        self, offset: npt.NDArray
    ) -> tuple[Callable, Callable]:
        """CoxPH's partial likelihood in ``beta`` with the offset as a
        last covariate whose coefficient is held at 1: its negative, and
        its score and information in ``beta``."""
        neg_ll, jac_hess = self.generator(
            self.x, np.column_stack([self.Z, offset]), self.c, self.w, self.tl
        )
        p = self.p

        def f(beta: npt.NDArray) -> float:
            return float(neg_ll(np.r_[beta, 1.0]))

        def jac(beta: npt.NDArray) -> tuple:
            score, info = jac_hess(np.r_[beta, 1.0])
            return np.atleast_1d(score)[:p], np.atleast_2d(info)[:p, :p]

        return f, jac

    def m_step(
        self, beta: npt.NDArray, offset: npt.NDArray
    ) -> tuple[npt.NDArray, float]:
        """The coefficients maximising the partial likelihood with the
        offset, from ``beta``, and the negative partial log-likelihood."""
        f, jac = self.partial_likelihood(offset)
        if self.p == 0:
            return beta, f(beta)
        with np.errstate(all="ignore"):
            score, info = jac(beta)
            res = newton_raphson(f, jac, beta, 1e-12, score, info)
            if res is None:
                # As CoxPH falls back where Newton-Raphson gives up (a
                # likelihood with no finite maximum, #392).
                res = minimize(f, beta, method="BFGS")
        return np.asarray(res.x, dtype=float), float(res.fun)

    # -- the E-step ---------------------------------------------------------

    def baseline(
        self, beta: npt.NDArray, offset: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        """CoxPH's baseline (Breslow, or Efron's increments) with the
        frailties as weights: ``(times, risk sums, deaths, increments)``
        at ``Z = center`` and ``u = 1``."""
        return CoxPH.baseline(
            np.r_[beta, 1.0],
            self.x,
            self.c,
            self.w,
            np.column_stack([self.Z, offset]),
            self.tl,
            self.tie_method,
        )

    def group_hazard(
        self, beta: npt.NDArray, offset: npt.NDArray
    ) -> npt.NDArray:
        """Each group's ``A_g``: the sum over its rows of ``w exp(beta'Z)``
        times the row's cumulative baseline hazard, in the form whose
        frailty-weighted total is the partial likelihood's expected count
        (its score in the group's offset is ``D_g - u_g A_g``). For
        Breslow's ties that is the Breslow cumulative hazard at the row's
        time. With Efron's, a row that dies at a tied time ``t`` (``m``
        deaths, risk sum ``r``, deaths' sum ``r_D``) is at risk for the
        fractions ``1 - l/m`` of the ``m`` steps, ``sum_l (1 - l/m) / (r -
        (l/m) r_D)``, where every other row at risk gets the full increment
        ``sum_l 1 / (r - (l/m) r_D)``."""
        times, r, d, h0 = self.baseline(beta, offset)
        k = np.searchsorted(times, self.x)
        H = np.cumsum(h0)[k]
        event = self.c == 0
        if self.tie_method == "efron":
            tied = np.flatnonzero(d > 1)
            if tied.size:
                risk_w = self.w * np.exp(self.Z @ beta + offset)
                r_D = np.zeros_like(times)
                np.add.at(r_D, k[event], risk_w[event])
                own = h0.copy()
                for t in tied:
                    m = int(round(float(d[t])))
                    frac = np.arange(m) / m
                    own[t] = np.sum((1.0 - frac) / (r[t] - frac * r_D[t]))
                H = np.where(event, H - h0[k] + own[k], H)
        weight = self.w * np.exp(self.Z @ beta) * H
        return np.bincount(self.inv, weights=weight, minlength=self.G)

    # -- EM at a given theta ----------------------------------------------

    def start(self, theta: float) -> tuple[npt.NDArray, npt.NDArray]:
        """The solved state nearest ``theta`` (in its log), or no frailty
        and the coefficients of the Cox fit."""
        if not self.solved:
            return self.cox_beta, np.zeros(self.G)
        nearest = min(self.solved, key=lambda t: abs(np.log(t / theta)))
        return self.solved[nearest]

    def update(
        self, theta: float, log_u: npt.NDArray, beta: npt.NDArray
    ) -> tuple[npt.NDArray, npt.NDArray]:
        """One EM step from the log-frailties ``log_u``: the M-step (from
        ``beta``), then the E-step, ``log((D + nu) / (A + nu))``, written
        to stay accurate as ``theta -> 0``, where both sides tend to
        ``nu``. Returns the new log-frailties and the coefficients."""
        beta, _ = self.m_step(beta, log_u[self.inv])
        A = self.group_hazard(beta, log_u[self.inv])
        return np.log1p(self.D * theta) - np.log1p(A * theta), beta

    def em(
        self, theta: float, tol: float = _EM_TOL
    ) -> tuple[npt.NDArray, npt.NDArray, float]:
        """``(beta, log u, negative partial log-likelihood)`` at the EM
        fixed point for ``theta``: where one EM step changes no
        log-frailty by more than ``tol``.

        EM converges linearly, slowly where the frailties carry much of the
        information (a large ``theta``, few events per group), so its steps
        are extrapolated by SQUAREM (Varadhan and Roland 2008, scheme S3):
        from two steps ``r = F(v) - v`` and ``s = F(F(v)) - 2 F(v) + v``,
        the point ``v - 2 a r + a^2 s`` with ``a = -max(1, |r| / |s|)``,
        then one more EM step. An extrapolation that is not finite, or
        whose step is longer than ``r``, is replaced by the two plain
        steps. The fixed point is EM's."""
        beta, log_u = self.start(theta)
        converged = False
        for _ in range(self.max_iter):
            u1, beta = self.update(theta, log_u, beta)
            r = u1 - log_u
            if float(np.max(np.abs(r))) < tol:
                log_u = u1
                converged = True
                break
            u2, beta2 = self.update(theta, u1, beta)
            s = (u2 - u1) - r
            size_r = float(np.linalg.norm(r))
            size_s = float(np.linalg.norm(s))
            jump = None
            if size_s > 0:
                a = -max(1.0, size_r / size_s)
                jump = log_u - 2.0 * a * r + a**2 * s
            log_u, beta = u2, beta2
            if jump is not None and np.all(np.isfinite(jump)):
                u3, beta3 = self.update(theta, jump, beta2)
                if np.all(np.isfinite(u3)) and (
                    np.linalg.norm(u3 - jump) <= size_r
                ):
                    log_u, beta = u3, beta3
        if not converged:
            self.not_converged += 1
        beta, neg_pl = self.m_step(beta, log_u[self.inv])
        self.solved[float(theta)] = (beta, log_u)
        return beta, log_u, neg_pl

    def i_loglik(
        self, theta: float, log_u: npt.NDArray, neg_pl: float
    ) -> float:
        """R's integrated log-likelihood at ``theta`` and the frailties
        ``log u`` of its EM fixed point (see the module docstring), each
        group's term written to stay accurate as ``theta -> 0``:
        ``log_rising(D, theta) - (D + nu) log1p(D theta) + nu log u + D``,
        whose sum over the groups then tends to 0."""
        D = self.D
        nu = 1.0 / theta
        term = (
            _log_rising_ratio(D, theta)
            - D * np.log1p(D * theta)
            - nu * (np.log1p(D * theta) - log_u)
            + D
        )
        return -neg_pl + float(np.sum(term))

    def profile(self, log_theta: float) -> float:
        """The negative integrated log-likelihood at ``exp(log_theta)``."""
        theta = float(np.exp(log_theta))
        _, log_u, neg_pl = self.em(theta)
        return -self.i_loglik(theta, log_u, neg_pl)

    # -- inference ----------------------------------------------------------

    def beta_covariance(
        self, theta: float, beta: npt.NDArray, log_u: npt.NDArray
    ) -> "npt.NDArray | None":
        """The coefficients' covariance at fixed ``theta``: the inverse of
        the penalised partial likelihood's information in the coefficients
        and the log-frailties, its coefficient block (R's ``se(coef)`` with
        ``sparse = FALSE``). The partial likelihood's information is
        CoxPH's, with one indicator column per group; the gamma penalty
        adds ``exp(omega_g) / theta`` to the frailties' diagonal."""
        if self.p == 0:
            return np.zeros((0, 0))
        indicators = np.zeros((self.x.shape[0], self.G))
        indicators[np.arange(self.x.shape[0]), self.inv] = 1.0
        _, jac_hess = self.generator(
            self.x,
            np.column_stack([self.Z, indicators]),
            self.c,
            self.w,
            self.tl,
        )
        with np.errstate(all="ignore"):
            info = np.array(jac_hess(np.r_[beta, log_u])[1], dtype=float)
        idx = np.arange(self.p, self.p + self.G)
        info[idx, idx] += np.exp(log_u) / theta
        try:
            cov = np.linalg.inv(info)[: self.p, : self.p]
        except np.linalg.LinAlgError:
            return None
        return cov if np.all(np.isfinite(cov)) else None

    def theta_variance(self, theta: float) -> float:
        """The variance of ``theta`` from the curvature of the profile
        integrated log-likelihood in ``log theta`` (a second difference),
        by the delta method: ``theta^2 / -d2l/d(log theta)^2``."""
        h = _CURVATURE_STEP
        mid = self.profile(np.log(theta))
        up = self.profile(np.log(theta) + h)
        down = self.profile(np.log(theta) - h)
        curvature = (up - 2.0 * mid + down) / h**2
        if not curvature > 0:
            return float("nan")
        return float(theta**2 / curvature)


class CoxFrailtyFitter:
    """
    The shared gamma frailty model with a Cox (unspecified) baseline: the
    semi-parametric counterpart of :func:`~surpyval.Frailty`, as ``CoxPH``
    is of ``WeibullPH``.

    .. math::
        h(t \\mid Z, u) = u \\, h_0(t) \\, e^{\\beta' Z},

    with the frailty ``u`` shared by every observation of a group, gamma of
    mean 1 and variance ``theta``, and ``h_0`` left to the data. Fitted by
    EM over the frailties with ``theta`` maximising the profile likelihood,
    which gives R's ``coxph(Surv(time, status) ~ ... + frailty(id, dist =
    "gamma"))`` (see :meth:`fit`). ``CoxFrailty`` is an instance.
    """

    def fit(
        self,
        x: Any,
        Z: Any = None,
        c: Any = None,
        n: Any = None,
        groups: Any = None,
        tie_method: str = "efron",
        theta: "float | None" = None,
    ) -> "CoxFrailtyModel":
        """
        Fit the shared gamma frailty Cox model.

        For a given frailty variance ``theta`` the coefficients, the
        baseline and each group's frailty come from EM: the E-step is each
        group's posterior mean frailty (closed form for the gamma), the
        M-step a ``CoxPH`` fit with the log-frailties as offsets and the
        frailty-weighted Breslow baseline (Klein 1992; Nielsen et al.
        1992). ``theta`` maximises the profile of the integrated
        likelihood (R's "I-likelihood"). The fit is that of R's
        ``coxph(Surv(time, status) ~ Z + frailty(groups, dist = "gamma"),
        ties = tie_method)``, whose penalised partial likelihood at a given
        ``theta`` has the EM fit as its maximum (Therneau, Grambsch and
        Pankratz 2003).

        Parameters
        ----------
        x : array_like
            The observed times.
        Z : array_like, optional
            Covariates, one row per observation. Omit for frailty alone.
            Rows with a missing or infinite covariate are dropped, with a
            warning; a column the partial likelihood cannot determine (a
            constant, or a combination of the others) is aliased as in
            ``CoxPH``: its coefficient is ``nan``, with a warning.
        c : array_like, optional
            Censoring flags: ``0`` event, ``1`` right-censored (the only
            two supported). Defaults to all events.
        n : array_like, optional
            Counts of each row. Defaults to 1.
        groups : array_like
            The group (cluster) of each row; at least two. Rows with a
            missing label are dropped, with a warning.
        tie_method : str, optional
            ``"efron"`` (the default, as for every Cox fit) or
            ``"breslow"``: the partial likelihood of the M-step and the
            baseline's increments, as in ``CoxPH``.
        theta : float, optional
            A fixed frailty variance (R's ``frailty(..., theta = )``), in
            place of its maximum likelihood estimate; it then has no
            standard error. ``0`` is the Cox model.

        Returns
        -------
        CoxFrailtyModel
            The coefficients ``beta``, the frailty variance ``theta``, each
            group's posterior frailty and the baseline at ``Z = 0``.

        Examples
        --------
        The kidney catheter data: two infection times for each of 38
        patients, with the patient's age and sex.

        >>> import numpy as np
        >>> from surpyval import CoxFrailty
        >>> from surpyval.datasets import load_kidney
        >>> df = load_kidney()
        >>> Z = np.column_stack([df["age"], df["sex"] == 2])
        >>> model = CoxFrailty.fit(
        ...     df["time"], Z=Z, c=1 - df["status"], groups=df["id"]
        ... )
        >>> model.beta.round(4), round(model.theta, 4)
        (array([ 0.0052, -1.5832]), 0.4078)
        """
        check_option(
            "tie_method",
            tie_method,
            _TIE_METHODS,
            "The frailty fit has no exact or Kalbfleisch-Prentice "
            "likelihood.",
        )
        if theta is not None and not (np.isfinite(theta) and theta >= 0):
            raise ValueError(
                "theta must be a finite, non-negative frailty variance; "
                "got {!r}.".format(theta)
            )
        x, Zm, c, w, labels, inv = grouped_data(x, Z, c, n, groups)
        n_obs = x.shape[0]
        n_groups = labels.shape[0]
        Zfull = np.zeros((n_obs, 0)) if Zm is None else Zm
        p_all = Zfull.shape[1]

        # The Cox fit without frailty: the start of EM, the profile at
        # theta = 0, and the aliasing of columns the partial likelihood
        # cannot determine (CoxPH's, with its warning). Its warnings are
        # this fit's: a monotone partial likelihood (#392) has no maximum
        # with a frailty either, and is said once, here.
        cox = None
        # What the fit reached (``maximum``): the Cox fit's, unless the
        # search for theta or EM says otherwise below.
        maximum = "verified"
        if p_all:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cox = CoxPH.fit(x, Zfull, c, w, tie_method=tie_method)
            maximum = cox.maximum
            for caught_warning in caught:
                message = str(caught_warning.message)
                warnings.warn(
                    message,
                    caught_warning.category,
                    stacklevel=_caller_stacklevel(),
                )
        kept = np.arange(p_all)
        if cox is not None and cox.aliased.size:
            kept = np.setdiff1d(kept, cox.aliased)
        Zk = Zfull[:, kept]
        center = covariate_center(Zk, w) if kept.size else np.zeros(0)
        em = _CoxFrailtyEM(x, Zk - center, c, w, inv, n_groups, tie_method)
        monotone = maximum == "no finite maximum"
        if monotone:
            # The coefficients run off to infinity whatever theta is: the
            # estimates mean nothing (said above), and EM would chase them
            # to its iteration limit at every theta.
            em.max_iter = 20
        em.cox_beta = (
            np.asarray(cox.beta, dtype=float)[kept]
            if cox is not None
            else np.zeros(0)
        )
        no_frailty = -em.m_step(em.cox_beta, np.zeros(n_obs))[1]

        theta_var = float("nan")
        if theta is None:
            res = minimize_scalar(
                em.profile,
                bounds=_LOG_THETA_BOUNDS,
                method="bounded",
                options={"xatol": 1e-7},
            )
            theta_hat = float(np.exp(res.x))
            if res.x - _LOG_THETA_BOUNDS[0] < 1e-3 and -res.fun <= (
                no_frailty + 1e-9
            ):
                # No detectable frailty: the profile is maximal at its
                # boundary, theta = 0, where the model is the Cox model
                # (R reports a theta of 5e-9 there).
                theta_hat = 0.0
            elif _LOG_THETA_BOUNDS[1] - res.x < 1e-3:
                maximum = "no finite maximum"
                warn_no_maximum(
                    "the frailty variance theta runs to the edge of its "
                    "search, 100, with the integrated likelihood still "
                    "increasing (groups that differ more than any finite "
                    "variance explains)",
                    "The reported theta, its standard error and its "
                    "bounds are meaningless",
                    "check for groups whose events all come first, or fix "
                    "theta with theta=",
                )
        else:
            theta_hat = float(theta)

        if theta_hat > 0:
            beta, log_u, neg_pl = em.em(theta_hat, tol=_EM_TOL / 100)
            loglik = em.i_loglik(theta_hat, log_u, neg_pl)
            cov_beta = em.beta_covariance(theta_hat, beta, log_u)
            if theta is None:
                theta_var = em.theta_variance(theta_hat)
        else:
            beta = em.cox_beta
            log_u = np.zeros(n_groups)
            loglik = no_frailty
            cov_beta = em.beta_covariance(1e-12, beta, log_u)
        if em.not_converged and maximum == "verified":
            maximum = "unverified"
            warn_unverified(
                "The Cox frailty fit",
                "the EM iteration over the frailties did not converge "
                "within {} iterations at {} value(s) of theta".format(
                    _EM_MAX_ITER, em.not_converged
                ),
            )

        # The baseline at Z = 0 and u = 1, as CoxPH reports it.
        times, r, d, h0 = em.baseline(beta, log_u[inv])
        if kept.size:
            r, h0 = baseline_at_origin(beta, center, Zk, r, h0)

        model = CoxFrailtyModel()
        model.tie_method = tie_method
        model.beta = expand(beta, kept, p_all) if p_all else np.zeros(0)
        model.theta = theta_hat
        model.x = times
        model.h0 = h0
        model.H0 = np.cumsum(h0)
        model.group_labels = list(labels)
        model.frailties = {
            str(lab): float(u) for lab, u in zip(labels, np.exp(log_u))
        }
        names = ["beta_{}".format(i) for i in range(p_all)] + ["theta"]
        covariance = np.full((p_all + 1, p_all + 1), np.nan)
        if cov_beta is not None and kept.size:
            covariance[np.ix_(kept, kept)] = cov_beta
            covariance[np.ix_(kept, [p_all])] = 0.0
            covariance[np.ix_([p_all], kept)] = 0.0
        covariance[p_all, p_all] = theta_var
        model.covariance = covariance
        model.parameter_names = names
        model.loglik = float(loglik)
        model.loglik_no_frailty = float(no_frailty)
        model.n_obs = n_obs
        model.n_events = int((c == 0).sum())
        model.n_groups = n_groups
        model._data_summary = data_summary(c, w, x=x)
        model._fit_data = {"x": x, "c": c, "n": w, "Z": Zfull}
        model.maximum = maximum
        return model

    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        group_col: str,
        Z_cols: "str | list[str] | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        formula: "str | None" = None,
        tie_method: str = "efron",
        theta: "float | None" = None,
    ) -> "CoxFrailtyModel":
        """Fit from a :class:`pandas.DataFrame` naming the columns.

        Either ``Z_cols`` or ``formula`` describes the covariates (or
        neither, for frailty alone); ``group_col`` names the group column.
        The fitted model predicts from DataFrames of the raw covariates.

        Parameters
        ----------
        df : DataFrame
            The data.
        x_col : str
            The column of times.
        group_col : str
            The column of group labels; rows with a missing label are
            dropped, with a warning.
        Z_cols : str or list of str, optional
            The covariate columns.
        c_col, n_col : str, optional
            The censoring-flag and count columns.
        formula : str, optional
            A formula (formulaic syntax) for the covariates, instead of
            ``Z_cols``.
        tie_method, theta : optional
            As for :meth:`fit`.

        Returns
        -------
        CoxFrailtyModel
            The fitted model.

        Examples
        --------
        >>> from surpyval import CoxFrailty
        >>> from surpyval.datasets import load_kidney
        >>> df = load_kidney()
        >>> df["censored"] = 1 - df["status"]
        >>> model = CoxFrailty.fit_from_df(
        ...     df, x_col="time", c_col="censored", group_col="id",
        ...     formula="age + C(sex)",
        ... )
        >>> model.feature_names
        ['age', 'C(sex)[T.2]']
        """
        x = df[x_col].values
        c = None if c_col is None else df[c_col].values
        n = None if n_col is None else df[n_col].values
        groups = df[group_col].values
        feature_names = None
        model_spec = None
        if Z_cols is None and formula is None:
            Z = None
        else:
            Z, feature_names, model_spec = design_matrix_from_df(
                df, Z_cols=Z_cols, formula=formula
            )
        with covariate_columns(feature_names, Z, model_spec):
            model = self.fit(
                x,
                Z=Z,
                c=c,
                n=n,
                groups=groups,
                tie_method=tie_method,
                theta=theta,
            )
        model.feature_names = feature_names
        model.formula = formula
        model._model_spec = model_spec
        return model


class CoxFrailtyModel(_SharedFrailty):
    """
    A fitted shared gamma frailty model with a Cox baseline, returned by
    :meth:`CoxFrailtyFitter.fit`.

    ``beta`` are the coefficients (hazard ratios ``exp(beta)`` within a
    group, given the frailty), ``theta`` the frailty variance, and
    ``frailties`` each group's posterior mean frailty, keyed by its label
    as a string. ``x``, ``h0`` and ``H0`` are the baseline's times, its
    increments and its cumulative hazard, of a unit at ``Z = 0`` with
    frailty 1 (Breslow's estimator with the frailties as weights, with
    Efron's increments after an Efron fit), as ``CoxPH`` reports them.

    The prediction methods give the *marginal* (population) curve by
    default, ``S(t | Z) = (1 + theta e^{beta'Z} H_0(t))^{-1/theta}``;
    ``group=`` conditions on an observed group's posterior frailty, and
    ``frailty=`` on a given one, ``S = exp(-u e^{beta'Z} H_0(t))``. As
    for ``CoxPH`` the baseline is a step function: ``hf`` and ``df`` are
    the jumps at the latest baseline time at or before ``x``, the survival
    is 1 before the first time and holds its last value after the last.

    ``params`` is ``beta`` then ``theta``, in the order of
    ``parameter_names`` and of ``covariance``. The coefficients' standard
    errors are R's (the inverse of the penalised partial likelihood's
    information at the estimated ``theta``, with every frailty in it --
    ``coxph``'s ``sparse = FALSE``); ``theta``'s is from the curvature of
    its profile likelihood, and its interval (:meth:`param_cb`) is formed
    on the log scale. ``loglik`` is the integrated log-likelihood (R's
    "I-likelihood") and ``loglik_no_frailty`` the Cox partial likelihood,
    its value at ``theta = 0``: twice their difference is the
    likelihood-ratio statistic for a frailty, whose null distribution is
    the 50:50 mixture of 0 and a chi-square on one degree of freedom
    (``theta`` is on its boundary under the null).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import CoxFrailty
    >>> from surpyval.datasets import load_kidney
    >>> df = load_kidney()
    >>> Z = np.column_stack([df["age"], df["sex"] == 2])
    >>> model = CoxFrailty.fit(
    ...     df["time"], Z=Z, c=1 - df["status"], groups=df["id"]
    ... )
    >>> round(model.loglik, 4), round(model.loglik_no_frailty, 4)
    (-181.6386, -184.3446)

    The marginal survival of a 45-year-old woman, and that of a new
    infection of patient 21, the most robust in the data:

    >>> model.sf([30, 100], [45, 1]).round(3)
    array([0.752, 0.577])
    >>> model.sf([30, 100], [45, 1], group=21).round(3)
    array([0.968, 0.936])
    """

    def __init__(self) -> None:
        super().__init__()
        self.kind = "CoxFrailty"
        self.tie_method = "efron"
        self.x: np.ndarray = np.array([])
        self.h0: np.ndarray = np.array([])
        self.H0: np.ndarray = np.array([])
        self.loglik: float = float("nan")
        self.loglik_no_frailty: float = float("nan")
        self._data_summary: "str | None" = None
        self._fit_data: "dict | None" = None

    # -- the step baseline --------------------------------------------------

    def _H0(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return SemiParametricRegressionModel._baseline_step(
            self.x, self.H0, x
        ).reshape(x.shape)

    def _h0(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return SemiParametricRegressionModel._baseline_step(
            self.x, self.h0, x
        ).reshape(x.shape)

    def _baseline_names(self) -> "list[str]":
        return []

    def hf(
        self, x: Any, Z: Any = None, group: Any = None, frailty: Any = None
    ) -> np.ndarray:
        """The hazard's jump at the latest baseline time at or before
        ``x``: the increase of :meth:`Hf` there (marginal by default, or
        given ``group=`` or ``frailty=``). With a step baseline there is
        no hazard rate; as ``CoxPH``'s ``hf``, this is a step size."""
        x = np.asarray(x, dtype=float)
        eta = self._eta(Z)
        u = self._resolve_frailty(group, frailty)
        H0 = self._H0(x)
        before = H0 - self._h0(x)
        return self._cumulative(eta * H0, u) - self._cumulative(
            eta * before, u
        )

    def _concordance_data(self) -> "tuple | None":
        data = self._fit_data
        if data is None:
            return None
        return data["x"], data["c"], data["n"], data["Z"]

    # -- printout and serialisation ----------------------------------------

    def __repr__(self) -> str:
        from .._summary import coefficient_repr, format_table

        out = (
            "Shared-Frailty Cox Regression SurPyval Model"
            "\n============================================"
            "\nBaseline            : unspecified (Cox); {} ties".format(
                self.tie_method
            )
            + f"\nFrailty             : {self._family_line()}"
            + f"\nGroups              : {self.n_groups}"
            + f"  (observations {self.n_obs}, events {self.n_events})"
        )
        if self._data_summary:
            out += "\nData                : " + self._data_summary
        if not self.parameter_names:
            return out
        table = self.summary()
        if self.beta.size:
            out += (
                "\nCoefficients        : exp(coef) is the hazard ratio "
                "given the frailty; Wald 95% intervals\n"
            ) + coefficient_repr(table.loc["coefficients"])
        estimates = {
            "coef": "estimate",
            "se(coef)": "se",
            "coef lower 95%": "lower 95%",
            "coef upper 95%": "upper 95%",
        }
        rows = table.loc["frailty"].rename(columns=estimates)
        rows.index.name = None
        out += (
            "\nFrailty variance    : profile-likelihood standard error; "
            "Wald 95% interval\n"
        ) + format_table(rows, list(estimates.values()))
        out += (
            "\nI-likelihood        : {:.4f} (Cox partial likelihood "
            "{:.4f})".format(self.loglik, self.loglik_no_frailty)
        )
        return out

    def to_dict(self) -> dict:
        """Serialise to a plain, JSON-serialisable ``dict``: the
        coefficients, ``theta``, the frailties and the baseline arrays."""
        out: dict[str, Any] = {
            "model": "CoxFrailtyModel",
            "kind": self.kind,
            "family": self.family,
            "tie_method": self.tie_method,
            "beta": np.asarray(self.beta, float).tolist(),
            "theta": float(self.theta),
            "x": np.asarray(self.x, float).tolist(),
            "h0": np.asarray(self.h0, float).tolist(),
            "H0": np.asarray(self.H0, float).tolist(),
            "parameter_names": list(self.parameter_names),
            "group_labels": [str(g) for g in self.group_labels],
            "frailties": {str(k): float(v) for k, v in self.frailties.items()},
            "n_obs": int(self.n_obs),
            "n_events": int(self.n_events),
            "n_groups": int(self.n_groups),
            "loglik": to_native(self.loglik),
            "loglik_no_frailty": to_native(self.loglik_no_frailty),
            "data_summary": self._data_summary,
            **maximum_entry(self.maximum),
        }
        if self.covariance is not None:
            out["covariance"] = np.asarray(self.covariance, float).tolist()
        serialise_covariate_meta(self, out)
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "CoxFrailtyModel":
        """Rebuild a model from a :meth:`to_dict` dictionary."""
        require_model_tag(model_dict, "CoxFrailtyModel", "a Cox frailty model")
        out = cls()
        out.family = model_dict.get("family", "gamma")
        out.tie_method = model_dict["tie_method"]
        out.beta = np.array(model_dict["beta"], dtype=float)
        out.theta = float(model_dict["theta"])
        out.x = np.array(model_dict["x"], dtype=float)
        out.h0 = np.array(model_dict["h0"], dtype=float)
        out.H0 = np.array(model_dict["H0"], dtype=float)
        out.parameter_names = list(model_dict["parameter_names"])
        out.group_labels = list(model_dict.get("group_labels", []))
        out.frailties = {
            k: float(v) for k, v in model_dict.get("frailties", {}).items()
        }
        out.n_obs = int(model_dict.get("n_obs", 0))
        out.n_events = int(model_dict.get("n_events", 0))
        out.n_groups = int(model_dict.get("n_groups", 0))
        out.loglik = float(model_dict.get("loglik", np.nan))
        out.loglik_no_frailty = float(
            model_dict.get("loglik_no_frailty", np.nan)
        )
        out._data_summary = model_dict.get("data_summary")
        if "covariance" in model_dict:
            out.covariance = np.array(model_dict["covariance"], dtype=float)
        restore_covariate_meta(out, model_dict)
        out.maximum = restored_maximum(model_dict)
        return out


CoxFrailty = CoxFrailtyFitter()
