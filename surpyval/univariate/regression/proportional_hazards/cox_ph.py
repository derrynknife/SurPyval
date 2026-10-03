# This code was created for and sponsored by Cartiga (www.cartiga.com).
# Cartiga makes no representations or warranties in connection with the code
# and waives any and all liability in connection therewith. Your use of the
# code constitutes acceptance of these terms.

# Copyright 2022 Cartiga LLC

from __future__ import annotations

import warnings
from copy import copy
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.typing as npt
from numpy.linalg import inv, pinv
from scipy.optimize import OptimizeResult, minimize, root
from scipy.stats import norm

if TYPE_CHECKING:
    import pandas as pd

from surpyval.univariate.information_criteria import ic_sample_size
from surpyval.univariate.nonparametric import (
    FlemingHarrington,
    KaplanMeier,
    NelsonAalen,
    Turnbull,
)
from surpyval.univariate.parametric.fitters import is_local_minimum
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils import (
    _caller_stacklevel,
    check_covariate_rows,
    finite_covariate_mask,
    validate_coxph,
    validate_coxph_df_inputs,
)
from surpyval.utils.fitter_repr import FitterRepr
from surpyval.utils.no_maximum import warn_no_maximum, warn_unverified
from surpyval.utils.pickling import Rebuilt

from .._aliasing import (
    aliased_columns,
    constant_columns,
    covariate_columns,
    expand,
    warn_aliased,
)
from .._fit_skeleton import covariate_center
from ..semi_parametric_regression_model import SemiParametricRegressionModel
from ..tvc_fit import fit_tvc_df

# at_risk_beta_Z, cox_at_risk_mask, efron_jac, efron_log_denominator and
# not_yet_entered are re-exported: they were defined here.
from .cox_likelihood import (  # noqa: F401
    CoxLikelihoodMixin,
    _combine_generators,
    at_risk_beta_Z,
    baseline_at_origin,
    combined_generators,
    cox_at_risk_mask,
    efron_jac,
    efron_log_denominator,
    newton_raphson,
    not_yet_entered,
    strata_labels,
)
from .tvc import handle_tvc, handle_tvc_timeline

nonparametric_dists = {
    "Nelson-Aalen": NelsonAalen,
    "Kaplan-Meier": KaplanMeier,
    "Fleming-Harrington": FlemingHarrington,
    "Turnbull": Turnbull,
}


def _baseline_method(tie_method: str) -> str:
    """The baseline-hazard estimator that goes with a tie method: Efron's
    tie correction for an Efron fit, Breslow's otherwise."""
    return "efron" if str(tie_method).lower() == "efron" else "breslow"


def _sub(a: "npt.ArrayLike | None", mask: npt.NDArray) -> "npt.NDArray | None":
    """Index ``a`` by ``mask``, passing ``None`` through unchanged."""
    if a is None:
        return None
    return np.asarray(a)[mask]


def _cox_aliased(
    info: npt.NDArray,
    Z: npt.NDArray,
    n: npt.NDArray,
    n_events: float,
    strata: "npt.NDArray | None" = None,
) -> npt.NDArray:
    """The columns whose coefficients the partial likelihood cannot
    determine (#409, #476), to be aliased (see
    :mod:`surpyval.univariate.regression._aliasing`).

    ``info`` is the information at ``beta = 0``: the sum over the event
    times of the risk sets' covariate covariance. A direction with no
    information is one in which the covariates do not vary within any
    risk set at an event time, and then the partial likelihood does not
    depend on it at all: a constant column (a Cox model has no
    intercept), one constant within each stratum, collinear columns
    (every level of a factor coded, with no intercept), or every unit at
    risk failing at once. Fitted anyway, a coefficient ran off to 5.5e14
    and every prediction was nan, with a misleading "monotone likelihood"
    warning. ``Z`` are the covariates with counts ``n`` (a column's
    spread over the data is what its information is judged against), and
    ``strata`` the stratum labels; with no event nothing is checked.
    """
    info = np.atleast_2d(np.asarray(info, dtype=float))
    p = info.shape[0]
    if not n_events > 0 or p == 0:
        return np.array([], dtype=int)
    Z = np.asarray(Z, dtype=float)
    n = np.asarray(n, dtype=float).reshape(-1)
    Zc = Z - covariate_center(Z, n)
    spread = n_events * (n @ Zc**2) / n.sum()
    return aliased_columns(
        info, Z.shape[0], constant_columns(Z, strata), spread
    )


def _solve_beta_and_p_values(
    neg_ll: Callable,
    jac: Callable,
    beta_init: npt.NDArray,
    tol: float,
    Z: npt.NDArray,
    n: npt.NDArray,
    n_events: float,
    strata: "npt.NDArray | None" = None,
) -> tuple[Any, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Maximise the partial likelihood by Newton-Raphson
    (:func:`newton_raphson`; the score's root-finder, then BFGS, if that
    fails) and compute Wald p-values from the observed information;
    shared by ``fit`` and ``_fit_stratified`` so the most-patched block
    in this file exists exactly once. The covariates ``Z``, counts ``n``,
    weighted number of events and stratum labels are for the aliasing
    check (:func:`_cox_aliased`).

    Returns ``(res, p_values, se, aliased)``: ``res.x`` has 0 at the
    aliased columns (the coefficients the predictions use), and their
    p-values and standard errors ``se`` are nan. ``res.maximum`` is what
    the search reached, for the model's ``maximum``: ``"no finite
    maximum"`` where the partial likelihood is monotone, else
    ``"verified"`` where the score is zero and the information positive
    definite (``is_local_minimum``, per event), and otherwise
    ``"unverified"``, which is said."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        score_at_start, info_at_start = jac(beta_init)
    p = len(np.atleast_1d(beta_init))
    aliased = _cox_aliased(info_at_start, Z, n, n_events, strata)
    kept = np.setdiff1d(np.arange(p), aliased)
    if aliased.size:
        warn_aliased(
            aliased,
            "the partial likelihood does not depend on them, as they do "
            "not vary within the risk sets at the event times beyond a "
            "combination of the other columns (a constant column -- a Cox "
            "model has no intercept --, one constant within each stratum, "
            "or a linear combination of the others, as the columns of "
            "every level of a factor are)",
        )
        full_neg_ll, full_jac = neg_ll, jac

        def embed(b: npt.NDArray) -> npt.NDArray:
            out = np.zeros(p)
            out[kept] = b
            return out

        def neg_ll(b: npt.NDArray) -> float:
            return full_neg_ll(embed(b))

        def jac(b: npt.NDArray) -> tuple:
            score, hess = full_jac(embed(b))
            score = np.atleast_1d(score)
            hess = np.atleast_2d(hess)
            return score[kept], hess[np.ix_(kept, kept)]

        info_at_start = np.atleast_2d(info_at_start)[np.ix_(kept, kept)]
        score_at_start = np.atleast_1d(score_at_start)[kept]
        beta_init = np.asarray(beta_init, dtype=float)[kept]
        if kept.size == 0:
            res = OptimizeResult(x=np.zeros(p), success=True, fun=0.0)
            # Nothing estimated: the answer is exact
            res.maximum = "verified"
            return res, np.full(p, np.nan), np.full(p, np.nan), aliased
    # Where the likelihood is monotone (below) the coefficients run off
    # towards infinity and the risk-set sums underflow to 0 on the way;
    # the resulting log(0) and 0/0 are that divergence, which is reported
    # by name below, not as a stream of RuntimeWarnings.
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = newton_raphson(
            neg_ll, jac, beta_init, tol, score_at_start, info_at_start
        )
        if res is not None:
            hessian_matrix = res.hess
        else:
            # Newton-Raphson gave up: a singular information (no events),
            # no decrease on halving, or a likelihood with no finite
            # maximum (#392). The score's root-finder (the solver before
            # #516) takes over; ``jac`` returns (score, hessian), hence
            # ``jac=True``.
            res = root(jac, beta_init, jac=True, tol=tol)

            # MINPACK's hybr root-finder can stall on delayed-entry data
            # with staggered risk sets (e.g. the start-stop representation
            # used for time-varying covariates) even though the partial
            # log-likelihood is well behaved there. Fall back to a direct
            # minimisation of the negative partial log-likelihood whenever
            # root-finding fails to converge or lands at a worse point, so
            # such fits still succeed.
            if not res.success:
                fallback = minimize(
                    lambda b: float(neg_ll(b)), beta_init, method="BFGS"
                )
                if float(neg_ll(fallback.x)) < float(neg_ll(res.x)):
                    res = fallback

            hessian_matrix = jac(res.x)[1]
    res.maximum = _maximum_reached(
        res, jac, hessian_matrix, info_at_start, kept, n_events
    )
    # An exactly singular information matrix raises before the
    # pseudo-inverse fallback can run (#259); route it there.
    try:
        var = np.diag(inv(hessian_matrix))
    except np.linalg.LinAlgError:
        var = np.full(len(np.atleast_1d(res.x)), -1.0)
    # Use the pseudo-inverse if the hessian does not have a diagonal that
    # is all positive.
    if np.any(var <= 0):
        var = np.diag(pinv(hessian_matrix))
    # A near-singular information matrix (e.g. a degenerate start-stop
    # design with duplicated rows) can still leave a non-positive
    # variance; the resulting standard error is simply unavailable (nan),
    # which is the correct signal, so suppress the sqrt-of-negative
    # warning rather than emit it. On a monotone likelihood the variance can
    # also round to exactly 0 (it does on some CPUs), so the z-score is
    # infinite: suppress that division warning too, as the fit has already
    # warned of the monotone likelihood.
    with np.errstate(invalid="ignore", divide="ignore"):
        se = np.sqrt(var)
        z_score = res.x / se
    p_values = 2 * (1 - norm.cdf(np.abs(z_score)))
    if aliased.size:
        res.x = embed(res.x)
        p_values = expand(p_values, kept, p)
        se = expand(se, kept, p)
    return res, p_values, se, aliased


def _maximum_reached(
    res: Any,
    jac: Callable,
    info: npt.NDArray,
    info_at_start: npt.NDArray,
    kept: npt.NDArray,
    n_events: float,
) -> str:
    """What the partial-likelihood search reached (see
    :func:`_solve_beta_and_p_values`), with its one warning: no finite
    maximum (:func:`_warn_if_monotone`), else a verified maximum -- the
    score and the information at ``res.x`` (``res.jac`` from
    Newton-Raphson, else ``jac``) pass ``is_local_minimum`` per event --
    or a search that did not reach one."""
    if _warn_if_monotone(info, info_at_start, kept):
        return "no finite maximum"
    score = getattr(res, "jac", None)
    if score is None or not isinstance(res.get("hess"), np.ndarray):
        score = jac(res.x)[0]
    verified = is_local_minimum(
        lambda _: 0.0,  # (only the derivatives are read)
        lambda _: np.atleast_1d(score),
        lambda _: np.atleast_2d(info),
        np.atleast_1d(res.x),
        obj_scale=max(n_events, 1.0),
    )
    if verified:
        return "verified"
    warn_unverified(
        "The partial-likelihood search",
        None if res.success else "it reported: {}".format(res.message),
    )
    return "unverified"


def _warn_if_monotone(
    info: npt.NDArray,
    info_at_start: npt.NDArray,
    columns: "npt.NDArray | None" = None,
) -> bool:
    """Warn when the partial likelihood has no finite maximum.

    When a covariate separates the events from the survivors (every
    failure at each event time has the largest -- or smallest -- value in
    its risk set), the partial likelihood keeps increasing as that
    coefficient grows, and the fit stops wherever the optimiser gave up
    (``beta`` of 35 with a p-value of 1 on such data). The symptom is that
    the information for that coefficient has collapsed: the risk sets'
    weighted covariate variance goes to 0 as the coefficient grows.
    ``columns`` are the columns of ``Z`` the matrices are for (all of
    them by default; the identified ones after aliasing). Returns whether
    it warned.
    """
    d = np.diag(np.atleast_2d(info))
    d0 = np.diag(np.atleast_2d(info_at_start))
    diverged = np.flatnonzero(
        (d0 > 0) & ~(np.nan_to_num(d, nan=0.0) > 1e-8 * d0)
    )
    if diverged.size:
        if columns is not None:
            diverged = np.asarray(columns)[diverged]
        warn_monotone(str(diverged.tolist()))
        return True
    return False


def warn_monotone(which: str) -> None:
    """Warn that the partial likelihood has no finite maximum in the
    coefficients ``which`` names (``"[0]"``, or ``"[0] (cause 'a')"``);
    shared with the Fine-Gray fit, a weighted partial likelihood (#392)."""
    warn_no_maximum(
        "the partial likelihood keeps increasing as coefficient(s) {} "
        "grow without bound, so the estimate is infinite (the covariate "
        "separates the events from the survivors)".format(which),
        "The reported value, its standard error and its p-value are "
        "meaningless",
        "consider removing or coarsening the covariate, or a penalised fit",
    )


class CoxPH_(FitterRepr, CoxLikelihoodMixin):
    """
    The Cox proportional hazards model: a baseline hazard left entirely
    to the data, multiplied by :math:`e^{\\beta' Z}`,

    .. math::
        h(x \\mid Z) = h_0(x)\\, e^{\\beta' Z}.

    The coefficients are estimated from the partial likelihood (with a
    choice of tie handling, Efron's by default) and the baseline by the
    Breslow estimator, with Efron's tie correction after an Efron fit.
    As in R's ``coxph``, the fit centres the covariates on their
    (``n``-weighted) means, which leaves the coefficients unchanged and
    keeps a covariate far from 0 (a year, a date) from overflowing
    :math:`e^{\\beta' Z}`. The baseline is then reported at ``Z = 0``
    (R's ``basehaz(fit, centered = FALSE)``), or, with ``center=True``, at
    the means, stored as the model's ``center``. Supports right censoring, left
    truncation (delayed entry), stratification and time-varying
    covariates in start-stop form; left- and interval-censored data are
    refused, as the partial likelihood has no term for them (use a
    parametric regression model).
    ``CoxPH`` is an instance of this class; its fit methods return a
    :class:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel`.
    """

    #: The ``repr`` (#614)
    fitter_kind = "semi-parametric proportional hazards fitter"

    # Best reference I can find that covers all the
    # possibilities for estimating betas
    # http://www-personal.umich.edu/~yili/lect4notes.pdf

    def baseline(
        self,
        beta: npt.NDArray,
        x: npt.NDArray,
        c: npt.NDArray,
        n: npt.NDArray,
        Z: npt.NDArray,
        tl: "npt.NDArray | None" = None,
        tie_method: str = "breslow",
    ) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
        # Baseline hazard increments at each distinct time -- the hazard of
        # a unit whose covariates are 0 in the ``Z`` given; ``fit`` passes
        # the centred covariates (#459), and moves the result to Z = 0
        # unless ``center=True`` (#463) -- returned with the risk weight
        # ``r`` and the deaths ``d``. Breslow's increment is
        # ``d / r``. With Efron ties (``tie_method="efron"``) the ``m`` tied
        # deaths at a time see the risk set step down, ``r - (l / m) * r_D``
        # for ``l = 0 .. m-1`` with ``r_D`` the tied deaths' own weight, and
        # the increment is ``sum_l 1 / (r - (l / m) * r_D)``: the
        # covariate-weighted Fleming-Harrington estimator, and the baseline
        # that matches the Efron partial likelihood (R's ``survfit.coxph``
        # does the same). Without ties the two are identical.
        #
        # The risk set at each event time ``tau_i``
        # follows ``cox_at_risk_mask`` (entered ``tl < tau_i``, not yet
        # exited ``x >= tau_i``), each row weighted by its count ``n`` and
        # hazard multiplier ``exp(Z'beta)``. Respecting ``tl`` is what makes
        # the baseline correct for left-truncated and time-varying-covariate
        # (start-stop) data. Computed by suffix sums — the risk set is
        # everyone with ``x >= tau_i`` minus the not-yet-entered
        # ``tl >= tau_i`` (valid because ``tl < x`` on every row) — the same
        # subtraction the Efron generator uses, replacing the previous
        # O(K·N) Python loop (#299).

        unique_x = np.unique(x)
        if tl is None:
            tl = np.full(x.shape[0], -np.inf)

        w = n * np.exp(Z @ beta)

        event = c == 0
        d = np.zeros_like(unique_x)
        np.add.at(d, np.searchsorted(unique_x, x[event]), n[event])

        r_exit = np.zeros_like(unique_x)
        np.add.at(r_exit, np.searchsorted(unique_x, x), w)
        r_exit = r_exit[::-1].cumsum()[::-1]

        # Bucket each row at the largest event time <= its entry time; the
        # suffix sum then gives, at each tau_i, the weight not yet entered.
        k = np.searchsorted(unique_x, tl, side="right") - 1
        entered_late = k >= 0
        r_pre = np.zeros_like(unique_x)
        np.add.at(r_pre, k[entered_late], w[entered_late])
        r_pre = r_pre[::-1].cumsum()[::-1]
        r = r_exit - r_pre

        with np.errstate(divide="ignore", invalid="ignore"):
            h0 = d / r
        if str(tie_method).lower() == "efron":
            r_tied = np.zeros_like(unique_x)
            np.add.at(r_tied, np.searchsorted(unique_x, x[event]), w[event])
            for t in np.flatnonzero(d > 1):
                m = int(round(float(d[t])))
                steps = r[t] - (np.arange(m) / m) * r_tied[t]
                h0[t] = np.sum(1.0 / steps)
        return unique_x, r, d, h0

    @dataframe_covariates
    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        strata: npt.ArrayLike | None = None,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """
        Fits Cox Proportional Hazards model to the provided data.

        Parameters
        ----------

        x: array-like
            The observed times of the events.
        Z: array-like
            The covariates of the model, one row per observation. Rows with
            a missing or infinite covariate are dropped, with a warning. A
            column whose coefficient the partial likelihood cannot
            determine -- a constant column (a Cox model has no
            intercept), one constant within each stratum, or a linear
            combination of the others (every level of a factor, with no
            intercept) -- is aliased, as R's ``coxph`` does: the fit runs
            on the other columns, its coefficient and p-value are ``nan``
            (``model.aliased`` lists it), predictions take it as 0, and
            one warning names it (#476).
        c: array-like, optional
            The censoring indicator. 0 if observed (event),
            1 if right-censored. Defaults to all observed. An exactly
            observed time must be finite. Left-censored
            (-1) and interval-censored (2) rows raise a ``ValueError``: the
            partial likelihood has no term for them, so fit such data with
            a parametric regression model (e.g. ``WeibullPH``) instead.
        n: array-like, optional
            The number of observations at each time point.
        tl: array-like, optional
            The left-truncation times of the observations.
        tie_method: str, optional
            The method to use for tie handling. One of ``'efron'``
            (default), ``'breslow'``, ``'exact'`` (the average-over-orderings
            exact partial likelihood, for ties from coarse rounding of
            continuous time) or ``'kalbfleisch-prentice'`` (alias ``'kp'`` --
            the exact discrete/conditional-logistic likelihood, for genuinely
            discrete time). Without ties they all agree. With ties Efron is
            far closer to the exact likelihood than Breslow, which biases
            coefficients towards zero, at almost no extra cost; the baseline
            hazard then uses the matching Efron (Fleming-Harrington style)
            increments. ``'exact'`` removes the remaining bias under heavy
            ties at several times the cost.
        tol: float, optional
            The convergence tolerance. The coefficients are found by
            Newton-Raphson with step-halving on the partial
            log-likelihood (as R's ``coxph`` and lifelines), which stops
            once a step is at most ``tol`` standard errors long (measured
            by the observed information), leaving an error of the order of
            its square. Should Newton-Raphson fail -- as it does where the
            likelihood has no finite maximum -- the score is root-found
            instead, to a relative change in ``beta`` of ``tol``.
        strata: array-like, optional
            Stratum label for each observation. When supplied the model is
            *stratified*: a separate baseline hazard is estimated per stratum
            while the coefficients ``beta`` are shared. The partial likelihood
            is summed within strata (risk sets never cross a stratum boundary),
            which is the standard remedy when proportional hazards fails for a
            nuisance covariate that you would rather not model. Prediction
            (``hf``/``Hf``/``sf``/``ff``/``df``) then takes a ``stratum``
            argument to select that stratum's baseline. Observations with a
            missing label (``None``, ``NaN`` or pandas ``NA``) are dropped,
            with a warning.
        center: bool, optional
            ``False`` (the default) reports the baseline (``h0``, ``H0``,
            each stratum's) at ``Z = 0``; ``True`` reports it at the
            covariate means, stored as ``model.center`` (as R's ``coxph``
            and ``basehaz(fit)``), and ``phi(Z)`` is then relative to them.
            The fit runs on centred covariates either way, so the
            coefficients and predictions are the same; the default refuses,
            with a ``ValueError``, covariates so far from 0 that the
            baseline there over- or underflows.

        Returns
        -------

        model: SemiParametricRegressionModel
            The fitted model: ``params`` (also ``beta``) are the
            coefficients and ``p_values`` their Wald p-values; the
            baseline (``h0``, ``H0``) is that of a unit at ``center``
            (zeros unless ``center=True``). If a
            covariate separates the events from the survivors the partial
            likelihood has no finite maximum; the fit then warns
            ("monotone partial likelihood") and the coefficient is
            meaningless.

        Examples
        --------
        In the Rossi recidivism data ``arrest`` is 1 for a subject
        arrested during follow-up, so the censoring flag is
        ``1 - arrest``:

        >>> from surpyval import CoxPH
        >>> from surpyval.datasets import load_rossi_static
        >>> df = load_rossi_static()
        >>> x, c = df["week"].values, 1 - df["arrest"].values
        >>> Z = df[["fin", "age", "prio"]].values
        >>> model = CoxPH.fit(x, Z, c=c)
        >>> model.params.round(4)
        array([-0.347 , -0.0671,  0.0969])
        >>> model.p_values.round(4)
        array([0.0682, 0.0013, 0.0004])
        >>> model.sf([20, 52], [1, 25, 3]).round(4)
        array([0.9326, 0.7963])
        """
        func_generator = self._resolve_func_generator(tie_method)

        if strata is not None:
            return self._fit_stratified(
                x, Z, c, n, tl, tie_method, tol, strata, func_generator, center
            )

        x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, tie_method)

        # Good initial guess assumes no impact
        beta_init = np.zeros(Z.shape[1])

        # Fitted on centred covariates, so exp(beta'Z) cannot overflow on a
        # column far from 0 (#459); see ``covariate_center``.
        mean = covariate_center(Z, n)
        Zc = Z - mean
        likelihood_args = (x, Zc, c, n, tl)
        neg_ll, jac = func_generator(*likelihood_args)

        res, p_values, se, aliased = _solve_beta_and_p_values(
            neg_ll, jac, beta_init, tol, Z, n, float(n[c == 0].sum())
        )

        model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
        model._neg_ll = float(neg_ll(res.x))
        model.p_values = p_values
        model.se = se
        # Kept as what they are built from, so the model pickles (#573)
        model.neg_ll_of = Rebuilt(
            func_generator, likelihood_args, item=0, built=neg_ll
        )
        # BIC's sample size, the events (R's ``nobs.coxph``, ``nevent``).
        model._ic_n = ic_sample_size(c, n)
        model.jac = Rebuilt(func_generator, likelihood_args, item=1, built=jac)
        model.tie_method = tie_method
        model.baseline_method = _baseline_method(tie_method)
        model.res = res
        model.maximum = res.maximum
        model.beta = copy(res.x)
        model.params = res.x
        if aliased.size:
            # Reported as nan (R's NA); ``res.x`` keeps the 0 the
            # predictions use (#476).
            model.beta = np.array(res.x, dtype=float)
            model.beta[aliased] = np.nan
            model.params = model.beta.copy()

        # Retain the per-observation training data (before ``baseline``
        # reassigns ``x`` to the unique event times) so the model can compute
        # residuals (Schoenfeld, martingale, ...) and the proportional-
        # hazards test.
        model._fit_data = {
            "x": np.asarray(x, dtype=float),
            "c": np.asarray(c, dtype=int),
            "n": np.asarray(n, dtype=float),
            "Z": np.asarray(Z, dtype=float),
            "tl": np.asarray(tl, dtype=float),
        }

        # The baseline of a unit at the centre, moved to Z = 0 by default.
        x, r, d, h0 = self.baseline(res.x, x, c, n, Zc, tl, tie_method)
        if center:
            model.center = mean
        else:
            r, h0 = baseline_at_origin(res.x, mean, Z, r, h0)
            model.center = np.zeros_like(mean)
        # Where the fit centred, for the residuals and diagnostics.
        model._fit_center = mean
        model.x = x
        model.r = r
        model.d = d
        model.tl = tl
        model.h0 = h0
        model.H0 = model.h0.cumsum()

        return model

    def _fit_stratified(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: "npt.ArrayLike | None",
        n: "npt.ArrayLike | None",
        tl: "npt.ArrayLike | None",
        tie_method: str,
        tol: float,
        strata: npt.ArrayLike,
        func_generator: Callable,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """Fit a stratified Cox model (shared ``beta``, per-stratum baseline).

        Each stratum is validated and turned into its own partial-likelihood
        generator; the generators are summed (see :func:`_combine_generators`)
        so the score equations are solved once for the shared coefficients.
        A separate baseline hazard is then estimated within each stratum.
        """
        labels_arr, missing = strata_labels(strata)
        if len(labels_arr) != len(np.atleast_1d(x)):
            raise ValueError("'strata' must have a label for each observation")
        # The per-observation arrays, subset together as rows are dropped.
        obs: list[Any] = [x, c, n, Z, tl]
        if Z is not None:
            # Checked before the per-stratum split, whose boolean mask
            # would otherwise raise a bare IndexError on a Z of the wrong
            # length.
            check_covariate_rows(np.asarray(Z), len(labels_arr))
            # Rows with a missing covariate are dropped here, once, rather
            # than by each stratum's validation (one warning per stratum,
            # each counting only that stratum's rows).
            keep = finite_covariate_mask(np.asarray(Z, dtype=float))
            if not keep.all():
                obs = [_sub(a, keep) for a in obs]
                labels_arr, missing = labels_arr[keep], missing[keep]
        if missing.any():
            # An observation without a stratum has no baseline to belong
            # to; it is dropped, as a row with a missing covariate is.
            if missing.all():
                raise ValueError(
                    "Every stratum label is missing; there is nothing to fit."
                )
            warnings.warn(
                "Dropped {} of {} rows with a missing stratum label.".format(
                    int(missing.sum()), missing.shape[0]
                ),
                UserWarning,
                stacklevel=_caller_stacklevel(),
            )
            obs = [_sub(a, ~missing) for a in obs]
            labels_arr = labels_arr[~missing]
        x_o, c_o, n_o, Z_o, tl_o = obs

        labels = np.unique(labels_arr)
        validated = []
        for s in labels:
            mask = labels_arr == s
            xs, cs, ns_, tls, Zs = validate_coxph(
                _sub(x_o, mask),
                _sub(c_o, mask),
                _sub(n_o, mask),
                _sub(Z_o, mask),
                _sub(tl_o, mask),
                tie_method,
            )
            validated.append((s, xs, cs, ns_, tls, Zs))

        if not validated:
            raise ValueError("no observations to fit")
        n_params = validated[0][5].shape[1]
        # One centre for every stratum, the mean over all the rows (as R's
        # coxph), so the strata's baselines stay comparable (#459).
        mean = covariate_center(
            np.vstack([v[5] for v in validated]),
            np.concatenate([v[3] for v in validated]),
        )
        per_stratum = []
        for s, xs, cs, ns_, tls, Zs in validated:
            Zcs = Zs - mean
            gen = func_generator(xs, Zcs, cs, ns_, tls)
            per_stratum.append((s, gen, (xs, cs, ns_, Zcs, tls)))

        gens = [g for _, g, _ in per_stratum]
        neg_ll, jac = _combine_generators(gens)
        strata_args = (
            func_generator,
            [
                (xs, Zcs, cs, ns_, tls)
                for _, _, (xs, cs, ns_, Zcs, tls) in per_stratum
            ],
        )

        beta_init = np.zeros(n_params)
        res, p_values, se, aliased = _solve_beta_and_p_values(
            neg_ll,
            jac,
            beta_init,
            tol,
            # The covariates, counts and strata, for the aliasing check.
            np.vstack([v[5] for v in validated]),
            np.concatenate([v[3] for v in validated]),
            sum(
                float(data[2][data[1] == 0].sum())
                for _, _, data in per_stratum
            ),
            np.concatenate(
                [np.full(len(v[1]), k) for k, v in enumerate(validated)]
            ),
        )

        model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
        model._neg_ll = float(neg_ll(res.x))
        model.p_values = p_values
        model.se = se
        # Kept as what they are built from, so the model pickles (#573)
        model.neg_ll_of = Rebuilt(
            combined_generators, strata_args, item=0, built=neg_ll
        )
        model._ic_n = ic_sample_size(
            np.concatenate([v[2] for v in validated]),
            np.concatenate([v[3] for v in validated]),
        )
        model.jac = Rebuilt(
            combined_generators, strata_args, item=1, built=jac
        )
        model.tie_method = tie_method
        model.baseline_method = _baseline_method(tie_method)
        model.res = res
        model.maximum = res.maximum
        model.beta = copy(res.x)
        model.params = res.x
        if aliased.size:
            # Reported as nan (R's NA); ``res.x`` keeps the 0 the
            # predictions use (#476).
            model.beta = np.array(res.x, dtype=float)
            model.beta[aliased] = np.nan
            model.params = model.beta.copy()
        model.center = mean if center else np.zeros_like(mean)
        model.is_stratified = True
        model.strata_labels = list(labels)

        # A separate baseline per stratum, each that of a unit at the
        # centre, moved to Z = 0 by default. Prediction selects the
        # stratum's baseline via the ``stratum`` argument to ``hf``/``Hf``.
        Z_all = np.vstack([v[5] for v in validated])
        baselines: dict[Any, dict[str, npt.NDArray]] = {}
        for s, _, (xs, cs, ns_, Zs, tls) in per_stratum:
            bx, br, bd, bh0 = self.baseline(
                res.x, xs, cs, ns_, Zs, tls, tie_method
            )
            if not center:
                br, bh0 = baseline_at_origin(res.x, mean, Z_all, br, bh0)
            baselines[s] = {
                "x": bx,
                "r": br,
                "d": bd,
                "h0": bh0,
                "H0": bh0.cumsum(),
            }
        model.strata_baselines = baselines

        # Expose the first stratum's baseline as the default so generic
        # attribute access (e.g. ``model.x``) still works; correct prediction
        # must pass an explicit ``stratum``.
        first = baselines[labels[0]]
        model.x = first["x"]
        model.r = first["r"]
        model.d = first["d"]
        model.h0 = first["h0"]
        model.H0 = first["H0"]
        model.tl = None

        return model

    def fit_from_df(
        self,
        df: "pd.DataFrame",
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        formula: str | None = None,
        tie_method: str = "efron",
        strata_col: str | None = None,
        tl_col: str | None = None,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """
        Fits a Cox PH model using a pandas dataframe as the input.

        Parameters
        ----------

        df: pandas.DataFrame
            The dataframe containing the data.
        x_col: str
            The column name of the observed times.
        Z_cols: str or list of str, optional
            The column name(s) of the covariates. Give this or ``formula``.
        c_col: str, optional
            The column name of the censoring indicator.
        n_col: str, optional
            The column name of the number of observations at each time point.
        formula: str, optional
            A ``formulaic`` formula for the covariates (e.g.
            ``"age + site"``), instead of ``Z_cols``; categorical columns get
            reference-level coding. Rows with a missing covariate (in
            ``Z_cols`` or a formula column) are dropped, with a warning.
        tie_method: str, optional
            The tie-handling method: ``'efron'`` (default), ``'breslow'``,
            ``'exact'`` or ``'kalbfleisch-prentice'`` (alias ``'kp'``). See
            :meth:`fit`.
        strata_col: str, optional
            The column name of the stratum label. When supplied the model is
            fitted stratified (a separate baseline hazard per stratum, shared
            coefficients); see :meth:`fit`. Rows with a missing label are
            dropped, with a warning.
        tl_col: str, optional
            The column name of the left-truncation (delayed-entry) times,
            passed to :meth:`fit` as ``tl``. A subject enters the risk sets
            only after its entry time.
        center: bool, optional
            Report the baseline at the covariate means (``model.center``)
            instead of at ``Z = 0``; see :meth:`fit`.

        Returns
        -------

        model: SemiParametricRegressionModel
            The fitted model.
        """
        x, c, n, tl, strata, Z, form, feature_names, model_spec = (
            validate_coxph_df_inputs(
                df,
                x_col,
                c_col,
                n_col,
                Z_cols,
                formula,
                tl_col=tl_col,
                strata_col=strata_col,
            )
        )

        with covariate_columns(feature_names, Z, model_spec):
            model = self.fit(
                x,
                Z,
                c,
                n,
                tl=tl,
                tie_method=tie_method,
                strata=strata,
                center=center,
            )
        model.formula = form
        model.feature_names = feature_names
        model._model_spec = model_spec

        return model

    def fit_tvc(
        self,
        i: npt.ArrayLike,
        xl: npt.ArrayLike,
        xr: npt.ArrayLike,
        c: npt.ArrayLike,
        Z: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """
        Fit a Cox model with time-varying covariates in start-stop format.

        Each row is one observation interval ``(xl, xr]`` of a subject
        (identified by ``i``) on which the covariate row ``Z`` is constant;
        ``c`` is ``0`` (event) only on the interval that ends at the subject's
        event and ``1`` (right-censored) otherwise. The rows are validated (see
        :func:`~surpyval.univariate.regression.proportional_hazards.tvc.
        handle_tvc`) and fitted as delayed-entry observations -- exact for the
        Cox partial likelihood.

        Parameters
        ----------
        i, xl, xr, c, Z : array_like
            The start-stop interval data: subject id, interval entry time
            ``xl``, exit time ``xr``, censoring flag ``c`` (``0`` event at
            ``xr``, ``1`` right-censored -- surpyval's convention), and the
            per-interval covariates.
        n : array_like, optional
            Count weight per interval row.
        tie_method : str, optional
            Tie-handling method: ``'efron'`` (default), ``'breslow'``,
            ``'exact'`` or ``'kalbfleisch-prentice'`` (``'kp'``); see
            :meth:`fit`.
        tol : float, optional
            Convergence tolerance; see :meth:`fit`.
        center : bool, optional
            Report the baseline at the covariate means of the interval rows
            (``model.center``) instead of at ``Z = 0``; see :meth:`fit`.

        Returns
        -------
        SemiParametricRegressionModel
            The fitted model, with ``is_tvc`` set. Evaluate it along a
            covariate path with ``sf_tvc`` / ``Hf_tvc`` (or the
            interval-oriented ``predict_tvc``); its cluster-robust standard
            errors cluster the rows by subject.

        Examples
        --------
        Seven subjects; four of them move from ``Z = 0`` to ``Z = 1`` part
        way through follow-up, so they contribute two rows each, and only
        the row ending in an event carries ``c = 0``:

        >>> from surpyval import CoxPH
        >>> from surpyval.univariate.regression import StepSchedule
        >>> i  = [0, 0, 1, 2, 2, 3, 4, 4, 5, 6]
        >>> xl = [0, 2, 0, 0, 1, 0, 0, 3, 0, 0]
        >>> xr = [2, 5, 3, 1, 4, 6, 3, 7, 2, 8]
        >>> c  = [1, 0, 0, 1, 0, 1, 1, 0, 0, 1]
        >>> Z  = [0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
        >>> model = CoxPH.fit_tvc(i, xl, xr, c, Z)
        >>> model.beta.round(4)
        array([1.6982])

        Survival of a unit that switches to ``Z = 1`` at time 2:

        >>> path = StepSchedule.from_changepoints([0, 2], [[0], [1]])
        >>> model.sf_tvc([1, 3, 5], path).round(4)
        array([1.    , 0.6513, 0.3171])
        """
        x, c, n_arr, tl, Z_arr, ident = handle_tvc(i, xl, xr, c, Z, n)
        model = self.fit(
            x=x,
            Z=Z_arr,
            c=c,
            n=n_arr,
            tl=tl,
            tie_method=tie_method,
            tol=tol,
            center=center,
        )
        model.is_tvc = True
        # Subject ids per *internal* (sorted) row, and the permutation from
        # the caller's row order to the internal order: residuals and
        # cluster-robust SEs align with the internal order, so user-supplied
        # per-row labels must be permuted the same way (#259).
        model.tvc_subject_ids = ident
        model.tvc_row_order = np.lexsort(
            (np.asarray(xl, dtype=float), np.asarray(i))
        )
        return model

    def fit_tvc_from_df(
        self,
        df: "pd.DataFrame",
        i_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: str | list[str] | None = None,
        n_col: str | None = None,
        tie_method: str = "efron",
        center: bool = False,
        formula: str | None = None,
    ) -> SemiParametricRegressionModel:
        """
        Fit a time-varying-covariate Cox model from a start-stop DataFrame.

        See :meth:`fit_tvc`; ``Z_cols`` names the covariate column(s) and the
        remaining arguments name the id / ``xl`` / ``xr`` / ``c`` columns.
        ``tie_method`` and ``center`` are as for :meth:`fit`. Instead of
        ``Z_cols``, ``formula`` (as in :meth:`fit_from_df`) gives the
        covariates as a ``formulaic`` formula, which codes categorical
        (e.g. ``"yes"`` / ``"no"``) columns.
        """
        return fit_tvc_df(
            self.fit_tvc,
            df,
            {"i": i_col, "xl": xl_col, "xr": xr_col, "c": c_col},
            Z_cols,
            formula,
            n_col,
            tie_method=tie_method,
            center=center,
        )

    def fit_tvc_timeline(
        self,
        i: npt.ArrayLike,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        tie_method: str = "efron",
        tol: float = 1e-10,
        center: bool = False,
    ) -> SemiParametricRegressionModel:
        """
        Fit a time-varying-covariate Cox model from a covariate *timeline*.

        This is the timeline / ``xicnt``-style alternative to
        :meth:`fit_tvc`'s explicit ``(start, stop]`` intervals. Each subject's
        rows give its covariate history: a covariate value ``Z`` takes effect
        at time ``x`` and holds until the subject's next row, with the terminal
        event / censoring marked on the last row's ``c``. The timeline is
        expanded to start-stop intervals (see
        :func:`~surpyval.univariate.regression.proportional_hazards.tvc.
        handle_tvc_timeline`) and fitted exactly as :meth:`fit_tvc`, so it
        gives an identical fit to the equivalent start-stop data.

        Parameters
        ----------
        i : array_like
            Subject identifier for each timeline row (the ``xicnt`` item id).
        x : array_like
            The time each row's covariate value takes effect. Strictly
            increasing within a subject; the first is the entry
            (delayed-entry) time, the last is the event / censoring time.
        Z : array_like
            The covariate vector effective from this row's ``x``. The value on
            a subject's terminal (last) row is ignored.
        c : array_like
            Censoring status; only each subject's last row is read (``0``
            event, ``1`` right-censored).
        n : array_like, optional
            Per-subject count weight (read from the terminal row).
        tie_method : str, optional
            Tie-handling method: ``'efron'`` (default), ``'breslow'``,
            ``'exact'`` or ``'kalbfleisch-prentice'`` (``'kp'``); see
            :meth:`fit`.
        tol : float, optional
            Convergence tolerance; see :meth:`fit`.
        center : bool, optional
            Report the baseline at the covariate means (``model.center``)
            instead of at ``Z = 0``; see :meth:`fit`.

        Returns
        -------
        SemiParametricRegressionModel
            The fitted model, with ``is_tvc`` set.
        """
        i_ss, xl, xr, c_ss, Z_ss, n_ss = handle_tvc_timeline(i, x, Z, c, n)
        return self.fit_tvc(
            i=i_ss,
            xl=xl,
            xr=xr,
            c=c_ss,
            Z=Z_ss,
            n=n_ss,
            tie_method=tie_method,
            tol=tol,
            center=center,
        )

    def fit_tvc_timeline_from_df(
        self,
        df: "pd.DataFrame",
        i_col: str,
        x_col: str,
        Z_cols: str | list[str] | None,
        c_col: str,
        n_col: str | None = None,
        tie_method: str = "efron",
        center: bool = False,
        formula: str | None = None,
    ) -> SemiParametricRegressionModel:
        """
        Fit a timeline TVC Cox model from a DataFrame.

        See :meth:`fit_tvc_timeline`; ``x_col`` names the change-point time
        column (``x``), ``Z_cols`` the covariate column(s) and ``c_col`` the
        terminal event / censoring column (``0`` event, ``1`` censored).
        ``tie_method`` and ``center`` are as for :meth:`fit`. Instead of
        ``Z_cols`` (pass ``None``), ``formula`` gives the covariates as a
        ``formulaic`` formula, as in :meth:`fit_from_df`.
        """
        return fit_tvc_df(
            self.fit_tvc_timeline,
            df,
            {"i": i_col, "x": x_col, "c": c_col},
            Z_cols,
            formula,
            n_col,
            tie_method=tie_method,
            center=center,
        )


CoxPH = CoxPH_()
