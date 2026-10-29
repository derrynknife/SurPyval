"""The penalised Cox fit, ``CoxPH.fit_penalised`` (see
:mod:`surpyval.univariate.regression._penalised` for the objective, the
solver and why a penalised model has no standard errors)."""

from __future__ import annotations

from collections.abc import Callable
from copy import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval.univariate.information_criteria import ic_sample_size
from surpyval.utils import validate_coxph
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.pickling import Rebuilt

from .._fit_skeleton import covariate_center
from .._penalised import (
    Penalty,
    check_penalty,
    covariate_scale,
    refuse_weights,
    solve_penalised,
)
from ..semi_parametric_regression_model import SemiParametricRegressionModel
from .cox_likelihood import baseline_at_origin, combined_generators

if TYPE_CHECKING:
    from .cox_ph import CoxPH_


class _Cached:
    """``jac(b)`` (the score and the Hessian of the negative partial
    log-likelihood) kept for the last ``b``, so the gradient and the
    Hessian at one point cost one evaluation."""

    def __init__(self, jac: Callable) -> None:
        self.jac = jac
        self.at: npt.NDArray | None = None
        self.value: Any = None

    def __call__(self, b: npt.NDArray) -> Any:
        if self.at is None or not np.array_equal(b, self.at):
            self.value = self.jac(b)
            self.at = np.array(b, dtype=float)
        return self.value


def _left_truncation(
    x: npt.ArrayLike,
    tl: npt.ArrayLike | None,
    tr: npt.ArrayLike | None,
    t: npt.ArrayLike | None,
) -> npt.ArrayLike | None:
    """The delayed entry ``tl`` of a penalised Cox fit given its ``tl``,
    ``tr`` or ``t``, refusing truncation on the right."""
    if tr is None and t is None:
        return tl
    from ..regression_data import truncation_window

    window = np.asarray(truncation_window(x, t, tl, tr), dtype=float)
    if window.ndim != 2 or window.shape[1] != 2:
        raise ValueError(
            "t must be an (N, 2) array of [tl, tr] rows; got shape "
            "{}.".format(window.shape)
        )
    if np.any(np.isfinite(window[:, 1])):
        raise ValueError(
            "CoxPH.fit_penalised takes left truncation (delayed entry, tl) "
            "only: the Cox partial likelihood has no term for right "
            "truncation, as for CoxPH.fit. For right-truncated data use a "
            "parametric penalised fit, such as WeibullPH.fit_penalised(x, "
            "Z, c=c, t=t, penalizer=...) (or the AFT and PO families), "
            "which takes every kind of truncation."
        )
    return window[:, 0]


def fit_cox_penalised(
    fitter: CoxPH_,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
    penalizer: Any,
    l1_ratio: float,
    standardise: bool,
    tie_method: str,
    center: bool,
    tol: float,
    strata: npt.ArrayLike | None = None,
    w: Any = None,
    tr: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
) -> SemiParametricRegressionModel:
    """:meth:`CoxPH_.fit_penalised`."""
    refuse_weights(w)
    tl = _left_truncation(x, tl, tr, t)
    func_generator = fitter._resolve_func_generator(tie_method)
    if strata is not None:
        return _fit_stratified_penalised(
            fitter,
            func_generator,
            x,
            Z,
            c,
            n,
            tl,
            strata,
            penalizer,
            l1_ratio,
            standardise,
            tie_method,
            center,
            tol,
        )
    x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, tie_method)
    from .cox_ph import _require_an_event

    _require_an_event(c, n)
    p = Z.shape[1]
    pen, a = check_penalty(penalizer, l1_ratio, standardise, p)
    mean = covariate_center(Z, n)
    scale = covariate_scale(Z, n, standardise)
    Zc = Z - mean
    N = float(np.sum(n))
    neg_ll_s, jac_s = func_generator(x, Zc / scale, c, n, tl)
    res, beta, penalty = _solve(
        neg_ll_s, jac_s, N, pen, a, standardise, scale, tol
    )

    likelihood_args = (x, Zc, c, n, tl)
    neg_ll, jac = func_generator(*likelihood_args)
    model = _penalised_model(fitter, res, beta, penalty, tie_method)
    model._neg_ll = float(neg_ll(beta))
    model.neg_ll_of = Rebuilt(
        func_generator, likelihood_args, item=0, built=neg_ll
    )
    model._ic_n = ic_sample_size(c, n)
    model.jac = Rebuilt(func_generator, likelihood_args, item=1, built=jac)
    model._fit_data = {
        "x": np.asarray(x, dtype=float),
        "c": np.asarray(c, dtype=int),
        "n": np.asarray(n, dtype=float),
        "Z": np.asarray(Z, dtype=float),
        "tl": np.asarray(tl, dtype=float),
    }
    x_b, r, d, h0 = fitter.baseline(beta, x, c, n, Zc, tl, tie_method)
    if center:
        model.center = mean
    else:
        r, h0 = baseline_at_origin(beta, mean, Z, r, h0)
        model.center = np.zeros_like(mean)
    model._fit_center = mean
    model.x = x_b
    model.r = r
    model.d = d
    model.tl = tl
    model.h0 = h0
    model.H0 = model.h0.cumsum()
    return model


def _solve(
    neg_ll_s: Callable,
    jac_s: Callable,
    N: float,
    pen: npt.NDArray,
    a: float,
    standardise: bool,
    scale: npt.NDArray,
    tol: float,
) -> tuple[Any, npt.NDArray, Penalty]:
    """Minimise the penalised objective of the partial likelihood of the
    standardised covariates, ``neg_ll_s`` (with ``jac_s``), over ``N``
    units: the solver's result, the coefficients on the covariates' own
    scale and the :class:`Penalty`."""
    p = pen.size
    cached = _Cached(jac_s)
    l1, l2 = Penalty(pen, a, bool(standardise), scale).weights()
    with np.errstate(all="ignore"):
        res = solve_penalised(
            lambda b: float(neg_ll_s(b)) / N,
            lambda b: np.atleast_1d(cached(b)[0]) / N,
            lambda b: np.atleast_2d(cached(b)[1]) / N,
            np.zeros(p),
            np.arange(p),
            l1,
            l2,
            tol=tol,
        )
    beta = res.x / scale
    if not res.verified:
        warn_unverified("The penalised Cox fit")
    penalty = Penalty(pen, a, bool(standardise), scale, res.effective_df)
    return res, beta, penalty


def _penalised_model(
    fitter: CoxPH_,
    res: Any,
    beta: npt.NDArray,
    penalty: Penalty,
    tie_method: str,
) -> SemiParametricRegressionModel:
    """The model of a penalised Cox fit with what every one holds: no
    standard errors or covariance, the penalty and the coefficients."""
    from .cox_ph import _baseline_method

    p = beta.size
    model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
    model.p_values = np.full(p, np.nan)
    model._se = np.full(p, np.nan)
    model._covariance = None
    model.penalty = penalty
    model.tie_method = tie_method
    model.baseline_method = _baseline_method(tie_method)
    model.res = res
    model.maximum = "verified" if res.verified else "unverified"
    model.beta = copy(beta)
    model.params = beta.copy()
    return model


def _fit_stratified_penalised(
    fitter: CoxPH_,
    func_generator: Callable,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
    strata: npt.ArrayLike,
    penalizer: Any,
    l1_ratio: float,
    standardise: bool,
    tie_method: str,
    center: bool,
    tol: float,
) -> SemiParametricRegressionModel:
    """The penalised stratified Cox fit: the partial likelihood summed
    over the strata, one set of coefficients, a baseline per stratum
    (glmnet's ``stratifySurv``). ``N`` is the units over every stratum,
    and the covariates are centred and standardised over all the rows."""
    from .cox_ph import _stratified_baselines, _stratified_rows

    labels, validated = _stratified_rows(x, Z, c, n, tl, tie_method, strata)
    Z_all = np.vstack([v[5] for v in validated])
    n_all = np.concatenate([v[3] for v in validated])
    p = Z_all.shape[1]
    pen, a = check_penalty(penalizer, l1_ratio, standardise, p)
    mean = covariate_center(Z_all, n_all)
    scale = covariate_scale(Z_all, n_all, standardise)
    N = float(np.sum(n_all))
    scaled = [
        (xs, (Zs - mean) / scale, cs, ns_, tls)
        for _, xs, cs, ns_, tls, Zs in validated
    ]
    neg_ll_s, jac_s = combined_generators(func_generator, scaled)
    res, beta, penalty = _solve(
        neg_ll_s, jac_s, N, pen, a, standardise, scale, tol
    )

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
    model = _penalised_model(fitter, res, beta, penalty, tie_method)
    model._neg_ll = float(neg_ll(beta))
    model.neg_ll_of = Rebuilt(
        combined_generators, strata_args, item=0, built=neg_ll
    )
    model._ic_n = ic_sample_size(
        np.concatenate([v[2] for v in validated]), n_all
    )
    model.jac = Rebuilt(combined_generators, strata_args, item=1, built=jac)
    model.center = mean if center else np.zeros_like(mean)
    model.is_stratified = True
    model.strata_labels = list(labels)
    _stratified_baselines(
        fitter,
        model,
        beta,
        mean,
        validated,
        per_stratum,
        labels,
        tie_method,
        center,
    )
    return model
