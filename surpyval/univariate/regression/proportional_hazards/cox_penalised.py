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
    PenalisedObjective,
    Penalty,
    penalised_fit,
    scaled_weights,
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
    units: tuple[npt.NDArray, npt.NDArray] | None = None,
) -> SemiParametricRegressionModel:
    """:meth:`CoxPH_.fit_penalised`. ``units``, ``(Z, n)`` with one row
    per unit (subject), gives the penalty's ``N`` and the covariates'
    standardisation where the rows are not the units (start-stop rows of
    :meth:`CoxPH_.fit_tvc_penalised`; with weights ``w``, each unit's
    ``n`` times its weight as given); ``None`` takes them over the
    rows."""
    tl = _left_truncation(x, tl, tr, t)
    if w is not None:
        return _fit_weighted_penalised(
            fitter,
            x,
            Z,
            c,
            n,
            tl,
            w,
            penalizer,
            l1_ratio,
            standardise,
            tie_method,
            center,
            tol,
            strata,
            units,
        )
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
            units,
        )
    x, c, n, tl, Z = validate_coxph(x, c, n, Z, tl, tie_method)
    from .cox_ph import _require_an_event

    _require_an_event(c, n)
    mean = covariate_center(Z, n)
    Zc = Z - mean
    Z_units, n_units = (Z, n) if units is None else units
    res, beta, penalty = _solve(
        lambda scale: func_generator(x, Zc / scale, c, n, tl),
        Z_units,
        n_units,
        penalizer,
        l1_ratio,
        standardise,
        tol,
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
    generators: Callable[[npt.NDArray], tuple[Callable, Callable]],
    Z: npt.NDArray,
    n: npt.NDArray,
    penalizer: Any,
    l1_ratio: Any,
    standardise: Any,
    tol: float,
) -> tuple[Any, npt.NDArray, Penalty]:
    """Minimise the penalised objective of the partial likelihood
    ``generators(scale)`` (the negative log-likelihood and its ``jac``)
    of the covariates ``Z`` (counts ``n``) divided by ``scale``, through
    the shared :func:`penalised_fit`: the solver's result, the
    coefficients on the covariates' own scale and the :class:`Penalty`."""

    def build(scale: npt.NDArray, N: float) -> PenalisedObjective:
        neg_ll_s, jac_s = generators(scale)
        cached = _Cached(jac_s)
        return PenalisedObjective(
            lambda b: float(neg_ll_s(b)) / N,
            lambda b: np.atleast_1d(cached(b)[0]) / N,
            lambda b: np.atleast_2d(cached(b)[1]) / N,
            np.zeros(Z.shape[1]),
        )

    p = Z.shape[1]
    fit = penalised_fit(
        build, Z, n, np.arange(p), penalizer, l1_ratio, standardise, tol
    )
    if not fit.result.verified:
        warn_unverified("The penalised Cox fit")
    return fit.result, fit.coefficients, fit.penalty


def _fit_weighted_penalised(
    fitter: CoxPH_,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
    w: Any,
    penalizer: Any,
    l1_ratio: float,
    standardise: bool,
    tie_method: str,
    center: bool,
    tol: float,
    strata: npt.ArrayLike | None,
    units: tuple[npt.NDArray, npt.NDArray] | None,
) -> SemiParametricRegressionModel:
    """The penalised Cox fit with weights ``w``: the weighted partial
    likelihood of :meth:`CoxPH_.fit` with ``w`` (R's tie rules with
    weights, :class:`WeightedCox`), over ``N_w`` the sum of ``n w``, with
    the covariates centred and standardised weighted by ``n w``.

    The search is on the weights scaled to sum to the units
    (:func:`scaled_weights`): the partial likelihood and ``N_w`` are both
    linear in the weights, so the objective is the same, and it is then
    the same function for ``w`` and ``c * w``."""
    from surpyval.utils.weights import (
        check_weights,
        effective_sample_size,
        refuse_weighted,
        warn_if_few_effective,
    )

    from ._weighted_cox import WeightedCox
    from .cox_likelihood import baselines_at_origin
    from .cox_ph import _require_an_event

    if strata is not None:
        raise refuse_weighted(
            "A stratified penalised Cox fit",
            "its weighted form is not built yet (as for CoxPH.fit)",
            "Fit each stratum with w, or put the stratum in Z.",
        )
    if tie_method not in ("efron", "breslow"):
        raise refuse_weighted(
            f"tie_method={tie_method!r}",
            "the exact partial likelihoods have no weighted form here",
            "Use tie_method='efron' (the default) or 'breslow'.",
        )
    x_arr = np.atleast_1d(np.asarray(x, dtype=float))
    w_arr = check_weights(w, x_arr.shape[0])
    Z_in = np.asarray(Z, dtype=float)
    Z_rows = Z_in.reshape(-1, 1) if Z_in.ndim <= 1 else Z_in
    if Z_rows.shape[0] != w_arr.shape[0]:
        raise ValueError(
            f"Z has {Z_rows.shape[0]} rows for {w_arr.shape[0]} values of "
            "x: give one row of covariates per value."
        )
    # The weights travel as a last column of Z, so that the rows the
    # validation drops or reorders take their weights with them (as
    # CoxPH.fit with w).
    x, c, n, tl, Z_aug = validate_coxph(
        x, c, n, np.column_stack([Z_rows, w_arr]), tl, tie_method
    )
    Z, w_arr = np.asarray(Z_aug[:, :-1], dtype=float), Z_aug[:, -1]
    _require_an_event(c, n)
    n = np.asarray(n, dtype=float)
    warn_if_few_effective(n, w_arr)
    w_s = scaled_weights(n, w_arr)
    omega = n * w_s
    mean = (omega[:, None] * Z).sum(axis=0) / omega.sum()
    Zc = Z - mean
    if units is None:
        Z_units, n_units = Z, omega
    else:
        # The units' n w as given, on the rows' scale of the weights
        Z_units = units[0]
        n_units = np.asarray(units[1], dtype=float) * (
            n.sum() / np.sum(n * w_arr)
        )

    def generators(scale: npt.NDArray) -> tuple[Callable, Callable]:
        lik = WeightedCox(x, c, n, w_s, tl, Zc / scale, tie_method)
        cached = _Cached(lik.derivatives)

        def neg_ll(b: npt.NDArray) -> float:
            return -cached(np.asarray(b, dtype=float))[0]

        def jac(b: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
            _, score, info = cached(np.asarray(b, dtype=float))
            return -score, info

        return neg_ll, jac

    res, beta, penalty = _solve(
        generators,
        Z_units,
        n_units,
        penalizer,
        l1_ratio,
        standardise,
        tol,
    )
    # The value with the weights as given, as CoxPH.fit with w reports it
    full = WeightedCox(x, c, n, w_arr, tl, Zc, tie_method)
    model = _penalised_model(fitter, res, beta, penalty, tie_method)
    model._neg_ll = float(full.neg_ll(beta))
    model.neg_ll_of = full.neg_ll
    model._ic_n = float(np.sum(n[np.asarray(c) == 0]))
    model._fit_data = {
        "x": np.asarray(x, dtype=float),
        "c": np.asarray(c, dtype=int),
        "n": n,
        "Z": Z,
        "tl": np.asarray(tl, dtype=float),
        "w": w_arr,
    }
    times, r, d, h0 = full.baseline(beta)
    moved = False
    if not center:
        [(r, h0)], moved = baselines_at_origin(
            beta, mean, Z, [(r, h0)], model.maximum
        )
    model.center = np.zeros_like(mean) if moved else mean
    model._fit_center = mean
    model.x = times
    model.r = r
    model.d = d
    model.tl = np.asarray(tl, dtype=float)
    model.h0 = h0
    model.H0 = np.cumsum(h0)
    model.weighted = True
    model.effective_n = effective_sample_size(n, w_arr)
    return model


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
    units: tuple[npt.NDArray, npt.NDArray] | None = None,
) -> SemiParametricRegressionModel:
    """The penalised stratified Cox fit: the partial likelihood summed
    over the strata, one set of coefficients, a baseline per stratum
    (glmnet's ``stratifySurv``). ``N`` is the units over every stratum,
    and the covariates are centred and standardised over all the rows."""
    from .cox_ph import _stratified_baselines, _stratified_rows

    labels, validated = _stratified_rows(x, Z, c, n, tl, tie_method, strata)
    Z_all = np.vstack([v[5] for v in validated])
    n_all = np.concatenate([v[3] for v in validated])
    mean = covariate_center(Z_all, n_all)

    def generators(scale: npt.NDArray) -> tuple[Callable, Callable]:
        scaled = [
            (xs, (Zs - mean) / scale, cs, ns_, tls)
            for _, xs, cs, ns_, tls, Zs in validated
        ]
        return combined_generators(func_generator, scaled)

    Z_units, n_units = (Z_all, n_all) if units is None else units
    res, beta, penalty = _solve(
        generators, Z_units, n_units, penalizer, l1_ratio, standardise, tol
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
