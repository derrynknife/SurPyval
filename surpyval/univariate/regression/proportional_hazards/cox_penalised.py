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
    solve_penalised,
)
from ..semi_parametric_regression_model import SemiParametricRegressionModel
from .cox_likelihood import baseline_at_origin

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
) -> SemiParametricRegressionModel:
    """:meth:`CoxPH_.fit_penalised`."""
    func_generator = fitter._resolve_func_generator(tie_method)
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

    likelihood_args = (x, Zc, c, n, tl)
    neg_ll, jac = func_generator(*likelihood_args)
    model = SemiParametricRegressionModel("Cox", "Semi-Parametric")
    model._neg_ll = float(neg_ll(beta))
    model.p_values = np.full(p, np.nan)
    model._se = np.full(p, np.nan)
    model._covariance = None
    model.penalty = Penalty(pen, a, bool(standardise), scale, res.effective_df)
    model.neg_ll_of = Rebuilt(
        func_generator, likelihood_args, item=0, built=neg_ll
    )
    model._ic_n = ic_sample_size(c, n)
    model.jac = Rebuilt(func_generator, likelihood_args, item=1, built=jac)
    model.tie_method = tie_method
    from .cox_ph import _baseline_method

    model.baseline_method = _baseline_method(tie_method)
    model.res = res
    model.maximum = "verified" if res.verified else "unverified"
    model.beta = copy(beta)
    model.params = beta.copy()
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
