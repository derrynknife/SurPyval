"""``fit_penalised`` for the parametric regression families (proportional
hazards, accelerated failure time, proportional odds): the objective, the
solver and the reasons a penalised model has no standard errors are in
:mod:`surpyval.univariate.regression._penalised`."""

from __future__ import annotations

import copy
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import grad as ag_grad
from autograd import hessian as ag_hessian
from scipy.optimize import OptimizeResult

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.univariate.regression._aliasing import dataframe_covariates
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.surpyval_data import SurpyvalData, accepts_surpyval_data
from surpyval.utils.units import takes_unit

from ._covariate_link import CovariateLink
from ._penalised import (
    Penalty,
    check_penalty,
    covariate_scale,
    solve_penalised,
)
from .parametric_regression_model import ParametricRegressionModel


def fit_parametric_penalised(
    fitter: Any,
    kind: str,
    link: CovariateLink,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    t: npt.ArrayLike | None,
    penalizer: Any,
    l1_ratio: float,
    standardise: bool,
    tol: float,
) -> ParametricRegressionModel:
    """The penalised fit of a parametric regression ``fitter`` of family
    ``kind``, whose covariates enter through ``link`` (its
    ``phi_param_map`` filled in here)."""
    from ._fit_skeleton import (
        assemble_regression_model,
        canonical_order,
        check_baseline_support,
        coefficient_map,
        drop_nonfinite_covariates,
    )

    data = SurpyvalData(x, c, n, t, group_and_sort=False)
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 1:
        Z_arr = Z_arr.reshape(-1, 1)
    data, Z_arr = drop_nonfinite_covariates(data, Z_arr)
    order = canonical_order(data, Z_arr)
    if np.any(order != np.arange(order.size)):
        data, Z_arr = data[order], Z_arr[order]
    data.add_covariates(Z_arr)
    check_baseline_support(fitter, data)

    p = Z_arr.shape[1]
    pen, a = check_penalty(penalizer, l1_ratio, standardise, p)
    w = np.asarray(data.n, dtype=float)
    scale = covariate_scale(Z_arr, w, standardise)
    scaled = _intervals_from_the_support(fitter, data)
    scaled.add_covariates(Z_arr / scale)
    N = float(w.sum())

    pmap = coefficient_map(Z_arr, fitter.param_map)
    k = fitter.k_dist
    bounds = (*fitter.bounds, *(((None, None),) * p))
    param_map = {**fitter.param_map, **{nm: j + k for nm, j in pmap.items()}}
    transform, inv_trans, const, _, _ = bounds_convert(
        data.x, bounds, None, param_map
    )
    with np.errstate(all="ignore"):
        baseline = fitter.dist.fit_from_surpyval_data(data).params
    init = np.array([*baseline, *np.zeros(p)], dtype=float)
    init_t = np.asarray(transform(init), dtype=float)

    def objective(theta: Any) -> Any:
        return fitter.neg_ll(scaled, *inv_trans(const(theta))) / N

    gradient = ag_grad(objective)
    curvature = ag_hessian(objective)
    l1, l2 = Penalty(pen, a, bool(standardise), scale).weights()
    with np.errstate(all="ignore"):
        res = solve_penalised(
            lambda th: float(objective(anp.asarray(th))),
            lambda th: np.asarray(gradient(anp.asarray(th)), dtype=float),
            lambda th: np.asarray(curvature(anp.asarray(th)), dtype=float),
            init_t,
            np.arange(k, k + p),
            l1,
            l2,
            tol=tol,
        )
    params = np.array(inv_trans(const(res.x)), dtype=float)
    params[k:] = params[k:] / scale
    with np.errstate(all="ignore"):
        neg_ll = float(fitter.neg_ll(data, *params))
    if not np.isfinite(neg_ll):
        raise ValueError(
            "The penalised fit did not reach a finite likelihood; check "
            "the data, or use a larger `penalizer`."
        )
    link.phi_param_map = pmap
    model = assemble_regression_model(
        fitter,
        kind,
        link,
        data,
        OptimizeResult(x=res.x, fun=res.fun, success=res.verified),
        params,
        bounds,
        pmap,
        {},
        neg_ll=neg_ll,
    )
    model.penalty = Penalty(pen, a, bool(standardise), scale, res.effective_df)
    model.maximum = "verified" if res.verified else "unverified"
    if not res.verified:
        warn_unverified("The penalised fit")
    return model


def _intervals_from_the_support(fitter: Any, data: SurpyvalData) -> Any:
    """A copy of ``data`` for the penalised objective, with each interval
    that starts at (or below) the lower end of the baseline's support
    written as the left-censored row it is: ``F(xr) - F(lower)`` is
    ``F(xr)``. The likelihood is the same, but its second derivatives at
    ``F(0)`` of a Weibull are ``0 * log(0)**2``, nan, and the solver's
    Newton steps and its check of the maximum need the Hessian."""
    lower = float(fitter.dist.support[0])
    c = np.asarray(data.c)
    x = np.asarray(data.x, dtype=float)
    from_lower = np.zeros(c.shape, dtype=bool)
    if np.isfinite(lower) and x.ndim == 2:
        from_lower = (c == 2) & (x[:, 0] <= lower)
    if not from_lower.any():
        return copy.copy(data)
    x = x.copy()
    x[from_lower, 0] = x[from_lower, 1]
    c = np.where(from_lower, -1, c)
    return SurpyvalData(x, c, data.n, data.t, group_and_sort=False)


class PenalisedFitMixin:
    """``fit_penalised`` for a parametric regression fitter; the family
    sets ``_penalised_kind`` and, where its covariate link is not
    ``exp(beta'Z)`` under the name ``NAME_EXP``, ``_penalised_link``."""

    _penalised_kind: str

    def _penalised_link(self) -> CovariateLink:
        from ._fit_skeleton import LogLinearPhi

        return LogLinearPhi(LogLinearPhi.NAME_EXP, {})

    @dataframe_covariates
    @takes_unit
    @accepts_surpyval_data
    def fit_penalised(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        penalizer: float | npt.ArrayLike = 0.1,
        l1_ratio: float = 0.0,
        standardise: bool = True,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        tol: float = 1e-10,
        w: npt.ArrayLike | float | None = None,
    ) -> ParametricRegressionModel:
        """
        Fit the model with a ridge, lasso or elastic-net penalty on the
        covariate coefficients (the baseline distribution's parameters
        are not penalised): for many covariates, or correlated ones.

        The fit minimises the negative log-likelihood divided by the
        number of units ``N`` (the sum of ``n``) plus

        .. math::
            \\sum_j \\lambda_j \\left( \\frac{1 - a}{2} b_j^2
            + a |b_j| \\right),

        with :math:`\\lambda_j` the ``penalizer`` and :math:`a` the
        ``l1_ratio``, as lifelines' ``penalizer`` and ``l1_ratio`` and
        glmnet's ``lambda`` and ``alpha``. A lasso or elastic-net penalty
        sets some coefficients exactly to 0: the L1 part is minimised
        exactly (each coefficient the difference of two non-negative
        parts, by L-BFGS-B, then Newton steps on those not at 0 with the
        optimality conditions of those at 0 checked), not smoothed.
        ``model.maximum`` is ``"verified"`` when those conditions hold.

        Every kind of censoring and truncation of :meth:`fit` is taken:
        left, right and interval censoring, and truncation on the left,
        the right or both. ``N`` counts the rows, weighted by ``n``.
        The model predicts as any fitted model does, but has no
        covariance, standard errors, confidence bounds or p-values:
        penalised estimates are biased towards 0 by design, and
        intervals from the penalised curvature would not have their
        stated coverage; those calls raise an error that says so (its
        ``summary()`` shows them as ``nan``).

        Parameters
        ----------
        x, Z, c, n, t, tl, tr : array_like
            As for :meth:`fit`.
        penalizer : float or array_like, optional
            :math:`\\lambda`: one value, or one per column of ``Z`` (0
            leaves a coefficient unpenalised). Default 0.1.
        l1_ratio : float, optional
            :math:`a`, from 0 (ridge, the default) to 1 (lasso).
        standardise : bool, optional
            ``True`` (the default) penalises the coefficients of the
            covariates divided by their standard deviations
            (count-weighted, divisor ``N``, as glmnet), so the penalty
            does not depend on their units; the coefficients are reported
            on the covariates' own scale either way.
        tol : float, optional
            The optimality tolerance on the gradient of the objective.
        w : array_like, optional
            Not taken yet: weights are refused, with a ``ValueError``. Fit
            the unpenalised model with ``w`` (:meth:`fit`).

        Returns
        -------
        ParametricRegressionModel
            The fitted model, with ``penalty`` recording the
            ``penalizer`` (per column), ``l1_ratio``, ``standardise`` and
            the columns' ``scale``.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import WeibullAFT
        >>> rng = np.random.default_rng(1)
        >>> Z = rng.normal(size=(200, 5))
        >>> x = 10 * rng.weibull(2, 200) * np.exp(-0.5 * Z[:, 0])
        >>> model = WeibullAFT.fit_penalised(x, Z, penalizer=0.05, l1_ratio=1)
        >>> model.params[2:].round(3)
        array([ 0.446, -0.035, -0.007,  0.   ,  0.   ])
        """
        from ._penalised import refuse_weights
        from .regression_data import truncation_window

        refuse_weights(w)
        t = truncation_window(x, t, tl, tr)
        return fit_parametric_penalised(
            self,
            self._penalised_kind,
            self._penalised_link(),
            x,
            Z,
            c,
            n,
            t,
            penalizer,
            l1_ratio,
            standardise,
            tol,
        )
