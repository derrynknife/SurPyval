"""The penalised shared-frailty fit, ``FrailtyFitter.fit_penalised`` (see
:mod:`surpyval.univariate.regression._penalised` for the objective, the
solver and why a penalised model has no standard errors).

The search is on the parameters the unpenalised fit searches (the
baseline's on their unconstrained scales, the coefficients, and
``log theta``). ``theta = 0``, the model without a shared frailty, is at
``log theta = -inf``, which no search reaches: where the search heads
there, the fit is also made with ``theta`` held at 0 (the penalised
proportional hazards fit of the same baseline), and the smaller penalised
objective is kept. At ``theta = 0`` the optimality conditions are those
of the coefficients and the baseline, with the objective not falling as
``theta`` leaves 0 (a bound's condition: the derivative in ``theta`` at
least 0).
"""

from __future__ import annotations

from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import grad as ag_grad
from autograd import hessian as ag_hessian

from surpyval.utils.covariates import coefficient_names
from surpyval.utils.no_maximum import warn_unverified
from surpyval.utils.surpyval_data import SurpyvalData

from .._aliasing import fit_columns
from .._fit_skeleton import check_baseline_support
from .._penalised import PenalisedObjective, Penalty, penalised_fit
from .families import lognormal_posterior_mean
from .frailty_fitter import _AUTOGRAD, _make_transforms, grouped_data
from .frailty_model import FrailtyModel

#: Below this the search is taken to head for ``theta = 0``, and the fit
#: with ``theta`` held there is made too (whichever is smaller is kept).
_NEAR_ZERO = 1e-3

#: How far ``theta`` leaves 0 for the check that the objective does not
#: fall there (its natural unit being 1), as the unpenalised fit's check.
_STEP_OFF_ZERO = 1e-6


def fit_frailty_penalised(
    fitter: Any,
    x: Any,
    Z: Any,
    c: Any,
    n: Any,
    groups: Any,
    penalizer: Any,
    l1_ratio: Any,
    standardise: Any,
    tol: float,
) -> FrailtyModel:
    """:meth:`FrailtyFitter.fit_penalised`."""
    x, Zm, c, w, labels, inv = grouped_data(x, Z, c, n, groups)
    assert Zm is not None
    check_baseline_support(
        fitter, SurpyvalData(x, c, w, None, group_and_sort=False)
    )
    k = fitter.k_dist
    p = Zm.shape[1]
    to_unc, to_nat = _make_transforms(fitter.dist, k)
    base = np.asarray(fitter.dist.fit(x=x, c=c, n=w).params, dtype=float)
    start = to_unc(np.concatenate([base, np.zeros(p), [0.5]]), p)
    index = np.arange(k, k + p)

    def neg_ll(Zs: npt.NDArray, nat: Any) -> Any:
        return fitter._neg_ll_natural(nat, x, c, w, Zs, inv, p, _AUTOGRAD)

    def builder(fun: Any, first: npt.NDArray) -> Any:
        # penalised_fit's builder of the objective fun, started at first
        def build(scale: npt.NDArray, N: float) -> PenalisedObjective:
            Zs = Zm / scale

            def f(u: Any) -> Any:
                return fun(Zs, u) / N

            gradient, curvature = ag_grad(f), ag_hessian(f)
            return PenalisedObjective(
                lambda u: float(f(anp.asarray(u))),
                lambda u: np.asarray(gradient(anp.asarray(u)), dtype=float),
                lambda u: np.asarray(curvature(anp.asarray(u)), dtype=float),
                first,
            )

        return build

    def interior(Zs: npt.NDArray, u: Any) -> Any:
        return neg_ll(Zs, to_nat(u, p, _AUTOGRAD))

    def at_zero(Zs: npt.NDArray, u: Any) -> Any:
        # theta held at 0 (the group terms are then -H exactly, the
        # proportional hazards likelihood)
        nat = to_nat(anp.concatenate([u, anp.zeros(1)]), p, _AUTOGRAD)
        return neg_ll(Zs, anp.concatenate([nat[:-1], anp.zeros(1)]))

    options = (penalizer, l1_ratio, standardise, tol)
    with np.errstate(all="ignore"):
        fit = penalised_fit(builder(interior, start), Zm, w, index, *options)
    res, penalty = fit.result, fit.penalty
    u_hat = np.asarray(res.x, dtype=float)
    theta = float(np.exp(u_hat[-1]))
    verified = bool(res.verified)
    if theta < _NEAR_ZERO or not verified:
        with np.errstate(all="ignore"):
            edge = penalised_fit(
                builder(at_zero, u_hat[:-1]), Zm, w, index, *options
            )
        slack = 1e-12 * max(1.0, abs(float(res.fun)))
        if np.isfinite(edge.result.fun) and (
            edge.result.fun <= res.fun + slack or not np.isfinite(res.fun)
        ):
            u_edge = np.asarray(edge.result.x, dtype=float)
            scale = edge.penalty.scale
            N = float(np.sum(w))
            here = float(at_zero(Zm / scale, u_edge)) / N
            off = float(
                interior(
                    Zm / scale,
                    np.concatenate([u_edge, [np.log(_STEP_OFF_ZERO)]]),
                )
            )
            # The objective must not fall as theta leaves 0.
            rises = off / N >= here - 1e-12 * max(1.0, abs(here))
            res = edge.result
            u_hat = np.concatenate([u_edge, [-np.inf]])
            theta = 0.0
            verified = bool(res.verified and rises)
            # theta, held at 0, is counted as the unpenalised fit counts
            # it: one parameter.
            penalty = Penalty(
                edge.penalty.penalizer,
                edge.penalty.l1_ratio,
                edge.penalty.standardise,
                scale,
                edge.penalty.effective_df + 1.0,
            )
    if not verified:
        warn_unverified("The penalised frailty fit")

    scale = penalty.scale
    nat = np.asarray(to_nat(u_hat[:-1].tolist() + [0.0], p), dtype=float)
    dist_params = nat[:k]
    beta = nat[k : k + p] / scale
    params = np.concatenate([dist_params, beta, [theta]])
    neg_ll_value = float(fitter._neg_ll_natural(params, x, c, w, Zm, inv, p))

    n_groups = labels.shape[0]
    H0 = fitter.dist.Hf(x, *dist_params)
    eta = np.exp(Zm @ beta)
    D = np.bincount(inv, weights=w * (c == 0), minlength=n_groups)
    H = np.bincount(inv, weights=w * eta * H0, minlength=n_groups)
    if fitter.family == "lognormal":
        post = lognormal_posterior_mean(D, H, theta)
    else:
        post = (1.0 + D * theta) / (1.0 + H * theta)

    coefs = coefficient_names(
        p, fit_columns(), [*fitter.dist.parameter_names, "theta"]
    )
    model = FrailtyModel()
    model.family = fitter.family
    model.dist = fitter.dist
    model.dist_params = np.asarray(dist_params, float)
    model.beta = np.asarray(beta, float)
    model.theta = theta
    model.k_dist = k
    model.group_labels = list(labels)
    model.frailties = {str(lab): float(v) for lab, v in zip(labels, post)}
    model._covariance = None
    model.parameter_names = [*fitter.dist.parameter_names, *coefs, "theta"]
    model.k = len(model.parameter_names)
    model.n_obs = x.shape[0]
    model.n_events = int((c == 0).sum())
    model.n_events_weighted = float(w[c == 0].sum())
    model.n_obs_weighted = float(w.sum())
    model.n_groups = n_groups
    model._neg_ll = neg_ll_value
    model.maximum = "verified" if verified else "unverified"
    model.penalty = penalty
    model._fit_data = {"x": x, "c": c, "w": w, "Z": Zm, "inv": inv}
    return model
