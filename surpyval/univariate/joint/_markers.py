"""Starting values for the joint fit (#891): the marker's path fitted to
each item alone, the spread of those fits, then the survival part with
each item's path plugged in.

1. **Per-item paths.** The path model's own ``fit(ages, readings)`` on
   each item with readings at as many ages as the path has parameters
   (the others are left out of this step). The readings' noise is the
   pooled residual variance of those fits.
2. **The population.** The fitted parameters regressed on the
   (centred) covariates give the means and the covariates' shifts; the
   spread of the varying ones about that, less their sampling variance
   (``sigma^2 (J'J)^-1``), is a moment estimate of ``D``.
3. **Survival plug-in.** Each item's most likely path given its readings
   and that population, as a known time-varying covariate of a Weibull
   proportional hazards fit, gives the survival part.

This two-stage estimate is biased (a predicted path is not the true one,
and the readings stop when an item fails); it is only where the joint
fit starts.
"""

from __future__ import annotations

import warnings
from typing import Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from autograd import value_and_grad
from scipy.optimize import minimize

from surpyval.utils.linalg import psd_floor

from ._likelihood import (
    JointArrays,
    Model,
    _path_params,
    log_cumhaz,
    log_event,
    log_fails_between,
    modes,
    path_jacobian,
)


def per_item_fits(
    path_model: Any, ages: list, values: list
) -> tuple[list, npt.NDArray, list, float]:
    """``(items, fits, jacobians, sigma2)``: the items whose readings the
    path could be fitted to alone, their fitted parameters ``(n, P)``,
    the path's Jacobian at each, and the pooled residual variance."""
    P = len(path_model.parameter_names)
    items, fits, jacs = [], [], []
    rss, dof = 0.0, 0
    for i, (t, y) in enumerate(zip(ages, values)):
        if len(np.unique(t)) < P:
            continue
        try:
            with warnings.catch_warnings(), np.errstate(all="ignore"):
                warnings.simplefilter("ignore")
                theta = np.asarray(path_model.fit(t, y), dtype=float)
        except (ValueError, RuntimeError, np.linalg.LinAlgError):
            continue
        with np.errstate(all="ignore"):
            r = y - np.asarray(path_model.path(t, *theta), dtype=float)
        if not (np.all(np.isfinite(theta)) and np.all(np.isfinite(r))):
            continue
        items.append(i)
        fits.append(theta)
        jacs.append(path_jacobian(path_model, t, np.tile(theta, (len(t), 1))))
        rss += float(r @ r)
        dof += len(t) - P
    all_values = np.concatenate([v for v in values if len(v)])
    if dof > 0 and rss > 0:
        sigma2 = rss / dof
    else:
        sigma2 = 1e-2 * max(float(np.var(all_values)), 1e-12)
    return items, np.array(fits).reshape(-1, P), jacs, sigma2


def marker_start(
    model: Model,
    ages: list,
    values: list,
    Z: npt.NDArray,
    w: npt.NDArray,
) -> dict:
    """The marker part of the start: ``mu``, ``Delta``, ``L`` (the
    Cholesky factor of ``D``) and ``sigma``."""
    layout = model.layout
    path_model = model.path.model
    items, fits, jacs, sigma2 = per_item_fits(path_model, ages, values)
    p = Z.shape[1]
    weights = np.sqrt(w[items])
    # each path parameter regressed on the covariates that shift it
    mu = np.zeros(layout.P)
    Delta = np.zeros((layout.P, p))
    fitted = np.zeros_like(fits)
    for j in range(layout.P):
        cols = np.flatnonzero(layout.shifts[j])
        X = np.column_stack([np.ones(len(items)), Z[items][:, cols]])
        coef, *_ = np.linalg.lstsq(
            X * weights[:, None], fits[:, j] * weights, rcond=None
        )
        mu[j] = coef[0]
        Delta[j, cols] = coef[1:]
        fitted[:, j] = X @ coef
    V = list(layout.varying)
    resid = (fits - fitted)[:, V]
    spread = np.atleast_2d(np.cov(resid.T, aweights=w[items]))
    sampling = []
    for J in jacs:
        info = J[:, V].T @ J[:, V]
        try:
            sampling.append(sigma2 * np.linalg.inv(info))
        except np.linalg.LinAlgError:
            continue
    D = spread - (np.mean(sampling, axis=0) if sampling else 0.0)
    # floored on the scale of each parameter's spread (their units differ:
    # a level and a rate a day), so that no variance is lost to another's
    size = np.sqrt(np.maximum(np.diag(spread), 1e-300))
    D = psd_floor(np.atleast_2d(D) / np.outer(size, size), 1e-2, 1e-2)
    D = D * np.outer(size, size)
    return {
        "mu": mu,
        "Delta": Delta,
        "L": np.linalg.cholesky(D),
        "sigma": np.sqrt(sigma2),
    }


def survival_start(model: Model, arrays: JointArrays, theta: dict) -> dict:
    """``theta`` (the marker part filled in) with the survival part --
    the Weibull ``alpha`` and ``beta``, ``gamma`` and the marker's effect
    ``zeta`` -- from the fit with each item's path taken as its most
    likely one given its readings: its lifetime given that path, from
    its entry age."""
    from surpyval import Weibull

    layout = model.layout
    p = layout.p
    # the hazard's estimated coefficients
    k = layout.n_hazard
    interval = np.isfinite(arrays.U)
    left = interval & (arrays.T <= 0)
    x = np.column_stack(
        [
            np.where(left, arrays.U, arrays.T),
            np.where(interval, arrays.U, arrays.T),
        ]
    )
    c = np.where(arrays.d > 0, 0, np.where(left, -1, np.where(interval, 2, 1)))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            plain = Weibull.fit(
                x,
                c=c,
                n=arrays.w,
                tl=arrays.L,
                tr=arrays.Rt,
            )
            alpha, beta = (float(v) for v in plain.params)
        except Exception:
            alpha, beta = float(np.mean(arrays.T)), 1.0
    if not (np.isfinite(alpha) and np.isfinite(beta)):
        alpha, beta = float(np.mean(arrays.T)), 1.0
    trial = dict(theta, alpha=alpha, beta=beta, gamma=np.zeros(p), zeta=0.0)
    zero = np.zeros(len(arrays.w))
    b, _ = modes(
        trial,
        layout,
        model.path,
        arrays,
        zero,
        zero,
        model.centre,
        readings=True,
    )
    params = np.asarray(
        _path_params(trial, layout, arrays.Z, b[:, None, :]), dtype=float
    )
    entered = arrays.L > 0
    late = np.isfinite(arrays.Rt)

    def objective(v: Any) -> Any:
        th = dict(trial)
        th["alpha"], th["beta"] = anp.exp(v[0]), anp.exp(v[1])
        th["gamma"], th["zeta"] = layout._gamma(v[2 : 2 + k]), v[2 + k]
        H_T = anp.exp(
            log_cumhaz(th, model.path, params, arrays, arrays.T, model.centre)
        )[:, 0]
        ll = (
            arrays.d
            * log_event(th, model.path, params, arrays, model.centre)[:, 0]
            - H_T
        )
        if interval.any():
            ll = (
                ll
                + log_fails_between(
                    th,
                    model.path,
                    params,
                    arrays,
                    arrays.T,
                    arrays.U,
                    model.centre,
                )[:, 0]
            )
        if entered.any():
            H_L = anp.exp(
                log_cumhaz(
                    th, model.path, params, arrays, arrays.L, model.centre
                )
            )[:, 0]
            ll = ll + anp.where(entered, H_L, 0.0)
        if late.any():
            ll = (
                ll
                - log_fails_between(
                    th,
                    model.path,
                    params,
                    arrays,
                    arrays.L,
                    arrays.Rt,
                    model.centre,
                )[:, 0]
            )
        return -anp.sum(arrays.w * ll) / anp.sum(arrays.w)

    v0 = np.concatenate([[np.log(alpha), np.log(beta)], np.zeros(k + 1)])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = minimize(
            value_and_grad(objective), v0, jac=True, method="BFGS"
        )
    v = result.x if np.all(np.isfinite(result.x)) else v0
    out = dict(trial)
    out["alpha"], out["beta"] = float(np.exp(v[0])), float(np.exp(v[1]))
    gamma = np.zeros(p)
    gamma[layout.hazard] = v[2 : 2 + k]
    out["gamma"], out["zeta"] = gamma, float(v[2 + k])
    return out
