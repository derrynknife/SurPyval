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
    X = np.column_stack([np.ones(len(items)), Z[items]])
    weights = np.sqrt(w[items])
    coef, *_ = np.linalg.lstsq(
        X * weights[:, None], fits * weights[:, None], rcond=None
    )
    mu = coef[0]
    Delta = coef[1:].T.reshape(layout.P, p)
    V = list(layout.varying)
    resid = (fits - X @ coef)[:, V]
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
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            plain = Weibull.fit(
                arrays.T,
                c=(1 - arrays.d).astype(int),
                n=arrays.w,
                tl=arrays.L,
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

    def objective(v: Any) -> Any:
        th = dict(trial)
        th["alpha"], th["beta"] = anp.exp(v[0]), anp.exp(v[1])
        th["gamma"], th["zeta"] = v[2 : 2 + p], v[2 + p]
        H_T = anp.exp(
            log_cumhaz(th, model.path, params, arrays, arrays.T, model.centre)
        )[:, 0]
        ll = (
            arrays.d
            * log_event(th, model.path, params, arrays, model.centre)[:, 0]
            - H_T
        )
        if entered.any():
            H_L = anp.exp(
                log_cumhaz(
                    th, model.path, params, arrays, arrays.L, model.centre
                )
            )[:, 0]
            ll = ll + anp.where(entered, H_L, 0.0)
        return -anp.sum(arrays.w * ll) / anp.sum(arrays.w)

    v0 = np.concatenate([[np.log(alpha), np.log(beta)], np.zeros(p + 1)])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = minimize(
            value_and_grad(objective), v0, jac=True, method="BFGS"
        )
    v = result.x if np.all(np.isfinite(result.x)) else v0
    out = dict(trial)
    out["alpha"], out["beta"] = float(np.exp(v[0])), float(np.exp(v[1]))
    out["gamma"], out["zeta"] = np.array(v[2 : 2 + p]), float(v[2 + p])
    return out
