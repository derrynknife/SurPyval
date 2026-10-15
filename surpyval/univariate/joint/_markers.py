"""Starting values for the joint fit (#891): the marker's mixed model
alone, then the survival part with each item's line plugged in.

1. **Per-item lines.** Ordinary least squares of the readings on
   ``[1, t, z]`` pooled over items gives the fixed effects; a line
   through each item's residuals (items with two or more reading ages)
   gives its random effects, whose spread, less their sampling
   variance, is a moment estimate of ``D``.
2. **Mixed model.** The REML fit of the linear mixed model
   (:func:`surpyval.degradation.population.reml_estimate`, the
   longitudinal part of ``DegradationAnalysis``), started from those
   moments, gives the fixed effects, ``D`` and ``sigma``, and each
   item's best linear unbiased prediction (BLUP) of its line.
3. **Survival plug-in.** A Weibull proportional hazards fit with each
   item's predicted line as a known time-varying covariate gives the
   survival part.

This two-stage estimate is biased (a predicted line is not the true
one, and the readings stop when an item fails); it is only where the
joint fit starts.
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

from ._likelihood import JointArrays, log_cumhaz, log_lifetime


def marker_start(
    ages: list, values: list, Z: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray, float, npt.NDArray]:
    """The mixed model's fixed effects ``[b0, b1, delta...]``, ``D``,
    ``sigma^2`` and every item's BLUP of its random effects ``(N, 2)``,
    from the readings ``ages[i]``, ``values[i]`` of each item and the
    covariates ``Z`` ``(N, p)``."""
    from surpyval.degradation.population import reml_estimate

    read = [k for k, t in enumerate(ages) if len(t)]
    X_all = np.vstack(
        [
            np.column_stack(
                [
                    np.ones(len(ages[k])),
                    ages[k],
                    np.tile(Z[k], (len(ages[k]), 1)),
                ]
            )
            for k in read
        ]
    )
    y_all = np.concatenate([values[k] for k in read])
    fixed, *_ = np.linalg.lstsq(X_all, y_all, rcond=None)
    # per-item lines through the residuals
    lines, sampling, resid_ss, resid_df = [], [], 0.0, 0
    for k in read:
        t = ages[k]
        if len(np.unique(t)) < 2:
            continue
        x = np.column_stack([np.ones(len(t)), t])
        r = (
            values[k]
            - np.column_stack([x, np.tile(Z[k], (len(t), 1))]) @ fixed
        )
        coef, *_ = np.linalg.lstsq(x, r, rcond=None)
        lines.append(coef)
        sampling.append(np.linalg.inv(x.T @ x))
        resid_ss += float(np.sum((r - x @ coef) ** 2))
        resid_df += len(t) - 2
    if resid_df > 0:
        sigma2 = resid_ss / resid_df
    else:
        sigma2 = float(np.var(y_all - X_all @ fixed)) / 2
    sigma2 = max(sigma2, 1e-8 * max(float(np.var(y_all)), 1e-12))
    if len(lines) >= 3:
        spread = np.cov(np.array(lines).T)
        D = spread - sigma2 * np.mean(sampling, axis=0)
    else:
        D = np.diag([max(float(np.var(y_all)) - sigma2, sigma2), sigma2])
    D = psd_floor(np.atleast_2d(D), 1e-4, max(sigma2 * 1e-6, 1e-12))
    # the mixed model, by REML, from those moments
    y_list = [np.asarray(values[k], dtype=float) for k in read]
    x_list = [np.column_stack([np.ones(len(ages[k])), ages[k]]) for k in read]
    a_list = [
        np.column_stack([x, np.tile(Z[k], (len(x), 1))])
        for x, k in zip(x_list, read)
    ]
    try:
        with warnings.catch_warnings(), np.errstate(all="ignore"):
            warnings.simplefilter("ignore")
            gamma, D_reml, s2_reml, _ = reml_estimate(
                y_list, x_list, D, sigma2, a_mat_list=a_list
            )
        if (
            np.all(np.isfinite(gamma))
            and np.all(np.isfinite(D_reml))
            and np.isfinite(s2_reml)
            and s2_reml > 0
        ):
            fixed, D, sigma2 = np.asarray(gamma), D_reml, float(s2_reml)
    except (np.linalg.LinAlgError, ValueError):
        pass  # the moments start the joint fit instead
    D = psd_floor(np.atleast_2d(D), 1e-4, max(sigma2 * 1e-6, 1e-12))
    # each item's BLUP: (X'X / s2 + D^-1)^-1 X'(y - A fixed) / s2
    Dinv = np.linalg.inv(D)
    blups = np.zeros((len(ages), 2))
    for k in read:
        x = np.column_stack([np.ones(len(ages[k])), ages[k]])
        r = (
            values[k]
            - np.column_stack([x, np.tile(Z[k], (len(x), 1))]) @ fixed
        )
        blups[k] = np.linalg.solve(x.T @ x / sigma2 + Dinv, x.T @ r / sigma2)
    return np.asarray(fixed, dtype=float), D, sigma2, blups


def survival_start(
    arrays: JointArrays, theta: dict, blups: npt.NDArray
) -> dict:
    """``theta`` (the marker's part filled in) with the survival part --
    the Weibull ``scale`` and ``shape``, ``gamma`` and the marker's
    effect ``assoc`` -- from the fit with each item's line taken as its
    BLUP: its lifetime given that line, from its entry age."""
    from surpyval import Weibull

    p = arrays.Z.shape[1]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        try:
            plain = Weibull.fit(
                arrays.T,
                c=(1 - arrays.d).astype(int),
                n=arrays.w,
                tl=arrays.L,
            )
            scale, shape = (float(v) for v in plain.params)
        except Exception:
            scale, shape = float(np.mean(arrays.T)), 1.0
    if not (np.isfinite(scale) and np.isfinite(shape)):
        scale, shape = float(np.mean(arrays.T)), 1.0
    entered = arrays.L > 0

    def objective(v: Any) -> Any:
        trial = dict(theta)
        trial["scale"], trial["shape"] = anp.exp(v[0]), anp.exp(v[1])
        trial["gamma"], trial["assoc"] = v[2 : 2 + p], v[2 + p]
        ll = log_lifetime(trial, arrays, blups)
        if entered.any():
            ll = ll + anp.where(
                entered,
                anp.exp(log_cumhaz(trial, arrays, blups, arrays.L)),
                0.0,
            )
        return -anp.sum(arrays.w * ll) / anp.sum(arrays.w)

    v0 = np.concatenate([[np.log(scale), np.log(shape)], np.zeros(p + 1)])
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        result = minimize(
            value_and_grad(objective), v0, jac=True, method="BFGS"
        )
    v = result.x if np.all(np.isfinite(result.x)) else v0
    out = dict(theta)
    out["scale"], out["shape"] = float(np.exp(v[0])), float(np.exp(v[1]))
    out["gamma"], out["assoc"] = np.array(v[2 : 2 + p]), float(v[2 + p])
    return out
