"""
Confidence bounds for the pseudo-failure-time degradation model.

The classic degradation analysis is a *two-stage* estimator: a path is fitted
to each unit and extrapolated to the threshold to give a pseudo failure time,
then a lifetime distribution is fitted to those pseudo failure times. The
life-model MLE covariance (what ``life_model.cb`` returns) treats the pseudo
failure times as exact data, so it captures only the between-unit scatter and
misses the *first-stage* uncertainty -- the measurement noise and extrapolation
error baked into each pseudo failure time. Its intervals are therefore too
narrow.

Two ways to fix this are provided:

* **analytic** (default) -- a delta-method / generated-regressor correction.
  Each pseudo failure time ``t_i`` carries a variance ``v_i`` propagated from
  the per-unit path-fit covariance through ``inv_path``. Perturbing ``t_i``
  moves the life-model MLE by ``dphi/dt_i = H^{-1} d(score)/dt_i`` (implicit
  function theorem; ``H`` is the life-model observed information). The
  corrected parameter covariance is

      Cov(phi) = H^{-1}  +  sum_i v_i (dphi/dt_i)(dphi/dt_i)',

  i.e. the usual MLE covariance plus the first-stage contribution. Fast (a few
  numerical derivatives, no refitting) and composes to any predicted quantity
  by one more delta step.

* **bootstrap** -- resample units with replacement, rerun the whole pipeline,
  and take percentiles. Slower but assumption-light; a good robustness check on
  the analytic bounds, especially with few units or heavy extrapolation where
  the first-order approximation is weakest.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from surpyval.utils.linalg import bound_signs as _bound_signs
from surpyval.utils.linalg import delta_method_se as _delta_se
from surpyval.utils.linalg import numerical_hessian as _num_hessian
from surpyval.utils.linalg import (
    safe_inv,
)

# -- delta-method helpers shared with the recurrent package (the two
# packages used to carry verbatim copies of these, the drift-prone
# pattern that produced #288) ---------------------------------------------


def _logit_bound(
    p_hat: npt.ArrayLike,
    se: npt.ArrayLike,
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    """Confidence bounds for a probability, taken on the logit scale so they
    stay in ``(0, 1)``. Two-sided puts ``[lower, upper]`` on the last axis."""
    p_arr = np.clip(np.asarray(p_hat, dtype=float), 1e-15, 1.0 - 1e-15)
    alpha, signs = _bound_signs(alpha_ci, bound)
    z = norm.ppf(1.0 - alpha)
    logit = np.log(p_arr / (1.0 - p_arr))
    with np.errstate(divide="ignore", invalid="ignore"):
        se_logit = np.asarray(se, dtype=float) / (p_arr * (1.0 - p_arr))
    lb = logit[..., None] + signs * z * se_logit[..., None]
    cb = 1.0 / (1.0 + np.exp(-lb))
    return cb if bound == "two-sided" else cb[..., 0]


# -- first stage: per-unit pseudo-failure-time variances ------------------


def _inv_path_grad(
    path_model: Any, threshold: float, theta: npt.NDArray
) -> npt.NDArray:
    """Gradient of ``inv_path(threshold; theta)`` w.r.t. the path parameters,
    by central differences."""
    theta = np.asarray(theta, dtype=float)
    g = np.zeros(theta.size)
    for j in range(theta.size):
        h = 1e-6 * max(abs(theta[j]), 1.0)
        up, dn = theta.copy(), theta.copy()
        up[j] += h
        dn[j] -= h
        t_up = float(path_model.inv_path(threshold, *up))
        t_dn = float(path_model.inv_path(threshold, *dn))
        g[j] = (t_up - t_dn) / (2.0 * h)
    return g


def pseudo_time_variances(model: Any) -> npt.NDArray:
    """
    Variance of each unit's pseudo failure time from the first stage.

    For unit ``i`` the least-squares path covariance is
    ``sigma^2 (J_i' J_i)^{-1}`` (``sigma^2`` the pooled measurement variance,
    ``J_i`` the path Jacobian at the fitted parameters), and the pseudo failure
    time ``t_i = inv_path(threshold; theta_i)`` has variance
    ``grad' Cov(theta_i) grad`` by the delta method. Censored units (whose
    "pseudo failure time" is an observed censoring time, not an extrapolation)
    get zero.
    """
    v = np.zeros(len(model.units))
    sigma2 = float(model.measurement_var)
    if not sigma2 > 0:
        return v  # exact path fits: no first-stage uncertainty
    for idx, unit in enumerate(model.units):
        if model.c[idx] == 1:
            continue
        theta = model.path_params[idx]
        x_unit = model.x[model.i == unit]
        J = np.atleast_2d(model.path_model.jacobian(x_unit, *theta))
        cov_theta = sigma2 * safe_inv(J.T @ J)
        g = _inv_path_grad(model.path_model, model.threshold, theta)
        v[idx] = float(g @ cov_theta @ g)
    return v


# -- second stage: corrected life-model covariance ------------------------


def _life_loglik(model: Any, phi: npt.NDArray, t: npt.ArrayLike) -> float:
    """Life-model log-likelihood at parameters ``phi`` and pseudo failure
    times ``t`` (events use the density, censored units the survival)."""
    lm = model.life_model
    gamma = float(getattr(lm, "gamma", 0.0) or 0.0)
    xt = np.asarray(t, dtype=float) - gamma
    tiny = 1e-300
    with np.errstate(divide="ignore", invalid="ignore"):
        logf = np.log(np.clip(lm.dist.df(xt, *phi), tiny, None))
        logs = np.log(np.clip(lm.dist.sf(xt, *phi), tiny, None))
    contrib = np.where(model.c == 0, logf, logs)
    return float(np.sum(contrib))


def life_parameter_covariance(
    model: Any, method: str = "analytic"
) -> npt.NDArray:
    """
    Covariance of the fitted life-model parameters.

    ``method='analytic'`` returns the two-stage-corrected covariance
    ``H^{-1} + sum_i v_i (dphi/dt_i)(dphi/dt_i)'``; the first term alone is the
    ordinary MLE covariance that treats the pseudo failure times as exact.
    """
    if method != "analytic":
        raise ValueError(
            "life_parameter_covariance supports method='analytic'"
        )
    lm = model.life_model
    if getattr(lm, "p", 1.0) != 1.0 or getattr(lm, "f0", 0.0) != 0.0:
        raise ValueError(
            "The analytic correction is implemented for a plain life model "
            "(no limited-failure-population or zero-inflation component); use "
            "method='bootstrap'."
        )
    phi = np.asarray(lm.params, dtype=float)
    t = np.asarray(model.pseudo_failure_times, dtype=float)

    H = -_num_hessian(lambda p: _life_loglik(model, p, t), phi)
    Hinv = safe_inv(H)

    v = pseudo_time_variances(model)
    events = np.where((model.c == 0) & (v > 0))[0]
    cov = Hinv.copy()
    if events.size:
        k = phi.size
        hphi = (np.finfo(float).eps ** (1.0 / 3.0)) * np.maximum(
            np.abs(phi), 1e-2
        )
        # Mixed partials d2 loglik / d phi_a d t_i for each event unit.
        S = np.zeros((k, events.size))
        for col, i in enumerate(events):
            ht = (np.finfo(float).eps ** (1.0 / 3.0)) * max(abs(t[i]), 1e-2)
            for a in range(k):
                pp, pm = phi.copy(), phi.copy()
                pp[a] += hphi[a]
                pm[a] -= hphi[a]
                tp, tm = t.copy(), t.copy()
                tp[i] += ht
                tm[i] -= ht
                d2 = (
                    _life_loglik(model, pp, tp)
                    - _life_loglik(model, pp, tm)
                    - _life_loglik(model, pm, tp)
                    + _life_loglik(model, pm, tm)
                ) / (4.0 * hphi[a] * ht)
                S[a, col] = d2
        # dphi/dt_i = H^{-1} d(score)/dt_i, weighted by v_i.
        dphi = Hinv @ S
        cov = cov + (dphi * v[events]) @ dphi.T
    return cov


# -- public confidence-bound entry points ---------------------------------


def _target(model: Any, x: npt.ArrayLike, on: str) -> Any:
    lm = model.life_model
    gamma = float(getattr(lm, "gamma", 0.0) or 0.0)
    x = np.atleast_1d(np.asarray(x, dtype=float))

    def sf_of(phi: npt.NDArray) -> npt.NDArray:
        return lm.dist.sf(x - gamma, *phi)

    return x, sf_of


def analytic_cb(
    model: Any,
    x: npt.ArrayLike,
    on: str,
    alpha_ci: float,
    bound: str,
) -> npt.NDArray:
    cov = life_parameter_covariance(model, method="analytic")
    phi = np.asarray(model.life_model.params, dtype=float)
    x, sf_of = _target(model, x, on)
    sf_hat = np.asarray(sf_of(phi), dtype=float)
    se = _delta_se(sf_of, phi, cov)

    if bound == "two-sided":
        sf_lo = _logit_bound(sf_hat, se, alpha_ci, "lower")
        sf_hi = _logit_bound(sf_hat, se, alpha_ci, "upper")
        if on in ("sf", "R"):
            return np.stack([sf_lo, sf_hi], axis=-1)
        elif on in ("ff", "F"):
            return np.stack([1.0 - sf_hi, 1.0 - sf_lo], axis=-1)
        else:  # Hf
            return np.stack([-np.log(sf_hi), -np.log(sf_lo)], axis=-1)

    # one-sided: ff and Hf decrease in sf, so flip the tail
    if on in ("sf", "R"):
        return _logit_bound(sf_hat, se, alpha_ci, bound)
    flip = "upper" if bound == "lower" else "lower"
    sf_b = _logit_bound(sf_hat, se, alpha_ci, flip)
    return (1.0 - sf_b) if on in ("ff", "F") else -np.log(sf_b)


def bootstrap_cb(
    model: Any,
    x: npt.ArrayLike,
    on: str,
    alpha_ci: float,
    bound: str,
    n_boot: int,
    seed: "int | None",
    Z: "npt.ArrayLike | None" = None,
) -> npt.NDArray:
    """
    Two-stage bootstrap confidence bounds: resample units with
    replacement and rerun the whole pipeline (per-unit path fit ->
    pseudo failure time -> life fit) on each resample, so the
    first-stage path/extrapolation uncertainty is folded in.

    For an *accelerated* (covariate) model pass the stress ``Z`` to
    evaluate at: each resampled unit carries its stress row and the
    covariate life fit is rerun per resample. The selected path model is
    held fixed across resamples (matching ``path="best"``'s chosen
    model), so the bound reflects life-fit and extrapolation
    variability, not path re-selection. Refits whose curve is not
    finite everywhere are dropped rather than poisoning the quantiles.
    """
    import warnings

    from .degradation_analysis import DegradationAnalysis

    x = np.atleast_1d(np.asarray(x, dtype=float))
    rng = np.random.default_rng(seed)
    method_name = _on_method(on)
    n_units = len(model.units)
    curves = []
    # Each resampled fit may emit the usual small-sample path-covariance
    # warnings; silence them here so a single bootstrap call does not surface
    # hundreds of duplicates.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_boot):
            picks = rng.choice(n_units, size=n_units)
            xs, ys, ids, Zs = [], [], [], []
            for new_id, idx in enumerate(picks):
                mask = model.i == model.units[idx]
                n_meas = int(mask.sum())
                xs.append(model.x[mask])
                ys.append(model.y[mask])
                ids.append(np.full(n_meas, new_id))
                if Z is not None:
                    Zs.append(np.tile(model.Z[idx], (n_meas, 1)))
            try:
                m = DegradationAnalysis.fit(
                    np.concatenate(xs),
                    np.concatenate(ys),
                    np.concatenate(ids),
                    threshold=model.threshold,
                    path=model.path_model,
                    distribution=model._distribution,
                    how=model._how,
                    Z=None if Z is None else np.concatenate(Zs),
                )
                curve_fn = getattr(m, method_name)
                curve = np.asarray(
                    curve_fn(x) if Z is None else curve_fn(x, Z), dtype=float
                )
                if np.isfinite(curve).all():
                    curves.append(curve)
            except Exception:
                continue
    if len(curves) < 2:
        detail = (
            " (a resample may not span enough stress levels to identify "
            "the covariate fit)"
            if Z is not None
            else ""
        )
        raise RuntimeError(
            "The degradation bootstrap produced too few successful refits "
            "to form a confidence bound" + detail + "."
        )
    curves_arr = np.asarray(curves)
    if bound == "two-sided":
        lo = np.quantile(curves_arr, alpha_ci / 2.0, axis=0)
        hi = np.quantile(curves_arr, 1.0 - alpha_ci / 2.0, axis=0)
        return np.stack([lo, hi], axis=-1)
    q = alpha_ci if bound == "lower" else 1.0 - alpha_ci
    return np.quantile(curves_arr, q, axis=0)


def _on_method(on: str) -> str:
    return {"sf": "sf", "R": "sf", "ff": "ff", "F": "ff", "Hf": "Hf"}[on]
