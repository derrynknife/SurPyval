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

from surpyval.utils.linalg import (
    cb_link,
)
from surpyval.utils.linalg import delta_method_se as _delta_se
from surpyval.utils.linalg import (
    percentile_bounds,
    safe_inv,
    scale_free_hessian,
    sf_link_bound,
)
from surpyval.utils.rng import as_generator
from surpyval.utils.shapes import covariate_rows
from surpyval.utils.warnings import ignoring_warnings

from ._clock import StressClock, stress_row
from .step_stress import path_time_derivative

# -- first stage: per-unit pseudo-failure-time variances ------------------


def _sd_steps(
    values: npt.NDArray, cov: npt.NDArray, power: float
) -> npt.NDArray:
    """Finite-difference steps ``eps**power`` of each value's standard
    error (``sqrt(cov[j, j])``), which move with the value's unit, so the
    derivatives do not depend on it; ``eps**power * max(|value|, 1e-2)``
    where there is no usable standard error. A step of a fraction of the
    value with a fixed floor is far too long for a value that is small in
    the unit it is written in (a path slope per second)."""
    h = np.finfo(float).eps ** power
    with np.errstate(invalid="ignore"):
        sd = np.sqrt(np.diag(np.asarray(cov, dtype=float)))
    usable = np.isfinite(sd) & (sd > 0)
    plain = h * np.maximum(np.abs(values), 1e-2)
    return np.where(usable, h * np.where(usable, sd, 1.0), plain)


def _inv_path_grad(
    path_model: Any,
    threshold: float,
    theta: npt.NDArray,
    cov_theta: npt.NDArray,
) -> npt.NDArray:
    """Gradient of ``inv_path(threshold; theta)`` w.r.t. the path parameters,
    by central differences in steps of their standard errors (from
    ``cov_theta``, see :func:`_sd_steps`)."""
    theta = np.asarray(theta, dtype=float)
    steps = _sd_steps(theta, cov_theta, 1.0 / 3.0)
    g = np.zeros(theta.size)
    for j in range(theta.size):
        h = steps[j]
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
    "pseudo failure time" is an observed censoring time, not an extrapolation
    -- right censored at the last measurement, or left censored at the first
    for a unit already past the threshold) get zero.
    """
    v = np.zeros(len(model.units))
    sigma2 = float(model.measurement_var)
    if not sigma2 > 0:
        return v  # exact path fits: no first-stage uncertainty
    for idx, unit in enumerate(model.units):
        if model.c[idx] != 0:
            continue
        theta = model.path_params[idx]
        x_unit = model.x[model.i == unit]
        J = np.atleast_2d(model.path_model.jacobian(x_unit, *theta))
        cov_theta = sigma2 * safe_inv(J.T @ J)
        g = _inv_path_grad(model.path_model, model.threshold, theta, cov_theta)
        v[idx] = float(g @ cov_theta @ g)
    return v


# -- second stage: corrected life-model covariance ------------------------


def _life_loglik(model: Any, phi: npt.NDArray, t: npt.ArrayLike) -> float:
    """Life-model log-likelihood at parameters ``phi`` and pseudo failure
    times ``t`` (events use the density, right-censored units the
    survival, and units already past the threshold at their first
    measurement -- left censored there -- the CDF)."""
    lm = model.life_model
    gamma = float(getattr(lm, "gamma", 0.0) or 0.0)
    xt = np.asarray(t, dtype=float) - gamma
    tiny = 1e-300
    with np.errstate(divide="ignore", invalid="ignore"):
        logf = np.log(np.clip(lm.dist.df(xt, *phi), tiny, None))
        logs = np.log(np.clip(lm.dist.sf(xt, *phi), tiny, None))
        logF = np.log(np.clip(lm.dist.ff(xt, *phi), tiny, None))
    contrib = np.where(model.c == 0, logf, np.where(model.c == -1, logF, logs))
    return float(np.sum(contrib))


def _generated_regressor_term(
    loglik: Any,
    phi: npt.NDArray,
    Hinv: npt.NDArray,
    t: npt.NDArray,
    affected: npt.NDArray,
) -> npt.NDArray:
    """
    ``dphi/dt`` for the units ``affected``, one column each: ``H^{-1}``
    times the mixed partials ``d2 loglik / d phi d t_i`` (implicit function
    theorem). The partials are central differences in steps of the
    parameters' standard errors (from ``Hinv``) and of each time, so they
    do not depend on the unit of time.
    """
    k = phi.size
    hphi = _sd_steps(phi, Hinv, 1.0 / 4.0)
    S = np.zeros((k, affected.size))
    for col, i in enumerate(affected):
        ht = (np.finfo(float).eps ** (1.0 / 4.0)) * (
            abs(t[i]) if t[i] != 0 else 1e-2
        )
        tp, tm = t.copy(), t.copy()
        tp[i] += ht
        tm[i] -= ht
        for a in range(k):
            pp, pm = phi.copy(), phi.copy()
            pp[a] += hphi[a]
            pm[a] -= hphi[a]
            S[a, col] = (
                loglik(pp, tp)
                - loglik(pp, tm)
                - loglik(pm, tp)
                + loglik(pm, tm)
            ) / (4.0 * hphi[a] * ht)
    return Hinv @ S


def _check_plain_life_model(lm: Any) -> None:
    if getattr(lm, "lfp_p", 1.0) != 1.0 or getattr(lm, "f0", 0.0) != 0.0:
        raise ValueError(
            "The analytic correction is implemented for a plain life model "
            "(no limited-failure-population or zero-inflation component); use "
            "method='bootstrap'."
        )


def _plain_information_inverse(model: Any, phi: npt.NDArray) -> npt.NDArray:
    """The inverse observed information of a plain life model at ``phi``,
    the pseudo failure times taken as exact. In steps found from the
    likelihood's curvature, so it does not depend on the unit of time."""
    t = np.asarray(model.pseudo_failure_times, dtype=float)
    H = scale_free_hessian(lambda p: -_life_loglik(model, p, t), phi)
    return safe_inv(H)


def life_parameter_covariance(
    model: Any, method: str = "analytic"
) -> npt.NDArray:
    """
    Covariance of the fitted life-model parameters.

    ``method='analytic'`` returns the two-stage-corrected covariance
    ``H^{-1} + sum_i v_i (dphi/dt_i)(dphi/dt_i)'``; the first term alone is the
    ordinary MLE covariance that treats the pseudo failure times as exact.
    An accelerated (covariate) model gets the same correction on its
    regression parameters (:func:`regression_parameter_covariance`), and a
    step-stress model the joint one of its life parameters and its clock
    (:func:`clock_parameter_covariance`).
    """
    if method != "analytic":
        raise ValueError(
            "life_parameter_covariance supports method='analytic'"
        )
    if getattr(model, "acceleration", None) == "clock":
        return clock_parameter_covariance(model)
    if model.is_accelerated:
        return regression_parameter_covariance(model)
    lm = model.life_model
    _check_plain_life_model(lm)
    phi = np.asarray(lm.params, dtype=float)
    t = np.asarray(model.pseudo_failure_times, dtype=float)
    Hinv = _plain_information_inverse(model, phi)

    v = pseudo_time_variances(model)
    events = np.where((model.c == 0) & (v > 0))[0]
    cov = Hinv.copy()
    if events.size:

        def loglik(p: npt.NDArray, times: npt.NDArray) -> float:
            return _life_loglik(model, p, times)

        dphi = _generated_regressor_term(loglik, phi, Hinv, t, events)
        cov = cov + (dphi * v[events]) @ dphi.T
    return cov


# -- accelerated (covariate) life fits ------------------------------------


class _RegressionLife:
    """
    The regression life model of an accelerated (covariate) degradation
    model as a function of its estimated parameters: the free ones (not
    held fixed, not aliased), with the rest held at their fitted values.
    """

    def __init__(self, model: Any) -> None:
        lm = model.life_model
        self.lm = lm
        names = list(lm.parameter_names)
        held = lm._held()
        self.free = np.array(
            [k for k, name in enumerate(names) if name not in held], dtype=int
        )
        self.base = np.asarray(lm._eval_params(), dtype=float)
        self.phi = self.base[self.free]
        self.n_cov = lm._n_covariates()
        self.Z_fit = self.rows(model.Z)
        self.c = np.asarray(model.c)

    def rows(self, Z: Any) -> npt.NDArray:
        """Covariates as the fitter takes them: prepared, as rows, and
        relative to the model's centre."""
        Zp = covariate_rows(self.lm._prepare_Z(Z), self.n_cov)
        return np.asarray(self.lm._centred(Zp), dtype=float)

    def full(self, p: npt.NDArray) -> npt.NDArray:
        out = self.base.copy()
        out[self.free] = p
        return out

    def loglik(self, p: npt.NDArray, t: npt.NDArray) -> float:
        fitter = self.lm.model
        params = self.full(p)
        tiny = 1e-300
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            f = fitter.df(t, self.Z_fit, *params)
            s = fitter.sf(t, self.Z_fit, *params)
            F = fitter.ff(t, self.Z_fit, *params)
            logf = np.log(np.clip(f, tiny, None))
            logs = np.log(np.clip(s, tiny, None))
            logF = np.log(np.clip(F, tiny, None))
        c = self.c
        contrib = np.where(c == 0, logf, np.where(c == -1, logF, logs))
        return float(np.sum(contrib))


def _regression_free_covariance(
    model: Any, life: _RegressionLife
) -> npt.NDArray:
    """The corrected covariance of the free regression parameters."""
    _check_plain_life_model(life.lm)
    Hinv = np.asarray(life.lm.covariance(), dtype=float)[
        np.ix_(life.free, life.free)
    ]
    if not np.isfinite(Hinv).all():
        raise ValueError(
            "The regression life fit has no usable covariance (its "
            "information matrix could not be inverted), so the analytic "
            "two-stage bound cannot be formed; use method='bootstrap'."
        )
    t = np.asarray(model.pseudo_failure_times, dtype=float)
    v = pseudo_time_variances(model)
    events = np.where((model.c == 0) & (v > 0))[0]
    cov = Hinv.copy()
    if events.size:
        dphi = _generated_regressor_term(
            life.loglik, life.phi, Hinv, t, events
        )
        cov = cov + (dphi * v[events]) @ dphi.T
    return cov


def regression_parameter_covariance(model: Any) -> npt.NDArray:
    """
    Two-stage covariance of an accelerated (covariate) model's regression
    life parameters, ordered as ``life_model.params``.

    Each unit's pseudo failure time carries the variance of its own path
    fit and extrapolation (:func:`pseudo_time_variances`); the regression
    fit's inverse observed information (``life_model.covariance()``) is
    corrected by the same generated-regressor term as the plain model's,
    ``H^{-1} + sum_i v_i (dphi/dt_i)(dphi/dt_i)'``, the derivative of the
    regression score taken in each unit's time at that unit's stress. A
    parameter held fixed has a zero row and column, an aliased one
    ``nan``, as in ``life_model.covariance()``.
    """
    life = _RegressionLife(model)
    cov_free = _regression_free_covariance(model, life)
    n = life.base.size
    cov = np.zeros((n, n))
    cov[np.ix_(life.free, life.free)] = cov_free
    aliased = life.lm.k_dist + np.asarray(life.lm.aliased, dtype=int)
    if aliased.size:
        cov[aliased, :] = np.nan
        cov[:, aliased] = np.nan
    return cov


def _regression_curve(
    model: Any, x: npt.NDArray, Z: Any
) -> tuple[npt.NDArray, npt.NDArray, str]:
    """``(sf, se, link)`` of an accelerated (covariate) model at ``x``
    under the stress ``Z``."""
    life = _RegressionLife(model)
    cov = _regression_free_covariance(model, life)
    rows = life.rows(Z)
    lower = life.lm.distribution.support[0]
    below = x < lower
    # below the support nothing has happened: survival 1, no spread
    x_eval = np.where(below, lower if np.isfinite(lower) else 0.0, x)

    def sf_of(p: npt.NDArray) -> npt.NDArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = life.lm.model.sf(x_eval, rows, *life.full(p))
        return np.broadcast_to(np.asarray(out, dtype=float), x.shape)

    sf_hat = np.asarray(model.sf(x, Z), dtype=float).reshape(x.shape)
    se = _delta_se(sf_of, life.phi, cov, scale_free=True)
    return sf_hat, np.where(below, 0.0, se), cb_link(life.lm.distribution)


# -- step-stress (acceleration="clock") fits ------------------------------


def _clock_unit_design(
    model: Any, idx: int
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Unit ``idx``'s time-ordered reference-stress times ``tau``, their
    derivative in the stress coefficients ``dtau/dgamma`` (``(n, q)``),
    the path's Jacobian in its parameters and its derivative in ``gamma``
    through the clock, ``g'(tau) dtau/dgamma``, at the fitted values."""
    gamma = np.asarray(model.gamma, dtype=float)
    z_ref = np.asarray(model.stress_ref, dtype=float)
    mask = np.flatnonzero(model.i == model.units[idx])
    order = mask[np.argsort(model.x[mask], kind="stable")]
    x_unit = np.asarray(model.x[order], dtype=float)
    s = np.asarray(model.Z[order], dtype=float) - z_ref
    exposure = np.diff(np.concatenate([[0.0], x_unit])) * np.exp(s @ gamma)
    tau = np.cumsum(exposure)
    dtau = np.cumsum(exposure[:, None] * s, axis=0)
    theta = np.asarray(model.path_params[idx], dtype=float)
    J = np.atleast_2d(model.path_model.jacobian(tau, *theta))
    C = path_time_derivative(model.path_model, tau, theta)[:, None] * dtau
    return tau, dtau, J, C


def clock_first_stage(
    model: Any,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    First-stage covariance of a step-stress model's pseudo failure times
    and of its clock: ``(Cov(t), Cov(t, gamma), Cov(gamma))``.

    With ``population_method="moments"`` the clock coefficients and every
    unit's path parameters are, jointly, the least-squares fit of the
    paths on the clock ``tau_i(t; gamma)``, so their covariance is the
    Gauss-Newton ``sigma^2 (J'J)^{-1}`` of that joint fit, ``J`` holding
    each unit's path Jacobian ``J_i`` and its clock column
    ``C_i = g'(tau_i) dtau_i/dgamma``. It is formed unit by unit through
    the Schur complement of the clock block,
    ``M = sum_i C_i'C_i - C_i'J_i B_i`` with ``B_i = (J_i'J_i)^{-1} J_i'C_i``.

    A unit whose path crosses the threshold has the pseudo failure time
    ``inv_path(threshold; theta_i)`` on the reference clock, with gradient
    ``u_i`` in ``theta_i``; a censored unit is censored at its first or
    last reference-stress time, which moves with ``gamma`` alone (gradient
    ``w_i``). With ``e_i = B_i'u_i - w_i``,

    .. code-block:: text

        Cov(t_i, t_j)   = sigma^2 (delta_ij u_i'(J_i'J_i)^{-1}u_i
                                   + e_i' M^{-1} e_j)
        Cov(t_i, gamma) = -sigma^2 e_i' M^{-1}
        Cov(gamma)      = sigma^2 M^{-1}

    Every pseudo failure time shares the one estimated clock, so the
    times are correlated.
    """
    if model.population_method != "moments":
        raise NotImplementedError(
            "The analytic two-stage bound of a step-stress "
            "(acceleration='clock') model is derived for the clock "
            "estimated by profile least squares "
            "(population_method='moments'); this model's clock is the "
            "mixed-model (population_method='reml') estimate, whose "
            "sampling covariance is not derived here. Use "
            "method='bootstrap', which re-estimates the clock on every "
            "resample, or refit with population_method='moments'."
        )
    n = len(model.units)
    q = np.asarray(model.gamma).size
    sigma2 = float(model.measurement_var)
    if not sigma2 > 0:
        # exact path fits: no first-stage uncertainty
        return np.zeros((n, n)), np.zeros((n, q)), np.zeros((q, q))
    own = np.zeros(n)
    E = np.zeros((n, q))
    M = np.zeros((q, q))
    for idx in range(n):
        tau, dtau, J, C = _clock_unit_design(model, idx)
        P = safe_inv(J.T @ J)
        B = P @ (J.T @ C)
        M += C.T @ C - (C.T @ J) @ B
        if model.c[idx] == 0:
            theta = np.asarray(model.path_params[idx], dtype=float)
            u = _inv_path_grad(
                model.path_model, model.threshold, theta, sigma2 * P
            )
            own[idx] = float(u @ P @ u)
            E[idx] = B.T @ u
        else:
            # right censored at its last reference time, left censored at
            # its first positive one: either moves with the clock only
            at = (
                int(np.argmax(tau))
                if model.c[idx] == 1
                else int(np.flatnonzero(tau > 0)[0])
            )
            E[idx] = -dtau[at]
    Minv = safe_inv(M)
    cov_t = sigma2 * (np.diag(own) + E @ Minv @ E.T)
    cov_tg = -sigma2 * E @ Minv
    cov_g = sigma2 * Minv
    return cov_t, cov_tg, cov_g


def clock_parameter_covariance(model: Any) -> npt.NDArray:
    """
    Two-stage covariance of a step-stress model's reference-stress life
    parameters and its stress coefficients, ``life_model.params`` then
    ``gamma``.

    The life parameters' block is ``H^{-1} + A Cov(t) A'`` with
    ``A = dphi/dt`` (as for the plain model, but over every unit: a
    censored unit's time moves with the clock too) and ``Cov(t)`` from
    :func:`clock_first_stage`; the cross block is ``A Cov(t, gamma)`` and
    the clock's own block ``Cov(gamma)``.
    """
    lm = model.life_model
    _check_plain_life_model(lm)
    phi = np.asarray(lm.params, dtype=float)
    t = np.asarray(model.pseudo_failure_times, dtype=float)
    cov_t, cov_tg, cov_g = clock_first_stage(model)
    Hinv = _plain_information_inverse(model, phi)
    k, q = phi.size, cov_g.shape[0]
    cov = np.zeros((k + q, k + q))
    cov[:k, :k] = Hinv
    cov[k:, k:] = cov_g
    affected = np.flatnonzero(np.diag(cov_t) > 0)
    if affected.size:

        def loglik(p: npt.NDArray, times: npt.NDArray) -> float:
            return _life_loglik(model, p, times)

        A = _generated_regressor_term(loglik, phi, Hinv, t, affected)
        cov[:k, :k] += A @ cov_t[np.ix_(affected, affected)] @ A.T
        cross = A @ cov_tg[affected]
        cov[:k, k:] = cross
        cov[k:, :k] = cross.T
    return cov


def _clock_curve(
    model: Any, x: npt.NDArray, Z: Any
) -> tuple[npt.NDArray, npt.NDArray, str]:
    """``(sf, se, link)`` of a step-stress model at ``x`` under the stress
    ``Z`` (a row or a StepSchedule): the reference-stress survival at the
    clock time, ``S0(tau_Z(x; gamma); phi)``, a function of the life
    parameters and of the clock."""
    lm = model.life_model
    cov = clock_parameter_covariance(model)
    phi = np.asarray(lm.params, dtype=float)
    gamma = np.asarray(model.gamma, dtype=float)
    z_ref = np.asarray(model.stress_ref, dtype=float)
    k, q = phi.size, gamma.size
    offset = float(getattr(lm, "gamma", 0.0) or 0.0)
    inside = np.isfinite(x) & (x > 0)
    x_eval = np.where(inside, x, 0.0)

    def sf_of(p: npt.NDArray) -> npt.NDArray:
        g = p[k:]

        def af(z: Any) -> float:
            row = stress_row(z, q, allow_nan=True)
            return float(np.exp(g @ (row - z_ref)))

        tau = StressClock(af, q, Z).tau(x_eval)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.asarray(lm.dist.sf(tau - offset, *p[:k]), dtype=float)

    sf_hat = np.asarray(model.sf(x, Z), dtype=float).reshape(x.shape)
    se = _delta_se(sf_of, np.concatenate([phi, gamma]), cov, scale_free=True)
    return sf_hat, np.where(inside, se, 0.0), cb_link(lm.dist)


# -- public confidence-bound entry points ---------------------------------


def analytic_cb(
    model: Any,
    x: npt.ArrayLike,
    on: str,
    alpha_ci: float,
    bound: str,
    Z: Any = None,
) -> npt.NDArray:
    """
    The analytic two-stage bound on ``on`` at ``x``: the delta method
    through the corrected parameter covariance, on the life
    distribution's probability-plot scale. ``Z`` is the stress of an
    accelerated (covariate) or step-stress model, already read as an
    array (or a StepSchedule).
    """
    x = np.atleast_1d(np.asarray(x, dtype=float))
    if getattr(model, "acceleration", None) == "clock":
        sf_hat, se, link = _clock_curve(model, x, Z)
    elif model.is_accelerated:
        sf_hat, se, link = _regression_curve(model, x, Z)
    else:
        cov = life_parameter_covariance(model, method="analytic")
        lm = model.life_model
        phi = np.asarray(lm.params, dtype=float)
        offset = float(getattr(lm, "gamma", 0.0) or 0.0)

        def sf_of(p: npt.NDArray) -> npt.NDArray:
            return lm.dist.sf(x - offset, *p)

        sf_hat = np.asarray(sf_of(phi), dtype=float)
        se = _delta_se(sf_of, phi, cov, scale_free=True)
        link = cb_link(lm.dist)

    if bound == "two-sided":
        # Each side is a one-sided bound, so it takes half the total tail
        # probability: with the full ``alpha_ci`` per side the "95%" band
        # was really a 90% one.
        sf_lo = sf_link_bound(sf_hat, se, alpha_ci / 2.0, "lower", link)
        sf_hi = sf_link_bound(sf_hat, se, alpha_ci / 2.0, "upper", link)
        if on in ("sf", "R"):
            return np.stack([sf_lo, sf_hi], axis=-1)
        elif on in ("ff", "F"):
            return np.stack([1.0 - sf_hi, 1.0 - sf_lo], axis=-1)
        else:  # Hf
            # 0.0 - log: where sf is 1, -log(1) is -0.0 (#746).
            return np.stack(
                [0.0 - np.log(sf_hi), 0.0 - np.log(sf_lo)], axis=-1
            )

    # one-sided: ff and Hf decrease in sf, so flip the tail
    if on in ("sf", "R"):
        return sf_link_bound(sf_hat, se, alpha_ci, bound, link)
    flip = "upper" if bound == "lower" else "lower"
    sf_b = sf_link_bound(sf_hat, se, alpha_ci, flip, link)
    return (1.0 - sf_b) if on in ("ff", "F") else 0.0 - np.log(sf_b)


def bootstrap_cb(
    model: Any,
    x: npt.ArrayLike,
    on: str,
    alpha_ci: float,
    bound: str,
    n_boot: int,
    random_state: "int | None",
    Z: "npt.ArrayLike | None" = None,
) -> npt.NDArray:
    """
    Two-stage bootstrap confidence bounds: resample units with
    replacement and rerun the whole pipeline (per-unit path fit ->
    pseudo failure time -> life fit) on each resample, so the
    first-stage path/extrapolation uncertainty is folded in.

    For an *accelerated* (covariate) model pass the stress ``Z`` to
    evaluate at: each resampled unit carries its stress row and the
    covariate life fit is rerun per resample. For a step-stress
    (``acceleration="clock"``) model ``Z`` is a stress row or a
    ``StepSchedule``; each resampled unit carries its stress history and
    the clock is re-estimated per resample. The selected path model is
    held fixed across resamples (matching ``path="best"``'s chosen
    model), so the bound reflects life-fit and extrapolation
    variability, not path re-selection. Refits whose curve is not
    finite everywhere are dropped rather than poisoning the quantiles.
    Refits whose life fit has no finite maximum are kept at the estimate
    they reached and counted, and resamples whose life fit has no
    maximum at all are left out and counted, in one warning
    (``warn_bootstrap_refits``), as the parametric bootstrap does.
    """
    from surpyval.univariate.parametric.parametric import Parametric
    from surpyval.utils.refits import DEGRADATION_REFIT

    from ._maximum import RefitTally

    x = np.atleast_1d(np.asarray(x, dtype=float))
    rng = as_generator(random_state)
    method_name = _on_method(on)
    # A missing time or stress makes the model's own curve nan there: the
    # bound is nan at those points, and only the others must be finite
    # for a refit to count (every refit used to be dropped, and the bound
    # raised).
    own = getattr(model, method_name)
    missing = np.isnan(np.asarray(own(x) if Z is None else own(x, Z)))
    if missing.all():
        shape = missing.shape + ((2,) if bound == "two-sided" else ())
        return np.full(shape, np.nan)
    # A step-stress (clock) model refits its clock on every resample: each
    # unit carries its own stress rows, the reference stress is held so the
    # refits describe the same reference-stress life, and gamma is
    # re-estimated by the model's own population method.
    clock = getattr(model, "acceleration", None) == "clock"
    clock_kwargs: dict = {}
    if clock:
        clock_kwargs = {
            "acceleration": "clock",
            "stress_ref": model.stress_ref,
            "population_method": model.population_method,
        }
    # A resample's units are the model's own: their path fits are reused,
    # and its life fit starts from the model's (see ``utils.refits``),
    # which made the 200 refits of a 200-unit analysis take 5.4 s, from
    # 9.7 s (#522).
    life = model.life_model
    refit: dict = {
        "units": {},
        "life_init": (
            np.asarray(life.params, dtype=float)
            if isinstance(life, Parametric)
            and not (life.offset or life.lfp or life.zi)
            else None
        ),
    }
    token = DEGRADATION_REFIT.set(refit)
    tally = RefitTally()
    try:
        curves = _bootstrap_curves(
            model,
            x,
            Z,
            method_name,
            missing,
            n_boot,
            rng,
            clock_kwargs,
            tally,
        )
    finally:
        DEGRADATION_REFIT.reset(token)
    if len(curves) < 2:
        detail = (
            " (a resample may not span enough stress levels to identify "
            "the covariate fit)"
            if Z is not None
            else ""
        )
        if model._distribution is None:
            detail = (
                " (the model does not know the lifetime-distribution "
                "fitter the refits need)"
            )
        raise RuntimeError(
            "The degradation bootstrap produced too few successful refits "
            "to form a confidence bound" + detail + "."
        )
    tally.warn(
        n_boot,
        advice="treat the bounds as rough; more units, or units followed "
        "closer to the threshold, would settle them",
    )
    return percentile_bounds(curves, alpha_ci, bound)


def _bootstrap_curves(
    model: Any,
    x: npt.NDArray,
    Z: "npt.ArrayLike | None",
    method_name: str,
    missing: npt.NDArray,
    n_boot: int,
    rng: Any,
    clock_kwargs: dict,
    tally: Any,
) -> list:
    """The curves of ``bootstrap_cb``'s refits: ``n_boot`` resamples of
    ``model``'s units, each refitted and its ``method_name`` curve taken at
    ``x`` (and ``Z``), those not finite where the model's own is dropped.
    ``tally`` (a ``RefitTally``, updated in place) counts the refits
    kept, by the maximum their life fit reached, and the resamples whose
    life fit has no maximum at all (left out)."""
    from .degradation_analysis import DegradationAnalysis

    def refit(
        xs: npt.NDArray,
        ys: npt.NDArray,
        ids: npt.NDArray,
        Zs: Any,
        censored: list,
    ) -> "tuple[npt.NDArray, Any] | None":
        # The resample's curve and what its life fit reached, or None
        # where the curve is not finite where the model's own is.
        m = DegradationAnalysis.fit(
            xs,
            ys,
            ids,
            threshold=model.threshold,
            path=model.path_model,
            distribution=model._distribution,
            how=model._how,
            Z=Zs,
            running=model.running,
            censored_units=censored or None,
            **clock_kwargs,
        )
        curve_fn = getattr(m, method_name)
        curve = np.asarray(
            curve_fn(x) if Z is None else curve_fn(x, Z), dtype=float
        )
        if not np.isfinite(curve[~missing]).all():
            return None
        reached = getattr(m.life_model, "maximum", None)
        return np.where(missing, np.nan, curve), reached

    n_units = len(model.units)
    clock = bool(clock_kwargs)
    # a unit given as censored stays so in every resample it is drawn in
    marked = set(model.censored_units or ())
    curves = []
    # Each resampled fit may emit the usual small-sample path-covariance
    # warnings; silence them here so a single bootstrap call does not surface
    # hundreds of duplicates.
    with ignoring_warnings():
        for _ in range(n_boot):
            picks = rng.choice(n_units, size=n_units)
            xs, ys, ids, Zs, censored = [], [], [], [], []
            for new_id, idx in enumerate(picks):
                if model.units[idx] in marked:
                    censored.append(new_id)
                mask = model.i == model.units[idx]
                n_meas = int(mask.sum())
                xs.append(model.x[mask])
                ys.append(model.y[mask])
                ids.append(np.full(n_meas, new_id))
                if clock:
                    Zs.append(model.Z[mask])
                elif Z is not None:
                    Zs.append(np.tile(model.Z[idx], (n_meas, 1)))
            found = tally.refit(
                refit,
                np.concatenate(xs),
                np.concatenate(ys),
                np.concatenate(ids),
                None if Z is None else np.concatenate(Zs),
                censored,
            )
            if found is not None:
                curves.append(found[0])
                tally.keep(found[1])
    return curves


def _on_method(on: str) -> str:
    return {"sf": "sf", "R": "sf", "ff": "ff", "F": "ff", "Hf": "Hf"}[on]
