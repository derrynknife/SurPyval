"""
Delta-method confidence bounds for the parametric regression models.

Every parametric regression fitter (AFT, parametric PH, PO, additive hazards,
accelerated life) produces a :class:`ParametricRegressionModel` fitted by
maximum likelihood, so its parameter uncertainty is the inverse of the
observed information (the Hessian of the negative log-likelihood at the MLE).
These helpers turn that covariance into

* Wald bounds on individual parameters (on a support-respecting scale), and
* delta-method bounds on the predicted ``sf``/``ff``/``Hf``/``hf``/``df`` at a
  covariate vector ``Z``.

The functions here are deliberately self-contained (numpy + scipy only) so the
regression package does not depend on any other fitted-model machinery.
"""

import numpy as np
from scipy.stats import norm


def numerical_hessian(func, x):
    """
    Central finite-difference Hessian of a scalar ``func`` at ``x``. Used to
    approximate the observed Fisher information from the negative
    log-likelihood, which the regression fitters minimise with a derivative
    -free optimiser.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    step = (np.finfo(float).eps ** (1.0 / 3.0)) * np.maximum(np.abs(x), 1e-2)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ei[i] = step[i]
            ej = np.zeros(n)
            ej[j] = step[j]
            H[i, j] = H[j, i] = (
                func(x + ei + ej)
                - func(x + ei - ej)
                - func(x - ei + ej)
                + func(x - ei - ej)
            ) / (4.0 * step[i] * step[j])
    return H


def delta_method_se(func, mle, cov):
    """
    Standard errors of the (vector-valued) ``func`` of the parameters at the
    MLE via the delta method with a central-difference Jacobian:
    ``se_i = sqrt(J_i' cov J_i)``.
    """
    mle = np.asarray(mle, dtype=float)
    step = (np.finfo(float).eps ** (1.0 / 3.0)) * np.maximum(np.abs(mle), 1e-2)
    cols = []
    for i in range(mle.size):
        ei = np.zeros(mle.size)
        ei[i] = step[i]
        cols.append(
            (
                np.asarray(func(mle + ei), dtype=float)
                - np.asarray(func(mle - ei), dtype=float)
            )
            / (2.0 * step[i])
        )
    J = np.stack(cols, axis=-1)
    var = np.einsum("...i,ij,...j->...", J, cov, J)
    with np.errstate(invalid="ignore"):
        return np.sqrt(var)


def bound_signs(alpha_ci, bound):
    """
    The per-bound one-sided tail probability and normal-quantile signs:
    ``[-1, 1]`` (lower, upper) for two-sided bounds, a single sign otherwise.
    """
    if bound == "two-sided":
        return alpha_ci / 2.0, np.array([-1.0, 1.0])
    elif bound == "lower":
        return alpha_ci, np.array([-1.0])
    elif bound == "upper":
        return alpha_ci, np.array([1.0])
    raise ValueError("`bound` must be 'two-sided', 'lower' or 'upper'")


def log_transformed_cb(estimate, se, alpha_ci=0.05, bound="two-sided"):
    """
    Log-transformed normal bounds ``est * exp(+/- z se / est)`` for a positive
    quantity (used for ``hf`` and ``df``). Two-sided bounds put ``[lower,
    upper]`` on the last axis; one-sided bounds keep the shape of ``estimate``.
    """
    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    alpha, signs = bound_signs(alpha_ci, bound)
    z = norm.ppf(1.0 - alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(estimate > 0, se / estimate, 0.0)
    cb = estimate[..., None] * np.exp(signs * z * ratio[..., None])
    if bound == "two-sided":
        return cb
    return cb[..., 0]


def logit_sf_bound(sf_hat, se, sign, alpha_tail):
    """
    A single survival-function confidence bound on the logit scale, which keeps
    it inside ``(0, 1)``: ``expit(logit(sf) + sign z se/(sf(1-sf)))``. ``sign``
    is ``-1`` for the lower bound and ``+1`` for the upper.
    """
    z = norm.ppf(1.0 - alpha_tail)
    est = np.clip(np.asarray(sf_hat, dtype=float), 1e-15, 1.0 - 1e-15)
    logit = np.log(est / (1.0 - est))
    with np.errstate(divide="ignore", invalid="ignore"):
        se_logit = np.asarray(se, dtype=float) / (est * (1.0 - est))
    return 1.0 / (1.0 + np.exp(-(logit + sign * z * se_logit)))
