"""REML estimation of the population path-parameter distribution.

The random-effects degradation model for path models that are linear in
their parameters is a linear mixed model: unit ``i``'s measurements are

    y_i = X_i theta_i + eps_i,   theta_i ~ MVN(mu, Sigma),
                                 eps_i ~ N(0, sigma^2 I)

so, with the random effects integrated out,

    y_i ~ N(X_i mu, V_i),   V_i = X_i Sigma X_i' + sigma^2 I.

This module maximises the REML log-likelihood of that marginal model:
the fixed effect ``mu`` is profiled out with generalised least squares
and the REML adjustment term ``logdet(sum_i X_i' V_i^-1 X_i)`` removes
the small-sample downward bias that plain ML variance components have
from estimating ``mu``. ``Sigma`` is parameterised by its Cholesky
factor (log-diagonal) so it stays positive definite by construction.

The two-stage moments estimate (computed in
``DegradationAnalysis.fit``) is used as the starting point.

Nonlinear path models
---------------------
For a path model that is *nonlinear* in its parameters -- exponential,
power, Gompertz, ... -- the measurement mean ``f(x_i, theta_i)`` is no
longer ``X_i theta_i`` and the marginal likelihood is intractable.
``reml_estimate_nonlinear`` handles this with the Lindstrom-Bates (1990)
alternating algorithm (the FOCE linearisation used by ``nlme``):

1. hold ``(mu, Sigma, sigma^2)`` fixed and find each unit's conditional
   mode ``theta_hat_i`` (its penalised-least-squares / MAP estimate);
2. linearise the path about that mode,
   ``f(x_i, theta) ~ f(x_i, theta_hat_i) + J_i (theta - theta_hat_i)``
   with ``J_i`` the path Jacobian at ``theta_hat_i``, forming the
   pseudo-response ``w_i = y_i - f(x_i, theta_hat_i) + J_i theta_hat_i``;
3. run the *linear* REML step above on ``(w_i, J_i)`` to update
   ``(mu, Sigma, sigma^2)``,

iterating 1-3 to convergence. For a linear-in-parameters path this
reduces exactly to the linear REML in one pass (``w_i = y_i`` and the
modes drop out), so the two routines agree.
"""

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

_LARGE = 1e30


def _n_z(p: int) -> int:
    """Length of the variance-parameter vector for ``p`` path params."""
    return p + p * (p - 1) // 2 + 1


def _chol_from_z(z: npt.NDArray, p: int) -> npt.NDArray:
    """Lower-triangular Cholesky factor of Sigma from the z vector."""
    chol = np.zeros((p, p))
    chol[np.diag_indices(p)] = np.exp(z[:p])
    if p > 1:
        chol[np.tril_indices(p, -1)] = z[p : p + p * (p - 1) // 2]
    return chol


def _z_from_init(
    cov_init: npt.NDArray, sigma2_init: float, p: int
) -> npt.NDArray:
    """Starting z vector from (possibly rank-deficient) moment estimates."""
    cov_init = (cov_init + cov_init.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov_init)
    floor = max(eigvals.max() * 1e-4, sigma2_init * 1e-6, 1e-12)
    eigvals = np.clip(eigvals, floor, None)
    cov_init = eigvecs @ np.diag(eigvals) @ eigvecs.T
    chol = np.linalg.cholesky(cov_init)
    z = np.empty(_n_z(p))
    z[:p] = np.log(np.diag(chol))
    if p > 1:
        z[p : p + p * (p - 1) // 2] = chol[np.tril_indices(p, -1)]
    z[-1] = 0.5 * np.log(sigma2_init)
    return z


def _reml_pieces(z, y_list, x_mat_list, p):
    """
    Evaluate the model at ``z``.

    Returns ``(neg_reml, mu, Sigma, sigma2)`` where ``neg_reml`` is the
    negative REML log-likelihood (up to a constant) with the fixed
    effect ``mu`` profiled out by GLS.
    """
    chol = _chol_from_z(z, p)
    covariance = chol @ chol.T
    sigma2 = np.exp(2.0 * z[-1])

    gls_information = np.zeros((p, p))  # sum X' V^-1 X
    gls_rhs = np.zeros(p)  # sum X' V^-1 y
    y_v_y = 0.0  # sum y' V^-1 y
    logdet_v = 0.0

    for y_i, x_mat in zip(y_list, x_mat_list):
        v_i = x_mat @ covariance @ x_mat.T + sigma2 * np.eye(len(y_i))
        cho = cho_factor(v_i, lower=True)
        logdet_v += 2.0 * np.log(np.diag(cho[0])).sum()
        v_inv_y = cho_solve(cho, y_i)
        v_inv_x = cho_solve(cho, x_mat)
        gls_information += x_mat.T @ v_inv_x
        gls_rhs += x_mat.T @ v_inv_y
        y_v_y += y_i @ v_inv_y

    mu = np.linalg.solve(gls_information, gls_rhs)
    quad = y_v_y - 2.0 * mu @ gls_rhs + mu @ gls_information @ mu
    sign, logdet_info = np.linalg.slogdet(gls_information)
    if sign <= 0:
        raise np.linalg.LinAlgError("GLS information not positive definite")
    neg_reml = 0.5 * (logdet_v + quad + logdet_info)
    return neg_reml, mu, covariance, sigma2


def reml_estimate(
    y_list: "list[npt.NDArray]",
    x_mat_list: "list[npt.NDArray]",
    cov_init: npt.NDArray,
    sigma2_init: float,
) -> tuple[npt.NDArray, npt.NDArray, float, bool]:
    """
    REML fit of ``y_i ~ N(X_i mu, X_i Sigma X_i' + sigma^2 I)``.

    Parameters
    ----------
    y_list : list of ndarray
        Each unit's measurement vector.
    x_mat_list : list of ndarray
        Each unit's design matrix (the path Jacobian, constant in the
        parameters for linear-in-parameter path models).
    cov_init, sigma2_init : ndarray, float
        Starting values for ``Sigma`` and ``sigma^2`` (typically the
        two-stage moment estimates); ``cov_init`` may be
        rank-deficient, its eigenvalues are floored.

    Returns
    -------
    (mu, Sigma, sigma2, converged)
    """
    p = x_mat_list[0].shape[1]

    def objective(z):
        try:
            return _reml_pieces(z, y_list, x_mat_list, p)[0]
        except np.linalg.LinAlgError:
            return _LARGE

    z0 = _z_from_init(cov_init, sigma2_init, p)
    result = minimize(
        objective,
        z0,
        method="Nelder-Mead",
        options={
            "maxiter": 20_000,
            "maxfev": 20_000,
            "xatol": 1e-10,
            "fatol": 1e-10,
        },
    )
    _, mu, covariance, sigma2 = _reml_pieces(result.x, y_list, x_mat_list, p)
    return mu, covariance, sigma2, bool(result.success)


def _prior_precision(cov: npt.NDArray, sigma2: float) -> npt.NDArray:
    """Inverse of ``Sigma`` with its eigenvalues floored positive.

    A rank-deficient or tiny ``Sigma`` gives a very tight (but proper)
    prior in the deficient directions rather than a singular precision.
    """
    cov = (cov + cov.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(cov)
    floor = max(eigvals.max() * 1e-8, sigma2 * 1e-12, np.finfo(float).tiny)
    inv_eigvals = 1.0 / np.clip(eigvals, floor, None)
    return eigvecs @ np.diag(inv_eigvals) @ eigvecs.T


def _conditional_mode(
    path_model,
    x: npt.NDArray,
    y: npt.NDArray,
    mu: npt.NDArray,
    prior_precision: npt.NDArray,
    sigma2: float,
    theta0: npt.NDArray,
    max_iter: int = 50,
) -> npt.NDArray:
    """
    Penalised-least-squares (MAP) mode of one unit's path parameters.

    Minimises ``||y - f(x, theta)||^2 / sigma^2 +
    (theta - mu)' Sigma^-1 (theta - mu)`` by damped Gauss-Newton with a
    backtracking line search, started from the unit's unpenalised
    least-squares fit ``theta0``.
    """
    theta = np.array(theta0, dtype=float)

    def penalised(t):
        resid = y - path_model.path(x, *t)
        delta = t - mu
        return (resid @ resid) / sigma2 + delta @ prior_precision @ delta

    obj = penalised(theta)
    if not np.isfinite(obj):
        return theta
    for _ in range(max_iter):
        jac = np.asarray(path_model.jacobian(x, *theta), dtype=float)
        resid = y - path_model.path(x, *theta)
        # gradient and Gauss-Newton Hessian of the penalised objective
        # (the common factor of 2 cancels in the Newton step)
        grad = -(jac.T @ resid) / sigma2 + prior_precision @ (theta - mu)
        hess = (jac.T @ jac) / sigma2 + prior_precision
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        alpha, improved = 1.0, False
        for _ in range(40):
            candidate = theta - alpha * step
            if np.isfinite(candidate).all():
                new_obj = penalised(candidate)
                if np.isfinite(new_obj) and new_obj < obj - 1e-14 * abs(obj):
                    theta, obj, improved = candidate, new_obj, True
                    break
            alpha *= 0.5
        if not improved:
            break
        scale = 1.0 + np.max(np.abs(theta))
        if np.max(np.abs(alpha * step)) <= 1e-10 * scale:
            break
    return theta


def reml_estimate_nonlinear(
    y_list: "list[npt.NDArray]",
    x_list: "list[npt.NDArray]",
    path_model,
    mean_init: npt.NDArray,
    cov_init: npt.NDArray,
    sigma2_init: float,
    theta_init: npt.NDArray,
    max_outer: int = 50,
    tol: float = 1e-5,
) -> tuple[npt.NDArray, npt.NDArray, float, bool]:
    """
    REML fit of a nonlinear random-effects degradation path by the
    Lindstrom-Bates (1990) FOCE linearisation.

    Parameters
    ----------
    y_list : list of ndarray
        Each unit's measurement vector.
    x_list : list of ndarray
        Each unit's measurement times (used to re-evaluate the path and
        its Jacobian at the conditional modes).
    path_model : PathModel
        The (nonlinear) degradation path model.
    mean_init, cov_init, sigma2_init : ndarray, ndarray, float
        Starting values for ``mu``, ``Sigma`` and ``sigma^2`` (typically
        the two-stage moment estimates).
    theta_init : ndarray
        Per-unit unpenalised least-squares path fits, one row per unit;
        the starting points for the conditional-mode search.
    max_outer : int, optional
        Maximum outer (linearise / LME) iterations. Default 50.
    tol : float, optional
        Relative convergence tolerance on ``(mu, Sigma, sigma^2)``.

    Returns
    -------
    (mu, Sigma, sigma2, converged)
    """
    mu = np.array(mean_init, dtype=float)
    covariance = np.array(cov_init, dtype=float)
    sigma2 = float(sigma2_init)
    theta_hat = np.array(theta_init, dtype=float)

    converged = False
    for _ in range(max_outer):
        prior_precision = _prior_precision(covariance, sigma2)
        # Step 1: conditional modes given the current population.
        w_list, jac_list = [], []
        for k, (y_i, x_i) in enumerate(zip(y_list, x_list)):
            theta_i = _conditional_mode(
                path_model,
                x_i,
                y_i,
                mu,
                prior_precision,
                sigma2,
                theta_hat[k],
            )
            theta_hat[k] = theta_i
            # Step 2: linearise the path about the mode.
            jac = np.asarray(path_model.jacobian(x_i, *theta_i), dtype=float)
            fitted = np.asarray(path_model.path(x_i, *theta_i), dtype=float)
            w_list.append(y_i - fitted + jac @ theta_i)
            jac_list.append(jac)

        # Step 3: linear REML step on the pseudo-data, warm-started from
        # the current variance components.
        mu_new, cov_new, sigma2_new, inner_ok = reml_estimate(
            w_list, jac_list, covariance, sigma2
        )

        prev = np.concatenate([mu, covariance.ravel(), [sigma2]])
        curr = np.concatenate([mu_new, cov_new.ravel(), [sigma2_new]])
        scale = np.maximum(np.abs(prev), np.abs(curr)) + 1e-12
        rel_change = float(np.max(np.abs(curr - prev) / scale))

        mu, covariance, sigma2 = mu_new, cov_new, sigma2_new
        if rel_change < tol:
            converged = bool(inner_ok)
            break

    return mu, covariance, sigma2, converged
