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
