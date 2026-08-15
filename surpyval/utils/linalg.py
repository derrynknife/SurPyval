"""
Shared numeric helpers: guarded linear algebra, finite-difference
derivatives, and the normal-approximation confidence-bound transforms
built on them.

Every function here existed first as a module-local helper -- several of
them as verbatim copies of each other. ``numerical_hessian``,
``delta_method_se``, ``bound_signs`` and ``log_transformed_cb`` were
duplicated wholesale between ``recurrent.inference`` and
``univariate.regression._bounds`` (the drift-prone pattern that produced
#288), the ``inv``-then-``pinv`` fallback was written out at seven call
sites, and the eigenvalue-surgery family below appeared five times across
the degradation package. This module is the single copy; the call sites
import it.

The implementations are transplants, not rewrites: each body is kept
bit-identical to the copies it replaced, so consolidating changed no
fitted numbers anywhere.
"""

from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

# -- guarded linear algebra ------------------------------------------------


def safe_inv(m: npt.NDArray) -> npt.NDArray:
    """
    Inverse of ``m``, falling back to the Moore-Penrose pseudo-inverse
    when ``m`` is singular -- or when ``inv`` "succeeds" but returns
    non-finite entries, which a near-singular matrix can do without
    raising.
    """
    try:
        out = np.linalg.inv(m)
        if not np.all(np.isfinite(out)):
            raise np.linalg.LinAlgError
        return out
    except np.linalg.LinAlgError:
        return np.linalg.pinv(m)


def safe_quadform(V: npt.NDArray, u: npt.NDArray) -> float:
    """
    The quadratic form ``u' V^{-1} u`` via ``solve``, falling back to the
    pseudo-inverse when ``V`` is singular. This is the test-statistic
    shape shared by the log-rank and Gray's tests, where a degenerate
    group leaves ``V`` without full rank.
    """
    try:
        return float(u @ np.linalg.solve(V, u))
    except np.linalg.LinAlgError:
        return float(u @ np.linalg.pinv(V) @ u)


# -- finite differences ----------------------------------------------------


def numerical_hessian(
    func: Callable[[npt.NDArray], float],
    x: npt.NDArray,
    step: "npt.NDArray | None" = None,
) -> npt.NDArray:
    """
    Central finite-difference Hessian of a scalar ``func`` at ``x``. Used
    to approximate the observed Fisher information from a negative
    log-likelihood minimised with a derivative-free optimiser.

    ``step`` is the per-parameter step array; the default is the usual
    cube-root-of-machine-epsilon rule for a second-derivative central
    difference, ``eps**(1/3) * max(|x|, 1e-2)``. Callers with their own
    convention (Royston-Parmar and the frailty fitter use
    ``1e-5 * max(|x|, 1)``) pass it explicitly.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if step is None:
        step = (np.finfo(float).eps ** (1.0 / 3.0)) * np.maximum(
            np.abs(x), 1e-2
        )
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


def delta_method_se(
    func: Callable[[npt.NDArray], Any],
    mle: npt.NDArray,
    cov: npt.NDArray,
) -> npt.NDArray:
    """
    Standard errors of the (possibly vector-valued) function ``func`` of
    the parameters, evaluated at the MLE, via the delta method with a
    central-difference Jacobian: ``se_i = sqrt(J_i' cov J_i)``.
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


# -- normal-approximation confidence bounds --------------------------------


def bound_signs(alpha_ci: float, bound: str) -> tuple[float, npt.NDArray]:
    """
    The one-sided tail probability and the signs of the normal quantile
    for each requested bound: ``[-1, 1]`` (lower, upper) for two-sided
    bounds, a single sign otherwise.
    """
    if bound == "two-sided":
        return alpha_ci / 2.0, np.array([-1.0, 1.0])
    elif bound == "lower":
        return alpha_ci, np.array([-1.0])
    elif bound == "upper":
        return alpha_ci, np.array([1.0])
    raise ValueError("`bound` must be 'two-sided', 'lower' or 'upper'")


def log_transformed_cb(
    estimate: npt.ArrayLike,
    se: npt.ArrayLike,
    alpha_ci: float = 0.05,
    bound: str = "two-sided",
) -> npt.NDArray:
    """
    Log-transformed normal confidence bounds ``est * exp(+/- z * se / est)``
    for a positive curve (the same construction as the exponential
    Greenwood bounds on the nonparametric MCF). Where the estimate is zero
    (e.g. a CIF at ``x = 0``) both bounds are zero.
    """
    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    alpha, signs = bound_signs(alpha_ci, bound)
    z = norm.ppf(1.0 - alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(estimate > 0, se / estimate, 0.0)
    cb = estimate[..., None] * np.exp(signs * z * ratio[..., None])
    return cb if bound == "two-sided" else cb[..., 0]


# -- eigenvalue surgery on symmetric matrices ------------------------------
#
# Three operations of one family: symmetrise, eigendecompose, repair the
# spectrum, reconstruct. They differ only in the repair and in what is
# rebuilt (the matrix, its inverse, or its square root). The floor
# conventions are the call sites' own and are preserved exactly:
# ``psd_precision`` includes the float-tiny guard in its floor where
# ``psd_floor`` does not, because the sites they replaced did the same.


def psd_project(matrix: npt.NDArray) -> tuple[npt.NDArray, bool]:
    """
    Project a symmetric matrix onto the positive semi-definite cone
    by clipping negative eigenvalues to zero.

    Returns the projected matrix and whether any eigenvalue was
    *materially* negative (beyond floating-point noise).
    """
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    tol = 1e-10 * max(np.abs(eigvals).max(), np.finfo(float).tiny)
    clipped = bool((eigvals < -tol).any())
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T, clipped


def psd_floor(
    matrix: npt.NDArray, rel_floor: float, abs_floor: float
) -> npt.NDArray:
    """
    Symmetrise ``matrix`` and floor its eigenvalues at
    ``max(eigmax * rel_floor, abs_floor)``, so a rank-deficient moment
    estimate becomes safely positive definite (e.g. before a Cholesky
    factorisation).
    """
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    floor = max(eigvals.max() * rel_floor, abs_floor)
    eigvals = np.clip(eigvals, floor, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def psd_precision(
    matrix: npt.NDArray, rel_floor: float, abs_floor: float
) -> npt.NDArray:
    """
    Inverse of a symmetric ``matrix`` with its eigenvalues floored at
    ``max(eigmax * rel_floor, abs_floor, tiny)`` before inversion: a
    rank-deficient or tiny matrix gives a very tight (but proper)
    precision in the deficient directions rather than a singular one.
    """
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    floor = max(eigvals.max() * rel_floor, abs_floor, np.finfo(float).tiny)
    inv_eigvals = 1.0 / np.clip(eigvals, floor, None)
    return eigvecs @ np.diag(inv_eigvals) @ eigvecs.T


def psd_root(matrix: npt.NDArray) -> npt.NDArray:
    """
    A square-root factor ``R`` of a symmetric ``matrix`` with its
    eigenvalues clipped non-negative, such that ``R R' = matrix`` (after
    the clip). Robust for multivariate-normal sampling from a covariance
    that may itself have been PSD-clipped: ``mean + z @ R.T``.
    """
    matrix = (matrix + matrix.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(matrix)
    return eigvecs @ np.diag(np.sqrt(np.clip(eigvals, 0.0, None)))
