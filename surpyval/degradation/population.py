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

Stress-dependent path parameters
--------------------------------
For an accelerated degradation test whose path parameters depend on
the unit's stress (``links`` in ``DegradationAnalysis.fit``) the fixed
effect is no longer a single mean but ``D_i gamma``, a per-unit
fixed-effects design times a coefficient vector, and the marginal model
is

    y_i ~ N(A_i gamma, V_i),  A_i = X_i D_i,  V_i = X_i Sigma X_i' + sigma^2 I.

Both routines take that fixed-effects design optionally
(``a_mat_list`` / ``d_mat_list``); without it ``D_i = I`` and ``gamma``
is the population mean ``mu`` as above. The random-effects design
``X_i`` is unchanged, so ``Sigma`` keeps its meaning as the
between-unit covariance of the path parameters *given* the stress.
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from surpyval.utils.linalg import psd_floor, psd_precision

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
    cov_init = psd_floor(cov_init, 1e-4, max(sigma2_init * 1e-6, 1e-12))
    chol = np.linalg.cholesky(cov_init)
    z = np.empty(_n_z(p))
    z[:p] = np.log(np.diag(chol))
    if p > 1:
        z[p : p + p * (p - 1) // 2] = chol[np.tril_indices(p, -1)]
    z[-1] = 0.5 * np.log(sigma2_init)
    return z


def _reml_pieces(
    z: npt.NDArray,
    y_list: list,
    x_mat_list: list,
    p: int,
    a_mat_list: list,
) -> tuple:
    """
    Evaluate the model at ``z``.

    Returns ``(neg_reml, gamma, Sigma, sigma2)`` where ``neg_reml`` is
    the negative REML log-likelihood (up to a constant) with the fixed
    effects ``gamma`` profiled out by GLS. ``x_mat_list`` is the
    random-effects design (it forms ``V_i``) and ``a_mat_list`` the
    fixed-effects design; the two are the same list for the plain
    population model.
    """
    chol = _chol_from_z(z, p)
    covariance = chol @ chol.T
    sigma2 = np.exp(2.0 * z[-1])

    m = a_mat_list[0].shape[1]
    gls_information = np.zeros((m, m))  # sum A' V^-1 A
    gls_rhs = np.zeros(m)  # sum A' V^-1 y
    y_v_y = 0.0  # sum y' V^-1 y
    logdet_v = 0.0

    for y_i, x_mat, a_mat in zip(y_list, x_mat_list, a_mat_list):
        v_i = x_mat @ covariance @ x_mat.T + sigma2 * np.eye(len(y_i))
        cho = cho_factor(v_i, lower=True)
        logdet_v += 2.0 * np.log(np.diag(cho[0])).sum()
        v_inv_y = cho_solve(cho, y_i)
        v_inv_a = cho_solve(cho, a_mat)
        gls_information += a_mat.T @ v_inv_a
        gls_rhs += a_mat.T @ v_inv_y
        y_v_y += y_i @ v_inv_y

    gamma = np.linalg.solve(gls_information, gls_rhs)
    quad = y_v_y - 2.0 * gamma @ gls_rhs + gamma @ gls_information @ gamma
    sign, logdet_info = np.linalg.slogdet(gls_information)
    if sign <= 0:
        raise np.linalg.LinAlgError("GLS information not positive definite")
    neg_reml = 0.5 * (logdet_v + quad + logdet_info)
    return neg_reml, gamma, covariance, sigma2


def reml_estimate(
    y_list: "list[npt.NDArray]",
    x_mat_list: "list[npt.NDArray]",
    cov_init: npt.NDArray,
    sigma2_init: float,
    a_mat_list: "list[npt.NDArray] | None" = None,
    diagnostics: "dict | None" = None,
) -> tuple[npt.NDArray, npt.NDArray, float, bool]:
    """
    REML fit of ``y_i ~ N(A_i gamma, X_i Sigma X_i' + sigma^2 I)``.

    Parameters
    ----------
    y_list : list of ndarray
        Each unit's measurement vector.
    x_mat_list : list of ndarray
        Each unit's random-effects design matrix (the path Jacobian,
        constant in the parameters for linear-in-parameter path
        models).
    cov_init, sigma2_init : ndarray, float
        Starting values for ``Sigma`` and ``sigma^2`` (typically the
        two-stage moment estimates); ``cov_init`` may be
        rank-deficient, its eigenvalues are floored.
    a_mat_list : list of ndarray, optional
        Each unit's fixed-effects design ``A_i = X_i D_i`` when the
        path parameters depend on the unit's stress. Default ``None``:
        ``A_i = X_i``, so the fixed effect is the population mean.
    diagnostics : dict, optional
        If given, ``diagnostics["on_boundary"]`` is set to whether the
        estimate of ``Sigma`` is on the boundary of the positive
        semi-definite cone (singular; see :func:`_on_boundary`).

    Returns
    -------
    (gamma, Sigma, sigma2, converged)
        ``gamma`` is the population mean ``mu`` when ``a_mat_list`` is
        not given.

    Notes
    -----
    The objective is :func:`_reml_pieces`, evaluated through the Woodbury
    identity by :func:`reml_estimate_woodbury` (a ``p x p`` computation per
    unit rather than an ``n_i x n_i`` factorisation) and searched by BFGS.
    """
    if a_mat_list is None:
        a_mat_list = x_mat_list
    gamma, covariance, sigma2, converged, _ = reml_estimate_woodbury(
        y_list,
        x_mat_list,
        cov_init,
        sigma2_init,
        a_mat_list,
        diagnostics=diagnostics,
    )
    return gamma, covariance, sigma2, converged


def _unit_summaries(y_list: list, x_mat_list: list, a_mat_list: list) -> dict:
    """The cross-products the Woodbury REML evaluation needs, stacked over
    units (``xtx`` is ``(units, p, p)``, ``xta`` is ``(units, p, m)``, ...)
    and summed where only the total is used."""
    return {
        "n": sum(len(y) for y in y_list),
        "xtx": np.stack([x.T @ x for x in x_mat_list]),
        "xty": np.stack([x.T @ y for x, y in zip(x_mat_list, y_list)]),
        "xta": np.stack([x.T @ a for x, a in zip(x_mat_list, a_mat_list)]),
        "yty": float(sum(y @ y for y in y_list)),
        "ata": sum(a.T @ a for a in a_mat_list),
        "aty": sum(a.T @ y for a, y in zip(a_mat_list, y_list)),
    }


def _reml_pieces_woodbury(
    z: npt.NDArray, summary: dict, p: int, reml: bool = True
) -> tuple:
    """
    :func:`_reml_pieces` through the Woodbury identity (with ``reml=False``
    the plain ML objective: no ``logdet`` adjustment for the fixed effects).

    With ``Sigma = L L'`` and ``M_i = I + L' X_i' X_i L / sigma^2``,
    ``V_i^-1 = (I - X_i L M_i^-1 L' X_i' / sigma^2) / sigma^2`` and
    ``logdet V_i = n_i log sigma^2 + logdet M_i``, so every quantity is a
    ``p x p`` computation on the unit's cross-products (batched over units)
    rather than an ``n_i x n_i`` factorisation.
    """
    return _reml_pieces_from_root(
        _chol_from_z(z, p), np.exp(2.0 * z[-1]), summary, reml
    )


def _reml_pieces_from_root(
    chol: npt.NDArray, sigma2: float, summary: dict, reml: bool = True
) -> tuple:
    """:func:`_reml_pieces_woodbury` at ``Sigma = chol chol'`` for any
    ``(p, r)`` root ``chol``, not only the Cholesky factor: every term
    depends on ``chol`` only through ``chol chol'``, so a root with
    ``r < p`` columns evaluates the objective at a singular ``Sigma``
    (on the boundary of the positive semi-definite cone)."""
    r = chol.shape[1]
    m_mat = (
        np.eye(r)
        + np.einsum("ji,ujk,kl->uil", chol, summary["xtx"], chol) / sigma2
    )
    m_chol = np.linalg.cholesky(m_mat)
    logdet_v = (
        summary["n"] * np.log(sigma2)
        + 2.0 * np.log(np.diagonal(m_chol, axis1=1, axis2=2)).sum()
    )
    lxy = np.einsum("ji,uj->ui", chol, summary["xty"])
    lxa = np.einsum("ji,ujk->uik", chol, summary["xta"])
    m_lxy = np.linalg.solve(m_mat, lxy[..., None])[..., 0]
    m_lxa = np.linalg.solve(m_mat, lxa)
    y_v_y = (summary["yty"] - np.sum(lxy * m_lxy) / sigma2) / sigma2
    gls_rhs = (
        summary["aty"] - np.einsum("uik,ui->k", lxa, m_lxy) / sigma2
    ) / sigma2
    gls_information = (
        summary["ata"] - np.einsum("uik,uil->kl", lxa, m_lxa) / sigma2
    ) / sigma2

    gamma = np.linalg.solve(gls_information, gls_rhs)
    quad = y_v_y - 2.0 * gamma @ gls_rhs + gamma @ gls_information @ gamma
    sign, logdet_info = np.linalg.slogdet(gls_information)
    if sign <= 0:
        raise np.linalg.LinAlgError("GLS information not positive definite")
    neg_reml = 0.5 * (logdet_v + quad + (logdet_info if reml else 0.0))
    return neg_reml, gamma, chol @ chol.T, sigma2


def reml_estimate_woodbury(
    y_list: "list[npt.NDArray]",
    x_mat_list: "list[npt.NDArray]",
    cov_init: npt.NDArray,
    sigma2_init: float,
    a_mat_list: "list[npt.NDArray]",
    reml: bool = True,
    diagnostics: "dict | None" = None,
) -> tuple[npt.NDArray, npt.NDArray, float, bool, float]:
    """
    :func:`reml_estimate` evaluated through the Woodbury identity: the same
    objective, at a cost independent of the number of measurements per
    unit, searched by BFGS (with a Nelder-Mead fallback). With
    ``reml=False`` it maximises the plain (ML) likelihood instead.
    Used inside the step-stress clock iteration, where the step is repeated
    many times.

    Returns ``(gamma, Sigma, sigma2, converged, objective)`` with
    ``objective`` the minimised negative (RE)ML log-likelihood on the
    original scale of the data. With a ``diagnostics`` dict,
    ``diagnostics["on_boundary"]`` is set to whether the estimate of
    ``Sigma`` is singular (see :func:`_on_boundary`).
    """
    p = x_mat_list[0].shape[1]
    # Column scaling: the REML fit is equivariant to rescaling the design
    # columns (Sigma becomes D Sigma D, the fixed effects divide by their
    # scale, the objective shifts by a constant), and with path parameters
    # of very different sizes it keeps the optimisation well conditioned.
    x_scale = _column_scale(x_mat_list)
    a_scale = _column_scale(a_mat_list)
    x_scaled = [x / x_scale for x in x_mat_list]
    a_scaled = [a / a_scale for a in a_mat_list]
    # Centring: GLS is equivariant to shifting y by A beta0, so take out an
    # OLS fit of the fixed effects first. Otherwise y' V^-1 y and the GLS
    # terms are huge and nearly cancel, and the round-off swamps the
    # finite-difference gradient.
    beta0, *_ = np.linalg.lstsq(
        np.vstack(a_scaled), np.concatenate(y_list), rcond=None
    )
    summary = _unit_summaries(
        [y - a @ beta0 for y, a in zip(y_list, a_scaled)],
        x_scaled,
        a_scaled,
    )
    cov_init = np.asarray(cov_init, dtype=float) * np.outer(x_scale, x_scale)

    def objective(z: npt.NDArray) -> float:
        # A trial step far from the optimum can overflow ``exp(z)`` or the
        # Woodbury products; such a point is simply rejected (``_LARGE``),
        # so the floating-point warnings it raises on the way are noise.
        try:
            with np.errstate(all="ignore"):
                value = _reml_pieces_woodbury(z, summary, p, reml)[0]
        except np.linalg.LinAlgError:
            return _LARGE
        return value if np.isfinite(value) else _LARGE

    z0 = _z_from_init(cov_init, sigma2_init, p)
    # the objective is smooth in z, so a quasi-Newton search converges in a
    # few hundred evaluations; Nelder-Mead polishes when it does not
    # the log-likelihood (and its gradient) grows with the number of
    # measurements, so the gradient tolerance does too
    result = minimize(
        objective,
        z0,
        method="BFGS",
        jac="3-point",
        options={"gtol": 1e-6 * summary["n"]},
    )
    # status 2 is BFGS's "precision loss": the finite-difference gradient
    # is at its noise floor, i.e. the optimum to numerical accuracy
    converged = bool(result.success or result.status == 2)
    if not converged:
        polish = minimize(
            objective,
            result.x if result.fun < _LARGE else z0,
            method="Nelder-Mead",
            options={
                "maxiter": 20_000,
                "maxfev": 20_000,
                "xatol": 1e-9,
                "fatol": 1e-12 * max(abs(float(result.fun)), 1.0),
            },
        )
        if polish.fun <= result.fun:
            result = polish
            converged = bool(polish.success)
    objective_value, gamma, covariance, sigma2 = _reml_pieces_woodbury(
        result.x, summary, p, reml
    )
    if diagnostics is not None:
        diagnostics["on_boundary"] = _on_boundary(result.x, summary, p, reml)
    covariance = covariance / np.outer(x_scale, x_scale)
    # undo the column scaling's constant shift of the objective: the
    # random-effects scaling cancels in V, the fixed-effects one enters only
    # the REML logdet term
    if reml:
        objective_value += float(np.sum(np.log(a_scale)))
    gamma = (gamma + beta0) / a_scale
    return gamma, covariance, sigma2, converged, objective_value


def _on_boundary(z: npt.NDArray, summary: dict, p: int, reml: bool) -> bool:
    """
    Whether the (RE)ML estimate at ``z`` is on the boundary of the positive
    semi-definite cone: ``Sigma`` with its smallest eigenvalue set to zero
    fits at least as well.

    ``Sigma`` is searched through its log-Cholesky factor, so it is
    positive definite at every trial point and a maximum on the boundary
    (a between-unit variance of zero, or a correlation of +-1) is only
    approached: the search stops where the shrinking gradient meets its
    tolerance, which leaves the smallest eigenvalue anywhere from 1e-10 to
    1e-4 of the largest (and unscaled, the ratio also depends on the units
    of time). No threshold on the eigenvalues separates that from a small
    genuine eigenvalue, so the test is on the objective instead. At an
    interior maximum the gradient is zero and removing the smallest
    eigenvalue ``lambda`` loses about ``f'' lambda^2 / 2`` of
    log-likelihood; at a maximum on the boundary the objective still
    improves towards it, so the singular ``Sigma`` is no worse. "No
    worse" is up to ``sqrt(eps)`` relative, the package's tolerance for
    two values being the same to numerical accuracy (as in ``Beta4`` and
    the destructive-degradation
    root search).

    The eigenvalues are those of the column-scaled ``Sigma`` the search
    works on, so the test does not depend on the units of the path
    parameters. ``sigma^2`` is held at its estimate; refitting it on the
    boundary could only lower the objective there further.
    """
    chol = _chol_from_z(z, p)
    sigma2 = float(np.exp(2.0 * z[-1]))
    try:
        with np.errstate(all="ignore"):
            estimate = _reml_pieces_from_root(chol, sigma2, summary, reml)[0]
            eigvals, eigvecs = np.linalg.eigh(chol @ chol.T)
            root = eigvecs[:, 1:] * np.sqrt(np.clip(eigvals[1:], 0.0, None))
            edge = _reml_pieces_from_root(root, sigma2, summary, reml)[0]
    except np.linalg.LinAlgError:
        return False
    if not (np.isfinite(estimate) and np.isfinite(edge)):
        return False
    roundoff = np.sqrt(np.finfo(float).eps) * max(abs(estimate), 1.0)
    return bool(edge <= estimate + roundoff)


def _column_scale(mats: list) -> npt.NDArray:
    """Root-mean-square of each design column over all units (1 where a
    column is identically zero)."""
    stacked = np.vstack(mats)
    rms = np.sqrt(np.mean(stacked**2, axis=0))
    return np.where(rms > 0, rms, 1.0)


def _prior_precision(cov: npt.NDArray, sigma2: float) -> npt.NDArray:
    """Inverse of ``Sigma`` with its eigenvalues floored positive.

    A rank-deficient or tiny ``Sigma`` gives a very tight (but proper)
    prior in the deficient directions rather than a singular precision.
    """
    return psd_precision(cov, 1e-8, sigma2 * 1e-12)


def _conditional_mode(
    path_model: Any,
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
    least-squares fit ``theta0``. It stops once a step would move no
    parameter by more than ``1e-10`` of the parameters' size, before or
    after taking it (the line search could not resolve a smaller one, and
    used to spend 40 halvings finding that out on every converged unit,
    #588), or when no step along the Gauss-Newton direction improves the
    objective. :func:`_conditional_modes` is the same search for many
    units at once.
    """
    theta = np.array(theta0, dtype=float)

    def penalised(t: npt.NDArray) -> float:
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
        if _negligible(step, theta):
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
        if _negligible(alpha * step, theta):
            break
    return theta


def _negligible(step: npt.NDArray, theta: npt.NDArray) -> "bool | Any":
    """Whether ``step`` (one per row, for many units) moves no parameter by
    more than ``1e-10`` of the parameters' size, the conditional-mode
    search's tolerance."""
    scale = 1.0 + np.max(np.abs(theta), axis=-1)
    return np.max(np.abs(step), axis=-1) <= 1e-10 * scale


def _elementwise(path_model: Any) -> bool:
    """Whether ``path_model`` is one of the built-in path models, whose
    ``path`` and ``jacobian`` act element by element on the times and the
    parameters, so that one call evaluates many units' paths (each row its
    own parameters)."""
    from .path_models import PATH_MODELS

    return any(type(path_model) is type(m) for m in PATH_MODELS.values())


class _PaddedUnits:
    """Many units' measurements padded to one ``(units, n)`` grid, for
    evaluating their paths in one call of an element-wise path model.

    Each unit's rows beyond its own measurements repeat its last time (so
    the path is finite there wherever it is at that time) and are masked
    out of every sum.
    """

    def __init__(self, path_model: Any, xs: list, ys: list) -> None:
        self.path_model = path_model
        lengths = np.array([len(x) for x in xs])
        n = int(lengths.max())
        self.mask = np.arange(n)[None, :] < lengths[:, None]
        last = np.array([x[-1] for x in xs], dtype=float)
        self.x = np.repeat(last[:, None], n, axis=1)
        self.y = np.zeros((len(xs), n))
        for k, (x, y) in enumerate(zip(xs, ys)):
            self.x[k, : len(x)] = x
            self.y[k, : len(y)] = y
        self.shape = self.x.shape

    def _flat(self, theta: npt.NDArray, units: npt.NDArray) -> tuple:
        n = self.shape[1]
        x = self.x[units].ravel()
        params = [np.repeat(theta[:, j], n) for j in range(theta.shape[1])]
        return x, params

    def resid(self, theta: npt.NDArray, units: npt.NDArray) -> npt.NDArray:
        """The residuals ``y - f(x, theta)`` of ``units`` (``theta`` one
        row per unit), 0 in the padding."""
        x, params = self._flat(theta, units)
        with np.errstate(all="ignore"):
            fitted = np.asarray(self.path_model.path(x, *params), dtype=float)
        fitted = fitted.reshape(len(units), -1)
        return np.where(self.mask[units], self.y[units] - fitted, 0.0)

    def jacobian(self, theta: npt.NDArray, units: npt.NDArray) -> npt.NDArray:
        """The path Jacobians of ``units``, ``(units, n, p)``, 0 in the
        padding."""
        x, params = self._flat(theta, units)
        jac = np.asarray(self.path_model.jacobian(x, *params), dtype=float)
        jac = jac.reshape(len(units), self.shape[1], -1)
        return np.where(self.mask[units][..., None], jac, 0.0)


def _conditional_modes(
    path_model: Any,
    xs: list,
    ys: list,
    means: npt.NDArray,
    prior_precision: npt.NDArray,
    sigma2: float,
    theta0: npt.NDArray,
    max_iter: int = 50,
) -> npt.NDArray:
    """
    :func:`_conditional_mode` of every unit (its times ``xs[k]``,
    measurements ``ys[k]``, prior mean ``means[k]`` and start
    ``theta0[k]``), one row per unit.

    The units are independent given the population, so for a built-in
    (element-wise) path model the damped Gauss-Newton search runs for all
    of them at once, each unit with its own step length and stopping as it
    converges: the same search, with the sums taken in a different order
    (#588). A custom path model is searched unit by unit.
    """
    theta = np.array(theta0, dtype=float)
    means = np.broadcast_to(np.asarray(means, dtype=float), theta.shape)
    if not _elementwise(path_model):
        for k, (x, y) in enumerate(zip(xs, ys)):
            theta[k] = _conditional_mode(
                path_model,
                x,
                y,
                means[k],
                prior_precision,
                sigma2,
                theta[k],
                max_iter,
            )
        return theta
    units = _PaddedUnits(path_model, xs, ys)

    def penalised(t: npt.NDArray, idx: npt.NDArray) -> npt.NDArray:
        resid = units.resid(t, idx)
        delta = t - means[idx]
        return np.einsum("un,un->u", resid, resid) / sigma2 + np.einsum(
            "ui,ij,uj->u", delta, prior_precision, delta
        )

    everyone = np.arange(len(theta))
    with np.errstate(all="ignore"):
        obj = penalised(theta, everyone)
    active = np.isfinite(obj)
    for _ in range(max_iter):
        idx = np.flatnonzero(active)
        if not idx.size:
            break
        jac = units.jacobian(theta[idx], idx)
        resid = units.resid(theta[idx], idx)
        grad = (
            -np.einsum("unp,un->up", jac, resid) / sigma2
            + (theta[idx] - means[idx]) @ prior_precision.T
        )
        hess = np.einsum("unp,unq->upq", jac, jac) / sigma2 + prior_precision
        step, solved = _batched_solve(hess, grad)
        stop = ~solved | _negligible(step, theta[idx])
        active[idx[stop]] = False
        idx, step = idx[~stop], step[~stop]
        alpha = _line_search(penalised, theta, obj, idx, step)
        # a unit whose search found no improvement, or whose accepted step
        # was negligible, has converged
        moved = alpha > 0
        active[idx[~moved]] = False
        done = _negligible(alpha[moved, None] * step[moved], theta[idx[moved]])
        active[idx[moved][done]] = False
    return theta


def _batched_solve(
    hess: npt.NDArray, grad: npt.NDArray
) -> tuple[npt.NDArray, npt.NDArray]:
    """The Gauss-Newton steps ``hess[k]^-1 grad[k]``, and which could be
    solved (a singular ``hess[k]`` stops that unit's search, as
    ``np.linalg.solve`` raising does in :func:`_conditional_mode`)."""
    try:
        return np.linalg.solve(hess, grad[..., None])[..., 0], np.ones(
            len(grad), dtype=bool
        )
    except np.linalg.LinAlgError:
        pass
    step = np.zeros_like(grad)
    solved = np.ones(len(grad), dtype=bool)
    for k in range(len(grad)):
        try:
            step[k] = np.linalg.solve(hess[k], grad[k])
        except np.linalg.LinAlgError:
            solved[k] = False
    return step, solved


def _line_search(
    penalised: Any,
    theta: npt.NDArray,
    obj: npt.NDArray,
    idx: npt.NDArray,
    step: npt.NDArray,
) -> npt.NDArray:
    """Backtrack each unit ``idx[k]`` along ``-step[k]``, halving from a
    full step up to 40 times until the objective falls, as
    :func:`_conditional_mode` does; ``theta`` and ``obj`` are updated in
    place where it does. Returns the step lengths taken (0 where none
    improved)."""
    alpha = np.ones(len(idx))
    taken = np.zeros(len(idx))
    searching = np.arange(len(idx))
    for _ in range(40):
        if not searching.size:
            break
        units = idx[searching]
        candidate = theta[units] - alpha[searching, None] * step[searching]
        finite = np.isfinite(candidate).all(axis=1)
        new_obj = np.full(searching.size, np.nan)
        if finite.any():
            with np.errstate(all="ignore"):
                new_obj[finite] = penalised(candidate[finite], units[finite])
        with np.errstate(invalid="ignore"):
            better = np.isfinite(new_obj) & (
                new_obj < obj[units] - 1e-14 * np.abs(obj[units])
            )
        theta[units[better]] = candidate[better]
        obj[units[better]] = new_obj[better]
        taken[searching[better]] = alpha[searching[better]]
        searching = searching[~better]
        alpha[searching] *= 0.5
    return taken


def reml_estimate_nonlinear(
    y_list: "list[npt.NDArray]",
    x_list: "list[npt.NDArray]",
    path_model: Any,
    mean_init: npt.NDArray,
    cov_init: npt.NDArray,
    sigma2_init: float,
    theta_init: npt.NDArray,
    max_outer: int = 50,
    tol: float = 1e-5,
    d_mat_list: "list[npt.NDArray] | None" = None,
    diagnostics: "dict | None" = None,
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
        Starting values for ``mu`` (or ``gamma``, see ``d_mat_list``),
        ``Sigma`` and ``sigma^2`` (typically the two-stage moment
        estimates).
    theta_init : ndarray
        Per-unit unpenalised least-squares path fits, one row per unit;
        the starting points for the conditional-mode search.
    max_outer : int, optional
        Maximum outer (linearise / LME) iterations. Default 50.
    tol : float, optional
        Relative convergence tolerance on ``(mu, Sigma, sigma^2)``.
    d_mat_list : list of ndarray, optional
        Each unit's ``(p, m)`` fixed-effects design ``D_i`` when the
        path parameters depend on the unit's stress: the unit's prior
        mean is ``D_i gamma`` and the linearised fixed-effects design is
        ``A_i = J_i D_i``. Default ``None``: ``D_i = I``.
    diagnostics : dict, optional
        As for :func:`reml_estimate`, for the last linearisation.

    Returns
    -------
    (gamma, Sigma, sigma2, converged)
        ``gamma`` is the population mean ``mu`` when ``d_mat_list`` is
        not given.
    """
    gamma = np.array(mean_init, dtype=float)
    covariance = np.array(cov_init, dtype=float)
    sigma2 = float(sigma2_init)
    theta_hat = np.array(theta_init, dtype=float)

    converged = False
    for _ in range(max_outer):
        prior_precision = _prior_precision(covariance, sigma2)
        # Step 1: conditional modes given the current population.
        prior_means = (
            gamma
            if d_mat_list is None
            else np.array([d @ gamma for d in d_mat_list])
        )
        theta_hat = _conditional_modes(
            path_model,
            x_list,
            y_list,
            prior_means,
            prior_precision,
            sigma2,
            theta_hat,
        )
        w_list, jac_list, a_list = [], [], []
        for k, (y_i, x_i) in enumerate(zip(y_list, x_list)):
            theta_i = theta_hat[k]
            # Step 2: linearise the path about the mode.
            jac = np.asarray(path_model.jacobian(x_i, *theta_i), dtype=float)
            fitted = np.asarray(path_model.path(x_i, *theta_i), dtype=float)
            w_list.append(y_i - fitted + jac @ theta_i)
            jac_list.append(jac)
            if d_mat_list is not None:
                a_list.append(jac @ d_mat_list[k])

        # Step 3: linear REML step on the pseudo-data, warm-started from
        # the current variance components.
        gamma_new, cov_new, sigma2_new, inner_ok = reml_estimate(
            w_list,
            jac_list,
            covariance,
            sigma2,
            a_mat_list=None if d_mat_list is None else a_list,
            diagnostics=diagnostics,
        )

        prev = np.concatenate([gamma, covariance.ravel(), [sigma2]])
        curr = np.concatenate([gamma_new, cov_new.ravel(), [sigma2_new]])
        scale = np.maximum(np.abs(prev), np.abs(curr)) + 1e-12
        rel_change = float(np.max(np.abs(curr - prev) / scale))

        gamma, covariance, sigma2 = gamma_new, cov_new, sigma2_new
        if rel_change < tol:
            converged = bool(inner_ok)
            break

    return gamma, covariance, sigma2, converged
