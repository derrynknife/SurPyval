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

import warnings
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.special import expit, log_ndtr, ndtr, ndtri, ndtri_exp

from surpyval.utils.validation import BOUNDS, option_error

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


def numerical_gradient(
    func: Callable[[npt.NDArray], float],
    x: npt.NDArray,
    step: "npt.NDArray | None" = None,
) -> npt.NDArray:
    """
    Central finite-difference gradient of a scalar ``func`` at ``x``, with
    the per-parameter ``step`` (default ``1e-6 * max(|x|, 1)``): with
    :func:`numerical_hessian`, what checks that a derivative-free search
    stopped at a maximum (``is_local_minimum``).

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.utils.linalg import numerical_gradient
    >>> f = lambda v: v[0] ** 2 + 3 * v[1]
    >>> numerical_gradient(f, np.array([1.0, 2.0])).round(6)
    array([2., 3.])
    """
    x = np.asarray(x, dtype=float)
    if step is None:
        step = 1e-6 * np.maximum(np.abs(x), 1.0)
    g = np.zeros(x.size)
    for i in range(x.size):
        e = np.zeros(x.size)
        e[i] = step[i]
        g[i] = (func(x + e) - func(x - e)) / (2.0 * step[i])
    return g


def delta_method_se(
    func: Callable[[npt.NDArray], Any],
    mle: npt.NDArray,
    cov: npt.NDArray,
) -> npt.NDArray:
    """
    Standard errors of the (possibly vector-valued) function ``func`` of
    the parameters, evaluated at the MLE, via the delta method with a
    central-difference Jacobian: ``se_i = sqrt(J_i' cov J_i)``.

    The step is ``eps**(1/3) * max(|p|, 1e-2)``. Where that step leaves
    the function's domain (a positive parameter smaller than the step:
    an accelerated life model's constant ``c`` of 6e-9 against a step of
    6e-8 gave a ``nan`` bound, #617) the difference is taken again in a
    step relative to the parameter alone.
    """
    h = np.finfo(float).eps ** (1.0 / 3.0)
    mle = np.asarray(mle, dtype=float)
    step = h * np.maximum(np.abs(mle), 1e-2)
    at = None
    cols = []
    for i in range(mle.size):

        def column(size: float) -> npt.NDArray:
            ei = np.zeros(mle.size)
            ei[i] = size
            return (
                np.asarray(func(mle + ei), dtype=float)
                - np.asarray(func(mle - ei), dtype=float)
            ) / (2.0 * size)

        col = column(step[i])
        small = h * abs(mle[i])
        if 0 < small < step[i] and not np.all(np.isfinite(col)):
            if at is None:
                at = np.asarray(func(mle), dtype=float)
            col = np.where(np.isfinite(at), column(small), col)
        cols.append(col)
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
    raise option_error("bound", bound, BOUNDS)


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
    from scipy.stats import norm

    estimate = np.asarray(estimate, dtype=float)
    se = np.asarray(se, dtype=float)
    alpha, signs = bound_signs(alpha_ci, bound)
    z = norm.ppf(1.0 - alpha)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(estimate > 0, se / estimate, 0.0)
    cb = estimate[..., None] * np.exp(signs * z * ratio[..., None])
    return cb if bound == "two-sided" else cb[..., 0]


# -- the scale of a Wald band on a survival function (#477, #504) ----------
#
# The univariate parametric, degradation and parametric regression models
# all form their Wald bands on ``sf``/``ff``/``Hf`` here, on the scale on
# which the family is a straight line in (log) time -- its
# probability-plot scale, the distribution's ``_cb_link``: ``log(-log S)``
# ("loglog") for the Weibull, Exponential, Rayleigh and Gumbel, the normal
# quantile of ``F`` ("probit") for the Normal and LogNormal, and the logit
# of ``F`` ("logit") for the Logistic, LogLogistic and every family with no
# such scale. Each scale is taken as ``u``, increasing in the cumulative
# hazard, and computed so that it stays accurate in both tails.


def percentile_bounds(
    draws: npt.ArrayLike, alpha_ci: float, bound: str = "two-sided"
) -> npt.NDArray:
    """The percentile bootstrap bound of ``draws`` (one resample per row):
    the ``alpha_ci`` quantile for ``bound="lower"``, the ``1 - alpha_ci``
    quantile for ``"upper"``, and otherwise the ``alpha_ci / 2`` and ``1 -
    alpha_ci / 2`` quantiles, stacked on a last axis of 2. Shared by the
    Kaplan-Meier, degradation, destructive-degradation and Buckley-James
    bootstraps (#351)."""
    draws = np.asarray(draws)
    if bound == "lower":
        return np.quantile(draws, alpha_ci, axis=0)
    if bound == "upper":
        return np.quantile(draws, 1.0 - alpha_ci, axis=0)
    lo = np.quantile(draws, alpha_ci / 2.0, axis=0)
    hi = np.quantile(draws, 1.0 - alpha_ci / 2.0, axis=0)
    return np.stack([lo, hi], axis=-1)


def cb_link(dist: Any) -> str:
    """The scale of a family's Wald band on ``sf``/``ff``/``Hf``: the
    distribution's ``_cb_link``, or ``"logit"`` for a family without one."""
    return getattr(dist, "_cb_link", "logit")


def sf_link_from_H(H: npt.ArrayLike, link: str) -> npt.NDArray:
    """The band scale ``u`` of the survival ``exp(-H)``, from the cumulative
    hazard: ``log H``, ``Phi^-1(F)`` or ``logit F``. It stays finite where
    ``exp(-H)`` underflows, so a bound formed on it has no ceiling on
    ``Hf`` (#418); ``H = 0`` and ``H = inf`` give ``-inf`` and ``inf``."""
    H = np.asarray(H, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if link == "loglog":
            return np.log(H)
        F = -np.expm1(-H)
        if link == "probit":
            # Phi^-1(F) = -Phi^-1(S), with S = exp(-H) taken in log space
            # in the right half.
            return np.where(F < 0.5, ndtri(F), -ndtri_exp(-H))
        # logit F = log(F / S) = log F + H
        return np.log(F) + H


def sf_link_from_sf(
    sf: npt.ArrayLike, ff: npt.ArrayLike, link: str
) -> npt.NDArray:
    """The band scale ``u`` (as :func:`sf_link_from_H`) from the survival
    and the failure probability, each used where it is the smaller, so
    accurate where it is small (``1 - sf`` is not)."""
    sf = np.asarray(sf, dtype=float)
    ff = np.asarray(ff, dtype=float)
    left = ff < 0.5
    with np.errstate(divide="ignore", invalid="ignore"):
        if link == "loglog":
            return np.log(np.where(left, -np.log1p(-ff), -np.log(sf)))
        if link == "probit":
            return np.where(left, ndtri(ff), -ndtri(sf))
        return np.log(ff) - np.log(sf)


def sf_from_link(u: npt.ArrayLike, link: str, on: str) -> npt.NDArray:
    """``sf``, ``ff`` or ``Hf`` (``on``) at band scale ``u``, each to full
    precision in its own small tail."""
    u = np.asarray(u, dtype=float)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if link == "loglog":
            H = np.exp(u)
            if on == "sf":
                return np.exp(-H)
            return -np.expm1(-H) if on == "ff" else H
        if link == "probit":
            if on == "sf":
                return ndtr(-u)
            return ndtr(u) if on == "ff" else -log_ndtr(-u)
        if on == "sf":
            return expit(-u)
        return expit(u) if on == "ff" else np.logaddexp(0.0, u)


def link_band(
    u_hat: npt.ArrayLike,
    se_u: npt.ArrayLike,
    alpha_ci: float,
    bound: str,
    link: str,
    on: str = "sf",
) -> npt.NDArray:
    """
    A Wald band on ``on`` (``"sf"``, ``"ff"`` or ``"Hf"``): ``u_hat`` +/-
    ``z se_u`` on the band scale ``link``, mapped to ``on``. Two-sided
    bounds put ``[lower, upper]`` on the last axis. Where ``u_hat`` is
    infinite (``sf`` exactly 1 or 0) the bounds are the estimate.

    ``u_hat`` and ``se_u`` come from :func:`sf_link_from_H` (or
    :func:`sf_link_from_sf`) and the delta method on it; on these scales
    the band is the envelope of the straight lines of the Wald ellipsoid,
    so it rises with time whenever the shape's own Wald interval excludes
    0, where a band on the logit of ``sf`` could turn back on small
    samples (#477).
    """
    from scipy.stats import norm

    u_hat = np.asarray(u_hat, dtype=float)
    se_u = np.asarray(se_u, dtype=float)
    alpha, signs = bound_signs(alpha_ci, bound)
    z = norm.ppf(1.0 - alpha)
    # sf falls as u rises; ff and Hf rise with it.
    direction = -1.0 if on == "sf" else 1.0
    with np.errstate(invalid="ignore"):
        u = u_hat[..., None] + direction * signs * z * se_u[..., None]
    u = np.where(np.isinf(u_hat)[..., None], u_hat[..., None], u)
    cb = sf_from_link(u, link, on)
    return cb if bound == "two-sided" else cb[..., 0]


def sf_link_bound(
    sf_hat: npt.ArrayLike,
    se: npt.ArrayLike,
    alpha_ci: float,
    bound: str,
    link: str = "logit",
    ff_hat: "npt.ArrayLike | None" = None,
    on: str = "sf",
) -> npt.NDArray:
    """
    The Wald band on ``on`` (``"sf"``, ``"ff"`` or ``"Hf"``) from the
    survival ``sf_hat`` and its delta-method standard error ``se``, on the
    band scale ``link`` (see :func:`link_band`). ``ff_hat``, when given,
    is the failure probability to full precision where it is small
    (otherwise ``1 - sf_hat``). Where ``sf`` or ``ff`` is below the normal
    range the bounds are the edge they are at: the transform degenerates
    to 0/0 there, and the variance is noise (#256).
    """
    from scipy.stats import norm

    sf_hat = np.asarray(sf_hat, dtype=float)
    ff_hat = 1.0 - sf_hat if ff_hat is None else np.asarray(ff_hat, float)
    u_hat = sf_link_from_sf(sf_hat, ff_hat, link)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        # |du / d sf|: 1 / (S H), 1 / phi(u) and 1 / (S F)
        if link == "loglog":
            slope = 1.0 / (sf_hat * np.exp(u_hat))
        elif link == "probit":
            slope = 1.0 / norm.pdf(u_hat)
        else:
            slope = 1.0 / (sf_hat * ff_hat)
        se_u = np.asarray(se, dtype=float) * slope
    tiny = np.finfo(float).tiny
    u_hat = np.where(ff_hat < tiny, -np.inf, u_hat)
    u_hat = np.where(sf_hat < tiny, np.inf, u_hat)
    return link_band(u_hat, se_u, alpha_ci, bound, link, on)


def wald_undefined(
    p_hat: float,
    var: float,
    lower: "float | None" = None,
    upper: "float | None" = None,
) -> "str | None":
    """
    Why a Wald bound on a parameter does not exist, or ``None`` when it
    does: the variance from the inverse observed information is negative
    or not finite (the information is not positive definite, typically
    because the estimate is at or near a boundary of the parameter
    space), or the estimate lies on the edge of the support that the
    bound's transformed scale (log or logit) needs it inside of (#411).
    A zero variance (a parameter fixed at fit time) is not undefined: its
    interval is the degenerate one at the estimate.
    """
    if not np.isfinite(var) or var < 0:
        return (
            "its variance from the inverse observed information is "
            f"{var:.3g}, so the information matrix is not positive "
            "definite (the estimate is at or near a boundary of the "
            "parameter space, or the likelihood is not regular there)"
        )
    if (lower is not None and p_hat <= lower) or (
        upper is not None and p_hat >= upper
    ):
        return (
            f"the estimate {p_hat:.6g} is on the edge of its support "
            f"({lower}, {upper}), where the likelihood is not regular"
        )
    return None


def warn_wald_undefined(what: str, reason: str, stacklevel: int = 3) -> None:
    """The one warning a Wald bound that does not exist gives (#411):
    the bound on ``what`` is undefined because of ``reason``."""
    warnings.warn(
        f"The Wald confidence bound on {what} is undefined: {reason}. "
        "nan is returned; a profile-likelihood or bootstrap interval, "
        "where the model has one, does not need the variance.",
        RuntimeWarning,
        stacklevel=stacklevel + 1,
    )


def param_name(name: "str | None") -> str:
    """How :func:`warn_wald_undefined` names a parameter."""
    return "the parameter" if name is None else f"the parameter {name!r}"


def wald_bound_on_support(
    p_hat: float,
    var: float,
    lower: "float | None",
    upper: "float | None",
    alpha_ci: float = 0.05,
    bound: str = "two-sided",
    name: "str | None" = None,
) -> npt.NDArray:
    """
    Wald confidence bound(s) on a single fitted parameter, computed on a
    transformed scale chosen from its support so the result respects it:
    a generalised logit for an interval-bounded parameter (e.g. a repair
    efficiency in ``(0, 1)``), log distance from the bound for a
    one-sided-bounded parameter (e.g. a positive rate), and the natural
    scale for an unbounded one.

    This is the core both ``param_cb`` implementations -- the
    recurrent-event inference mixin's and the parametric regression
    model's -- shared verbatim; each supplies ``p_hat``/``var`` and the
    parameter's ``(lower, upper)`` from its own bookkeeping.

    Where the bound does not exist (see :func:`wald_undefined`) it is
    ``nan``, with a warning naming the parameter ``name`` and why; it
    used to be ``nan`` with only numpy's raw "invalid value encountered
    in sqrt", or a ``ZeroDivisionError`` for an estimate on the edge of
    an interval support (#411).
    """
    from scipy.stats import norm

    alpha, signs = bound_signs(alpha_ci, bound)
    reason = wald_undefined(p_hat, var, lower, upper)
    if reason is not None:
        # wald_bound_on_support -> param_cb -> the caller
        warn_wald_undefined(param_name(name), reason, stacklevel=3)
        return np.full(signs.shape, np.nan)
    offsets = signs * norm.ppf(1.0 - alpha) * np.sqrt(var)

    if lower is not None and upper is not None:
        # Bounds on the generalised logit keep the result in (lower,
        # upper).
        width = upper - lower
        frac = (p_hat - lower) / width
        u_hat = np.log(frac / (1.0 - frac))
        du = offsets / (width * frac * (1.0 - frac))
        return lower + width / (1.0 + np.exp(-(u_hat + du)))
    elif lower is not None:
        # Bounds on log(p - lower) keep the result above ``lower``.
        return lower + (p_hat - lower) * np.exp(offsets / (p_hat - lower))
    elif upper is not None:
        # Bounds on log(upper - p) keep the result below ``upper``.
        return upper - (upper - p_hat) * np.exp(-offsets / (upper - p_hat))
    return p_hat + offsets


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
