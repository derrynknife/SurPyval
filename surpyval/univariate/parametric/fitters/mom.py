import warnings
from math import comb
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..parametric import Parametric

import numpy.typing as npt
from scipy.optimize import minimize

from surpyval import np
from surpyval.univariate.parametric.fitters import (
    preconditioned_bfgs,
    search_floor,
)


def raw_to_central(moments: npt.NDArray) -> npt.NDArray:
    """``(mean, var, mu3, ...)`` from raw moments ``E[X], E[X^2], ...``.

    ``mu_k = sum_j C(k, j) (-1)^(k-j) E[X^j] mean^(k-j)``, with the mean
    kept in the leading slot because the first central moment is zero by
    construction and carries no information.

    The transform is exact and bijective, so matching the first ``k``
    central moments is the same estimator as matching the first ``k``
    raw ones -- it is only better conditioned. See ``mom_fun``.
    """
    mean = moments[0]
    central = [mean]
    for k in range(2, len(moments) + 1):
        acc = (-1) ** k * mean**k  # the j = 0 term, where E[X^0] = 1
        for j in range(1, k + 1):
            acc = acc + comb(k, j) * (-1) ** (k - j) * moments[j - 1] * (
                mean ** (k - j)
            )
        central.append(acc)
    return np.array(central)


def mom_fun(
    params: npt.NDArray,
    dist: Any,
    inv_trans: Callable[..., Any],
    const: Callable[..., Any],
    offset: bool,
    moments: npt.NDArray,
    sd: float | None = None,
) -> Any:
    """Squared mismatch between the sample and model moments.

    Compared as *central* moments scaled by the sample's own standard
    deviation, so every term is dimensionless and of comparable size:
    the mean in units of sigma, then the relative variance error, then
    the skewness difference, and so on.

    Raw moments describe the same estimator but hide it. Once a
    distribution is offset, ``E[X^k]`` is dominated by ``gamma^k`` and
    the shape contributes only a fractional correction -- 0.5% of
    ``E[X^3]`` for a Gamma(3, 4) shifted by 10. The optimiser is then
    reading three parameters off the third decimal place of a large
    number, and offset fits converged to parameter sets that matched the
    sample moments *better than the true parameters did* while being
    nowhere near them: a shape of 17.7 against a true 3.

    Central moments remove the offset by construction, so the shape
    information is the whole of ``mu_3`` rather than a rounding error in
    it. For unshifted fits the two agree to several decimal places,
    because there the conditioning was never the problem.

    ``sd`` is the sample's standard deviation. It used to be read off
    the sample's second central moment, which is only there when two or
    more moments are matched; a one-parameter fit (a Rayleigh, or a
    Weibull with one parameter fixed) fell back to ``sigma = 1`` and its
    mismatch was in the data's squared units. That made the objective,
    and with it the optimiser's stopping test, scale dependent: a
    Rayleigh fitted to data in thousandths started with a mismatch of
    2e-9 already, stopped where it began and came back 1.2% off. Without
    ``sd`` the old behaviour is kept.
    """
    dist_moments = dist.mom_moment_gen(
        *inv_trans(const(params)), offset=offset, k=len(moments)
    )
    sample = raw_to_central(moments)
    model = raw_to_central(dist_moments)

    # sigma^k puts every term on a common footing. Taken from the sample
    # alone so the scale is a constant of the problem, not something the
    # optimiser can shrink to flatter itself.
    if sd is not None:
        sigma = sd
    else:
        sigma = np.sqrt(np.abs(sample[1])) if len(sample) > 1 else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    scale = np.array([sigma ** (k + 1) for k in range(len(sample))])

    value = (((sample - model) / scale) ** 2).sum()
    # Where the model's moments do not exist (a LogLogistic with shape at
    # or below the moment's order, say) the mismatch is nan. nan compares
    # false against everything, so an optimiser that stepped there lost
    # track of the best point and the search ended *on* the nan. A huge
    # finite penalty instead reads as "worse than anything", and the line
    # searches and Nelder-Mead back away from it. (Not inf: the gradient
    # search's finite differences would then take inf - inf.)
    return value if np.isfinite(value) else _NO_MOMENTS


# The objective's value where the model's moments do not exist.
_NO_MOMENTS = 1e300


def mom(model: "Parametric") -> Any:
    """
    MOM: Method of Moments.

    This is one of the simplest ways to calculate the parameters of a
    distribution. This method is quick but only works with uncensored data.
    """
    dist = model.dist
    x, n = model.data["x"], model.data["n"]

    const = model.fitting_info["const"]
    inv_trans = model.fitting_info["inv_trans"]
    init = model.fitting_info["init"]
    offset = model.offset

    x_ = np.repeat(x, n)

    # The closed-form moment estimate cannot honour fixed parameters or
    # an offset, so only use it for a plain fit
    if (
        hasattr(dist, "_mom")
        and not offset
        and not model.fitting_info["fixed_idx"]
    ):
        return {"params": np.atleast_1d(dist._mom(x_)), "gamma": 0.0}
    # Likewise an exact offset solution, where the distribution has one
    # and the sample admits it (see ``LogNormal._mom_offset``)
    if (
        hasattr(dist, "_mom_offset")
        and offset
        and not model.fitting_info["fixed_idx"]
    ):
        closed = dist._mom_offset(x_)
        if closed is not None:
            return {"params": np.atleast_1d(closed[1:]), "gamma": closed[0]}

    # One equation per *free* parameter. A fixed parameter is known, so
    # matching a moment for it too over-determined the system: the fit
    # could only match every moment if the fixed value happened to agree
    # with the data, and otherwise warned of a failed match -- routinely,
    # e.g. for a Weibull with its shape fixed.
    n_free = model.k - len(model.fitting_info["fixed_idx"])
    if n_free == 0:
        # Every parameter is fixed; there is nothing to match.
        params = inv_trans(const(np.array(init)))
        return _mom_results(params, offset, None)

    moments = np.zeros(n_free)

    for i in range(0, n_free):
        moments[i] = (x_ ** (i + 1)).mean()
    # The spread ``mom_fun`` would take from the second central moment,
    # but available however few moments are matched
    sd = float(np.std(x_))
    args = (dist, inv_trans, const, offset, moments, sd)

    # A start at which the model's moments do not exist (a heavy tail, as
    # for the Beta-Geometric at a <= 2) makes the objective nan, and every
    # optimiser then stops where it began and reports it -- the starting
    # point came back as the fit, and since ``nan > 1e-2`` is False not
    # even the mismatch warning below fired.
    with np.errstate(all="ignore"):
        start_value = mom_fun(np.array(init), *args)
    if not start_value < _NO_MOMENTS:
        raise ValueError(
            f"Method of moments cannot start: the {dist.name} moments are "
            "not finite at the initial guess "
            f"{np.asarray(inv_trans(const(np.array(init))))} (they may not "
            "exist there). Pass `init` with parameters at which the first "
            f"{n_free} moment(s) exist, or use how='MLE'."
        )

    # A loose tolerance here silently returned parameters far from the
    # moment-matching solution for offset/fixed fits (#275): use a tight
    # tolerance, polish with Nelder-Mead if needed, and warn when the
    # relative moment mismatch remains large.
    # BFGS rescaled as for the other estimators (see
    # ``preconditioned_bfgs``); plain BFGS judged convergence by an
    # absolute gradient, in whatever units the transformed parameters
    # happened to be in. Differenced gradients, as before: the moment
    # generators are not all differentiable by autograd.
    with np.errstate(all="ignore"):
        res = preconditioned_bfgs(
            mom_fun, np.array(init), args, floor=search_floor(model)
        )
    # Reported as ``model.optimizer``
    res.optimizer = "BFGS"
    if not res.success or res.fun > 1e-8:
        res_nm = minimize(
            mom_fun,
            res.x if np.all(np.isfinite(res.x)) else np.array(init),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12},
            args=args,
        )
        if np.isfinite(res_nm.fun) and res_nm.fun < res.fun:
            res = res_nm
            res.optimizer = "Nelder-Mead"
    # The objective is a sum of squared standardised-moment differences
    # (see ``mom_fun``), so this threshold is in those units. Healthy
    # fits land in one of two places: ~1e-12 when the moment equations
    # have an exact solution, or ~1e-3 when sampling noise means no
    # parameter vector reproduces the sample moments exactly and the
    # optimiser returns the closest one -- a third central moment is
    # noisy enough at n=5000 for that to be routine. A fit that has
    # actually failed sits near 0.5. 1e-2 separates them with roughly an
    # order of magnitude of clearance on either side; the previous 1e-4
    # was calibrated against the old raw-moment objective and fires on
    # ordinary sampling noise under this one.
    if not np.isfinite(res.fun):
        # Never report a nan objective as a fit (see the start check).
        raise ValueError(
            f"Method of moments failed for {dist.name}: the model moments "
            "became non-finite during the search. Try a different `init`, "
            "or how='MLE'."
        )
    if res.fun > 1e-2:
        warnings.warn(
            "MOM optimisation did not match the sample moments (squared "
            f"standardised-moment mismatch {res.fun:.3g}); the returned "
            "parameters may be unreliable. Consider how='MLE'."
        )

    params = inv_trans(const(res.x))
    return _mom_results(params, offset, res)


def _mom_results(params: npt.NDArray, offset: bool, res: Any) -> Any:
    """Split the full parameter vector into the results dict."""
    results = {}
    if offset:
        results["gamma"] = params[0]
        results["params"] = params[1:]
    else:
        results["gamma"] = 0.0
        results["params"] = params

    results["res"] = res

    return results
