import warnings
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from ..parametric import Parametric

import numpy.typing as npt
from autograd import hessian, jacobian

from surpyval import np

from . import fallback_minimize


def mps_fun(
    params: npt.NDArray,
    dist: Any,
    x: npt.NDArray,
    inv_trans: Callable[..., Any],
    const: Callable[..., Any],
    c: npt.NDArray | None,
    n: npt.NDArray,
    tl: npt.NDArray | None,
    tr: npt.NDArray | None,
    offset: bool,
) -> Any:
    if offset:
        gamma = inv_trans(const(params))[0]
        x_new = x - gamma
        # The truncation bounds live on the same (observed) timescale as
        # x, so they must be shifted with it; leaving them unshifted
        # evaluated F(tl) on the unshifted distribution and made the
        # objective infinite at the true parameters (#268). Infinite
        # bounds are unaffected by the shift. A shifted left bound at or
        # below the base distribution's support carries F = 0 exactly --
        # evaluate it as -inf rather than feeding a negative time to the
        # base CDF (NaN for e.g. Weibull, which fails the whole fit).
        s0 = dist.support[0]
        tl = tl - gamma
        tr = tr - gamma
        if np.isfinite(tl) and tl <= s0:
            tl = -np.inf
        if np.isfinite(tr) and tr <= s0:
            # The whole window sits below the support: F(tr) = 0 makes
            # the spacings denominator collapse, which neg_mean_D
            # reports as an infinite objective.
            tr = s0
        params = inv_trans(const(params))[1:]
    else:
        params = inv_trans(const(params))
        x_new = x.copy()
    D = dist.neg_mean_D(x_new, c, n, tl, tr, *params)
    return D


def mps(model: "Parametric") -> Any:
    """
    MPS: Maximum Product Spacing

    This is the method to get the largest (geometric) average distance
    between all points. This method works really well when all points are
    unique. Some complication comes in when using repeated data. This method
    is quite good for offset distributions.
    """

    dist = model.dist
    x, c, n = model.data["x"], model.data["c"], model.data["n"]
    const = model.fitting_info["const"]
    inv_trans = model.fitting_info["inv_trans"]
    init = model.fitting_info["init"]
    offset = model.offset
    tl = model.tl
    tr = model.tr

    jac = jacobian(mps_fun)
    hess = hessian(mps_fun)

    args = (dist, x, inv_trans, const, c, n, tl, tr, offset)
    res = fallback_minimize(mps_fun, init, args, jac, hess, newton_tol=1e-15)

    if (res.success is False) or (np.isnan(res.x).any()):
        warnings.warn("MPS FAILED: Try alternate estimation method")

    results = {}
    params = inv_trans(const(res.x))
    results["res"] = res

    if offset:
        results["gamma"] = params[0]
        results["params"] = params[1::]
    else:
        results["gamma"] = 0.0
        results["params"] = params

    return results
