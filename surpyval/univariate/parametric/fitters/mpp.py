from copy import copy
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..parametric import Parametric

import numpy.typing as npt
from scipy.optimize import minimize
from scipy.stats import pearsonr

from surpyval import np
from surpyval.univariate.nonparametric import plotting_positions


def _rr_fit(a: npt.NDArray, b: npt.NDArray) -> Any:
    """
    Least-squares line of ``b`` on ``a``, guarding the degenerate case.

    A probability plot needs at least two distinct abscissae to have a
    slope. With fewer -- a single observation, or a sample whose values
    are all tied -- ``polyfit`` is rank deficient and returns a nan
    slope. That nan is not caught anywhere: it becomes the seed for the
    maximum likelihood fit, which then starts at nan, produces a nan
    hessian, and finally dies inside numdifftools with an ``IndexError``
    from an empty list of finite difference steps. The cause is four
    steps removed from the symptom.

    The fallback is a unit slope through the centroid. A *zero* slope
    would be the more literal reading of "no information", but every
    ``unpack_rr`` divides by the slope to recover a scale -- Weibull
    takes ``alpha = exp(-intercept / slope)`` -- so zero merely moves the
    nan one step later. Unit slope keeps each distribution's own
    ``unpack_rr`` in charge of turning the line into correctly typed
    parameters, and lands on a usable seed: three tied observations at 10
    give ``alpha = 11.2, beta = 1`` rather than ``nan``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() >= 2 and np.unique(a[finite]).size >= 2:
        params = np.polyfit(a[finite], b[finite], 1)
        if np.isfinite(params).all():
            return params
    if finite.any():
        intercept = float(np.mean(b[finite]) - np.mean(a[finite]))
    else:
        intercept = 0.0
    return np.array([1.0, intercept])


def mpp_from_ecfd(
    dist: Any, x: npt.ArrayLike, F: npt.ArrayLike
) -> dict[str, Any]:
    x_pp = np.asarray(copy(x))
    y_pp = np.asarray(copy(F))

    mask = (y_pp != 0) & (y_pp != 1)
    y_pp = y_pp[mask]
    x_pp = x_pp[mask]

    with np.errstate(all="ignore"):
        y_pp = dist.mpp_y_transform(y_pp)
        x_pp = dist.mpp_x_transform(x_pp)
        params = _rr_fit(x_pp, y_pp)

    params = np.array(dist.unpack_rr(params, "y"))

    results = {}
    results["params"] = params
    return results


def mpp(model: "Parametric") -> dict[str, Any]:
    """
    MPP: Method of Probability Plotting

    This is the classic probability plotting paper method. This method
    creates the plotting points, transforms it to Weibull scale and then fits
    the line of best fit.
    """
    dist = model.dist
    x, c, n, t = (
        model.data["x"],
        model.data["c"],
        model.data["n"],
        model.data["t"],
    )

    heuristic = model.fitting_info["heuristic"]
    on_d_is_0 = model.fitting_info["on_d_is_0"]
    offset = model.offset
    rr = model.fitting_info["rr"]
    turnbull_estimator = model.fitting_info["turnbull_estimator"]

    if rr not in ["x", "y"]:
        raise ValueError("rr must be either 'x' or 'y'")

    if hasattr(dist, "mpp"):
        results = dist.mpp(
            x,
            c,
            n,
            t=t,
            heuristic=heuristic,
            rr=rr,
            on_d_is_0=on_d_is_0,
            offset=offset,
        )
        results["params"] = np.atleast_1d(results["params"])
        results.setdefault("gamma", 0.0)
        return results

    x_, r, d, F = plotting_positions(
        x=x,
        c=c,
        n=n,
        t=t,
        heuristic=heuristic,
        turnbull_estimator=turnbull_estimator,
    )

    x_mask = np.isfinite(x_)
    x_ = x_[x_mask]
    F = F[x_mask]
    d = d[x_mask]
    r = r[x_mask]

    if not on_d_is_0:
        x_ = x_[d > 0]
        y_ = F[d > 0]
    else:
        y_ = F

    mask = (y_ != 0) & (y_ != 1)
    y_pp = y_[mask]
    x_pp = x_[mask]
    y_pp = dist.mpp_y_transform(y_pp)

    if offset:
        x_min = np.min(x_pp)

        def fun(gamma: float) -> Any:
            g = x_min - np.exp(-gamma)
            out = -pearsonr(dist.mpp_x_transform(x_pp - g), y_pp)[0]
            return out

        res = minimize(fun, 0.0)
        gamma = x_min - np.exp(-res.x[0])
        x_pp = x_pp - gamma

    x_pp = dist.mpp_x_transform(x_pp)

    if rr == "y":
        params = _rr_fit(x_pp, y_pp)
    elif rr == "x":
        params = _rr_fit(y_pp, x_pp)

    params = np.array(dist.unpack_rr(params, rr))

    results = {}

    if offset:
        results["gamma"] = gamma
    else:
        results["gamma"] = 0.0

    results["params"] = params

    return results
