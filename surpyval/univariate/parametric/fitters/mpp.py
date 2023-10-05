from copy import copy

from scipy.optimize import minimize
from scipy.stats import pearsonr

from surpyval import np
from surpyval.univariate.nonparametric import plotting_positions


def mpp_from_ecfd(dist, x, F):
    old_err_state = np.seterr(all="ignore")

    x_pp = copy(x)
    y_pp = copy(F)

    mask = (y_pp != 0) & (y_pp != 1)
    y_pp = y_pp[mask]
    x_pp = x_pp[mask]
    y_pp = dist.mpp_y_transform(y_pp)
    x_pp = dist.mpp_x_transform(x_pp)

    params = np.polyfit(x_pp, y_pp, 1)

    params = np.array(dist.unpack_rr(params, "y"))

    results = {}
    results["params"] = params
    np.seterr(**old_err_state)
    return results


def mpp(model):
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
        return dist.mpp(
            x,
            c,
            n,
            heuristic=heuristic,
            rr=rr,
            on_d_is_0=on_d_is_0,
            offset=offset,
        )

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
        # I think this should be x[c != 1] and not in xl
        x_min = np.min(x_pp)

        # fun = lambda gamma : -pearsonr(np.log(x - gamma), y_)[0]
        def fun(gamma):
            g = x_min - np.exp(-gamma)
            out = -pearsonr(dist.mpp_x_transform(x_pp - g), y_pp)[0]
            return out

        res = minimize(fun, 0.0)
        gamma = x_min - np.exp(-res.x[0])
        x_pp = x_pp - gamma

    x_pp = dist.mpp_x_transform(x_pp)

    if rr == "y":
        params = np.polyfit(x_pp, y_pp, 1)
    elif rr == "x":
        params = np.polyfit(y_pp, x_pp, 1)

    params = np.array(dist.unpack_rr(params, rr))

    results = {}

    if offset:
        results["gamma"] = gamma

    results["params"] = params
    results["rr"] = rr

    return results
