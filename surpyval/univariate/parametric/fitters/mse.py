from autograd import hessian, jacobian

from surpyval import np
from surpyval.univariate.nonparametric import fleming_harrington, turnbull
from surpyval.utils import xcnt_to_xrd

from . import fallback_minimize


def mse_fun(params, dist, x, F, inv_trans, const, offset):
    params = inv_trans(const(params))
    if offset:
        x = x - params[0]
        params = params[1:]
    return np.sum((dist.ff(x, *params) - F) ** 2)


def mse(model):
    """
    MSE: Mean Square Error
    This is simply fitting the curve to the best estimate from a non-parametric
    estimate.

    This is slightly different in that it fits it to untransformed data on
    the x and y axis. The MPP method fits the curve to the transformed data.
    This is simply fitting a the CDF sigmoid to the nonparametric estimate.
    """
    dist = model.dist
    x, c, n, t = (
        model.data["x"],
        model.data["c"],
        model.data["n"],
        model.data["t"],
    )

    const = model.fitting_info["const"]
    inv_trans = model.fitting_info["inv_trans"]
    init = model.fitting_info["init"]
    offset = model.offset

    if (-1 in c) or (2 in c):
        out = turnbull(x, c, n, t, estimator="Fleming-Harrington")
        F = 1 - out["R"]
        x = out["x"]
    else:
        x, r, d = xcnt_to_xrd(x, c, n, t)
        R = fleming_harrington(r, d)
        F = 1 - R

    mask = np.isfinite(x)
    F = F[mask]
    x = x[mask]

    jac = jacobian(mse_fun)
    hess = hessian(mse_fun)

    args = (dist, x, F, inv_trans, const, offset)
    res = fallback_minimize(mse_fun, init, args, jac, hess)

    results = {}
    results["res"] = res
    params = inv_trans(const(res.x))

    if offset:
        results["gamma"] = params[0]
        results["params"] = params[1:]
    else:
        results["gamma"] = 0.0
        results["params"] = params

    return results
