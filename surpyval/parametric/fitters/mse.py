from autograd import hessian, jacobian
from scipy.optimize import minimize

from surpyval import nonparametric as nonp
from surpyval import np
from surpyval.utils import xcnt_to_xrd


def mse_fun(params, dist, x, F, inv_trans, const):
    return np.sum(((dist.ff(x, *inv_trans(const(params)))) - F) ** 2)


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

    if (-1 in c) or (2 in c):
        out = nonp.turnbull(x, c, n, t, estimator="Fleming-Harrington")
        F = 1 - out["R"]
        x = out["x"]
    else:
        x, r, d = xcnt_to_xrd(x, c, n, t)
        R = nonp.fleming_harrington(r, d)
        F = 1 - R

    mask = np.isfinite(x)
    F = F[mask]
    x = x[mask]

    jac = jacobian(mse_fun)
    hess = hessian(mse_fun)

    old_err_state = np.seterr(all="ignore")

    res = minimize(
        mse_fun,
        init,
        method="Newton-CG",
        jac=jac,
        hess=hess,
        args=(dist, x, F, inv_trans, const),
    )

    if (res.success is False) or (np.isnan(res.x).any()):
        res = minimize(
            mse_fun,
            init,
            method="BFGS",
            jac=jac,
            args=(dist, x, F, inv_trans, const),
        )

    if (res.success is False) or (np.isnan(res.x).any()):
        res = minimize(mse_fun, init, args=(dist, x, F, inv_trans, const))

    results = {}
    results["res"] = res
    results["params"] = inv_trans(const(res.x))
    np.seterr(**old_err_state)
    return results
