import sys

from autograd import hessian, jacobian
from scipy.optimize import minimize

from surpyval import np


def mps_fun(params, dist, x, inv_trans, const, c, n, tl, tr, offset):
    if offset:
        x_new = x - inv_trans(const(params))[0]
        params = inv_trans(const(params))[1:]
    else:
        params = inv_trans(const(params))
        x_new = x.copy()
    D = dist.neg_mean_D(x_new, c, n, tl, tr, *params)
    return D


def mps(model):
    """
    MPS: Maximum Product Spacing

    This is the method to get the largest (geometric) average distance
    between all points. This method works really well when all points are
    unique. Some complication comes in when using repeated data. This method
    is quite good for offset distributions.
    """

    old_err_state = np.seterr(all="ignore")

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

    res = minimize(
        mps_fun,
        init,
        method="Newton-CG",
        jac=jac,
        hess=hess,
        tol=1e-15,
        args=(dist, x, inv_trans, const, c, n, tl, tr, offset),
    )

    if (res.success is False) or (np.isnan(res.x).any()):
        res = minimize(
            mps_fun,
            init,
            method="BFGS",
            jac=jac,
            args=(dist, x, inv_trans, const, c, n, tl, tr, offset),
        )

    if (res.success is False) or (np.isnan(res.x).any()):
        res = minimize(
            mps_fun,
            init,
            args=(dist, x, inv_trans, const, c, n, tl, tr, offset),
        )

    if (res.success is False) or (np.isnan(res.x).any()):
        print("MPS FAILED: Try alternate estimation method", file=sys.stderr)

    results = {}
    params = inv_trans(const(res.x))
    results["res"] = res

    if offset:
        results["gamma"] = params[0]
        results["params"] = params[1::]
    else:
        results["params"] = params

    results["jac"] = jac

    np.seterr(**old_err_state)

    return results
