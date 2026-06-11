import warnings

from autograd import hessian, jacobian

from surpyval import np

from . import fallback_minimize


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
        results["params"] = params

    results["jac"] = jac

    return results
