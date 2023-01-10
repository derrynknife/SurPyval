from scipy.optimize import minimize

from surpyval import np


def mom_fun(params, dist, inv_trans, const, offset, moments):
    dist_moments = dist.mom_moment_gen(
        *inv_trans(const(params)), offset=offset
    )
    # Unsure which is better
    # return (np.abs(moments - dist_moments) / moments).sum()
    return np.log(np.abs(moments - dist_moments) / moments).sum()


def mom(model):
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

    if hasattr(dist, "_mom"):
        return {"params": dist._mom(x_)}

    moments = np.zeros(model.k)

    for i in range(0, model.k):
        moments[i] = (x_ ** (i + 1)).mean()

    res = minimize(
        mom_fun,
        np.array(init),
        tol=1e-1,
        args=(dist, inv_trans, const, offset, moments),
    )

    params = inv_trans(const(res.x))

    results = {}
    if offset:
        results["gamma"] = params[0]
        results["params"] = params[1:]
    else:
        results["params"] = params

    results["res"] = res
    results["offset"] = offset

    return results
