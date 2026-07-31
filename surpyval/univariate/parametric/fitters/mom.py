import warnings

from scipy.optimize import minimize

from surpyval import np


def mom_fun(params, dist, inv_trans, const, offset, moments):
    dist_moments = dist.mom_moment_gen(
        *inv_trans(const(params)), offset=offset
    )
    return ((moments - dist_moments) ** 2 / moments).sum()


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

    # The closed-form moment estimate cannot honour fixed parameters or
    # an offset, so only use it for a plain fit
    if (
        hasattr(dist, "_mom")
        and not offset
        and not model.fitting_info["fixed_idx"]
    ):
        return {"params": np.atleast_1d(dist._mom(x_)), "gamma": 0.0}

    moments = np.zeros(model.k)

    for i in range(0, model.k):
        moments[i] = (x_ ** (i + 1)).mean()

    # A loose tolerance here silently returned parameters far from the
    # moment-matching solution for offset/fixed fits (#275): use a tight
    # tolerance, polish with Nelder-Mead if needed, and warn when the
    # relative moment mismatch remains large.
    res = minimize(
        mom_fun,
        np.array(init),
        args=(dist, inv_trans, const, offset, moments),
    )
    if not res.success or res.fun > 1e-8:
        res_nm = minimize(
            mom_fun,
            res.x if np.all(np.isfinite(res.x)) else np.array(init),
            method="Nelder-Mead",
            options={"maxiter": 10000, "xatol": 1e-12, "fatol": 1e-12},
            args=(dist, inv_trans, const, offset, moments),
        )
        if np.isfinite(res_nm.fun) and res_nm.fun < res.fun:
            res = res_nm
    if res.fun > 1e-4:
        warnings.warn(
            "MOM optimisation did not match the sample moments (relative "
            f"squared mismatch {res.fun:.3g}); the returned parameters may "
            "be unreliable. Consider how='MLE'."
        )

    params = inv_trans(const(res.x))

    results = {}
    if offset:
        results["gamma"] = params[0]
        results["params"] = params[1:]
    else:
        results["gamma"] = 0.0
        results["params"] = params

    results["res"] = res

    return results
