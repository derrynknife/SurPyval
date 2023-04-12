import sys

from autograd import hessian, jacobian
from autograd.numpy.linalg import inv
from scipy.optimize import minimize

from surpyval import np


def mle(model):
    """
    Maximum Likelihood Estimation (MLE)

    """
    # Function that adds in any fixed parameters
    const = model.fitting_info["const"]
    # Inverse transform function for parameters. i.e. from (None, None) to
    # correct bounded values
    inv_trans = model.fitting_info["inv_trans"]
    # Initial guess
    init = model.fitting_info["init"]
    # Offset, Limited Failure Population, Zero Inflated logic.
    offset, lfp, zi = model.offset, model.lfp, model.zi

    if hasattr(model.dist, "mle"):
        return model.dist.mle(model.surv_data)

    results = {}

    """
    Need to flag entries where truncation is inf or -inf so that the autograd
    doesn't fail. Because autograd fails if it encounters any inf, nan, -inf
    etc even if they don't affect the gradient. A must for autograd
    """

    def fun(
        params,
        offset=False,
        lfp=False,
        zi=False,
        transform=True,
        gamma=0,
        f0=0,
        p=1,
    ):

        # Transform parameters from (-Inf, Inf) range to parameter
        # to correct bounded values
        if transform:
            params = inv_trans(const(params))

        # Unpack offset, zi, lfp parameters
        if offset:
            gamma, *params = params

        if zi:
            *params, f0 = params

        if lfp:
            *params, p = params

        return model.dist._neg_ll_func(model.surv_data, *params, gamma, f0, p)

    old_err_state = np.seterr(all="ignore")
    use_initial = False
    jac = jacobian(fun)
    hess = hessian(fun)

    # Try easiest, to most complex optimisations
    for method, jac_i, hess_i in [
        ("Nelder-Mead", None, None),
        ("BFGS", None, None),
        ("TNC", jac, None),
        ("Newton-CG", jac, hess),
    ]:
        res = minimize(
            fun,
            init,
            args=(offset, lfp, zi, True),
            method=method,
            jac=jac_i,
            hess=hess_i,
        )
        if res.success:
            break

    if "Desired error not necessarily" in res["message"]:
        print(
            "Precision was lost, try:"
            + "\n- Using alternate fitting method"
            + "\n- visually checking model fit"
            + "\n- change data to be closer to 1.",
            file=sys.stderr,
        )

    elif (not res.success) | (np.isnan(res.x).any()):
        print(
            "MLE Failed, using MPP results instead. "
            + "Try making the values of the data closer to "
            + "1 by dividing or multiplying by some constant."
            + "\n\nAlternately try setting the `init` keyword in the `fit()`"
            + " method to a value you believe is closer."
            + "A good way to do this is to set any shape parameter to 1. "
            + "and any scale parameter to be the mean of the data "
            + "(or it's inverse)"
            + "\n\nModel returned with inital guesses (MPP)",
            file=sys.stderr,
        )

        use_initial = True

    if use_initial:
        params = inv_trans(const(init))
    else:
        params = inv_trans(const(res.x))

    if offset:
        gamma = params[0]
        params = params[1:]
    else:
        gamma = 0.0

    results["gamma"] = gamma

    if zi:
        f0 = params[-1]
        params = params[0:-1]
    else:
        f0 = 0.0
    results["f0"] = f0

    if lfp:
        p = params[-1]
        params = params[0:-1]
    else:
        p = 1.0

    results["p"] = p
    results["params"] = params
    # Do not account for variation of gamma, f0, p in confidence bounds.
    results["hess_inv"] = inv(
        hess(params, *(False, False, False, False, gamma, f0, p))
    )
    results["_neg_ll"] = res["fun"]
    results["res"] = res

    np.seterr(**old_err_state)

    return results
