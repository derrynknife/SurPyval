import warnings

from autograd import hessian, jacobian
from autograd.numpy.linalg import inv
from numdifftools import Hessian  # type: ignore
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

    use_initial = False
    jac = jacobian(fun)
    hess = hessian(fun)

    best = np.inf
    best_result = None
    best_method = None

    with np.errstate(all="ignore"):
        # Try easiest to most complex optimisations, warm-starting each
        # method from the best point found so far.
        x0 = np.array(init, dtype=float)
        for method, jac_i, hess_i in [
            ("Nelder-Mead", None, None),
            ("Powell", None, None),
            ("BFGS", None, None),
            ("TNC", jac, None),
            ("Newton-CG", jac, hess),
        ]:
            opts = {"maxfun": 1000} if method == "TNC" else {"maxiter": 1000}
            res = minimize(
                fun,
                x0,
                args=(offset, lfp, zi, True),
                method=method,
                jac=jac_i,
                hess=hess_i,
                options=opts,
            )
            if (
                res.success
                and np.isfinite(res.fun)
                and not np.isnan(res.x).any()
            ):
                if res.fun < best:
                    best_result = res
                    best_method = method
                    best = res.fun
                    x0 = res.x

        if best_result is not None:
            res = best_result

        winning_message = (
            best_result.get("message", "")
            if best_result is not None
            else res.get("message", "")
        )

        if "Desired error not necessarily" in winning_message:
            warnings.warn(
                "Precision was lost, try:"
                + "\n- Using alternate fitting method"
                + "\n- visually checking model fit"
                + "\n- change data to be closer to 1."
            )

        elif (not res.success) or (np.isnan(res.x).any()):
            warnings.warn(
                "MLE Failed, using MPP results instead. "
                + "Try making the values of the data closer to "
                + "1 by dividing or multiplying by some constant."
                + "\n\nAlternately try setting the `init` keyword in"
                + " the `fit()`"
                + " method to a value you believe is closer."
                + "A good way to do this is to set any shape parameter to 1. "
                + "and any scale parameter to be the mean of the data "
                + "(or it's inverse)"
                + "\n\nModel returned with inital guesses (MPP)"
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

        # Confidence bounds: compute the Hessian in the transformed
        # (unbounded) space the optimiser worked in, then map the
        # covariance back to the bounded parameter space with the delta
        # method. Evaluating curvature in the same space, and at the same
        # point, as the optimiser's solution is far more reliable for
        # bounded parameters than differentiating in the bounded space.
        # The covariance of gamma, f0, and p is marginalised out of the
        # returned block, and fixed parameters get zero variance.
        x_opt = np.array(init, dtype=float) if use_initial else res.x
        try:
            hess_t = hess(x_opt, offset, lfp, zi, True)
            if np.isnan(hess_t).any():
                hess_t = Hessian(lambda x: fun(x, offset, lfp, zi, True))(
                    x_opt
                )
            cov_t = inv(hess_t)
            # Jacobian of the free, transformed parameters to the full
            # vector of bounded parameters; rows for fixed parameters
            # are zero.
            jac_t = jacobian(lambda x: inv_trans(const(x)))(x_opt)
            cov = jac_t @ cov_t @ jac_t.T
            # Slice out the distribution's parameters; gamma is first
            # and p / f0 are last when present.
            start = 1 if offset else 0
            stop = cov.shape[0] - int(zi) - int(lfp)
            hess_inv = cov[start:stop, start:stop]
            if np.isnan(hess_inv).any():
                hess_inv = None
        except np.linalg.LinAlgError:
            hess_inv = None

        results["hess_inv"] = hess_inv
        results["_neg_ll"] = res["fun"]
        results["log_likelihood"] = -res["fun"]
        results["res"] = res
        results["optimizer"] = (
            best_method if best_method is not None else method
        )

    return results
