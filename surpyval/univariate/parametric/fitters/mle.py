import warnings

from autograd import hessian, jacobian
from autograd.numpy.linalg import inv
from numdifftools import Hessian  # type: ignore
from scipy.optimize import OptimizeResult, minimize

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
        if len(init) == 0:
            # Every parameter is fixed; there is nothing to optimise
            res = OptimizeResult(
                x=init,
                success=True,
                fun=fun(init, offset, lfp, zi, True),
                message="",
            )
            best_result = res
            best_method = method = "all parameters fixed"
            methods = []
        else:
            methods = [
                ("BFGS", jac, None),
                ("TNC", jac, None),
                ("Newton-CG", jac, hess),
                ("Nelder-Mead", None, None),
                ("Powell", None, None),
            ]

        # Gradient methods first, stopping at the first that converges;
        # the derivative free pair is the fallback for when they do not.
        #
        # All five used to run on every fit, and the last four were
        # almost always confirming what an earlier one had already found:
        # over 102 fits across eleven distributions the whole ladder
        # agreed on the objective to 1e-10. The cost was not small.
        # Nelder-Mead and Powell are derivative free, so they pay for
        # their robustness in function evaluations -- 50 and 22 of them
        # against BFGS's 21 -- and every evaluation is O(n). On a
        # million observations those two alone were 42% of the fit.
        #
        # Order and early exit have to change together. Stopping early
        # without reordering halts at Nelder-Mead, which is both the
        # most expensive rung and the one with the worst objective;
        # reordering without stopping early saves nothing at all.
        #
        # The derivative free methods still start from the cold initial
        # guess when they are reached, so the multi-start behaviour that
        # motivated the original order survives for the fits that
        # actually need it -- they are simply no longer paid for by the
        # fits that do not.
        for method, jac_i, hess_i in methods:
            opts = {"maxfun": 1000} if method == "TNC" else {"maxiter": 1000}
            if method in ("Nelder-Mead", "Powell") or best_result is None:
                x0 = init
            else:
                x0 = best_result.x
            res = minimize(
                fun,
                x0,
                args=(offset, lfp, zi, True),
                method=method,
                jac=jac_i,
                hess=hess_i,
                options=opts,
            )
            if res.success and res.fun < best:
                best_result = res
                best_method = method
                best = res.fun
                break

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
                "\n- Using alternate fitting method"
                "\n- visually checking model fit"
                "\n- change data to be closer to 1."
            )

        elif (not res.success) or (np.isnan(res.x).any()):
            warnings.warn(
                "MLE Failed, using MPP results instead. "
                "Try making the values of the data closer to "
                "1 by dividing or multiplying by some constant."
                "\n\nAlternately try setting the `init` keyword in"
                " the `fit()`"
                " method to a value you believe is closer."
                "A good way to do this is to set any shape parameter to 1. "
                "and any scale parameter to be the mean of the data "
                "(or it's inverse)"
                "\n\nModel returned with inital guesses (MPP)"
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

        # The covariance of the parameters is found from the Hessian in
        # the transformed (unbounded) space used during optimisation,
        # then mapped back to the bounded parameter space with the delta
        # method. p and f0 are estimated parameters and are included in
        # the covariance. User-fixed parameters are known, not
        # estimated, so they carry no variance and the free parameters
        # get their conditional variance. gamma is also held at its
        # estimate since the threshold parameter of an offset model is
        # non-regular and a Wald variance for it would be misleading.
        fixed_idx = model.fitting_info["fixed_idx"]
        u_full = const(init) if use_initial else const(res.x)
        n_head = 1 if offset else 0
        n_core = len(params)
        n_total = len(u_full)
        var_idx = np.array(
            [i for i in range(n_head, n_total) if i not in fixed_idx],
            dtype=int,
        )

        # Embed the variance-carrying sub-vector into the full
        # transformed vector; the matrix form keeps the held entries
        # constant under autograd
        embed = np.zeros((n_total, len(var_idx)))
        embed[var_idx, np.arange(len(var_idx))] = 1.0
        u_held = np.where(embed.sum(axis=1) == 0, u_full, 0.0)

        def transformed_fun(u):
            theta = inv_trans(embed @ u + u_held)[n_head:]
            if zi:
                *theta, f0_i = theta
            else:
                f0_i = f0
            if lfp:
                *theta, p_i = theta
            else:
                p_i = p
            return model.dist._neg_ll_func(
                model.surv_data, *theta, gamma, f0_i, p_i
            )

        def u_to_phi(u):
            return inv_trans(embed @ u + u_held)[n_head:]

        try:
            if len(var_idx) == 0:
                cov_matrix = np.zeros((n_total - n_head, n_total - n_head))
            else:
                u_var = u_full[var_idx]
                hess_u = hessian(transformed_fun)(u_var)
                # A corrupted autograd Hessian (e.g. a primitive whose
                # VJP silently drops second-order terms) shows up as
                # asymmetry; recompute numerically rather than invert
                # garbage (#270).
                asym = np.max(np.abs(hess_u - hess_u.T)) > 1e-4 * max(
                    np.max(np.abs(hess_u)), 1.0
                )
                if np.isnan(hess_u).any() or asym:
                    hess_u = Hessian(transformed_fun)(u_var)
                cov_u = inv(hess_u)
                if np.isnan(cov_u).any():
                    cov_u = inv(Hessian(transformed_fun)(u_var))
                jac_u = jacobian(u_to_phi)(u_var)
                # Covariance of the extended vector (*params, p?, f0?);
                # fixed parameters have zero rows and columns
                cov_matrix = jac_u @ cov_u @ jac_u.T
            hess_inv = cov_matrix[:n_core, :n_core]
        except np.linalg.LinAlgError:
            cov_matrix = None
            hess_inv = None

        results["cov_matrix"] = cov_matrix
        results["hess_inv"] = hess_inv
        # On the fallback path the returned parameters are the initial
        # guess, so the reported likelihood must be evaluated there — not
        # taken from the failed optimizer (#261).
        if use_initial:
            with np.errstate(all="ignore"):
                neg_ll_val = float(fun(init, offset, lfp, zi, True))
        else:
            neg_ll_val = float(res["fun"])
        results["_neg_ll"] = neg_ll_val
        results["log_likelihood"] = -neg_ll_val
        results["res"] = res
        results["optimizer"] = (
            best_method if best_method is not None else method
        )

    return results
