import warnings
from math import comb

from scipy.optimize import minimize

from surpyval import np


def raw_to_central(moments):
    """``(mean, var, mu3, ...)`` from raw moments ``E[X], E[X^2], ...``.

    ``mu_k = sum_j C(k, j) (-1)^(k-j) E[X^j] mean^(k-j)``, with the mean
    kept in the leading slot because the first central moment is zero by
    construction and carries no information.

    The transform is exact and bijective, so matching the first ``k``
    central moments is the same estimator as matching the first ``k``
    raw ones -- it is only better conditioned. See ``mom_fun``.
    """
    mean = moments[0]
    central = [mean]
    for k in range(2, len(moments) + 1):
        acc = (-1) ** k * mean**k  # the j = 0 term, where E[X^0] = 1
        for j in range(1, k + 1):
            acc = acc + comb(k, j) * (-1) ** (k - j) * moments[j - 1] * (
                mean ** (k - j)
            )
        central.append(acc)
    return np.array(central)


def mom_fun(params, dist, inv_trans, const, offset, moments):
    """Squared mismatch between the sample and model moments.

    Compared as *central* moments scaled by the sample's own standard
    deviation, so every term is dimensionless and of comparable size:
    the mean in units of sigma, then the relative variance error, then
    the skewness difference, and so on.

    Raw moments describe the same estimator but hide it. Once a
    distribution is offset, ``E[X^k]`` is dominated by ``gamma^k`` and
    the shape contributes only a fractional correction -- 0.5% of
    ``E[X^3]`` for a Gamma(3, 4) shifted by 10. The optimiser is then
    reading three parameters off the third decimal place of a large
    number, and offset fits converged to parameter sets that matched the
    sample moments *better than the true parameters did* while being
    nowhere near them: a shape of 17.7 against a true 3.

    Central moments remove the offset by construction, so the shape
    information is the whole of ``mu_3`` rather than a rounding error in
    it. For unshifted fits the two agree to several decimal places,
    because there the conditioning was never the problem.
    """
    dist_moments = dist.mom_moment_gen(
        *inv_trans(const(params)), offset=offset
    )
    sample = raw_to_central(moments)
    model = raw_to_central(dist_moments)

    # sigma^k puts every term on a common footing. Taken from the sample
    # alone so the scale is a constant of the problem, not something the
    # optimiser can shrink to flatter itself.
    sigma = np.sqrt(np.abs(sample[1])) if len(sample) > 1 else 1.0
    if not np.isfinite(sigma) or sigma <= 0:
        sigma = 1.0
    scale = np.array([sigma ** (k + 1) for k in range(len(sample))])

    return (((sample - model) / scale) ** 2).sum()


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
    # The objective is a sum of squared standardised-moment differences
    # (see ``mom_fun``), so this threshold is in those units. Healthy
    # fits land in one of two places: ~1e-12 when the moment equations
    # have an exact solution, or ~1e-3 when sampling noise means no
    # parameter vector reproduces the sample moments exactly and the
    # optimiser returns the closest one -- a third central moment is
    # noisy enough at n=5000 for that to be routine. A fit that has
    # actually failed sits near 0.5. 1e-2 separates them with roughly an
    # order of magnitude of clearance on either side; the previous 1e-4
    # was calibrated against the old raw-moment objective and fires on
    # ordinary sampling noise under this one.
    if res.fun > 1e-2:
        warnings.warn(
            "MOM optimisation did not match the sample moments (squared "
            f"standardised-moment mismatch {res.fun:.3g}); the returned "
            "parameters may be unreliable. Consider how='MLE'."
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
