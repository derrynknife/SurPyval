"""
Delta-method confidence bounds for the parametric regression models.

Every parametric regression fitter (AFT, parametric PH, PO, additive hazards,
accelerated life) produces a :class:`ParametricRegressionModel` fitted by
maximum likelihood, so its parameter uncertainty is the inverse of the
observed information (the Hessian of the negative log-likelihood at the MLE).
These helpers turn that covariance into

* Wald bounds on individual parameters (on a support-respecting scale), and
* delta-method bounds on the predicted ``sf``/``ff``/``Hf``/``hf``/``df`` at a
  covariate vector ``Z``.

The numeric machinery -- ``numerical_hessian``, ``delta_method_se``,
``bound_signs`` and ``log_transformed_cb`` -- lives in
``surpyval.utils.linalg`` and is shared with the recurrent-event inference
mixin; this module used to carry verbatim copies of all four (the
drift-prone pattern that produced #288) and now keeps only the logit-scale
survival bound, which is its own.
"""

import numpy as np
import numpy.typing as npt
from scipy.stats import norm


def logit_sf_bound(
    sf_hat: npt.ArrayLike,
    se: npt.ArrayLike,
    sign: float,
    alpha_tail: float,
) -> npt.NDArray:
    """
    A single survival-function confidence bound on the logit scale, which keeps
    it inside ``(0, 1)``: ``expit(logit(sf) + sign z se/(sf(1-sf)))``. ``sign``
    is ``-1`` for the lower bound and ``+1`` for the upper.
    """
    z = norm.ppf(1.0 - alpha_tail)
    est = np.clip(np.asarray(sf_hat, dtype=float), 1e-15, 1.0 - 1e-15)
    logit = np.log(est / (1.0 - est))
    with np.errstate(divide="ignore", invalid="ignore"):
        se_logit = np.asarray(se, dtype=float) / (est * (1.0 - est))
    return 1.0 / (1.0 + np.exp(-(logit + sign * z * se_logit)))
