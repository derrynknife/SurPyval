import autograd.numpy as np
import numpy.typing as npt


from .._likelihood import regression_neg_ll
from .._fit_skeleton import (
    HazardIdentitiesMixin,
    LogLinearPhi,
    assemble_regression_model,
    optimise_nm_tnc,
    prepare_regression_fit,
)
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin
from .aft_tvc_fit import AFTTVCFitMixin


class _LogLinearPhiModel:
    """Internal phi object: phi(Z) = exp(beta'Z)."""

    name = "Log Linear [exp(beta'Z)]"

    def phi(self, Z, *params):
        return np.exp(np.dot(Z, np.array(params)))

    def phi_bounds(self, Z):
        return ((None, None),) * Z.shape[1]

    def phi_param_map(self, Z):
        return {"beta_" + str(i): i for i in range(Z.shape[1])}


class AFTFitter(
    HazardIdentitiesMixin, AFTTVCFitMixin, DataFrameRegressionMixin
):
    """
    Accelerated Failure Time fitter using exp(beta'Z) as the acceleration
    factor.

    The cumulative hazard is:
        H(x | Z) = H_0(exp(beta'Z) * x)

    A positive beta coefficient means higher covariate values accelerate
    failure (shorter life), consistent with the PH sign convention.
    """

    def __init__(self, distribution):
        self.dist = distribution
        self.k_dist = len(distribution.param_names)
        self.bounds = distribution.bounds
        self.support = distribution.support
        self.param_names = distribution.param_names
        self.param_map = {v: i for i, v in enumerate(distribution.param_names)}
        self._phi_model = _LogLinearPhiModel()
        self.Hf_dist = distribution.Hf
        self.hf_dist = distribution.hf
        self.sf_dist = distribution.sf
        self.ff_dist = distribution.ff

    def _phi(self, Z, *phi_params):
        return self._phi_model.phi(Z, *phi_params)

    def Hf(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        return self.Hf_dist(self._phi(Z, *phi_params) * x, *dist_params)

    def hf(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi_val = self._phi(Z, *phi_params)
        return phi_val * self.hf_dist(phi_val * x, *dist_params)

    def neg_ll(self, data, *params):
        return regression_neg_ll(self, data, *params)

    def fit(
        self,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
    ) -> ParametricRegressionModel:
        data, prep = prepare_regression_fit(
            self,
            x,
            Z,
            c,
            n,
            t,
            init,
            fixed,
            self._phi_model.phi_bounds,
            self._phi_model.phi_param_map,
        )
        init_t, bounds, pmap, transform, inv_trans, const, fixed = prep

        with np.errstate(all="ignore"):

            def fun(params):
                return self.neg_ll(data, *inv_trans(const(params)))

            res = optimise_nm_tnc(fun, init_t)

        params = inv_trans(const(res.x))
        reg_model = LogLinearPhi(_LogLinearPhiModel.name, pmap)

        return assemble_regression_model(
            self,
            "Accelerated Failure Time",
            reg_model,
            data,
            res,
            params,
            bounds,
            pmap,
            fixed,
        )


def AFT(distribution):
    """
    Create an Accelerated Failure Time fitter for the given distribution.

    Uses exp(beta'Z) as the acceleration factor — the standard statistical
    parameterisation for AFT models.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Weibull``, ``LogNormal``).

    Returns
    -------
    AFTFitter
        A configured fitter with a ``.fit(x, Z, ...)`` method.

    Examples
    --------
    >>> from surpyval import Weibull
    >>> from surpyval import AFT
    >>> model = AFT(Weibull).fit(x, Z=covariates, c=c)
    """
    return AFTFitter(distribution)
