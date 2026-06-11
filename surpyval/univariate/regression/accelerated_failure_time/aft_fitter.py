import autograd.numpy as np
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.surpyval_data import SurpyvalData

from ..parametric_regression_model import ParametricRegressionModel


class _LogLinearPhiModel:
    """Internal phi object: phi(Z) = exp(beta'Z)."""

    name = "Log Linear [exp(beta'Z)]"

    def phi(self, Z, *params):
        return np.exp(np.dot(Z, np.array(params)))

    def phi_bounds(self, Z):
        return ((None, None),) * Z.shape[1]

    def phi_param_map(self, Z):
        return {"beta_" + str(i): i for i in range(Z.shape[1])}


class AFTFitter:
    """
    Accelerated Failure Time fitter using exp(beta'Z) as the acceleration factor.

    The cumulative hazard is:
        H(x | Z) = H_0(exp(beta'Z) * x)

    A positive beta coefficient means higher covariate values accelerate failure
    (shorter life), consistent with the PH sign convention.
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

    def sf(self, x, Z, *params):
        return np.exp(-self.Hf(x, Z, *params))

    def ff(self, x, Z, *params):
        return -np.expm1(-self.Hf(x, Z, *params))

    def df(self, x, Z, *params):
        return self.hf(x, Z, *params) * self.sf(x, Z, *params)

    def log_sf(self, x, Z, *params):
        return -self.Hf(x, Z, *params)

    def log_ff(self, x, Z, *params):
        return np.log(self.ff(x, Z, *params))

    def log_df(self, x, Z, *params):
        return np.log(self.hf(x, Z, *params)) - self.Hf(x, Z, *params)

    def neg_ll(self, Z, x, c, n, *params):
        like = np.zeros_like(x, dtype=float)
        like = np.where(c == 0,  self.log_df(x, Z, *params), like)
        like = np.where(c == 1,  self.log_sf(x, Z, *params), like)
        like = np.where(c == -1, self.log_ff(x, Z, *params), like)
        return -np.sum(n * like)

    def fit(self, x, Z, c=None, n=None, t=None, init=None, fixed=None):
        data = SurpyvalData(x, c, n, t, group_and_sort=False)
        data.add_covariates(Z)

        if fixed is None:
            fixed = {}

        if init is None:
            ps = self.dist.fit_from_surpyval_data(data).params
            phi_init = np.zeros(data.Z.shape[1])
            init = np.array([*ps, *phi_init])
        else:
            init = np.array(init)

        phi_bounds = self._phi_model.phi_bounds(data.Z)
        phi_param_map = self._phi_model.phi_param_map(data.Z)
        bounds = (*self.bounds, *phi_bounds)

        param_map = {
            **self.param_map,
            **{k: v + len(self.param_map) for k, v in phi_param_map.items()},
        }

        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            data.x, bounds, fixed, param_map
        )
        init = transform(init)[not_fixed]

        with np.errstate(all="ignore"):

            def fun(params):
                return self.neg_ll(
                    data.Z, data.x, data.c, data.n, *inv_trans(const(params))
                )

            res = minimize(fun, init, method="Nelder-Mead", options={"maxiter": 1000})
            res2 = minimize(fun, res.x, method="TNC")
            res = res2 if res2.success else res

        params = inv_trans(const(res.x))

        # Build a lightweight phi model object for ParametricRegressionModel
        _pm = phi_param_map

        class _PhiModel:
            name = _LogLinearPhiModel.name
            phi_param_map = _pm

            def phi(self, Z, *p):
                return np.exp(np.dot(Z, np.array(p)))

        model = ParametricRegressionModel()
        model.model = self
        model.reg_model = _PhiModel()
        model.kind = "Accelerated Failure Time"
        model.distribution = self.dist
        model.params = np.array(params)
        model.dist_params = np.array(params[: self.k_dist])
        model.phi_params = np.array(params[self.k_dist :])
        model.res = res
        model._neg_ll = res.fun
        model.fixed = fixed
        model.k_dist = self.k_dist
        model.k = len(bounds)
        model.data = data

        return model


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
    >>> from surpyval.regression import AFT
    >>> model = AFT(Weibull).fit(x, Z=covariates, c=c)
    """
    return AFTFitter(distribution)
