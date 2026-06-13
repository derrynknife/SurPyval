import autograd.numpy as np
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.surpyval_data import SurpyvalData

from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin


class ProportionalOddsFitter(DataFrameRegressionMixin):
    """
    Proportional Odds model fitter using exp(beta'Z) as the odds multiplier.

    The survival odds satisfy:
        O(x | Z) = O_0(x) * exp(beta'Z)   where O(x) = S(x) / F(x)

    This gives:
        sf(x | Z) = exp(beta'Z) * S_0(x) / (F_0(x) + exp(beta'Z) * S_0(x))
        ff(x | Z) = F_0(x) / (F_0(x) + exp(beta'Z) * S_0(x))
        hf(x | Z) = h_0(x) / (F_0(x) + exp(beta'Z) * S_0(x))

    A positive beta coefficient means higher covariate values increase the
    survival odds (protective effect — longer life). To match the PH sign
    convention (positive beta = shorter life), negate your covariates or betas.
    """

    def __init__(self, distribution):
        self.dist = distribution
        self.k_dist = len(distribution.param_names)
        self.bounds = distribution.bounds
        self.support = distribution.support
        self.param_names = distribution.param_names
        self.param_map = {v: i for i, v in enumerate(distribution.param_names)}
        self.Hf_dist = distribution.Hf
        self.hf_dist = distribution.hf
        self.sf_dist = distribution.sf
        self.ff_dist = distribution.ff
        self.df_dist = distribution.df

    def _phi_bounds(self, Z):
        return ((None, None),) * Z.shape[1]

    def _phi_param_map(self, Z):
        return {"beta_" + str(i): i for i in range(Z.shape[1])}

    def _phi(self, Z, *phi_params):
        return np.exp(np.dot(Z, np.array(phi_params)))

    def sf(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return phi * S0 / (F0 + phi * S0)

    def ff(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return F0 / (F0 + phi * S0)

    def hf(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        h0 = self.hf_dist(x, *dist_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return h0 / (F0 + phi * S0)

    def Hf(self, x, Z, *params):
        return -np.log(self.sf(x, Z, *params))

    def df(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        f0 = self.df_dist(x, *dist_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        denom = F0 + phi * S0
        return phi * f0 / (denom * denom)

    def log_sf(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return np.log(phi) + np.log(S0) - np.log(F0 + phi * S0)

    def log_ff(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return np.log(F0) - np.log(F0 + phi * S0)

    def log_df(self, x, Z, *params):
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        f0 = self.df_dist(x, *dist_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        denom = F0 + phi * S0
        return np.log(phi) + np.log(f0) - 2.0 * np.log(denom)

    def neg_ll(self, data, *params):
        return regression_neg_ll(self, data, *params)

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

        phi_bounds = self._phi_bounds(data.Z)
        phi_param_map = self._phi_param_map(data.Z)
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
                return self.neg_ll(data, *inv_trans(const(params)))

            res = minimize(
                fun, init, method="Nelder-Mead", options={"maxiter": 1000}
            )
            res2 = minimize(fun, res.x, method="TNC")
            res = res2 if res2.success else res

        params = inv_trans(const(res.x))

        _pm = phi_param_map

        class _PhiModel:
            name = "Log Linear [exp(beta'Z)]"
            phi_param_map = _pm

            def phi(self, Z, *p):
                return np.exp(np.dot(Z, np.array(p)))

        model = ParametricRegressionModel()
        model.model = self
        model.reg_model = _PhiModel()
        model.kind = "Proportional Odds"
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


def PO(distribution):
    """
    Create a Proportional Odds fitter for the given distribution.

    Uses exp(beta'Z) as the odds multiplier — the standard parameterisation
    for proportional odds survival models.

    Parameters
    ----------
    distribution : ParametricFitter
        A surpyval parametric distribution (e.g. ``Logistic``,
        ``LogLogistic``).

    Returns
    -------
    ProportionalOddsFitter
        A configured fitter with a ``.fit(x, Z, ...)`` method.

    Examples
    --------
    >>> from surpyval import Logistic
    >>> from surpyval import PO
    >>> model = PO(Logistic).fit(x, Z=covariates, c=c)
    """
    return ProportionalOddsFitter(distribution)
