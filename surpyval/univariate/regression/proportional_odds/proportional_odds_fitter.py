import autograd.numpy as np
import numpy.typing as npt


from .._likelihood import regression_neg_ll
from .._fit_skeleton import (
    LogLinearPhi,
    assemble_regression_model,
    optimise_nm_tnc,
    prepare_regression_fit,
)
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
            self._phi_bounds,
            self._phi_param_map,
        )
        init_t, bounds, pmap, transform, inv_trans, const, fixed = prep

        with np.errstate(all="ignore"):

            def fun(params):
                return self.neg_ll(data, *inv_trans(const(params)))

            res = optimise_nm_tnc(fun, init_t)

        params = inv_trans(const(res.x))
        reg_model = LogLinearPhi("Log Linear [exp(beta'Z)]", pmap)

        return assemble_regression_model(
            self,
            "Proportional Odds",
            reg_model,
            data,
            res,
            params,
            bounds,
            pmap,
            fixed,
        )


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
