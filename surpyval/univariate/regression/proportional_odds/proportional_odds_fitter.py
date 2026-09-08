from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.utils.surpyval_data import SurpyvalData

from .._fit_skeleton import (
    LogLinearPhi,
    MirroredDistributionAttrs,
    assemble_regression_model,
    make_objective,
    mirror_distribution,
    optimise_nm_tnc,
    prepare_regression_fit,
)
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin


class ProportionalOddsFitter(
    MirroredDistributionAttrs, DataFrameRegressionMixin
):
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

    def __init__(self, distribution: Any) -> None:
        mirror_distribution(self, distribution)
        self.Hf_dist = distribution.Hf
        self.hf_dist = distribution.hf
        self.sf_dist = distribution.sf
        self.ff_dist = distribution.ff
        self.df_dist = distribution.df

    def _phi(self, Z: Numeric, *phi_params: Boxable) -> Boxable:
        return LogLinearPhi.phi(Z, *phi_params)

    def sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return phi * S0 / (F0 + phi * S0)

    def ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return F0 / (F0 + phi * S0)

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        h0 = self.hf_dist(x, *dist_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return h0 / (F0 + phi * S0)

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        return -np.log(self.sf(x, Z, *params))

    def df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
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

    def log_sf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return np.log(phi) + np.log(S0) - np.log(F0 + phi * S0)

    def log_ff(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi = self._phi(Z, *phi_params)
        S0 = self.sf_dist(x, *dist_params)
        F0 = self.ff_dist(x, *dist_params)
        return np.log(F0) - np.log(F0 + phi * S0)

    def log_df(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
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

    def neg_ll(self, data: SurpyvalData, *params: Boxable) -> Boxable:
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
            LogLinearPhi.phi_bounds,
            LogLinearPhi.make_param_map,
        )
        init_t, bounds, pmap, transform, inv_trans, const, fixed = prep

        with np.errstate(all="ignore"):

            fun = make_objective(self, data, inv_trans, const)

            res = optimise_nm_tnc(fun, init_t)

        params = inv_trans(const(res.x))
        reg_model = LogLinearPhi(LogLinearPhi.NAME_EXP, pmap)

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


def PO(distribution: Any) -> "ProportionalOddsFitter":
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
    >>> import numpy as np
    >>> from surpyval import Logistic
    >>> from surpyval import PO
    >>> np.random.seed(1)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Logistic.random(100, 10, 2) + 2.0 * Z[:, 0]
    >>> model = PO(Logistic).fit(x, Z=Z)
    >>> model.params.round(3)
    array([9.708, 2.337, 0.918])
    """
    return ProportionalOddsFitter(distribution)
