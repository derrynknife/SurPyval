from typing import Any

import autograd.numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric.parametric_fitter import (
    Boxable,
    Numeric,
)
from surpyval.utils.surpyval_data import SurpyvalData

from .._fit_skeleton import (
    HazardIdentitiesMixin,
    LogLinearPhi,
    assemble_regression_model,
    make_objective,
    mirror_distribution,
    optimise_nm_tnc,
    prepare_regression_fit,
)
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin
from .aft_tvc_fit import AFTTVCFitMixin


class AFTFitter(
    HazardIdentitiesMixin,
    AFTTVCFitMixin,
    DataFrameRegressionMixin,
):
    """
    Accelerated Failure Time fitter using exp(beta'Z) as the acceleration
    factor.

    The cumulative hazard is:
        H(x | Z) = H_0(exp(beta'Z) * x)

    A positive beta coefficient means higher covariate values accelerate
    failure (shorter life), consistent with the PH sign convention.
    """

    def __init__(self, distribution: Any) -> None:
        mirror_distribution(self, distribution)
        self.Hf_dist = distribution.Hf
        self.hf_dist = distribution.hf
        self.sf_dist = distribution.sf
        self.ff_dist = distribution.ff

    def _phi(self, Z: Numeric, *phi_params: Boxable) -> Boxable:
        return LogLinearPhi.phi(Z, *phi_params)

    def Hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        return self.Hf_dist(self._phi(Z, *phi_params) * x, *dist_params)

    def hf(self, x: Numeric, Z: Numeric, *params: Boxable) -> Boxable:
        x = np.atleast_1d(np.asarray(x, dtype=float))
        Z = np.atleast_2d(np.asarray(Z, dtype=float))
        dist_params = params[: self.k_dist]
        phi_params = params[self.k_dist :]
        phi_val = self._phi(Z, *phi_params)
        return phi_val * self.hf_dist(phi_val * x, *dist_params)

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
            "Accelerated Failure Time",
            reg_model,
            data,
            res,
            params,
            bounds,
            pmap,
            fixed,
        )


def AFT(distribution: Any) -> "AFTFitter":
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
    >>> import numpy as np
    >>> from surpyval import Weibull
    >>> from surpyval import AFT
    >>> np.random.seed(1)
    >>> Z = np.random.binomial(1, 0.5, 100).reshape(-1, 1)
    >>> x = Weibull.random(100, 10, 2) * np.exp(-0.5 * Z[:, 0])
    >>> model = AFT(Weibull).fit(x, Z=Z)
    >>> model.params.round(3)
    array([9.629, 1.751, 0.473])
    """
    return AFTFitter(distribution)
