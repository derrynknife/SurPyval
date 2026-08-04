import inspect
from typing import Any

import autograd.numpy as np
import numpy.typing as npt
from .._fit_skeleton import (
    HazardIdentitiesMixin,
    assemble_regression_model,
    optimise_ph,
    prepare_regression_fit,
)
from .._likelihood import regression_neg_ll
from ..parametric_regression_model import ParametricRegressionModel
from ..regression_data import DataFrameRegressionMixin
from ..tvc_fit import TVCFitMixin


class Phi:
    # Lightweight namespace whose attributes are populated by the fitter.
    phi: Any
    phi_param_map: Any
    name: str


class ProportionalHazardsFitter(
    HazardIdentitiesMixin, TVCFitMixin, DataFrameRegressionMixin
):
    def __init__(
        self,
        name,
        dist,
        phi,
        phi_name,
        phi_bounds,
        phi_param_map,
        phi_init=None,
    ):
        if str(inspect.signature(phi)) != "(Z, *params)":
            raise ValueError(
                "PH function must have the signature '(Z, *params)'"
            )

        self.name = name
        self.dist = dist
        self.k_dist = len(self.dist.param_names)
        self.bounds = self.dist.bounds
        self.support = self.dist.support
        self.param_names = self.dist.param_names
        self.param_map = {v: i for i, v in enumerate(self.dist.param_names)}
        self.phi = phi
        self.phi_name = phi_name
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df
        self.phi_init = phi_init
        self.phi_bounds = phi_bounds
        self.phi_param_map = phi_param_map

    def Hf(self, x, Z, *params):
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        Hf_raw = self.Hf_dist(x, *dist_params)
        return self.phi(Z, *phi_params) * Hf_raw

    def hf(self, x, Z, *params):
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        hf_raw = self.hf_dist(x, *dist_params)
        return self.phi(Z, *phi_params) * hf_raw

    def _parameter_initialiser_dist(self, x, c=None, n=None, t=None):
        out = []
        for low, high in self.bounds:
            if (low is None) and (high is None):
                out.append(0)
            elif high is None:
                out.append(low + 1.0)
            elif low is None:
                out.append(high - 1.0)
            else:
                out.append((high + low) / 2.0)

        return out

    def mpp_inv_y_transform(self, y, *params):
        return y

    def mpp_y_transform(self, y, *params):
        return y

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def random(self, size, Z, *params):
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        Z_arr = np.atleast_2d(np.asarray(Z, dtype=float))
        x = []
        Z_out = []
        for row in Z_arr:
            phi = self.phi(row, *phi_params)
            U = np.random.uniform(0, 1, size)
            # S(x|Z) = S0(x)^phi, so inverting S(x|Z) = U gives
            # x = qf(1 - U^(1/phi)); U^phi inverts the wrong quantity.
            x.append(self.dist.qf(1 - U ** (1.0 / phi), *dist_params))
            Z_out.append(np.tile(row, (size, 1)))
        return np.concatenate(x), np.vstack(Z_out)

    def neg_ll(self, data, *params):
        return regression_neg_ll(self, data, *params)

    @staticmethod
    def create(distribution):
        """
        Create a Proportional Hazards fitter for the given distribution using
        exp(beta'Z) as the hazard multiplier.

        Parameters
        ----------
        distribution : ParametricFitter
            A surpyval parametric distribution (e.g. ``Weibull``,
            ``Exponential``).

        Returns
        -------
        ProportionalHazardsFitter
            A configured fitter with a ``.fit(x, Z, ...)`` method.
        """
        return ProportionalHazardsFitter.create_general_log_linear_fitter(
            f"{distribution.name}PH", distribution
        )

    @classmethod
    def create_general_log_linear_fitter(cls, name, distribution):
        return cls(
            name,
            distribution,
            lambda Z, *params: np.exp(np.dot(Z, np.array(params))),
            "Log Linear [e^(beta'Z)]",
            lambda Z: (((None, None),) * Z.shape[1]),
            phi_param_map=lambda Z: {
                "beta_" + str(i): i for i in range(Z.shape[1])
            },
            phi_init=lambda Z: np.zeros(Z.shape[1]),
        )

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
        """
        Fit the proportional hazards model to the data.

        Parameters
        ----------

        x : array_like
            The observed event times.
        Z : array_like
            The covariates to fit the model to.
        c : array_like, optional
            The censoring indicators.
        n : array_like, optional
            The number of observations at each time.
        t : array_like, optional
            The time intervals.
        init : array_like, optional
            The initial values for the parameters.
        fixed : dict, optional
            A dictionary of parameters to fix to a specific value.

        Returns
        -------

        ParametricRegressionModel
            The fitted model.

        Examples
        --------

        >>> from surpyval import WeibullPH
        >>> from surpyval.datasets import load_tires_data
        >>> data = load_tires_data()
        >>> x = data['Survival'].values
        >>> c = data['Censoring'].values
        >>> Z = data[[
        ...     'Wedge gauge', 'Interbelt gauge', 'Peel force',
        ...     'Wedge gauge×peel force'
        ... ]].values
        >>> model = WeibullPH.fit(x=x, Z=Z, c=c)
        >>> model
        Parametric Regression SurPyval Model
        ====================================
        Kind                : Proportional Hazard
        Distribution        : Weibull
        Regression Model    : Log Linear [e^(beta'Z)]
        Fitted by           : MLE
        Distribution        :
            alpha: 0.24255136...
            beta: 16.057785...
        Regression Model    :
            beta_0: -9.1650627...
            beta_1: -7.9985730...
            beta_2: -27.503185...
            beta_3: 18.385445...
        >>> model = WeibullPH.fit(x=x, Z=Z, c=c, fixed={"beta": 15})
        >>> model
        Parametric Regression SurPyval Model
        ====================================
        Kind                : Proportional Hazard
        Distribution        : Weibull
        Regression Model    : Log Linear [e^(beta'Z)]
        Fitted by           : MLE
        Distribution        :
            alpha: 0.23772966...
            beta: 15.0
        Regression Model    :
            beta_0: -8.6283269...
            beta_1: -7.6175293...
            beta_2: -25.952367...
            beta_3: 17.270148...
        """
        data, prep = prepare_regression_fit(
            self,
            x,
            Z,
            c,
            n,
            t,
            init,
            fixed,
            self.phi_bounds,
            self.phi_param_map,
            self.phi_init,
        )
        init_t, bounds, pmap, transform, inv_trans, const, fixed = prep

        with np.errstate(all="ignore"):

            def fun(params):
                return self.neg_ll(data, *inv_trans(const(params)))

            res = optimise_ph(fun, init_t)

        params = inv_trans(const(res.x))

        # Keep this fitter's possibly-custom phi (and its historical
        # serialisation name) rather than assuming log-linear.
        reg_model = Phi()
        reg_model.phi = self.phi
        reg_model.phi_param_map = pmap
        reg_model.name = self.phi_name

        return assemble_regression_model(
            self,
            "Proportional Hazard",
            reg_model,
            data,
            res,
            params,
            bounds,
            pmap,
            fixed,
        )
