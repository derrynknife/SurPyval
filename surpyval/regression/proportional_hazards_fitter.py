import inspect

import autograd.numpy as np
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.surpyval_data import SurpyvalData

from .parametric_regression_model import ParametricRegressionModel


class Phi:
    pass


class ProportionalHazardsFitter:
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

    def df(self, x, Z, *params):
        return self.hf(x, Z, *params) * np.exp(-self.Hf(x, Z, *params))

    def sf(self, x, Z, *params):
        return np.exp(-self.Hf(x, Z, *params))

    def ff(self, x, Z, *params):
        return 1 - np.exp(-self.Hf(x, Z, *params))

    def _parameter_initialiser_dist(self, x, c=None, n=None, t=None):
        out = []
        for low, high in self.bounds:
            if (low is None) & (high is None):
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

    def log_df(self, x, Z, *params):
        return np.log(self.hf(x, Z, *params)) - self.Hf(x, Z, *params)

    def log_sf(self, x, Z, *params):
        return -self.Hf(x, Z, *params)

    def log_ff(self, x, Z, *params):
        return np.log(self.ff(x, Z, *params))

    def random(self, size, Z, *params):
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        U = np.random.uniform(0, 1, size)
        x = self.dist.qf(U ** (self.phi(Z, *phi_params)), *dist_params)
        Z_out = np.ones_like(x) * Z
        return x.flatten(), Z_out.flatten()

    def neg_ll(self, data, *params):
        ll = (
            (data.n_o * self.log_df(data.x_o, data.Z_o, *params)).sum()
            + (data.n_r * self.log_sf(data.x_r, data.Z_r, *params)).sum()
            + (data.n_l * self.log_ff(data.x_l, data.Z_l, *params)).sum()
            + (
                data.n_i
                * (
                    self.log_ff(data.x_ir, data.Z_i, *params)
                    - self.log_ff(data.x_il, data.Z_i, *params)
                )
            ).sum()
        )
        if (np.isfinite(data.tl).any()) | (np.isfinite(data.tr).any()):
            ll -= (
                data.n
                * (
                    self.log_ff(data.tr, data.Z, *params)
                    - self.log_ff(data.tl, data.Z, *params)
                )
            ).sum()
        return -ll

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

    def fit(self, Z, x, c=None, n=None, t=None, init=[], fixed={}):
        """
        Fit the proportional hazards model to the data.

        Parameters
        ----------

        Z : array_like
            The covariates to fit the model to.
        x : array_like
            The observed event times.
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
        >>> from autograd import numpy as anp
        >>> import numpy as np
        >>>
        >>> data = load_tires_data()
        >>>
        >>> x = data['Survival'].values
        >>> c = data['Censoring'].values
        >>> Z = data[[
            'Wedge gauge', 'Interbelt gauge', 'Peel force',
            'Wedge gauge×peel force'
        ]].values
        >>> model = WeibullPH.fit(x=x, Z=Z, c=c)
        >>> model
        Parametric Regression SurPyval Model
        ====================================
        Kind                : Proportional Hazard
        Distribution        : Weibull
        Regression Model    : Log Linear [e^(beta'Z)]
        Fitted by           : MLE
        Distribution        :
            alpha: 0.24255054642143947
            beta: 16.057791674515805
        Regression Model    :
            beta_0: -9.165062641226692
            beta_1: -7.998599877425742
            beta_2: -27.503283340963034
            beta_3: 18.38550143851751
        >>> model = WeibullPH.fit(x=x, Z=Z, c=c, fixed={"beta": 15})
        >>> model
        Parametric Regression SurPyval Model
        ====================================
        Kind                : Proportional Hazard
        Distribution        : Weibull
        Regression Model    : Log Linear [e^(beta'Z)]
        Fitted by           : MLE
        Distribution        :
            alpha: 0.23772915681951018
            beta: 15.0
        Regression Model    :
            beta_0: -8.628333861229965
            beta_1: -7.617541980158942
            beta_2: -25.952407717383302
            beta_3: 17.270173771235655
        """
        data = SurpyvalData(x, c, n, t, group_and_sort=False)
        data.add_covariates(Z)

        # Need to convert t to be at the edges of the support, if not
        # within it.
        # data.tl = data.t[:, 0]
        # data.tr = t[:, 1]

        # if np.isfinite(self.support[0]):
        # tl = np.where(tl < self.support[0], self.support[0], tl)

        # if np.isfinite(self.support[1]):
        # tr = np.where(tl > self.support[1], self.support[1], tr)

        if init == []:
            ps = self.dist.fit_from_surpyval_data(data).params
            if callable(self.phi_init):
                init_phi = self.phi_init(Z)

            init = np.array([*ps, *init_phi])
        else:
            init = np.array(init)

        # Dynamic or static bounds determination for models where the
        # number of covariates is not fixed in advance.
        if callable(self.phi_bounds):
            bounds = (*self.bounds, *self.phi_bounds(data.Z))
        else:
            bounds = (*self.bounds, *self.phi_bounds)

        # Dynamic or static parameter mapping for models where the
        # number of covariates is not fixed in advance.
        if callable(self.phi_param_map):
            phi_param_map = self.phi_param_map(data.Z)
        else:
            phi_param_map = self.phi_param_map

        param_map = {**self.param_map, **phi_param_map}

        # Create functions to make parameters unbounded for optimisation
        # Also create function to insert fixed values.
        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            x, bounds, fixed, param_map
        )

        init = transform(init)[not_fixed]

        with np.errstate(all="ignore"):

            def fun(params):
                return self.neg_ll(data, *inv_trans(const(params)))

            res = minimize(fun, init)
            res = minimize(fun, res.x, method="TNC")

        # Unpack parameters found from optimisation to actual values.
        params = inv_trans(const(res.x))

        # Create "Phi" model
        reg_model = Phi()
        reg_model.phi = self.phi
        reg_model.phi_param_map = phi_param_map
        reg_model.name = self.phi_name

        # Create regression model.
        model = ParametricRegressionModel()
        model.distribution_param_map = self.param_map
        model.phi_param_map = phi_param_map
        model.model = self
        model.reg_model = reg_model
        model.kind = "Proportional Hazard"
        model.distribution = self.dist
        model.params = np.array(params)
        model.res = res
        model._neg_ll = res["fun"]
        model.fixed = fixed
        model.k_dist = self.k_dist
        model.phi_param_map = phi_param_map
        model.data = data

        return model
