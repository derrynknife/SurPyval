import inspect
import warnings

import autograd.numpy as np
from scipy.optimize import minimize

from surpyval.univariate.parametric.fitters import bounds_convert
from surpyval.utils.surpyval_data import SurpyvalData

from .parametric_regression_model import ParametricRegressionModel


class ParameterSubstitutionFitter:
    def __init__(
        self,
        kind,
        name,
        distribution,
        life_model,
        life_parameter,
        baseline=[],
        param_transform=None,
        inverse_param_transform=None,
    ):
        if type(baseline) != list:
            # Baseline used if using a function that deviates from some number,
            # e.g. np.exp(np.dot(Z, beta))
            baseline = [baseline]

        self.name = name
        self.kind = kind
        self.dist = distribution
        self.life_model = life_model
        self.k_dist = len(self.dist.param_names)
        self.bounds = self.dist.bounds
        self.support = self.dist.support
        self.param_names = self.dist.param_names
        self.param_map = {v: i for i, v in enumerate(self.dist.param_names)}
        self.phi = life_model.phi
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df
        self.baseline = baseline
        self.life_parameter = life_parameter
        self.fixed = {life_parameter: 1.0}

        if param_transform is None:
            self.param_transform = lambda x: x
            self.inverse_param_transform = lambda x: x
        else:
            self.param_transform = param_transform
            self.inverse_param_transform = inverse_param_transform

    def Hf(self, x, Z, *params):
        x = np.array(x)
        if np.isscalar(Z):
            Z = np.ones_like(x) * Z
        else:
            Z = np.array(Z)

        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])

        Hf = np.zeros_like(x)
        stresses = np.unique(Z, axis=0)
        for stress in stresses:
            life_param_mask = (
                np.array(range(0, len(dist_params)))
                == self.param_map[self.life_parameter]
            )
            dist_params_i = np.where(
                life_param_mask,
                self.param_transform(self.phi(stress, *phi_params)),
                dist_params,
            )
            mask = (Z == stress).all(axis=1)
            Hf = np.where(mask, self.Hf_dist(x, *dist_params_i), Hf)

        return Hf

    def hf(self, x, Z, *params):
        x = np.array(x)
        if np.isscalar(Z):
            Z = np.ones_like(x) * Z
        else:
            Z = np.array(Z)

        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])

        hf = np.zeros_like(x)
        for stress in np.unique(Z, axis=0):
            life_param_mask = (
                np.array(range(0, len(dist_params)))
                == self.param_map[self.life_parameter]
            )
            params = np.where(
                life_param_mask,
                self.param_transform(self.phi(stress, *phi_params)),
                dist_params,
            )
            mask = (Z == stress).all(axis=1)
            hf = np.where(mask, self.hf_dist(x, *params), hf)

        return hf

    def df(self, x, Z, *params):
        x = np.array(x)
        if np.isscalar(Z):
            Z = np.ones_like(x) * Z
        else:
            Z = np.array(Z)
        return self.hf(x, Z, *params) * np.exp(-self.Hf(x, Z, *params))

    def sf(self, x, Z, *params):
        x = np.array(x)
        if np.isscalar(Z):
            Z = np.ones_like(x) * Z
        else:
            Z = np.array(Z)
        return np.exp(-self.Hf(x, Z, *params))

    def ff(self, x, Z, *params):
        x = np.array(x)
        if np.isscalar(Z):
            Z = np.ones_like(x) * Z
        else:
            Z = np.array(Z)
        return -np.expm1(-self.Hf(x, Z, *params))

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

        x = []
        Z_out = []
        if type(Z) == tuple:
            Z = np.random.uniform(*Z, size)

        for stress in np.unique(Z, axis=0):
            life_param_mask = (
                np.array(range(0, len(dist_params)))
                == self.param_map[self.life_parameter]
            )
            dist_params = np.where(
                life_param_mask,
                self.param_transform(self.phi(stress, *phi_params)),
                dist_params,
            )

            U = np.random.uniform(0, 1, size)
            x.append(self.dist.qf(U, *dist_params))
            if np.isscalar(stress):
                cols = 1
            else:
                cols = len(stress)
            Z_out.append(np.ones((size, cols)) * stress)
        return np.array(x).flatten(), np.concatenate(Z_out)

    def neg_ll(self, Z, x, c, n, *params):
        like = np.zeros_like(x).astype(float)
        like = np.where(c == 0, self.log_df(x, Z, *params), like)
        like = np.where(c == 1, self.log_sf(x, Z, *params), like)
        like = np.where(c == -1, self.log_ff(x, Z, *params), like)
        like = np.multiply(n, like)
        like = -np.sum(like)
        return like

    def fit(self, Z, x, c=None, n=None, t=None, init=[], fixed={}):
        data = SurpyvalData(x=x, c=c, n=n, t=t, group_and_sort=False)
        data.add_covariates(Z)
        life_parameter_idx = self.param_map[self.life_parameter]
        if init == []:
            stress_data = []
            params_at_Z = []

            # How do I make this work when there is only one failure per
            # stress?
            base_line_dist_init = self.dist.fit_from_surpyval_data(data).params

            for s in np.unique(data.Z, axis=0):
                mask = (data.Z == s).all(axis=1)
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    try:
                        params_at_s = self.dist.fit_from_surpyval_data(
                            data[mask]
                        ).params
                        params_at_Z.append(params_at_s)
                    except:  # noqa: E722
                        params_at_s = np.copy(base_line_dist_init)
                        params_at_s[life_parameter_idx] = x[mask].mean()
                        params_at_Z.append(params_at_s)
                    finally:
                        stress_data.append(s)

            params_at_Z = np.array(params_at_Z)
            dist_init = params_at_Z.mean(axis=0)

            stress_data = np.array(stress_data)

            if len(params_at_Z) < 2:
                raise ValueError(
                    "Insufficient data at separate Z values. Try manually \
                    setting initial guess using `init` keyword in `fit`"
                )

            parameter_data = params_at_Z[:, life_parameter_idx]

            parameter_data = self.inverse_param_transform(parameter_data)

            if callable(self.life_model.phi_init):
                if str(inspect.signature(self.life_model.phi_init)) == "(Z)":
                    phi_init = self.life_model.phi_init(Z)
                else:
                    phi_init = self.life_model.phi_init(
                        parameter_data, stress_data
                    )
            else:
                phi_init = self.life_model.phi_init
            init = np.array([*dist_init, *phi_init])
        else:
            init = np.array(init)

        if self.baseline != []:
            baseline_model = self.dist.fit_from_surpyval_data(data)
            baseline_fixed = {
                k: baseline_model.params[baseline_model.param_map[k]]
                for k in self.baseline
            }
            fixed = {**baseline_fixed, **fixed}

        if self.fixed != {}:
            fixed = {**self.fixed, **fixed}

        # Dynamic or static bounds determination
        if callable(self.life_model.phi_bounds):
            bounds = (*self.bounds, *self.life_model.phi_bounds(data.Z))
        else:
            bounds = (*self.bounds, *self.life_model.phi_bounds)

        if callable(self.life_model.phi_param_map):
            phi_param_map = self.life_model.phi_param_map(data.Z)
        else:
            phi_param_map = self.life_model.phi_param_map

        param_map = {
            **self.param_map,
            **{k: v + len(self.param_map) for k, v in phi_param_map.items()},
        }
        self.param_map = param_map

        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            x, bounds, fixed, param_map
        )

        init = transform(init)[not_fixed]

        with np.errstate(all="ignore"):

            def fun(params):
                return self.neg_ll(
                    data.Z, data.x, data.c, data.n, *inv_trans(const(params))
                )

            res1 = minimize(
                fun, init, method="Nelder-Mead", options={"maxiter": 1000}
            )
            res2 = minimize(
                fun,
                res1.x,
                method="TNC",
                # tol=1e-20,
                # options={"maxiter": 1000},
            )
            if not res2.success:
                res = res1
            else:
                res = res2

        params = inv_trans(const(res.x))
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])

        model = ParametricRegressionModel()
        model.model = self
        model.kind = self.kind
        model.distribution = self.dist
        model.reg_model = self.life_model
        model.params = np.array(params)
        model.dist_params = dist_params
        model.phi_params = phi_params
        model.res = res
        model._neg_ll = res.fun
        model.fixed = self.fixed
        model.k_dist = self.k_dist
        model.fun = fun

        model.k = len(bounds)

        model.data = {"x": x, "c": c, "n": n, "t": t}
        model.data = data

        return model
