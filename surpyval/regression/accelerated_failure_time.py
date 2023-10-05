import autograd.numpy as np
from scipy.optimize import minimize

import surpyval
from surpyval.univariate.parametric.fitters import bounds_convert

from .regression import Regression


class AcceleratedFailureTimeFitter:
    def __init__(self, name, distribution, acc_model):
        self.name = name
        self.dist = distribution
        self.acc_model = acc_model
        self.k_dist = len(self.dist.param_names)
        self.bounds = self.dist.bounds
        self.support = self.dist.support
        self.param_names = self.dist.param_names
        self.param_map = {v: i for i, v in enumerate(self.dist.param_names)}
        self.phi = acc_model.phi
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df

    def Hf(self, x, Z, *params):
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        return self.Hf_dist(self.phi(Z, *phi_params) * x, *dist_params)

    def hf(self, x, Z, *params):
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])
        return self.hf_dist(self.phi(Z, *phi_params) * x, *dist_params)

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

    def neg_ll(self, Z, x, c, n, *params):
        params = np.array(params)

        like = np.zeros_like(x).astype(float)
        like = np.where(c == 0, self.log_df(x, Z, *params), like)
        like = np.where(c == 1, self.log_sf(x, Z, *params), like)
        like = np.where(c == -1, self.log_ff(x, Z, *params), like)

        like = np.multiply(n, like)
        return -np.sum(like)

    def random(self, size, Z, *params):
        dist_params = np.array(params[0 : self.k_dist])
        phi_params = np.array(params[self.k_dist :])

        x = []
        Z_out = []

        for stress in np.unique(Z):
            U = np.random.uniform(0, 1, size)
            x.append(
                self.dist.qf(U, *dist_params) / self.phi(stress, *phi_params)
            )
            Z_out.append(np.ones(size) * stress)
        return np.concatenate(x), np.concatenate(Z_out)

    def fit(self, Z, x, c=None, n=None, t=None, init=[], fixed={}):
        x, c, n, t = surpyval.xcnt_handler(
            x=x, c=c, n=n, t=t, group_and_sort=False
        )

        if init == []:
            stress_data = np.unique(Z, axis=0)
            params_at_Z = []
            for s in stress_data:
                params_at_Z.append(
                    self.dist.fit(x[Z == s], c[Z == s], n[Z == s]).params
                )

            params_at_Z = np.array(params_at_Z)
            dist_init = params_at_Z.mean(axis=0)

            acc_parameter_data = params_at_Z[
                :, self.param_map[self.fixed_parameter]
            ]
            acc_parameter_data = self.acc_parameter_relationship(
                acc_parameter_data
            )

            if callable(self.acc_model.phi_init):
                phi_init = self.acc_model.phi_init(
                    acc_parameter_data, stress_data
                )
            else:
                phi_init = self.acc_model.phi_init

            init = np.array([*dist_init, *phi_init])
        else:
            init = np.array(init)

        if self.fixed != {}:
            fixed = {**self.fixed, **fixed}

        # Dynamic or static bounds determination
        if callable(self.acc_model.phi_bounds):
            bounds = (*self.bounds, *self.acc_model.phi_bounds(Z))
        else:
            bounds = (*self.bounds, *self.acc_model.phi_bounds)

        if callable(self.acc_model.phi_param_map):
            phi_param_map = self.acc_model.phi_param_map(Z)
        else:
            phi_param_map = self.acc_model.phi_param_map

        param_map = {
            **self.param_map,
            **{k: v + len(self.param_map) for k, v in phi_param_map.items()},
        }

        transform, inv_trans, const, fixed_idx, not_fixed = bounds_convert(
            x, bounds, fixed, param_map
        )

        init = transform(init)[not_fixed]

        with np.errstate(all="ignore"):

            def fun(params):
                return self.neg_ll(Z, x, c, n, *inv_trans(const(params)))

            # fun  = lambda params : self.neg_ll(Z, x, c, n, *params)
            # jac = jacobian(fun)
            # hess = hessian(fun)
            res = minimize(fun, init)
            res = minimize(fun, res.x, method="TNC")
            # res = minimize(fun, init, jac=jac, method='BFGS')
            # res = minimize(fun, init, method='Newton-CG', jac=jac)

        params = inv_trans(const(res.x))
        model = Regression()
        model.model = self
        model.reg_model = self.acc_model
        model.kind = "Accelerated Failure Time"
        model.distribution = self.dist
        model.params = np.array(params)
        model.res = res
        model._neg_ll = res["fun"]
        model.fixed = self.fixed
        model.k_dist = self.k_dist
        model.k = len(bounds)

        model.data = {"x": x, "c": c, "n": n, "t": t}

        return model
