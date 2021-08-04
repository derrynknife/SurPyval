import autograd.numpy as np
from autograd import jacobian, hessian, elementwise_grad
from scipy.optimize import minimize
import surpyval
import inspect

from ..parametric.fitters import bounds_convert, fix_idx_and_function
from .regression import Regression

class AcceleratedFailureTimeFitter():
    def __init__(self, name, distribution, acc_model):

        if str(inspect.signature(acc_model.phi)) != '(X, *params)':
            raise ValueError('PH function must have the signature \'(X, *params)\'')

        self.name = name
        self.dist = distribution
        self.acc_model = acc_model
        self.k_dist = len(self.dist.param_names)
        self.bounds = self.dist.bounds
        self.support = self.dist.support
        self.param_names = self.dist.param_names
        self.param_map = {v : i for i, v in enumerate(self.dist.param_names)}
        self.phi = acc_model.phi
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df

    def Hf(self, x, X, *params):
        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])
        return self.Hf_dist(self.phi(X, *phi_params) * x, *dist_params)

    def hf(self, x, X, *params):
        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])
        return self.hf_dist(self.phi(X, *phi_params) * x, *dist_params)

    def df(self, x, X, *params):
        return self.hf(x, X, *params) * np.exp(-self.Hf(x, X, *params))

    def sf(self, x, X, *params):
        return np.exp(-self.Hf(x, X, *params))

    def ff(self, x, X, *params):
        return 1 - np.exp(-self.Hf(x, X, *params))

    def _parameter_initialiser_dist(self, x, c=None, n=None, t=None):
        out = []
        for low, high in self.bounds:
            if (low is None) & (high is None):
                out.append(0)
            elif high is None:
                out.append(low + 1.)
            elif low is None:
                out.append(high - 1.)
            else:
                out.append((high + low)/2.)

        return out

    def mpp_inv_y_transform(self, y, *params):
        return y

    def mpp_y_transform(self, y, *params):
        return y

    def mpp_inv_y_transform(self, y, *params):
        return y

    def mpp_x_transform(self, x, gamma=0):
        return x - gamma

    def log_df(self, x, X, *params):
        return np.log(self.hf(x, X, *params)) - self.Hf(x, X, *params)

    def log_sf(self, x, X, *params):
        return -self.Hf(x, X, *params)

    def log_ff(self, x, X, *params):
        return np.log(self.ff(x, X, *params))

    def neg_ll(self, X, x, c, n, *params):
        params = np.array(params)

        like = np.zeros_like(x).astype(float)
        like = np.where(c ==  0, self.log_df(x, X, *params), like)
        like = np.where(c ==  1, self.log_sf(x, X, *params), like)
        like = np.where(c ==  -1, self.log_ff(x, X, *params), like)

        like = np.multiply(n, like)
        return -np.sum(like)

    def random(self, size, X, *params):
        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])

        x = []
        X_out = []

        for stress in np.unique(X):
            U = np.random.uniform(0, 1, size)
            x.append(self.dist.qf(U, *dist_params)/self.phi(stress, *phi_params))
            X_out.append(np.ones(size) * stress)
        return np.concatenate(x), np.concatenate(X_out)

    def fit(self, X, x, c=None, n=None, t=None, init=[], fixed={}):
        x, c, n, t = surpyval.xcnt_handler(x=x, c=c, n=n, t=t, group_and_sort=False)

        if init == []:
            stress_data = np.unique(X, axis=0)
            params_at_X = []
            for s in stress_data:
                params_at_X.append(self.dist.fit(x[X == s], c[X == s], n[X == s]).params)

            params_at_X = np.array(params_at_X)
            dist_init = params_at_X.mean(axis=0)

            acc_parameter_data = params_at_X[:, self.param_map[self.fixed_parameter]]
            acc_parameter_data = self.acc_parameter_relationship(acc_parameter_data)

            if callable(self.acc_model.phi_init):
                phi_init = self.acc_model.phi_init(acc_parameter_data, stress_data)
            else:
                phi_init = self.acc_model.phi_init


            init = np.array([*dist_init, *phi_init])
        else:
            init = np.array(init)

        if self.fixed != {}:
            fixed = {**self.fixed, **fixed}

        # Dynamic or static bounds determination
        if callable(self.acc_model.phi_bounds):
            bounds = (*self.bounds, *self.acc_model.phi_bounds(X))
        else:
            bounds = (*self.bounds, *self.acc_model.phi_bounds)

        if callable(self.acc_model.phi_param_map):
            phi_param_map = self.acc_model.phi_param_map(X)
        else:
            phi_param_map = self.acc_model.phi_param_map

        param_map = {**self.param_map, **{k : v + len(self.param_map) for k, v in phi_param_map.items()}}

        transform, inv_trans, funcs, inv_f = bounds_convert(x, bounds)
        const, fixed_idx, not_fixed = fix_idx_and_function(fixed, param_map, funcs)

        init = transform(init)[not_fixed]

        with np.errstate(all='ignore'):
            fun  = lambda params : self.neg_ll(X, x, c, n, *inv_trans(const(params)))
            # fun  = lambda params : self.neg_ll(X, x, c, n, *params)
            # jac = jacobian(fun)
            # hess = hessian(fun)
            res = minimize(fun, init)
            res = minimize(fun, res.x, method='TNC')
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
        model._neg_ll = res['fun']
        model.fixed = self.fixed
        model.k_dist = self.k_dist
        model.k = len(bounds)

        model.data = {
            'x' : x,
            'c' : c,
            'n' : n,
            't' : t
        }

        return model
