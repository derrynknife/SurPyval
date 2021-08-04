import autograd.numpy as np
from autograd import jacobian, hessian, elementwise_grad
from scipy.optimize import minimize
import surpyval
import inspect
import warnings

from ..parametric.fitters import bounds_convert, fix_idx_and_function
from .regression import Regression

class ParameterSubstitutionFitter():
    def __init__(self, kind, name, distribution, life_model, life_parameter, baseline=[],
                 param_transform=None, inverse_param_transform=None):

        if str(inspect.signature(life_model.phi)) != '(X, *params)':
            raise ValueError('PH function must have the signature \'(X, *params)\'')

        if type(baseline) != list:
            # Baseline used if using a function that deviates from some number, e.g. np.exp(np.dot(X, beta))
            baseline = [baseline]

        self.name = name
        self.kind = kind
        self.dist = distribution
        self.life_model = life_model
        self.k_dist = len(self.dist.param_names)
        self.bounds = self.dist.bounds
        self.support = self.dist.support
        self.param_names = self.dist.param_names
        self.param_map = {v : i for i, v in enumerate(self.dist.param_names)}
        self.phi = life_model.phi
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df
        self.baseline = baseline
        self.life_parameter = life_parameter
        self.fixed = {life_parameter : 1.}

        if param_transform is None:
            self.param_transform = lambda x: x
            self.inverse_param_transform = lambda x: x
        else:
            self.param_transform = param_transform
            self.inverse_param_transform = inverse_param_transform

    def Hf(self, x, X, *params):
        x = np.array(x)
        if np.isscalar(X):
            X = np.ones_like(x) * X
        else:
            X = np.array(X)

        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])

        Hf = np.zeros_like(x)
        stresses = np.unique(X, axis=0)
        for stress in stresses:
            life_param_mask = np.array(range(0, len(dist_params))) == self.param_map[self.life_parameter]
            dist_params_i = np.where(life_param_mask,
                self.param_transform(self.phi(stress, *phi_params)),
                dist_params)
            mask = (X == stress).all(axis=1)
            Hf = np.where(mask, self.Hf_dist(x, *dist_params_i), Hf)

        return Hf

    def hf(self, x, X, *params):
        x = np.array(x)
        if np.isscalar(X):
            X = np.ones_like(x) * X
        else:
            X = np.array(X)

        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])

        hf = np.zeros_like(x)
        for stress in np.unique(X, axis=0):
            life_param_mask = np.array(range(0, len(dist_params))) == self.param_map[self.life_parameter]
            params = np.where(life_param_mask,
                self.param_transform(self.phi(stress, *phi_params)),
                dist_params)
            mask = (X == stress).all(axis=1)
            hf = np.where(mask, self.hf_dist(x, *params), hf)

        return hf

    def df(self, x, X, *params):
        x = np.array(x)
        if np.isscalar(X):
            X = np.ones_like(x) * X
        else:
            X = np.array(X)
        return self.hf(x, X, *params) * np.exp(-self.Hf(x, X, *params))

    def sf(self, x, X, *params):
        x = np.array(x)
        if np.isscalar(X):
            X = np.ones_like(x) * X
        else:
            X = np.array(X)
        return np.exp(-self.Hf(x, X, *params))

    def ff(self, x, X, *params):
        x = np.array(x)
        if np.isscalar(X):
            X = np.ones_like(x) * X
        else:
            X = np.array(X)
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

    def random(self, size, X, *params):
        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])

        x = []
        X_out = []
        if type(X) == tuple:
            X = np.random.uniform(*X, size)

        for stress in np.unique(X, axis=0):
            life_param_mask = np.array(range(0, len(dist_params))) == self.param_map[self.life_parameter]
            dist_params = np.where(life_param_mask,
                                   self.param_transform(self.phi(stress, *phi_params)),
                                   dist_params)

            U = np.random.uniform(0, 1, size)
            x.append(self.dist.qf(U, *dist_params))
            if np.isscalar(stress):
                cols = 1
            else:
                cols = len(stress)
            X_out.append((np.ones((size, cols)) * stress))
        return np.array(x).flatten(), np.concatenate(X_out)

    def neg_ll(self, X, x, c, n, *params, verbose=False):
        like = np.zeros_like(x).astype(float)
        like = np.where(c ==  0, self.log_df(x, X, *params), like)
        like = np.where(c ==  1, self.log_sf(x, X, *params), like)
        like = np.where(c ==  -1, self.log_ff(x, X, *params), like)
        like = np.multiply(n, like)
        like = -np.sum(like)
        if verbose:
            print('   ' * 100, end='\r')
            print(like, ": ", params, end='\r')
        return like

    def fit(self, X, x, c=None, n=None, t=None, init=[], fixed={}, verbose=False):
        x, c, n, t = surpyval.xcnt_handler(x=x, c=c, n=n, t=t, group_and_sort=False)

        if init == []:
            stress_data = []
            params_at_X = []

            # How do I make this work when there is only one failure per stress?
            for s in np.unique(X, axis=0):
                mask = (X == s).all(axis=1)
                with warnings.catch_warnings():
                    warnings.filterwarnings('error')
                    try:
                        params_at_X.append(self.dist.fit(x[mask], c[mask], n[mask]).params)
                        stress_data.append(s)
                    except np.RankWarning:
                        pass
                    finally:
                        pass

            params_at_X = np.array(params_at_X)
            stress_data = np.array(stress_data)
            dist_init = params_at_X.mean(axis=0)

            i = self.param_map[self.life_parameter]

            if len(params_at_X) < 2:
                raise ValueError("Insufficient data at separate X values. Try manually setting initial guess using `init` keyword in `fit`")

            parameter_data = params_at_X[:, i]

            parameter_data = self.inverse_param_transform(parameter_data)

            if callable(self.life_model.phi_init):
                if str(inspect.signature(self.life_model.phi_init)) == '(X)':
                    phi_init = self.life_model.phi_init(X)
                else:
                    phi_init = self.life_model.phi_init(parameter_data, stress_data)
            else:
                phi_init = self.life_model.phi_init

            init = np.array([*dist_init, *phi_init])
        else:
            init = np.array(init)

        if self.baseline != []:
            baseline_model = self.dist.fit(x, c, n, t)
            baseline_fixed = {k : baseline_model.params[baseline_model.param_map[k]] for k in self.baseline}
            fixed = {**baseline_fixed, **fixed}

        if self.fixed != {}:
            fixed = {**self.fixed, **fixed}

        # Dynamic or static bounds determination
        if callable(self.life_model.phi_bounds):
            bounds = (*self.bounds, *self.life_model.phi_bounds(X))
        else:
            bounds = (*self.bounds, *self.life_model.phi_bounds)

        if callable(self.life_model.phi_param_map):
            phi_param_map = self.life_model.phi_param_map(X)
        else:
            phi_param_map = self.life_model.phi_param_map

        param_map = {**self.param_map, **{k : v + len(self.param_map) for k, v in phi_param_map.items()}}
        self.param_map = param_map

        transform, inv_trans, funcs, inv_f = bounds_convert(x, bounds)
        const, fixed_idx, not_fixed = fix_idx_and_function(fixed, param_map, funcs)

        init = transform(init)[not_fixed]

        with np.errstate(all='ignore'):
            fun  = lambda params, verbose : self.neg_ll(X, x, c, n, *inv_trans(const(params)), verbose=verbose)
            # jac = jacobian(fun)
            # hess = hessian(fun)
            res = minimize(fun, init, args=(verbose))
            res = minimize(fun, res.x, args=(verbose), method='TNC')
            # res = minimize(fun, init, jac=jac, method='BFGS')
            # res = minimize(fun, init, method='Newton-CG', jac=jac)

        params = inv_trans(const(res.x))
        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])

        model = Regression()
        model.model = self
        model.kind = self.kind
        model.distribution = self.dist
        model.reg_model = self.life_model
        model.params = np.array(params)
        model.dist_params = dist_params
        model.phi_params = phi_params
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
