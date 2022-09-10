from autograd import jacobian, hessian
import autograd.numpy as np
import inspect
from autograd import elementwise_grad

from scipy.optimize import minimize
import surpyval

from ..parametric.fitters import bounds_convert, fix_idx_and_function
from .regression import Regression

from ..utils import _get_idx

class ProportionalHazardsModel():
    """
    A Proportional Hazard Model class.
    Currently only implemented for semi-parametric
    would need to change should we want to allow
    fully parametric models.
    # """
    def __init__(self, kind, parameterization):
        self.kind = kind
        self.parameterization = parameterization

    def __repr__(self):
        out = ('Regression SurPyval Model'
            + '\n========================='
            + '\nType                : Proportional Hazards'
            + '\nKind                : {kind}'
            + '\nParameterization    : {parameterization}'
            ).format(kind=self.kind,
                    parameterization=self.parameterization)
            
        out = (out + '\nParameters          :\n')
        for i, p in enumerate(self.parameters):
            out += '   beta_{i}  :  {p}\n'.format(i=i, p=p)
        return out

    def hf(self, x, Z):
        idx, rev = _get_idx(self.x, x)
        return (self.h0[idx] * self.phi(Z))[rev]

    def Hf(self, x, Z):
        idx, rev = _get_idx(self.x, x)
        return (self.H0[idx] * self.phi(Z))[rev]

    def sf(self, x, Z):
        return np.exp(-self.Hf(x, Z))

    def ff(self, x, Z):
        return 1 - self.sf(x, Z)

    def df(self, x, Z):
        return self.hf(x, Z) * self.sf(x, Z)


class ProportionalHazardsFitter():
    def __init__(self, name, dist, phi, phi_bounds, phi_param_map,
                 baseline=[], fixed={}, phi_init=None):

        if str(inspect.signature(phi)) != '(X, *params)':
            raise ValueError('PH function must have the signature \'(X, *params)\'')

        if type(baseline) != list:
            # If passed a single string..
            baseline = [baseline]

        self.name = name
        self.dist = dist
        self.k_dist = len(self.dist.param_names)
        self.bounds = self.dist.bounds
        self.support = self.dist.support
        self.param_names = self.dist.param_names
        self.param_map = {v : i for i, v in enumerate(self.dist.param_names)}
        self.phi = phi
        self.Hf_dist = self.dist.Hf
        self.hf_dist = self.dist.hf
        self.sf_dist = self.dist.sf
        self.ff_dist = self.dist.ff
        self.df_dist = self.dist.df
        self.baseline = baseline
        self.fixed = fixed
        self.phi_init = phi_init
        self.phi_bounds = phi_bounds
        self.phi_param_map = phi_param_map

    def Hf(self, x, X, *params):
        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])
        Hf_raw = self.Hf_dist(x, *dist_params)
        return self.phi(X, *phi_params) * Hf_raw

    def hf(self, x, X, *params):
        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])
        hf_raw = self.hf_dist(x, *dist_params)
        return self.phi(X, *phi_params) * hf_raw

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

    def random(self, size, X, *params):
        dist_params = np.array(params[0:self.k_dist])
        phi_params = np.array(params[self.k_dist:])
        random = []
        U = np.random.uniform(0, 1, size)
        x = self.dist.qf(U**(self.phi(X, *phi_params)), *dist_params)
        X_out = np.ones_like(x) * X
        return x.flatten(), X_out.flatten()

    def neg_ll(self, X, x, c, n, *params):
        params = np.array(params)
        like = np.zeros_like(x).astype(float)
        like = np.where(c ==  0, self.log_df(x, X, *params), like)
        like = np.where(c ==  1, self.log_sf(x, X, *params), like)
        like = np.where(c ==  -1, self.log_ff(x, X, *params), like)
        like = np.multiply(n, like)
        return -np.sum(like)

    def fit(self, X, x, c=None, n=None, t=None, init=[], fixed={}):
        x, c, n, t = surpyval.xcnt_handler(x, c, n, t, group_and_sort=False)

        if init == []:
            ps = self.dist.fit(x, c=c, n=n, t=t).params
            if callable(self.phi_init):
                init_phi = self.phi_init(X)

            init = np.array([*ps, *init_phi])
        else:
            init = np.array(init)

        if self.baseline != []:
            baseline_model = self.dist.fit(x, c, n, t)
            baseline_fixed = {k : baseline_model.params[baseline_model.param_map[k]] for k in self.baseline}
            fixed = {**baseline_fixed, **fixed}

        # Dynamic or static bounds determination
        if callable(self.phi_bounds):
            bounds = (*self.bounds, *self.phi_bounds(X))
        else:
            bounds = (*self.bounds, *self.phi_bounds)

        if callable(self.phi_param_map):
            phi_param_map = self.phi_param_map(X)
        else:
            phi_param_map = self.phi_param_map

        param_map = {**self.param_map, **phi_param_map}

        transform, inv_trans, funcs, inv_f = bounds_convert(x, bounds)
        const, fixed_idx, not_fixed = fix_idx_and_function(fixed, param_map, funcs)

        init = transform(init)[not_fixed]

        with np.errstate(all='ignore'):
            fun  = lambda params : self.neg_ll(X, x, c, n, *inv_trans(const(params)))
            # jac = jacobian(fun)
            # hess = hessian(fun)
            res = minimize(fun, init)
            res = minimize(fun, res.x, method='TNC')

        params = inv_trans(const(res.x))

        model = Regression()
        model.model = self
        model.reg_model = self.phi
        model.kind = "Proportional Hazard"
        model.distribution = self.dist
        model.params = np.array(params)
        model.res = res
        model._neg_ll = res['fun']
        model.fixed = self.fixed
        model.k_dist = self.k_dist
        model.phi_param_map = phi_param_map

        print(res)

        return model
