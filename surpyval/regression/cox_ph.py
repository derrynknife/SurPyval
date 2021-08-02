from autograd import jacobian, hessian
import autograd.numpy as np
import inspect
from autograd import elementwise_grad

from scipy.optimize import minimize
import surpyval

from ..parametric.fitters import bounds_convert, fix_idx_and_function

from ..nonparametric import NelsonAalen
from ..nonparametric import KaplanMeier
from ..nonparametric import FlemingHarrington
from ..nonparametric import Turnbull

nonparametric_dists = {
    'Nelson-Aalen' : NelsonAalen,
    'Kaplan-Meier' : KaplanMeier,
    'Fleming-Harrington' : FlemingHarrington,
    'Turnbull' : Turnbull
}

class CoxProportionalHazardsFitter():
    def __init__(self):
        self.name = 'CoxPH'
        self.phi = lambda X, *params: np.exp(np.dot(X, np.array(params)))
        self.phi_init = phi_init=lambda X: np.zeros(X.shape[1])
        self.phi_bounds = lambda X: (((None, None),) * X.shape[1])
        self.phi_param_map = lambda X: {'beta_' + str(i) : i for i in range(X.shape[1])}

    def Hf(self, x, X, *params):
        return self.phi(X, *params) * self.baseline.Hf(x)

    def hf(self, x, X, *params):
        return self.phi(X, *params) * self.baseline.hf(x)

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
        like = np.where(x ==  0, 0, like)
        like = np.where(c ==  1, self.log_sf(x, X, *params), like)
        like = np.where(c ==  -1, self.log_ff(x, X, *params), like)
        like = np.multiply(n, like)
        return -np.sum(like)

    def fit(self, X, x, c=None, n=None, t=None, init=[], fixed={}, baseline='Fleming-Harrington'):
        # x, c, n, t = surpyval.xcnt_handler(x, c, n, t)
        if c is None:
            c = np.zeros_like(x)
        if n is None:
            n = np.ones_like(x)

        if init == []:
            init = self.phi_init(X)
        else:
            init = np.array(init)


        self.baseline = nonparametric_dists[baseline].fit(x, c, n, t)

        # Dynamic or static bounds determination
        bounds = self.phi_bounds(X)
        param_map = self.phi_param_map(X)

        transform, inv_trans, funcs, inv_f = bounds_convert(x, bounds)
        const, fixed_idx, not_fixed = fix_idx_and_function(fixed, param_map, funcs)


        init = transform(init)[not_fixed]

        fun  = lambda params : self.neg_ll(X, x, c, n, *inv_trans(const(params)))
        jac = jacobian(fun)
        hess = hessian(fun)

        with np.errstate(all='ignore'):
            res = minimize(fun, init, jac=jac)

        params = inv_trans(const(res.x))

        # res = minimize(fun, init, method='BFGS', jac=jac)
        return params
