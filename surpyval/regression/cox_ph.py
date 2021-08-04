from autograd import jacobian, hessian
import autograd.numpy as np
import inspect
from autograd import elementwise_grad
from autograd.numpy.linalg import inv

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
        # like = np.where(x ==  0, 0, like)
        like = np.where(c ==  1, self.log_sf(x, X, *params), like)
        like = np.where(c ==  -1, self.log_ff(x, X, *params), like)
        like = np.multiply(n, like)
        return -np.sum(like)


    def neg_ll_breslow(self, X, x, c, n, *params):
        params = np.array(params)
        like = np.dot(X, params)
        theta = np.exp(like)
        theta = theta.sum() - theta.cumsum() + theta[0]
        like = n * (c == 0).astype(int) * (like - np.log(theta))
        return -like.sum()

    def neg_ll_breslow(self, X, x, c, n, *params):
        params = np.array(params)
        like = np.dot(X, params)
        theta = np.exp(like)
        theta = theta.sum() - theta.cumsum() + theta[0]
        like = n * (c == 0).astype(int) * (like - np.log(theta))
        return -like.sum()

    def fit(self, X, x, c=None, n=None, t=None, baseline='Fleming-Harrington', init=[]):
        x, c, n, t = surpyval.xcnt_handler(x, c, n, t, group_and_sort=False)

        if init == []:
            init = self.phi_init(X)
        else:
            init = np.array(init)

        if baseline == 'Breslow':
            fun  = lambda params : self.neg_ll_cox(X, x, c, n, *params)
        else:
            self.baseline = nonparametric_dists[baseline].fit(x, c, n, t)
            fun  = lambda params : self.neg_ll(X, x, c, n, *params)
        
        jac = jacobian(fun)
        hess = hessian(fun)

        with np.errstate(all='ignore'):
            res = minimize(fun, init, jac=jac)
            res = minimize(fun, res.x, jac=jac, method='TNC', tol=1e-20)

        params = res.x
        se = np.sqrt(np.diag(inv(hess(res.x))))

        return {'params' : params, 'exp(param)' : np.exp(params),'se' : se}
