from autograd import jacobian, hessian
import autograd.numpy as np

from scipy.optimize import minimize
import surpyval

class PH():
    def __init__(self, dist, func):
        assert callable(func) == True
        self.dist = dist
        self.func = func

    def log_like_f(self, phi, x, *params):
        l_phi = np.log(phi)
        outf = self.dist.df(x, *params)
        outf = np.where(outf <= 0, surpyval.TINIEST, outf)
        outf = np.where(outf <  1, outf, 1)

        outr = self.dist.sf(x, *params)
        outr = np.where(outr <= 0, surpyval.TINIEST, outr)
        outr = np.where(outr < 1,  outr, 1)

        outr = np.multiply(np.log(outr), phi - 1)
        return np.log(outf) + l_phi + outr

    def log_like_r(self, phi, x, *params):
        out = self.dist.sf(x, *params)
        out = np.where(out <= 0, surpyval.TINIEST, out)
        out = np.where(out <  1, out, 1)

        return np.multiply(phi, np.log(out))

    def log_like_l(self, phi, x, *params):
        out = self.dist.sf(x, *params)
        out = 1 - np.power(out, phi)
        out = np.where(out <= 0, surpyval.TINIEST, out)
        out = np.where(out <  1, out, 1)

        return np.log(out)    

    def neg_ll(self, X, x, c=None, n=None, *params):
        dist_params = params[0:self.dist.k]
        phm_params  = params[self.dist.k::]
        phi_func = lambda p : self.func(X, p)
        phi  = phi_func(phm_params)

        like = np.zeros_like(x).astype(surpyval.NUM)
        like = np.where(c ==  0, self.log_like_f(phi, x, *dist_params), like)
        like = np.where(c ==  1, self.log_like_r(phi, x, *dist_params), like)
        like = np.where(c == -1, self.log_like_l(phi, x, *dist_params), like)

        like = np.multiply(n, like)
        return -np.sum(like)

    def fit(self, X, x, c=None, n=None, guess=(1., 1.)):
        #x, c, n = surv.xcn_handler(x, c, n)
        c = np.zeros_like(x)
        n = np.ones_like(x)
        fun  = lambda t : self.neg_ll(X, x, c, n, *t)
        jac = jacobian(fun)
        hess = hessian(fun)
        ps = self.dist.parameter_initialiser(x, c=c, n=n)
        
        init = [*ps, *guess]
        res = minimize(fun, init, jac=jac, hess=hess, tol=1e-10)
        self.res = res