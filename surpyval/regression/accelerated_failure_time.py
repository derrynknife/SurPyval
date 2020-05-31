from autograd import jacobian, hessian
import autograd.numpy as np

from scipy.optimize import minimize
import surpyval

class AFT():
	def __init__(self, dist, func):
		assert callable(func) == True
		self.dist = dist
		self.func = func

	def like(self, x, c=None, n=None, *params):
		like = np.zeros_like(x).astype(surpyval.NUM)
		like = np.where(c ==  0, self.dist.df(x, *params), like)
		like = np.where(c == -1, self.dist.ff(x, *params), like)
		like = np.where(c ==  1, self.dist.sf(x, *params), like)
		return like

	def neg_ll(self, X, x, c=None, n=None, *params):
		dist_params = params[0:self.dist.k]
		aftm_params = params[self.dist.k::]
		new_x = self.func(X, aftm_params)

		like = self.like(new_x, c, n, *dist_params)
		like = np.where(like <= 0, surpyval.TINIEST, like)
		like = np.where(like < 1, like, 1)
		like = np.log(like)
		like = np.multiply(n, like)
		return -np.sum(like)

	def fit(self, X, x, c=None, n=None, guess=(1., 1.)):
		c = np.zeros_like(x)
		n = np.ones_like(x)

		fun  = lambda t : self.neg_ll(X, x, c, n, *t)
		jac = jacobian(fun)
		hess = hessian(fun)
		ps = self.dist.parameter_initialiser(x, c=c, n=n)

		init = [*ps, *guess]
		res = minimize(fun, init, jac=jac, hess=hess, tol=1e-10)
		self.res = res



