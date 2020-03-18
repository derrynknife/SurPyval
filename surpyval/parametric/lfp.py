import autograd.numpy as np
from scipy.stats import uniform
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z

from scipy.optimize import minimize

import surpyval
from surpyval import parametric as para
from surpyval.parametric.surpyval_dist import SurpyvalDist

TINIEST = np.finfo(np.float64).tiny

class LFP():
	def __init__(self, name, dist):
		self.name = name
		self.dist = dist

	def neg_ll(self, x, c, n, *params):
		params = np.array(params)
		w = params[0]
		params = params[1::]
		like = np.zeros_like(x)
		like = np.where(c ==  0, self.df(x, w, *params), like)
		like = np.where(c == -1, self.ff(x, w, *params), like)
		like = np.where(c ==  1, self.sf(x, w, *params), like)
		like += TINIEST
		like = np.where(like < 1, like, 1)
		like = np.log(like)
		like = np.multiply(n, like)
		return -like.sum()

	def parameter_initialiser(self, x, c=None, n=None):
		return self.dist.parameter_initialiser(x, c, n)

	def sf(self, x, w, *params):
		return 1 - self.ff(x, w, *params)

	def ff(self, x, w, *params):
		return w * self.dist.ff(x, *params)

	def df(self, x, w, *params):
		return w * self.dist.df(x, *params)

	def hf(self, x, w, *params):
		return self.df(x, w, *params) / self.sf(x, w, *params)

	def Hf(self, x, w, *params):
		return -np.log(self.sf(x, w, *params))

	def _mle(self, x, c, n):
		fun = lambda params : self.neg_ll(x, c, n, *params)
		bounds = tuple(((0, 1), *self.dist.bounds))
		init = (1., *self.parameter_initialiser(x, c, n))
		res = minimize(fun, init, bounds=bounds, tol=1e-10)
		self.res = res

	def fit(self, x, c=None, n=None):
		x, c, n = surpyval.xcn_handler(x, c, n)
		self._mle(x, c, n)