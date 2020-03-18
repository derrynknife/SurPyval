import autograd.numpy as np
from autograd import jacobian, hessian
from scipy.stats import uniform
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z

from scipy.optimize import minimize

import surpyval
from surpyval import parametric as para
from surpyval.parametric.surpyval_dist import SurpyvalDist

TINIEST = np.finfo(np.float64).tiny

class LFP(SurpyvalDist):
	def __init__(self, dist):
		self.dist = dist
		self.bounds = tuple(((0, 1), *self.dist.bounds))
		self.use_autograd = True

	def parameter_initialiser(self, x, c=None, n=None):
		return tuple((1, *self.dist.parameter_initialiser(x, c, n)))

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