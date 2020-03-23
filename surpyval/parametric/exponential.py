import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z

import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Exponential_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		self.k = 1
		self.bounds = ((0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.05, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['lambda']

	def parameter_initialiser(self, x, c=None, n=None):
		x, c, n = surpyval.xcn_handler(x, c, n)
		c = (c == 0).astype(np.int64)
		return [(n * c).sum()/x.sum()]

	def sf(self, x, failure_rate):
		return np.exp(-failure_rate * x)

	def cs(self, x, X, failure_rate):
		# The exponential distribution is memoryless so of course it is the same as the survival function
		return self.sf(x, failure_rate)

	def ff(self, x, failure_rate):
		return 1 - np.exp(-failure_rate * x)

	def df(self, x, failure_rate):
		return failure_rate * np.exp(-failure_rate * x)

	def hf(self, x, failure_rate):
		return failure_rate

	def Hf(self, x, failure_rate):
		return failure_rate * x

	def qf(self, p, failure_rate):
		return -np.log(p)/failure_rate

	def mean(self, failure_rate):
		return 1. / failure_rate

	def moment(self, n, failure_rate):
		return factorial(n) / (failure_rate ** n)

	def entropy(self, failure_rate):
		return 1 - np.log(failure_rate)

	def random(self, size, failure_rate):
		U = uniform.rvs(size=size)
		return self.qf(U, failure_rate)

	def mpp_x_transform(self, x):
		return x

	def mpp_y_transform(self, y):
		return -np.log(1 - y)

	def mpp_inv_y_transform(self, y):
		return 1 - np.exp(y)

	def _mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y', on_d_is_0=False):
		assert rr in ['x', 'y']
		x_, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		if not on_d_is_0:
			x_ = x_[d > 0]
			F = F[d > 0]
		
		# Linearise
		y_ = self.mpp_y_transform(F)

		if   rr == 'y':
			x_ = x_[:,np.newaxis]
			failure_rate = np.linalg.lstsq(x_, y_, rcond=None)[0]
		elif rr == 'x':
			y_ = y_[:,np.newaxis]
			mttf = np.linalg.lstsq(y_, x_, rcond=None)[0]
			failure_rate = 1. / mttf
		return tuple([failure_rate[0]])

	def lambda_cb(self, x, failure_rate, cv_matrix, cb=0.05):
		return failure_rate * np.exp(np.array([-1, 1]).reshape(2, 1) * (z(cb/2) * 
									np.sqrt(cv_matrix.item()) / failure_rate))

	def R_cb(self, x, failure_rate, cv_matrix, cb=0.05):
		return np.exp(-self.lambda_cb(x, failure_rate, cv_matrix, cb=0.05) * x).T

	def jacobian(self, x, failure_rate, c=None, n=None):
		"""
		The jacobian for a two parameter Weibull distribution.
		Not used, but will need for cb
		"""
		if c is None:
			c = np.zeros_like(x)

		if n is None:
			n = np.ones_like(x)
		
		f = c == 0
		l = c == -1
		r = c == 1

		dll_dlambda = (
			np.sum(n[f] * (1./failure_rate - x[f])) -
			np.sum(n[r] * x[r]) -
			np.sum(n[l] * ((-x[l] * np.exp(-failure_rate * x[l]))/(1 - np.exp(-failure_rate*x[l]))))
		)

		return -np.array([dll_dlambda])

Exponential = Exponential_('Exponential')