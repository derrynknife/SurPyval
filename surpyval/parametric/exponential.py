import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z

import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter
from scipy.special import factorial
from .fitters.mpp import mpp

import warnings

class Exponential_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		self.k = 1
		self.bounds = ((0, None),)
		self.support = (0, np.inf)
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.05, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['lambda']
		self.param_map = {
			'lambda' : 0,
		}

	def parameter_initialiser(self, x, c=None, n=None, offset=False):
		x, c, n = surpyval.xcn_handler(x, c, n)
		c = (c == 0).astype(np.int64)
		rate = (n * c).sum()/x.sum()
		if offset:
			return np.min(x) - (np.max(x) - np.min(x))/10., rate
		else:
			return np.array([rate])

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

	def mpp_x_transform(self, x, gamma=0):
		return x - gamma

	def mpp_y_transform(self, y, *params):
		mask = ((y == 0) | (y == 1))
		out = np.zeros_like(y)
		out[~mask] = -np.log(1 - y[~mask])
		out[mask] = np.nan
		return out

	def mpp_inv_y_transform(self, y, *params):
		return 1 - np.exp(y)

	def mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y', on_d_is_0=False, offset=False):
		assert rr in ['x', 'y']
		x_pp, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		if not on_d_is_0:
			x_pp = x_pp[d > 0]
			F    = F[d > 0]
		
		# Linearise
		y_pp = self.mpp_y_transform(F)

		mask = np.isfinite(y_pp)
		if mask.any():
			warnings.warn("Some Infinite values encountered in plotting points and have been ignored.", stacklevel=2)
			y_pp = y_pp[mask]
			x_pp = x_pp[mask]

		if offset:
			if   rr == 'y':
				params = np.polyfit(x_pp, y_pp, 1)
			elif rr == 'x':
				params = np.polyfit(y_pp, x_pp, 1)
			failure_rate = params[0]
			offset = -params[1] * (1./failure_rate)
			return tuple([offset, failure_rate])
		else:
			if   rr == 'y':
				x_pp = x_pp[:,np.newaxis]
				failure_rate = np.linalg.lstsq(x_pp, y_pp, rcond=None)[0]
			elif rr == 'x':
				y_pp = y_pp[:,np.newaxis]
				mttf = np.linalg.lstsq(y_pp, x_pp, rcond=None)[0]
				failure_rate = 1. / mttf
			return tuple([failure_rate[0]])

	def lambda_cb(self, x, failure_rate, cv_matrix, cb=0.05):
		return failure_rate * np.exp(np.array([-1, 1]).reshape(2, 1) * (z(cb/2) * 
									np.sqrt(cv_matrix.item()) / failure_rate))

	def R_cb(self, x, failure_rate, cv_matrix, cb=0.05):
		return np.exp(-self.lambda_cb(x, failure_rate, cv_matrix, cb=0.05) * x).T

Exponential = Exponential_('Exponential')