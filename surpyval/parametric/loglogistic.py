import autograd.numpy as np
from scipy.stats import uniform
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z

import surpyval
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class LogLogistic_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((0, None), (0, None),)
		self.support = (0, np.inf)
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['alpha', 'beta']
		self.param_map = {
			'alpha' : 0,
			'beta' : 1
		}

	def parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
		if offset:
			return *self.fit(x, c, n, t, how='MPP').params, 1.
			# x, c, n = surpyval.xcn_handler(x, c, n)
			# flag = (c == 0).astype(int)
			# value_range = np.max(x) - np.min(x)
			# gamma_init = np.min(x) - value_range / 10
			# return gamma_init, x.sum() / (n * flag).sum(), 2.
		else:
			return self.fit(x, c, n, t, how='MPP').params
			# x, c, n = surpyval.xcn_handler(x, c, n)
			# flag = (c == 0).astype(int)
			# return x.sum() / (n * flag).sum(), 2.

	def sf(self, x, alpha, beta):
		return 1 - self.ff(x, alpha, beta)

	def cs(self, x, X, alpha, beta):
		return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

	def ff(self, x, alpha, beta):
		return 1. / (1 + (x/alpha)**-beta)

	def df(self, x, alpha, beta):
		return ((beta/alpha)*(x/alpha)**(beta-1.))/((1. + (x/alpha)**beta)**2.)

	def hf(self, x, alpha, beta):
		return self.pdf(x, alpha, beta) / self.sf(x, alpha, beta)

	def Hf(self, x, alpha, beta):
		return -np.log(self.sf(x, alpha, beta))

	def qf(self, p, alpha, beta):
		return alpha * (p/(1 - p))**(1./beta)

	def mean(self, alpha, beta):
		if beta > 1:
			return (alpha * np.pi / beta) / (np.sin(np.pi/beta))
		else:
			return np.nan

	def random(self, size, alpha, beta):
		U = uniform.rvs(size=size)
		return self.qf(U, alpha, beta)

	def mpp_x_transform(self, x, gamma=0):
		return np.log(x - gamma)

	def mpp_y_transform(self, y, *params):
		mask = ((y == 0) | (y == 1))
		out = np.zeros_like(y)
		out[~mask] = -np.log(1./y[~mask] - 1)
		out[mask] = np.nan
		return out

	def mpp_inv_y_transform(self, y, *params):
		return 1./(np.exp(-y) + 1)

	def unpack_rr(self, params, rr):
		if rr == 'y':
			beta  = params[0]
			alpha = np.exp(params[1]/-beta)
		elif rr == 'x':
			beta  = 1./params[0]
			alpha = np.exp(params[1] / (beta * params[0]))
		return alpha, beta

LogLogistic = LogLogistic_('LogLogistic')