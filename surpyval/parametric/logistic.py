import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z

import surpyval
from surpyval.parametric.parametric_fitter import ParametricFitter

class Logistic_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((None, None), (0, None),)
		self.support = (-np.inf, np.inf)
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['mu', 'sigma']
		self.param_map = {
			'mu'    : 0,
			'sigma' : 1
		}

	def parameter_initialiser(self, x, c=None, n=None, t=None, offset=False):
		return self.fit(x, c, n, t, how='MPP').params
		x, c, n = surpyval.xcn_handler(x, c, n)
		flag = (c == 0).astype(int)
		if offset:
			return x.sum() / (n * flag).sum(), 1., 1.
		else:
			return x.sum() / (n * flag).sum(), 1.

	def sf(self, x, mu, sigma):
		return 1 - self.ff(x, mu, sigma)

	def cs(self, x, X, mu, sigma):
		return self.sf(x + X, mu, sigma) / self.sf(X, mu, sigma)

	def ff(self, x, mu, sigma):
		z = (x - mu) / sigma
		return 1. / (1 + np.exp(-z))

	def df(self, x, mu, sigma):
		z = (x - mu) / sigma
		return np.exp(-z) / (sigma * (1 + np.exp(-z))**2)

	def hf(self, x, mu, sigma):
		return self.pdf(x, mu, sigma) / self.sf(x, mu, sigma)

	def Hf(self, x, mu, sigma):
		return -np.log(self.sf(x, mu, sigma))

	def qf(self, p, mu, sigma):
		return mu + sigma * np.log(p/(1 - p))

	def mean(self, mu, sigma):
		return mu

	def random(self, size, mu, sigma):
		U = uniform.rvs(size=size)
		return self.qf(U, mu, sigma)

	def mpp_x_transform(self, x, gamma=0):
		return x - gamma

	def mpp_y_transform(self, y, *params):
		mask = ((y == 0) | (y == 1))
		out = np.zeros_like(y)
		out[~mask] = -np.log(1./y[~mask] - 1)
		out[mask] = np.nan
		return out

	def mpp_inv_y_transform(self, y, *params):
		return 1./(np.exp(-y) + 1)

	def unpack_rr(self, params, rr):
		if   rr == 'y':
			sigma = 1/params[0]
			mu    = -sigma * params[1]
		elif rr == 'x':
			sigma  = 1./params[0]
			mu = np.exp(params[1] / (beta * params[0]))
		return mu, sigma

Logistic = Logistic_('Logistic')