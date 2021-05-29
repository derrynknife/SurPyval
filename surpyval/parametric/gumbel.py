import autograd.numpy as np
from scipy.stats import uniform
from numpy import euler_gamma
from scipy.special import ndtri as z

import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Gumbel_(ParametricFitter):
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

	def parameter_initialiser(self, x, c=None, n=None):
		return para.Gumbel.fit(x, c, n, how='MPP').params

	def sf(self, x, mu, sigma):
		return np.exp(-np.exp((x - mu)/sigma))

	def cs(self, x, X, mu, sigma):
		return self.sf(x + X, mu, sigma) / self.sf(X, mu, sigma)

	def ff(self, x, mu, sigma):
		return 1 - np.exp(-np.exp((x - mu)/sigma))

	def df(self, x, mu, sigma):
		z = (x - mu) / sigma
		return (1/sigma) * np.exp(z - np.exp(z))

	def hf(self, x, mu, sigma):
		z = (x - mu) / sigma
		return (1/sigma) * np.exp(z)

	def Hf(self, x, mu, sigma):
		return np.exp((x - mu)/sigma)

	def qf(self, p, mu, sigma):
		return mu + sigma * (np.log(-np.log(1 - p)))

	def mean(self, mu, sigma):
		return mu - sigma * euler_gamma

	def random(self, size, mu, sigma):
		U = uniform.rvs(size=size)
		return self.qf(U, mu, sigma)

	def mpp_x_transform(self, x, gamma=0):
		return x - gamma

	def mpp_y_transform(self, y, *params):
		mask = ((y == 0) | (y == 1))
		out = np.zeros_like(y)
		out[~mask] = np.log(-np.log((1 - y[~mask])))
		out[mask] = np.nan
		return out

	def mpp_inv_y_transform(self, y, *params):
		return 1 - np.exp(-np.exp(y))

	def unpack_rr(self, params, rr):
		if   rr == 'y':
			sigma = 1/params[0]
			mu    = -sigma * params[1]
		elif rr == 'x':
			sigma  = 1./params[0]
			mu = np.exp(params[1] / (beta * params[0]))
		return mu, sigma

	def var_z(self, x, mu, sigma, cv_matrix):
		z_hat = (x - mu)/sigma
		var_z = (1./sigma)**2 * (cv_matrix[0, 0] + z_hat**2 * cv_matrix[1, 1] + 
			2 * z_hat * cv_matrix[0, 1])
		return var_z

	def z_cb(self, x, mu, sigma, cv_matrix, cb=0.05):
		z_hat = (x - mu)/sigma
		var_z = self.var_z(x, mu, sigma, cv_matrix)
		bounds = z_hat + np.array([1., -1.]).reshape(2, 1) * z(cb/2) * np.sqrt(var_z)
		return bounds

	def R_cb(self, x, mu, sigma, cv_matrix, cb=0.05):
		return self.sf(self.z_cb(x, mu, sigma, cv_matrix, cb=0.05), 0, 1).T

Gumbel = Gumbel_('Gumbel')