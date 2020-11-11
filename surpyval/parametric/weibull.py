import autograd.numpy as np
from scipy.stats import uniform
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z

from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Weibull_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		# Set 'k', the number of parameters
		self.k = 2
		self.bounds = ((0, None), (0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['alpha', 'beta']

	def parameter_initialiser(self, x, c=None, n=None):
		log_x = np.log(x)
		log_x[np.isnan(log_x)] = -np.inf
		gumb = para.Gumbel.fit(log_x, c, n, how='MLE')
		if not gumb.res.success:
			gumb = para.Gumbel.fit(log_x, c, n, how='MPP')
		mu, sigma = gumb.params
		alpha, beta = np.exp(mu), 1. / sigma
		if (np.isinf(alpha) | np.isnan(alpha)):
			alpha = np.median(x)
		if (np.isinf(beta) | np.isnan(beta)):
			beta = 1.
		return alpha, beta

	def sf(self, x, alpha, beta):
		return np.exp(-(x / alpha)**beta)

	def ff(self, x, alpha, beta):
		return 1 - np.exp(-(x / alpha)**beta)

	def cs(self, x, X, alpha, beta):
		return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

	def df(self, x, alpha, beta):
		return (beta / alpha) * (x / alpha)**(beta-1) * np.exp(-(x / alpha)**beta)

	def hf(self, x, alpha, beta):
		return (beta / alpha) * (x / alpha)**(beta - 1)

	def Hf(self, x, alpha, beta):
		return (x / alpha)**beta

	def qf(self, p, alpha, beta):
		return alpha * (-np.log(1 - p))**(1/beta)

	def mean(self, alpha, beta):
		return alpha * gamma_func(1 + 1./beta)

	def moment(self, n, alpha, beta):
		return alpha**n * gamma_func(1 + n/beta)

	def entropy(self, alhpa, beta):
		return euler_gamma * (1 - 1/beta) + np.log(alpha / beta) + 1

	def random(self, size, alpha, beta):
		U = uniform.rvs(size=size)
		return self.qf(U, alpha, beta)

	def mpp_x_transform(self, x):
		return np.log(x)

	def mpp_y_transform(self, y):
		return np.log(-np.log((1 - y)))

	def mpp_inv_y_transform(self, y):
		return 1 - np.exp(-np.exp(y))

	def unpack_rr(self, params, rr):
		if rr == 'y':
			beta  = params[0]
			alpha = np.exp(params[1]/-beta)
		elif rr == 'x':
			beta  = 1./params[0]
			alpha = np.exp(params[1] / (beta * params[0]))
		return alpha, beta

	def u(self, x, alpha, beta):
		return beta * (np.log(x) - np.log(alpha))

	def u_cb(self, x, alpha, beta, cv_matrix, cb=0.05):
		u = self.u(x, alpha, beta)
		var_u = self.var_u(x, alpha, beta, cv_matrix)
		diff = z(cb/2) * np.sqrt(var_u)
		bounds = u + np.array([1., -1.]).reshape(2, 1) * diff
		return bounds

	def du(self, x, alpha, beta):
		du_dbeta = np.log(x) - np.log(alpha)
		du_dalpha  = -beta/alpha
		return du_dalpha, du_dbeta

	def var_u(self, x, alpha, beta, cv_matrix):
		da, db = self.du(x, alpha, beta)
		var_u = (da**2 * cv_matrix[0, 0] + db**2 * cv_matrix[1, 1] + 
			2 * da * db * cv_matrix[0, 1])
		return var_u

	def R_cb(self, x, alpha, beta, cv_matrix, cb=0.05):
		return np.exp(-np.exp(self.u_cb(x, alpha, beta, cv_matrix, cb))).T

	def jacobian(self, x, alpha, beta, c=None, n=None):
		"""
		The jacobian for a two parameter Weibull distribution.

		Please report mistakes if found!
		"""
		f = c == 0
		l = c == -1
		r = c == 1
		dll_dbeta = (
			1./beta * np.sum(n[f]) +
			np.sum(n[f] * np.log(x[f]/alpha)) - 
			np.sum(n[f] * (x[f]/alpha)**beta * np.log(x[f]/alpha)) - 
			np.sum(n[r] * (x[r]/alpha)**beta * np.log(x[r]/alpha)) +
			np.sum(n[l] * (x[l]/alpha)**beta * np.log(x[l]/alpha) *
				np.exp(-(x[l]/alpha)**beta) / 
				(1 - np.exp(-(x[l]/alpha)**beta)))
		)

		dll_dalpha = ( 0 -
			beta/alpha * np.sum(n[f]) +
			beta/alpha * np.sum(n[f] * (x[f]/alpha)**beta) +
			beta/alpha * np.sum(n[r] * (x[r]/alpha)**beta) -
			beta/alpha * np.sum(n[l] * (x[l]/alpha)**beta * 
				np.exp(-(x[l]/alpha)**beta) /
				(1 - np.exp(-(x[l]/alpha)**beta)))
		)
		return -np.array([dll_dalpha, dll_dbeta])

Weibull = Weibull_('Weibull')