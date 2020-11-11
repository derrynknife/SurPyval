import autograd.numpy as np
from scipy.stats import uniform
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z
from scipy import integrate
from scipy.optimize import minimize

from surpyval import parametric as para
from surpyval import nonparametric as nonp
from surpyval.parametric.parametric_fitter import ParametricFitter

class ExpoWeibull_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		# Set 'k', the number of parameters
		self.k = 3
		self.bounds = ((0, None), (0, None), (0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['alpha', 'beta', 'mu']

	def parameter_initialiser(self, x, c=None, n=None):
		log_x = np.log(x)
		log_x[np.isnan(log_x)] = 0
		gumb = para.Gumbel.fit(log_x, c, n, how='MLE')
		if not gumb.res.success:
			gumb = para.Gumbel.fit(log_x, c, n, how='MPP')
		mu, sigma = gumb.params
		alpha, beta = np.exp(mu), 1. / sigma
		if (np.isinf(alpha) | np.isnan(alpha)):
			alpha = np.median(x)
		if (np.isinf(beta) | np.isnan(beta)):
			beta = 1.
		return alpha, beta, 1.

	def sf(self, x, alpha, beta, mu):
		return 1 - np.power(1 - np.exp(-(x / alpha)**beta), mu)

	def ff(self, x, alpha, beta, mu):
		return np.power(1 - np.exp(-(x / alpha)**beta), mu)

	def cs(self, x, X, alpha, beta, mu):
		return self.sf(x + X, alpha, beta, mu) / self.sf(X, alpha, beta, mu)

	def df(self, x, alpha, beta, mu):
		return (beta * mu * x**(beta - 1)) / (alpha**beta) \
				* (1 - np.exp(-(x/alpha)**beta))**(mu - 1) \
				* np.exp(-(x/alpha)**beta)

	def hf(self, x, alpha, beta, mu):
		return self.df(x, alpha, beta, mu) / self.sf(x, alpha, beta, mu)

	def Hf(self, x, alpha, beta, mu):
		return -np.log(self.sf(x, alpha, beta, mu))

	def qf(self, p, alpha, beta, mu):
		return alpha * (-np.log(1 - p**(1./mu)))**(1/beta)

	def mean(self, alpha, beta, mu):
		func = lambda x : x * self.df(x, alpha, beta, mu)
		top = 2 * self.qf(0.999, alpha, beta, mu)
		return integrate.quadrature(func, 0, top)[0]

	def random(self, size, alpha, beta, mu):
		U = uniform.rvs(size=size)
		return self.qf(U, alpha, beta, mu)

	def _mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y', on_d_is_0=False):
		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		if on_d_is_0:
			pass
		else:
			F = F[d > 0]
			x = x[d > 0]

		init = self.parameter_initialiser(x, c, n)
		init = np.array(init)

		if rr == 'y':
			# Three parameter
			#beta = np.polyfit(x, self.mpp_y_transform(F, alpha), deg=1)[0]
			y = self.mpp_y_transform(F)
			fun = lambda params : np.sum((y - self.mpp_y_transform(self.ff(x, *params))) ** 2)
			res = minimize(fun, init, bounds=self.bounds)
		else:
			# Three parameter
			#beta = 1./np.polyfit(self.mpp_y_transform(F, alpha), x, deg=1)[0]
			fun = lambda params : np.sum((np.log(x) - np.log(self.qf(F, *params))) ** 2)
			res = minimize(fun, init, bounds=self.bounds)
		return res.x

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
		return alpha, beta, 1.

ExpoWeibull = ExpoWeibull_('ExpoWeibull')