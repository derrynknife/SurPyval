import autograd.numpy as np
from scipy.stats import uniform
from autograd import jacobian
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z
from scipy import integrate
from scipy.optimize import minimize

from surpyval import parametric as para
from surpyval import nonparametric as nonp
from surpyval.parametric.parametric_fitter import ParametricFitter

from .fitters.mpp import mpp

class ExpoWeibull_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		# Set 'k', the number of parameters
		self.k = 3
		self.bounds = ((0, None), (0, None), (0, None),)
		self.support = (0, np.inf)
		# self.use_autograd = True
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['alpha', 'beta', 'mu']
		self.param_map = {
			'alpha' : 0,
			'beta'  : 1,
			'mu'    : 2
		}

	def parameter_initialiser(self, x, c=None, n=None, offset=False):
		if offset:
			return self._mpp(dist=self, x=x, c=c, n=n, on_d_is_0=True, offset=True)
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

		y = self.mpp_y_transform(F)
		fun = lambda a : np.linalg.lstsq(x, self.mpp_y_transform(F, x))[1]
		res = minimize(fun, 1., bounds=((0, None),))
		mu = res.x[0]

		if rr == 'y':
			fun = lambda params : np.sum((y - self.mpp_y_transform(self.ff(x, *params[0:-1]), mu)) ** 2)
			res = minimize(fun, init[0:2], bounds=self.bounds)
		else:
			pass
		return np.array(*res.x, mu)

	def mpp_x_transform(self, x, gamma=0):
		return np.log(x - gamma)

	def mpp_y_transform(self, y, *params):
		mu = params[self.param_map['mu']]
		return np.log(-np.log((1 - y**(1./mu))))

	def mpp_inv_y_transform(self, y, *params):
		mu = params[self.param_map['mu']]
		return (1 - np.exp(-np.exp(y)))**mu

	def unpack_rr(self, params, rr):
		#UPDATE ME
		if rr == 'y':
			beta  = params[0]
			alpha = np.exp(params[1]/-beta)
		elif rr == 'x':
			beta  = 1./params[0]
			alpha = np.exp(params[1] / (beta * params[0]))
		return alpha, beta, 1.
		
ExpoWeibull = ExpoWeibull_('ExpoWeibull')