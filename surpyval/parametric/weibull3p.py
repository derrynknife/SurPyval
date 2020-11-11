import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize
from scipy.stats import pearsonr
import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Weibull3p_(ParametricFitter):
	"""
	class for the three parameter weibull distribution.
	"""
	def __init__(self, name):
		self.name = name
		self.k = 3
		self.bounds = ((0, None), (0, None), (None, None),)
		self.use_autograd = True
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['alpha', 'beta', 'gamma']

	def parameter_initialiser(self, x, c=None, n=None):
		x, c, n = surpyval.xcn_handler(x, c, n)
		diff = (np.max(x) - np.min(x))/10
		init_mpp = Weibull3p.fit(x, c=c, n=n, how='MPP', heuristic='Turnbull').params
		init = init_mpp[0], init_mpp[1], np.min(x) - diff
		self.bounds = ((0, None), (0, None), (None, np.min(x)))
		return init

	def sf(self, x, alpha, beta, gamma):
		return np.exp(-((x - gamma) / alpha)**beta)

	def cs(self, x, X, alpha, beta, gamma):
		return self.sf(x + X, alpha, beta, gamma) / self.sf(X, alpha, beta, gamma)

	def ff(self, x, alpha, beta, gamma):
		return 1 - np.exp(-((x - gamma) / alpha)**beta)

	def df(self, x, alpha, beta, gamma):
		return (beta / alpha) * ((x - gamma) / alpha)**(beta-1) \
			* np.exp(-((x - gamma) / alpha)**beta)

	def hf(self, x, alpha, beta, gamma):
		return (beta / alpha) * ((x - gamma) / alpha)**(beta - 1)

	def Hf(self, x, alpha, beta, gamma):
		return ((x - gamma) / alpha)**beta

	def qf(self, p, alpha, beta, gamma):
		return alpha * (-np.log(1 - p))**(1/beta) + gamma

	def mean(self, alpha, beta, gamma):
		return alpha * gamma_func(1 + 1/beta) + gamma

	def moment(self, n, alhpa, beta, gamma):
		return alpha**n * gamma_func(1 + n/beta) + gamma

	def random(self, size, alpha, beta, gamma):
		U = uniform.rvs(size=size)
		return self.qf(U, alpha, beta, gamma)

	def mpp_x_transform(self, x):
		return np.log(x)

	def mpp_y_transform(self, y):
		return np.log(-np.log((1 - y)))

	def mpp_inv_y_transform(self, y):
		return 1 - np.exp(-np.exp(y))

	def du(self, x, alpha, beta, gamma):
		du_dbeta = np.log(x - gamma) - np.log(alpha)
		du_dalpha  = -beta/alpha
		du_dgamma = beta / (gamma - x)
		return du_dalpha, du_dbeta, du_dgamma

	def var_u(self, x, alpha, beta, gamma, cv_matrix):
		da, db, dg = self.du(x, alpha, beta, gamma)
		var_u = (da**2 * cv_matrix[0, 0] + db**2 * cv_matrix[1, 1] + 
				 dg**2 * cv_matrix[2, 2] + 2 * da * db * cv_matrix[0, 1] +
				 2 * da * dg * cv_matrix[0, 2] + 2 * db * dg * cv_matrix[1, 2])
		return var_u

	def u(self, x, alpha, beta, gamma):
		return beta * (np.log(x - gamma) - np.log(alpha))

	def u_cb(self, x, alpha, beta, gamma, cv_matrix, cb=0.05):
		u = self.u(x, alpha, beta, gamma)
		var_u = self.var_u(x, alpha, beta, gamma, cv_matrix)
		diff = z(cb/2) * np.sqrt(var_u)
		bounds = u + np.array([1., -1.]).reshape(2, 1) * diff
		return bounds

	def R_cb(self, x, alpha, beta, gamma, cv_matrix, cb=0.05):
		return np.exp(-np.exp(self.u_cb(x, alpha, beta, gamma, cv_matrix, cb))).T

	def _mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y'):
		assert rr in ['x', 'y']
		"""
		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model using cumulative probability from x values. 
		"""
		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)
		
		# Linearise
		y_ = np.log(np.log(1/(1 - F)))

		mask = ((~np.isnan(y_)) & (~np.isinf(y_)))
		y_ = y_[mask]
		x  = x[mask]

		# Find gamma with maximum correlation
		gamma = np.min(x) - (np.max(x) - np.min(x))/10

		fun = lambda gamma : -pearsonr(np.log(x - gamma), y_)[0]
		res = minimize(fun, gamma, bounds=[(None, np.min(x))])
		gamma = res.x[0]

		if   rr == 'y':
			model = np.polyfit(np.log(x - gamma), y_, 1)
			beta  = model[0]
			alpha = np.exp(model[1]/-beta)
		elif rr == 'x':
			model = np.polyfit(y_, np.log(x - gamma), 1)
			beta  = 1./model[0]
			alpha = np.exp(model[1] / (beta * model[0]))
		return alpha, beta, gamma

	def jacobian(self, x, alpha, beta, gamma, c=None, n=None):
		# Done in a past iteration, and now don't have the heart to delete
		if c is None:
			c = np.zeros_like(x)

		if n is None:
			n = np.ones_like(x)
		
		f = c == 0
		l = c == -1
		r = c == 1

		dll_dbeta = (
			1./beta * np.sum(n[f]) +
			np.sum(n[f] * np.log((x[f] - gamma)/alpha)) - 
			np.sum(n[f] * ((x[f] - gamma)/alpha)**beta * np.log((x[f] - gamma)/alpha)) - 
			np.sum(n[r] * ((x[r] - gamma)/alpha)**beta * np.log((x[r] - gamma)/alpha)) +
			np.sum(n[l] * (((x[l] - gamma) / alpha) ** beta * np.log((x[l] - gamma)/alpha) * np.exp(-((x[l] - gamma)/alpha)**beta)) / 
				(1 - np.exp(-((x[l] - gamma)/alpha)**beta)))
			)

		dll_dalpha = (0 -
			beta/alpha * np.sum(n[f]) +
			beta/alpha * np.sum(n[f] * ((x[f] - gamma)/alpha)**beta) +
			beta/alpha * np.sum(n[r] * ((x[r] - gamma)/alpha)**beta) -
			beta/alpha * np.sum(n[l] * ((x[l] - gamma)/alpha)**beta * np.log((x[l] - gamma)/alpha) * np.exp(-((x[l] - gamma)/alpha)**beta) /
				(1 - np.exp(-((x[l] - gamma)/alpha)**beta)))
		)

		dll_dgamma = (
			(1 - beta) * np.sum(n[f]/(x[f] - gamma)) + 
			np.sum(n[f] * ((x[f] - gamma) / alpha) ** beta * (beta / (x[f] - gamma))) +
			np.sum(n[r] * ((x[r] - gamma) / alpha) ** beta * (beta / (x[r] - gamma))) +
			np.sum(n[l] * (-beta/(x[l] - gamma) * ((x[l] - gamma)/alpha)**beta * np.exp(-((x[l] - gamma)/alpha)**beta)) / 
				(1 - np.exp(-((x[l] - gamma)/alpha)**beta)))
		)

		return -np.array([dll_dalpha, dll_dbeta, dll_dgamma])

Weibull3p = Weibull3p_('Weibull3p')