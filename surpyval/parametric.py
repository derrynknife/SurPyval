import re
from autograd import jacobian, hessian
import autograd.numpy as np
from autograd.numpy.linalg import inv
from autograd.scipy.stats import norm
from scipy.stats import norm as scipy_norm
from scipy.stats import uniform
from numpy import euler_gamma

from scipy.special import gamma as gamma_func
from scipy.special import gammainc, gammaincinv
from scipy.special import factorial
from scipy.optimize import minimize
from scipy.special import ndtri as z
from scipy.optimize import approx_fprime
from scipy.stats import pearsonr

from surpyval import nonparametric as nonp

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

NUM     = np.float64
TINIEST = np.finfo(NUM).tiny
EPS     = np.sqrt(np.finfo(NUM).eps)

def round_sig(points, sig=2):
    places = sig - np.floor(np.log10(np.abs(points))) - 1
    output = []
    for p, i in zip(points, places):
        output.append(np.round(p, np.int(i)))
    return output

class SurpyvalDist():
	def neg_ll_1(self, x, c=None, n=None, *params):
		like = n * (self.ff(x, *params) * c * (c - 1.) / 2. +
					self.df(x, *params) * (1. - c**2.) +
					self.sf(x, *params) * c * (c + 1.) / 2.)
		like += TINIEST
		like = np.where(like < 1, like, 1)
		return -np.sum(np.log(like))

	def neg_ll(self, x, c=None, n=None, *params):
		# Use this neg_ll, will make it much easier to implement interval cens
		like = np.zeros_like(x).astype(NUM)
		like = np.where(c == 0, self.df(x, *params), like)
		like = np.where(c == -1, self.ff(x, *params), like)
		like = np.where(c == 1, self.sf(x, *params), like)
		like += TINIEST
		like = np.where(like < 1, like, 1)
		return -np.sum(np.log(like))

	def neg_mean_D(self, x, c=None, n=None, *params):
		idx = np.argsort(x)
		F  = self.ff(x[idx], *params)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if c is not None:
			Dr = self.sf(x[c == 1], *params)
			Dl = self.ff(x[c == -1], *params)
			D = np.concatenate([Dl, D, Dr])
		D[D < TINIEST] = TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		return M

	def mom_moment_gen(self, *params):
		moments = np.zeros(self.k)
		for i in range(0, self.k):
			n = i + 1
			moments[i] = self.moment(n, *params)
		return moments

	def _mse(self, x, c=None, n=None, heuristic='Nelson-Aalen'):
		"""
		MSE: Mean Square Error
		This is simply fitting the curve to the best estimate from a non-parametric estimate.

		This is slightly different in that it fits it to untransformed data.
		The transformation is how the "Probability Plotting Method" works.

		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""
		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)
		init = self.parameter_initialiser(x, c=c, n=n)
		fun = lambda t : np.sum(((self.ff(x, *t)) - F)**2)
		res = minimize(fun, init, bounds=self.bounds)
		self.res = res
		return res

	def _mps(self, x, c=None, n=None):
		"""
		MPS: Maximum Product Spacing

		This is the method to get the largest (geometric) average distance between all points

		This method works really well when all points are unique. Some complication comes in when using repeated data.

		This method is exceptional for when using three parameter distributions.
		"""
		init = self.parameter_initialiser(x, c=c, n=n)
		bounds = self.bounds
		fun = lambda t : self.neg_mean_D(x, c, n, *t)
		res = minimize(fun, init, bounds=bounds)
		return res

	def _mle(self, x, c=None, n=None):
		"""
		MLE: Maximum Likelihood estimate
		"""
		if n is None:
			n = np.ones_like(x).astype(np.int64)

		if c is None:
			c = np.zeros_like(x).astype(np.int64)

		x_ = np.copy(x)
		c_ = np.copy(c).astype(np.int64)
		n_ = np.copy(n).astype(np.int64)

		init = self.parameter_initialiser(x_, c_, n_)

		if self.use_autograd:
			try:
				fun  = lambda t : self.neg_ll(x_, c_, n_, *t)
				jac = jacobian(fun)
				hess = hessian(fun)
				res = minimize(fun, init, method='trust-exact', jac=jac, hess=hess, tol=1e-10)
				hess_inv = inv(res.hess)
			except:
				with np.errstate(all='ignore'):
					fun = lambda t : self.neg_ll(x_, c_, n_, *t)
					jac = lambda t : approx_fprime(t, fun, EPS)
					res = minimize(fun, init, method='BFGS', jac=jac)
					hess_inv = res.hess_inv

		else:
			fun = lambda t : self.neg_ll(x_, c_, n_, *t)
			jac = lambda t : approx_fprime(t, fun, EPS)
			#hess = lambda t : approx_fprime(t, jac, eps)
			res = minimize(fun, init, method='BFGS', jac=jac)
			hess_inv = res.hess_inv

		return res, jac, hess_inv

	def _mom(self, x, n=None):
		"""
		MOM: Method of Moments.

		This is one of the simplest ways to calculate the parameters of a distribution.

		This method is quick but only works with uncensored data.
		# Can I add a simple sum(c) instead of length to work with censoring?
		"""
		if n is not None:
			x = np.repeat(x, n)

		moments = np.zeros(self.k)
		for i in range(0, self.k):
			moments[i] = np.sum(x**(i+1)) / len(x)

		fun = lambda t : np.sum((moments - self.mom_moment_gen(*t))**2)
		res = minimize(fun, 
					   self.parameter_initialiser(x), 
					   bounds=self.bounds)
		return res

	def _mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y', on_d_is_0=False):
		assert rr in ['x', 'y']
		"""
		MPP: Method of Probability Plotting
		Yes, the order of this language was invented to keep MXX format consistent
		This is the classic probability plotting paper method.

		This method creates the plotting points, transforms it to Weibull scale and then fits the line of best fit.

		Fit a two parameter Weibull distribution from data.
		
		Fits a Weibull model using cumulative probability from x values. 
		"""
		x_, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)
		
		if not on_d_is_0:
			x_ = x_[d > 0]
			F = F[d > 0]

		# Linearise
		x_ = self.mpp_x_transform(x_)
		y_ = self.mpp_y_transform(F)

		if rr == 'y':
			params = np.polyfit(x_, y_, 1)
		elif rr == 'x':
			params = np.polyfit(y_, x_, 1)

		params = self.unpack_rr(params, rr)
		return params

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		x = np.array(x, dtype=NUM)
		assert x.ndim == 1

		model = Parametric()
		model.method = how
		model.raw_data = {
			'x' : x,
			'c' : c,
			'n' : n
		}

		if c is None:
			c = np.zeros_like(x).astype(np.int64)
		else:
			c = np.array(c, dtype=np.int64)

		assert c.ndim == 1
		assert c.shape == x.shape

		if n is None:
			n = np.ones_like(x).astype(np.int64)
		if n is None:
			n = np.array(n, dtype=np.int64)

		assert n.ndim == 1
		assert n.shape == x.shape

		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}

		heuristic = kwargs.get('heuristic', 'Nelson-Aalen')
		model.heuristic = heuristic
		model.dist = self

		if   how == 'MLE':
			# Maximum Likelihood
			model.res, model.jac, model.hess_inv = self._mle(x, c=c, n=n)
			model.params = tuple(model.res.x)
		elif how == 'MPS':
			# Maximum Product Spacing
			if model.raw_data['c'] is not None:
				raise Exception('Maximum product spacing doesn\'t support censoring')
			if model.raw_data['c'] is not None:
				raise Exception('Maximum product spacing doesn\'t support counts')
			model.res = self._mps(x)
			model.params = tuple(model.res.x)
		elif how == 'MOM':
			if model.raw_data['c'] is not None:
				raise Exception('Method of moments doesn\'t support censoring')
			model.res = self._mom(x, n=n)
			model.params = tuple(model.res.x)
		elif how == 'MPP':
			rr = kwargs.get('rr', 'y')
			model.params = self._mpp(x, n=n, c=c, rr=rr, heuristic=heuristic)
		elif how == 'MSE':
			model.res = self._mse(x, c=c, n=n, heuristic=heuristic)
			model.params = tuple(model.res.x)
		
		return model
class Parametric():
	def __init__(self):
		self.jac_r = np.vectorize(jacobian(lambda x : self.dist.sf(x, *self.params)))

	def sf(self, x):
		return self.dist.sf(x, *self.params)

	def ff(self, x):
		return self.dist.ff(x, *self.params)

	def df(self, x): 
		return self.dist.df(x, *self.params)

	def hf(self, x):
		return self.dist.hf(x, *self.params)

	def Hf(self, x):
		return self.dist.Hf(x, *self.params)

	def qf(self, p):
		return self.dist.qf(p, *self.params)

	def random(self, size):
		U = uniform.rvs(size=size)
		return self.qf(U, *self.params)

	def mean(self):
		return self.dist.mean(*self.params)

	def moment(self):
		return self.dist.moment(*self.params)

	def entropy(self):
		return self.dist.entropy(*self.params)

	def cb(self, x, sig):
		if self.method != 'MLE':
			raise Exception('Only MLE has confidence bounds')
			
		du = z(sig) * np.sqrt(var_u)
		u_u = u_hat + du
		u_l = u_hat - du
		return np.vstack([np.exp(-np.exp(u_u)), np.exp(-np.exp(u_l))])

	def ll(self):
		if hasattr(self, 'log_like'):
			return self.log_like
		else:
			x = self.data['x']
			c = self.data['c']
			n = self.data['n']
			self.log_like = -self.dist.neg_ll(x, c, n, *self.params)
			return self.log_like

	def aic(self):
		if hasattr(self, 'aic_'):
			return self.aic_
		else:
			x = self.data['x']
			c = self.data['c']
			n = self.data['n']
			alpha = self.alpha
			beta  = self.beta
			k = len(self.params)
			self.aic_ = 2 * k + 2 * self.dist.neg_ll(x, c, n, *self.params)
			return self.aic_

	def aic_c(self):
		if hasattr(self, 'aic_c_'):
			return self.aic_c_
		else:
			k = len(self.params)
			n = len(self.data['x'])
			self.aic_c_ = self.aic() + (2*k**2 + 2*k)/(n - k - 1)
			return self.aic_c_

	def get_plot_data(self, heuristic, plot_bounds=True, cb=0.05):
		"""
		Looking a little less ugly now.
		"""
		x = self.data['x']
		x_, r, d, F = nonp.plotting_positions(
			self.data['x'], 
			c=self.data['c'], 
			n=self.data['n'], 
			heuristic=heuristic)

		if self.dist.name in ['Weibull3p']:
			x_ = x_ - self.params[2]
			x = x - self.params[2]
		y_scale_min = np.min(F[F > 0])/2
		y_scale_max = (1 - (1 - np.max(F[F < 1]))/10)

		# x-axis
		x_min = np.min(x)
		x_max = np.max(x)
		if self.dist.plot_x_scale == 'log':
			x_min = np.log10(x_min)
			x_max = np.log10(x_max)
			vals_non_sig = 10 ** np.linspace(x_min, x_max, 7)
			x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
			x_minor_ticks = ((10**x_minor_ticks * np.array(np.arange(1, 11))
												    .reshape((10, 1)))
												    .flatten())
			diff = (x_max - x_min)/10
			x_scale_min = 10**(x_min - diff)
			x_scale_max = 10**(x_max + diff)
			x_model = 10**np.linspace(x_min - diff, x_max + diff, 100)
		else:
			vals_non_sig = np.linspace(x_min, x_max, 7)
			x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
			diff = (x_max - x_min) / 10
			x_scale_min = x_min - diff
			x_scale_max = x_max + diff
			x_model = np.linspace(x_scale_min, x_scale_max, 100)

		if self.dist.name in ['Weibull3p']:
			cdf = self.ff(x_model + self.params[2])
		else:
			cdf = self.ff(x_model)
		not_different = True
		i = 1
		while not_different:
		    x_ticks = np.array(round_sig(vals_non_sig, i))
		    not_different = (np.diff(x_ticks) == 0).any()
		    i += 1

		y_ticks = np.array(self.dist.y_ticks)
		y_ticks = y_ticks[np.where((y_ticks > y_scale_min) & 
								   (y_ticks < y_scale_max))[0]]
		y_ticks_labels = [str(int(y))+'%' if (re.match('([0-9]+\.0+)', str(y)) is not None) & 
							(y > 1) else str(y) for y in y_ticks * 100]

		x_ticks_labels = [str(int(x)) if (re.match('([0-9]+\.0+)', str(x)) is not None) & 
							(x > 1) else str(x) for x in x_ticks]

		if plot_bounds:
			if self.dist.name == 'Weibull3p':
				cbs = 1 - self.dist.R_cb(x_model + self.params[2], *self.params, self.hess_inv, cb=cb)
			else:
				cbs = 1 - self.dist.R_cb(x_model, *self.params, self.hess_inv, cb=cb)
		else:
			cbs = []

		plot_data = {
			'x_scale_min' : x_scale_min,
			'x_scale_max' : x_scale_max,
			'y_scale_min' : y_scale_min,
			'y_scale_max' : y_scale_max,
			'y_ticks' : y_ticks,
			'y_ticks_labels' : y_ticks_labels,
			'x_ticks' : x_ticks,
			'x_ticks_labels' : x_ticks_labels,
			'cdf' : cdf,
			'x_model' : x_model,
			'x_minor_ticks' : x_minor_ticks,
			'cbs' : cbs,
			'x_scale' : self.dist.plot_x_scale,
			'x_' : x_,
			'F' : F
		}
		return plot_data

	def plot(self, heuristic, plot_bounds=True, cb=0.05):
		d = self.get_plot_data(heuristic, plot_bounds=plot_bounds, cb=cb)
		# MAKE THE PLOT
		# Set the y limits
		plt.gca().set_ylim([d['y_scale_min'], d['y_scale_max']])
		
		# Set the x scale
		plt.xscale(d['x_scale'])
		# Set the y scale
		if self.dist.name == 'Gamma':
			# The y scale for the gamm distribution is dependent on the shape
			plt.gca().set_yscale('function',
				functions=(lambda x : self.dist.mpp_y_transform(x, self.params[0]),
						   lambda x : self.dist.mpp_inv_y_transform(x, self.params[0])))
		else:
			plt.gca().set_yscale('function', 
				functions=(self.dist.mpp_y_transform, 
						   self.dist.mpp_inv_y_transform))
		
		# Set Major Y axis ticks
		plt.yticks(d['y_ticks'], labels=d['y_ticks_labels'])
		# Set Minor Y axis ticks
		plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 51)))
		# Set Major X axis ticks
		plt.xticks(d['x_ticks'], labels=d['x_ticks_labels'])
		# Set Minor X axis ticks if log scale.
		if d['x_scale'] == 'log':
			plt.gca().set_xticks(d['x_minor_ticks'], minor=True)
			plt.gca().set_xticklabels([], minor=True)

		# Turn on the grid
		plt.grid(b=True, which='major', color='g', alpha=0.4, linestyle='-')
		plt.grid(b=True, which='minor', color='g', alpha=0.1, linestyle='-')

		# Label it
		plt.title('{} Probability Plot'.format(self.dist.name))
		plt.ylabel('CDF')
		plt.scatter(d['x_'], d['F'])
		plt.gca().set_xlim([d['x_scale_min'], d['x_scale_max']])
		if plot_bounds:
			plt.plot(d['x_model'], d['cbs'], color='r')
		return plt.plot(d['x_model'], d['cdf'], color='k', linestyle='--')
class Weibull_(SurpyvalDist):
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

	def parameter_initialiser(self, x, c=None, n=None):
		gumb = Gumbel.fit(np.log(x), c, n, how='MLE')
		if not gumb.res.success:
			gumb = Gumbel.fit(np.log(x), c, n, how='MPP')
		mu, sigma = gumb.params
		alpha, beta = np.exp(mu), 1. / sigma
		return np.exp(mu), 1 / sigma

	def sf(self, x, alpha, beta):
		return np.exp(-(x / alpha)**beta)

	def ff(self, x, alpha, beta):
		return 1 - np.exp(-(x / alpha)**beta)

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
class Gumbel_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((None, None), (0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]

	def parameter_initialiser(self, x, c=None, n=None):
		if n is None:
		    n = np.ones_like(x).astype(np.int32)

		if c is None:
			c = np.zeros_like(x).astype(np.int32)

		flag = (c == 0).astype(NUM)

		return x.sum() / (n * flag).sum(), np.std(x)

	def sf(self, x, mu, sigma):
		return np.exp(-np.exp((x - mu)/sigma))

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

	def mpp_x_transform(self, x):
		return x

	def mpp_y_transform(self, y):
		return np.log(-np.log((1 - y)))

	def mpp_inv_y_transform(self, y):
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
class Exponential_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		self.k = 1
		self.bounds = ((0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.05, 0.4, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]

	def parameter_initialiser(self, x, c=None, n=None):
		if n is None:
		    n = np.ones_like(x)

		if c is None:
			c = np.zeros_like(x)

		c = (c == 0).astype(NUM)

		return [(n * c).sum()/x.sum()]

	def sf(self, x, failure_rate):
		return np.exp(-failure_rate * x)

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
class Weibull3p_(SurpyvalDist):
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

	def parameter_initialiser(self, x, c=None, n=None):
		if n is None:
		    n = np.ones_like(x).astype(np.int64)

		if c is None:
			c = np.zeros_like(x).astype(np.int64)

		flag = (c == 0).astype(NUM)

		xx = np.copy(x)
		nn = np.copy(n)
		cc = np.copy(c)

		diff = (np.max(x) - np.min(x))/10

		init_mpp = Weibull3p.fit(x, c=c, n=n, how='MPP').params
		init = init_mpp[0], init_mpp[1], np.min(x) - diff
		#init = x.sum() / (n * flag).sum(), 1., np.min(x) - 1
		self.bounds = ((0, None), (0, None), (None, np.min(x)))
		return init

	def sf(self, x, alpha, beta, gamma):
		return np.exp(-((x - gamma) / alpha)**beta)

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
class Normal_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((None, None), (0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 
				0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]


	def parameter_initialiser(self, x, c=None, n=None):
		if n is None:
		    n = np.ones_like(x)

		if c is None:
			c = np.zeros_like(x)

		c = (c == 0).astype(NUM)

		return x.sum() / (n * c).sum(), np.std(x)

	def sf(self, x, mu, sigma):
		return norm.sf(x, mu, sigma)

	def ff(self, x, mu, sigma):
		return norm.cdf(x, mu, sigma)

	def df(self, x, mu, sigma):
		return norm.pdf(x, mu, sigma)

	def hf(self, x, mu, sigma):
		return norm.pdf(x, mu, sigma) / self.sf(x, mu, sigma)

	def Hf(self, x, mu, sigma):
		return -np.log(norm.sf(x, mu, sigma))

	def qf(self, p, mu, sigma):
		return scipy_norm.ppf(p, mu, sigma)

	def mean(self, mu, sigma):
		return mu

	def moment(self, n, mu, sigma):
		return norm.moment(n, mu, sigma)

	def random(self, size, mu, sigma):
		U = uniform.rvs(size=size)
		return self.qf(U, mu, sigma)

	def mpp_x_transform(self, x):
		return x

	def mpp_y_transform(self, y):
		return self.qf(y, 0, 1)

	def mpp_inv_y_transform(self, y):
		return self.ff(y, 0, 1)

	def unpack_rr(self, params, rr):
		if rr == 'y':
			sigma, mu = params
			mu = -mu/sigma
			sigma = 1./sigma
		elif rr == 'x':
			sigma, mu = params
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
class LogNormal_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((0, None), (0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'log'
		self.y_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]

	def parameter_initialiser(self, x, c=None, n=None):
		norm_mod = Normal.fit(np.log(x), c=c, n=n, how='MLE')
		mu, sigma = norm_mod.params
		return mu, sigma

	def sf(self, x, mu, sigma):
		return 1 - self.ff(x, mu, sigma)

	def ff(self, x, mu, sigma):
		return norm.cdf(np.log(x), mu, sigma)

	def df(self, x, mu, sigma):
		return 1./x * norm.pdf(np.log(x), mu, sigma)

	def hf(self, x, mu, sigma):
		return self.pdf(x, mu, sigma) / self.sf(x, mu, sigma)

	def Hf(self, x, mu, sigma):
		return -np.log(self.sf(x, mu, sigma))

	def qf(self, p, mu, sigma):
		return np.exp(scipy_norm.ppf(p, mu, sigma))

	def mean(self, mu, sigma):
		return np.exp(mu + (sigma**2)/2)

	def random(self, size, mu, sigma):
		return np.exp(Normal.random(size, mu, sigma))

	def mpp_x_transform(self, x):
		return np.log(x)

	def mpp_y_transform(self, y):
		return Normal.qf(y, 0, 1)

	def mpp_inv_y_transform(self, y):
		return Normal.ff(y, 0, 1)

	def unpack_rr(self, params, rr):
		if rr == 'y':
			sigma, mu = params
			mu = -mu/sigma
			sigma = 1./sigma
		elif rr == 'x':
			sigma, mu = params
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
		t = np.log(x)
		return Normal.sf(self.z_cb(t, mu, sigma, cv_matrix, cb=0.05), 0, 1).T
class Gamma_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((0, None), (0, None),)
		self.use_autograd = False
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]

	def parameter_initialiser(self, x, c=None, n=None):
		# These equations are truly magical
		s = np.log(x.sum()/len(x)) - np.log(x).sum()/len(x)
		alpha = (3 - s + np.sqrt((s - 3)**2 + 24*s)) / (12*s)
		beta = x.sum()/(len(x)*alpha)
		return alpha, 1./beta

	def sf(self, x, alpha, beta):
		return 1 - self.ff(x, alpha, beta)

	def ff(self, x, alpha, beta):
		return gammainc(alpha, beta * x)

	def df(self, x, alpha, beta):
		return ((beta ** alpha) * x ** (alpha - 1) * np.exp(-(x * beta)) / (gamma_func(alpha)))

	def hf(self, x, alpha, beta):
		return self.df(x, alpha, beta) / self.sf(x, alpha, beta)

	def Hf(self, x, ahlpa, beta):
		return -np.log(self.sf(x, alpha, beta))

	def qf(self, p, alpha, beta):
		return gammaincinv(alpha, p) / beta

	def mean(self, alpha, beta):
		return alpha / beta

	def moment(self, n, alpha, beta):
		return gamma_func(n + alpha) / (beta**n * gamma_func(alpha))

	def random(self, size, alpha, beta):
		U = uniform.rvs(size=size)
		return self.qf(U, alpha, beta)

	def mpp_y_transform(self, y, alpha):
		return gammaincinv(alpha, y)

	def mpp_inv_y_transform(self, y, alpha):
		return gammainc(alpha, y)

	def mpp_x_transform(self, x):
		return x

	def var_R(self, dR, cv_matrix):
		dr_dalpha = dR[:, 0]
		dr_dbeta  = dR[:, 1]
		var_r = (dr_dalpha**2 * cv_matrix[0, 0] + 
				 dr_dbeta**2  * cv_matrix[1, 1] + 
				 2 * dr_dalpha * dr_dbeta * cv_matrix[0, 1])
		return var_r

	def R_cb(self, x, alpha, beta, cv_matrix, cb=0.05):
		R_hat = self.sf(x, alpha, beta)
		dR_f = lambda t : self.sf(*t)
		jac = lambda t : approx_fprime(t, dR_f, EPS)[1::]
		x_ = np.array(x)
		if x_.size == 1:
			dR = jac((x_, alpha, beta))
			dR = dR.reshape(1, 2)
		else:
			out = []
			for xx in x_:
				out.append(jac((xx, alpha, beta)))
			dR = np.array(out)
		K = z(cb/2)
		exponent = K * np.array([-1, 1]).reshape(2, 1) * np.sqrt(self.var_R(dR, cv_matrix))
		exponent = exponent/(R_hat*(1 - R_hat))
		R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
		return R_cb.T
class Uniform_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		# Set 'k', the number of parameters
		self.k = 2
		self.bounds = ((None, None), (None, None),)
		self.use_autograd = True
		self.plot_x_scale = 'linear'
		self.y_ticks = np.linspace(0, 1, 21)[1:-1]

	def parameter_initialiser(self, x, c=None, n=None):
		return np.min(x), np.max(x)

	def sf(self, x, a, b):
		return 1 - self.ff(x, a, b)

	def ff(self, x, a, b):
		f = np.zeros_like(x)
		f = np.where(x < a, 0, f)
		f = np.where(x > b, 1, f)
		f = np.where(((x <= b) & (x >= a)), (x - a)/(b - a), f)
		return f

	def df(self, x, a, b):
		d = np.zeros_like(x)
		d = np.where(x < a, 0, d)
		d = np.where(x > b, 0, d)
		d = np.where(((x <= b) & (x >= a)), 1./(b - a), d)
		return d

	def hf(self, x, a, b):
		return self.df(x, a, b) / self.sf(x, a, b)

	def Hf(self, x, a, b):
		return np.log(self.sf(x, a, b))

	def qf(self, p, a, b):
		return a + p*(b - a)

	def mean(self, a, b):
		return 0.5 * (a + b)

	def moment(self, n, a, b):
		if n == 0:
			return 1
		else:
			out = np.zeros(n)
			for i in range(n):
				out[i] = a**i * b**(n-i)
			return np.sum(out)/(n + 1)

	def random(self, size, a, b):
		U = uniform.rvs(size=size)
		return self.qf(U, a, b)

	def mpp_x_transform(self, x):
		return x

	def mpp_y_transform(self, y):
		return y

	def mpp_inv_y_transform(self, y):
		return y
class Weibull_Mix_Two_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		self.n = 2
		# Set 'k', the number of parameters
		self.k = 4
		self.bounds = ((0, None), (0, None), (0, None), (0, None),)
		self.use_autograd = False
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]

	def parameter_initialiser(self, x, c=None, n=None):
		return np.mean(x), 1.

	def df(self, x, n, w, alphas, betas): 
		f = np.zeros_like(x) 
		for i in range(self.n): 
			f += w[i] * Weibull.df(x, alphas[i], betas[i]) 
		return f 

	def ff(self, x, n, w, alphas, betas): 
		F = np.zeros_like(x) 
		for i in range(self.n): 
			F += w[i] * Weibull.ff(x, alphas[i], betas[i]) 
		return F

	def sf(self, x, n, w, alphas, betas):
		return 1 - self.ff(x, n, w, alphas, betas)

	def mean(self, n, w, alphas, betas):
		mean = 0
		for i in range(n):
			mean += w[i] * alphas[i] * gamma_func(1 + 1./betas[i])
		return mean

	def mpp_x_transform(self, x):
		return np.log(x)

	def mpp_y_transform(self, y):
		return np.log(-np.log((1 - y)))

	def mpp_inv_y_transform(self, y):
		return 1 - np.exp(-np.exp(y))

	def fit(self, x):
		x = np.array(x, dtype=NUM)
		assert x.ndim == 1
		model = Parametric()

		model.method = 'EM'
		model.dist = self
		model.raw_data = {
			'x' : x,
			'c' : None,
			'n' : None
		}

		c = np.zeros_like(x).astype(np.int64)
		n = np.ones_like(x, dtype=np.int64)

		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}

		wmm = WMM(data=x)
		wmm.fit()
		model.log_like = wmm.ll()

		model.params = tuple((self.n, wmm.w, wmm.alphas, wmm.betas))

		return model

class WMM(): 
	def __init__(self, **kwargs): 
		assert 'data' in kwargs 
		self.n = kwargs.pop('n', 2) 
		self.x = np.sort(kwargs.pop('data')) 
		self.N = len(self.x) 
		self.alphas = [np.mean(x) for x in np.array_split(self.x, self.n)] 
		self.betas = [1.] * self.n 
		self.w = np.ones(shape=(self.n)) / self.n 
		self.p = np.ones(shape=(self.n, len(self.x))) / self.n 
		# Set 'k', the number of parameters
		self.k = 2
		self.bounds = ((0, None), (0, None),)
		self.use_autograd = False
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.method = 'EM'
		self.dist = Weibull

	def Q_prime(self, params): 
		tmp_params = params.reshape(self.n, 2) 
		f = np.zeros_like(self.p) 
		for i in range(self.n): 
			like = np.log(Weibull.df(self.x, tmp_params[0, i], tmp_params[1, i])) 
			mask = np.isinf(like) 
			if any(mask): 
				#print(np.sum(like[~np.isneginf(like)])) 
				like[np.isneginf(like)] = np.log(TINIEST) 
				#print('max\'d') 
			f[i] = np.multiply(self.p[i], like) 
		f = -np.sum(f)
		self.loglike = f
		return f 

	def expectation(self): 
		for i in range(self.n): 
			self.p[i] = self.w[i] * Weibull.df(self.x, self.alphas[i], self.betas[i]) 
			self.p = np.divide(self.p, np.sum(self.p, axis=0)) 
			self.w = np.sum(self.p, axis=1) / self.N 

	def maximisation(self): 
		params = np.concatenate((self.alphas, self.betas)) 
		bounds = ((0, None), (0, None),) * self.n 
		res = minimize(self.Q_prime, params, bounds=bounds) 
		if not res.success: 
			print(res) 
			raise Exception('Max failed') 
		ab = res.x.reshape(self.n, 2) 
		self.alphas = ab[0] 
		self.betas = ab[1]

	def EM(self):
		self.expectation() 
		self.maximisation()

	def fit(self, tol=1e-6):
		self.EM()
		f0 = self.loglike
		self.EM()
		f1 = self.loglike
		while np.abs(f0 - f1) > tol:
			f0 = f1
			self.EM()
			f1 = self.loglike
	def ll(self):
		return self.loglike

	def mean(self):
		mean = 0
		for i in range(self.n):
			mean += self.w[i] * self.alphas[i] * gamma_func(1 + 1./self.betas[i])
		return mean

	def __str__(self): 
		print(self.alphas) 
		print(self.betas) 
		print(self.w) 
		return "Done" 

	def df(self, t): 
		f = np.zeros_like(t) 
		for i in range(self.n): 
			f += self.w[i] * Weibull.df(t, self.alphas[i], self.betas[i]) 
		return f 

	def ff(self, t): 
		F = np.zeros_like(t) 
		for i in range(self.n): 
			F += self.w[i] * Weibull.ff(t, self.alphas[i], self.betas[i]) 
		return F

	def sf(self, t): 
		return 1 - self.ff



Weibull = Weibull_('Weibull')
Gumbel = Gumbel_('Gumbel')
Exponential = Exponential_('Exponential')
Gamma = Gamma_('Gamma')
Weibull3p = Weibull3p_('Weibull3p')
Normal = Normal_('Normal')
LogNormal = LogNormal_('LogNormal')
Uniform = Uniform_('Uniform')
Weibull_Mix_Two = Weibull_Mix_Two_('Weibull_Mix_Two')

