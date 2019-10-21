import numpy as np
from numpy import euler_gamma
#from .errors import InputError
from scipy.special import gamma as gamma_func
from scipy.special import factorial
from scipy.optimize import minimize
from scipy.special import ndtri as z
from scipy.stats import norm
from scipy.stats import lognorm
try: 
	import SurPyval.nonparametric as nonp
except:
	import nonparametric as nonp

#import WeibullScale

NUM = np.float64
TINIEST = np.finfo(NUM).tiny

class Parametric():
	def __init__(self):
		pass

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
		U = np.random.uniform(size=size)
		return self.qf(U, *self.params)

	def mean(self):
		return self.dist.mean(*self.params)

	def moment(self):
		return self.dist.moment(*self.params)

	def entropy(self):
		return self.dist.entropy(*self.params)

	def cb(self, x, sig):
		if self.method != 'MLE':
			raise InvalidError('Only MLE has confidence bounds')
			
		u_hat = self.beta * (np.log(x) - np.log(self.alpha))
		var_alpha = self.res.hess_inv[0, 0]
		var_beta  = self.res.hess_inv[1, 1]
		covar_ab  = self.res.hess_inv[1, 0]
		var_u = (
			(u_hat/ self.beta)**2 * var_beta +
			(self.beta/self.alpha)**2 * var_alpha - 
			2 * u_hat / self.alpha * covar_ab
		)
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
			self.log_like = -self.dist.neg_ll(x, *self.params, c=c, n=n)
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
			self.aic_ = 2 * k + 2 * self.dist.neg_ll(x, *self.params, c=c, n=n)
			return self.aic_

	def aic_c(self):
		if hasattr(self, 'aic_c_'):
			return self.aic_c_
		else:
			k = len(self.params)
			n = len(self.data['x'])
			self.aic_c_ = self.aic() + (2*k**2 + 2*k)/(n - k - 1)
			return self.aic_c_
class LFP_():
	"""
	class for the generic weibull distribution.

	Can be used to create 

	N.B careful with parallelisation!
	"""
	def __init__(self, name):
		self.name = name

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
		U = np.random.uniform(size=size)
		return self.qf(U, *self.params)

	def mean(self):
		return self.dist.mean(*self.params)

	def moment(self):
		return self.dist.moment(*self.params)

	def entropy(self):
		return self.dist.entropy(*self.params)

	def _mle(self, x, c=None, n=None, 
			 dist="Weibull", model=None):
		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : dist.neg_ll(x, t[0], t[1], c, n)
		jac = lambda t : dist.jacobian(x, t[0], t[1], c, n)
		res = minimize(fun, 
					   init,
					   method='BFGS',
					   jac=jac)
		if model is not None:
			model.res = res
		return res.x[0], res.x[1]

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		model = Parametric()
		model.method = how
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}
		model.dist = "Weibull"
		model.dist = self
		if   how == 'MLE':
			params = self._mle(x, c, n, model=model)
		# Store params with redundancy...
		model.alpha, model.beta  = params
		model.params = params
		return model
class Weibull_():
	"""
	class for the generic weibull distribution.

	Can be used to create 

	"""
	def __init__(self, name):
		self.name = name

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

	def moment(self, n, alhpa, beta):
		return alpha**n * gamma_func(1 + n/beta)

	def entropy(self, alhpa, beta):
		return euler_gamma * (1 - 1/beta) + np.log(alpha / beta) + 1

	def random(self, size, alpha, beta):
		U = np.random.uniform(size=size)
		return self.qf(U, alpha, beta)

	def neg_mean_D(self, x, alpha, beta, c=None, n=None):
		idx = np.argsort(x)
		F  = self.ff(x[idx], alpha, beta)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if c is not None:
			Dr = self.sf(x[c == 1],  alpha, beta)
			Dl = self.ff(x[c == -1], alpha, beta)
			D = np.concatenate([Dl, D, Dr])
		D[D < TINIEST] = TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		return M

	def neg_ll(self, x, alpha, beta, c=None, n=None):
		if n is None:
		    n = np.ones_like(x)
		    
		if c is None:
			like = n * self.df(x, alpha, beta)
		
		else:
			l = c == -1
			f = c ==  0
			r = c ==  1

			like_l = n[l] * self.ff(x[l], alpha, beta)
			like_f = n[f] * self.df(x[f], alpha, beta)
			like_r = n[r] * self.sf(x[r], alpha, beta)
			like = np.concatenate([like_l, like_f, like_r])
			
		like[like < TINIEST] = TINIEST

		return -np.sum(np.log(like))

	def _mse(self, x, c=None, n=None,
			heuristic='Nelson-Aalen'):
		"""
		MSE: Mean Square Error
		This is simply fitting the curve to the best estimate from a non-parametric estimate.

		This is slightly different in that it fits it to untransformed data.
		The transformation is how the "Probability Plotting Method" works.

		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""

		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : np.sum(((self.ff(x, t[0], t[1])) - F)**2)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

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
		x_ = np.log(x_)
		y_ = np.log(np.log(1/(1 - F)))

		if   rr == 'y':
			model = np.polyfit(x_, y_, 1)
			beta  = model[0]
			alpha = np.exp(model[1]/-beta)
		elif rr == 'x':
			model = np.polyfit(y_, x_, 1)
			beta  = 1./model[0]
			alpha = np.exp(model[1] / (beta * model[0]))
		return alpha, beta

	#TODO: add MSE
	def _mom(self, x, n=None):
		"""
		MOM: Method of Moments.

		This is one of the simplest ways to calculate the parameters of a distribution.

		This method is quick but only works with uncensored data.
		# Can I add a simple sum(c) instead of length to work with censoring?
		"""
		if n is not None:
			x = np.rep(x, n)
		m1 = np.sum(x) / len(x)
		m2 = np.sum(x**2) / len(x)
		fun = lambda t : ((m1**2/m2) - (gamma(1 + 1./t)**2/gamma(1 + 2./t)))**2
		res = minimize(fun, 1, bounds=((0, None),))
		beta = res.x[0]
		alpha = m1 / gamma(1 + 1./beta)
		return alpha, beta

	def _mps(self, x, c=None, n=None):
		"""
		MPS: Maximum Product Spacing

		This is the method to get the largest (geometric) average distance between all points

		This method works really well when all points are unique. Some complication comes in when using repeated data.

		This method is exceptional for when using three parameter distributions.
		"""
		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : self.neg_mean_D(x, t[0], t[1])
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

	def _mle(self, x, c=None, n=None, model=None):
		"""
		MLE: Maximum Likelihood estimate

		This is the MLE, the king of parameter estimation.
		"""

		gumb = Gumbel.fit(np.log(x), c, n, how='MLE')
		mu, sigma = gumb.params
		init = np.exp(mu), 1 / sigma
		bounds = ((0, None), (0, None))
		fun = lambda t : self.neg_ll(x, t[0], t[1], c, n)
		jac = lambda t : self.jacobian(x, t[0], t[1], c, n)
		res = minimize(fun, 
					   init,
					   method='BFGS',
					   jac=jac)
		if model is not None:
			model.res = res
		return res.x[0], res.x[1]

	def jacobian(self, x, alpha, beta, c=None, n=None):
		"""
		The jacobian for a two parameter Weibull distribution.

		Please report mistakes if found!
		"""
		if c is None:
			c = np.zeros_like(x)

		if n is None:
			n = np.ones_like(x)
		
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

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		model = Parametric()
		model.method = how
		x = np.array(x, dtype=NUM)
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}
		model.dist = "Weibull"
		model.dist = self
		if   how == 'MLE':
			# Maximum Likelihood
			params = self._mle(x, c=c, n=n, model=model)
		elif how == 'MPS':
			# Maximum Product Spacing
			if c is not None:
				raise InputError('Maximum product spacing doesn\'t support censoring')
			if n is not None:
				raise InputError('Maximum product spacing doesn\'t support counts')
			params = self._mps(x)
		elif how == 'MOM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			params = self._mom(x, n=n)
		elif how == 'MPP':
			heuristic = kwargs.get('heuristic')
			if heuristic is None:
				heuristic = 'Nelson-Aalen'
			if 'rr' in kwargs:
				rr = kwargs['rr']
			else:
				rr = 'y'
			params = self._mpp(x, n=n, c=c, rr=rr, heuristic=heuristic)
		elif how == 'MSE':
			params = self._mse(x, c=c, n=n)
		# Store params with redundancy...
		model.alpha, model.beta  = params
		model.params = params
		return model
class Gumbel_():
	"""
	class for the generic weibull distribution.

	Can be used to create 

	"""
	def __init__(self, name):
		self.name = name

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
		U = np.random.uniform(size=size)
		return self.qf(U, alpha, beta)

	def neg_mean_D(self, x, mu, sigma, c=None, n=None):
		idx = np.argsort(x)
		F  = self.ff(x[idx], mu, sigma)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if c is not None:
			Dr = self.sf(x[c == 1],  mu, sigma)
			Dl = self.ff(x[c == -1], mu, sigma)
			D = np.concatenate([Dl, D, Dr])
		D[D < TINIEST] = TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		return M

	def neg_ll(self, x, mu, sigma, c=None, n=None):
		if n is None:
		    n = np.ones_like(x)
		    
		if c is None:
			like = n * self.df(x, mu, sigma)
		
		else:
			l = c == -1
			f = c ==  0
			r = c ==  1

			like_l = n[l] * self.ff(x[l], mu, sigma)
			like_f = n[f] * self.df(x[f], mu, sigma)
			like_r = n[r] * self.sf(x[r], mu, sigma)
			like = np.concatenate([like_l, like_f, like_r])
			
		like[like < TINIEST] = TINIEST

		return -np.sum(np.log(like))

	def _mse(self, x, c=None, n=None,
			heuristic='Nelson-Aalen'):
		"""
		MSE: Mean Square Error
		This is simply fitting the curve to the best estimate from a non-parametric estimate.

		This is slightly different in that it fits it to untransformed data.
		The transformation is how the "Probability Plotting Method" works.

		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""

		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		init = [np.mean(x), np.std(x)]
		bounds = ((None, None), (None, None))
		fun = lambda t : np.sum(((self.ff(x, t[0], t[1])) - F)**2)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

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
		x_ = x_
		y_ = np.log(-np.log(1 - F))

		if   rr == 'y':
			model = np.polyfit(x_, y_, 1)
			sigma = 1/model[0]
			mu    = -sigma * model[1]
		elif rr == 'x':
			model = np.polyfit(y_, x_, 1)
			sigma  = 1./model[0]
			mu = np.exp(model[1] / (beta * model[0]))
		return mu, sigma

	#TODO: add MSE
	def _mom(self, x, n=None):
		"""
		MOM: Method of Moments.

		This is one of the simplest ways to calculate the parameters of a distribution.

		This method is quick but only works with uncensored data.
		# Can I add a simple sum(c) instead of length to work with censoring?
		"""
		if n is not None:
			x = np.rep(x, n)
		m1 = np.sum(x) / len(x)
		m2 = np.sum(x**2) / len(x)
		fun = lambda t : ((m1**2/m2) - (gamma(1 + 1./t)**2/gamma(1 + 2./t)))**2
		res = minimize(fun, 1, bounds=((0, None),))
		beta = res.x[0]
		alpha = m1 / gamma(1 + 1./beta)
		return alpha, beta

	def _mps(self, x, c=None, n=None):
		"""
		MPS: Maximum Product Spacing

		This is the method to get the largest (geometric) average distance between all points

		This method works really well when all points are unique. Some complication comes in when using repeated data.

		This method is exceptional for when using three parameter distributions.
		"""
		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : self.neg_mean_D(x, t[0], t[1])
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

	def _mle(self, x, c=None, n=None, model=None):
		"""
		MLE: Maximum Likelihood estimate

		This is the MLE, the king of parameter estimation.
		"""
		if n is None:
			n = np.ones_like(x)
		if c is None:
			c = np.zeros_like(x)

		#init = self._mpp(x, c=c, n=n, on_d_is_0=True)
		init = np.mean(x), 1
		bounds = ((None, None), (0, None))
		fun = lambda t : self.neg_ll(x, t[0], t[1], c, n)
		res = minimize(fun, init)

		if model is not None:
			model.res = res
		return res.x[0], res.x[1]

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		model = Parametric()
		model.method = how
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}
		model.dist = "Gumbel"
		model.dist = self
		if   how == 'MLE':
			# Maximum Likelihood
			params = self._mle(x, c=c, n=n, model=model)
		elif how == 'MPS':
			# Maximum Product Spacing
			if c is not None:
				raise InputError('Maximum product spacing doesn\'t support censoring')
			if n is not None:
				raise InputError('Maximum product spacing doesn\'t support counts')
			params = self._mps(x)
		elif how == 'MOM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			params = self._mom(x, n=n)
		elif how == 'MPP':
			heuristic = 'Nelson-Aalen'
			if c is not None:
				heuristic = 'Nelson-Aalen'
			if 'rr' in kwargs:
				rr = kwargs['rr']
			else:
				rr = 'y'
			params = self._mpp(x, n=n, c=c, rr=rr, heuristic=heuristic)
		elif how == 'MSE':
			params = self._mse(x, c=c, n=n)
		# Store params with redundancy...
		model.mu, model.sigma  = params
		model.params = params
		return model
class Exponential_():
	"""
	class for the generic weibull distribution.

	Can be used to create 

	"""
	def __init__(self, name):
		self.name = name

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
		U = np.random.uniform(size=size)
		return self.qf(U, alpha, beta)

	def neg_mean_D(self, x, failure_rate, c=None, n=None):
		idx = np.argsort(x)
		F  = self.ff(x[idx], failure_rate)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if c is not None:
			Dr = self.sf(x[c == 1],  failure_rate)
			Dl = self.ff(x[c == -1], failure_rate)
			D = np.concatenate([Dl, D, Dr])
		D[D < TINIEST] = TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		return M

	def neg_ll(self, x, failure_rate, c=None, n=None):
		if n is None:
		    n = np.ones_like(x)
		    
		if c is None:
			like = n * self.df(x, failure_rate)
		
		else:
			l = c == -1
			f = c ==  0
			r = c ==  1

			like_l = n[l] * self.ff(x[l], failure_rate)
			like_f = n[f] * self.df(x[f], failure_rate)
			like_r = n[r] * self.sf(x[r], failure_rate)
			like = np.concatenate([like_l, like_f, like_r])
			
		like[like < TINIEST] = TINIEST

		return -np.sum(np.log(like))

	def _mse(self, x, c=None, n=None,
			heuristic='Nelson-Aalen'):
		"""
		MSE: Mean Square Error
		This is simply fitting the curve to the best estimate from a non-parametric estimate.

		This is slightly different in that it fits it to untransformed data.
		The transformation is how the "Probability Plotting Method" works.

		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""

		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : np.sum(((self.ff(x, t[0], t[1])) - F)**2)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

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
		y_ = -np.log(1 - F)

		if   rr == 'y':
			xx = x_[:,np.newaxis]
			failure_rate, _, _, _ = np.linalg.lstsq(xx, y_, rcond=None)
		elif rr == 'x':
			y_ = y_[:,np.newaxis]
			mttf, _, _, _ = np.linalg.lstsq(y_, x_, rcond=None)
			failure_rate = 1. / mttf
		return failure_rate

	#TODO: add MSE
	def _mom(self, x, n=None):
		"""
		MOM: Method of Moments.

		This is one of the simplest ways to calculate the parameters of a distribution.

		This method is quick but only works with uncensored data.
		# Can I add a simple sum(c) instead of length to work with censoring?
		"""
		if n is not None:
			x = np.rep(x, n)
		m1 = np.sum(x) / len(x)
		m2 = np.sum(x**2) / len(x)
		fun = lambda t : ((m1**2/m2) - (gamma(1 + 1./t)**2/gamma(1 + 2./t)))**2
		res = minimize(fun, 1, bounds=((0, None),))
		beta = res.x[0]
		alpha = m1 / gamma(1 + 1./beta)
		return alpha, beta

	def _mps(self, x, c=None, n=None):
		"""
		MPS: Maximum Product Spacing

		This is the method to get the largest (geometric) average distance between all points

		This method works really well when all points are unique. Some complication comes in when using repeated data.

		This method is exceptional for when using three parameter distributions.
		"""
		init = np.mean(x)
		bounds = (0, None)
		fun = lambda t : self.neg_mean_D(x, t)
		res = minimize(fun, init, bounds=bounds)
		return res.x

	def _mle(self, x, c=None, n=None, model=None):
		"""
		MLE: Maximum Likelihood estimate

		This is the MLE, the king of parameter estimation.
		"""
		if n is None:
		    n = np.ones_like(x)

		if c is None:
			c = np.zeros_like(x)

		res = (n * (1 - c)).sum() / x.sum()
		return [res]

	def jacobian(self, x, failure_rate, c=None, n=None):
		"""
		The jacobian for a two parameter Weibull distribution.

		Please report mistakes if found!
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

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		model = Parametric()
		model.method = how
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}
		model.dist = "Exponential"
		model.dist = self
		if   how == 'MLE':
			# Maximum Likelihood
			params = self._mle(x, c=c, n=n, model=model)
		elif how == 'MPS':
			# Maximum Product Spacing
			if c is not None:
				raise InputError('Maximum product spacing doesn\'t support censoring')
			if n is not None:
				raise InputError('Maximum product spacing doesn\'t support counts')
			params = self._mps(x)
		elif how == 'MOM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			params = self._mom(x, n=n)
		elif how == 'MPP':
			heuristic = 'Nelson-Aalen'
			if c is not None:
				heuristic = 'Nelson-Aalen'
			if 'rr' in kwargs:
				rr = kwargs['rr']
			else:
				rr = 'x'
			params = self._mpp(x, n=n, c=c, rr=rr, heuristic=heuristic)
		elif how == 'MSE':
			params = self._mse(x, c=c, n=n)
		# Store params with redundancy...
		model.failure_rate = params[0]
		model.params = params
		return model
class Weibull3p_():
	"""
	class for the three parameter weibull distribution.
	"""
	def __init__(self, name):
		self.name = name

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
		U = np.random.uniform(size=size)
		return self.qf(U, alpha, beta, gamma)		

	def neg_mean_D(self, x, alpha, beta, gamma, c=None, n=None):
		#print(alpha, beta, gamma)
		if gamma > np.min(x):
			gamma = np.min(x) - TINIEST
		idx = np.argsort(x)
		F  = self.ff(x[idx], alpha, beta, gamma)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if n is not None:
			# For each point
			D = np.repeat(D[0:-1], n[idx])
		if c is not None:
			Dr = self.sf(x[c == 1],  alpha, beta, gamma)
			Dl = self.ff(x[c == -1], alpha, beta, gamma)
			D = np.concatenate([Dl, D, Dr])
		D[D < TINIEST] = TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		return M

	def neg_ll(self, x, alpha, beta, gamma, c=None, n=None):
		if n is None:
		    n = np.ones_like(x)
		    
		if c is None:
			like = n * self.df(x, alpha, beta, gamma)
		
		else:
			l = c == -1
			f = c ==  0
			r = c ==  1

			like_l = n[l] * self.ff(x[l], alpha, beta, gamma)
			like_f = n[f] * self.df(x[f], alpha, beta, gamma)
			like_r = n[r] * self.sf(x[r], alpha, beta, gamma)
			like = np.concatenate([like_l, like_f, like_r])
			
		like[like < TINIEST] = TINIEST

		return -np.sum(np.log(like))

	def _mse(self, x, c=None, n=None,
			heuristic='Nelson-Aalen'):
		"""
		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""

		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		init = [np.mean(x), 1., 0.]
		bounds = ((0, None), (0, None), (None, np.min(x)))
		fun = lambda t : np.sum(((self.ff(x, t[0], t[1], t[2])) - F)**2)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1], res.x[2]

	def _mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y'):
		assert rr in ['x', 'y']
		"""
		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model using cumulative probability from x values. 
		"""
		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)
		
		# Linearise
		x_ = np.log(x)
		y_ = np.log(np.log(1/(1 - F)))

		if   rr == 'y':
			model = np.polyfit(x_, y_, 1)
			beta  = model[0]
			alpha = np.exp(model[1]/-beta)
		elif rr == 'x':
			model = np.polyfit(y_, x_, 1)
			beta  = 1./model[0]
			alpha = np.exp(model[1] / (beta * model[0]))
		return alpha, beta, 0

	def _mps(self, x, c=None, n=None):
		init = (np.mean(x), 1., np.min(x) - 1e-6)
		bounds = ((0, None), (0, None), (None, np.min(x)))
		fun = lambda t : self.neg_mean_D(x, t[0], t[1], t[2], c=c, n=n)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1], res.x[2]

	def _mle(self, x, c=None, n=None, model=None):
		init_mps = self.fit(x, c=c, n=n, how='MPS').params
		init_mpp = Weibull.fit(x - init_mps[2], c=c, n=n, how='MPP').params
		init = init_mpp[0], init_mpp[1], init_mps[2]
		bounds = ((0, None), (0, None), (None, np.min(x)))
		#fun = lambda t : self.neg_ll(x, t[0], t[1], t[2], c, n)
		#res = minimize(fun, init, bounds=bounds)
		fun = lambda t : self.neg_ll(x, t[0], t[1], t[2], c, n)
		jac = lambda t : self.jacobian(x, t[0], t[1], t[2], c, n)
		res_a = minimize(fun, 
					   init,
					   method='Nelder-Mead')
		if res_a.success:
			init = res_a.x
		res_b = minimize(fun, 
					     init,
					     method='TNC',
					     jac=jac,
					     bounds=bounds,
					     tol=1e-10)
		if res_a.success:
			res = res_a
		else:
			res = res_b
		
		if model is not None:
			model.res = res
		return res.x[0], res.x[1], res.x[2]

	def jacobian(self, x, alpha, beta, gamma, c=None, n=None):
		# If by some chance I can solve this on paper...
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

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		model = Parametric()
		model.method = how
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}
		model.dist = "Weibull"
		model.dist = self
		if   how == 'MLE':
			params = self._mle(x, c, n, model=model)
		elif how == 'MPS':
			params = self._mps(x, c=c, n=n)
		elif how == 'MOM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			params = self._mom(x, n=n)
		elif how == 'MPP':
			if 'rr' in kwargs:
				rr = kwargs['rr']
			else:
				rr = 'x'
			params = self._mpp(x, rr=rr)
		elif how == 'MSE':
			params = self._mse(x, c=c, n=n)
		# Store params with redundancy...
		model.alpha, model.beta, model.gamma  = params
		model.params = params
		return model
class Normal_():
	"""
	class for the generic weibull distribution.

	Can be used to create 

	"""
	def __init__(self, name):
		self.name = name

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
		return norm.ppf(p, mu, sigma)

	def mean(self, mu, sigma):
		return mu

	def random(self, size, mu, sigma):
		return norm.rvs(mu, sigma, size)

	def neg_mean_D(self, x, mu, sigma, c=None, n=None):
		idx = np.argsort(x)
		F  = self.ff(x[idx], mu, sigma)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if c is not None:
			Dr = self.sf(x[c == 1],  mu, sigma)
			Dl = self.ff(x[c == -1], mu, sigma)
			D = np.concatenate([Dl, D, Dr])
		D[D < TINIEST] = TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		return M

	def neg_ll(self, x, mu, sigma, c=None, n=None):
		if n is None:
		    n = np.ones_like(x)
		    
		if c is None:
			like = n * self.df(x, mu, sigma)
		
		else:
			l = c == -1
			f = c ==  0
			r = c ==  1

			like_l = n[l] * self.ff(x[l], mu, sigma)
			like_f = n[f] * self.df(x[f], mu, sigma)
			like_r = n[r] * self.sf(x[r], mu, sigma)
			like = np.concatenate([like_l, like_f, like_r])
			
		like[like < TINIEST] = TINIEST

		return -np.sum(np.log(like))

	def _mse(self, x, c=None, n=None,
			heuristic='Nelson-Aalen'):
		"""
		MSE: Mean Square Error
		This is simply fitting the curve to the best estimate from a non-parametric estimate.

		This is slightly different in that it fits it to untransformed data.
		The transformation is how the "Probability Plotting Method" works.

		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""

		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		init = [np.mean(x), np.std(x)]
		bounds = ((None, None), (None, None))
		fun = lambda t : np.sum(((self.ff(x, t[0], t[1])) - F)**2)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

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
		x_ = x_
		y_ = np.log(-np.log(1 - F))

		if   rr == 'y':
			model = np.polyfit(x_, y_, 1)
			sigma = 1/model[0]
			mu    = -sigma * model[1]
		elif rr == 'x':
			model = np.polyfit(y_, x_, 1)
			sigma  = 1./model[0]
			mu = np.exp(model[1] / (beta * model[0]))
		return mu, sigma

	#TODO: add MSE
	def _mom(self, x, n=None):
		"""
		MOM: Method of Moments.

		This is one of the simplest ways to calculate the parameters of a distribution.

		This method is quick but only works with uncensored data.
		# Can I add a simple sum(c) instead of length to work with censoring?
		"""
		if n is not None:
			x = np.rep(x, n)
		m1 = np.sum(x) / len(x)
		m2 = np.sum(x**2) / len(x)
		fun = lambda t : ((m1**2/m2) - (gamma_func(1 + 1./t)**2/gamma_func(1 + 2./t)))**2
		res = minimize(fun, 1, bounds=((0, None),))
		beta = res.x[0]
		alpha = m1 / gamma_func(1 + 1./beta)
		return mu, sigma

	def _mps(self, x, c=None, n=None):
		"""
		MPS: Maximum Product Spacing

		This is the method to get the largest (geometric) average distance between all points

		This method works really well when all points are unique. Some complication comes in when using repeated data.

		This method is exceptional for when using three parameter distributions.
		"""
		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : self.neg_mean_D(x, t[0], t[1])
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

	def _mle(self, x, c=None, n=None, model=None):
		"""
		MLE: Maximum Likelihood estimate

		This is the MLE, the king of parameter estimation.
		"""
		if n is None:
			n = np.ones_like(x)
		if c is None:
			c = np.zeros_like(x)

		#init = self._mpp(x, c=c, n=n, on_d_is_0=True)
		init = np.mean(x), np.std(x)
		bounds = ((None, None), (0, None))
		fun = lambda t : self.neg_ll(x, t[0], t[1], c, n)
		res = minimize(fun, init)

		if model is not None:
			model.res = res
		return res.x[0], res.x[1]

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		model = Parametric()
		model.method = how
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}
		model.dist = "Normal"
		model.dist = self
		if   how == 'MLE':
			# Maximum Likelihood
			params = self._mle(x, c=c, n=n, model=model)
		elif how == 'MPS':
			# Maximum Product Spacing
			if c is not None:
				raise InputError('Maximum product spacing doesn\'t support censoring')
			if n is not None:
				raise InputError('Maximum product spacing doesn\'t support counts')
			params = self._mps(x)
		elif how == 'MOM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			params = self._mom(x, n=n)
		elif how == 'MPP':
			heuristic = 'Nelson-Aalen'
			if c is not None:
				heuristic = 'Nelson-Aalen'
			if 'rr' in kwargs:
				rr = kwargs['rr']
			else:
				rr = 'y'
			params = self._mpp(x, n=n, c=c, rr=rr, heuristic=heuristic)
		elif how == 'MSE':
			params = self._mse(x, c=c, n=n)
		# Store params with redundancy...
		model.mu, model.sigma = params
		model.params = params
		return model
class LogNormal_():
	"""
	class for the generic weibull distribution.

	Can be used to create 

	"""
	def __init__(self, name):
		self.name = name

	def sf(self, x, mu, sigma):
		return lognorm.sf(x, mu, sigma)

	def ff(self, x, mu, sigma):
		return lognorm.cdf(x, mu, sigma)

	def df(self, x, mu, sigma):
		return lognorm.pdf(x, mu, sigma)

	def hf(self, x, mu, sigma):
		return lognorm.pdf(x, mu, sigma) / self.sf(x, mu, sigma)

	def Hf(self, x, mu, sigma):
		return -np.log(lognorm.sf(x, mu, sigma))

	def qf(self, p, mu, sigma):
		return lognorm.ppf(p, mu, sigma)

	def mean(self, mu, sigma):
		return np.exp(mu + (sigma**2)/2)

	def random(self, size, mu, sigma):
		return lognorm.rvs(mu, sigma, size=size)

	def neg_mean_D(self, x, mu, sigma, c=None, n=None):
		idx = np.argsort(x)
		F  = self.ff(x[idx], mu, sigma)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if c is not None:
			Dr = self.sf(x[c == 1],  mu, sigma)
			Dl = self.ff(x[c == -1], mu, sigma)
			D = np.concatenate([Dl, D, Dr])
		D[D < TINIEST] = TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		return M

	def neg_ll(self, x, mu, sigma, c=None, n=None):
		if n is None:
		    n = np.ones_like(x)
		    
		if c is None:
			like = n * self.df(x, mu, sigma)
		
		else:
			l = c == -1
			f = c ==  0
			r = c ==  1

			like_l = n[l] * self.ff(x[l], mu, sigma)
			like_f = n[f] * self.df(x[f], mu, sigma)
			like_r = n[r] * self.sf(x[r], mu, sigma)
			like = np.concatenate([like_l, like_f, like_r])
			
		like[like < TINIEST] = TINIEST

		return -np.sum(np.log(like))

	def _mse(self, x, c=None, n=None,
			heuristic='Nelson-Aalen'):
		"""
		MSE: Mean Square Error
		This is simply fitting the curve to the best estimate from a non-parametric estimate.

		This is slightly different in that it fits it to untransformed data.
		The transformation is how the "Probability Plotting Method" works.

		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""

		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)

		init = [np.mean(x), np.std(x)]
		bounds = ((None, None), (None, None))
		fun = lambda t : np.sum(((self.ff(x, t[0], t[1])) - F)**2)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

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
		x_ = x_
		y_ = np.log(-np.log(1 - F))

		if   rr == 'y':
			model = np.polyfit(x_, y_, 1)
			sigma = 1/model[0]
			mu    = -sigma * model[1]
		elif rr == 'x':
			model = np.polyfit(y_, x_, 1)
			sigma  = 1./model[0]
			mu = np.exp(model[1] / (beta * model[0]))
		return mu, sigma

	#TODO: add MSE
	def _mom(self, x, n=None):
		"""
		MOM: Method of Moments.

		This is one of the simplest ways to calculate the parameters of a distribution.

		This method is quick but only works with uncensored data.
		# Can I add a simple sum(c) instead of length to work with censoring?
		"""
		if n is not None:
			x = np.rep(x, n)
		m1 = np.sum(x) / len(x)
		m2 = np.sum(x**2) / len(x)
		fun = lambda t : ((m1**2/m2) - (gamma_func(1 + 1./t)**2/gamma_func(1 + 2./t)))**2
		res = minimize(fun, 1, bounds=((0, None),))
		beta = res.x[0]
		alpha = m1 / gamma_func(1 + 1./beta)
		return mu, sigma

	def _mps(self, x, c=None, n=None):
		"""
		MPS: Maximum Product Spacing

		This is the method to get the largest (geometric) average distance between all points

		This method works really well when all points are unique. Some complication comes in when using repeated data.

		This method is exceptional for when using three parameter distributions.
		"""
		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : self.neg_mean_D(x, t[0], t[1])
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

	def _mle(self, x, c=None, n=None, model=None):
		"""
		MLE: Maximum Likelihood estimate

		This is the MLE, the king of parameter estimation.
		"""
		if n is None:
			n = np.ones_like(x)
		if c is None:
			c = np.zeros_like(x)

		#init = self._mpp(x, c=c, n=n, on_d_is_0=True)
		init = np.mean(x), 1
		bounds = ((None, None), (0, None))
		fun = lambda t : self.neg_ll(x, t[0], t[1], c, n)
		res = minimize(fun, init)

		if model is not None:
			model.res = res
		return res.x[0], res.x[1]

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		model = Parametric()
		model.method = how
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}
		model.dist = "LogNormal"
		model.dist = self
		if   how == 'MLE':
			# Maximum Likelihood
			params = Normal._mle(np.log(x), c=c, n=n, model=model)
		elif how == 'MPS':
			# Maximum Product Spacing
			if c is not None:
				raise InputError('Maximum product spacing doesn\'t support censoring')
			if n is not None:
				raise InputError('Maximum product spacing doesn\'t support counts')
			params = self._mps(x)
		elif how == 'MOM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			params = self._mom(x, n=n)
		elif how == 'MPP':
			heuristic = 'Nelson-Aalen'
			if c is not None:
				heuristic = 'Nelson-Aalen'
			if 'rr' in kwargs:
				rr = kwargs['rr']
			else:
				rr = 'y'
			params = self._mpp(x, n=n, c=c, rr=rr, heuristic=heuristic)
		elif how == 'MSE':
			params = self._mse(x, c=c, n=n)
		# Store params with redundancy...
		model.mu, model.sigma = params
		model.params = params
		return model
class WMM(): 
	def Q_prime(self, params): 
		tmp_params = params.reshape(self.n, 2) 
		f = np.zeros_like(self.p) 
		for i in range(self.n): 
			like = np.log(Weibull.pdf(self.x, tmp_params[0, i], tmp_params[1, i])) 
			mask = np.isinf(like) 
			if any(mask): 
				#print(np.sum(like[~np.isneginf(like)])) 
				like[np.isneginf(like)] = MAX_LOG 
				#print('max\'d') 
			f[i] = np.multiply(self.p[i], like) 
			f = -np.sum(f) 
			return f 

	def expectation(self): 
		for i in range(self.n): 
			self.p[i] = self.w[i] * Weibull.pdf(self.x, self.alphas[i], self.betas[i]) 
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

	def __str__(self): 
		print(self.alphas) 
		print(self.betas) 
		print(self.w) 
		return "Done" 

	def pdf(self, t): 
		f = np.zeros_like(t) 
		for i in range(self.n): 
			f += self.w[i] * Weibull.pdf(t, self.alphas[i], self.betas[i]) 
		return f 

	def cdf(self, t): 
		F = np.zeros_like(t) 
		for i in range(self.n): 
			F += self.w[i] * Weibull.cdf(t, self.alphas[i], self.betas[i]) 
		return F 

	def __init__(self, **kwargs): 
		assert 'data' in kwargs 
		self.n = kwargs.pop('n', 2) 
		self.x = np.sort(kwargs.pop('data')) 
		self.N = len(self.x) 
		self.alphas = [np.mean(x) for x in np.array_split(self.x, self.n)] 
		self.betas = [1.] * self.n 
		self.w = np.ones(shape=(self.n)) / self.n 
		self.p = np.ones(shape=(self.n, len(self.x))) / self.n 
	
Exponential = Exponential_('Exponential')
Weibull = Weibull_('Weibull')
Gumbel = Gumbel_('Gumbel')
Weibull3p = Weibull3p_('Weibull3p')
Normal = Normal_('Normal')
LogNormal = LogNormal_('LogNormal')

