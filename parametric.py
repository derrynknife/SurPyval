import numpy as np
from numpy import euler_gamma
from .errors import InputError
from scipy.special import gamma
from scipy.optimize import minimize
from scipy.special import ndtri as z
from .nonparametric import plotting_positions

TINIEST = np.finfo(np.float64).tiny

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
			self.log_like = self.dist.neg_ll(x, *self.params, c=c, n=n)
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
			 dist=dist, model=None):
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
		return alpha * gamma(1 + 1/beta)

	def moment(self, n, alhpa, beta):
		return alpha**n * gamma(1 + n/beta)

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

		F = plotting_positions(x, c=c, n=n, heuristic=heuristic)
		x = np.unique(x)

		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : np.sum(((self.ff(x, t[0], t[1])) - F)**2)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

	def _ppm(self, x, c=None, n=None,
			heuristic="Blom", rr='y'):
		assert rr in ['x', 'y']
		"""
		PPM: Probability Plotting Method
		This is the classif probability plotting paper method.

		This method creates the plotting points, transforms it to Weibull scale and then fits the line of best fit.

		Fit a two parameter Weibull distribution from data.
		
		Fits a Weibull model using cumulative probability from x values. 
		"""
		pp = plotting_positions(x, c=c, n=n,
							    heuristic=heuristic)
		
		# Linearise
		x_ = np.log(x)
		y_ = np.log(np.log(1/(1 - pp)))

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
		MPS: Maximum Product Spaceing

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
		init = [np.mean(x), 1.]
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

	def jacobian(x, alpha, beta, c=None, n=None):
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
			if c is not None:
				raise InputError('Maximum product spacing doesn\'t support censoring')
			if n is not None:
				raise InputError('Maximum product spacing doesn\'t support counts')
			params = self._mps(x)
		elif how == 'MOM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			params = self._mom(x, n=n)
		elif how == 'PPM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			if 'rr' in kwargs:
				rr = kwargs['rr']
			else:
				rr = 'x'
			params = self._lsm(x, rr=rr)
		elif how == 'MSE':
			params = self._mse(x, c=c, n=n)
		# Store params with redundancy...
		model.alpha, model.beta  = params
		model.params = params
		return model

class Weibull3p_():
	"""
	class for the three parameter weibull distribution.
	"""
	def __init__(self, name):
		self.name = name

	def sf(self, x, alpha, beta, gamma):
		return np.exp(-((x - gamma)/ alpha)**beta)

	def ff(self, x, alpha, beta, gamma):
		return 1 - np.exp(-((x-gamma) / alpha)**beta)

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
		return alpha * gamma(1 + 1/beta) + gamma

	def moment(self, n, alhpa, beta, gamma):
		return alpha**n * gamma(1 + n/beta) + gamma

	def random(self, size, alpha, beta):
		U = np.random.uniform(size=size)
		return self.qf(U, alpha, beta, gamma)

	def neg_mean_D(self, x, alpha, beta, gamma, c=None, n=None):
		idx = np.argsort(x)
		F  = self.ff(x[idx], alpha, beta)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if c is not None:
			Dr = self.sf(x[c == 1],  alpha, beta, gamma)
			Dl = self.ff(x[c == -1], alpha, beta, gamma)
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
		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""

		F = plotting_positions(x, c=c, n=n, heuristic=heuristic)
		x = np.unique(x)

		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None), (None, np.min(x)))
		fun = lambda t : np.sum(((self.ff(x, t[0], t[1], t[2])) - F)**2)
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1], res.x[2]

	def _lsm(self, x, c=None, n=None,
			heuristic="Blom", rr='y'):
		assert rr in ['x', 'y']
		"""
		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model using cumulative probability from x values. 
		"""
		pp = plotting_positions(x, c=c, n=n,
							    heuristic=heuristic)
		
		# Linearise
		x_ = np.log(x)
		y_ = np.log(np.log(1/(1 - pp)))

		if   rr == 'y':
			model = np.polyfit(x_, y_, 1)
			beta  = model[0]
			alpha = np.exp(model[1]/-beta)
		elif rr == 'x':
			model = np.polyfit(y_, x_, 1)
			beta  = 1./model[0]
			alpha = np.exp(model[1] / (beta * model[0]))
		return alpha, beta, gamma

	def _mps(self, x, c=None, n=None):
		init = [np.mean(x), 1.]
		bounds = ((0, None), (0, None))
		fun = lambda t : self.neg_mean_D(x, t[0], t[1])
		res = minimize(fun, init, bounds=bounds)
		return res.x[0], res.x[1]

	def _mle(self, x, c=None, n=None, model=None):
		init = [np.mean(x), 1.]
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

	def jacobian(x, alpha, beta, c=None, n=None):
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
			if c is not None:
				raise InputError('Maximum product spacing doesn\'t support censoring')
			if n is not None:
				raise InputError('Maximum product spacing doesn\'t support counts')
			params = self._mps(x)
		elif how == 'MOM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			params = self._mom(x, n=n)
		elif how == 'LSM':
			if c is not None:
				raise InputError('Method of moments doesn\'t support censoring')
			if 'rr' in kwargs:
				rr = kwargs['rr']
			else:
				rr = 'x'
			params = self._lsm(x, rr=rr)
		elif how == 'MSE':
			params = self._mse(x, c=c, n=n)
		# Store params with redundancy...
		model.alpha, model.beta  = params
		model.params = params
		return model

	
Weibull = Weibull_('Weibull')
Weibull3p = Weibull3p_('Weibull3p')
