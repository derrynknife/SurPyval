import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z
from scipy.optimize import minimize

from surpyval import parametric as para

TINIEST = np.finfo(np.float64).tiny

class MixtureModel():
	"""
	Generalised from algorithm found here
	https://www.sciencedirect.com/science/article/pii/S0307904X12002545
	"""
	def __init__(self, **kwargs): 
		assert 'x' in kwargs
		assert 'dist' in kwargs
		self.dist = kwargs.pop('dist', para.Weibull)
		self.m = kwargs.pop('m', 2)
		raw_data = {}
		x = kwargs.pop('x')
		c = kwargs.pop('c', None)
		n = kwargs.pop('n', None)
		raw_data['x'] = x
		raw_data['c'] = c
		raw_data['n'] = n
		self.raw_data = raw_data

		x = np.array(x)
		assert x.ndim == 1
		assert len(x) > self.m * (self.dist.k + 1)
		if c is not None:
			c = np.array(c)
			assert c.ndim == 1
			assert c.shape == x.shape
		else:
			c = np.zeros_like(x)

		if n is not None:
			n = np.array(n)
			assert n.ndim == 1
			assert n.shape == x.shape
		else:
			n = np.ones_like(x)

		self.x = x
		self.c = c
		self.n = n

		self.N = n.sum()
		splits_x = np.array_split(x, self.m)
		splits_c = np.array_split(c, self.m)
		splits_n = np.array_split(n, self.m)
		params = np.zeros(shape=(self.m, self.dist.k))

		for i in range(self.m):
			params[i, :] = self.dist.fit(x=splits_x[i], c=splits_c[i], 
										 n=splits_n[i]).params
		self.params = params
		self.w = np.ones(shape=(self.m)) / self.m
		self.p = np.ones(shape=(self.m, len(self.x))) / self.m

		self.method = 'EM'

	def Q(self, params):
		params = params.reshape(self.m, self.dist.k)
		f = np.zeros_like(self.p)
		for i in range(self.m):
			like = self.dist.like(self.x, self.c, self.n, *params[i])
			
			like += TINIEST
			like = np.where(like < 1, like, 1)
			like = np.log(like)
			like = np.multiply(self.n, like)

			f[i] = np.multiply(self.p[i], like) 
		f = -np.sum(f)
		self.loglike = f
		return f 

	def expectation(self): 
		for i in range(self.m):
			like = self.dist.like(self.x, self.c, self.n, *self.params[i])
			like = np.multiply(self.w[i], like)
			self.p[i] = like
		self.p = np.divide(self.p, np.sum(self.p, axis=0)) 
		self.w = np.sum(self.p, axis=1) / self.N 

	def maximisation(self): 
		bounds = self.dist.bounds * self.m
		res = minimize(self.Q, self.params, bounds=bounds)
		#if not res.success: 
		#	print(res) 
		#	raise Exception('Max failed') 
		self.params = res.x.reshape(self.m, self.dist.k)

	def EM(self):
		self.expectation() 
		self.maximisation()

	def _em(self, tol=1e-6, max_iter=1000):
		i = 0
		self.EM()
		f0 = self.loglike
		self.EM()
		f1 = self.loglike
		while (np.abs(f0 - f1) > tol) and (i < max_iter):
			f0 = f1
			self.EM()
			f1 = self.loglike
			i += 1
		if i >= 1000:
			print('Max iterations reached')

	def neg_ll(self, x, c, n, *params):
		f = np.zeros_like(self.p)
		params = np.reshape(params, (self.m, self.dist.k + 1))
		#params = params.reshape(self.m, self.dist.k + 1)
		for i in range(self.m):
			like = self.dist.like(x, c, n, *params[i, 1::])
			like = np.multiply(params[i, 0], like)
			like += TINIEST
			like = np.where(like < 1, like, 1)
			f[i] = like
		f = np.sum(f, axis=0)
		f = np.log(f)
		f = np.multiply(n, f)
		f = -np.sum(f)
		return f


	def _mle(self):
		"""
		What I've learned from this is that EM is way better. Like way better.
		So do not use.
		"""
		fun = lambda x : self.neg_ll(self.x, self.c, self.n, *x)
		bounds = tuple(((0, 1), *self.dist.bounds)) * self.m
		init = np.hstack([np.atleast_2d(self.w).T, self.params])
		def w_sums_to_1(x):
			xx = np.reshape(x, (self.m, self.dist.k + 1))
			return xx[:, 0].sum() - 1
		cons = {'type':'eq', 'fun': w_sums_to_1}
		res = minimize(fun, init.flatten(), bounds=bounds, constraints=[cons])
		self.res = res

	def fit(self, how='EM'):
		if how == 'EM':
			self._em()
		else:
			print('Only EM available now')

	def ll(self):
		return self.loglike

	def mean(self):
		mean = 0
		for i in range(self.m):
			mean += self.w[i] * self.dist.mean(*self.params[i])
		return mean

	def df(self, t): 
		f = np.zeros_like(t) 
		for i in range(self.m): 
			f += self.w[i] * self.dist.df(t, *self.params[i]) 
		return f 

	def ff(self, t): 
		F = np.zeros_like(t) 
		for i in range(self.m): 
			F += self.w[i] * self.dist.ff(t, *self.params[i]) 
		return F

	def sf(self, t): 
		return 1 - self.ff(t)


