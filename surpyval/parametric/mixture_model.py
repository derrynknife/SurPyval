import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z
from scipy.optimize import minimize

from surpyval import parametric as para
from surpyval.parametric.surpyval_dist import SurpyvalDist

TINIEST = np.finfo(np.float64).tiny

class MixtureModel(): 
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

	def Q_prime(self, params): 
		tmp_params = params.reshape(self.m, self.dist.k) 
		f = np.zeros_like(self.p) 
		for i in range(self.m):
			like = np.zeros_like(self.x).astype(np.float64)
			like = np.where(self.c == 0, self.dist.df(self.x, *tmp_params[i, :]), like)
			like = np.where(self.c == -1, self.dist.ff(self.x, *tmp_params[i, :]), like)
			like = np.where(self.c == 1, self.dist.sf(self.x, *tmp_params[i, :]), like)
			like += TINIEST
			like = np.where(like < 1, like, 1)
			like = np.log(like)
			like = np.multiply(self.n, like) 

			#like = np.log(para.Weibull.df(self.x, *tmp_params[i, :])) 
			f[i] = np.multiply(self.p[i], like) 
		f = -np.sum(f)
		self.loglike = f
		return f 

	def expectation(self): 
		for i in range(self.m):
			like = np.zeros_like(self.x).astype(np.float64)
			like = np.where(self.c ==  0, self.dist.df(self.x, *self.params[i, :]), like)
			like = np.where(self.c == -1, self.dist.ff(self.x, *self.params[i, :]), like)
			like = np.where(self.c ==  1, self.dist.sf(self.x, *self.params[i, :]), like)
			like = np.multiply(self.w[i], like)
			self.p[i] = like
		self.p = np.divide(self.p, np.sum(self.p, axis=0)) 
		self.w = np.sum(self.p, axis=1) / self.N 

	def maximisation(self): 
		bounds = self.dist.bounds * self.m
		res = minimize(self.Q_prime, self.params, bounds=bounds) 
		if not res.success: 
			print(res) 
			raise Exception('Max failed') 
		self.params = res.x.reshape(self.m, self.dist.k)

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
			f += self.w[i] * para.Weibull.df(t, self.alphas[i], self.betas[i]) 
		return f 

	def ff(self, t): 
		F = np.zeros_like(t) 
		for i in range(self.n): 
			F += self.w[i] * para.Weibull.ff(t, self.alphas[i], self.betas[i]) 
		return F

	def sf(self, t): 
		return 1 - self.ff