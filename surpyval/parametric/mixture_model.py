import autograd.numpy as np
from scipy.linalg import inv
from scipy.stats import uniform
from scipy.optimize import minimize
from scipy.special import ndtri as z

from autograd import jacobian, hessian
import surpyval
from surpyval import parametric as para

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

		x, c, n = surpyval.xcn_handler(x, c, n)
		assert len(x) > self.m * (self.dist.k + 1)

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

	def Q(self, params):
		params = params.reshape(self.m, self.dist.k)
		f = np.zeros_like(self.p)
		for i in range(self.m):
			like = self.dist.like(self.x, self.c, self.n, *params[i])
			like += surpyval.TINIEST
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
		"""
		Handles autograd
		"""
		f = np.zeros_like(self.p)
		params = np.reshape(params, (self.m, self.dist.k + 1))
		f = np.zeros_like(x)
		for i in range(self.m):
			like = self.dist.like(x, c, n, *params[i, 1::])
			like = np.multiply(params[i, 0], like)
			f = f + like
		f = np.where(f <= 0, surpyval.TINIEST, f)
		f = np.where(f < 1, f, 1)
		f = np.log(f)
		f = np.multiply(n, f)
		f = -np.sum(f)
		return f

	def _mle(self):
		fun = lambda x : self.neg_ll(self.x, self.c, self.n, *x)
		bounds = tuple(((0, 1), *self.dist.bounds)) * self.m
		init = np.hstack([np.atleast_2d(self.w).T, self.params])
		def w_sums_to_1(x):
			xx = np.reshape(x, (self.m, self.dist.k + 1))
			return xx[:, 0].sum() - 1
		cons = {'type':'eq', 'fun': w_sums_to_1}
		jac = jacobian(fun)
		hess= hessian(fun)
		res = minimize(fun, init.flatten(), jac=jac, bounds=bounds, constraints=[cons])
		self.hess_inv = inv(hess(res.x))
		self.res = res
		self.params = np.reshape(res.x, (self.m, self.dist.k + 1))[:, 1::]
		self.w = np.reshape(res.x, (self.m, self.dist.k + 1))[:, 0]

	def fit(self, how='EM'):
		if how == 'EM':
			self._em()
			self.method = 'EM'
		elif how == 'MLE':
			self._mle()
			self.method = 'MLE'

	def ll(self):
		return self.loglike

	def R_cb(self, t, cb=0.05):
		"""
		Nailed this. Can be used elsewhere if needed
		"""
		def ssf(params):
			params = np.reshape(params, (self.m, self.dist.k + 1))
			F = np.zeros_like(t) 
			for i in range(self.m): 
				F = F + params[i, 0] * self.dist.ff(t, *params[i, 1::]) 
			return 1 - F

		pvars = self.hess_inv[np.triu_indices(self.hess_inv.shape[0])]
		with np.errstate(all='ignore'):
			jac = jacobian(ssf)(self.res.x)
			
		var_u = []
		for i, j in enumerate(jac):
			j = np.atleast_2d(j).T * j
			j = j[np.triu_indices(j.shape[0])] 
			var_u.append(np.sum(j * pvars))
		diff = z(cb/2) * np.sqrt(np.array(var_u)) * np.array([1., -1.]).reshape(2, 1)
		R_hat = self.sf(t)
		exponent = diff/(R_hat*(1 - R_hat))
		R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
		return R_cb.T

	def mean(self):
		mean = 0
		for i in range(self.m):
			mean += self.w[i] * self.dist.mean(*self.params[i])
		return mean

	def random(self, size):
		sizes = np.random.multinomial(size, self.w)
		rvs = np.zeros(size)
		s_last = 0
		for i, s in enumerate(sizes):
			rvs[s_last:s+s_last] = self.dist.random(s, *self.params[i, :])
			s_last = s
		np.random.shuffle(rvs)
		return rvs

	def df(self, t): 
		f = np.zeros_like(t) 
		for i in range(self.m): 
			f += self.w[i] * self.dist.df(t, *self.params[i]) 
		return f 

	def ff(self, t): 
		F = np.zeros_like(t) 
		for i in range(self.m): 
			F = F + self.w[i] * self.dist.ff(t, *self.params[i]) 
		return F

	def sf(self, t): 
		return 1 - self.ff(t)

	def cs(self, t, T): 
		return self.sf(t + T) / self.sf(T)


