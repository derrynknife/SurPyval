import numpy as np 
from matplotlib import pyplot as plt 
from scipy.optimize import minimize 
from scipy.stats.distributions import weibull_min 
from nonparametric import plotting_positions as pp 
import WeibullScale

MAX_LOG = np.log(np.finfo(np.float64).tiny) 

def weib_scale(x): 
	return np.log(np.log(1/(1-x))) 

class Weibull(): 
	def __init__(self, **kwargs): 
		self.x = kwargs.pop('data') 

	@classmethod 
	def sf(self, x, alpha, beta):
		return np.exp(-np.power(np.divide(x, alpha), beta)) 

	@classmethod 
	def cdf(self, x, alpha, beta): 
		return 1 - self.sf(x, alpha, beta) 

	@classmethod 
	def pdf(self, x, alpha, beta): 
		one = (np.multiply(np.divide(beta, alpha), np.power(np.divide(x, alpha), beta - 1))) 
		two = self.sf(x, alpha, beta) 
		return one * two 

	def ll(self, x, alpha, beta): 
		return np.sum(np.log(self.pdf(x, alpha, beta))) 

	def fit(self): 
		alpha0 = np.mean(self.x) 
		beta0 = 1. 
		params = np.array([alpha0, beta0]) 
		fun = lambda x : -self.ll(self.x, x[0], x[1]) 
		bounds = ((0, None), (0, None)) 
		res = minimize(fun, params, bounds=bounds) 
		if res.success: 
			self.alpha = res.x[0] 
			self.beta = res.x[1] 
		else: 
			raise Exception('No good on Max step') 

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






