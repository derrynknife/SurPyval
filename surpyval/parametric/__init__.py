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
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

NUM     = np.float64
TINIEST = np.finfo(NUM).tiny
EPS     = np.sqrt(np.finfo(NUM).eps)

from surpyval import nonparametric as nonp
from surpyval.parametric.surpyval_dist import SurpyvalDist
from surpyval.parametric.parametric_dist import Parametric
from surpyval.parametric.weibull import Weibull
from surpyval.parametric.weibull3p import Weibull3p
from surpyval.parametric.gumbel import Gumbel
from surpyval.parametric.exponential import Exponential
from surpyval.parametric.normal import Normal
from surpyval.parametric.lognormal import LogNormal
from surpyval.parametric.gamma import Gamma

def round_sig(points, sig=2):
    places = sig - np.floor(np.log10(np.abs(points))) - 1
    output = []
    for p, i in zip(points, places):
        output.append(np.round(p, np.int(i)))
    return output


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
class Logistic_(SurpyvalDist):
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
			n = np.ones_like(x)

		if c is None:
			c = np.zeros_like(x)

		flag = (c == 0).astype(np.int)
		return x.sum() / (n * flag).sum(), 1.

	def sf(self, x, mu, sigma):
		return 1 - self.ff(x, mu, sigma)

	def ff(self, x, mu, sigma):
		z = (x - mu) / sigma
		return 1. / (1 + np.exp(-z))

	def df(self, x, mu, sigma):
		z = (x - mu) / sigma
		return np.exp(-z) / (sigma * (1 + np.exp(-z))**2)

	def hf(self, x, mu, sigma):
		return self.pdf(x, mu, sigma) / self.sf(x, mu, sigma)

	def Hf(self, x, mu, sigma):
		return -np.log(self.sf(x, mu, sigma))

	def qf(self, p, mu, sigma):
		return mu + sigma * np.log(p/(1 - p))

	def mean(self, mu, sigma):
		return mu

	def random(self, size, mu, sigma):
		U = uniform.rvs(size=size)
		return self.qf(U, mu, sigma)

	def mpp_x_transform(self, x):
		return x

	def mpp_y_transform(self, y):
		return -np.log(1./y - 1)

	def mpp_inv_y_transform(self, y):
		return 1./(np.exp(-y) + 1)

	def unpack_rr(self, params, rr):
		if   rr == 'y':
			sigma = 1/params[0]
			mu    = -sigma * params[1]
		elif rr == 'x':
			sigma  = 1./params[0]
			mu = np.exp(params[1] / (beta * params[0]))
		return mu, sigma
class LogLogistic_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((None, None), (0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]

	def parameter_initialiser(self, x, c=None, n=None):
		if n is None:
			n = np.ones_like(x)

		if c is None:
			c = np.zeros_like(x)

		flag = (c == 0).astype(np.int)
		return x.sum() / (n * flag).sum(), 2.

	def sf(self, x, alpha, beta):
		return 1 - self.ff(x, alpha, beta)

	def ff(self, x, alpha, beta):
		return 1. / (1 + (x/alpha)**-beta)

	def df(self, x, alpha, beta):
		return ((beta/alpha)*(x/alpha)**(beta-1.))/((1. + (x/alpha)**beta)**2.)

	def hf(self, x, alpha, beta):
		return self.pdf(x, alpha, beta) / self.sf(x, alpha, beta)

	def Hf(self, x, alpha, beta):
		return -np.log(self.sf(x, alpha, beta))

	def qf(self, p, alpha, beta):
		return alpha * (p/(1 - p))**(1./beta)

	def mean(self, alpha, beta):
		if beta > 1:
			return (alpha * np.pi / beta) / (np.sin(np.pi/beta))
		else:
			return np.nan

	def random(self, size, alpha, beta):
		U = uniform.rvs(size=size)
		return self.qf(U, alpha, beta)

	def mpp_x_transform(self, x):
		return np.log(x)

	def mpp_y_transform(self, y):
		return -np.log(1./y - 1)

	def mpp_inv_y_transform(self, y):
		return 1./(np.exp(-y) + 1)

	def unpack_rr(self, params, rr):
		if rr == 'y':
			beta  = params[0]
			alpha = np.exp(params[1]/-beta)
		elif rr == 'x':
			beta  = 1./params[0]
			alpha = np.exp(params[1] / (beta * params[0]))
		return alpha, beta

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

Uniform = Uniform_('Uniform')
Weibull_Mix_Two = Weibull_Mix_Two_('Weibull_Mix_Two')
Logistic = Logistic_('Logistic')
LogLogistic = LogLogistic_('LogLogistic')