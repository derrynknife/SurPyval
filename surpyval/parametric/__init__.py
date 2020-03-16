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

from surpyval import nonparametric as nonp
from surpyval.parametric.surpyval_dist import SurpyvalDist
from surpyval.parametric.parametric_dist import Parametric
from surpyval.parametric.weibull import Weibull
from surpyval.parametric.gumbel import Gumbel

NUM     = np.float64
TINIEST = np.finfo(NUM).tiny
EPS     = np.sqrt(np.finfo(NUM).eps)

def round_sig(points, sig=2):
    places = sig - np.floor(np.log10(np.abs(points))) - 1
    output = []
    for p, i in zip(points, places):
        output.append(np.round(p, np.int(i)))
    return output

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

Exponential = Exponential_('Exponential')
Gamma = Gamma_('Gamma')
Weibull3p = Weibull3p_('Weibull3p')
Normal = Normal_('Normal')
LogNormal = LogNormal_('LogNormal')
Uniform = Uniform_('Uniform')
Weibull_Mix_Two = Weibull_Mix_Two_('Weibull_Mix_Two')
Logistic = Logistic_('Logistic')
LogLogistic = LogLogistic_('LogLogistic')

