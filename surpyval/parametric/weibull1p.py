import autograd.numpy as np
from scipy.stats import uniform
from numpy import euler_gamma
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z

from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Weibull1p_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		# Set 'k', the number of parameters
		self.k = 1
		self.bounds = ((0, None),)
		self.use_autograd = True
		self.plot_x_scale = 'log'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['alpha', 'beta']

	def parameter_initialiser(self, x, c=None, n=None):
		x, c, n = surpyval.xcn_handler(x, c, n)
		flag = (c == 0)
		return (n * x).sum() / (n * flag).sum()

	def sf(self, x, alpha):
		return np.exp(-(x / alpha)**self.beta)

	def ff(self, x, alpha):
		return 1 - np.exp(-(x / alpha)**self.beta)

	def cs(self, x, X, alpha):
		return self.sf(x + X, alpha, self.beta) / self.sf(X, alpha, self.beta)

	def df(self, x, alpha):
		return (self.beta / alpha) * (x / alpha)**(self.beta-1) * np.exp(-(x / alpha)**self.beta)

	def hf(self, x, alpha):
		return (self.beta / alpha) * (x / alpha)**(self.beta - 1)

	def Hf(self, x, alpha):
		return (x / alpha)**self.beta

	def qf(self, p, alpha):
		return alpha * (-np.log(1 - p))**(1/self.beta)

	def mean(self, alpha):
		return alpha * gamma_func(1 + 1./self.beta)

	def moment(self, n, alpha):
		return alpha**n * gamma_func(1 + n/self.beta)

	def entropy(self, alhpa):
		return euler_gamma * (1 - 1/self.beta) + np.log(alpha / self.beta) + 1

	def random(self, size, alpha):
		U = uniform.rvs(size=size)
		return self.qf(U, alpha, self.beta)

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

Weibull1p = Weibull1p_('Weibull1p')