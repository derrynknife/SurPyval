import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z

from autograd import hessian
from autograd.numpy.linalg import pinv
from scipy.optimize import minimize

from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Uniform_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		# Set 'k', the number of parameters
		self.k = 2
		self.bounds = ((None, None), (None, None),)
		self.support = (-np.inf, np.inf)
		self.plot_x_scale = 'linear'
		self.y_ticks = np.linspace(0, 1, 21)[1:-1]
		self.param_names = ['a', 'b']
		self.param_map = {
			'a' : 0,
			'b' : 1
		}

	def parameter_initialiser(self, x, c=None, n=None):
		return np.min(x) - 1., np.max(x) + 1.

	def sf(self, x, a, b):
		return 1 - self.ff(x, a, b)

	def cs(self, x, X, a, b):
		return self.sf(x + X, a, b) / self.sf(X, a, b)

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

	def p(self, c, N):
		return 1 - 2 * (1 + c)**(1. - n) + (1 + 2*c)**(1. - n)

	def random(self, size, a, b):
		U = uniform.rvs(size=size)
		return self.qf(U, a, b)

	def ab_cb(self, x, a, b, N, alpha=0.05):
		# Parameter confidence intervals from here:
		# https://mathoverflow.net/questions/278675/confidence-intervals-for-the-endpoints-of-the-uniform-distribution
		#
		sample_range = np.max(x) - np.min(x)
		fun = lambda c : self.p(c, N)
		c_hat = minimize(fun, 1.).x
		return a - c_hat*sample_range, b + c_hat*sample_range

	def mle(self, x, c, n, t, const, trans, inv_fs, init, fixed_idx, offset):
		params = np.array([np.min(x), np.max(x)])
		return None, None, None, params

	def mpp_x_transform(self, x):
		return x

	def mpp_y_transform(self, y, *params):
		return y

	def mpp_inv_y_transform(self, y, *params):
		return y

Uniform = Uniform_('Uniform')