import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z

from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Uniform_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		# Set 'k', the number of parameters
		self.k = 2
		self.bounds = ((None, None), (None, None),)
		self.use_autograd = True
		self.plot_x_scale = 'linear'
		self.y_ticks = np.linspace(0, 1, 21)[1:-1]
		self.param_names = ['a', 'b']

	def parameter_initialiser(self, x, c=None, n=None):
		return np.min(x), np.max(x)

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

	def random(self, size, a, b):
		U = uniform.rvs(size=size)
		return self.qf(U, a, b)

	def mpp_x_transform(self, x):
		return x

	def mpp_y_transform(self, y):
		return y

	def mpp_inv_y_transform(self, y):
		return y

Uniform = Uniform_('Uniform')