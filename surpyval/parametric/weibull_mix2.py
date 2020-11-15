import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import gamma as gamma_func
from scipy.special import ndtri as z

import surpyval
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter
from surpyval.parametric.parametric import Parametric

class Weibull_Mix_Two_(ParametricFitter):
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
		self.param_names = ['alpha1', 'beta1', 'alpha2', 'beta2']
		self.param_map = {
			'alpha1' : 0,
			'beta1'  : 1,
			'alpha2' : 2,
			'beta2'  : 3
		}

	def parameter_initialiser(self, x, c=None, n=None):
		return np.mean(x), 1.

	def df(self, x, n, w, alphas, betas): 
		f = np.zeros_like(x) 
		for i in range(self.n): 
			f += w[i] * para.Weibull.df(x, alphas[i], betas[i]) 
		return f 

	def ff(self, x, n, w, alphas, betas): 
		F = np.zeros_like(x) 
		for i in range(self.n): 
			F += w[i] * para.Weibull.ff(x, alphas[i], betas[i]) 
		return F

	def sf(self, x, n, w, alphas, betas):
		return 1 - self.ff(x, n, w, alphas, betas)

	def cs(self, x, X, n, w, alphas, betas):
		return self.sf(x + X, n, w, alphas, betas) / self.sf(X, n, w, alphas, betas)

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

	def fit(self, x, c=None, n=None):
		model = Parametric()

		model.method = 'EM'
		model.dist = self
		model.raw_data = {
			'x' : x,
			'c' : None,
			'n' : None
		}

		x, c, n = surpyval.xcn_handler(x, c, n)

		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}

		wmm = para.WMM(data=x)
		wmm.fit()
		model.log_like = wmm.ll()

		model.params = tuple((self.n, wmm.w, wmm.alphas, wmm.betas))

		return model

Weibull_Mix_Two = Weibull_Mix_Two_('Weibull_Mix_Two')