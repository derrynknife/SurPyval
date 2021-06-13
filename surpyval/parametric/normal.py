import autograd.numpy as np
from scipy.stats import uniform
from scipy.special import ndtri as z
from autograd.scipy.stats import norm
from scipy.stats import norm as scipy_norm
import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter

class Normal_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((None, None), (0, None),)
		self.support = (-np.inf, np.inf)
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 
				0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.999]
		self.param_names = ['mu', 'sigma']
		self.param_map = {
			'mu'    : 0,
			'sigma' : 1
		}

	def parameter_initialiser(self, x, c=None, n=None, t=None):
		return para.Normal.fit(x, c, n, t, how='MPP').params
		# x, c, n = surpyval.xcn_handler(x, c, n)
		# flag = (c == 0).astype(int)
		# return x.sum() / (n * flag).sum(), np.std(x)

	def sf(self, x, mu, sigma):
		return norm.sf(x, mu, sigma)

	def cs(self, x, X, mu, sigma):
		return self.sf(x + X, mu, sigma) / self.sf(X, mu, sigma)

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

	def mpp_y_transform(self, y, *params):
		return self.qf(y, 0, 1)

	def mpp_inv_y_transform(self, y, *params):
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

Normal = Normal_('Normal')