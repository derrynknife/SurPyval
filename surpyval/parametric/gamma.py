import autograd.numpy as np
from scipy.stats import uniform
from scipy.optimize import approx_fprime
from scipy.special import gamma as gamma_func
from scipy.special import gammainc, gammaincinv
from autograd_gamma import gammainc as agammainc
from scipy.special import ndtri as z

from surpyval import parametric as para
from surpyval.parametric.surpyval_dist import SurpyvalDist

class Gamma_(SurpyvalDist):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((0, None), (0, None),)
		self.use_autograd = True
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
		return agammainc(alpha, beta * x)

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
		return agammainc(alpha, y)

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
		jac = lambda t : approx_fprime(t, dR_f, para.EPS)[1::]
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

Gamma = Gamma_('Gamma')