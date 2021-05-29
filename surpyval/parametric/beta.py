import types
import autograd.numpy as np
from autograd import jacobian
from scipy.stats import uniform
from scipy.optimize import approx_fprime
from scipy.special import betainc, betaincinv
from scipy.special import gamma as gamma_func
from scipy.special import gammainc, gammaincinv
from autograd_gamma import gammainc as agammainc
from autograd.scipy.special import beta as abeta
from autograd_gamma import betainc as abetainc
from scipy.special import ndtri as z

from scipy.optimize import minimize
from scipy.stats import pearsonr

import surpyval
from surpyval import parametric as para
from surpyval import nonparametric as nonp
from surpyval.parametric.parametric_fitter import ParametricFitter

class Beta_(ParametricFitter):
	def __init__(self, name):
		self.name = name
		self.k = 2
		self.bounds = ((0, None), (0, None),)
		self.support = (0, 1)
		self.plot_x_scale = 'linear'
		self.y_ticks = [0.0001, 0.0002, 0.0003, 0.001, 0.002, 
			0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 
			0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
			0.9, 0.95, 0.99, 0.999, 0.9999]
		self.param_names = ['alpha', 'beta']
		self.param_map = {
			'alpha' : 0,
			'beta'  : 1
		}

	def parameter_initialiser(self, x, c=None, n=None):
		if (c is not None) & ((c == 0).all()):
			p = tuple(self._mom(x, n, [1., 1.]).x)
		else:
			p = 1., 1.
		return p

	def sf(self, x, alpha, beta):
		return 1 - self.ff(x, alpha, beta)

	def cs(self, x, X, alpha, beta):
		return self.sf(x + X, alpha, beta) / self.sf(X, alpha, beta)

	def ff(self, x, alpha, beta):
		return abetainc(alpha, beta, x)

	def df(self, x, alpha, beta):
		return (x**(alpha - 1) * (1 - x)**(beta - 1)) / abeta(alpha, beta)

	def hf(self, x, alpha, beta):
		return self.df(x, alpha, beta) / self.sf(x, alpha, beta)

	def Hf(self, x, ahlpa, beta):
		return -np.log(self.sf(x, alpha, beta))

	def qf(self, p, alpha, beta):
		return betaincinv(alpha, beta, p)

	def mean(self, alpha, beta):
		return alpha / (alpha + beta)

	def moment(self, n, alpha, beta):
		return gamma_func(n + alpha) / (beta**n * gamma_func(alpha))

	def random(self, size, alpha, beta):
		U = uniform.rvs(size=size)
		return self.qf(U, alpha, beta)

	def mpp_y_transform(self, y, alpha, beta):
		return self.qf(y, alpha, beta)

	def mpp_inv_y_transform(self, y, alpha, beta):
		return abetainc(y, alpha, beta)

	def mpp_x_transform(self, x, gamma=0):
		return x - gamma

	def _mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y', on_d_is_0=False):
		raise NotImplementedError("Probability Plotting Method for Beta distribution")

	def _mom(self, x, n, init):
		"""
		MOM: Method of Moments for the beta distribution has an analytic answer.
		"""
		mean = np.repeat(x, n).mean()
		var = np.repeat(x, n).var()
		term1 = ((mean*(1 - mean)/var) - 1)
		alpha =  term1 * mean
		beta = term1 * (1 - mean)

		# Need to keep fitter constant
		res = types.SimpleNamespace()
		res.x = [alpha, beta]
		return res

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
		jac = jacobian(dR_f)
		#jac = lambda t : approx_fprime(t, dR_f, surpyval.EPS)[1::]
		x_ = np.array(x)
		if x_.size == 1:
			dR = jac(np.array((x_, alpha, beta))[1::])
			dR = dR.reshape(1, 2)
		else:
			out = []
			for xx in x_:
				out.append(jac(np.array((xx, alpha, beta)))[1::])
			dR = np.array(out)
		K = z(cb/2)
		exponent = K * np.array([-1, 1]).reshape(2, 1) * np.sqrt(self.var_R(dR, cv_matrix))
		exponent = exponent/(R_hat*(1 - R_hat))
		R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
		return R_cb.T

Beta = Beta_('Beta')
