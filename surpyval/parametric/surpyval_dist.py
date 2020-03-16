from autograd import jacobian, hessian
import autograd.numpy as np
from autograd.numpy.linalg import inv

from scipy.optimize import minimize
from scipy.optimize import approx_fprime

from surpyval import nonparametric as nonp
from surpyval import parametric as para

NUM     = np.float64
TINIEST = np.finfo(NUM).tiny
EPS     = np.sqrt(np.finfo(NUM).eps)

class SurpyvalDist():
	def neg_ll(self, x, c=None, n=None, *params):
		# Use this neg_ll, will make it much easier to implement interval cens
		like = np.zeros_like(x).astype(NUM)
		like = np.where(c == 0, self.df(x, *params), like)
		like = np.where(c == -1, self.ff(x, *params), like)
		like = np.where(c == 1, self.sf(x, *params), like)
		like += TINIEST
		like = np.where(like < 1, like, 1)
		return -np.sum(np.log(like))

	def neg_mean_D(self, x, c=None, n=None, *params):
		idx = np.argsort(x)
		F  = self.ff(x[idx], *params)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])
		if c is not None:
			Dr = self.sf(x[c == 1], *params)
			Dl = self.ff(x[c == -1], *params)
			D = np.concatenate([Dl, D, Dr])
		D[D < TINIEST] = TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		return M

	def mom_moment_gen(self, *params):
		moments = np.zeros(self.k)
		for i in range(0, self.k):
			n = i + 1
			moments[i] = self.moment(n, *params)
		return moments

	def _mse(self, x, c=None, n=None, heuristic='Nelson-Aalen'):
		"""
		MSE: Mean Square Error
		This is simply fitting the curve to the best estimate from a non-parametric estimate.

		This is slightly different in that it fits it to untransformed data.
		The transformation is how the "Probability Plotting Method" works.

		Fit a two parameter Weibull distribution from data
		
		Fits a Weibull model to pp points 
		"""
		x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)
		init = self.parameter_initialiser(x, c=c, n=n)
		fun = lambda t : np.sum(((self.ff(x, *t)) - F)**2)
		res = minimize(fun, init, bounds=self.bounds)
		self.res = res
		return res

	def _mps(self, x, c=None, n=None):
		"""
		MPS: Maximum Product Spacing

		This is the method to get the largest (geometric) average distance between all points

		This method works really well when all points are unique. Some complication comes in when using repeated data.

		This method is exceptional for when using three parameter distributions.
		"""
		init = self.parameter_initialiser(x, c=c, n=n)
		bounds = self.bounds
		fun = lambda t : self.neg_mean_D(x, c, n, *t)
		res = minimize(fun, init, bounds=bounds)
		return res

	def _mle(self, x, c=None, n=None):
		"""
		MLE: Maximum Likelihood estimate
		"""
		if n is None:
			n = np.ones_like(x).astype(np.int64)

		if c is None:
			c = np.zeros_like(x).astype(np.int64)

		x_ = np.copy(x)
		c_ = np.copy(c).astype(np.int64)
		n_ = np.copy(n).astype(np.int64)

		init = self.parameter_initialiser(x_, c_, n_)

		if self.use_autograd:
			try:
				fun  = lambda t : self.neg_ll(x_, c_, n_, *t)
				jac = jacobian(fun)
				hess = hessian(fun)
				res = minimize(fun, init, method='trust-exact', jac=jac, hess=hess, tol=1e-10)
				hess_inv = inv(res.hess)
			except:
				with np.errstate(all='ignore'):
					fun = lambda t : self.neg_ll(x_, c_, n_, *t)
					jac = lambda t : approx_fprime(t, fun, EPS)
					res = minimize(fun, init, method='BFGS', jac=jac)
					hess_inv = res.hess_inv

		else:
			fun = lambda t : self.neg_ll(x_, c_, n_, *t)
			jac = lambda t : approx_fprime(t, fun, EPS)
			#hess = lambda t : approx_fprime(t, jac, eps)
			res = minimize(fun, init, method='BFGS', jac=jac)
			hess_inv = res.hess_inv

		return res, jac, hess_inv

	def _mom(self, x, n=None):
		"""
		MOM: Method of Moments.

		This is one of the simplest ways to calculate the parameters of a distribution.

		This method is quick but only works with uncensored data.
		# Can I add a simple sum(c) instead of length to work with censoring?
		"""
		if n is not None:
			x = np.repeat(x, n)

		moments = np.zeros(self.k)
		for i in range(0, self.k):
			moments[i] = np.sum(x**(i+1)) / len(x)

		fun = lambda t : np.sum((moments - self.mom_moment_gen(*t))**2)
		res = minimize(fun, 
					   self.parameter_initialiser(x), 
					   bounds=self.bounds)
		return res

	def _mpp(self, x, c=None, n=None, heuristic="Nelson-Aalen", rr='y', on_d_is_0=False):
		assert rr in ['x', 'y']
		"""
		MPP: Method of Probability Plotting
		Yes, the order of this language was invented to keep MXX format consistent
		This is the classic probability plotting paper method.

		This method creates the plotting points, transforms it to Weibull scale and then fits the line of best fit.

		Fit a two parameter Weibull distribution from data.
		
		Fits a Weibull model using cumulative probability from x values. 
		"""
		x_, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)
		
		if not on_d_is_0:
			x_ = x_[d > 0]
			F = F[d > 0]

		# Linearise
		x_ = self.mpp_x_transform(x_)
		y_ = self.mpp_y_transform(F)

		if rr == 'y':
			params = np.polyfit(x_, y_, 1)
		elif rr == 'x':
			params = np.polyfit(y_, x_, 1)

		params = self.unpack_rr(params, rr)
		return params

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		x = np.array(x, dtype=NUM)
		assert x.ndim == 1

		model = para.Parametric()
		model.method = how
		model.raw_data = {
			'x' : x,
			'c' : c,
			'n' : n
		}

		if c is None:
			c = np.zeros_like(x).astype(np.int64)
		else:
			c = np.array(c, dtype=np.int64)

		assert c.ndim == 1
		assert c.shape == x.shape

		if n is None:
			n = np.ones_like(x).astype(np.int64)
		if n is None:
			n = np.array(n, dtype=np.int64)

		assert n.ndim == 1
		assert n.shape == x.shape

		model.data = {
			'x' : x,
			'c' : c,
			'n' : n
		}

		heuristic = kwargs.get('heuristic', 'Nelson-Aalen')
		model.heuristic = heuristic
		model.dist = self

		if   how == 'MLE':
			# Maximum Likelihood
			model.res, model.jac, model.hess_inv = self._mle(x, c=c, n=n)
			model.params = tuple(model.res.x)
		elif how == 'MPS':
			# Maximum Product Spacing
			if model.raw_data['c'] is not None:
				raise Exception('Maximum product spacing doesn\'t support censoring')
			if model.raw_data['c'] is not None:
				raise Exception('Maximum product spacing doesn\'t support counts')
			model.res = self._mps(x)
			model.params = tuple(model.res.x)
		elif how == 'MOM':
			if model.raw_data['c'] is not None:
				raise Exception('Method of moments doesn\'t support censoring')
			model.res = self._mom(x, n=n)
			model.params = tuple(model.res.x)
		elif how == 'MPP':
			rr = kwargs.get('rr', 'y')
			model.params = self._mpp(x, n=n, c=c, rr=rr, heuristic=heuristic)
		elif how == 'MSE':
			model.res = self._mse(x, c=c, n=n, heuristic=heuristic)
			model.params = tuple(model.res.x)
		
		return model