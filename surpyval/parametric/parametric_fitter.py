from autograd import jacobian, hessian
import autograd.numpy as np
from autograd.numpy.linalg import inv

from scipy.optimize import minimize
from scipy.optimize import approx_fprime

import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para

import pandas as pd

class ParametricFitter():
	def parameter_initialiser_(self, x, c=None, n=None):
		x, c, n, R = nonp.turnbull(x, c, n, estimator='Nelson-Aalen')
		F = 1 - R
		F = F[~np.isinf(x)]
		x = x[~np.isinf(x)]
		init = np.ones(self.k)
		fun = lambda params : np.sum(((self.ff(x, *params)) - F)**2)
		res = minimize(fun, init, bounds=self.bounds)

		return res.x

	def like_with_interval(self, x, c=None, n=None, *params):
		xl = x[:, 0]
		xr = x[:, 1]
		like = np.zeros_like(xl).astype(surpyval.NUM)
		like_i = np.zeros_like(xl).astype(surpyval.NUM)
		like = np.where(c ==  0, self.df(xl, *params), like)
		like = np.where(c == -1, self.ff(xl, *params), like)
		like = np.where(c ==  1, self.sf(xl, *params), like)
		like_i = np.where(c ==  2, self.ff(xr, *params) - self.ff(xl, *params), like_i)
		return like + like_i

	def like(self, x, c=None, n=None, *params):
		like = np.zeros_like(x).astype(surpyval.NUM)
		like = np.where(c ==  0, self.df(x, *params), like)
		like = np.where(c == -1, self.ff(x, *params), like)
		like = np.where(c ==  1, self.sf(x, *params), like)
		return like

	def like_t(self, t, *params):
		tl = t[:, 0]
		tr = t[:, 1]
		t_denom = self.ff(tr, *params) - self.ff(tl, *params)
		return t_denom

	def neg_ll_trunc(self, x, c=None, n=None, t=None, *params):
		# This is going to get difficult
		if 2 in c:
			like = self.like_with_interval(x, c, n, *params)
		else:
			like = self.like(x, c, n, *params)

		like = np.where(like <= surpyval.TINIEST, surpyval.TINIEST, like)
		like = np.where(like < 1, like, 1)
		if t is not None:
			like = np.log(like) - np.log(self.like_t(t, *params))
		else:
			like = np.log(like)
		like = np.multiply(n, like)
		return -np.sum(like)

	def neg_ll(self, x, c=None, n=None, t=None, *params):
		# Where the magic happens
		if 2 in c:
			like = self.like_with_interval(x, c, n, *params)
		else:
			like = self.like(x, c, n, *params)
		like = np.where(like <= 0, surpyval.TINIEST, like)
		like = np.where(like < 1, like, 1)
		if t is not None:
			like = np.log(like) - np.log(self.like_t(t, *params))
		else:
			like = np.log(like)
		like = np.multiply(n, like)
		return -np.sum(like)

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
		D[D < surpyval.TINIEST] = surpyval.TINIEST
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
		# What the fuck is this...
		x_, r, d, R = nonp.turnbull(x, c, n, estimator='Nelson-Aalen')
		F = 1 - R
		mask = np.isfinite(x_)
		F = F[mask]
		x = x[mask]
		init = self.parameter_initialiser(x, c=c, n=n)
		fun = lambda params : np.sum(((self.ff(x_, *params)) - F)**2)
		res = minimize(fun, init, bounds=self.bounds)
		# OLD MSE - changed to Turnbull for better use
		# x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)
		# init = self.parameter_initialiser(x, c=c, n=n)
		# fun = lambda t : np.sum(((self.ff(x, *t)) - F)**2)
		# res = minimize(fun, init, bounds=self.bounds)
		# self.res = res
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

	def _mle(self, x, c, n, t):
		"""
		MLE: Maximum Likelihood estimate
		"""
		# This might help to be able to hold a parameter constant:
		# https://stackoverflow.com/questions/24185589/minimizing-a-function-while-keeping-some-of-the-variables-constant

		def constraints(p):
			# Add constraint handling here
			# i.e. in comes [1, 0] and it needs to make [1, 2.3, 0] for the func
			# assumes 2.3 is provided as the constant
			return p

		init = self.parameter_initialiser(x, c, n)

		# Change logic here for selecting the neg_ll function.
		# if 2 in c: func = self.neg_ll_with_int

		if self.use_autograd:
				try:
					fun  = lambda xx : self.neg_ll(x, c, n, t, *constraints(xx))
					jac = jacobian(fun)
					hess = hessian(fun)
					res = minimize(fun, init, 
									method='trust-exact', 
									jac=jac, 
									hess=hess, 
									tol=1e-10)
					hess_inv = inv(res.hess)
				except:
					print("Autograd attempt failed, using without hessian")
					fun = lambda xx : self.neg_ll(x, c, n, t, *constraints(xx))
					jac = lambda xx : approx_fprime(xx, fun, surpyval.EPS)
					res = minimize(fun, init, method='BFGS', jac=jac)
					hess_inv = res.hess_inv

		else:
			fun = lambda xx : self.neg_ll(x, c, n, t, *constraints(xx))
			jac = lambda xx : approx_fprime(xx, fun, surpyval.EPS)
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

	def _mpp(self, x, c=None, n=None, heuristic="Turnbull", rr='y', on_d_is_0=False):
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
		model = para.Parametric()
		model.method = how
		#Truncated data
		t = kwargs.get('t', None)

		model.raw_data = {
			'x' : x,
			'c' : c,
			'n' : n,
			't' : t
		}

		x, c, n = surpyval.xcn_handler(x, c, n)

		if t is not None:
			assert t.shape[0] == n.shape[0]
			assert t.shape[1] == 2
		
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n,
			't' : t
		}

		heuristic = kwargs.get('heuristic', 'Turnbull')
		model.heuristic = heuristic
		model.dist = self

		if   how == 'MLE':
			# Maximum Likelihood
			with np.errstate(all='ignore'):
				model.res, model.jac, model.hess_inv = self._mle(x, c=c, n=n, t=t)
				model.params = tuple(model.res.x)
		elif how == 'MPS':
			# Maximum Product Spacing
			if model.raw_data['c'] is not None:
				raise Exception('Maximum product spacing doesn\'t support censoring')
			if model.raw_data['n'] is not None:
				raise Exception('Maximum product spacing doesn\'t support counts')
			if model.raw_data['t'] is not None:
				raise Exception('Maximum product spacing doesn\'t support tuncation')
			model.res = self._mps(x)
			model.params = tuple(model.res.x)
		elif how == 'MOM':
			if model.raw_data['c'] is not None:
				raise Exception('Method of moments doesn\'t support censoring')
			if model.raw_data['t'] is not None:
				raise Exception('Maximum product spacing doesn\'t support tuncation')
			model.res = self._mom(x, n=n)
			model.params = tuple(model.res.x)
		elif how == 'MPP':
			rr = kwargs.get('rr', 'y')
			if model.raw_data['t'] is not None:
				raise Exception('Method of probability plotting doesn\'t support tuncation')
			model.params = self._mpp(x, n=n, c=c, rr=rr, heuristic=heuristic)
		elif how == 'MSE':
			if model.raw_data['t'] is not None:
				raise Exception('Maximum product spacing doesn\'t support tuncation')
			model.res = self._mse(x, c=c, n=n, heuristic=heuristic)
			model.params = tuple(model.res.x)
		
		return model

	def fit_from_df(self, df, **kwargs):
		assert type(df) == pd.DataFrame

		heuristic = kwargs.get('heuristic', 'Nelson-Aalen')
		how = kwargs.get('how', 'MLE')
		x_col = kwargs.pop('x', 'x')
		c_col = kwargs.pop('c', 'c')
		n_col = kwargs.pop('n', 'n')

		#raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

		x = df[x_col].astype(surpyval.NUM)
		assert x.ndim == 1

		if c_col in df:
			c = df[c_col].values.astype(np.int64)
		else:
			c = None

		if n_col in df:
			n = df[n_col].values.astype(np.int64)
		else:
			n = None

		return self.fit(x, c, n, how, **kwargs)

	def from_params(self, params):
		model = para.Parametric()
		assert self.k == len(params), "Must have {k} params for {dist} distribution".format(k=self.k, dist=self.dist.name)
		model.params = params
		for i, (low, upp) in enumerate(self.bounds):
			if low is None:
				l = -np.inf
			else:
				l = low
			if upp is None:
				u = np.inf
			else:
				u = upp

			assert (l < params[i]) & (params[i] < u), "Params must be in bounds {}".format(self.bounds)
		model.dist = self
		return model










