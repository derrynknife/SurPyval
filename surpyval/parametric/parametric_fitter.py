from autograd import jacobian, hessian
import autograd.numpy as np
from autograd.numpy.linalg import inv

from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from scipy.stats import pearsonr

import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para

import pandas as pd

PARA_METHODS = ['MPP', 'MLE', 'MPS', 'MSE', 'MOM']

class ParametricFitter():
	def parameter_initialiser_(self, x, c, n):
		x, c, n, R = nonp.turnbull(x, c, n, estimator='Turnbull')
		F = 1 - R
		F = F[np.isfinite(x)]
		x = x[np.isfinite(x)]
		init = np.ones(self.k)
		fun = lambda params : np.sum(((self.ff(x, *params)) - F)**2)
		res = minimize(fun, init, bounds=self.bounds)

		return res.x

	def like_with_interval(self, x, c, n, *params):
		xl = x[:, 0]
		xr = x[:, 1]
		like = np.zeros_like(xl).astype(surpyval.NUM)
		like_i = np.zeros_like(xl).astype(surpyval.NUM)
		like = np.where(c ==  0, self.df(xl, *params), like)
		like = np.where(c == -1, self.ff(xl, *params), like)
		like = np.where(c ==  1, self.sf(xl, *params), like)
		like_i = np.where(c ==  2, self.ff(xr, *params) - self.ff(xl, *params), like_i)
		return like + like_i

	def like(self, x, c, n, *params):
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

	def neg_ll_trunc(self, x, c, n, t, *params):
		# This is going to get difficult
		# if 2 in c:
		like = self.like_with_interval(x, c, n, *params)
		# else:
			# like = self.like(x, c, n, *params)

		like = np.where(like <= surpyval.TINIEST, surpyval.TINIEST, like)
		like = np.where(like < 1, like, 1)
		if t is not None:
			like = np.log(like) - np.log(self.like_t(t, *params))
		else:
			like = np.log(like)
		like = np.multiply(n, like)
		return -np.sum(like)

	def neg_ll(self, x, c, n, t, *params):
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

	# def neg_mean_D(self, x, c, n, *params):
	# 	idx = np.argsort(x)
	# 	F  = self.ff(x[idx], *params)
	# 	D0 = F[0]
	# 	Dn = 1 - F[-1]
	# 	D = np.diff(F)
	# 	D = np.concatenate([[D0], D, [Dn]])
	# 	if c is not None:
	# 		Dr = self.sf(x[c == 1], *params)
	# 		Dl = self.ff(x[c == -1], *params)
	# 		D = np.concatenate([Dl, D, Dr])
	# 	D[D < surpyval.TINIEST] = surpyval.TINIEST
	# 	M = np.log(D)
	# 	M = -np.sum(M)/(M.shape[0])
	# 	return M

	def neg_mean_D(self, x, c, n, *params):
		mask = c == 0
		x_obs = x[mask]
		n_obs = n[mask]

		# Assumes already ordered
		F  = self.ff(x_obs, *params)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])

		Dr = self.sf(x[c ==  1], *params)
		Dl = self.ff(x[c == -1], *params)

		if (n_obs > 1).any():
			n_ties = (n_obs - 1).sum()
			Df = self.df(x_obs, *params)
			Df = Df[Df != 0]
			LL = np.concatenate([Dl, Df, Dr])
			ll_n = np.concatenate([n[c == -1], (n_obs - 1), n[c == 1]])
		else:
			Df = []
			n_ties = n_obs.sum()
			LL = np.concatenate([Dl, Dr])
			ll_n = np.concatenate([n[c == -1], n[c == 1]])
		
		# D = np.concatenate([Dl, D, Dr, Df])
		D[D < surpyval.TINIEST] = surpyval.TINIEST
		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		
		LL[LL < surpyval.TINIEST] = surpyval.TINIEST
		LL = -(np.log(LL) * ll_n).sum()/(n.sum() - n_obs.sum() + n_ties)
		return M + LL

	def mom_moment_gen(self, *params):
		moments = np.zeros(self.k)
		for i in range(0, self.k):
			n = i + 1
			moments[i] = self.moment(n, *params)
		return moments

	def _mse(self, x, c, n, init):
		"""
		MSE: Mean Square Error
		This is simply fitting the curve to the best estimate from a non-parametric estimate.

		This is slightly different in that it fits it to untransformed data on the x and 
		y axis. The MPP method fits the curve to the transformed data. This is simply fitting
		a the CDF sigmoid to the nonparametric estimate.
		"""

		x_, r, d, R = nonp.turnbull(x, c, n, estimator='Nelson-Aalen')
		F = 1 - R
		mask = np.isfinite(x_)
		F  = F[mask]
		x_ = x_[mask]
		fun = lambda params : np.sum(((self.ff(x_, *params)) - F)**2)
		res = minimize(fun, init, bounds=self.bounds)
		# OLD MSE - changed to Turnbull for better use
		# x, r, d, F = nonp.plotting_positions(x, c=c, n=n, heuristic=heuristic)
		# init = self.parameter_initialiser(x, c=c, n=n)
		# fun = lambda t : np.sum(((self.ff(x, *t)) - F)**2)
		# res = minimize(fun, init, bounds=self.bounds)
		# self.res = res
		return res

	def _mps(self, x, c, n, init):
		"""
		MPS: Maximum Product Spacing

		This is the method to get the largest (geometric) average distance between all points

		This method works really well when all points are unique. Some complication comes in when using repeated data.

		This method is exceptional for when using three parameter distributions.
		"""
		bounds = self.bounds
		fun = lambda params : self.neg_mean_D(x, c, n, *params)
		res = minimize(fun, init, bounds=bounds)
		return res

	def _mle(self, x, c, n, t, init, fixed=None):
		"""
		MLE: Maximum Likelihood estimate
		"""
		# This might help to be able to hold a parameter constant:
		# https://stackoverflow.com/questions/24185589/minimizing-a-function-while-keeping-some-of-the-variables-constant

		def constraints(p, not_fixed):
			params = np.empty(self.k)
			for k, v in fixed.items():
				params[self.param_map[k]] = v
			for i, v in zip(not_fixed, p):
				params[i] = v
			return params

		fail = False

		if fixed is not None:
			fixed_idx = [self.param_map[x] for x in fixed.keys()]
			not_fixed = np.array([x for x in range(self.k) if x not in fixed_idx])
			fun = lambda params: self.neg_ll(x, c, n, t, *constraints(params, not_fixed))
			init = init[not_fixed]
		else:
			fun = lambda params: self.neg_ll(x, c, n, t, *params)


		if self.use_autograd:
				try:
					jac  = jacobian(fun)
					hess = hessian(fun)
					res  = minimize(fun, init, 
									method='trust-exact', 
									jac=jac, 
									hess=hess, 
									tol=1e-10)
					hess_inv = inv(res.hess)
				except:
					print("Autograd attempt failed, using without hessian")
					fail = True

		if (fail) | (not self.use_autograd):
			jac = lambda xx : approx_fprime(xx, fun, surpyval.EPS)
			res = minimize(fun, init, method='BFGS', jac=jac)
			hess_inv = res.hess_inv

		# It's working!! 
		# unpack the constraints
		if fixed is not None:
			np.matrix(np.empty(shape=(self.k, self.k)))
			return res, jac, None, constraints(res.x, not_fixed)
		else:
			return res, jac, hess_inv, tuple(res.x)

	def _mom(self, x, n, init):
		"""
		MOM: Method of Moments.

		This is one of the simplest ways to calculate the parameters of a distribution.

		This method is quick but only works with uncensored data.
		# Can I add a simple sum(c) instead of length to work with censoring?
		"""
		x_ = np.repeat(x, n)

		moments = np.zeros(self.k)
		for i in range(0, self.k):
			moments[i] = np.sum(x_**(i+1)) / len(x_)

		fun = lambda params : np.sum((moments - self.mom_moment_gen(*params))**2)
		res = minimize(fun, init, bounds=self.bounds)
		return res

	def _mpp(self, x, c, n, heuristic="Turnbull", rr='y', on_d_is_0=False):
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
			y_ = F[d > 0]

		# Linearise
		x_pp = self.mpp_x_transform(x_)
		y_pp = self.mpp_y_transform(y_)

		mask = np.isfinite(y_pp)
		y_pp = y_pp[mask]
		x_pp = x_pp[mask]

		if   rr == 'y':
			params = np.polyfit(x_pp, y_pp, 1)
		elif rr == 'x':
			params = np.polyfit(y_pp, x_pp, 1)

		params = self.unpack_rr(params, rr)

		return params

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		#Truncated data
		t = kwargs.pop('t', None)

		if how not in PARA_METHODS:
			raise ValueError('"how" must be one of: ' + str(PARA_METHODS))

		if t is not None and how == 'MPS':
			raise ValueError('Maximum product spacing doesn\'t support tuncation')
		if t is not None and how == 'MSE':
			raise NotImplementedError('Mean square error doesn\'t yet support tuncation')
		if t is not None and how == 'MPP':
			raise NotImplementedError('Method of probability plotting doesn\'t yet support tuncation')
		if t is not None and how == 'MOM':
			raise ValueError('Maximum product spacing doesn\'t support tuncation')

		x, c, n = surpyval.xcn_handler(x, c, n)
		# TODO: Add xcnt_handler

		if surpyval.utils.check_no_censoring(c) and (how == 'MOM'):
			raise ValueError('Method of moments doesn\'t support censoring')

		heuristic = kwargs.pop('heuristic', 'Turnbull')
		if (surpyval.utils.no_left_or_int(c)) and (how == 'MPP') and (not heuristic == 'Turnbull'):
			raise ValueError('Probability plotting estimation with left or interval censoring only works with Turnbull heuristic')

		# Passed checks
		model = para.Parametric()
		model.method = how
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n,
			't' : t
		}

		model.heuristic = heuristic
		model.dist = self
		if how != 'MPP':
			init = np.array(self.parameter_initialiser(x, c, n))

		if how == 'MLE':
			# Maximum Likelihood
			fixed = kwargs.pop('fixed', None)
			with np.errstate(all='ignore'):
				model.res, model.jac, model.hess_inv, params = self._mle(x=x, c=c, n=n, t=t, fixed=fixed, init=init)
				model.params = tuple(params)

		elif how == 'MPS':
			# Maximum Product Spacing
			model.res = self._mps(x=x, c=c, n=n, init=init)
			model.params = tuple(model.res.x)

		elif how == 'MOM':
			# Method of Moments
			model.res = self._mom(x=x, n=n, init=init)
			model.params = tuple(model.res.x)

		elif how == 'MPP':
			# Method of Probability Plotting
			rr = kwargs.get('rr', 'y')
			model.params = tuple(self._mpp(x=x, n=n, c=c, rr=rr, heuristic=heuristic))

		elif how == 'MSE':
			# Mean Square Error
			model.res = self._mse(x=x, c=c, n=n, init=init)
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










