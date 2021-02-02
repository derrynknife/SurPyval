from autograd import jacobian, hessian
import autograd.numpy as np
from autograd.numpy.linalg import inv

from scipy.optimize import minimize
from scipy.optimize import approx_fprime
from scipy.stats import pearsonr
from scipy.integrate import quad

import surpyval
from surpyval import nonparametric as nonp
from surpyval import parametric as para

import pandas as pd
from copy import deepcopy

from .fitters.mom import mom
from .fitters.mle import mle
from .fitters.mps import mps
from .fitters.mse import mse
from .fitters.mpp import mpp

from .fitters import bounds_convert, fix_idx_and_function

PARA_METHODS = ['MPP', 'MLE', 'MPS', 'MSE', 'MOM']

class ParametricFitter():
	def parameter_initialiser_(self, x, c, n, offset=False):
		# Change this to simply invoke MPP
		x, c, n, R = nonp.turnbull(x, c, n, estimator='Turnbull')
		F = 1 - R
		F = F[np.isfinite(x)]
		x = x[np.isfinite(x)]
		# Need to add offset step.
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
		if 2 in c:
			like = self.like_with_interval(x, c, n, *params)
		else:
			like = self.like(x, c, n, *params)
		like = np.log(like) - np.log(self.like_t(t, *params))
		like = np.multiply(n, like)
		return -np.sum(like)

	def neg_ll(self, x, c, n, *params):
		# Where the magic happens
		if 2 in c:
			like = self.like_with_interval(x, c, n, *params)
		else:
			like = self.like(x, c, n, *params)
		# like = np.where(like <= 0, surpyval.TINIEST, like)
		# like = np.where(like < 1, like, 1)
		like = np.log(like)
		like = np.multiply(n, like)
		like = -np.sum(like)
		return like

	def neg_mean_D(self, x, c, n, *params, offset=False):
		mask = c == 0
		x_obs = x[mask]
		n_obs = n[mask]

		if offset:
			gamma = params[0]
			params = params[1::]
		else:
			gamma = 0

		# Assumes already ordered
		F  = self.ff(x_obs - gamma, *params)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([[D0], D, [Dn]])

		Dr = self.sf(x[c ==  1]  - gamma, *params)
		Dl = self.ff(x[c == -1]  - gamma, *params)

		if (n_obs > 1).any():
			n_ties = (n_obs - 1).sum()
			Df = self.df(x_obs  - gamma, *params)
			# Df = Df[Df != 0]
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

	def _moment(self, n, *params, offset=False):
		## Worth doing an analytic offset moment?
		if offset:
			gamma = params[0]
			params = params[1::]
			fun = lambda x : x**n * self.df((x - gamma), *params)
			m = quad(fun, gamma, np.inf)[0]
		else:
			if hasattr(self, 'moment'):
				m = self.moment(n, *params)
			else:
				fun = lambda x : x**n * self.df(x, *params)
				m = quad(fun, *self.support)[0]
		return m
		

	def mom_moment_gen(self, *params, offset=False):
		if offset:
			k = self.k + 1
		else:
			k = self.k
		moments = np.zeros(k)
		for i in range(0, k):
			n = i + 1
			moments[i] = self._moment(n, *params, offset=offset)
		return moments

	def fit(self, x, c=None, n=None, how='MLE', **kwargs):
		#Truncated data
		t = kwargs.pop('t', None)
		offset = kwargs.pop('offset', False)

		if offset and self.name in ['Normal', 'Beta', 'Uniform', 'Gumbel', 'Logistic']:
			raise ValueError('{dist} distribution cannot be offset'.format(dist=self.name))

		if how not in PARA_METHODS:
			raise ValueError('"how" must be one of: ' + str(PARA_METHODS))

		if how == 'MPP' and self.name == 'ExpoWeibull':
			raise ValueError('ExpoWeibull distribution does not work with probability plot fitting')			

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
		if offset:
			model = para.OffsetParametric()
		else:
			model = para.Parametric()

		model.method = how
		model.heuristic = heuristic
		model.dist = self
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n,
			't' : t
		}

		fixed = kwargs.pop('fixed', None)

		bounds = deepcopy(self.bounds)
		param_map = self.param_map.copy()
		if offset:
			bounds = ((None, np.min(x)), *bounds)
			offset_index_inc = 1
			gamma_map = {'gamma' : -1}
			param_map.update(gamma_map)
		else:
			offset_index_inc = 0

		model.bounds = bounds

		transform, inv_trans, funcs, inv_f = bounds_convert(x, bounds)
		const, fixed_idx, not_fixed = fix_idx_and_function(self, fixed, param_map, offset_index_inc, funcs)

		if how != 'MPP':
			# Need a better general fitter to include offset
			try:
				init = np.array(self.parameter_initialiser(x, c, n, offset=offset))
			except:
				init = np.array(self.parameter_initialiser(x, c, n))
			# This should happen in the optimiser
			init = transform(init)
			init = init[not_fixed]
		else:
			# Probability plotting method does not need an initial estimate
			pass

		fix_and_const_kwargs = {
			'const' : const,
			'trans' : transform,
			'inv_fs' : inv_trans,
			'fixed_idx' : fixed_idx,
			'offset' : offset
		}

		if how == 'MLE':
			# Maximum Likelihood Estimation
			model.res, model.jac, model.hess_inv, params = mle(dist=self, x=x, c=c, n=n, t=t, init=init, **fix_and_const_kwargs)

		elif how == 'MPS':
			# Maximum Product Spacing
			model.res = mps(dist=self, x=x, c=c, n=n, init=init, offset=offset)
			params = model.res.x

		elif how == 'MOM':
			# Method of Moments
			model.res = mom(dist=self, x=x, n=n, init=init, offset=offset)
			params = tuple(model.res.x)

		elif how == 'MPP':
			# Method of Probability Plotting
			rr = kwargs.get('rr', 'y')
			params = mpp(dist=self, x=x, n=n, c=c, rr=rr, heuristic=heuristic, offset=offset)

		elif how == 'MSE':
			# Mean Square Error
			model.res = mse(dist=self, x=x, c=c, n=n, init=init)
			params = tuple(model.res.x)

		# Unpack params and offset
		if offset:
			model.gamma = params[0]
			model.params = tuple(params[1::])
		else:
			model.params = tuple(params)

		return model

	def fit_from_df(self, df, **kwargs):
		"""
		For x, need to allow either:
			- x for single, OR
			- xl and xr for left and right interval
		For t, need to have (for left and right interval):
			- tl, and
			- tr

		"""
		if not type(df) == pd.DataFrame:
			raise ValueError("df must be a pandas DataFrame")

		how = kwargs.pop('how', 'MLE')
		x_col = kwargs.pop('x', 'x')
		c_col = kwargs.pop('c', 'c')
		n_col = kwargs.pop('n', 'n')

		#raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

		x = df[x_col].astype(surpyval.NUM)

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
		if self.k != len(params):
			raise ValueError("Must have {k} params for {dist} distribution".format(k=self.k, dist=self.name))

		model = para.Parametric()
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

			if not ((l < params[i]) & (params[i] < u)):
				raise ValueError("Params {params} must be in bounds {bounds}".format(params=', '.join(self.param_names), bounds=self.bounds))
		model.dist = self
		return model
