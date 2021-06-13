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
from copy import deepcopy, copy

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

	def like(self, x, c, n, *params):
		like = np.zeros_like(x).astype(surpyval.NUM)
		like = np.where(c ==  0, self.df(x, *params), like)
		like = np.where(c == -1, self.ff(x, *params), like)
		like = np.where(c ==  1, self.sf(x, *params), like)
		return like

	def like_i(self, x, c, n, inf_c_flags, *params):
		# This makes sure that any intervals that are at the boundaries of support or
		# are infinite will not cause the autograd functions to fail.
		ir = np.where(inf_c_flags[:, 1] == 1, 1, self.ff(x[:, 1], *params))
		il = np.where(inf_c_flags[:, 0] == 1, 0, self.ff(x[:, 0], *params))
		like_i = ir - il
		return like_i

	def like_t(self, t, t_flags, *params):
		tr_denom = np.where(t_flags[:, 1] == 1, self.ff(t[:, 1], *params), 1.)
		tl_denom = np.where(t_flags[:, 0] == 1, self.ff(t[:, 0], *params), 0.)
		t_denom = tr_denom - tl_denom
		return t_denom

	def neg_ll(self, x, c, n, inf_c_flags, t, t_flags, *params):
		if 2 in c:
			like_i = self.like_i(x, c, n, inf_c_flags, *params)
			x_ = copy(x[:, 0])
			x_[x_ == 0] = 1
		else:
			like_i = 0
			x_ = copy(x)
			
		like = self.like(x_, c, n, *params)
		like = like + like_i
		like = np.log(like) - np.log(self.like_t(t, t_flags, *params))
		like = np.multiply(n, like)
		return -np.sum(like)

	def neg_mean_D(self, x, c, n, *params):
		mask = c == 0
		x_obs = x[mask]
		n_obs = n[mask]

		gamma = 0

		# Assumes already ordered
		F  = self.ff(x_obs - gamma, *params)
		D0 = F[0]
		Dn = 1 - F[-1]
		D = np.diff(F)
		D = np.concatenate([np.array([D0]), D.T, np.array([Dn])]).T

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
		

		M = np.log(D)
		M = -np.sum(M)/(M.shape[0])
		
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

	def fit(self, x, c=None, n=None, t=None, how='MLE', **kwargs):
		# Check inputs
		offset = kwargs.pop('offset', False)
		fixed = kwargs.pop('fixed', None)
		tl = kwargs.pop('tl', None)
		tr = kwargs.pop('tr', None)

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

		# Needs to handle tl and tr
		if t is not None and how == 'MOM':
			raise ValueError('Maximum product spacing doesn\'t support tuncation')

		x, c, n, t = surpyval.xcnt_handler(x, c, n, t=t, tl=tl, tr=tr)

		# Turnbull should be avoided as the alpha and beta matrix can be memory expensive!
		if (~np.isfinite(t)).any() & ((-1 not in c) & (2 not in c)):
			heuristic = kwargs.pop('heuristic', 'Nelson-Aalen')
		else:
			heuristic = kwargs.pop('heuristic', 'Turnbull')

		if surpyval.utils.check_no_censoring(c) and (how == 'MOM'):
			raise ValueError('Method of moments doesn\'t support censoring')
		
		if (surpyval.utils.no_left_or_int(c)) and (how == 'MPP') and (not heuristic == 'Turnbull'):
			raise ValueError('Probability plotting estimation with left or interval censoring only works with Turnbull heuristic')

		if not offset:
			if x.ndim == 2:
				if ((x[:, 0] <= self.support[0]) & (c == 0)).any():
					raise ValueError("Observed values must be in support of distribution; are some of your observed values 0, -Inf, or Inf?")
			else:
				if ((x <= self.support[0]) & (c == 0)).any():
					raise ValueError("Observed values must be in support of distribution; are some of your observed values 0, -Inf, or Inf?")

		# Passed checks
		if offset:
			model = para.OffsetParametric()
		else:
			model = para.Parametric()

		# There is hope in this!
		# model = para.ParametricO(offset)

		model.method = how
		model.heuristic = heuristic
		model.dist = self
		model.data = {
			'x' : x,
			'c' : c,
			'n' : n,
			't' : t
		}

		bounds = deepcopy(self.bounds)
		param_map = self.param_map.copy()

		if offset:
			bounds = ((None, np.min(x)), *bounds)
			offset_index_inc = 1
			param_map.update({'gamma' : -1})
		else:
			offset_index_inc = 0

		model.bounds = bounds

		transform, inv_trans, funcs, inv_f = bounds_convert(x, bounds)
		const, fixed_idx, not_fixed = fix_idx_and_function(self, fixed, param_map, offset_index_inc, funcs)

		if how != 'MPP':
			# Need a better general fitter to include offset
			if 'init' in kwargs:
				init = kwargs.pop('init')
			else:
				if self.name in ['Gumbel', 'Beta', 'Normal']:
					init = np.array(self.parameter_initialiser(x, c, n))
				else:
					init = np.array(self.parameter_initialiser(x, c, n, offset=offset))
					
			# This should happen in the optimiser
			init = transform(init)
			init = init[not_fixed]
		else:
			# Probability plotting method does not need an initial estimate
			init = None

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
			model.res, model.jac, model.hess_inv, params = mps(dist=self, x=x, c=c, n=n, init=init, **fix_and_const_kwargs)

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
		if params is not None:
			if offset:
				model.gamma = params[0]
				model.params = tuple(params[1::])
			else:
				model.params = tuple(params)
		else:
			model.params = []

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
			c = df[c_col].values.astype(int)
		else:
			c = None

		if n_col in df:
			n = df[n_col].values.astype(int)
		else:
			n = None

		return self.fit(x, c, n, how, **kwargs)

	def from_params(self, params, offset=None):
		if self.k != len(params):
			raise ValueError("Must have {k} params for {dist} distribution".format(k=self.k, dist=self.name))

		if offset is not None and self.name in ['Normal', 'Beta', 'Uniform', 'Gumbel', 'Logistic']:
			raise ValueError('{dist} distribution cannot be offset'.format(dist=self.name))

		if offset is not None:
			model = para.OffsetParametric()
			model.gamma = offset
			model.bounds = ((None, offset), *deepcopy(self.bounds))
		else:
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
