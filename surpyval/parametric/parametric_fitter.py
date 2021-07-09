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

	def fit(self, x=None, c=None, n=None, t=None, how='MLE',
			offset=False, tl=None, tr=None, xl=None, xr=None,
			fixed=None, heuristic='Nelson-Aalen', init=[], rr='y'):

		r"""

		The central feature to SurPyval's capability. This function aimed to have an API to mimic the 
		simplicity of the scipy API. That is, to use a simple :code:`fit()` call, with as many or as few
		parameters as is needed.

		Parameters
		----------

		x : array like, optional
			Array of observations of the random variables. If x is :code:`None`, xl and xr must be provided.
		c : array like, optional
			Array of censoring flag. -1 is left censored, 0 is observed, 1 is right censored, and 2 is intervally
			censored. If not provided will assume all values are observed.
		n : array like, optional
			Array of counts for each x. If data is proivded as counts, then this can be provided. If :code:`None`
			will assume each observation is 1.
		t : 2D-array like, optional
			2D array like of the left and right values at which the respective observation was truncated. If
			not provided it assumes that no truncation occurs.
		how : {'MLE', 'MPP', 'MOM', 'MSE', 'MPS'}, optional
			Method to estimate parameters, these are:

				- MLE : Maximum Likelihood Estimation
				- MPP : Method of Probability Plotting
				- MOM : Method of Moments
				- MSE : Mean Square Error
				- MPS : Maximum Product Spacing

		offset : boolean, optional
			If :code:`True` finds the shifted distribution. If not provided assumes not a shifted distribution.
			Only works with distributions that are supported on the half-real line.

		tl : array like or scalar, optional
			Values of left truncation for observations. If it is a scalar value assumes each observation is
			left truncated at the value. If an array, it is the respective 'late entry' of the observation

		tr : array like or scalar, optional
			Values of right truncation for observations. If it is a scalar value assumes each observation is
			right truncated at the value. If an array, it is the respective right truncation value for each
			observation

		xl : array like, optional
			Array like of the left array for 2-dimensional input of x. This is useful for data that is all
			intervally censored. Must be used with the :code:`xr` input.

		xr : array like, optional
			Array like of the right array for 2-dimensional input of x. This is useful for data that is all
			intervally censored. Must be used with the :code:`xl` input.

		fixed : dict, optional
			Dictionary of parameters and their values to fix. Fixes parameter by name.

		heuristic : {'"Blom", "Median", "ECDF", "Modal", "Midpoint", "Mean", "Weibull", "Benard", "Beard", "Hazen", "Gringorten", "None", "Tukey", "DPW", "Fleming-Harrington", "Kaplan-Meier", "Nelson-Aalen", "Filliben", "Larsen", "Turnbull"}
			Plotting method to use, if using the probability plotting, MPP, method.

		init : array like, optional
			initial guess of parameters. Useful if method is failing.

		rr : {'y', 'x'}
			The dimension on which to minimise the spacing between the line and the observation.
			If 'y' the mean square error between the line and vertical distance to each point is minimised.
			If 'x' the mean square error between the line and horizontal distance to each point is minimised.

		Returns
		-------

		model : Parametric
			A parametric model with the fitted parameters and methods for all functions of the distribution using the 
			fitted parameters.


		Examples
		--------
		>>> from surpyval import Weibull
		>>> x = Weibull.random(100, 10, 4)
		>>> model = Weibull.fit(x)
		>>> print(model)
		Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters [10.25563233  3.68512055]

		>>> model = Weibull.fit(x, how='MPS', fixed={'alpha' : 10})
		>>> print(model)
		Parametric Surpyval model with Weibull distribution fitted by MPS yielding parameters [10.          3.45512434]

		>>> model = Weibull.fit(xl=x, xr=x+2, how='MPP')
		>>> print(model)
		Parametric Surpyval model with Weibull distribution fitted by MPP yielding parameters (11.465337182989183, 9.217130068358955)

		>>> c = np.zeros_like(x)
		>>> c[x > 13] = 1
		>>> x[x > 13] = 13
		>>> c = c[x > 6]
		>>> x = x[x > 6]
		>>> model = Weibull.fit(x=x, c=c, tl=6)
		>>> print(model)
		Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters [9.70132936 3.03139549]
		"""

		if (x is not None) & ((xl is not None) | (xr is not None)):
			raise ValueError("Must use either 'x' of both 'xl and 'xr'")

		if (x is None) & ((xl is None) | (xr is None)):
			raise ValueError("Must use either 'x' of both 'xl and 'xr'")

		if x is None:
			xl = np.array(xl).astype(float)
			xr = np.array(xr).astype(float)
			x = np.vstack([xl, xr]).T

		if offset and self.name in ['Normal', 'Beta', 'Uniform', 'Gumbel', 'Logistic']:
			raise ValueError('{dist} distribution cannot be offset'.format(dist=self.name))

		if how not in PARA_METHODS:
			raise ValueError('"how" must be one of: ' + str(PARA_METHODS))

		if how == 'MPP' and self.name == 'ExpoWeibull':
			raise ValueError('ExpoWeibull distribution does not work with probability plot fitting')			

		if t is not None and how == 'MPS':
			raise ValueError('Maximum product spacing doesn\'t yet support tuncation')

		if t is not None and how == 'MSE':
			raise NotImplementedError('Mean square error doesn\'t yet support tuncation')

		if t is not None and how == 'MPP':
			raise NotImplementedError('Method of probability plotting doesn\'t yet support tuncation')

		if t is not None and how == 'MOM':
			raise ValueError('Maximum product spacing doesn\'t support tuncation')

		x, c, n, t = surpyval.xcnt_handler(x=x, c=c, n=n, t=t, tl=tl, tr=tr)

		if x.ndim == 2:
			heuristic = 'Turnbull'

		# Turnbull should be avoided as the alpha and beta matrix can be memory expensive!
		# if (~np.isfinite(t)).any() & ((-1 not in c) & (2 not in c)):
		# 	heuristic = kwargs.pop('heuristic', 'Nelson-Aalen')
		# else:
		# 	heuristic = kwargs.pop('heuristic', 'Turnbull')

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
			if init == []:
				if self.name in ['Gumbel', 'Beta', 'Normal', 'Uniform']:
					init = np.array(self._parameter_initialiser(x, c, n))
				else:
					init = np.array(self._parameter_initialiser(x, c, n, offset=offset))
					
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
			results = mle(dist=self, x=x, c=c, n=n, t=t, init=init, **fix_and_const_kwargs)

		elif how == 'MPS':
			# Maximum Product Spacing
			results = mps(dist=self, x=x, c=c, n=n, init=init, **fix_and_const_kwargs)

		elif how == 'MOM':
			# Method of Moments
			# results = mom(dist=self, x=x, n=n, init=init, offset=offset)
			results = mom(dist=self, x=x, n=n, init=init, **fix_and_const_kwargs)

		elif how == 'MPP':
			# Method of Probability Plotting
			results = mpp(dist=self, x=x, n=n, c=c, rr=rr, heuristic=heuristic, offset=offset)

		elif how == 'MSE':
			# Mean Square Error
			results = mse(dist=self, x=x, c=c, n=n, t=t, init=init, **fix_and_const_kwargs)

		for k, v in results.items():
			setattr(model, k, v)

		return model

	def fit_from_df(self, df, x=None, c=None, n=None,
					xl=None, xr=None, tl=None, tr=None,
					**fit_options):

		r"""

		The central feature to SurPyval's capability. This function aimed to have an API to mimic the 
		simplicity of the scipy API. That is, to use a simple :code:`fit()` call, with as many or as few
		parameters as is needed.

		Parameters
		----------

		df : DataFrame
			DataFrame of data to be used to create surpyval model

		x : string, optional
			column name for the column in df containing the variable data. If not provided must provide
			both xl and xr

		c : string, optional
			column name for the column in df containing the censor flag of x. If not provided assumes
			all values of x are observed.

		n : string, optional
			column name in for the column in df containing the counts of x. If not provided assumes
			each x is one observation.

		tl : string or scalar, optional
			If string, column name in for the column in df containing the left truncation data. If scalar
			assumes each x is left truncated by that value. If not provided assumes x is not left truncated.

		tr : string or scalar, optional
			If string, column name in for the column in df containing the right truncation data. If scalar
			assumes each x is right truncated by that value. If not provided assumes x is not right truncated.

		xl : string, optional
			column name for the column in df containing the left interval for interval censored data.
			If left interval is -Inf, assumes left censored. If xl[i] == xr[i] assumes observed. Cannot
			be provided with x, must be provided with xr.

		xr : string, optional
			column name for the column in df containing the right interval for interval censored data.
			If right interval is Inf, assumes right censored. If xl[i] == xr[i] assumes observed. Cannot
			be provided with x, must be provided with xl.

		fit_options : dict, optional
			dictionary of fit options that will be passed to the :code:`fit` method, see that method for options.

		Returns
		-------

		model : Parametric
			A parametric model with the fitted parameters and methods for all functions of the distribution using the 
			fitted parameters.


		"""

		if not type(df) == pd.DataFrame:
			raise ValueError("df must be a pandas DataFrame")

		if (x is not None) and ((xl is not None) or (xr is not None)):
			raise ValueError("Must use either 'x' or 'xl' and 'xr'; cannot use both")

		if x is not None:
			x = df[x].astype(float)
		else:
			xl = df[xl].astype(float)
			xr = df[xr].astype(float)
			x = np.vstack([xl, xr]).T

		#raise TypeError('Unepxected kwargs provided: %s' % list(kwargs.keys()))

		if c is not None:
			c = df[c].values.astype(int)

		if n is not None:
			n = df[n].values.astype(int)

		if tl is not None:
			if type(tl) == str:
				tl = df[tl].values.astype(float)
			elif np.isscalar(tl):
				tl = (np.ones(df.shape[0]) * tl).astype(float)
			else:
				raise ValueError('tl must be scalar or column label')
		else:
			tl = np.ones(df.shape[0]) * -np.inf

		if tr is not None:
			if type(tr) == str:
				tr = df[tr].values.astype(float)
			elif np.isscalar(tr):
				tr = (np.ones(df.shape[0]) * tr).astype(float)
			else:
				raise ValueError('tr must be scalar or column label')
		else:
			tr = np.ones(df.shape[0]) * np.inf

		t = np.vstack([tl, tr]).T

		return self.fit(x=x, c=c, n=n, t=t, **fit_options)

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
		model.params = np.array(params)
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
