import numpy as np
from surpyval import nonparametric as nonp
from scipy.stats import t, norm
from .kaplan_meier import KaplanMeier
from .nelson_aalen import NelsonAalen
from .fleming_harrington import FlemingHarrington_

import matplotlib.pyplot as plt

class NonParametric():
	"""
	Result of ``.fit()`` method for every non-parametric surpyval distribution. This means that each of the 
	methods in this class can be called with a model created from the ``NelsonAalen``, ``KaplanMeier``, 
	``FlemingHarrington``, or ``Turnbull`` estimators.
	"""
	def __repr__(self):
		return "{model} survival model".format(model=self.model)

	def sf(self, x, how='step'):
		r"""

		Surival (or Reliability) function with the non-parametric estimates from the data

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the survival function will be calculated 

		Returns
		-------

		sf : scalar or numpy array 
			The value(s) of the survival function at each x


		Examples
		--------
		>>> from surpyval import NelsonAalen
		>>> x = np.array([1, 2, 3, 4, 5])
		>>> model = NelsonAalen.fit(x)
		>>> model.sf(2)
		array([0.63762815])
		>>> model.sf([1., 1.5, 2., 2.5])
		array([0.81873075, 0.81873075, 0.63762815, 0.63762815])
		"""
		x = np.atleast_1d(x)
		# Let's not assume we can predict above the highest measurement
		if how == 'step':
			idx = np.searchsorted(self.x, x, side='right') - 1
			R = self.R[idx]
			R[np.where(x < self.x.min())] = 1
			R[np.where(x > self.x.max())] = np.nan
			R[np.where(x < 0)] = np.nan
			return R
		elif how == 'interp':
			R = np.hstack([[1], self.R])
			x_data = np.hstack([[0], self.x])
			R = np.interp(x, x_data, R)
			R[np.where(x > self.x.max())] = np.nan
			return R

	def ff(self, x, how='step'):
		r"""

		CDF (failure or unreliability) function with the non-parametric estimates from the data

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the survival function will be calculated 

		Returns
		-------

		ff : scalar or numpy array 
			The value(s) of the failure function at each x


		Examples
		--------
		>>> from surpyval import NelsonAalen
		>>> x = np.array([1, 2, 3, 4, 5])
		>>> model = NelsonAalen.fit(x)
		>>> model.ff(2)
		array([0.36237185])
		>>> model.ff([1., 1.5, 2., 2.5])
		array([0.18126925, 0.18126925, 0.36237185, 0.36237185])
		"""
		return 1 - self.sf(x, how=how)

	def hf(self, x, how='step'):
		r"""

		Instantaneous hazard function with the non-parametric estimates from the data. This is
		calculated using simply the difference between consecutive H(x).

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the survival function will be calculated 

		Returns
		-------

		hf : scalar or numpy array 
			The value(s) of the failure function at each x


		Examples
		--------
		>>> from surpyval import NelsonAalen
		>>> x = np.array([1, 2, 3, 4, 5])
		>>> model = NelsonAalen.fit(x)
		>>> model.ff(2)
		array([0.36237185])
		>>> model.ff([1., 1.5, 2., 2.5])
		array([0.18126925, 0.18126925, 0.36237185, 0.36237185])
		"""
		return np.diff(np.hstack([[0], self.Hf(x, how=how)]))


	def df(self, x, how='step'):
		r"""

		Density function with the non-parametric estimates from the data. This is calculated using 
		the relationship between the hazard function and the density:

		.. math::
			f(x) = h(x)e^{-H(x)}

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the survival function will be calculated 

		Returns
		-------

		df : scalar or numpy array 
			The value(s) of the density function at x


		Examples
		--------
		>>> from surpyval import NelsonAalen
		>>> x = np.array([1, 2, 3, 4, 5])
		>>> model = NelsonAalen.fit(x)
		>>> model.df(2)
		array([0.28693267])
		>>> model.df([1., 1.5, 2., 2.5])
		array([0.16374615, 0.        , 0.15940704, 0.        ])
		"""
		return self.hf(x, how=how) * np.exp(-self.Hf(x, how=how))

	def Hf(self, x, how='step'):
		r"""

		Cumulative hazard rate with the non-parametric estimates from the data. This is calculated using 
		the relationship between the hazard function and the density:

		.. math::
			H(x) = -\ln(R(x))

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the survival function will be calculated 

		Returns
		-------

		Hf : scalar or numpy array 
			The value(s) of the density function at x


		Examples
		--------
		>>> from surpyval import NelsonAalen
		>>> x = np.array([1, 2, 3, 4, 5])
		>>> model = NelsonAalen.fit(x)
		>>> model.Hf(2)
		array([0.45])
		>>> model.df([1., 1.5, 2., 2.5])
		model.Hf([1., 1.5, 2., 2.5])
		"""
		H = -np.log(self.sf(x, how=how))
		return H

	def R_cb(self, x, bound='two-sided', how='step', confidence=0.95, bound_type='exp', dist='z'):
		r"""

		Cumulative hazard rate with the non-parametric estimates from the data. This is calculated using 
		the relationship between the hazard function and the density:

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the confidence bounds will be calculated
		bound : ('two-sided', 'upper', 'lower'), str, optional
			Compute either the two-sided, upper or lower confidence bound(s). Defaults to two-sided
		how : ('step', 'interp'), optional
			How to compute the values between observations. Survival statistics traditionally uses
			step functions, but can use interpolated values if desired. Defaults to step.
		confidence : scalar, optional
			The confidence level at which the bound will be computed.
		bound_type : ('exp', 'regular'), str, optional
			The method with which the bounds will be calculated. Using regular will allow for the 
			bounds to exceed 1 or be less than 0. Defaults to exp as this ensures the bounds are 
			within 0 and 1.
		dist : ('t', 'z'), str, optional
			The distribution to use in finding the bounds. Defaults to the normal (z) distribution.

		Returns
		-------

		R_cb : scalar or numpy array 
			The value(s) of the upper, lower, or both confidence bound(s) of the survival function at x

		Examples
		--------
		>>> from surpyval import NelsonAalen
		>>> x = np.array([1, 2, 3, 4, 5])
		>>> model = NelsonAalen.fit(x)
		>>> model.R_cb([1., 1.5, 2., 2.5], bound='lower', dist='t')
		array([0.11434813, 0.11434813, 0.04794404, 0.04794404])
		>>> model.R_cb([1., 1.5, 2., 2.5])
		array([[0.97789387, 0.16706394],
		       [0.97789387, 0.16706394],
		       [0.91235117, 0.10996882],
		       [0.91235117, 0.10996882]])

		References
		----------
		
		http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis

		"""
		if bound_type not in ['exp', 'normal']:
			return ValueError("'bound_type' must be in ['exp', 'normal']")
		if dist not in ['t', 'z']:
			return ValueError("'dist' must be in ['t', 'z']")

		old_err_state = np.seterr(all='ignore')
			
		x = np.atleast_1d(x)
		if bound in ['upper', 'lower']:
			if dist == 't':
				stat = t.ppf(1 - confidence, self.r - 1)
			else:
				stat = norm.ppf(1 - confidence, 0, 1)
			if bound == 'upper' : stat = -stat
		elif bound == 'two-sided':
			if dist == 't':
				stat = t.ppf((1 - confidence)/2, self.r - 1)
			else:
				stat = norm.ppf((1 - confidence)/2, 0, 1)
			stat = np.array([-1, 1]).reshape(2, 1) * stat

		if bound_type == 'exp':
			# Exponential Greenwood confidence
			R_out = self.greenwood * 1./(np.log(self.R)**2)
			R_out = np.log(-np.log(self.R)) - stat * np.sqrt(R_out)
			R_out = np.exp(-np.exp(R_out))
		else:
			# Normal Greenwood confidence
			R_out = self.R + np.sqrt(self.greenwood * self.R**2) * stat
		# Let's not assume we can predict above the highest measurement
		if how == 'step':
			R_out[np.where(x < self.x.min())] = 1
			R_out[np.where(x > self.x.max())] = np.nan
			# R_out[np.where(x < 0)] = np.nan
			idx = np.searchsorted(self.x, x, side='right') - 1
			if bound == 'two-sided':
				R_out = R_out[:, idx].T
			else:
				R_out = R_out[idx]
		elif how == 'interp':
			if bound == 'two-sided':
				R1 = np.interp(x, self.x, R_out[0, :])
				R2 = np.interp(x, self.x, R_out[1, :])
				R_out = np.vstack([R1, R2]).T
			else:
				R_out = np.interp(x, self.x, R_out)
			R_out[np.where(x > self.x.max())] = np.nan

		np.seterr(**old_err_state)

		return R_out

	def random(self, size):
		return np.random.choice(self.x, size=size)

	def get_plot_data(self, **kwargs):
		y_scale_min = 0
		y_scale_max = 1

		# x-axis
		x_min = 0
		x_max = np.max(self.x)

		diff = (x_max - x_min) / 10
		x_scale_min = x_min
		x_scale_max = x_max + diff

		cbs = self.R_cb(self.x, **kwargs)

		plot_data = {
			'x_scale_min' : x_scale_min,
			'x_scale_max' : x_scale_max,
			'y_scale_min' : y_scale_min,
			'y_scale_max' : y_scale_max,
			'cbs' : cbs,
			'x_' : self.x,
			'R' : self.R,
			'F' : self.F
		}
		return plot_data

	def plot(self, **kwargs):
		r"""
		Creates a plot of the survival function.
		"""
		plot_bounds = kwargs.pop('plot_bounds', True)
		how = kwargs.pop('how', 'step')
		bound = kwargs.pop('how', 'two-sided')
		confidence = kwargs.pop('confidence', 0.95)
		bound_type = kwargs.pop('bound_type', 'exp')
		dist = kwargs.pop('dist', 'z')

		d = self.get_plot_data( how=how,
								bound=bound,
								confidence=confidence,
								bound_type=bound_type,
								dist=dist)
		# MAKE THE PLOT
		# Set the y limits
		plt.gca().set_ylim([d['y_scale_min'], d['y_scale_max']])

		# Label it
		plt.title('Model Survival Plot')
		plt.ylabel('R')
		if how == 'interp':
			if plot_bounds:
				plt.plot(d['x_'], d['cbs'], color='r')
			return plt.plot(d['x_'], d['R'])
		else:
			if plot_bounds:
				plt.step(d['x_'], d['cbs'], color='r')
			return plt.step(d['x_'], d['R'])


