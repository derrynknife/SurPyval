import re
from autograd import jacobian
import autograd.numpy as np
from scipy.stats import uniform

import surpyval
from scipy.special import ndtri as z
from surpyval import nonparametric as nonp

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

def _round_vals(x):
	not_different = True
	i = 1
	while not_different:
		x_ticks = np.array(surpyval.round_sig(x, i))
		not_different = (np.diff(x_ticks) == 0).any()
		i += 1
	return x_ticks

class Parametric():
	"""
	Result of ``.fit()`` or ``.from_params()`` method for every surpyval distribution.
	
	Instances of this class are very useful when a user needs the other functions of a distribution for plotting, optimizations, monte carlo analysis and numeric integration.

	"""
	def __str__(self):
		return 'Parametric Surpyval model with {dist} distribution fitted by {method} yielding parameters {params}'.format(dist=self.dist.name, method=self.method, params=self.params)

	def sf(self, x):
		r"""

		Surival (or Reliability) function for a distribution using the parameters found in the ``.params`` attribute.

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the survival function will be calculated 

		Returns
		-------

		sf : scalar or numpy array 
			The scalar value of the survival function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the survival function at each corresponding value in the input array.


		Examples
		--------
		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> model.sf(2)
		0.9920319148370607
		>>> model.sf([1, 2, 3, 4, 5])
		array([0.9990005 , 0.99203191, 0.97336124, 0.938005  , 0.8824969 ])
		"""
		if type(x) == list:
			x = np.array(x)
		return self.dist.sf(x, *self.params)

	def ff(self, x):
		r"""

		The cumulative distribution function, or failure function, for a distribution using the parameters found in the ``.params`` attribute.

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the failure function (CDF) will be calculated

		Returns
		-------

		ff : scalar or numpy array 
			The scalar value of the CDF of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the CDF at each corresponding value in the input array.


		Examples
		--------

		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> model.ff(2)
		0.007968085162939342
		>>> model.ff([1, 2, 3, 4, 5])
		array([0.0009995 , 0.00796809, 0.02663876, 0.061995  , 0.1175031 ])
		"""
		if type(x) == list:
			x = np.array(x)
		return self.dist.ff(x, *self.params)

	def df(self, x): 
		r"""

		The density function for a distribution using the parameters found in the ``.params`` attribute.

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the density function will be calculated

		Returns
		-------

		df : scalar or numpy array 
			The scalar value of the density function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the density function at each corresponding value in the input array.


		Examples
		--------

		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> model.df(2)
		0.01190438297804473
		>>> model.df([1, 2, 3, 4, 5])
		array([0.002997  , 0.01190438, 0.02628075, 0.04502424, 0.06618727])
		"""
		if type(x) == list:
			x = np.array(x)
		return self.dist.df(x, *self.params)

	def hf(self, x):
		r"""

		The instantaneous hazard function for a distribution using the parameters found in the ``.params`` attribute.

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the instantaneous hazard function will be calculated

		Returns
		-------

		hf : scalar or numpy array 
			The scalar value of the instantaneous hazard function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the instantaneous hazard function at each corresponding value in the input array.


		Examples
		--------

		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> model.hf(2)
		0.012000000000000002
		>>> model.hf([1, 2, 3, 4, 5])
		array([0.003, 0.012, 0.027, 0.048, 0.075])
		"""
		if type(x) == list:
			x = np.array(x)
		return self.dist.hf(x, *self.params)

	def Hf(self, x):
		r"""

		The cumulative hazard function for a distribution using the parameters found in the ``.params`` attribute.

		Parameters
		----------

		x : array like or scalar
			The values of the random variables at which the cumulative hazard function will be calculated

		Returns
		-------

		Hf : scalar or numpy array 
			The scalar value of the cumulative hazard function of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the cumulative hazard function at each corresponding value in the input array.


		Examples
		--------

		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> model.Hf(2)
		0.008000000000000002
		>>> model.Hf([1, 2, 3, 4, 5])
		array([0.001, 0.008, 0.027, 0.064, 0.125])
		"""
		if type(x) == list:
			x = np.array(x)
		return self.dist.Hf(x, *self.params)

	def qf(self, p):
		r"""

		The quantile function for a distribution using the parameters found in the ``.params`` attribute.

		Parameters
		----------
		p : array like or scalar
			The values, which must be between 0 and 1, at which the the quantile will be calculated

		Returns
		-------
		qf : scalar or numpy array 
			The scalar value of the quantile of the distribution if a scalar was passed. If an array like object was passed then a numpy array is returned with the value of the quantile at each corresponding value in the input array.


		Examples
		--------
		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> model.qf(0.2)
		6.06542793124108
		>>> model.qf([.1, .2, .3, .4, .5])
		array([4.72308719, 6.06542793, 7.09181722, 7.99387877, 8.84997045])
		"""
		if type(p) == list:
			p = np.array(p)
		return self.dist.qf(p, *self.params)

	def cs(self, x, X):
		r"""

		The conditional survival of the model.

		Parameters
		----------

		x : array like or scalar
			The values at which conditional survival is to be calculated.
		X : array like or scalar
			The value(s) at which it is known the item has survived

		Returns
		-------

		cs : array
			The conditional survival probability.

		Examples
		--------

		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> model.cs(11, 10)
		0.00025840046151723767
		"""
		return self.dist.cs(x, X, *self.params)

	def random(self, size):
		r"""

		A method to draw random samples from the distributions using the parameters found in the ``.params`` attribute.

		Parameters
		----------
		size : int
			The number of random samples to be drawn from the distribution.

		Returns
		-------
		random : numpy array 
			Returns a numpy array of size ``size`` with random values drawn from the distribution.


		Examples
		--------
		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> np.random.seed(1)
		>>> model.random(1)
		array([8.14127103])
		>>> model.random(10)
		array([10.84103403,  0.48542084,  7.11387062,  5.41420125,  4.59286657,
        		5.90703589,  7.5124326 ,  7.96575225,  9.18134126,  8.16000438])
		"""
		return self.dist.qf(uniform.rvs(size=size), *self.params)

	def mean(self):
		r"""
		A method to draw random samples from the distributions using the parameters found in the ``.params`` attribute.

		Parameters
		----------
		None

		Returns
		-------
		mean : float
			Returns the mean of the distribution.


		Examples
		--------
		>>> from surpyval import Weibull
		>>> model = Weibull.from_params([10, 3])
		>>> model.mean()
		8.929795115692489
		"""
		if not hasattr(self, '_mean'):
			self._mean = self.dist.mean(*self.params)
		return self._mean

	def moment(self, n):
		r"""

		A method to draw random samples from the distributions using the parameters found in the ``.params`` attribute.

		Parameters
		----------
		n : integer
			The degree of the moment to be computed

		Returns
		-------
		moment[n] : float
			Returns the n-th moment of the distribution

		References
		----------
		INSERT WIKIPEDIA HERE

		Examples
		--------
		>>> from surpyval import Normal
		>>> model = Normal.from_params([10, 3])
		>>> model.moment(1)
		10.0
		>>> model.moment(5)
		202150.0
		"""
		return self.dist.moment(n, *self.params)

	def entropy(self):
		r"""

		A method to draw random samples from the distributions using the parameters found in the ``.params`` attribute.

		Parameters
		----------

		None

		Returns
		-------

		entropy : float
			Returns entropy of the distribution

		References
		----------
		
		ENTROPY REF

		Examples
		--------
		>>> from surpyval import Normal
		>>> model = Normal.from_params([10, 3])
		>>> model.entropy()
		2.588783247593625
		"""
		return self.dist.entropy(*self.params)

	# def cb(self, x, sig):
	# 	if self.method != 'MLE':
	# 		raise Exception('Only MLE has confidence bounds')
			
	# 	du = z(sig) * np.sqrt(var_u)
	# 	u_u = u_hat + du
	# 	u_l = u_hat - du
	# 	return np.vstack([np.exp(-np.exp(u_u)), np.exp(-np.exp(u_l))])

	# def R_cb(self, x, cb=0.05):
	# 	return self.dist.R_cb(x, *self.params, self.hess_inv, cb=cb)

	def R_cb(self, t, cb=0.05):
		"""
		.. warning:: The confidence bounds are still a work in progress.

		"""
		if hasattr(self.dist, 'R_cb'):
			return self.dist.R_cb(t, *self.params, self.hess_inv, cb=cb)

		# This can be changed to a general lookup so that ANY function can have it's CB computed!
		sf_func = lambda params : self.dist.sf(t, *params)

		pvars = self.hess_inv[np.triu_indices(self.hess_inv.shape[0])]
		with np.errstate(all='ignore'):
			jac = jacobian(sf_func)(np.array(self.params))


		# See general process here:
		# http://reliawiki.org/index.php/Confidence_Bounds#Fisher_Matrix_Confidence_Bounds
		# I'm pretty sure this is quite wrong.
		# Even if it is wrong it needs to be cleaned for clarity
		# Need to break into a diag calc and an triupper (withouth diag) part.

		# This interpretation comes from:
		# http://reliawiki.org/index.php/The_Gamma_Distribution#Bounds_on_Reliability
		# This is a general pattern
		var_u = []
		for i, j in enumerate(jac):
			j = np.atleast_2d(j).T * j
			j = j[np.triu_indices(j.shape[0])] 
			var_u.append(np.sum(j * pvars))
		diff = z(cb/2) * np.sqrt(np.array(var_u)) * np.array([1., -1.]).reshape(2, 1)
		R_hat = self.sf(t)
		exponent = diff/(R_hat*(1 - R_hat))
		R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
		return R_cb.T

	def neg_ll(self):
		r"""

		The the negative log-likelihood for the model, if it was fit with the ``fit()`` method. Not available if fit with the ``from_params()`` method.

		Parameters
		----------

		None

		Returns
		-------

		neg_ll : float
			The negative log-likelihood of the model

		Examples
		--------

		>>> from surpyval import Weibull
		>>> import numpy as np
		>>> np.random.seed(1)
		>>> x = Weibull.random(100, 10, 3)
		>>> model = Weibull.fit(x)
		>>> model.neg_ll()
		262.52685642385734
		"""
		if not hasattr(self, 'data'):
			raise ValueError("Must have been fit with data")

		return self._neg_ll

	def aic(self):
		r"""

		The the Aikake Information Criterion (AIC) for the model, if it was fit with the ``fit()`` method. Not available if fit with the ``from_params()`` method.

		Parameters
		----------

		None

		Returns
		-------

		aic : float
			The AIC of the model

		Examples
		--------

		>>> from surpyval import Weibull
		>>> import numpy as np
		>>> np.random.seed(1)
		>>> x = Weibull.random(100, 10, 3)
		>>> model = Weibull.fit(x)
		>>> model.aic()
		529.0537128477147
		"""
		if hasattr(self, '_aic'):
			return self._aic
		else:
			k = len(self.params)
			self._aic = 2 * k + 2 * self.neg_ll()
			return self._aic

	def aic_c(self):
		r"""

		The the Corrected Aikake Information Criterion (AIC) for the model, if it was fit with the ``fit()`` method. Not available if fit with the ``from_params()`` method.

		Parameters
		----------

		None

		Returns
		-------

		aic_c : float
			The Corrected AIC of the model

		Examples
		--------

		>>> from surpyval import Weibull
		>>> import numpy as np
		>>> np.random.seed(1)
		>>> x = Weibull.random(100, 10, 3)
		>>> model = Weibull.fit(x)
		>>> model.aic()
		529.1774241879209
		"""
		if hasattr(self, '_aic_c'):
			return self._aic_c
		else:
			k = len(self.params)
			n = self.data['n'].sum()
			self._aic_c = self.aic() + (2*k**2 + 2*k)/(n - k - 1)
			return self._aic_c

	def get_plot_data(self, heuristic='Turnbull', cb=0.05):
		r"""

		A method to gather plot data

		Parameters
		----------
		heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean', 'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None', 'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier', 'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
			The method that the plotting point on the probablility plot will be calculated.

		cb : float, optional
			The confidence with which the confidence bounds, if able, will be calculated. Defaults to 0.95.

		Returns
		-------
		data : dict
			Returns dictionary containing the data needed to do a plot.

		Examples
		--------
		>>> from surpyval import Weibull
		>>> x = Weibull.random(100, 10, 3)
		>>> model = Weibull.fit(x)
		>>> data = model.get_plot_data()
		"""
		x = self.data['x']
		x_, r, d, F = nonp.plotting_positions(
			x=self.data['x'], 
			c=self.data['c'], 
			n=self.data['n'], 
			heuristic=heuristic)

		mask = np.isfinite(x_)
		x_ = x_[mask]
		r  =  r[mask]
		d  =  d[mask]
		F  =  F[mask]

		if np.isfinite(self.data['t']).any():
			Ftl = self.ff(np.min(self.data['t'][:, 0]))
			Ftr = self.ff(np.max(self.data['t'][:, 1]))
			F = Ftl + F * (Ftr - Ftl)

		if self.dist.name in ['Weibull3p']:
			x_ = x_ - self.params[2]
			x  = x  - self.params[2]

		y_scale_min = np.min(F[F > 0])/2
		#y_scale_max = np.max(F[F < 1]) + (1 - np.max(F[F < 1]))/2
		y_scale_max = (1 - (1 - np.max(F[F < 1]))/10)

		# x-axis
		if self.dist.plot_x_scale == 'log':
			log_x = np.log10(x_[x_ > 0])
			x_min = np.min(log_x)
			x_max = np.max(log_x)
			vals_non_sig = 10 ** np.linspace(x_min, x_max, 7)
			x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
			x_minor_ticks = ((10**x_minor_ticks * np.array(np.arange(1, 11))
													.reshape((10, 1)))
													.flatten())
			diff = (x_max - x_min)/10
			x_scale_min = 10**(x_min - diff)
			x_scale_max = 10**(x_max + diff)
			x_model = 10**np.linspace(x_min - diff, x_max + diff, 100)
		elif self.dist.name in ('Beta'):
			x_min = np.min(x_)
			x_max = np.max(x_)
			x_scale_min = 0
			x_scale_max = 1
			vals_non_sig = np.linspace(x_scale_min, x_scale_max, 11)[1:-1]
			x_minor_ticks = np.linspace(x_scale_min, x_scale_max, 22)[1:-1]
			x_model = np.linspace(x_scale_min, x_scale_max, 102)[1:-1]
		elif self.dist.name in ('Uniform'):
			x_min = np.min(self.params)
			x_max = np.max(self.params)
			x_scale_min = x_min
			x_scale_max = x_max
			vals_non_sig = np.linspace(x_scale_min, x_scale_max, 11)[1:-1]
			x_minor_ticks = np.linspace(x_scale_min, x_scale_max, 22)[1:-1]
			x_model = np.linspace(x_scale_min, x_scale_max, 102)[1:-1]
		else:
			x_min = np.min(x_)
			x_max = np.max(x_)
			vals_non_sig = np.linspace(x_min, x_max, 7)
			x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
			diff = (x_max - x_min) / 10
			x_scale_min = x_min - diff
			x_scale_max = x_max + diff
			x_model = np.linspace(x_scale_min, x_scale_max, 100)

		if self.dist.name in ['Weibull3p']:
			cdf = self.ff(x_model + self.params[2])
		else:
			cdf = self.ff(x_model)

		x_ticks = _round_vals(vals_non_sig)

		y_ticks = np.array(self.dist.y_ticks)
		y_ticks = y_ticks[np.where((y_ticks > y_scale_min) & 
								   (y_ticks < y_scale_max))[0]]
		y_ticks_labels = [str(int(y))+'%' if (re.match('([0-9]+\.0+)', str(y)) is not None) & 
							(y > 1) else str(y) for y in y_ticks * 100]

		if self.dist.name in ['Weibull3p']:		
			x_ticks_labels = [str(int(x)) if (re.match('([0-9]+\.0+)', str(x)) is not None) & 
							(x > 1) else str(x) for x in _round_vals(x_ticks + self.params[2])]
		else:
			x_ticks_labels = [str(int(x)) if (re.match('([0-9]+\.0+)', str(x)) is not None) & 
							(x > 1) else str(x) for x in x_ticks]

		# if hasattr(self.dist, 'R_cb') & hasattr(self, 'hess_inv'):
		# 	if self.hess_inv is not None:
		# 		if self.dist.name == 'Weibull3p':
		# 			cbs = 1 - self.dist.R_cb(x_model + self.params[2], *self.params, self.hess_inv, cb=cb)
		# 		else:
		# 			cbs = 1 - self.dist.R_cb(x_model, *self.params, self.hess_inv, cb=cb)
		# 	else:
		# 		cbs = []
		# else:
		# 	cbs = []

		if hasattr(self, 'hess_inv'):
			if (self.hess_inv is not None):
				if self.dist.name == 'Weibull3p':
					# cbs = 1 - self.R_cb(x_model + self.params[2], *self.params, self.hess_inv, cb=cb)
					cbs = 1 - self.R_cb(x_model + self.params[2], cb=cb)
				else:
					cbs = 1 - self.R_cb(x_model, cb=cb)
			else:
				cbs = []
		else:
			cbs = []

		plot_data = {
			'x_scale_min' : x_scale_min,
			'x_scale_max' : x_scale_max,
			'y_scale_min' : y_scale_min,
			'y_scale_max' : y_scale_max,
			'y_ticks' : y_ticks,
			'y_ticks_labels' : y_ticks_labels,
			'x_ticks' : x_ticks,
			'x_ticks_labels' : x_ticks_labels,
			'cdf' : cdf,
			'x_model' : x_model,
			'x_minor_ticks' : x_minor_ticks,
			'cbs' : cbs,
			'x_scale' : self.dist.plot_x_scale,
			'x_' : x_,
			'F' : F
		}
		return plot_data

	def plot(self, heuristic='Turnbull', plot_bounds=True, cb=0.05):
		r"""
		A method to do a probability plot

		Parameters
		----------
		heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean', 'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None', 'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier', 'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
			The method that the plotting point on the probablility plot will be calculated.

		plot_bounds : Boolean, optional
			A Boolean value to indicate whehter you want the probability bounds to be calculated.

		cb : float, optional
			The confidence with which the confidence bounds, if able, will be calculated. Defaults to 0.95.

		Returns
		-------
		plot : list
			list of a matplotlib plot object

		Examples
		--------
		>>> from surpyval import Weibull
		>>> x = Weibull.random(100, 10, 3)
		>>> model = Weibull.fit(x)
		>>> model.plot()
		"""
		d = self.get_plot_data(heuristic=heuristic, cb=cb)
		# MAKE THE PLOT
		# Set the y limits
		plt.gca().set_ylim([d['y_scale_min'], d['y_scale_max']])
		
		# Set the x scale
		plt.xscale(d['x_scale'])
		# Set the y scale
		plt.gca().set_yscale('function',
			functions=(lambda x : self.dist.mpp_y_transform(x, *self.params), 
					   lambda x : self.dist.mpp_inv_y_transform(x, *self.params)))
		# if self.dist.name == 'Gamma':
		# 	# The y scale for the gamma distribution is dependent on the shape
		# 	plt.gca().set_yscale('function',
		# 		functions=(lambda x : self.dist.mpp_y_transform(x, self.params[0]),
		# 				   lambda x : self.dist.mpp_inv_y_transform(x, self.params[0])))
		# elif self.dist.name == 'Beta':
		# 	# The y scale for the beta distribution is dependent on the shape
		# 	plt.gca().set_yscale('function',
		# 		functions=(lambda x : self.dist.mpp_y_transform(x, *self.params),
		# 				   lambda x : self.dist.mpp_inv_y_transform(x, *self.params)))
		# elif self.dist.name == 'ExpoWeibull':
		# 	# The y scale for the beta distribution is dependent on the shape
		# 	plt.gca().set_yscale('function',
		# 		functions=(lambda x : self.dist.mpp_y_transform(x, *self.params),
		# 				   lambda x : self.dist.mpp_inv_y_transform(x, *self.params)))
		# else:
		# 	plt.gca().set_yscale('function', 
		# 		functions=(self.dist.mpp_y_transform, 
		# 				   self.dist.mpp_inv_y_transform))
		
		# Set Major Y axis ticks
		plt.yticks(d['y_ticks'], labels=d['y_ticks_labels'])
		# Set Minor Y axis ticks
		plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 51)))
		# Set Major X axis ticks
		plt.xticks(d['x_ticks'], labels=d['x_ticks_labels'])
		# Set Minor X axis ticks if log scale.
		if d['x_scale'] == 'log':
			plt.gca().set_xticks(d['x_minor_ticks'], minor=True)
			plt.gca().set_xticklabels([], minor=True)

		# Turn on the grid
		plt.grid(b=True, which='major', color='g', alpha=0.4, linestyle='-')
		plt.grid(b=True, which='minor', color='g', alpha=0.1, linestyle='-')

		# Label it
		plt.title('{} Probability Plot'.format(self.dist.name))
		plt.ylabel('CDF')
		plt.scatter(d['x_'], d['F'])
		plt.gca().set_xlim([d['x_scale_min'], d['x_scale_max']])
		if plot_bounds & (len(d['cbs']) != 0):
			plt.plot(d['x_model'], d['cbs'], color='r')
		return plt.plot(d['x_model'], d['cdf'], color='k', linestyle='--')

class _Parametric():
	def init(self, params, offset):
		"""
		Draft to use when ready.
		"""
		self.offset = offset
		if offset:
			self.gamma = params[0]
			self.params = params[1::]
			self.off = lambda x : x - self.gamma
			self.off_inv = lambda x : x + self.gamma
		else:
			self.params = params
			self.off = lambda x : x
			self.off_inv = lambda x : x

	def __str__(self):
		return 'Parametric Surpyval model with {dist} distribution fitted by {method} yielding parameters {params}'.format(dist=self.dist.name, method=self.method, params=self.params)

	def sf(self, x):
		return self.dist.sf(self.off(x), *self.params)

	def ff(self, x):
		return self.dist.ff(self.off(x), *self.params)

	def df(self, x): 
		return self.dist.df(self.off(x), *self.params)

	def hf(self, x):
		return self.dist.hf(self.off(x), *self.params)

	def Hf(self, x):
		return self.dist.Hf(self.off(x), *self.params)

	def qf(self, p):
		return self.off_inv(self.dist.qf(p, *self.params))

	def cs(self, x, X):
		return self.dist.cs(self.off(x), X, *self.params)

	def random(self, size):
		return self.off_inv(self.dist.qf(uniform.rvs(size=size), *self.params))

	def mean(self):
		return self.dist.mean(*self.params, offset=self.offset)

	def moment(self):
		return self.dist.moment(*self.params, offset=self.offset)

	def entropy(self):
		return self.dist.entropy(*self.params, offset=self.offset)

	# def cb(self, x, sig):
	# 	if self.method != 'MLE':
	# 		raise Exception('Only MLE has confidence bounds')
			
	# 	du = z(sig) * np.sqrt(var_u)
	# 	u_u = u_hat + du
	# 	u_l = u_hat - du
	# 	return np.vstack([np.exp(-np.exp(u_u)), np.exp(-np.exp(u_l))])

	# def R_cb(self, x, cb=0.05):
	# 	return self.dist.R_cb(x, *self.params, self.hess_inv, cb=cb)

	def R_cb(self, t, cb=0.05):
		"""
		Nailed this. Can be used elsewhere if needed
		"""
		if hasattr(self.dist, 'R_cb'):
			return self.dist.R_cb(t, *self.params, self.hess_inv, cb=cb)

		sf_func = lambda params : self.dist.sf(t, *params)

		pvars = self.hess_inv[np.triu_indices(self.hess_inv.shape[0])]
		with np.errstate(all='ignore'):
			jac = jacobian(sf_func)(np.array(self.params))


			
		var_u = []
		for i, j in enumerate(jac):
			j = np.atleast_2d(j).T * j
			j = j[np.triu_indices(j.shape[0])] 
			var_u.append(np.sum(j * pvars))
		diff = z(cb/2) * np.sqrt(np.array(var_u)) * np.array([1., -1.]).reshape(2, 1)
		R_hat = self.sf(t)
		exponent = diff/(R_hat*(1 - R_hat))
		R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
		return R_cb.T

	def ll(self):
		if hasattr(self, 'log_like'):
			return self.log_like
		else:
			x = self.data['x']
			c = self.data['c']
			n = self.data['n']
			t = self.data['t']
			self.log_like = -self.dist.neg_ll(x, c, n, t, *self.params)
			return self.log_like

	def aic(self):
		if hasattr(self, 'aic_'):
			return self.aic_
		else:
			x = self.data['x']
			c = self.data['c']
			n = self.data['n']
			t = self.data['t']
			k = len(self.params)
			self.aic_ = 2 * k + 2 * self.dist.neg_ll(x, c, n, t, *self.params)
			return self.aic_

	def aic_c(self):
		if hasattr(self, 'aic_c_'):
			return self.aic_c_
		else:
			k = len(self.params)
			n = self.data['n'].sum()
			self.aic_c_ = self.aic() + (2*k**2 + 2*k)/(n - k - 1)
			return self.aic_c_

	def get_plot_data(self, heuristic='Turnbull', cb=0.05):
		"""
		Looking a little less ugly now. But not great
		"""
		x = self.data['x']
		x_, r, d, F = nonp.plotting_positions(
			x=self.data['x'], 
			c=self.data['c'], 
			n=self.data['n'], 
			heuristic=heuristic)

		mask = np.isfinite(x_)
		x_ = x_[mask]
		r  =  r[mask]
		d  =  d[mask]
		F  =  F[mask]

		# This needs to be changed to find the F at lowest and highest x (not t)
		if self.data['t'] is not None:
			Ftl = self.ff(np.min(self.data['t'][:, 0]))
			Ftr = self.ff(np.max(self.data['t'][:, 1]))
			F = Ftl + F * (Ftr - Ftl)

		if self.dist.name in ['Weibull3p']:
			x_ = x_ - self.params[2]
			x  = x  - self.params[2]

		y_scale_min = np.min(F[F > 0])/2
		#y_scale_max = np.max(F[F < 1]) + (1 - np.max(F[F < 1]))/2
		y_scale_max = (1 - (1 - np.max(F[F < 1]))/10)

		# x-axis
		if self.dist.plot_x_scale == 'log':
			log_x = np.log10(x_[x_ > 0])
			x_min = np.min(log_x)
			x_max = np.max(log_x)
			vals_non_sig = 10 ** np.linspace(x_min, x_max, 7)
			x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
			x_minor_ticks = ((10**x_minor_ticks * np.array(np.arange(1, 11))
													.reshape((10, 1)))
													.flatten())
			diff = (x_max - x_min)/10
			x_scale_min = 10**(x_min - diff)
			x_scale_max = 10**(x_max + diff)
			x_model = 10**np.linspace(x_min - diff, x_max + diff, 100)
		else:
			x_min = np.min(x_)
			x_max = np.max(x_)
			vals_non_sig = np.linspace(x_min, x_max, 7)
			x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
			diff = (x_max - x_min) / 10
			x_scale_min = x_min - diff
			x_scale_max = x_max + diff
			x_model = np.linspace(x_scale_min, x_scale_max, 100)

		if self.dist.name in ['Weibull3p']:
			cdf = self.ff(x_model + self.params[2])
		else:
			cdf = self.ff(x_model)

		x_ticks = _round_vals(vals_non_sig)

		y_ticks = np.array(self.dist.y_ticks)
		y_ticks = y_ticks[np.where((y_ticks > y_scale_min) & 
								   (y_ticks < y_scale_max))[0]]
		y_ticks_labels = [str(int(y))+'%' if (re.match('([0-9]+\.0+)', str(y)) is not None) & 
							(y > 1) else str(y) for y in y_ticks * 100]

		if self.dist.name in ['Weibull3p']:		
			x_ticks_labels = [str(int(x)) if (re.match('([0-9]+\.0+)', str(x)) is not None) & 
							(x > 1) else str(x) for x in _round_vals(x_ticks + self.params[2])]
		else:
			x_ticks_labels = [str(int(x)) if (re.match('([0-9]+\.0+)', str(x)) is not None) & 
							(x > 1) else str(x) for x in x_ticks]

		# if hasattr(self.dist, 'R_cb') & hasattr(self, 'hess_inv'):
		# 	if self.hess_inv is not None:
		# 		if self.dist.name == 'Weibull3p':
		# 			cbs = 1 - self.dist.R_cb(x_model + self.params[2], *self.params, self.hess_inv, cb=cb)
		# 		else:
		# 			cbs = 1 - self.dist.R_cb(x_model, *self.params, self.hess_inv, cb=cb)
		# 	else:
		# 		cbs = []
		# else:
		# 	cbs = []

		if hasattr(self, 'hess_inv'):
			if self.dist.name == 'Weibull3p':
				# cbs = 1 - self.R_cb(x_model + self.params[2], *self.params, self.hess_inv, cb=cb)
				cbs = 1 - self.R_cb(x_model + self.params[2], cb=cb)
			else:
				cbs = 1 - self.R_cb(x_model, cb=cb)
		else:
			cbs = []

		plot_data = {
			'x_scale_min' : x_scale_min,
			'x_scale_max' : x_scale_max,
			'y_scale_min' : y_scale_min,
			'y_scale_max' : y_scale_max,
			'y_ticks' : y_ticks,
			'y_ticks_labels' : y_ticks_labels,
			'x_ticks' : x_ticks,
			'x_ticks_labels' : x_ticks_labels,
			'cdf' : cdf,
			'x_model' : x_model,
			'x_minor_ticks' : x_minor_ticks,
			'cbs' : cbs,
			'x_scale' : self.dist.plot_x_scale,
			'x_' : x_,
			'F' : F
		}
		return plot_data

	def plot(self, heuristic='Turnbull', plot_bounds=True, cb=0.05):
		d = self.get_plot_data(heuristic=heuristic, cb=cb)
		# MAKE THE PLOT
		# Set the y limits
		plt.gca().set_ylim([d['y_scale_min'], d['y_scale_max']])
		
		# Set the x scale
		plt.xscale(d['x_scale'])
		# Set the y scale
		if self.dist.name == 'Gamma':
			# The y scale for the gamma distribution is dependent on the shape
			plt.gca().set_yscale('function',
				functions=(lambda x : self.dist.mpp_y_transform(x, self.params[0]),
						   lambda x : self.dist.mpp_inv_y_transform(x, self.params[0])))
		elif self.dist.name == 'Beta':
			# The y scale for the beta distribution is dependent on the shape
			plt.gca().set_yscale('function',
				functions=(lambda x : self.dist.mpp_y_transform(x, *self.params),
						   lambda x : self.dist.mpp_inv_y_transform(x, *self.params)))
		else:
			plt.gca().set_yscale('function', 
				functions=(self.dist.mpp_y_transform, 
						   self.dist.mpp_inv_y_transform))
		
		# Set Major Y axis ticks
		plt.yticks(d['y_ticks'], labels=d['y_ticks_labels'])
		# Set Minor Y axis ticks
		plt.gca().yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 51)))
		# Set Major X axis ticks
		plt.xticks(d['x_ticks'], labels=d['x_ticks_labels'])
		# Set Minor X axis ticks if log scale.
		if d['x_scale'] == 'log':
			plt.gca().set_xticks(d['x_minor_ticks'], minor=True)
			plt.gca().set_xticklabels([], minor=True)

		# Turn on the grid
		plt.grid(b=True, which='major', color='g', alpha=0.4, linestyle='-')
		plt.grid(b=True, which='minor', color='g', alpha=0.1, linestyle='-')

		# Label it
		plt.title('{} Probability Plot'.format(self.dist.name))
		plt.ylabel('CDF')
		plt.scatter(d['x_'], d['F'])
		plt.gca().set_xlim([d['x_scale_min'], d['x_scale_max']])
		if plot_bounds & (len(d['cbs']) != 0):
			plt.plot(d['x_model'], d['cbs'], color='r')
		return plt.plot(d['x_model'], d['cdf'], color='k', linestyle='--')



