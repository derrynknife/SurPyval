import re
from autograd import jacobian
import autograd.numpy as np
from scipy.stats import uniform

import surpyval
from scipy.special import ndtri as z
from surpyval import nonparametric as nonp

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator

def round_vals(x):
	not_different = True
	i = 1
	while not_different:
		x_ticks = np.array(surpyval.round_sig(x, i))
		not_different = (np.diff(x_ticks) == 0).any()
		i += 1
	return x_ticks

class OffsetParametric():
	def __str__(self):
		return 'Offset Parametric Surpyval model with {dist} distribution fitted by {method} yielding parameters {params} with offset of {gamma}'.format(dist=self.dist.name, method=self.method, params=self.params, gamma=self.gamma)

	def sf(self, x):
		return self.dist.sf(x - self.gamma, *self.params)

	def ff(self, x):
		return self.dist.ff(x - self.gamma, *self.params)

	def df(self, x): 
		return self.dist.df(x - self.gamma, *self.params)

	def hf(self, x):
		return self.dist.hf(x - self.gamma, *self.params)

	def Hf(self, x):
		return self.dist.Hf(x - self.gamma, *self.params)

	def qf(self, p):
		return self.dist.qf(p, *self.params) + self.gamma

	def cs(self, x, X):
		return self.dist.cs(x - self.gamma, X, *self.params)

	def random(self, size):
		return self.dist.qf(uniform.rvs(size=size), *self.params) + self.gamma

	def mean(self):
		return self.dist.mean(*self.params) + self.gamma

	def moment(self):
		return self.dist.moment(*self.params)

	def entropy(self):
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
		Need to add an H_cb function that can be used to convert to any
		other function, i.e. hf, df, ff, sf OR....
		Use this to convert. grad(log(R)) would work!
		"""
		sf_func = lambda params : self.dist.sf(t - params[0], *params[1::])

		pvars = self.hess_inv[np.triu_indices(self.hess_inv.shape[0])]
		with np.errstate(all='ignore'):
			jac = jacobian(sf_func)(np.array([self.gamma, *self.params]))

			
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
			self.log_like = -self.dist.neg_ll(x - self.gamma, c, n, t, *self.params)
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
			self.aic_ = 2 * k + 2 * self.dist.neg_ll(x - self.gamma, c, n, t, *self.params)
			return self.aic_

	def aic_c(self):
		if hasattr(self, 'aic_c_'):
			return self.aic_c_
		else:
			k = len(self.params) + 1
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
		if np.isfinite(self.data['t']).any():
			Ftl = self.ff(np.min(self.data['t'][:, 0]))
			Ftr = self.ff(np.max(self.data['t'][:, 1]))
			F = Ftl + F * (Ftr - Ftl)

		x_ = x_ - self.gamma
		x  = x  - self.gamma

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

		cdf = self.ff(x_model + self.gamma)

		x_ticks = round_vals(vals_non_sig)

		y_ticks = np.array(self.dist.y_ticks)
		y_ticks = y_ticks[np.where((y_ticks > y_scale_min) & 
								   (y_ticks < y_scale_max))[0]]
		y_ticks_labels = [str(int(y))+'%' if (re.match('([0-9]+\.0+)', str(y)) is not None) & 
							(y > 1) else str(y) for y in y_ticks * 100]

		
		x_ticks_labels = [str(int(x)) if (re.match('([0-9]+\.0+)', str(x)) is not None) & 
						(x > 1) else str(x) for x in round_vals(x_ticks + self.gamma)]

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
			cbs = 1 - self.R_cb(x_model + self.gamma, cb=cb)
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
		plt.title('{} Probability Plot with Offset'.format(self.dist.name))
		plt.ylabel('CDF')
		plt.scatter(d['x_'], d['F'])
		plt.gca().set_xlim([d['x_scale_min'], d['x_scale_max']])
		if plot_bounds & (len(d['cbs']) != 0):
			plt.plot(d['x_model'], d['cbs'], color='r')
		return plt.plot(d['x_model'], d['cdf'], color='k', linestyle='--')

class ParametricO():
	def __init__(self, offset=False):
		self.offset = offset
		if not offset:
			self.gamma = 0

	def __str__(self):
		if self.offset:
			return 'Parametric Surpyval model with {dist} distribution fitted by {method} yielding parameters {params} with offset of {gamma}'.format(dist=self.dist.name, method=self.method, params=self.params, gamma=self.gamma)
		else:
			return 'Parametric Surpyval model with {dist} distribution fitted by {method} yielding parameters {params}'.format(dist=self.dist.name, method=self.method, params=self.params)

	def sf(self, x):
		return self.dist.sf(x - self.gamma, *self.params)

	def ff(self, x):
		return self.dist.ff(x - self.gamma, *self.params)

	def df(self, x): 
		return self.dist.df(x - self.gamma, *self.params)

	def hf(self, x):
		return self.dist.hf(x - self.gamma, *self.params)

	def Hf(self, x):
		return self.dist.Hf(x - self.gamma, *self.params)

	def qf(self, p):
		return self.dist.qf(p, *self.params) + self.gamma

	def cs(self, x, X):
		return self.dist.cs(x - self.gamma, X, *self.params)

	def random(self, size):
		return self.dist.qf(uniform.rvs(size=size), *self.params) + self.gamma

	def mean(self):
		return self.dist.mean(*self.params) + self.gamma

	def moment(self):
		return self.dist.moment(*self.params)

	def entropy(self):
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
		Need to add an H_cb function that can be used to convert to any
		other function, i.e. hf, df, ff, sf OR....
		Use this to convert. grad(log(R)) would work!
		"""
		sf_func = lambda params : self.dist.sf(t - params[0], *params[1::])

		pvars = self.hess_inv[np.triu_indices(self.hess_inv.shape[0])]
		with np.errstate(all='ignore'):
			jac = jacobian(sf_func)(np.array([self.gamma, *self.params]))

			
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
			self.log_like = -self.dist.neg_ll(x - self.gamma, c, n, t, *self.params)
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
			self.aic_ = 2 * k + 2 * self.dist.neg_ll(x - self.gamma, c, n, t, *self.params)
			return self.aic_

	def aic_c(self):
		if hasattr(self, 'aic_c_'):
			return self.aic_c_
		else:
			k = len(self.params) + 1
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

		x_ = x_ - self.gamma
		x  = x  - self.gamma

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

		cdf = self.ff(x_model + self.gamma)

		x_ticks = round_vals(vals_non_sig)

		y_ticks = np.array(self.dist.y_ticks)
		y_ticks = y_ticks[np.where((y_ticks > y_scale_min) & 
								   (y_ticks < y_scale_max))[0]]
		y_ticks_labels = [str(int(y))+'%' if (re.match('([0-9]+\.0+)', str(y)) is not None) & 
							(y > 1) else str(y) for y in y_ticks * 100]

		
		x_ticks_labels = [str(int(x)) if (re.match('([0-9]+\.0+)', str(x)) is not None) & 
						(x > 1) else str(x) for x in round_vals(x_ticks + self.gamma)]

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
			cbs = 1 - self.R_cb(x_model + self.gamma, cb=cb)
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
		plt.title('{} Probability Plot with Offset'.format(self.dist.name))
		plt.ylabel('CDF')
		plt.scatter(d['x_'], d['F'])
		plt.gca().set_xlim([d['x_scale_min'], d['x_scale_max']])
		if plot_bounds & (len(d['cbs']) != 0):
			plt.plot(d['x_model'], d['cbs'], color='r')
		return plt.plot(d['x_model'], d['cdf'], color='k', linestyle='--')