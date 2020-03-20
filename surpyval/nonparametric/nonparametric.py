import numpy as np
import surpyval.nonparametric as nonp
from scipy.stats import t, norm
from .kaplan_meier import KaplanMeier
from .nelson_aalen import NelsonAalen
from .fleming_harrington import FlemingHarrington_

class NonParametric():
	"""
	Class to capture all data and meta data on non-parametric sur(py)val model

	Needs to have:
	f = None - or empirical
	confidence bounds

	TODO: add confidence bounds
	standard: h_u, H_u or Ru, Rl

	"""
	def __str__(self):
		return "{model} survival model".format(model=self.model)

	def sf(self, x, how='step'):
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
		return 1 - self.sf(x, how=how)

	def hf(self, x, how='step'):
		return np.diff(self.Hf(x, how=how))

	def Hf(self, x, how='step'):
		H = -np.log(self.sf(x, how=how))
		H[H == 0] = 0
		return H

	def R_cb(self, x, bound='upper', how='step', confidence=0.95, bound_type='exp', dist='t'):
		# Greenwoods variance using t-stat. Ref found:
		# http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis
		assert bound_type in ['exp', 'normal']
		assert dist in ['t', 'z']
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
			R_out[np.where(x < 0)] = np.nan
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
		return R_out

	def random(self, size):
		return np.random.choice(self.x, size=size)