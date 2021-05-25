import numpy as np
from surpyval import nonparametric as nonp
from surpyval.utils import xcn_handler

class NonParametricFitter():
	"""
	Class to capture all data and meta data on non-parametric sur(py)val model

	Needs to have:
	f = None - or empirical
	confidence bounds

	TODO: add confidence bounds
	standard: h_u, H_u or Ru, Rl
	"""

	def fit(self, x, c=None, n=None, sig=0.05):
		x, c, n = xcn_handler(x, c, n)

		data = {}
		data['x'] = x
		data['c'] = c
		data['n'] = n
		out = nonp.NonParametric()
		out.data = data
		out.model = self.how
		
		x_, r, d, R = nonp.FIT_FUNCS[self.how](x, c=c, n=n)

		out.x = x_
		out.max_x = np.max(out.x)
		out.r = r
		out.d = d
		with np.errstate(divide='ignore'):
			out.H = -np.log(R)
		out.R = R
		out.F = 1 - out.R

		with np.errstate(divide='ignore'):
			var = out.d / (out.r * (out.r - out.d))
		
		with np.errstate(invalid='ignore'):
			greenwood = np.cumsum(var)
		out.greenwood = greenwood

		return out