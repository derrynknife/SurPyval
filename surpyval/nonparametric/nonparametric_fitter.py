import numpy as np
from surpyval import nonparametric as nonp
from surpyval.utils import xcnt_handler

class NonParametricFitter():
	"""
	Class to capture all data and meta data on non-parametric sur(py)val model

	Needs to have:
	f = None - or empirical
	confidence bounds

	TODO: add confidence bounds
	standard: h_u, H_u or Ru, Rl
	"""

	def fit(self, x, c=None, n=None, t=None,
			xl=None, xr=None, tl=None, tr=None,
			estimator='Fleming-Harrington'):


		x, c, n, t = xcnt_handler(x, c, n, t=t, tl=tl, tr=tr)

		data = {}
		data['x'] = x
		data['c'] = c
		data['n'] = n
		data['t'] = t
		out = nonp.NonParametric()
		
		x_, r, d, R = nonp.FIT_FUNCS[self.how](**data)

		out.data = data
		out.model = self.how
		out.x = x_
		out.max_x = np.max(out.x)
		out.r = r
		out.d = d
		with np.errstate(all='ignore'):
			out.H = -np.log(R)
		out.R = R
		out.F = 1 - out.R

		with np.errstate(all='ignore'):
			var = out.d / (out.r * (out.r - out.d))
		
		with np.errstate(all='ignore'):
			greenwood = np.cumsum(var)
		out.greenwood = greenwood

		return out