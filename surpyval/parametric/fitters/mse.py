import autograd.numpy as np

from surpyval import nonparametric as nonp
from scipy.optimize import minimize

def mse(dist, x, c, n, t, init, const, trans, inv_fs, fixed_idx, offset):
	"""
	MSE: Mean Square Error
	This is simply fitting the curve to the best estimate from a non-parametric estimate.

	This is slightly different in that it fits it to untransformed data on the x and 
	y axis. The MPP method fits the curve to the transformed data. This is simply fitting
	a the CDF sigmoid to the nonparametric estimate.
	"""

	# Need to make the Turnbull estimate much much much faster (and less memory before I unlock this)
	# x_, r, d, R = nonp.turnbull(x, c, n, t, estimator='Nelson-Aalen')
	x_, r, d, R = nonp.nelson_aalen(x, c, n)

	F = 1 - R
	mask = np.isfinite(x_)
	F  = F[mask]
	x_ = x_[mask]
	
	fun = lambda params : np.sum(((dist.ff(x_, *inv_fs(const(params)))) - F)**2)
	res = minimize(fun, init)

	results = {}
	results['res'] = res
	results['params'] = inv_fs(const(res.x))

	return results