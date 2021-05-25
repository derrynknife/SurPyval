import autograd.numpy as np

from surpyval import nonparametric as nonp
from scipy.optimize import minimize

def mse(dist, x, c, n, init):
	"""
	MSE: Mean Square Error
	This is simply fitting the curve to the best estimate from a non-parametric estimate.

	This is slightly different in that it fits it to untransformed data on the x and 
	y axis. The MPP method fits the curve to the transformed data. This is simply fitting
	a the CDF sigmoid to the nonparametric estimate.
	"""

	x_, r, d, R = nonp.turnbull(x, c, n, estimator='Nelson-Aalen')

	F = 1 - R
	mask = np.isfinite(x_)
	F  = F[mask]
	x_ = x_[mask]
	
	fun = lambda params : np.sum(((dist.ff(x_, *params)) - F)**2)
	res = minimize(fun, init, bounds=dist.bounds)

	return res