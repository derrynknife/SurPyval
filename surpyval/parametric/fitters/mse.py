import autograd.numpy as np

from surpyval import nonparametric as nonp
from scipy.optimize import minimize

from autograd import jacobian, hessian, value_and_grad

def mse(dist, x, c, n, t, init, const, trans, inv_fs, fixed_idx, offset):
	"""
	MSE: Mean Square Error
	This is simply fitting the curve to the best estimate from a non-parametric estimate.

	This is slightly different in that it fits it to untransformed data on the x and 
	y axis. The MPP method fits the curve to the transformed data. This is simply fitting
	a the CDF sigmoid to the nonparametric estimate.
	"""

	# Need to make the Turnbull estimate much much much faster (and less memory before I unlock this)
	if (-1 in c) or (2 in c):
		x_, r, d, R = nonp.turnbull(x, c, n, t, estimator='Nelson-Aalen')
	else:
		x_, r, d, R = nonp.nelson_aalen(x, c, n)

	F = 1 - R
	mask = np.isfinite(x_)
	F  = F[mask]
	x_ = x_[mask]

	fun = lambda params : np.sum(((dist.ff(x_, *inv_fs(const(params)))) - F)**2)
	jac = jacobian(fun)
	hess = hessian(fun)

	try:
		res = minimize(fun, init, method='Newton-CG', jac=jac, hess=hess)
	except:
		try:
			res = minimize(fun, init, method='BFGS', jac=jac)
		except:
			res = minimize(fun, init)

	results = {}
	results['res'] = res
	results['params'] = inv_fs(const(res.x))

	return results