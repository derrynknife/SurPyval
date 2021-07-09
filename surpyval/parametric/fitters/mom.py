import autograd.numpy as np
from autograd import jacobian, hessian

from scipy.optimize import minimize

def mom_old(dist, x, n, init, offset):
	"""
	MOM: Method of Moments.

	This is one of the simplest ways to calculate the parameters of a distribution.

	This method is quick but only works with uncensored data.
	"""
	x_ = np.repeat(x, n)

	if hasattr(dist, '_mom'):
		return {'params' : dist._mom(x)}

	if offset:
		k = dist.k + 1
		bounds = ((None, np.min(x)), *dist.bounds)
	else:
		k = dist.k
		bounds = dist.bounds

	moments = np.zeros(k)

	for i in range(0, k):
		moments[i] = (x_**(i+1)).mean()

	fun = lambda params : np.sum((moments - dist.mom_moment_gen(*params, offset=offset))**2)
	res = minimize(fun, init, bounds=bounds, tol=1e-15)

	results = {}
	if offset:
		results['gamma'] = res.x[0]
		results['params'] = res.x[1::]
	else:
		results['params'] = res.x
	results['res'] = res
	return results

def mom(dist, x, n, init, const, trans, inv_fs, fixed_idx, offset):
	"""
	MOM: Method of Moments.

	This is one of the simplest ways to calculate the parameters of a distribution.

	This method is quick but only works with uncensored data.
	"""
	x_ = np.repeat(x, n)

	if hasattr(dist, '_mom'):
		return {'params' : dist._mom(x)}

	if offset:
		k = dist.k + 1
		bounds = ((None, np.min(x)), *dist.bounds)
	else:
		k = dist.k
		bounds = dist.bounds

	moments = np.zeros(k)

	for i in range(0, k):
		moments[i] = (x_**(i+1)).mean()

	fun = lambda params : np.log(((np.abs(moments - dist.mom_moment_gen(*inv_fs(const(params)), offset=offset))/moments))).sum()
	
	res = minimize(fun, np.array(init), tol=1e-15)

	params = inv_fs(const(res.x))

	results = {}
	if offset:
		results['gamma'] = params[0]
		results['params'] = params[1:]
	else:
		results['params'] = params
	results['res'] = res
	return results
