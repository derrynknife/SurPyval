import autograd.numpy as np
from scipy.optimize import minimize

def mom(dist, x, n, init, offset):
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