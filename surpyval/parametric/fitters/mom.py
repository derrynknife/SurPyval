import autograd.numpy as np
from scipy.optimize import minimize

def mom(dist, x, n, init, offset):
	"""
	MOM: Method of Moments.

	This is one of the simplest ways to calculate the parameters of a distribution.

	This method is quick but only works with uncensored data.
	"""
	x_ = np.repeat(x, n)

	if offset:
		k = dist.k + 1
		bounds = ((None, np.min(x)), *dist.bounds)
	else:
		k = dist.k
		bounds = dist.bounds
	moments = np.zeros(k)

	for i in range(0, k):
		moments[i] = np.sum(x_**(i+1)) / len(x_)

	fun = lambda params : np.sum((moments - dist.mom_moment_gen(*params, offset=offset))**2)
	res = minimize(fun, init, bounds=bounds)
	return res