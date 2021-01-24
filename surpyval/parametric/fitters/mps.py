from scipy.optimize import minimize

def mps(dist, x, c, n, init, offset):
	"""
	MPS: Maximum Product Spacing

	This is the method to get the largest (geometric) average distance between all points

	This method works really well when all points are unique. Some complication comes in when using repeated data.

	This method is exceptional for when using three parameter distributions.
	"""
	if offset:
		k = dist.k + 1
		bounds = ((None, None), *dist.bounds)
	else:
		k = dist.k
		bounds = dist.bounds

	fun = lambda params : dist.neg_mean_D(x, c, n, *params, offset=offset)
	res = minimize(fun, init, bounds=bounds)
	return res