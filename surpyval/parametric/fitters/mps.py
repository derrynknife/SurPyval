from scipy.optimize import minimize
from scipy.optimize import approx_fprime

import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy.linalg import inv
from autograd.numpy.linalg import pinv
import surpyval
import warnings
import sys


def mps_old(dist, x, c, n, init, offset):
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

def mps(dist, x, c, n, const, trans, inv_fs, init, fixed_idx, offset):
	"""
	MPS: Maximum Product Spacing

	This is the method to get the largest (geometric) average distance between all points

	This method works really well when all points are unique. Some complication comes in when using repeated data.

	This method is exceptional for when using three parameter distributions.
	"""

	# Can change this to raise invalid and divide errors when MPS takes
	old_err_state = np.seterr(invalid='ignore', divide='ignore', over='ignore', under='ignore')

	if offset:
		fun = lambda params: dist.neg_mean_D(x - inv_fs(const(params))[0], c, n, *inv_fs(const(params))[1::])
		fun_hess = lambda params: dist.neg_mean_D(x - params[0], c, n, *params[1::])
	else:
		fun = lambda params: dist.neg_mean_D(x, c, n, *inv_fs(const(params)))
		fun_hess = lambda params: dist.neg_mean_D(x, c, n, *params)

	jac  = jacobian(fun)
	hess = hessian(fun)

	try:
		jac  = jacobian(fun)
		hess = hessian(fun)
		res  = minimize(fun, init,
						method='Newton-CG', 
						jac=jac, 
						hess=hess, 
						tol=1e-15)
	except:
		print("MPS with autodiff hessian and jacobian failed, trying without hessian", file=sys.stderr)
		try:
			jac  = jacobian(fun)
			res = minimize(fun, init, method='BFGS', jac=jac)
		except:
			print("MPS with autodiff jacobian failed, trying without jacobian or hessian", file=sys.stderr)
			try:
				res = minimize(fun, init)
			except:
				print("MPS FAILED: Try alternate estimation method", file=sys.stderr)
				np.seterr(**old_err_state)
				return {}

	results = {}
	params = inv_fs(const(res.x))

	if offset:
		results['gamma'] = params[0]
		results['params'] = params[1::]
	else:
		results['params'] = params

	results['hess_inv'] = inv(hessian(fun_hess)(params))
	results['jac'] = jac

	np.seterr(**old_err_state)

	return results










