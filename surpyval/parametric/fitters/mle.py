import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy.linalg import inv
from autograd.numpy.linalg import pinv

from scipy.optimize import minimize
from scipy.optimize import approx_fprime

import surpyval
import warnings
import sys

def mle(dist, x, c, n, t, const, trans, inv_fs, init, fixed_idx, offset):
		"""
		MLE: Maximum Likelihood estimate
		"""
		old_err_state = np.seterr(invalid='raise', divide='raise', over='ignore', under='ignore')

		if t is None:
			if offset:
				fun = lambda params: dist.neg_ll(x - inv_fs(const(params))[0], c, n, *inv_fs(const(params))[1::])
				# fun_hess = lambda params: dist.neg_ll(x - const(psarams)[0], c, n, *const(params)[1::])
				fun_hess = lambda params: dist.neg_ll(x - params[0], c, n, *params[1::])
			else:
				fun = lambda params: dist.neg_ll(x, c, n, *inv_fs(const(params)))
				fun_hess = lambda params: dist.neg_ll(x, c, n, *params)
				# fun_hess = lambda params: dist.neg_ll(x, c, n, *const(params))
		else:
			# fun = lambda params: dist.neg_ll_trunc(x, c, n, t, *inv_fs(const(params)))
			# fun_hess = lambda params: dist.neg_ll(x, c, n, t, *params)
			if offset:
				fun = lambda params: dist.neg_ll_trunc(x - inv_fs(const(params))[0], c, n, t, *inv_fs(const(params))[1::])
				# fun_hess = lambda params: dist.neg_ll(x - const(psarams)[0], c, n, *const(params)[1::])
				fun_hess = lambda params: dist.neg_ll_trunc(x - params[0], c, n, t, *params[1::])
			else:
				fun = lambda params: dist.neg_ll_trunc(x, c, n, t, *inv_fs(const(params)))
				fun_hess = lambda params: dist.neg_ll_trunc(x, c, n, t, *params)

		if hasattr(dist, 'mle'):
			return dist.mle(x, c, n, t, const, trans, inv_fs, init, fixed_idx, offset)

		try:
			jac  = jacobian(fun)
			hess = hessian(fun)
			res  = minimize(fun, init,
							method='Newton-CG', 
							jac=jac, 
							hess=hess, 
							tol=1e-15)
		except:
			print("Autodifferentiation with hessian failed, trying without hessian", file=sys.stderr)
			try:
				jac  = jacobian(fun)
				res = minimize(fun, init, method='BFGS', jac=jac)
			except:
				print("MLE FAILED: Likelihood function appears undefined; try alternate estimation method", file=sys.stderr)
				return None, jac, None, None

		p_hat = inv_fs(const(res.x))
		hess_inv = inv(hessian(fun_hess)(p_hat))

		np.seterr(**old_err_state)

		return res, jac, hess_inv, p_hat