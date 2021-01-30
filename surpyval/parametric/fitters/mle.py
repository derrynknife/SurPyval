import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy.linalg import inv
from autograd.numpy.linalg import pinv

from scipy.optimize import minimize
from scipy.optimize import approx_fprime

import surpyval

def mle(dist, x, c, n, t, const, trans, inv_fs, init, fixed_idx, offset):
		"""
		MLE: Maximum Likelihood estimate
		"""
		# fail = False
		if t is None:
			if offset:
				fun = lambda params: dist.neg_ll(x - inv_fs(const(params))[0], c, n, *inv_fs(const(params))[1::])
				fun_hess = lambda params: dist.neg_ll(x - params[0], c, n, *params[1::])
			else:
				fun = lambda params: dist.neg_ll(x, c, n, *inv_fs(const(params)))
				fun_hess = lambda params: dist.neg_ll(x, c, n, *params)
		else:
			fun = lambda params: dist.neg_ll_trunc(x, c, n, t, *inv_fs(const(params)))
			fun_hess = lambda params: dist.neg_ll(x, c, n, t, *params)

		if dist.name == 'Uniform':
			p_hat = np.array([np.min(x), np.max(x)])
			jac  = jacobian(fun)
			hess_inv = pinv(hessian(fun_hess)(p_hat))
			return None, jac, hess_inv, p_hat

		try:
			jac  = jacobian(fun)
			hess = hessian(fun)
			res  = minimize(fun, init, 
							method='trust-exact', 
							jac=jac, 
							hess=hess, 
							tol=1e-10)
			hess_inv = inv(res.hess)
		except:
			print("Autograd attempt failed, using without hessian")
			# fail = True

		# if (fail) | (not dist.use_autograd):
			jac = lambda xx : approx_fprime(xx, fun, surpyval.EPS)
			res = minimize(fun, init, method='BFGS', jac=jac)
			hess_inv = res.hess_inv

		p_hat = inv_fs(const(res.x))
		# hess_inv = inv(hessian(fun_hess)(p_hat))


		return res, jac, hess_inv, p_hat