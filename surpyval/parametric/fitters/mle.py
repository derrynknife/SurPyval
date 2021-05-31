import autograd.numpy as np
from autograd import jacobian, hessian
from autograd.numpy.linalg import inv
from autograd.numpy.linalg import pinv
import copy

from scipy.optimize import minimize
from scipy.optimize import approx_fprime

import surpyval
import warnings
import sys

def mle(dist, x, c, n, t, const, trans, inv_fs, init, fixed_idx, offset):
		"""
		MLE: Maximum Likelihood estimate
		"""
		if hasattr(dist, 'mle'):
			return dist.mle(x, c, n, t, const, trans, inv_fs, init, fixed_idx, offset)

		old_err_state = np.seterr(invalid='raise', divide='raise', over='ignore', under='ignore')

		# Need to flag entries where truncation is inf or -inf so that the autograd doesn't fail
		# autograd fails if it encounters any inf, nan, -inf etc even if they don't affect the gradient
		# Very important to the autograd! And it worked, yay!
		t_flags = np.ones_like(t)
		t_mle = copy.copy(t)
		# Create flags to indicate where the truncation values are infinite
		t_flags[:, 0] = np.where(np.isfinite(t[:, 0]), 1, 0)
		t_flags[:, 1] = np.where(np.isfinite(t[:, 1]), 1, 0)
		# Convert the infinite values to a finite value to ensure the autodiff functions don't fail
		t_mle[:, 0] = np.where(t_flags[:, 0] == 1, t[:, 0], 1)
		t_mle[:, 1] = np.where(t_flags[:, 1] == 1, t[:, 1], 1)

		# Create the objective function
		def fun(params, offset=False, transform=True):
			x_mle = np.copy(x)
			if transform:
				params = inv_fs(const(params))
			if offset:
				gamma = params[0]
				params = params[1::]
			else:
				gamma = 0

			if 2 in c:
				l_flag = x_mle[:, 0] <= np.min([gamma, dist.support[0]])
				r_flag = x_mle[:, 1] >= dist.support[1]
				mask = np.vstack([l_flag, r_flag]).T
				inf_c_flags = (mask).astype(int)
				x_mle[(c == 2).reshape(-1, 1) & mask] = 1
			else:
				inf_c_flags = np.zeros_like(x)
			x_mle = x_mle - gamma
			return dist.neg_ll(x_mle, c, n, inf_c_flags, t_mle, t_flags, *params)

		jac  = jacobian(fun)
		hess = hessian(fun)

		try:
			# First attempt to be with with jacobian and hessian from autograd
			res  = minimize(fun, init, args=(offset, True), method='Newton-CG', jac=jac, hess=hess)
			if not res.success:
				raise Exception
		except:
			# If first attempt fails, try a second time with only jacobian from autograd
			print("MLE with autodiff hessian and jacobian failed, trying without hessian", file=sys.stderr)
			try:
				res = minimize(fun, init, args=(offset, True), method='BFGS', jac=jac)
				if not res.success:
					raise Exception
			except:
				# If second attempt fails, try a third time with only jthe objective function
				print("MLE with autodiff jacobian failed, trying without jacobian or hessian", file=sys.stderr)
				try:
					res = minimize(fun, init, args=(offset, True))
					p_hat = inv_fs(const(res.x))
					np.seterr(**old_err_state)
					return res, None, res.hess_inv, p_hat
				except:
					# Something really went wrong
					print("MLE FAILED: Likelihood function appears undefined; try alternate estimation method", file=sys.stderr)
					np.seterr(**old_err_state)
					return None, None, None, None

		p_hat = inv_fs(const(res.x))
		hess_inv = inv(hessian(fun)(p_hat, *(offset, False)))

		np.seterr(**old_err_state)

		return res, jac, hess_inv, p_hat




