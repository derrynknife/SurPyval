import autograd.numpy as np

# Functions Supporting the transformation of parameters for optimisation.
def pass_through(x):
	return x

def adj_relu(x):
	return np.where(x >= 0, x + 1, np.exp(x))

def inv_adj_relu(x):
	return np.where(x >= 1, x - 1, np.log(x))

def rev_adj_relu(x):
	return -np.where(x >= 0, x + 1, np.exp(x))

def inv_rev_adj_relu(x):
	return np.where(x < -1, -x - 1, np.log(-x))


def bounds_convert(x, bounds):
	funcs = []
	inv_f = []
	for l, u in bounds:
		if (l is None) and (u is None):
			funcs.append(lambda x : pass_through(x))
			inv_f.append(lambda x : pass_through(x))
		elif (u is None):
			funcs.append(lambda x : inv_adj_relu(x))
			inv_f.append(lambda x : adj_relu(x))
		elif (l is None):
			if u != 0:
				upper = np.min(x)
			else:
				upper = 0
			funcs.append(lambda x : inv_rev_adj_relu(x - upper))
			inv_f.append(lambda x : upper + rev_adj_relu(x))

	transform = lambda params : np.array([f(p) for p, f in zip(params, funcs)])
	inv_trans = lambda params : np.array([f(p) for p, f in zip(params, inv_f)])

	return transform, inv_trans, funcs, inv_f


# Functions to support fixing parameters
def fix_idx_and_function(dist, fixed, param_map, offset_index_inc, funcs):
	if fixed is not None:
		"""
		Record to the model that parameters were fixed
		"""
		fixed_idx = [param_map[x] + offset_index_inc for x in fixed.keys()]
		not_fixed = np.array([x for x in range(dist.k + offset_index_inc) if x not in fixed_idx])

		def constraints(p):
			params = [0] * (dist.k + offset_index_inc)
			# params = np.empty(self.k + offset_index_inc)
			for k, v in fixed.items():
				params[param_map[k] + offset_index_inc] = funcs[param_map[k] + offset_index_inc](v)
			for i, v in zip(not_fixed, p):
				params[i] = v
			return np.array(params)

		const = constraints
	else:
		const = lambda x : x
		fixed_idx = []
		not_fixed = np.array([x for x in range(dist.k + offset_index_inc)])

	return const, fixed_idx, not_fixed