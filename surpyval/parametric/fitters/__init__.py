import autograd.numpy as np


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
    for lower, upper in bounds:
        if (lower is None) and (upper is None):
            funcs.append(lambda x: pass_through(x))
            inv_f.append(lambda x: pass_through(x))
        elif (upper is None):
            funcs.append(lambda x: inv_adj_relu(x))
            inv_f.append(lambda x: adj_relu(x))
        elif (lower == 0) and (upper == 0):
            funcs.append(lambda x: 1000 * np.arctanh((2 * x)-1))
            inv_f.append(lambda x: (np.tanh(x/1000) + 1)/2)
        elif (lower is None):
            if upper != 0:
                limit = np.min(x)
            else:
                limit = 0
            funcs.append(lambda x: inv_rev_adj_relu(x - limit))
            inv_f.append(lambda x: limit + rev_adj_relu(x))
        else:
            funcs.append(lambda x: pass_through(x))
            inv_f.append(lambda x: pass_through(x))

    def transform(params):
        return np.array([f(p) for p, f in zip(params, funcs)])

    def inv_trans(params):
        return np.array([f(p) for p, f in zip(params, inv_f)])

    return transform, inv_trans, funcs, inv_f


# Functions to support fixing parameters
def fix_idx_and_function(dist, fixed, param_map, offset_index_inc, funcs):
    if fixed is not None:
        """
        Record to the model that parameters were fixed
        """
        fixed_idx = [param_map[x] + offset_index_inc for x in fixed.keys()]
        not_fixed = [x for x in
                     range(dist.k + offset_index_inc) if x not in fixed_idx]
        not_fixed = np.array(not_fixed)

        def constraints(p):
            params = [0] * (dist.k + offset_index_inc)
            # params = np.empty(self.k + offset_index_inc)
            for k, v in fixed.items():
                params[param_map[k] + offset_index_inc] = (
                    funcs[param_map[k] + offset_index_inc](v))
            for i, v in zip(not_fixed, p):
                params[i] = v
            return np.array(params)

        const = constraints
    else:

        def const(x):
            return x

        fixed_idx = []
        not_fixed = np.array([x for x in range(dist.k + offset_index_inc)])

    return const, fixed_idx, not_fixed
