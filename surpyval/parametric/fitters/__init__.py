from surpyval import np

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
    for i, (lower, upper) in enumerate(bounds):
        def add_to_funcs(low, upp, i):
            if (low is None) and (upp is None):
                funcs.append(lambda x: pass_through(x))
                inv_f.append(lambda x: pass_through(x))
            elif (low == 0) and (upp == 1):
                D = 10
                funcs.append(lambda x: D * np.arctanh((2 * x) - 1))
                inv_f.append(lambda x: (np.tanh(x / D) + 1) / 2)
            elif (upp is None):
                funcs.append(lambda x: (inv_adj_relu(x - low)))
                inv_f.append(lambda x: (adj_relu(x) + low))
            elif (low is None):
                funcs.append(lambda x: inv_rev_adj_relu(x - np.copy(upp)))
                inv_f.append(lambda x: np.copy(upp) + rev_adj_relu(x))
            else:
                funcs.append(lambda x: pass_through(x))
                inv_f.append(lambda x: pass_through(x))
        add_to_funcs(lower, upper, i)

    def transform(params):
        return np.array([f(p) for p, f in zip(params, funcs)])

    def inv_trans(params):
        return np.array([f(p) for p, f in zip(params, inv_f)])

    return transform, inv_trans, funcs, inv_f


def fix_idx_and_function(fixed, param_map, funcs):
    n_params = len(param_map)
    if fixed is not None:
        """
        Record to the model that parameters were fixed
        """
        fixed_idx = [param_map[x] for x in fixed.keys()]
        not_fixed = [x for x in range(n_params) if x not in fixed_idx]
        not_fixed = np.array(not_fixed)

        def constraints(p):
            params = [0] * (n_params)
            for k, v in fixed.items():
                params[param_map[k]] = funcs[param_map[k]](v)
            for i, v in zip(not_fixed, p):
                params[i] = v
            return np.array(params)

        const = constraints
    else:

        def const(x):
            return x

        fixed_idx = []
        not_fixed = np.array([x for x in range(n_params)])

    return const, fixed_idx, not_fixed
