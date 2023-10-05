from surpyval import np


def adj_relu(x):
    return np.where(x >= 0, x + 1, np.exp(x))


def inv_adj_relu(x):
    return np.where(x >= 1, x - 1, np.log(x))


def rev_adj_relu(x):
    return -np.where(x >= 0, x + 1, np.exp(x))


def inv_rev_adj_relu(x):
    return np.where(x < -1, -x - 1, np.log(-x))


def add_to_funcs(low, upp, i, funcs, inv_f):
    if (low is None) and (upp is None):
        funcs.append(lambda x: x)
        inv_f.append(lambda x: x)
    elif (low == 0) and (upp == 1):
        D = 10
        funcs.append(lambda x: D * np.arctanh((2 * x) - 1))
        inv_f.append(lambda x: (np.tanh(x / D) + 1) / 2)
    elif upp is None:
        funcs.append(lambda x: (inv_adj_relu(x - np.copy(low))))
        inv_f.append(lambda x: (adj_relu(x) + np.copy(low)))
    elif low is None:
        funcs.append(lambda x: inv_rev_adj_relu(x - np.copy(upp)))
        inv_f.append(lambda x: np.copy(upp) + rev_adj_relu(x))
    else:
        funcs.append(lambda x: x)
        inv_f.append(lambda x: x)


def bounds_convert(x, bounds, fixed, param_map):
    """
    This function is used to transform the parameters from the bounded
    parameter space to the unbounded parameter space. This is an improvement
    over using the scipy.optimize.minimize function's bounds parameter as
    it allows us to avoid the use of the constrained optimization methods.
    """
    bounded_to_unbounded_transforms = []
    unbounded_to_bounded_transforms = []

    for i, (lower, upper) in enumerate(bounds):
        add_to_funcs(
            lower,
            upper,
            i,
            bounded_to_unbounded_transforms,
            unbounded_to_bounded_transforms,
        )

    def transform_params_to_unbounded(params):
        return np.array(
            [f(p) for p, f in zip(params, bounded_to_unbounded_transforms)]
        )

    def transform_unbounded_value_to_params(params):
        return np.array(
            [f(p) for p, f in zip(params, unbounded_to_bounded_transforms)]
        )

    n_params = len(param_map)

    if fixed is not None:
        fixed_idx = [param_map[x] for x in fixed.keys()]
        not_fixed = [x for x in range(n_params) if x not in fixed_idx]
        not_fixed = np.array(not_fixed)

        def constraints(p):
            params = [0] * (n_params)
            for k, v in fixed.items():
                params[param_map[k]] = bounded_to_unbounded_transforms[
                    param_map[k]
                ](v)
            for i, v in zip(not_fixed, p):
                params[i] = v
            return np.array(params)

        const = constraints
    else:

        def const(x):
            return x

        fixed_idx = []
        not_fixed = np.array([x for x in range(n_params)])

    return (
        transform_params_to_unbounded,
        transform_unbounded_value_to_params,
        const,
        fixed_idx,
        not_fixed,
    )
