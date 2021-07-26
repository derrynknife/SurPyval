import autograd.numpy as np
from autograd import jacobian, hessian

from scipy.optimize import minimize

def mom(model):
    """
    MOM: Method of Moments.

    This is one of the simplest ways to calculate the parameters of a distribution.

    This method is quick but only works with uncensored data.
    """
    dist = model.dist
    x, c, n, t = model.data['x'], model.data['c'], model.data['n'], model.data['t']

    const = model.fitting_info['const']
    inv_trans = model.fitting_info['inv_trans']
    init = model.fitting_info['init']
    offset = model.offset
    bounds = model.bounds

    x_ = np.repeat(x, n)

    if hasattr(dist, '_mom'):
        return {'params' : dist._mom(x_)}

    moments = np.zeros(model.k)

    for i in range(0, model.k):
        moments[i] = (x_**(i+1)).mean()

    fun = lambda params : np.log(((np.abs(moments - dist.mom_moment_gen(*inv_trans(const(params)), offset=offset))/moments))).sum()
    
    res = minimize(fun, np.array(init), tol=1e-15)

    params = inv_trans(const(res.x))

    results = {}
    if offset:
        results['gamma'] = params[0]
        results['params'] = params[1:]
    else:
        results['params'] = params

    results['res'] = res
    results['offset'] = offset

    return results
