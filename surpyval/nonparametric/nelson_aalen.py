import numpy as np
from surpyval.utils import xcnt_to_xrd
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter

def na(r, d):
    H = np.cumsum(d/r)
    H[np.isnan(H)] = np.inf
    R = np.exp(-H)
    return R

def nelson_aalen(x, c, n, t):
    xrd = xcnt_to_xrd(x, c, n, t)
    out = {k : v for k, v in zip(['x', 'r', 'd'], xrd)}
    out['R'] = na(out['r'], out['d'])
    return out

class NelsonAalen_(NonParametricFitter):
    r"""
    Nelson-Aalen estimator class. Returns a `NonParametric` object from method :code:`fit()` Calculates the Non-Parametric estimate of the survival function using:

    .. math::
        R(x) = e^{-\sum_{i:x_{i} \leq x}^{} \frac{d_{i} }{r_{i}}}
    
    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import NelsonAalen
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = NelsonAalen.fit(x)
    >>> model.R
    array([0.81873075, 0.63762815, 0.45688054, 0.27711205, 0.10194383])
    """
    def __init__(self):
        self.how = 'Nelson-Aalen'

NelsonAalen = NelsonAalen_()