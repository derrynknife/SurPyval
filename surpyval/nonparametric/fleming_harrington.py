import numpy as np
from surpyval.utils import xcnt_to_xrd
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter


def fh_h(r_i, d_i):
    out = 0
    while d_i > 1:
        out += 1. / r_i
        r_i -= 1
        d_i -= 1
    out += d_i / r_i
    return out


def fh(r, d):
    Y = np.array([fh_h(r_i, d_i) for r_i, d_i in zip(r, d)])
    H = Y.cumsum()
    H[np.isnan(H)] = np.inf
    R = np.exp(-H)
    return R


def fleming_harrington(x, c, n, t):
    xrd = xcnt_to_xrd(x, c, n, t)
    out = {k: v for k, v in zip(['x', 'r', 'd'], xrd)}
    out['R'] = fh(out['r'], out['d'])
    return out


class FlemingHarrington_(NonParametricFitter):
    r"""
    Fleming-Harrington estimation of survival distribution.
    Returns a `NonParametric` object from method :code:`fit()`
    calculates the Non-Parametric estimate of the survival function using:

    .. math::

        R = e^{-\sum_{i:x_{i} \leq x} \sum_{i=0}^{d_x-1} \frac{1}{r_x - i}}

    See 'NonParametric section for detailed estimate of how H is computed.'

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import FlemingHarrington
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = FlemingHarrington.fit(x)
    >>> model.R
    array([0.81873075, 0.63762815, 0.45688054, 0.27711205, 0.10194383])
    """
    def __init__(self):
        self.how = 'Fleming-Harrington'


FlemingHarrington = FlemingHarrington_()
