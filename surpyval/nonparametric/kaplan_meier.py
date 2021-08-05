import numpy as np
from surpyval.utils import xcnt_to_xrd
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter


def km(r, d):
    R = 1 - (d / r)
    R[np.isnan(R)] = 0
    R = np.cumprod(R)
    return R


def kaplan_meier(x, c, n, t):
    xrd = xcnt_to_xrd(x, c, n, t)
    out = {k: v for k, v in zip(['x', 'r', 'd'], xrd)}
    out['R'] = km(out['r'], out['d'])
    return out


class KaplanMeier_(NonParametricFitter):
    r"""
    Kaplan-Meier estimator class. Calculates the Non-Parametric
    estimate of the survival function using:

    .. math::
        R(x) = \prod_{i:x_{i} \leq x}^{}
            \left ( 1 - \frac{d_{i} }{r_{i}}  \right )

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import KaplanMeier
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> model = KaplanMeier.fit(x)
    >>> model.R
    array([0.8, 0.6, 0.4, 0.2, 0. ])
    """
    def __init__(self):
        self.how = 'Kaplan-Meier'


KaplanMeier = KaplanMeier_()
