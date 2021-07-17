import surpyval
import numpy as np
from surpyval import nonparametric as nonp
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter

def kaplan_meier(x, c=None, n=None, **kwargs):
    # Could consider accepting on xrd for this
    x, r, d = surpyval.xcnt_to_xrd(x, c, n, **kwargs)
    
    R = np.cumprod(1 - d/r)
    return x, r, d, R

class KaplanMeier_(NonParametricFitter):
    r"""
    Kaplan-Meier estimator class. Calculates the Non-Parametric estimate of the survival function using:

    .. math::
        R(x) = \prod_{i:x_{i} \leq x}^{} \left ( 1 - \frac{d_{i} }{r_{i}}  \right )
    
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


