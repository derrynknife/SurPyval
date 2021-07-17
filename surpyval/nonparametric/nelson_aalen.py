import surpyval
import numpy as np
from surpyval import nonparametric as nonp
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter

def nelson_aalen(x, c=None, n=None, **kwargs):
    x, r, d = surpyval.xcnt_to_xrd(x, c, n, **kwargs)

    h = d/r
    H = np.cumsum(h)
    R = np.exp(-H)
    return x, r, d, R

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