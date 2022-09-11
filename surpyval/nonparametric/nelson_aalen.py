import numpy as np
from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter


def nelson_aalen(r, d):
    H = np.cumsum(d / r)
    H[np.isnan(H)] = np.inf
    R = np.exp(-H)
    return R


class NelsonAalen_(NonParametricFitter):
    r"""
    Nelson-Aalen estimator class. Returns a `NonParametric`
    object from method :code:`fit()` Calculates the Non-Parametric
    estimate of the survival function using:

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
