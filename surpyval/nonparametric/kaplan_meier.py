import numpy as np

from surpyval.nonparametric.nonparametric_fitter import NonParametricFitter


def kaplan_meier(r, d):
    R = 1 - (d / r)
    R[np.isnan(R)] = 0
    old_err_state = np.seterr(under="raise")

    try:
        R = np.cumprod(R)
    except FloatingPointError:
        R = np.cumsum(np.log(1 - (d / r)))
        R = np.exp(R)

    np.seterr(**old_err_state)
    return R


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
        self.how = "Kaplan-Meier"


KaplanMeier = KaplanMeier_()
