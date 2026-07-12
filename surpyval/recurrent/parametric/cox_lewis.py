import numpy as np

from surpyval.utils.fitter import singleton_fitter

from .nhpp_fitter import NHPPFitter


@singleton_fitter
class CoxLewis(NHPPFitter):
    """
    A class to represent the Cox-Lewis model for non-homogeneous Poisson
    processes (NHPP). This model is used in reliability analysis to predict
    failure rates based on historical data.

    Examples
    --------

    >>> from surpyval import Exponential
    >>> from surpyval.recurrent import CoxLewis
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1).cumsum()
    >>> model = CoxLewis.fit(x)
    >>> print(model)
    Parametric Recurrence SurPyval Model
    ==================================
    Process             : Cox-Lewis
    Fitted by           : MLE
    Parameters          :
        alpha: 0.3848528712360503
        beta: 0.1939447728437042
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([ 1.6215655 ,  3.59019342,  5.98016527,  8.88166096, 12.40416155,
           16.68058024])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([1.78389227, 2.16569736, 2.62921991, 3.19194983, 3.87512041,
           4.70450946])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([0.63923607, 1.20789182, 1.72001505, 2.18583659, 2.61303902,
           3.00753845])
    """

    def __init__(self):
        self.name = "Cox-Lewis"
        self.param_names = ["alpha", "beta"]
        self.bounds = ((0, None), (None, None))
        self.support = (0.0, np.inf)

    def parameter_initialiser(self, x):
        return np.array([1.0, 1.0])

    def cif(self, x, *params):
        # The Cox-Lewis intensity is log-linear, so its cumulative intensity
        # is the integral of ``exp(alpha + beta * x)`` from 0 to ``x``.
        alpha = params[0]
        beta = params[1]
        return np.exp(alpha) / beta * (np.exp(beta * x) - 1.0)

    def iif(self, x, *params):
        alpha = params[0]
        beta = params[1]
        return np.exp(alpha + beta * x)

    def log_iif(self, x, *params):
        alpha = params[0]
        beta = params[1]
        return alpha + beta * x

    def inv_cif(self, N, *params):
        alpha = params[0]
        beta = params[1]
        # For an improving system (beta < 0) the cumulative intensity is
        # bounded above by exp(alpha) / -beta, so counts at or beyond that
        # asymptote are never reached: return inf rather than log of a
        # non-positive number.
        arg = 1.0 + np.asarray(N, dtype=float) * beta * np.exp(-alpha)
        reached = arg > 0.0
        safe_arg = np.where(reached, arg, 1.0)
        return np.where(reached, np.log(safe_arg) / beta, np.inf)
