import numpy as np
import numpy.typing as npt

from surpyval.recurrent.parametric.counting_process import Boxable
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
         alpha: 0.384812737762836
          beta: 0.19396672109211047
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([ 1.62151879,  3.59013322,  5.98014113,  8.88174429, 12.40445268,
           16.6812175 ])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([1.78385983, 2.16570551, 2.62928751, 3.19210196, 3.87539016,
           4.70494021])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([0.63925589, 1.20792021, 1.72004461, 2.18586234, 2.61305756,
           3.00754742])
    """

    def __init__(self) -> None:
        self.name = "Cox-Lewis"
        self.param_names = ["alpha", "beta"]
        # alpha is the *log*-intensity intercept and is legitimately
        # negative whenever the baseline rate is below one event per
        # time unit; the old (0, None) bound silently pinned such fits
        # at alpha = 0 (#286).
        self.bounds = ((None, None), (None, None))
        self.support = (0.0, np.inf)

    def parameter_initialiser(self, x: npt.ArrayLike) -> npt.NDArray:
        return np.array([1.0, 1.0])

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        # The Cox-Lewis intensity is log-linear, so its cumulative intensity
        # is the integral of ``exp(alpha + beta * x)`` from 0 to ``x``.
        alpha = params[0]
        beta = params[1]
        return np.exp(alpha) / beta * (np.exp(beta * x) - 1.0)

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return np.exp(alpha + beta * x)

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return alpha + beta * x

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
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
