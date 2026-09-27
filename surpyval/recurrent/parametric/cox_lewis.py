import numpy as np
import numpy.typing as npt

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .nhpp_fitter import NHPPFitter


@singleton_fitter
class CoxLewis(NHPPFitter):
    """
    The Cox-Lewis (log-linear) non-homogeneous Poisson process, with

    .. math::
        \\lambda(t) = e^{\\alpha + \\beta t}, \\qquad
        \\Lambda(t) = \\frac{e^{\\alpha}}{\\beta}
        \\left(e^{\\beta t} - 1\\right).

    ``alpha`` is the log of the intensity at ``t = 0`` and ``beta`` its
    proportional change per unit time: positive is deteriorating, negative
    improving. With a negative ``beta`` the cumulative intensity levels off
    at ``exp(alpha) / -beta``, and ``inv_cif`` returns ``inf`` beyond it.
    ``CoxLewis`` is an instance of this class; ``fit`` and ``from_params``
    return a ``ParametricRecurrenceModel``.

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
        # The log-linear intensity is defined at any time, so an item
        # observed from a negative ``tl`` may have events at negative times.
        self.support = (-np.inf, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        # The Cox-Lewis intensity is log-linear, so its cumulative intensity
        # is the integral of ``exp(alpha + beta * x)`` from 0 to ``x``,
        # ``exp(alpha) * (exp(beta x) - 1) / beta``. Written with expm1 and
        # its beta -> 0 limit ``exp(alpha) * x`` (an HPP): the direct form
        # is 0/0 at beta = 0 and loses digits for a tiny beta.
        alpha = params[0]
        beta = params[1]
        x = np.asarray(x, dtype=float)
        return np.exp(alpha) * _expm1_over(beta, x)

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
        # non-positive number. log1p and the beta -> 0 limit
        # ``N exp(-alpha)`` keep a tiny or zero beta exact.
        scaled = np.asarray(N, dtype=float) * np.exp(-alpha)
        arg = scaled * beta
        reached = arg > -1.0
        safe_arg = np.where(reached, arg, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                beta == 0,
                scaled,
                np.log1p(safe_arg) / np.where(beta == 0, 1.0, beta),
            )
        return np.where(reached, ratio, np.inf)


def _expm1_over(beta: Boxable, x: npt.NDArray) -> Boxable:
    """``(exp(beta * x) - 1) / beta``, with its limit ``x`` at beta = 0."""
    safe_beta = np.where(beta == 0, 1.0, beta)
    return np.where(beta == 0, x, np.expm1(beta * x) / safe_beta)
