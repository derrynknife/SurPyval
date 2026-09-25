import numpy as np

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .nhpp_fitter import NHPPFitter


@singleton_fitter
class CrowAMSAA(NHPPFitter):
    """
    The Crow-AMSAA (power-law) non-homogeneous Poisson process, with
    cumulative intensity and intensity

    .. math::
        \\Lambda(t) = \\left(\\frac{t}{\\alpha}\\right)^{\\beta}, \\qquad
        \\lambda(t) = \\frac{\\beta}{\\alpha^{\\beta}} t^{\\beta - 1}.

    ``beta`` above 1 is a rising event rate (deterioration), below 1 a
    falling one (reliability growth) and 1 the HPP with rate
    ``1 / alpha``; ``alpha`` is the time by which one event is expected.
    ``CrowAMSAA`` is an instance of this class; ``fit`` and ``from_params``
    return a ``ParametricRecurrenceModel``.

    Examples
    --------

    >>> from surpyval import Exponential
    >>> from surpyval.recurrent import CrowAMSAA
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1e-3).cumsum()
    >>> model = CrowAMSAA.fit(x)
    >>> print(model)
    Parametric Recurrence SurPyval Model
    ==================================
    Process             : Crow-AMSAA
    Fitted by           : MLE
    Parameters          :
         alpha: 913.8466210685444
          beta: 1.4781707110680866
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([4.20072057e-05, 1.17030084e-04, 2.13103439e-04, 3.26040266e-04,
           4.53440995e-04, 5.93696079e-04])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([6.20938211e-05, 8.64952211e-05, 1.05001087e-04, 1.20485793e-04,
           1.34052640e-04, 1.46264026e-04])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([ 913.84662107, 1460.57434899, 1921.54912724, 2334.39329941,
           2714.78099355, 3071.15581638])
    """

    def __init__(self) -> None:
        self.name = "Crow-AMSAA"
        self.param_names = ["alpha", "beta"]
        # beta > 0: the intensity beta / alpha**beta * x**(beta - 1) and
        # log(beta) are undefined below zero (a negative beta was allowed and
        # the optimiser could stop there, with a decreasing MCF).
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return (x / alpha) ** beta

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return (beta / alpha**beta) * (x ** (beta - 1))

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return np.log(beta) - beta * np.log(alpha) + (beta - 1) * np.log(x)

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        alpha = params[0]
        beta = params[1]
        return alpha * (N ** (1.0 / beta))
