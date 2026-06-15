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
        """
        Cumulative intensity function (CIF) of the Cox-Lewis model.

        This is the integral of the instantaneous intensity from 0 to ``x``,
        i.e. the expected number of events by ``x``. It satisfies
        ``cif(0) == 0``.

        Parameters
        ----------

        x : float
            The value at which CIF is evaluated.
        params : tuple
            Parameters of the Cox-Lewis model.

        Returns
        -------

        float
            The CIF value.
        """
        alpha = params[0]
        beta = params[1]
        return np.exp(alpha) / beta * (np.exp(beta * x) - 1.0)

    def iif(self, x, *params):
        """
        Instantaneous intensity function (IIF) or the failure rate of the
        Cox-Lewis model. This is the log-linear intensity that defines the
        Cox-Lewis model.

        Parameters
        ----------

        x : float
            The value at which IIF is evaluated.
        params : tuple
            Parameters of the Cox-Lewis model.

        Returns
        -------

        float
            The IIF value.
        """
        alpha = params[0]
        beta = params[1]
        return np.exp(alpha + beta * x)

    def log_iif(self, x, *params):
        """
        Natural logarithm of the instantaneous intensity function (IIF) of
        the Cox-Lewis model.

        Parameters
        ----------

        x : float
            The value at which log(IIF) is evaluated.
        params : tuple
            Parameters of the Cox-Lewis model.

        Returns
        -------

        float
            The log(IIF) value.
        """
        alpha = params[0]
        beta = params[1]
        return alpha + beta * x

    def inv_cif(self, N, *params):
        """
        Inverse of the cumulative intensity function (CIF) of the
        Cox-Lewis model.

        Parameters
        ----------

        N : float
            The number of events expected to have occured.
        params : tuple
            Parameters of the Cox-Lewis model.

        Returns
        -------

        float
            The value of x at which N events are expected to have occured.
        """
        alpha = params[0]
        beta = params[1]
        return np.log(1.0 + N * beta * np.exp(-alpha)) / beta
