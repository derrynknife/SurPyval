import numpy as np

from .nhpp_fitter import NHPPFitter


class CoxLewis_(NHPPFitter):
    """
    A class to represent the Cox-Lewis model for non-homogeneous Poisson
    processes (NHPP). This model is used in reliability analysis to predict
    failure rates based on historical data.

    Examples
    --------

    >>> from surpyval import CoxLewis, Exponential
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
        alpha: 0.7177727527489457
        beta: 0.08828975097259131
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([2.23907426, 2.44575105, 2.67150506, 2.91809719, 3.1874509 ,
           3.4816672 ])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([0.19768731, 0.21593475, 0.23586652, 0.25763807, 0.28141925,
           0.30739553])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([-8.12974037, -0.27891768,  4.3135192 ,  7.57190502, 10.09930541,
           12.16434189])
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

        Parameters
        ----------

        x : float
            The value at which CIF is evaluated.
        params : tuple
            Parameters of the Duane model.

        Returns
        -------

        float
            The CIF value.
        """
        alpha = params[0]
        beta = params[1]
        return np.exp(alpha + beta * x)

    def iif(self, x, *params):
        """
        Instantaneous intensity function (IIF) or the failure rate of the
        Cox-Lewis model.

        Parameters
        ----------

        x : float
            The value at which IIF is evaluated.
        params : tuple
            Parameters of the Duane model.

        Returns
        -------

        float
            The IIF value.
        """
        alpha = params[0]
        beta = params[1]
        return beta * np.exp(alpha + beta * x)

    def log_iif(self, x, *params):
        """
        Natural logarithm of the instantaneous intensity function (IIF) of
        the Cox-Lewis model.

        Parameters
        ----------

        x : float
            The value at which log(IIF) is evaluated.
        params : tuple
            Parameters of the Duane model.

        Returns
        -------

        float
            The log(IIF) value.
        """
        alpha = params[0]
        beta = params[1]
        return np.log(beta) + alpha + beta * x

    def inv_cif(self, N, *params):
        """
        Inverse of the cumulative intensity function (CIF) of the
        Cox-Lewis model.

        Parameters
        ----------

        N : float
            The number of events expected to have occured.
        params : tuple
            Parameters of the Duane model.

        Returns
        -------

        float
            The value of x at which N events are expected to have occured.
        """
        alpha = params[0]
        beta = params[1]
        return (np.log(N) - alpha) / beta


CoxLewis = CoxLewis_()
