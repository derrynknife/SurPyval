import numpy as np

from .nhpp_fitter import NHPPFitter


class Crow_(NHPPFitter):
    """
    A class to represent the Crow model for non-homogeneous Poisson
    processes (NHPP). This model is used in reliability analysis to predict
    failure rates based on historical data.

    Examples
    --------

    >>> from surpyval import Crow, Exponential
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1e-3).cumsum()
    >>> model = Crow.fit(x)
    >>> print(model)
    Parametric Recurrence SurPyval Model
    ==================================
    Process             : Crow
    Fitted by           : MLE
    Parameters          :
        alpha: 21466.48892388212
        beta: 1.2392590066132205
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([4.65842366e-05, 1.09974784e-04, 1.81767316e-04, 2.59625443e-04,
           3.42329130e-04, 4.29111278e-04])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([5.77299348e-05, 6.81436206e-05, 7.50855947e-05, 8.04357921e-05,
           8.48468915e-05, 8.86300027e-05])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([ 3129.2801836 ,  5474.64210552,  7593.6351178 ,  9577.82762328,
           11467.45338053, 13284.98316248])
    """

    def __init__(self):
        self.name = "Crow"
        self.param_names = ["alpha", "beta"]
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def parameter_initialiser(self, x):
        return np.array([1.0, 1.0])

    def cif(self, x, *params):
        """
        Cumulative intensity function (CIF) of the Crow model.

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
        return (x**beta) / alpha

    def iif(self, x, *params):
        """
        Instantaneous intensity function (IIF) or the failure rate of the
        Crow model.

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
        return (beta / alpha) * (x ** (beta - 1))

    def log_iif(self, x, *params):
        """
        Natural logarithm of the instantaneous intensity function (IIF) of
        the Crow model.

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
        return np.log(beta) - np.log(alpha) + (beta - 1) * np.log(x)

    def inv_cif(self, N, *params):
        """
        Inverse of the cumulative intensity function (CIF) of the
        Crow model.

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
        return (alpha * N) ** (1.0 / beta)


Crow = Crow_()
