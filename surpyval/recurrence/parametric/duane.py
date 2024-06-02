import numpy as np

from .nhpp_fitter import NHPPFitter


class Duane_(NHPPFitter):
    """
    Represents the Duane Non-Homogeneous Poisson Process model.
    This class includes methods to evaluate various statistical functions of
    the model and perform parameter estimation based on input data.

    Examples
    --------

    >>> from surpyval import Duane, Exponential
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> x = Exponential.random(10, 1e-3).cumsum()
    >>> model = Duane.fit(x)
    >>> print(model)
    Parametric Recurrence SurPyval Model
    ==================================
    Process             : Duane
    Fitted by           : MLE
    Parameters          :
        alpha: 1.2392945732132952
            b: 4.6568641229556424e-05
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([4.65686412e-05, 1.09940677e-04, 1.81713565e-04, 2.59551323e-04,
           3.42234115e-04, 4.28994958e-04])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([5.77122644e-05, 6.81244421e-05, 7.50655449e-05, 8.04151364e-05,
           8.48257764e-05, 8.86085206e-05])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([ 3129.4028404 ,  5474.76881015,  7593.73955973,  9577.89554535,
           11467.47544342, 13284.95262974])
    """

    def __init__(self):
        self.name = "Duane"
        self.param_names = ["alpha", "b"]
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def parameter_initialiser(self, x):
        return np.array([1.0, 1.0])

    def cif(self, x, *params):
        """
        Cumulative intensity function (CIF) of the Duane model.

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
        return params[1] * x ** params[0]

    def iif(self, x, *params):
        """
        Instantaneous intensity function (IIF) or the failure rate of the
        Duane model.

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
        return params[0] * params[1] * x ** (params[0] - 1.0)

    def log_iif(self, x, *params):
        """
        Natural logarithm of the instantaneous intensity function (IIF) of the
        Duane model.

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
        return (
            np.log(params[0]) + np.log(params[1]) + (params[0] - 1) * np.log(x)
        )

    def inv_cif(self, N, *params):
        """
        Inverse of the cumulative intensity function (CIF) of the Duane model.

        Parameters
        ----------

        N : float
            The value at which inverse CIF is evaluated.
        params : tuple
            Parameters of the Duane model.

        Returns
        -------

        float
            The inverse CIF value.
        """
        return (N / params[1]) ** (1.0 / params[0])


Duane = Duane_()
