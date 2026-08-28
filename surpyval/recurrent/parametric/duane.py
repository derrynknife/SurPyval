import numpy as np
import numpy.typing as npt

from surpyval.recurrent.parametric.counting_process import Boxable
from surpyval.utils.fitter import singleton_fitter

from .nhpp_fitter import NHPPFitter


@singleton_fitter
class Duane(NHPPFitter):
    """
    Represents the Duane Non-Homogeneous Poisson Process model.
    This class includes methods to evaluate various statistical functions of
    the model and perform parameter estimation based on input data.

    Examples
    --------

    >>> from surpyval import Exponential
    >>> from surpyval.recurrent import Duane
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
         alpha: 1.478202089169939
             b: 4.199455086392048e-05
    >>> model.cif([1, 2, 3, 4, 5, 6])
    array([4.19945509e-05, 1.16997373e-04, 2.13046585e-04, 3.25956224e-04,
           4.53327287e-04, 5.93550595e-04])
    >>>
    >>> model.iif([1, 2, 3, 4, 5, 6])
    array([6.20764328e-05, 8.64728804e-05, 1.04975302e-04, 1.20457293e-04,
           1.34021869e-04, 1.46231288e-04])
    >>>
    >>> model.inv_cif([1, 2, 3, 4, 5, 6])
    array([ 913.90063856, 1460.64614435, 1921.63239303, 2334.48481047,
           2714.87871658, 3071.25832646])
    """

    def __init__(self) -> None:
        self.name = "Duane"
        self.param_names = ["alpha", "b"]
        self.bounds = ((0, None), (0, None))
        self.support = (0.0, np.inf)

    def parameter_initialiser(self, x: npt.ArrayLike) -> npt.NDArray:
        return np.array([1.0, 1.0])

    def cif(self, x: Boxable, *params: Boxable) -> Boxable:
        return params[1] * x ** params[0]

    def iif(self, x: Boxable, *params: Boxable) -> Boxable:
        return params[0] * params[1] * x ** (params[0] - 1.0)

    def log_iif(self, x: Boxable, *params: Boxable) -> Boxable:
        return (
            np.log(params[0]) + np.log(params[1]) + (params[0] - 1) * np.log(x)
        )

    def inv_cif(self, N: Boxable, *params: Boxable) -> Boxable:
        return (N / params[1]) ** (1.0 / params[0])
