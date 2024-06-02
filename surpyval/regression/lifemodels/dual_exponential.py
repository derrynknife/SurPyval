from autograd import numpy as np
from numpy import ndarray

from surpyval.regression.lifemodels.lifemodel import LifeModel


class DualExponential_(LifeModel):
    """
    Dual Exponential Life Model

    This class represents a dual exponential life model, which is used for
    survival analysis.

    Attributes
    ----------
        name (str): The name of the life model.
        param_names (dict): A dictionary specifying the parameter names and
        their indices in the parameter vector.
        param_bounds (tuple): A tuple specifying the parameter bounds.

    Methods:
    --------
        phi(Z: ndarray, *params: float) -> ndarray:
            Calculate the life parameter for a distribution using the
            covariates / stresses, Z, and the parameters of the dual
            exponential model.

        phi_init(life: float, Z: ndarray) -> list[float]:
            Initialize the parameters of the dual exponential model based
            on observed data.
    """

    def __init__(self):
        """
        Initialize the DualExponential_ class.

        The class is initialized with default parameter names and bounds for
        the dual exponential distribution.

        """
        super().__init__(
            "DualExponential",
            {"a": 0, "b": 1, "c": 2},
            ((None, None), (None, None), (0, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        """
        Calculate the life parameter for a distribution using the covariates /
        stresses, Z, and the parameters of the dual exponential model.

        Args:
            Z (ndarray): An array of shape (n_samples, 2) containing the
                predictor variables.
            *params (float): Parameters 'a', 'b', and 'c' of the dual
                exponential distribution.

        Returns:
            ndarray: An array of shape (n_samples,) containing the PDF values.

        """
        Z = np.atleast_2d(Z)
        Z1 = Z[:, 0]
        Z2 = Z[:, 1]
        a = params[0]
        b = params[1]
        c = params[2]
        return c * np.exp(a / Z1) * np.exp(b / Z2)

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        """
        Initialize the parameters of the dual exponential model for the initial
        guess of the optimization.

        Parameters:
        -----------

            life (float): The observed lifetime.
            Z (ndarray): An array of shape (n_samples, 2) containing the
            covariates / stresses.

        Returns:
        --------

            list[float]: A list of parameters [a, b, c] for the dual
            exponential model.

        """
        A = np.atleast_2d(Z)
        A = 1.0 / np.hstack([np.ones(Z.shape[0]).reshape(-1, 1), Z])
        y = np.log(life)
        c, a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return [a, b, np.exp(c)]


DualExponential = DualExponential_()
