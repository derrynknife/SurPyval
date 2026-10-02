from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class DualExponential_(LifeModel):
    r"""
    The dual exponential life model, for two thermal-like stresses (for
    example temperature and humidity, the temperature-humidity model),
    each entering as Arrhenius does:

    .. math::
        L(U, V) = c\, e^{a / U + b / V},

    with ``U`` and ``V`` the two columns of ``Z`` (temperature in kelvin).

    Parameters (as the fitted model reports them):

    - ``a``: the sensitivity to ``U``: for a temperature, the activation
      energy over Boltzmann's constant, in kelvin, as for
      :class:`Exponential`.
    - ``b``: the sensitivity to ``V``, likewise.
    - ``c`` (> 0): a scale constant, the life as both stresses grow
      without bound.

    With equal (or proportional) stress columns ``a`` and ``b`` cannot be
    told apart: ``b`` is reported as aliased (NaN), and the fit is the
    :class:`Exponential` fit.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> Z = np.array([[358.0, 0.85]])
    >>> life_models.DualExponential.phi(Z, 8000.0, 0.5, 1e-5).round(1)
    array([91279.2])
    """

    n_stresses = 2

    def __init__(self) -> None:
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

    def _stress_terms(
        self, Z: ndarray
    ) -> "tuple[ndarray, tuple[str, ...], bool] | None":
        # log L = log c + a / s1 + b / s2
        return 1.0 / np.atleast_2d(Z), ("a", "b"), True


DualExponential = DualExponential_()
