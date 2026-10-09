from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class DualPower_(LifeModel):
    r"""
    The dual power life model, for two non-thermal stresses (for example
    voltage and frequency), each a power law:

    .. math::
        L(U, V) = c\, U^{m} V^{n},

    with ``U`` and ``V`` the two columns of ``Z``, both positive.

    Parameters (as the fitted model reports them):

    - ``c`` (> 0): the life at unit stresses (``U = V = 1``).
    - ``m``: the power of ``U``: doubling ``U`` multiplies the life by
      :math:`2^{m}`.
    - ``n``: the power of ``V``, likewise.

    With equal (or proportional) stress columns ``m`` and ``n`` cannot be
    told apart: ``n`` is reported as aliased (NaN), and the fit is the
    :class:`Power` fit.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import life_models
    >>> life_models.DualPower.phi(np.array([[2.0, 3.0]]), 100.0, -1.0, -0.5)
    array([28.86751346])
    """

    n_stresses = 2
    positive_stress_columns = (0, 1)
    phi_takes_rows = True
    log_scale_parameters = ("c",)

    def __init__(self) -> None:
        super().__init__(
            "DualPower",
            {"c": 0, "m": 1, "n": 2},
            ((0, None), (None, None), (None, None)),
        )

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        # One exponent, which a factor alone can overflow (#634)
        return self._phi_from_log_life(Z, params)

    def log_life(self, Z: ndarray, *params: float) -> ndarray:
        Z = np.atleast_2d(Z)
        log_c, m, n = params[0], params[1], params[2]
        return log_c + m * np.log(Z[:, 0]) + n * np.log(Z[:, 1])

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        A = np.atleast_2d(Z)
        A = np.hstack([np.ones(Z.shape[0]).reshape(-1, 1), np.log(Z)])
        y = np.log(life)
        c, m, n = np.linalg.lstsq(A, y, rcond=None)[0]
        return [np.exp(c), m, n]

    def _stress_terms(
        self, Z: ndarray
    ) -> "tuple[ndarray, tuple[str, ...], bool] | None":
        # log L = log c + m log s1 + n log s2
        return np.log(np.atleast_2d(Z)), ("m", "n"), True


DualPower = DualPower_()
