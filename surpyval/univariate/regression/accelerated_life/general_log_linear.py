from collections.abc import Sequence

from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class GeneralLogLinear_(LifeModel):
    """
    The general log-linear life model, for any number of stresses:

    .. math::

        L(Z) = c \\exp\\left(\\sum_{j} \\beta_j Z_j\\right),

    with one coefficient ``beta_j`` per column of ``Z`` and a constant
    factor ``c > 0`` (ReliaSoft's :math:`e^{\\alpha_0 + \\sum_j \\alpha_j
    X_j}`, with :math:`c = e^{\\alpha_0}`). A stress enters as itself; for
    an Arrhenius or inverse-power effect pass its transform as the column
    (``1 / T``, ``log V``). With ``Weibull`` it is the Weibull accelerated
    failure time model, and with ``LogNormal`` the log-normal one.

    The number of coefficients is the number of columns of ``Z``, which
    the fit reads from the data (:meth:`resolve`): ``GeneralLogLinear`` is
    the unresolved model, and a fitted model carries the one for its
    columns.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import AcceleratedLife, Weibull
    >>> from surpyval.life_models import GeneralLogLinear
    >>> rng = np.random.default_rng(1)
    >>> Z = np.column_stack(
    ...     [np.repeat([1.0, 2.0, 3.0], 40), np.tile([0.0, 1.0], 60)]
    ... )
    >>> life = 50.0 * np.exp(-0.5 * Z[:, 0] + 0.3 * Z[:, 1])
    >>> x = life * rng.weibull(2.0, 120)
    >>> model = AcceleratedLife(Weibull, GeneralLogLinear).fit(x, Z)
    >>> model.reg_model.phi_param_map
    {'c': 0, 'coef_0': 1, 'coef_1': 2}
    >>> model.phi_params.round(3)
    array([58.897, -0.589,  0.357])
    """

    phi_takes_rows = True
    log_scale_parameters = ("c",)

    def __init__(
        self,
        n_stresses: "int | None" = None,
        names: "Sequence[str] | None" = None,
    ) -> None:
        # ``None``: not yet resolved to a number of columns, so the
        # parameters are only the constant factor until the fit sees Z.
        # The coefficients are ``coef_j``, or ``names`` (a fit's columns,
        # :meth:`named`; #614).
        k = 0 if n_stresses is None else int(n_stresses)
        if names is None or len(names) != k:
            names = ["coef_" + str(j) for j in range(k)]
        names = ["c", *names]
        bounds: tuple[tuple[int | None, int | None], ...] = ((0, None),) + (
            (None, None),
        ) * k
        super().__init__(
            "GeneralLogLinear", {nm: i for i, nm in enumerate(names)}, bounds
        )
        self.n_stresses = None if n_stresses is None else k

    def resolve(self, n_stresses: int) -> "GeneralLogLinear_":
        if self.n_stresses is not None:
            return self
        return GeneralLogLinear_(n_stresses)

    def named(self, names: "Sequence[str]") -> "GeneralLogLinear_":
        if self.n_stresses is None or list(names) == list(
            self.coefficient_columns()
        ):
            return self
        return GeneralLogLinear_(self.n_stresses, names)

    def coefficient_columns(self) -> "dict[str, int]":
        # Every parameter but ``c`` is a column's coefficient
        names = sorted(self.phi_param_map, key=self.phi_param_map.__getitem__)
        return {name: j for j, name in enumerate(names[1:])}

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        # One row per stress vector, so a single row ``[Z_0, Z_1]`` gives
        # one life of shape (1,), as the other multi-stress models do. (A
        # 0-d life, from ``dot`` of two 1-D arrays, broke autograd's
        # gradient of the fit's ``where``: #530.)
        # One exponent, which a factor alone can overflow (#634)
        return self._phi_from_log_life(Z, params)

    def log_life(self, Z: ndarray, *params: float) -> ndarray:
        Z = np.atleast_2d(Z)
        return params[0] + np.dot(Z, np.array(params[1:]))

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        # Least squares of log L = log c + beta'Z through the lives at
        # each stress level (the minimum-norm solution where a column is
        # constant or collinear; the fit holds those aliased at 0).
        Z = np.atleast_2d(Z)
        A = np.hstack([np.ones((Z.shape[0], 1)), Z])
        coef = np.linalg.lstsq(A, np.log(life), rcond=None)[0]
        return [float(np.exp(coef[0])), *coef[1:].tolist()]

    def _stress_terms(
        self, Z: ndarray
    ) -> "tuple[ndarray, tuple[str, ...], bool] | None":
        # log L = log c + beta'Z: each column is its own term. One column
        # at one level is refused as for the other one-stress models.
        Z = np.atleast_2d(Z)
        if Z.shape[1] == 1:
            return None
        names = tuple(self.coefficient_columns())
        return Z, names, True


GeneralLogLinear = GeneralLogLinear_()
