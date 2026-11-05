from collections.abc import Sequence

from autograd import numpy as np
from numpy import ndarray

from surpyval.univariate.regression.accelerated_life.lifemodel import LifeModel


class GeneralLinear_(LifeModel):
    r"""
    The general linear life model, for any number of stresses: the life
    parameter moves by a fixed amount per unit of each stress,

    .. math::

        L(Z) = a + \sum_{j} \beta_j Z_j,

    with one coefficient ``beta_j`` per column of ``Z`` and a constant
    ``a`` (the life parameter where every stress is 0). It is ``Linear``
    for several stresses, as ``GeneralLogLinear`` is ``Exponential``'s
    log-linear form for several.

    It is meant for a location parameter, the ``mu`` of ``Normal``,
    ``Gumbel`` or ``Logistic``, which can take any sign: the stresses then
    shift the distribution rather than stretch time, so the fit does not
    depend on where the zero of the measurement is (adding a constant to
    every time moves ``a`` alone). That is the model for a quantity on
    the whole real line with an arbitrary zero (a level in decibels, a
    temperature in degrees Celsius), and is what R's ``survreg`` calls a
    ``"gaussian"`` or ``"logistic"`` model. On a scale parameter (the
    ``alpha`` of ``Weibull``, a life) a fitted plane can reach zero or
    negative life outside the stresses tested, where a life is undefined.

    Coefficients take the sign of the life, as in ``Linear`` and
    ``GeneralLogLinear``: a positive ``beta_j`` means a later (longer)
    life as stress ``j`` rises.

    The number of coefficients is the number of columns of ``Z``, which
    the fit reads from the data (:meth:`resolve`): ``GeneralLinear`` is
    the unresolved model, and a fitted model carries the one for its
    columns.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval import AcceleratedLife, Normal
    >>> from surpyval.life_models import GeneralLinear
    >>> rng = np.random.default_rng(1)
    >>> Z = np.column_stack(
    ...     [np.repeat([1.0, 2.0, 3.0], 40), np.tile([0.0, 1.0], 60)]
    ... )
    >>> x = 50.0 - 5.0 * Z[:, 0] + 3.0 * Z[:, 1] + rng.normal(0, 2, 120)
    >>> model = AcceleratedLife(Normal, GeneralLinear).fit(x, Z)
    >>> model.reg_model.phi_param_map
    {'a': 0, 'coef_0': 1, 'coef_1': 2}
    >>> model.phi_params.round(2)
    array([49.96, -5.01,  2.93])
    """

    phi_takes_rows = True
    #: How the fit's warning explains a coefficient the data cannot
    #: determine (an aliased column)
    aliasing_reason = (
        "the {} life model's life is linear in the stresses, and these "
        "columns are constant (the constant a absorbs them) or a linear "
        "combination of the other columns"
    )

    def __init__(
        self,
        n_stresses: "int | None" = None,
        names: "Sequence[str] | None" = None,
    ) -> None:
        # ``None``: not yet resolved to a number of columns, so the
        # parameters are only the constant until the fit sees Z. The
        # coefficients are ``coef_j``, or ``names`` (a fit's columns,
        # :meth:`named`), as ``GeneralLogLinear``'s.
        k = 0 if n_stresses is None else int(n_stresses)
        if names is None or len(names) != k:
            names = ["coef_" + str(j) for j in range(k)]
        names = ["a", *names]
        bounds: tuple[tuple[int | None, int | None], ...] = ((None, None),) * (
            k + 1
        )
        super().__init__(
            "GeneralLinear", {nm: i for i, nm in enumerate(names)}, bounds
        )
        self.n_stresses = None if n_stresses is None else k

    def resolve(self, n_stresses: int) -> "GeneralLinear_":
        if self.n_stresses is not None:
            return self
        return GeneralLinear_(n_stresses)

    def named(self, names: "Sequence[str]") -> "GeneralLinear_":
        if self.n_stresses is None or list(names) == list(
            self.coefficient_columns()
        ):
            return self
        return GeneralLinear_(self.n_stresses, names)

    def coefficient_columns(self) -> "dict[str, int]":
        # Every parameter but ``a`` is a column's coefficient
        names = sorted(self.phi_param_map, key=self.phi_param_map.__getitem__)
        return {name: j for j, name in enumerate(names[1:])}

    def phi(self, Z: ndarray, *params: float) -> ndarray:
        # One row per stress vector, so a single row ``[Z_0, Z_1]`` gives
        # one life of shape (1,), as ``GeneralLogLinear``'s
        Z = np.atleast_2d(Z)
        return params[0] + np.dot(Z, np.array(params[1:]))

    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        # Least squares of L = a + beta'Z through the lives at each
        # stress level (the minimum-norm solution where a column is
        # constant or collinear; the fit holds those aliased at 0).
        Z = np.atleast_2d(Z)
        A = np.hstack([np.ones((Z.shape[0], 1)), Z])
        lives = np.asarray(life, dtype=float).ravel()
        coef = np.linalg.lstsq(A, lives, rcond=None)[0]
        return [float(c) for c in coef]

    def _stress_terms(
        self, Z: ndarray
    ) -> "tuple[ndarray, tuple[str, ...], bool] | None":
        # L = a + beta'Z: each column is its own term, and ``a`` absorbs a
        # constant one. One column at one level is refused as for the
        # other one-stress models.
        Z = np.atleast_2d(Z)
        if Z.shape[1] == 1:
            return None
        names = tuple(self.coefficient_columns())
        return Z, names, True


GeneralLinear = GeneralLinear_()
