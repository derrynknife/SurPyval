from abc import ABC, abstractmethod

from numpy import ndarray


class LifeModel(ABC):
    """
    Base class for the stress-life relationships used by
    ``AcceleratedLife``: a function :math:`L(Z)` giving the life parameter
    of a distribution at stress :math:`Z`.

    A subclass passes its ``name``, a ``phi_param_map`` from parameter
    name to position, and ``phi_bounds`` (one ``(lower, upper)`` pair per
    parameter, ``None`` for unbounded) to this constructor, and implements
    :meth:`phi` and :meth:`phi_init`. The built-in life models are
    instances of subclasses: ``Power``, ``InversePower``, ``Eyring``,
    ``InverseEyring``, ``ExponentialLifeModel``, ``InverseExponential``,
    ``Linear``, ``DualExponential``, ``DualPower``, ``PowerExponential``
    and ``GeneralLogLinear``.

    A life model whose parameters depend on the number of stress columns
    (``GeneralLogLinear``, one coefficient per column) sets
    ``n_stresses = None`` and overrides :meth:`resolve`, which the fit
    calls with the number of columns of ``Z`` to get the model with a
    fixed ``phi_param_map`` and ``phi_bounds``.

    Examples
    --------
    ``Power`` is one, with :math:`L(Z) = a Z^n`:

    >>> import numpy as np
    >>> from surpyval.life_models import LifeModel, Power
    >>> isinstance(Power, LifeModel)
    True
    >>> Power.phi_param_map
    {'a': 0, 'n': 1}
    >>> Power.phi(np.array([1.0, 2.0, 4.0]), 1000.0, -2.0)
    array([1000.  ,  250.  ,   62.5])
    """

    #: The number of stress columns ``Z`` has (``None`` when it depends on
    #: the data, as for ``GeneralLogLinear``). Lets a single 1-D row
    #: ``[T, V]`` be read as one two-stress row rather than two stresses.
    n_stresses: "int | None" = 1
    #: Stress columns that must be strictly positive: the life model takes
    #: a power or logarithm of them (``Z**n``, ``log Z``), or reads them as
    #: an absolute temperature.
    positive_stress_columns: "tuple[int, ...]" = ()

    def __init__(
        self,
        name: str,
        phi_param_map: dict[str, int],
        phi_bounds: tuple[tuple[int | None, int | None], ...],
    ) -> None:
        self.name = name
        self.phi_param_map = phi_param_map
        self.phi_bounds = phi_bounds

    def resolve(self, n_stresses: int) -> "LifeModel":
        """
        The life model for ``n_stresses`` stress columns. A model with a
        fixed number of parameters is the same for any number, and
        returns itself (a wrong number of columns is refused by the fit);
        ``GeneralLogLinear`` returns the model with one coefficient per
        column.

        Examples
        --------
        >>> from surpyval.life_models import GeneralLogLinear, Power
        >>> Power.resolve(1) is Power
        True
        >>> GeneralLogLinear.resolve(2).phi_param_map
        {'c': 0, 'beta_0': 1, 'beta_1': 2}
        """
        return self

    @abstractmethod
    def phi(self, Z: ndarray, *params: float) -> ndarray:
        """
        The life :math:`L(Z)` at stress ``Z`` for the life-model
        parameters ``params``. Must be written with ``autograd.numpy`` so
        the fit can differentiate it.
        """

    @abstractmethod
    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        """
        Starting values for the life-model parameters, given the life
        estimated separately at each distinct stress level (``life``, one
        value per row of ``Z``). Typically a least-squares fit of the
        linearised relationship.
        """

    def _stress_terms(
        self, Z: ndarray
    ) -> "tuple[ndarray, tuple[str, ...], bool] | None":
        """
        The terms of the stresses the log-life is linear in, for the
        check of which stress effects the data determine (#503): one
        column per stress column of ``Z`` (``log s`` for a power term,
        ``1 / s`` for an exponential one), the life-model parameter each
        multiplies, and whether the model has a free constant factor
        (an intercept on the log scale, which absorbs a constant
        stress). ``None`` for a life model with no such form, or with one
        stress, where one stress level is refused already.
        """
        return None
