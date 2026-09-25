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
    ``Linear``, ``DualExponential``, ``DualPower`` and
    ``PowerExponential``.
    """

    def __init__(
        self,
        name: str,
        phi_param_map: dict[str, int],
        phi_bounds: tuple[tuple[int | None, int | None], ...],
    ) -> None:
        self.name = name
        self.phi_param_map = phi_param_map
        self.phi_bounds = phi_bounds

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
