from abc import ABC, abstractmethod

from numpy import ndarray


class LifeModel(ABC):
    def __init__(
        self,
        name: str,
        phi_param_map: dict[str, int],
        phi_bounds: tuple[tuple[int | None, int | None]],
    ):
        self.name = name
        self.phi_param_map = phi_param_map
        self.phi_bounds = phi_bounds

    @abstractmethod
    def phi(
        self, Z: ndarray, *params: float
    ) -> ndarray:
        ...

    @abstractmethod
    def phi_init(self, life: float, Z: ndarray) -> list[float]:
        ...
