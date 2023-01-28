from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class Distribution(ABC):
    """
    An abstract base class that all surpyval distributions inherit from,
    implementing the following methods:
    - .sf()
    - .ff()
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def sf(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        ...

    @abstractmethod
    def ff(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        ...

    @abstractmethod
    def entropy(self, *args, **kwargs) -> ArrayLike:
        ...

    @abstractmethod
    def moment(self, n: ArrayLike, *args, **kwargs) -> ArrayLike:
        ...

    @abstractmethod
    def random(self, size: ArrayLike, *args, **kwargs) -> ArrayLike:
        ...
    
    @abstractmethod
    def to_dict(self) -> dict:
        ...
