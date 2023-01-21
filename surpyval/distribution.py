from abc import ABC, abstractmethod
from typing import Optional
from numpy.typing import ArrayLike


class Distribution(ABC):
    """
    An abstract base class that all surpyval distributions inherit from,
    implementing the following methods:
    - .sf()
    - .ff()
    """

    @abstractmethod
    def sf(this, x: Optional[ArrayLike]) -> ArrayLike:
        """Survival (or Reliability) function for the Exponential Distribution:

        Parameters
        ----------
        this : _type_
            _description_
        x : Optional[ArrayLike]
            _description_

        Returns
        -------
        ArrayLike
            _description_
        """
        pass
    
    @abstractmethod
    def ff(this, x: Optional[ArrayLike]) -> ArrayLike:
        pass

    @abstractmethod
    def to_dict(this) -> dict:
        pass
