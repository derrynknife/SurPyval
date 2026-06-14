from abc import ABC, abstractmethod

from numpy.typing import ArrayLike


class CountingProcess(ABC):
    """
    Abstract base class for parametric counting-process intensity models.

    A counting process ``N(t)`` counts the number of events observed up to
    time ``t``. Every parametric counting process in surpyval is described
    by functions of time and its parameters:

    - ``iif`` the instantaneous intensity function (the rate of events),
    - ``cif`` the cumulative intensity function (the expected number of
      events by ``t``, i.e. the integral of ``iif`` from the origin), and
    - ``log_iif`` the natural logarithm of the instantaneous intensity,
      used directly in the log-likelihood.

    Concrete subclasses also expose a ``param_names`` attribute listing the
    names of the model's parameters. This shared base lets fitters such as
    :class:`ProportionalIntensityNHPP` verify, with a simple ``isinstance``
    check, that the intensity model handed to them really is a counting
    process.
    """

    @abstractmethod
    def iif(self, x: ArrayLike, *params) -> ArrayLike:
        """Instantaneous intensity function (event rate) at ``x``."""
        ...

    @abstractmethod
    def cif(self, x: ArrayLike, *params) -> ArrayLike:
        """Cumulative intensity (expected event count) by ``x``."""
        ...

    @abstractmethod
    def log_iif(self, x: ArrayLike, *params) -> ArrayLike:
        """Natural logarithm of the instantaneous intensity at ``x``."""
        ...
