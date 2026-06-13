from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike


class Distribution(ABC):
    """
    Root abstract base class that every surpyval model inherits from.

    The contract shared by all models -- parametric, nonparametric,
    mixtures, compositions and degenerate models -- is the survival
    interface:

    - ``sf()`` the survival (reliability) function
    - ``ff()`` the cumulative distribution (failure) function

    ``Hf()`` is provided as a default derived from ``sf()`` and may be
    overridden by models with a closed form. Statistical summaries such
    as ``moment``, ``entropy``, ``random`` and ``to_dict`` are not part
    of this minimal contract; the ``ParametricDistribution`` and
    ``NonParametricDistribution`` subclasses add the ones appropriate to
    their model family.
    """

    @abstractmethod
    def sf(self, x: ArrayLike, *args, **kwargs) -> ArrayLike: ...

    @abstractmethod
    def ff(self, x: ArrayLike, *args, **kwargs) -> ArrayLike: ...

    def Hf(self, x: ArrayLike, *args, **kwargs) -> ArrayLike:
        # Cumulative hazard derived from the survival function. Models
        # with a closed-form cumulative hazard override this.
        return -np.log(self.sf(x, *args, **kwargs))


class ParametricDistribution(Distribution):
    """
    A fully specified parametric model. In addition to the survival
    interface it supports random sampling and the standard statistical
    summaries (moments, entropy) and can be serialised with ``to_dict``.
    """

    @abstractmethod
    def random(self, size: ArrayLike, *args, **kwargs) -> ArrayLike: ...

    @abstractmethod
    def moment(self, n: ArrayLike, *args, **kwargs) -> ArrayLike: ...

    @abstractmethod
    def entropy(self, *args, **kwargs) -> ArrayLike: ...

    @abstractmethod
    def to_dict(self) -> dict: ...


class NonParametricDistribution(Distribution):
    """
    An empirical model produced by a nonparametric estimator
    (Kaplan-Meier, Nelson-Aalen, Fleming-Harrington or Turnbull). Adds
    random sampling from the fitted estimate to the survival interface.
    """

    @abstractmethod
    def random(self, size: ArrayLike, *args, **kwargs) -> ArrayLike: ...
