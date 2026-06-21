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
    def random(
        self, size: int | tuple[int, ...], *args, **kwargs
    ) -> ArrayLike: ...

    @abstractmethod
    def moment(self, n: int, *args, **kwargs) -> ArrayLike: ...

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
    def random(self, size: int, *args, **kwargs) -> ArrayLike: ...


class MultivariateDistribution(ABC):
    """
    A jointly-specified model of several correlated event-time series
    (the ``multivariate`` outcome-dimension axis of ``MODEL_ATLAS.md``).

    Unlike :class:`Distribution`, whose functions take a single random
    variable, the multivariate interface is evaluated at a *point in
    several dimensions* -- ``x`` is array-like with one column per series.
    Concrete implementations (e.g. copula models) glue together ordinary
    univariate surpyval margins with a dependence structure, so the
    contract is the joint survival interface plus sampling:

    - ``cdf()`` the joint cumulative distribution ``P(X_1<=x_1, ...)``
    - ``sf()``  the joint survival ``P(X_1>x_1, ...)``
    - ``pdf()`` the joint density
    - ``random()`` draw correlated samples (one row per realisation)
    """

    @abstractmethod
    def cdf(self, x: ArrayLike, *args, **kwargs) -> ArrayLike: ...

    @abstractmethod
    def sf(self, x: ArrayLike, *args, **kwargs) -> ArrayLike: ...

    @abstractmethod
    def pdf(self, x: ArrayLike, *args, **kwargs) -> ArrayLike: ...

    @abstractmethod
    def random(
        self, size: int | tuple[int, ...], *args, **kwargs
    ) -> ArrayLike: ...
