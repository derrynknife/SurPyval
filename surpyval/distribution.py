from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

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

    Examples
    --------
    It is abstract; every fitted model is one, whatever its family:

    >>> import numpy as np
    >>> from surpyval import KaplanMeier, Weibull
    >>> from surpyval.distribution import Distribution
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> models = [Weibull.fit(x), KaplanMeier.fit(x)]
    >>> all(isinstance(m, Distribution) for m in models)
    True
    >>> [m.sf([2.5]).round(4) for m in models]
    [array([0.609]), array([0.6])]
    """

    @abstractmethod
    def sf(self, x: ArrayLike, *args: Any, **kwargs: Any) -> ArrayLike:
        """The survival (reliability) function at ``x``."""

    @abstractmethod
    def ff(self, x: ArrayLike, *args: Any, **kwargs: Any) -> ArrayLike:
        """The failure (cumulative distribution) function at ``x``."""

    def Hf(self, x: ArrayLike, *args: Any, **kwargs: Any) -> ArrayLike:
        """The cumulative hazard, ``-log sf(x)`` unless overridden."""
        # Cumulative hazard derived from the survival function. Models
        # with a closed-form cumulative hazard override this.
        return -np.log(self.sf(x, *args, **kwargs))


class ParametricDistribution(Distribution):
    """
    A fully specified parametric model. In addition to the survival
    interface it supports random sampling and the standard statistical
    summaries (moments, entropy) and can be serialised with ``to_dict``.

    Examples
    --------
    It is abstract; a model from a parametric fitter is one:

    >>> from surpyval import Weibull
    >>> from surpyval.distribution import ParametricDistribution
    >>> model = Weibull.from_params([10, 2])
    >>> isinstance(model, ParametricDistribution)
    True
    >>> model.sf([5, 10]).round(4)
    array([0.7788, 0.3679])
    >>> round(float(model.moment(1)), 4)
    8.8623
    """

    @abstractmethod
    def random(
        self, size: int | tuple[int, ...], *args: Any, **kwargs: Any
    ) -> ArrayLike:
        """Draw random samples from the model. ``random_state`` (an int or
        a ``numpy.random.Generator``) gives a draw of its own; ``None``
        draws from numpy's global stream, so ``np.random.seed``
        reproduces it (Conventions, "Random draws and seeds")."""

    @abstractmethod
    def moment(self, n: int, *args: Any, **kwargs: Any) -> ArrayLike:
        """The ``n``-th raw moment."""

    @abstractmethod
    def entropy(self, *args: Any, **kwargs: Any) -> ArrayLike:
        """The differential entropy."""

    @abstractmethod
    def to_dict(self) -> dict:
        """Serialise the model to a plain dictionary."""


class NonParametricDistribution(Distribution):
    """
    An empirical model produced by a nonparametric estimator
    (Kaplan-Meier, Nelson-Aalen, Fleming-Harrington or Turnbull). Adds
    random sampling from the fitted estimate to the survival interface.

    Examples
    --------
    It is abstract; a model from a non-parametric fitter is one:

    >>> import numpy as np
    >>> from surpyval import KaplanMeier
    >>> from surpyval.distribution import NonParametricDistribution
    >>> model = KaplanMeier.fit(np.array([1, 2, 3, 4, 5]))
    >>> isinstance(model, NonParametricDistribution)
    True
    >>> model.sf([2.5, 4]).round(4)
    array([0.6, 0.2])
    """

    @abstractmethod
    def random(self, size: int, *args: Any, **kwargs: Any) -> ArrayLike:
        """Draw random samples from the fitted estimate, with
        ``random_state`` as for :meth:`ParametricDistribution.random`."""


class MultivariateDistribution(ABC):
    """
    A jointly-specified model of several correlated event-time series.

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

    Examples
    --------
    It is abstract; a copula model is one:

    >>> from surpyval import Weibull
    >>> from surpyval.distribution import MultivariateDistribution
    >>> from surpyval.multivariate import Clayton
    >>> margins = [
    ...     Weibull.from_params([10, 2]),
    ...     Weibull.from_params([20, 3]),
    ... ]
    >>> model = Clayton.from_params([2.0], margins)
    >>> isinstance(model, MultivariateDistribution)
    True
    >>> model.sf([[5, 15], [10, 20]]).round(4)
    array([0.624 , 0.2354])
    """

    @abstractmethod
    def cdf(self, x: ArrayLike, *args: Any, **kwargs: Any) -> ArrayLike:
        """The joint CDF at each row of ``x``."""

    @abstractmethod
    def sf(self, x: ArrayLike, *args: Any, **kwargs: Any) -> ArrayLike:
        """The joint survival function at each row of ``x``."""

    @abstractmethod
    def pdf(self, x: ArrayLike, *args: Any, **kwargs: Any) -> ArrayLike:
        """The joint density at each row of ``x``."""

    @abstractmethod
    def random(
        self, size: int | tuple[int, ...], *args: Any, **kwargs: Any
    ) -> ArrayLike:
        """Draw correlated samples, one row per realisation, with
        ``random_state`` as for :meth:`ParametricDistribution.random`."""
