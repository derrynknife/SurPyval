"""

Parametric Analysis
===================

.. code:: python

    import surpyval.parametric as para

    model = para.Weibull.fit(x)

"""

from typing import TYPE_CHECKING as _TYPE_CHECKING
from typing import Any as _Any

from .discrete_fitter import DiscreteParametricFitter
from .distributions import (
    Bernoulli,
    Beta,
    Beta4,
    BetaGeometric,
    Binomial,
    CustomDistribution,
    DiscreteWeibull,
    Discretize,
    DiscretizedFitter,
    ExactEventTime,
    Exponential,
    ExpoWeibull,
    FixedEventProbability,
    Galton,
    Gamma,
    Gauss,
    Geometric,
    Gumbel,
    GumbelLEV,
    Hypoexponential,
    InstantlyOccurs,
    Logistic,
    LogLogistic,
    LogNormal,
    NegativeBinomial,
    NeverOccurs,
    Normal,
    Poisson,
    Rayleigh,
    Uniform,
    Weibull,
)
from .likelihood_contour import (
    LikelihoodContour,
    LikelihoodContourPlot,
    plot_likelihood_contour,
)
from .mixture_model import MixtureModel
from .parametric import Parametric
from .parametric_fitter import OptimisedFitMixin, ParametricFitter
from .royston_parmar import RoystonParmar, RoystonParmarModel
from .weibayes import weibayes

if _TYPE_CHECKING:
    from ._bootstrap import ParametricBootstrap as ParametricBootstrap


def __getattr__(name: str) -> _Any:
    # ``ParametricBootstrap`` on first use: its module imports the
    # regressions' bootstrap, and with it the regression package, pandas,
    # formulaic and scipy.stats, which ``import surpyval`` leaves alone.
    if name == "ParametricBootstrap":
        from . import _bootstrap

        globals()[name] = _bootstrap.ParametricBootstrap
        return _bootstrap.ParametricBootstrap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"ParametricBootstrap"})
