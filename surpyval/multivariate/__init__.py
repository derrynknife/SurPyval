"""Multivariate (jointly-modelled, correlated) survival models.

This module opens the ``multivariate`` outcome-dimension axis of
``MODEL_ATLAS.md``: several correlated event-time series modelled jointly.
v1 provides bivariate **copulas** built on top of the existing univariate
surpyval distributions, with full censoring and truncation support in the
joint likelihood.

Accessed as a package, mirroring ``surpyval.recurrent``::

    import surpyval as surv
    from surpyval.multivariate import Clayton

    model = Clayton.fit(
        [x1, x2], margins=[surv.Weibull, surv.LogNormal], c=[c1, c2]
    )
    model.kendall_tau()
    model.random(1000)
"""

from .parametric import (
    Clayton,
    Copula,
    CopulaModel,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
    MultivariateSurpyvalData,
)

__all__ = [
    "Copula",
    "CopulaModel",
    "MultivariateSurpyvalData",
    "Independence",
    "Clayton",
    "Gumbel",
    "Frank",
    "Gaussian",
]
