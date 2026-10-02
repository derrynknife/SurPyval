"""The covariate link a fitted parametric regression model carries.

A fitted ``ParametricRegressionModel`` keeps, as ``reg_model``, how its
covariates act: a :class:`CovariateLink` for the proportional hazards,
accelerated failure time, proportional odds and additive hazards
families (a life model, ``LifeModel``, for accelerated life). The
printed model shows its
``name``, ``to_dict`` stores its ``name`` and ``phi_param_map``, and
``phi()`` evaluates its ``phi``.
"""

from __future__ import annotations

from typing import Any, Callable


class CovariateLink:
    """How the covariates enter a fitted regression model.

    Parameters
    ----------
    name : str
        The display name of the link, shown in the model's ``repr``; for
        the built-in links also the name ``to_dict`` stores and
        ``from_dict`` rebuilds it from.
    phi_param_map : dict
        ``{coefficient name: position}`` of the covariate coefficients,
        positions counted from the first coefficient.
    phi : callable, optional
        The covariate function ``phi(Z, *coefficients)`` that multiplies
        the baseline (``exp(beta'Z)``, or a custom one). ``None``, the
        default, for an additive link, whose ``beta'Z`` is added to the
        hazard rather than multiplying it.
    """

    #: The display (and, for the built-in links, serialisation) name.
    name: str
    #: ``{coefficient name: position}`` of the covariate coefficients.
    phi_param_map: dict[str, int]
    #: The multiplier ``phi(Z, *coefficients)``; ``None`` for an additive
    #: link, which has none.
    phi: "Callable[..., Any] | None" = None

    def __init__(
        self,
        name: str,
        phi_param_map: dict[str, int],
        phi: "Callable[..., Any] | None" = None,
    ) -> None:
        self.name = name
        self.phi_param_map = phi_param_map
        if phi is not None:
            self.phi = phi
