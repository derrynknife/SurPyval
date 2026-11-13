"""
Life models: how a distribution's life parameter depends on stress, for
:class:`~surpyval.univariate.regression.accelerated_life.AcceleratedLife`.

Each life model is a function :math:`L(Z)` of the stress (or of several
stresses); ``AcceleratedLife(dist, model)`` puts it in place of the
distribution's life (scale) parameter:

==========================  ===============================================
``Power``                   :math:`L(V) = a V^{n}`
``InversePower``            :math:`L(V) = 1 / (a V^{n})`
``Exponential``             :math:`L(V) = b e^{a / V}` (the Arrhenius model)
``Arrhenius``               another name for ``Exponential``
``InverseExponential``      :math:`L(V) = 1 / (b e^{a / V})`
``Eyring``                  :math:`L(V) = (1 / V) e^{-(b - a / V)}`
``InverseEyring``           :math:`L(V) = V e^{c - a / V}`
``Linear``                  :math:`L(V) = a + b V`
``DualPower``               :math:`L(U, V) = c U^{m} V^{n}`
``DualExponential``         :math:`L(U, V) = c e^{a / U + b / V}`
``PowerExponential``        :math:`L(U, V) = c e^{a / U} V^{n}`
``GeneralLogLinear``        :math:`L(Z) = c e^{\\sum_j \\beta_j Z_j}`
``GeneralLinear``           :math:`L(Z) = a + \\sum_j \\beta_j Z_j`
==========================  ===============================================

``Arrhenius`` is another name for ``Exponential``: a model built with it
saves and loads as ``Arrhenius`` (its repr says so too) and is otherwise
identical, with the same fit, the same numbers and the same checks.

``LifeModel`` is their base class, for a life model of your own. The
docstring of each model gives its exact form.

``GeneralLogLinear`` with a distribution that has an AFT fitter (the
Weibull, LogNormal and others) is that AFT model: prefer ``WeibullAFT``
and the like, which also take formulas and DataFrames.

``Linear`` and ``GeneralLinear`` are meant for a location parameter (the
``mu`` of ``Normal``, ``Gumbel`` or ``Logistic``): the stresses shift the
distribution, so the fit does not depend on where zero is. On a scale
parameter a fitted line or plane can reach zero or negative life outside
the stresses tested.

Examples
--------
>>> import numpy as np
>>> from surpyval import AcceleratedLife, Weibull, life_models
>>> rng = np.random.default_rng(0)
>>> V = np.repeat([1.0, 2.0, 4.0], 20)
>>> x = Weibull.random(60, 100, 2, random_state=rng) * V ** -1.5
>>> model = AcceleratedLife(Weibull, life_models.Power).fit(x, V)
>>> model.params[2:].round(1)  # a and n
array([106.9,  -1.6])
"""

from surpyval.univariate.regression.accelerated_life import (
    ArrheniusLifeModel,
    DualExponential,
    DualPower,
)
from surpyval.univariate.regression.accelerated_life import (
    ExponentialLifeModel as Exponential,
)
from surpyval.univariate.regression.accelerated_life import (
    Eyring,
    GeneralLinear,
    GeneralLogLinear,
    InverseExponential,
    InverseEyring,
    InversePower,
    LifeModel,
    Linear,
    Power,
    PowerExponential,
)

#: The exponential life model by the name it is known by for temperature:
#: the Arrhenius model, with ``V`` an absolute temperature in kelvin and
#: ``a`` the activation energy over Boltzmann's constant. Another name for
#: ``Exponential``: a model built with it carries the name ``Arrhenius``
#: (its repr, what it saves and loads as) and is otherwise identical.
Arrhenius = ArrheniusLifeModel

__all__ = [
    "Arrhenius",
    "DualExponential",
    "DualPower",
    "Exponential",
    "Eyring",
    "GeneralLinear",
    "GeneralLogLinear",
    "InverseExponential",
    "InverseEyring",
    "InversePower",
    "LifeModel",
    "Linear",
    "Power",
    "PowerExponential",
]
