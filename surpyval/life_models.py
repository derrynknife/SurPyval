"""
Life models: how a distribution's life parameter depends on stress, for
:class:`~surpyval.univariate.regression.accelerated_life.AcceleratedLife`.

Each life model is a function :math:`L(Z)` of the stress (or of several
stresses); ``AcceleratedLife(dist, model)`` puts it in place of the
distribution's life (scale) parameter:

==========================  ===============================================
``Power``                   :math:`L(V) = a V^{n}`
``InversePower``            :math:`L(V) = 1 / (a V^{n})`
``Exponential``             :math:`L(V) = b e^{a / V}`
``InverseExponential``      :math:`L(V) = 1 / (b e^{a / V})`
``Eyring``                  :math:`L(V) = (1 / V) e^{-(b - a / V)}`
``InverseEyring``           :math:`L(V) = V e^{c - a / V}`
``Linear``                  :math:`L(V) = a + b V`
``DualPower``               :math:`L(U, V) = c U^{m} V^{n}`
``DualExponential``         :math:`L(U, V) = c e^{a / U + b / V}`
``PowerExponential``        :math:`L(U, V) = c e^{a / U} V^{n}`
``GeneralLogLinear``        :math:`L(Z) = c e^{\\sum_j \\beta_j Z_j}`
==========================  ===============================================

``LifeModel`` is their base class, for a life model of your own. The
docstring of each model gives its exact form.

``GeneralLogLinear`` with a distribution that has an AFT fitter (the
Weibull, LogNormal and others) is that AFT model: prefer ``WeibullAFT``
and the like, which also take formulas and DataFrames.

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
    DualExponential,
    DualPower,
)
from surpyval.univariate.regression.accelerated_life import (
    ExponentialLifeModel as Exponential,
)
from surpyval.univariate.regression.accelerated_life import (
    Eyring,
    GeneralLogLinear,
    InverseExponential,
    InverseEyring,
    InversePower,
    LifeModel,
    Linear,
    Power,
    PowerExponential,
)

__all__ = [
    "DualExponential",
    "DualPower",
    "Exponential",
    "Eyring",
    "GeneralLogLinear",
    "InverseExponential",
    "InverseEyring",
    "InversePower",
    "LifeModel",
    "Linear",
    "Power",
    "PowerExponential",
]
