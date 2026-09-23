Shared Frailty Models
=====================

A frailty model adds an unobserved multiplicative random effect shared
by every observation in a group, so that units from the same group are
correlated:

.. math::

    h(x \mid Z, w) = w \, h_0(x) \, e^{\beta' Z}

with the frailty :math:`w` drawn once per group from a distribution with
mean one and variance :math:`\theta`. Setting :math:`\theta = 0`
recovers the ordinary proportional-hazards model, so the fitted
:math:`\theta` measures how much of the variation is between groups
rather than within them.

Use this when observations arrive in clusters that share something you
have not measured -- repairs on the same machine, patients at the same
hospital, components from the same batch. Treating them as independent
understates the uncertainty.

Factory::

    from surpyval import Frailty, Weibull
    model = Frailty(Weibull).fit(x, Z=Z, c=c, groups=unit_id)

Pre-built instances: ``ExponentialFrailty``, ``WeibullFrailty``,
``LogNormalFrailty``, ``GammaFrailty``.

.. autoclass:: surpyval.univariate.regression.frailty.FrailtyFitter
    :members:

.. autoclass:: surpyval.univariate.regression.frailty.frailty_model.FrailtyModel
    :members:
