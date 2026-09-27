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

The frailty is Gamma-distributed (the only family currently available),
so it integrates out of each group's likelihood in closed form. Only
observed and right-censored data are supported, and at least two groups
are needed. The fitted model predicts the *marginal* (population) curve
by default, or the curve conditional on an observed group's posterior
frailty (``group=``) or on a given frailty (``frailty=``). Like the
parametric regression models it reports ``neg_ll()``, ``aic()``,
``bic()`` and ``aic_c()`` (``theta`` counted as a parameter), so a
frailty fit can be compared directly with the proportional-hazards fit it
reduces to at ``theta = 0``.

.. autofunction:: surpyval.univariate.regression.frailty.Frailty

.. autoclass:: surpyval.univariate.regression.frailty.frailty_fitter.FrailtyFitter
    :members: fit, fit_from_df

.. autoclass:: surpyval.univariate.regression.frailty.frailty_model.FrailtyModel
    :members:
