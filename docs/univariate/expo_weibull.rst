Exponentiated Weibull Distribution
==================================

A Weibull CDF raised to a power: parameters ``alpha`` (scale),
``beta`` and ``mu`` (shapes), with
:math:`F(x) = \left[1 - e^{-(x/\alpha)^\beta}\right]^{\mu}`. The
extra shape lets the hazard be bathtub-shaped or unimodal as well as
monotone; ``mu = 1`` is the Weibull. ``ExpoWeibull`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.expo_weibull.ExpoWeibull_
   :members:
   :inherited-members:
