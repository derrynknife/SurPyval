Weibull Distribution
====================

The workhorse of reliability engineering, with scale ``alpha``
(:math:`\alpha`, the characteristic life, reached by 63.2% of the
population) and shape ``beta`` (:math:`\beta`):
:math:`R(x) = e^{-(x/\alpha)^\beta}`. The hazard falls with age when
:math:`\beta < 1` (infant mortality), is constant when
:math:`\beta = 1` (the Exponential) and rises when :math:`\beta > 1`
(wear-out). ``Weibull`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.weibull.Weibull_
   :members:
   :inherited-members:
