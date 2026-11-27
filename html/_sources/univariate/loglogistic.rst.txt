LogLogistic Distribution
========================

A variable whose logarithm is Logistic, with scale ``alpha`` (the
median) and shape ``beta``:
:math:`R(x) = 1/\left(1 + (x/\alpha)^\beta\right)`. For
:math:`\beta > 1` its hazard rises and then falls, which suits
failures that become less likely once a unit has survived a while. ``LogLogistic`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.loglogistic.LogLogistic_
   :members:
   :inherited-members:
