Logistic Distribution
=====================

Location ``mu`` and scale ``sigma``:
:math:`F(x) = 1/\left(1 + e^{-(x - \mu)/\sigma}\right)`. Shaped like
the Normal with heavier tails, and supported on the whole real line. ``Logistic`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.logistic.Logistic_
   :members:
   :inherited-members:
