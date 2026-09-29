Gumbel Distribution
===================

The smallest-extreme-value (left-skewed) Gumbel distribution, with
location ``mu`` and scale ``sigma``:
:math:`R(x) = e^{-e^{(x - \mu)/\sigma}}`. It is the distribution of
the logarithm of a Weibull variable, and is supported on the whole real
line. For the largest-extreme-value version see :doc:`gumbel_lev`. ``Gumbel`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.gumbel.Gumbel_
   :members:
   :inherited-members:
