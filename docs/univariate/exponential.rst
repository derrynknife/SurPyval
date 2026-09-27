Exponential Distribution
========================

The constant-hazard distribution: the failure rate
:math:`\lambda` (parameter ``failure_rate``) is the same at every age,
so :math:`R(x) = e^{-\lambda x}` and the mean life is
:math:`1/\lambda`. It is the model for purely random failures, and the
only memoryless continuous distribution. ``Exponential`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.exponential.Exponential_
   :members:
   :inherited-members:
