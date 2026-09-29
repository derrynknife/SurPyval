Uniform Distribution
====================

Equally likely anywhere between ``a`` and ``b``:
:math:`F(x) = (x - a)/(b - a)`. Both ends of the support are fitted
parameters. ``Uniform`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.uniform.Uniform_
   :members:
   :inherited-members:
