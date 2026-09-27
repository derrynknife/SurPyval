LogNormal Distribution
======================

A variable whose logarithm is Normal with mean ``mu`` and standard
deviation ``sigma``:
:math:`F(x) = \Phi\left((\ln x - \mu)/\sigma\right)`, so the median
life is :math:`e^{\mu}`. Common for fatigue and repair times. Also
exported as ``Galton``. ``LogNormal`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.lognormal.LogNormal_
   :members:
   :inherited-members:
