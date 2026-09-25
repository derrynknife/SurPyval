Normal Distribution
===================

The Normal (Gaussian) distribution with mean ``mu`` and standard
deviation ``sigma``. It is supported on the whole real line, so as a
lifetime model it suits wear-out data far from zero. Also exported as
``Gauss``. ``Normal`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.normal.Normal_
   :members:
   :inherited-members:
