Gamma Distribution
==================

Shape ``alpha`` (:math:`\alpha`) and *rate* ``beta``
(:math:`\beta`, the reciprocal of the scale):
:math:`F(x) = \gamma(\alpha, \beta x)/\Gamma(\alpha)`, mean
:math:`\alpha/\beta`. With integer :math:`\alpha` it is the time to
the :math:`\alpha`-th event of a Poisson process (the Erlang
distribution). ``Gamma`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.gamma.Gamma_
   :members:
   :inherited-members:
