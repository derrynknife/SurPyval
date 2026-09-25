Four-Parameter Beta Distribution
================================

The Beta distribution stretched from :math:`[0, 1]` to an interval
:math:`[a, b]`: shapes ``alpha`` and ``beta``, and support ends ``a``
and ``b``, all four fitted. For the unit interval see :doc:`beta`. ``Beta4`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters; both return a
:doc:`Parametric <parametric_class>` model.

.. autoclass:: surpyval.univariate.parametric.distributions.beta4.Beta4_
   :members:
   :inherited-members:
