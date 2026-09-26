Four-Parameter Beta Distribution
================================

The Beta distribution stretched from :math:`[0, 1]` to an interval
:math:`[a, b]`: shapes ``alpha`` and ``beta``, and support ends ``a``
and ``b``, all four fitted. For the unit interval see :doc:`beta`. ``Beta4`` is an instance of the class
below: call its ``fit`` to estimate the parameters from data, or
``from_params`` for known parameters (which requires ``a < b``); both
return a :doc:`Parametric <parametric_class>` model.

A caution: like a three-parameter Weibull, the likelihood is unbounded
when a shape parameter is below 1 and the matching end of the support
moves onto the most extreme observation (``alpha < 1`` with ``a`` at the
smallest value, or ``beta < 1`` with ``b`` at the largest), since the
density there is infinite. The fit returns a local maximum away from
that edge; check that the fitted ``a`` and ``b`` are plausible, and
consider ``how='MPS'``, which is not drawn to the edge.

.. autoclass:: surpyval.univariate.parametric.distributions.beta4.Beta4_
   :members:
   :inherited-members:
