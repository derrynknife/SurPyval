Custom Distributions
====================

Create a distribution that SurPyval does not have by supplying only its
cumulative hazard function :math:`H(x)` as ``fun(x, *params)`` (the
star-argument may have any name, or each parameter its own named
argument, ``fun(x, nu, b)``), with the parameter names, their bounds and
the support. The survival
function, CDF, hazard and density follow from :math:`H` (the
derivatives by automatic differentiation, so ``fun`` must be written
with ``autograd.numpy``). The result is a fitter like any built-in
distribution: its ``fit`` returns a :doc:`Parametric <parametric_class>`
model. The names ``gamma`` and ``f0`` are reserved (for the offset and
zero-inflated parameters) and cannot be used as parameter names, nor
can the names of a fitted model's own attributes (``k``, ``dist``,
``data``, ``method``, ``sf``, ...), since the model exposes each
parameter as an attribute; a parameter named ``p`` is allowed, and is the model's ``p``
(the limited-failure proportion is ``lfp_p``, a reserved name). The distribution also has a
numerical ``qf`` (so ``random`` works) and moments computed on its own
scale. ``qf`` inverts every probability at once when ``fun`` broadcasts
(elementwise ``autograd.numpy`` arithmetic does); a ``fun`` that indexes
or reduces its argument is detected -- on an array it does not give each
point's own value -- and is inverted one probability at a time, with the
same answers, more slowly. A model of it can be saved with ``to_dict`` and read back with
``from_dict`` in any session that has constructed the distribution
again under the same name (see the class docstring); constructing a
second distribution under a name already used in the session warns that
it replaces the first for that purpose.

.. autoclass:: surpyval.univariate.parametric.distributions.custom_distribution.CustomDistribution
   :members:
