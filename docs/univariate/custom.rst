Custom Distributions
====================

Create a distribution that SurPyval does not have by supplying only its
cumulative hazard function :math:`H(x)` as ``fun(x, *params)``, with
the parameter names, their bounds and the support. The survival
function, CDF, hazard and density follow from :math:`H` (the
derivatives by automatic differentiation, so ``fun`` must be written
with ``autograd.numpy``). The result is a fitter like any built-in
distribution: its ``fit`` returns a :doc:`Parametric <parametric_class>`
model. The names ``p``, ``gamma`` and ``f0`` are reserved (for the
limited-failure, offset and zero-inflated parameters) and cannot be used
as parameter names.

.. autoclass:: surpyval.univariate.parametric.distributions.custom_distribution.CustomDistribution
   :members:
