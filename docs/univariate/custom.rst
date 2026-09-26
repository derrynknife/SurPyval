Custom Distributions
====================

Create a distribution that SurPyval does not have by supplying only its
cumulative hazard function :math:`H(x)` as ``fun(x, *params)``, with
the parameter names, their bounds and the support. The survival
function, CDF, hazard and density follow from :math:`H` (the
derivatives by automatic differentiation, so ``fun`` must be written
with ``autograd.numpy``). The result is a fitter like any built-in
distribution: its ``fit`` returns a :doc:`Parametric <parametric_class>`
model. The names ``gamma`` and ``f0`` are reserved (for the offset and
zero-inflated parameters) and cannot be used as parameter names; a
parameter named ``p`` is allowed, and the limited-failure proportion of
such a model is then called ``lfp_p``. The distribution also has a
numerical ``qf`` (so ``random`` works) and moments computed on its own
scale. A model of it can be saved with ``to_dict`` and read back with
``from_dict`` in any session that has constructed the distribution
again under the same name (see the class docstring).

.. autoclass:: surpyval.univariate.parametric.distributions.custom_distribution.CustomDistribution
   :members:
