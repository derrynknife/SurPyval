Mixture Models
==============

A population made of ``m`` sub-populations, each following the same
distribution family with its own parameters, mixed in proportions
(weights) ``w``: :math:`F(x) = \sum_i w_i F_i(x)`. Create the fitter
with ``MixtureModel(dist=Weibull, m=2)`` and call its ``fit``, which
fits in place -- the fitted parameters and weights are stored on the
object, which is then used as the model. Worked examples are in
:doc:`../Parametric SurPyval Modelling`.

.. automodule:: surpyval.univariate.parametric.mixture_model
   :members:
   :inherited-members:
