Multivariate Modelling
======================

Joint models for several *correlated* event-time series. A copula
separates the problem into two independent choices: the marginal
distribution of each dimension, which is any surpyval univariate
distribution, and the dependence structure between them, which is the
copula itself.

For a narrative introduction with worked examples, see
:doc:`Multivariate Modelling with SurPyval`.

.. note::

   ``surpyval.multivariate.Gumbel`` is the Gumbel *copula* and is a
   different object from ``surpyval.Gumbel``, the univariate Gumbel
   distribution. The two are unrelated; import the copulas from
   ``surpyval.multivariate`` to keep them apart.

Copulas
-------

Each copula below is exported as a ready-to-use instance --
``Clayton``, ``Gumbel``, ``Frank``, ``Gaussian`` and ``Independence`` --
in the same way the univariate distributions are. The classes are
documented here; the instances carry the same methods.

.. autoclass:: surpyval.multivariate.parametric.copula.copula.Copula
   :members:

.. autoclass::
   surpyval.multivariate.parametric.copula.archimedean.IndependenceCopula
   :members:

.. autoclass::
   surpyval.multivariate.parametric.copula.archimedean.ClaytonCopula
   :members:

.. autoclass::
   surpyval.multivariate.parametric.copula.archimedean.GumbelCopula
   :members:

.. autoclass::
   surpyval.multivariate.parametric.copula.archimedean.FrankCopula
   :members:

.. autoclass::
   surpyval.multivariate.parametric.copula.elliptical.GaussianCopula
   :members:

Fitted Model
------------

.. autoclass:: surpyval.multivariate.parametric.copula.copula_model.CopulaModel
   :members:

Data
----

.. autoclass:: surpyval.multivariate.parametric.data.MultivariateSurpyvalData
   :members:
