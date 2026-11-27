Non-Parametric Model
====================

The fitted model returned by every non-parametric estimator
(``KaplanMeier``, ``NelsonAalen``, ``FlemingHarrington``, ``Turnbull``).
It holds the estimate at the distinct times of the data -- ``x``, the
numbers at risk ``r`` and dying ``d``, and the survival ``R`` -- and
provides the survival, failure, hazard and quantile functions between
those times, pointwise confidence bounds (``cb``) and simultaneous bands
(``band``), the mean and restricted mean survival time, and plotting.
How to use it is shown in :doc:`../Non-Parametric SurPyval Modelling`.

.. autoclass:: surpyval.univariate.nonparametric.nonparametric.NonParametric
   :members:
