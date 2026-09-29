Semi-Parametric Additive Hazards
================================

The Lin & Ying additive hazards model: covariates *add* a constant
amount to the baseline hazard rather than multiplying it,
:math:`h(x \mid Z) = h_0(x) + \beta' Z`, with the baseline left to the
data. ``AdditiveHazards`` is an instance of the fitter class below. The
fully parametric version, with a Weibull (or other) baseline, is the
``AH`` family in :doc:`parametric`. Supports observed and right-censored
data.

Nothing keeps the estimated hazard :math:`h_0(x) + \beta' Z` positive: the
Lin-Ying cumulative hazard falls between the event times, where its
baseline drifts down by :math:`\beta' \bar{Z}(x)`, and for a covariate row
the model takes below zero. The model therefore predicts with the running
maximum of the estimate from time 0, the smallest non-decreasing cumulative
hazard at or above it and 0 (#376): ``sf`` stays in :math:`[0, 1]` and never
increases, and ``hf`` is 0 wherever ``Hf`` is held. Where the estimate is at
its running maximum (most of the time for a covariate row inside the data)
the prediction is the estimate itself; the fitted baseline ``H0`` is the raw
estimate at :math:`Z = 0`, as R's ``timereg`` reports it.

.. autoclass:: surpyval.univariate.regression.additive_hazards.additive_hazards.AdditiveHazards_
   :members:
   :inherited-members:

The fitted model:

.. autoclass:: surpyval.univariate.regression.additive_hazards.additive_hazards.AdditiveHazardsModel
   :members:
