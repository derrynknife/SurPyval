Semi-Parametric Additive Hazards
================================

The Lin & Ying additive hazards model: covariates *add* a constant
amount to the baseline hazard rather than multiplying it,
:math:`h(x \mid Z) = h_0(x) + \beta' Z`, with the baseline left to the
data. ``AdditiveHazards`` is an instance of the fitter class below. The
fully parametric version, with a Weibull (or other) baseline, is the
``AH`` family in :doc:`parametric`. Supports observed and right-censored
data.

.. autoclass:: surpyval.univariate.regression.additive_hazards.additive_hazards.AdditiveHazards_
   :members:
   :inherited-members:

The fitted model:

.. autoclass:: surpyval.univariate.regression.additive_hazards.additive_hazards.AdditiveHazardsModel
   :members:
