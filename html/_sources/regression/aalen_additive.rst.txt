Non-Parametric Additive Hazards (Aalen)
=======================================

Aalen's additive hazards model: each covariate adds to the hazard an
amount that may change freely with time,
:math:`h(x \mid Z) = b_0(x) + b(x)' Z`. The fit estimates the
cumulative regression functions :math:`B_j(x) = \int_0^x b_j`, a
least-squares fit at each event time, with their optional-variation
variance (R's ``timereg::aalen`` with ``robust = 0``). For effects
constant in time see :doc:`additive_hazards` (Lin and Ying). Supports
observed and right-censored data, with delayed entry.

As for the Lin-Ying model, nothing keeps the estimated hazard positive,
and the model predicts with the running maximum of the estimated
cumulative hazard from time 0, so ``sf`` stays in :math:`[0, 1]` and
never increases.

.. autoclass:: surpyval.univariate.regression.additive_hazards.aalen_additive.AalenAdditive_
   :members:
   :inherited-members:

The fitted model:

.. autoclass:: surpyval.univariate.regression.additive_hazards.aalen_additive.AalenAdditiveModel
   :members:
