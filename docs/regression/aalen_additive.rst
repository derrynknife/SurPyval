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

**Where the fit stops.** Each increment is a least-squares fit over the
units at risk, so it needs enough of them, and varied enough, to separate
the covariates. Where they run out before the last event (for example,
every unit with a binary covariate at 1 has left), the fit stops at the
last event time it can estimate. It warns with that time, the units at
risk at the next event time and the number of events left out, and
records the time in ``stopped_at``. Every prediction after it (``sf``,
``ff``, ``Hf``, ``hf``, ``qf``, ``cb``, ``cumulative_coefficients`` and
their bounds) is ``nan``, and a ``random`` draw past it is ``nan``: what
happens after the stop is unknown. R's ``timereg`` holds the cumulative
coefficients at their last values to the end of the data instead, as
SurPyval did before v0.30, which understates the risk wherever events
go on. Up to the stop the two agree. To estimate further, use fewer
covariates or a ridge penalty (``penalizer=``). A fit that does not stop
has ``stopped_at`` ``nan`` and holds its curves after the last event
time, as before.

.. autoclass:: surpyval.univariate.regression.additive_hazards.aalen_additive.AalenAdditive_
   :members:
   :inherited-members:

The fitted model:

.. autoclass:: surpyval.univariate.regression.additive_hazards.aalen_additive.AalenAdditiveModel
   :members:
