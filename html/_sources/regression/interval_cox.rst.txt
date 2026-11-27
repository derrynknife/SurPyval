Interval-Censored Cox (Semi-Parametric PH)
==========================================

The proportional hazards model with the baseline left to the data, for
interval-censored data: :math:`S(x \mid Z) = S_0(x)^{\exp(\beta' Z)}`,
where :math:`S_0` is a non-parametric maximum-likelihood estimate on the
Turnbull intervals, fitted jointly with :math:`\beta` (R's
``icenReg::ic_sp(model = "ph")``). Takes exact, right-, left- and
interval-censored rows, counts and left truncation. ``IntervalCoxPH`` is
an instance of the fitter class below. The guide is
:doc:`../Interval-Censored Cox Regression`.

.. autoclass:: surpyval.univariate.regression.proportional_hazards.interval_cox.IntervalCoxPH_
   :members:
   :inherited-members:

The fitted model:

.. autoclass:: surpyval.univariate.regression.proportional_hazards.interval_cox.IntervalCoxPHModel
   :members:
