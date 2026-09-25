Cox Proportional Hazards
========================

The semi-parametric proportional hazards model: the covariates multiply
a baseline hazard that is estimated non-parametrically,
:math:`h(x \mid Z) = h_0(x) e^{\beta' Z}`. ``CoxPH`` is an instance of
the fitter class below. Its fit methods cover ordinary data (``fit``,
``fit_from_df``, with right censoring, left truncation, a choice of tie
handling and optional strata) and time-varying covariates in start-stop
form (``fit_tvc``, ``fit_tvc_timeline`` and their ``_from_df``
versions). The theory is in :doc:`../regression analysis` and worked
examples are in :doc:`../Regression Modelling with SurPyval`.

Fitter
------

.. autoclass:: surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_
   :members:
   :inherited-members:
   :exclude-members: baseline, create_efron_ll_jac_hess, create_breslow_ll_jac_hess, create_exact_ll_jac_hess, create_kalbfleisch_prentice_ll_jac_hess

Fitted model
------------

The object returned by :meth:`CoxPH.fit <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit>`
and the other fit methods. It exposes the usual survival functions at a
covariate vector (``sf``, ``ff``, ``df``, ``hf``, ``Hf``) and, for a
time-varying covariate, the survival along a covariate path: ``sf_tvc`` /
``Hf_tvc`` take a piecewise-constant
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` (or
``(xl, Z)`` arrays), the same interface the parametric families use, while the
older interval-oriented ``predict_tvc`` returns the survival at the baseline
jump times along a subject's ``(xl, xr]`` intervals.

.. autoclass:: surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel
   :members:
