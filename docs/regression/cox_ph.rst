Cox Proportional Hazards
========================

Fitter
------

.. autoclass:: surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_
   :members:
   :inherited-members:

Fitted model
------------

The object returned by :meth:`CoxPH.fit` / :meth:`CoxPH.fit_tvc`. It exposes
the usual survival functions at a covariate vector (``sf``, ``ff``, ``df``,
``hf``, ``Hf``) and, for a time-varying covariate, the survival along a
covariate path: ``sf_tvc`` / ``Hf_tvc`` take a piecewise-constant
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` (or
``(xl, Z)`` arrays), the same interface the parametric families use, while the
older interval-oriented ``predict_tvc`` returns the survival at the baseline
jump times along a subject's ``(xl, xr]`` intervals.

.. autoclass:: surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel
   :members:
