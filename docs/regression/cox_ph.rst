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
``hf``, ``Hf``) and, for models fitted from time-varying-covariate (start-stop)
data, ``predict_tvc`` for the survival of a subject along a covariate path.

.. autoclass:: surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel
   :members:
