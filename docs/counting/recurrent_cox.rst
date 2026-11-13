Semi-Parametric Recurrent-Event Regression
==========================================

Regression for repeated events with the baseline left unspecified: the
Andersen-Gill, Prentice-Williams-Peterson (total and gap time) and
Wei-Lin-Weissfeld models. Each is the Cox partial likelihood on its own
layout of the rows, with the coefficients' robust (sandwich) covariance
clustered by item, as R's ``coxph(... + cluster(id))``. What each model
assumes about a repeat event is set out on :doc:`/Recurrent Event
Regression Analysis`, with worked examples on :doc:`/Recurrent Event
Regression Modelling with SurPyval`.

The fitted model
----------------

.. autoclass:: surpyval.recurrent.regression.recurrent_cox.RecurrentCoxModel
   :members:
   :inherited-members:

Andersen-Gill
-------------

.. autodata:: surpyval.recurrent.regression.recurrent_cox.AndersenGill
   :no-value:

   .. automethod:: surpyval.recurrent.regression.recurrent_cox.AndersenGill.fit
   .. automethod:: surpyval.recurrent.regression.recurrent_cox.AndersenGill.fit_from_df
   .. automethod:: surpyval.recurrent.regression.recurrent_cox.AndersenGill.fit_tvc
   .. automethod:: surpyval.recurrent.regression.recurrent_cox.AndersenGill.fit_tvc_from_df

Prentice-Williams-Peterson
--------------------------

.. autodata:: surpyval.recurrent.regression.recurrent_cox.PrenticeWilliamsPeterson
   :no-value:

   .. automethod:: surpyval.recurrent.regression.recurrent_cox.PrenticeWilliamsPeterson.fit
   .. automethod:: surpyval.recurrent.regression.recurrent_cox.PrenticeWilliamsPeterson.fit_from_df
   .. automethod:: surpyval.recurrent.regression.recurrent_cox.PrenticeWilliamsPeterson.fit_tvc
   .. automethod:: surpyval.recurrent.regression.recurrent_cox.PrenticeWilliamsPeterson.fit_tvc_from_df

Wei-Lin-Weissfeld
-----------------

.. autodata:: surpyval.recurrent.regression.recurrent_cox.WeiLinWeissfeld
   :no-value:

   .. automethod:: surpyval.recurrent.regression.recurrent_cox.WeiLinWeissfeld.fit
   .. automethod:: surpyval.recurrent.regression.recurrent_cox.WeiLinWeissfeld.fit_from_df
