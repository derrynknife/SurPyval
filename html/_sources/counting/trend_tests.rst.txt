Trend Tests and Diagnostics
===========================

Hypothesis tests for whether a system's rate of events is changing
over time -- the question that decides between an HPP and an NHPP --
plus the result classes returned by the tests and by the per-model
goodness-of-fit method.

The corresponding per-model diagnostics (``residuals``,
``trend_test``, ``cramer_von_mises``) are methods on the fitted model
classes; see :doc:`parametric_recurrence_model`.

.. autofunction:: surpyval.recurrent.tests.laplace

.. autofunction:: surpyval.recurrent.tests.mil_hdbk_189c

.. autoclass:: surpyval.recurrent.tests.TrendTestResult
   :members:

.. autoclass:: surpyval.recurrent.diagnostics.GoodnessOfFitResult
   :members:
