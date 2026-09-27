Comparison Tests and Validation Metrics
=======================================

Tools that compare models or groups rather than fit one: hypothesis tests
of whether groups share a survival (or cumulative-incidence) curve, the
restricted-mean difference between two groups, automatic selection of the
best-fitting distribution, and metrics that score how well a model's
predicted survival matches held-out data. All are importable directly
from ``surpyval``. The tests are explained with the estimators they
build on in :doc:`Non-Parametric Estimation` (log-rank, restricted mean)
and :doc:`Competing Risks Analysis` (Gray's test); model selection in
:doc:`Parametric SurPyval Modelling`; and the validation metrics in
:doc:`Regression Modelling with SurPyval`.

Group-comparison tests
-----------------------

The (weighted, optionally stratified) log-rank test for comparing survival
distributions across groups, and the result it returns:

.. autofunction:: surpyval.univariate.nonparametric.logrank.logrank

.. autoclass:: surpyval.univariate.nonparametric.logrank.LogRankResult
   :no-members:

Gray's test for comparing cumulative incidence functions across groups under
competing risks:

.. autofunction:: surpyval.univariate.competing_risks.nonparametric.gray_test.gray_test

.. autoclass:: surpyval.univariate.competing_risks.nonparametric.gray_test.GrayTestResult
   :members:
   :exclude-members: count, index

Restricted mean survival time
-----------------------------

The two-group restricted-mean-survival-time difference (the per-model
``rmst`` method lives on the
:doc:`non-parametric model class <univariate/non_parametric_class>`):

.. autofunction:: surpyval.univariate.nonparametric.nonparametric.rmst_diff

Automatic distribution selection
--------------------------------

Fit every candidate continuous distribution and keep the one with the
best information criterion:

.. autofunction:: surpyval.fit_best.fit_best

Prediction-validation metrics
-----------------------------

Right-censored-standard metrics for scoring a predicted survival function
(Brier / integrated Brier score and Uno's time-dependent AUC), plus the helper
that builds a predicted-survival matrix from any fitted model exposing
``sf(x, Z)``.

.. automodule:: surpyval.metrics.validation
   :members:
