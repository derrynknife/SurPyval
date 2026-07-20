Comparison Tests and Validation Metrics
=======================================

Group-comparison tests
-----------------------

The (weighted, optionally stratified) log-rank test for comparing survival
distributions across groups:

.. autofunction:: surpyval.univariate.nonparametric.logrank.logrank

Gray's test for comparing cumulative incidence functions across groups under
competing risks:

.. autofunction:: surpyval.univariate.competing_risks.nonparametric.gray_test.gray_test

Restricted mean survival time
-----------------------------

The two-group restricted-mean-survival-time difference (the per-model
``rmst`` method lives on the non-parametric model class):

.. autofunction:: surpyval.univariate.nonparametric.nonparametric.rmst_diff

Prediction-validation metrics
-----------------------------

Right-censored-standard metrics for scoring a predicted survival function
(Brier / integrated Brier score and Uno's time-dependent AUC), plus the helper
that builds a predicted-survival matrix from any fitted model exposing
``sf(x, Z)``.

.. automodule:: surpyval.metrics.validation
   :members:
