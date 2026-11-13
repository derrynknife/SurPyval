Comparison Tests and Validation Metrics
=======================================

Tools that compare models or groups rather than fit one: hypothesis tests
of whether groups share a survival (or cumulative-incidence) curve, the
restricted-mean difference between two groups, automatic selection of the
best-fitting distribution, forecasts of a fleet's failures from its current
ages, and metrics that score how well a model's predicted survival matches
held-out data. All are importable directly
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

The log-rank tests of every pair of groups, with the p-values adjusted for
the number of pairs, and the adjustment on its own:

.. autofunction:: surpyval.univariate.nonparametric.logrank.pairwise_logrank

.. autoclass:: surpyval.univariate.nonparametric.logrank.PairwiseLogRankResult
   :members: to_dict

.. autofunction:: surpyval.univariate.nonparametric.logrank.adjust_p_values

The log-rank and Wilcoxon-type tests of interval-censored data (any mix of
exact, right, left and interval-censored rows), and the result they return:

.. autofunction:: surpyval.univariate.nonparametric.interval_logrank.interval_logrank

.. autoclass:: surpyval.univariate.nonparametric.interval_logrank.IntervalLogRankResult
   :members: to_dict

The max-combo test of two groups, the largest of several
Fleming-Harrington weighted log-rank statistics with a p-value that allows
for taking the largest (for hazards that are not proportional):

.. autofunction:: surpyval.univariate.nonparametric.logrank_maxcombo.maxcombo_logrank

.. autoclass:: surpyval.univariate.nonparametric.logrank_maxcombo.MaxComboResult
   :members: to_dict

The log-rank test for trend across ordered groups (one degree of freedom):

.. autofunction:: surpyval.univariate.nonparametric.logrank_trend.trend_logrank

.. autoclass:: surpyval.univariate.nonparametric.logrank_trend.TrendLogRankResult
   :members: to_dict

The log-rank test with an exact or Monte Carlo permutation p-value, for
small samples:

.. autofunction:: surpyval.univariate.nonparametric.logrank_permutation.permutation_logrank

.. autoclass:: surpyval.univariate.nonparametric.logrank_permutation.PermutationLogRankResult
   :members: to_dict

The maximally selected log-rank statistic: the cut-point of a continuous
covariate, with a p-value corrected for the search:

.. autofunction:: surpyval.univariate.nonparametric.maxstat.maxstat_logrank

.. autoclass:: surpyval.univariate.nonparametric.maxstat.MaxStatResult
   :members: to_dict

The test of two groups' survival at one time:

.. autofunction:: surpyval.univariate.nonparametric.survival_difference.sf_diff

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

Adjusted survival curves
------------------------

A comparison of groups that were not assigned at random, on one mix of
covariates: direct adjustment (standardisation) of a fitted regression
model, and the checks on a set of weights before a weighted comparison.
The guide is :doc:`Adjusted Survival Curves`.

.. autofunction:: surpyval.direct_adjustment

.. autoclass:: surpyval.AdjustedCurves
   :members: sf, ff, cb, sf_diff, rmst, rmst_diff, plot

.. autofunction:: surpyval.weight_diagnostics

Automatic distribution selection
--------------------------------

Fit every candidate continuous distribution and keep the one with the
best information criterion:

.. autofunction:: surpyval.fit_best.fit_best

Forecasting from the current state
----------------------------------

The expected failures of units in service at their current ages over a
horizon, from any univariate or regression model, with a prediction
interval for the count, and every failure of repairable units from a
recurrent-event model (a Poisson process, or a renewal model simulated from
each unit's own history) (worked examples in
:doc:`Parametric SurPyval Modelling`,
:doc:`Regression Modelling with SurPyval` and
:doc:`Recurrent Event Modelling with SurPyval`):

.. autofunction:: surpyval.forecasting.forecast

.. autoclass:: surpyval.forecasting.Forecast
   :members: unit_expected

Prediction-validation metrics
-----------------------------

Right-censored-standard metrics for scoring a predicted survival function
(Brier / integrated Brier score and Uno's time-dependent AUC), the same
metrics with standard errors, confidence intervals and paired tests between
models (``score_predictions``), all of them also for one cause with competing
risks (``e=`` and ``cause=``, scoring predicted cumulative incidences), plus
the helper that builds a predicted-survival matrix from a fitted regression
model or forest.

.. automodule:: surpyval.metrics.validation
   :members:

Harrell's concordance index of any risk score, with Therneau's treatment
of tied event times by default, as R and lifelines (every regression model
also has a ``concordance`` method that picks its family's score):

.. autofunction:: surpyval.metrics.concordance.concordance_index

Harrell's or Uno's C with its standard error and confidence interval, and
the comparison of two scores' C on the same subjects (see "How sure is C,
and Uno's version" on the :doc:`regression analysis` page):

.. autofunction:: surpyval.metrics.concordance.concordance_estimate

.. autoclass:: surpyval.metrics.concordance.ConcordanceResult

.. autofunction:: surpyval.metrics.concordance.compare_concordance

.. autoclass:: surpyval.metrics.concordance.ConcordanceComparison
