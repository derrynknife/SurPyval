Bayesian Fitting
================

Bayesian fits of the parametric distributions: a prior (``surpyval.prior``)
times the likelihood of the data, by ``Distribution.fit_bayes`` and
``fit_bayes_from_df``. The result is a
:class:`~surpyval.bayes.BayesianParametric`. Worked examples, in plain
words first, are in :doc:`Bayesian Fitting with SurPyval`.

.. py:module:: surpyval.bayes

Fitting
-------

.. automethod:: surpyval.univariate.parametric.ParametricFitter.fit_bayes

.. automethod:: surpyval.univariate.parametric.ParametricFitter.fit_bayes_from_df

The model
---------

.. autoclass:: surpyval.bayes.BayesianParametric
   :members:

.. autoclass:: surpyval.bayes.Posterior
   :members:

Every family's fit
------------------

The regressions' and the other families' Bayesian fits share one layer:
a family gives a :class:`~surpyval.bayes.Target` (its likelihood, names,
ranges, units and simulation), and ``fit_target`` chooses the engine
(closed form where conjugate, the grid for one to three free parameters,
the Laplace approximation beyond) and returns a
:class:`~surpyval.bayes.BayesianResult`.

.. autoclass:: surpyval.bayes.BayesianResult
   :members:

.. autoclass:: surpyval.bayes.PredictiveCheck
   :members:

.. autoclass:: surpyval.bayes.BayesianDiagnosis
   :members:

.. autofunction:: surpyval.bayes.fit_target

.. autoclass:: surpyval.bayes.Target
   :members:

Recurrent-event processes
-------------------------

``CrowAMSAA.fit_bayes`` and ``HPP.fit_bayes``; see "Reliability growth
and repairable systems" in :doc:`Bayesian Fitting with SurPyval`. The
results, ``BayesianRecurrence``, ``BayesianProportionalIntensity`` and
``BayesianCauseSpecific``, are importable from ``surpyval.recurrent``
(for ``isinstance`` checks and loading saved fits); they are made by
``fit_bayes``, not built directly.

.. automethod:: surpyval.recurrent.parametric._bayes_fit.BayesFitMixin.fit_bayes

.. autoclass:: surpyval.recurrent.BayesianRecurrence
   :members:

.. autoclass:: surpyval.recurrent.bayes.CountPrediction
   :members:

Every NHPP intensity (``Duane``, ``CoxLewis``, the software reliability
growth models) has the same ``fit_bayes``. With covariates, and by
failure cause:

.. automethod:: surpyval.recurrent.ProportionalIntensityNHPP.fit_bayes

.. automethod:: surpyval.recurrent.ProportionalIntensityHPP.fit_bayes

.. autoclass:: surpyval.recurrent.BayesianProportionalIntensity
   :members:

.. automethod:: surpyval.recurrent.CauseSpecificNHPP.fit_bayes

.. autoclass:: surpyval.recurrent.BayesianCauseSpecific
   :members:

Comparing fits
--------------

.. autofunction:: surpyval.bayes.compare

.. autoclass:: surpyval.bayes.PredictiveCriterion
   :members:

.. autoclass:: surpyval.bayes.PredictiveComparison
   :members:

Priors
------

.. py:module:: surpyval.bayes.prior

.. autofunction:: surpyval.bayes.prior.around

.. autofunction:: surpyval.bayes.prior.between

.. autofunction:: surpyval.bayes.prior.uniform

.. autofunction:: surpyval.bayes.prior.flat

.. autofunction:: surpyval.bayes.prior.fixed

.. autofunction:: surpyval.bayes.prior.from_distribution

.. autofunction:: surpyval.bayes.prior.on_quantities

.. autofunction:: surpyval.bayes.prior.on_function

.. autofunction:: surpyval.bayes.prior.from_model

.. autofunction:: surpyval.bayes.prior.from_data

.. autofunction:: surpyval.bayes.prior.from_fit

.. autofunction:: surpyval.bayes.prior.from_params

.. autofunction:: surpyval.bayes.prior.check

.. autoclass:: surpyval.bayes.prior.MultivariateNormal

.. autoclass:: surpyval.bayes.prior.PriorSpec

The likelihood
--------------

.. autofunction:: surpyval.bayes.log_likelihood_fn

.. autoclass:: surpyval.bayes.LogLikelihood
