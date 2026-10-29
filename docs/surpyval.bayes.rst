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

.. autofunction:: surpyval.bayes.prior.from_model

.. autofunction:: surpyval.bayes.prior.from_data

.. autofunction:: surpyval.bayes.prior.from_params

.. autofunction:: surpyval.bayes.prior.check

.. autoclass:: surpyval.bayes.prior.MultivariateNormal

.. autoclass:: surpyval.bayes.prior.PriorSpec

The likelihood
--------------

.. autofunction:: surpyval.bayes.log_likelihood_fn

.. autoclass:: surpyval.bayes.LogLikelihood
