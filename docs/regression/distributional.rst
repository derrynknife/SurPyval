Distributional Regression
=========================

Every parameter of the distribution has its own covariates and link:
:math:`g_k(\theta_k(z)) = g_k(\theta_k) + x_k(z)' b_k`. The guide, with a
worked example, is :doc:`../Distributional Regression`.

Pre-built instances: ``WeibullDistributional``,
``ExponentialDistributional``, ``GammaDistributional``,
``LogNormalDistributional``, ``LogLogisticDistributional``,
``NormalDistributional``, ``GumbelDistributional`` and
``LogisticDistributional``.

.. autofunction:: surpyval.univariate.regression.distributional.fitter.Distributional

.. autoclass:: surpyval.univariate.regression.distributional.fitter.DistributionalFitter
    :members: fit, fit_from_df, sf, ff, df, hf, Hf, random

.. autoclass:: surpyval.univariate.regression.distributional.model.DistributionalRegressionModel
    :members: parameters_at, distribution_at, qf, mean, covariance, aliased, phi, to_dict, from_dict
