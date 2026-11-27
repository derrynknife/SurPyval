Distributional Regression
=========================

Every parameter of the distribution has its own covariates and link:
:math:`g_k(\theta_k(z)) = g_k(\theta_k) + s_k x_k(z)' b_k`, with SurPyval's
sign :math:`s_k` (a positive coefficient means a shorter life, as in the
AFT models: :math:`s_k = -1` on a scale or a location). The guide, with a
worked example and the rule for each kind of parameter, is
:doc:`../Distributional Regression`.

Pre-built instances: ``WeibullDistributional``,
``ExponentialDistributional``, ``GammaDistributional``,
``LogNormalDistributional``, ``LogLogisticDistributional``,
``NormalDistributional``, ``GumbelDistributional`` and
``LogisticDistributional``.

.. autofunction:: surpyval.univariate.regression.distributional.fitter.Distributional

.. autoclass:: surpyval.univariate.regression.distributional.fitter.DistributionalFitter
    :members: fit, fit_from_df, sf, ff, df, hf, Hf, random

.. autoclass:: surpyval.univariate.regression.distributional.model.DistributionalRegressionModel
    :members: coefficient_signs, parameters_at, parameter_cb, plot_parameter, distribution_at, qf, mean, likelihood_ratio_test, covariance, aliased, phi, to_dict, from_dict
