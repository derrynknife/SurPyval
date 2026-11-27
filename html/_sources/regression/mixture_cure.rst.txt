Mixture-Cure Regression
=======================

A fraction of the units never fails, and covariates act both on that
fraction and on the lifetime of the rest:
:math:`S(t \mid z) = 1 - \pi(z) + \pi(z) S_u(t \mid z)`. The model is the
limited-failure fit (``lfp=True``) with covariates, and a member of the
:doc:`distributional regressions <distributional>`, whose fits, bounds
and saved form it shares. The guide, with worked examples, is
:doc:`../Mixture-Cure Regression`.

Pre-built instances: ``WeibullMixtureCure``, ``LogNormalMixtureCure``,
``LogLogisticMixtureCure``, ``GammaMixtureCure`` and
``ExponentialMixtureCure``.

.. autofunction:: surpyval.univariate.regression.distributional.mixture_cure.MixtureCure

.. autoclass:: surpyval.univariate.regression.distributional.mixture_cure.MixtureCureFitter
    :members: fit, fit_from_df, random

.. autoclass:: surpyval.univariate.regression.distributional.mixture_cure.MixtureCureModel
    :members: susceptible_probability, cure_probability, cure_given_survival, sf, ff, df, hf, Hf, qf, mean, follow_up, distribution_at, parameters_at, parameter_cb, to_dict
