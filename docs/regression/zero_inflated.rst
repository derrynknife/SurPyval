Zero-Inflated Regression
========================

A share of the units fails at time 0 (dead on arrival), and covariates
act both on that share and on the lifetime of the rest:
:math:`F(t \mid z) = f_0(z) + (1 - f_0(z)) F_u(t \mid z)`. The model is
the zero-inflated fit (``zi=True``) with covariates, the mirror image of
the :doc:`mixture-cure regression <mixture_cure>`, and a member of the
:doc:`distributional regressions <distributional>`, whose fits, bounds
and saved form it shares. Where every row says whether its unit was
dead on arrival it is fitted in two stages (a binary regression, then
the latency's regression on the units that worked), which is the joint
maximum; otherwise both parts are fitted together. The guide, with
worked examples, is :doc:`../Zero-Inflated Regression`.

Pre-built instances: ``WeibullZeroInflated``, ``LogNormalZeroInflated``,
``LogLogisticZeroInflated``, ``GammaZeroInflated`` and
``ExponentialZeroInflated``.

.. autofunction:: surpyval.univariate.regression.distributional.zero_inflated.ZeroInflated

.. autoclass:: surpyval.univariate.regression.distributional.zero_inflated.ZeroInflatedFitter
    :members: fit, fit_from_df, fit_penalised, random

.. autoclass:: surpyval.univariate.regression.distributional.zero_inflated.ZeroInflatedModel
    :members: doa_probability, sf, ff, df, hf, Hf, qf, mean, distribution_at, parameters_at, parameter_cb, to_dict
