Equivalent Time Scales
======================

One life on two clocks, combined into one age :math:`T = x_1 + k\,x_2`
(equivalent hours: hours plus :math:`k` hours per start), with
:math:`k \ge 0` estimated by maximum likelihood together with the
distribution on :math:`T`. The likelihood is that of the first clock,
with the clock-rate factor :math:`1 + k\rho_i` on each failure. The
guide, with a worked example, is :doc:`../Equivalent Time Scales`.

Pre-built instances: ``WeibullEquivalence``, ``LogNormalEquivalence``,
``LogLogisticEquivalence``, ``GammaEquivalence`` and
``ExponentialEquivalence``.

.. automodule:: surpyval.univariate.equivalence
    :exclude-members: Equivalence, EquivalenceFitter, EquivalenceModel

.. autofunction:: surpyval.univariate.equivalence.Equivalence

.. autoclass:: surpyval.univariate.equivalence.EquivalenceFitter
    :members: fit, fit_from_df, from_params

.. autoclass:: surpyval.univariate.equivalence.EquivalenceModel
    :members: k, dist_params, equivalent, sf, ff, df, hf, Hf, qf, mean, sf_at_usage, ff_at_usage, hf_at_usage, qf_at_usage, mean_at_usage, cb, cb_at_usage, quantile_cb_at_usage, param_cb, k_lr_test, covariance, standard_errors, random, reading, to_dict, from_dict
