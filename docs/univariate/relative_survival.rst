Relative Survival
=================

Net survival and excess hazards beyond a population's own rates: the
population's hazards in a :class:`~surpyval.RateTable`, net survival by
Pohar Perme's estimator or Ederer II (:class:`~surpyval.RelativeSurvival`)
and excess hazard regression (:class:`~surpyval.ExcessHazard`). The
guide, with worked examples, is :doc:`../Relative Survival`.

.. autoclass:: surpyval.RateTable
   :members: hf, Hf, sf, by_year, stratum_index, from_df, to_dict, from_dict

.. autoclass:: surpyval.RelativeSurvival
   :members: fit, fit_from_df, sf, ff, Hf, cb, standard_errors, table, to_dict, from_dict

.. autoclass:: surpyval.ExcessHazard
   :members: fit, fit_from_df, parameter_names, intervals, n_intervals, rates, coef, sf, ff, hf, Hf, df, qf, random, observed_sf, cb, param_cb, covariance, standard_errors, summary, to_dict, from_dict
