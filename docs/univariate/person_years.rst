Exposure (Person-Years) Tables
==============================

The total time at risk, the events and the units in each cell of a
cross-classification by fixed factors and by time bands the units move
through (attained age, calendar period, time since entry), with each
cell's event rate and its exact bounds, rate ratios between cells, and
observed to expected ratios given reference rates. Worked examples are
in :doc:`../Non-Parametric SurPyval Modelling`.

.. autofunction:: surpyval.person_years

.. autoclass:: surpyval.PersonYears
   :members: margins, rate_ratio, to_dict, from_dict, dimensions, has_expected

The exact (Garwood) bounds of a single rate:

.. autofunction:: surpyval.poisson_rate_bounds
