Non-Parametric Counting (MCF)
=============================

The non-parametric mean cumulative function (MCF) estimator for
recurrent events: the expected number of events per item by time
:math:`t`, estimated without assuming a process. It supports exact
events, right-censored end-of-observation rows, left truncation (delayed
entry) and gapped multi-window observation, and gives confidence bounds
from the Lawless-Nadeau robust variance via ``mcf_cb``.
``NonParametricCounting.fit`` returns a fitted instance of the class
below.

.. autodata:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting
   :no-value:

   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.fit
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.from_xrd
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.mcf
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.mcf_cb
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.plot
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.to_dict
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.from_dict
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.to_json
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting.from_json
