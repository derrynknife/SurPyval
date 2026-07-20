Non-Parametric Counting (MCF)
=============================

The non-parametric mean cumulative function (MCF) estimator for recurrent
events. It supports exact events, right-censored end-of-observation rows, left
truncation (delayed entry) and gapped multi-window observation, and provides
Greenwood confidence bounds via ``mcf_cb``.

.. class:: NonParametricCounting

   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting_.fit
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting_.fit_from_recurrent_data
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting_.mcf
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting_.mcf_cb
   .. automethod:: surpyval.recurrent.nonparametric.mcf.NonParametricCounting_.plot
