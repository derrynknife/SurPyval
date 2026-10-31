Plotting Non-Parametric Estimates
=================================

One figure of one or more non-parametric estimates, with their bounds,
censoring ticks, quantile lines and a number-at-risk table under the x
axis. Import it as ``surpyval.plot_nonparametric``; worked examples are
in :doc:`../Non-Parametric SurPyval Modelling`.

.. autofunction:: surpyval.plot_nonparametric

.. autoclass:: surpyval.NonParametricPlot

The table's counts without the plot (no plotting library is loaded):

.. autofunction:: surpyval.at_risk_table

The actuarial life table of the data, by interval:

.. autofunction:: surpyval.life_table
