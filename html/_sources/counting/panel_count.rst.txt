Panel Counts (Events Counted at Inspections)
============================================

Recurrent events seen only as counts: each unit is inspected at its own
times and the number of events since the previous inspection is recorded,
never the event times. ``PanelCountMCF`` estimates the mean cumulative
function without assuming a process (maximum likelihood or maximum
pseudo-likelihood over non-decreasing step functions), and
``PanelCountRegression`` fits the proportional means model
:math:`E[N(t) \mid Z] = \Lambda_0(t) e^{\beta' Z}` with the baseline left
unspecified. Bounds and standard errors come from resampling units (the
bootstrap). The theory is in :doc:`/Recurrent Event Analysis` and worked
examples in :doc:`/Recurrent Event Modelling with SurPyval`. Each
``fit`` returns a fitted instance of its class.

The parametric intensity models (``CrowAMSAA``, ``CoxLewis``, ...) take
panel counts as interval-counted rows (``c=2``, ``x`` the ``[previous,
this]`` inspection times, ``n`` the count, 0 allowed).

.. autoclass:: surpyval.recurrent.panel_count.PanelCountMCF
   :members:
   :inherited-members:

.. autoclass:: surpyval.recurrent.panel_count.PanelCountRegression
   :members:
   :inherited-members:

.. autoclass:: surpyval.recurrent.panel_count.PanelCountConvergenceWarning
