Cause-Specific MCF
==================

The non-parametric mean cumulative function of each event type in a
recurrent process with several kinds of event (marks ``e``): one
:doc:`NonParametricCounting <nonparametric_mcf>` estimate per cause,
sharing the at-risk set. Import it from ``surpyval.recurrent``; ``fit``
returns a fitted instance, whose per-cause models are in
``models[cause]``.

.. autoclass:: surpyval.recurrent.competing_risks.nonparametric.cause_specific_mcf.CauseSpecificMCF
   :members:
