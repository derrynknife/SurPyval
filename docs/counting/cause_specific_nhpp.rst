Cause-Specific NHPP
===================

A parametric intensity per event type in a recurrent process with
several kinds of event (marks ``e``): one NHPP (Crow-AMSAA by default)
fitted per cause, sharing each item's observation window. The total
intensity is the sum of the cause-specific ones (``total_cif``). Import
it from ``surpyval.recurrent``; ``fit`` returns a fitted instance, whose
per-cause models, each a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>`, are in
``models[cause]``.

.. autoclass:: surpyval.recurrent.competing_risks.parametric.cause_specific_nhpp.CauseSpecificNHPP
   :members:
