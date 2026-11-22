Testing Effort and Change Points
================================

Two ways to fit an intensity model to testing that was not steady. Each
wraps any of the intensity models (``GoelOkumoto``, ``YamadaDelayedS``,
``InflectionS``, ``MusaOkumoto``, ``CrowAMSAA``, ``Duane``, ``CoxLewis``,
``HPP``) and is itself a fitter, whose ``fit`` takes the model's data
and returns a
:doc:`ParametricRecurrenceModel <parametric_recurrence_model>` with
every answer in time.

``EffortClock(model, effort)`` runs the model on the testing effort
spent (tester hours, test cases run) in place of time,
:math:`\Lambda(t) = \Lambda_0(W(t))`. The effort is a table of amounts
per period (an ``EffortTable``, or the pair ``(ends, amounts)``) or a
``WeibullEffort`` curve.

``ChangePoint(model, tau=None)`` multiplies the model's rates by
``ratio``, a further parameter, from the time ``tau`` on. ``tau`` is
given, or estimated by profile likelihood from one system's failure
times; ``change_point_cb`` then bounds it and ``change_point_test``
tests for no change by parametric bootstrap.

See "When the testing was not steady" in
:doc:`../Software Reliability Growth`.

.. autoclass:: surpyval.recurrent.EffortClock

.. autoclass:: surpyval.recurrent.EffortTable
   :members:

.. autoclass:: surpyval.recurrent.WeibullEffort
   :members:

.. autoclass:: surpyval.recurrent.ChangePoint

.. autofunction:: surpyval.recurrent.change_point_cb

.. autofunction:: surpyval.recurrent.change_point_test
