Degradation Analysis
====================

Degradation Analysis Fitter
---------------------------

.. autoclass:: surpyval.degradation.degradation_analysis.DegradationAnalysis_
   :members:

Degradation Model
-----------------

.. autoclass:: surpyval.degradation.degradation_analysis.DegradationModel
   :members:

.. autoclass:: surpyval.degradation.degradation_analysis.RULPrediction
   :members:

.. autoclass:: surpyval.degradation.degradation_analysis.InducedFailureDistribution
   :members:

Path Models
-----------

.. autoclass:: surpyval.degradation.path_models.PathModel
   :members:

.. autoclass:: surpyval.degradation.path_models.LinearPath_
   :members:

.. autoclass:: surpyval.degradation.path_models.QuadraticPath_
   :members:

.. autoclass:: surpyval.degradation.path_models.ExponentialPath_
   :members:

.. autoclass:: surpyval.degradation.path_models.OffsetExponentialPath_
   :members:

.. autoclass:: surpyval.degradation.path_models.PowerPath_
   :members:

.. autoclass:: surpyval.degradation.path_models.LogarithmicPath_
   :members:

.. autoclass:: surpyval.degradation.path_models.LloydLipowPath_
   :members:

.. autoclass:: surpyval.degradation.path_models.GompertzPath_
   :members:

.. autoclass:: surpyval.degradation.path_models.MichaelisMentenPath_
   :members:

.. autofunction:: surpyval.degradation.path_models.get_path_model

Stress-Dependent Path Parameters
--------------------------------

For accelerated degradation tests whose *mechanism* depends on stress
(``links`` in :meth:`DegradationAnalysis.fit`): the path parameters are
modelled on a link scale, ``eta_i = D(z_i) gamma + u_i``, so a
log-linked rate with ``Z = 1/T`` follows the Arrhenius relationship.

.. autoclass:: surpyval.degradation.stress.LinkedPathModel
   :members:

.. autofunction:: surpyval.degradation.stress.stress_design

.. autofunction:: surpyval.degradation.stress.fixed_effect_names

Step-Stress: the Accelerated Clock
----------------------------------

For tests whose stress changes *during* a unit's test
(``acceleration="clock"`` in :meth:`DegradationAnalysis.fit`): stress
speeds up the clock of every unit's path, ``AF(z) = exp(gamma' (z -
stress_ref))``, and the path is the ordinary path model on the
reference-stress time the unit has aged. The fitted
:class:`~surpyval.degradation.degradation_analysis.DegradationModel` then
carries ``gamma`` and ``stress_ref``, its life methods take the stress as
one row or a :class:`~surpyval.StepSchedule`, and its trajectory methods
take the unit's stress history ``Z`` and a planned ``Z_future``. How the
stress coefficients are estimated:

.. automodule:: surpyval.degradation.step_stress

Stochastic Process Models
-------------------------

Where a path model treats degradation as a deterministic curve with
noise, these treat it as a stochastic process in its own right: the
Wiener process for degradation that can go down as well as up, and the
gamma process for monotone accumulation such as wear or crack growth.
Both give a first-passage distribution to the threshold in closed form,
and so a remaining-useful-life prediction with bounds. Both also take a
stress ``Z`` (one row per measurement), which may change between or
during units' tests: stress accelerates the process clock, and the life
under any stress profile stays in closed form.

.. autoclass:: surpyval.degradation.process_models.WienerProcess
   :members:

.. autoclass:: surpyval.degradation.process_models.WienerProcessModel
   :members:

.. autoclass:: surpyval.degradation.process_models.GammaProcess
   :members:

.. autoclass:: surpyval.degradation.process_models.GammaProcessModel
   :members:

.. autoclass:: surpyval.degradation.process_models.ProcessRUL
   :members:

Destructive Degradation
-----------------------

For tests that destroy the unit being measured, so each unit yields one
observation at one time rather than a path. The degradation
distribution at each time is modelled directly, and the failure
distribution follows from the threshold crossing.

.. autoclass:: surpyval.degradation.destructive.DestructiveDegradation_
   :members:

.. autoclass:: surpyval.degradation.destructive.DestructiveDegradationModel
   :members:
