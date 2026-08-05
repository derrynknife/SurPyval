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

Stochastic Process Models
-------------------------

Where a path model treats degradation as a deterministic curve with
noise, these treat it as a stochastic process in its own right: the
Wiener process for degradation that can go down as well as up, and the
gamma process for monotone accumulation such as wear or crack growth.
Both give a first-passage distribution to the threshold in closed form,
and so a remaining-useful-life prediction with bounds.

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
