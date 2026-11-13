Cause-Specific Proportional Intensity
=====================================

Proportional-intensity regression for a recurrent process with several
kinds of event (marks ``e``): every event type has its own baseline
intensity (Duane by default, or Crow-AMSAA, Cox-Lewis or the constant-rate
HPP via ``baseline``) and its own covariate coefficients, so a covariate can
raise one type's rate and leave another's alone. It is the regression
counterpart of :doc:`CauseSpecificNHPP <cause_specific_nhpp>`. Import it from
``surpyval.recurrent``; ``fit`` returns a fitted instance, whose per-type
models, each a
:doc:`ProportionalIntensityModel <proportional_intensity_models>`, are in
``models[event]``. The worked example is "Several failure modes, each with
its own covariate effects" in
:doc:`../Recurrent Event Regression Modelling with SurPyval`.

.. autoclass:: surpyval.recurrent.competing_risks.regression.cause_specific_proportional_intensity.CauseSpecificProportionalIntensity
   :members:
