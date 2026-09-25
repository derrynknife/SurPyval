NHPP Regression
===============

Proportional-intensity NHPP regression: a time-varying baseline intensity
(Duane by default, or Crow-AMSAA or Cox-Lewis via ``dist``) scaled by the
covariate factor ``exp(Z @ beta)``. The fit returns a
:doc:`ProportionalIntensityModel <proportional_intensity_models>`, which
carries the prediction methods (``cif``, ``iif``, ``inv_cif``, each
taking the covariates alongside the time).

.. autodata:: surpyval.recurrent.regression.nhpp_proportional_intensity.ProportionalIntensityNHPP
   :no-value:

   .. automethod:: surpyval.recurrent.regression.nhpp_proportional_intensity.ProportionalIntensityNHPP.fit
   .. automethod:: surpyval.recurrent.regression.nhpp_proportional_intensity.ProportionalIntensityNHPP.fit_from_recurrent_data
