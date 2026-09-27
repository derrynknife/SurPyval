HPP Regression
==============

Proportional-intensity HPP regression: a constant baseline rate scaled by
the covariate factor ``exp(Z @ beta)``. The fit returns a
:doc:`ProportionalIntensityModel <proportional_intensity_models>`, which
carries the prediction methods (``cif``, ``iif``, ``inv_cif``, each
taking the covariates alongside the time).

.. autodata:: surpyval.recurrent.regression.hpp_proportional_intensity.ProportionalIntensityHPP
   :no-value:

   .. automethod:: surpyval.recurrent.regression.hpp_proportional_intensity.ProportionalIntensityHPP.fit
   .. automethod:: surpyval.recurrent.regression.hpp_proportional_intensity.ProportionalIntensityHPP.fit_from_recurrent_data
