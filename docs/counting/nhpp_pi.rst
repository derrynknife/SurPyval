NHPP Regression
===============

Proportional-intensity NHPP regression: a time-varying baseline (Duane by default) scaled by the covariate factor ``exp(Z @ beta)``.

Fitting
-------

.. class:: ProportionalIntensityNHPP

   .. automethod:: surpyval.recurrent.regression.nhpp_proportional_intensity.ProportionalIntensityNHPP.fit
   .. automethod:: surpyval.recurrent.regression.nhpp_proportional_intensity.ProportionalIntensityNHPP.fit_from_recurrent_data

Prediction
----------

Unlike the HPP case, the prediction methods are on the model the fit
returns rather than on the fitter -- ``cif``, ``iif`` and ``inv_cif``
take the covariates alongside the time. The full model API is under
:doc:`proportional_intensity_models`.

.. automethod:: surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel.cif
.. automethod:: surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel.iif
.. automethod:: surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel.inv_cif
