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

Unlike the HPP case, the prediction methods are not on the fitter: the
fit returns a
:class:`~surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel`,
and it is that object which carries
:meth:`~surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel.cif`,
:meth:`~surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel.iif`
and
:meth:`~surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel.inv_cif`
-- each taking the covariates alongside the time. They are documented in
full, with the rest of the model API, under
:doc:`proportional_intensity_models`.
