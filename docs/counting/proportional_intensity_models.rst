Proportional Intensity Regression Model
=======================================

The fitted model returned by the proportional-intensity regression
fitters, :doc:`ProportionalIntensityHPP <hpp_pi>` and
:doc:`ProportionalIntensityNHPP <nhpp_pi>`. Every prediction takes the
covariates alongside the time -- ``cif(x, Z)``, ``iif(x, Z)``,
``inv_cif(x, Z)`` -- because the intensity is the baseline scaled by
:math:`e^{\beta' Z}`. The fitted baseline parameters are in ``params``
and the covariate coefficients in ``coeffs``. Like the other fitted
recurrence models it provides confidence bounds, likelihood inference,
diagnostics, simulation (per covariate vector) and serialisation.

.. autoclass:: surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel
   :members:
   :inherited-members:
