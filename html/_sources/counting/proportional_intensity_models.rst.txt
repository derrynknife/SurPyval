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

For covariates that change over time, ``cif_tvc(x, Z)``, ``iif_tvc(x, Z)``
and ``cb_tvc(x, Z)`` predict along a step schedule (a
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`), with
the names and arguments of the life regressions' ``sf_tvc`` and
``cb_tvc``. A model fitted to such covariates (``fit_tvc``) has ``is_tvc``
set, and the methods that read one covariate row per fitted item
(``residuals``, ``cramer_von_mises``, ``trend_test``, ``diagnose``,
``plot``) refuse for it.

.. autoclass:: surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel
   :members:
   :inherited-members:
