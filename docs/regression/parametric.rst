Parametric Proportional Hazards
===============================

Batteries Included
------------------

A proportional hazard model is on in which the hazard rate is a function of
the covariates and time. Very generally, the hazard rate for a proportional
hazard model is:

.. math::
    h(x | Z) = h(x) \phi(Z)

Where :math:`h(t)` is the hazard rate of the underlying distribution and 
:math:`\phi` is the function of the covariates.

SurPyval allows you to use very general models for fitting, or to create much
more specific models for your use case. For the general models SurPyval has 
the following options immediately available:

    - ExponentialPH
    - WeibullPH


These two are similar to the Cox Proportional Hazards model in that that have
a log-linear function of the covariates. The difference is that the baseline
disstribution is estimated using either the Exponential or Weibull
distributions. The log-linear function is:

.. math::
    \phi(Z) = e^{\beta_1 z_1 + \beta_2 z_2 + ...}

The number of coefficients depends only on the number of covariates that you
pass to the ``fit`` call. The coefficients are named ``beta_0``, ``beta_1``, 
etc. You can pass any number of covariates to the function, and the model will
automatically create the correct number of coefficients.

Custom PH Models
----------------



.. autoclass:: surpyval.regression.proportional_hazards_fitter.ProportionalHazardsFitter
    :members: