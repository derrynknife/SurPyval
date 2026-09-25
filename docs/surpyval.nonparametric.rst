Non-Parametric
==============

Estimators that make no assumption about the shape of the distribution:
the survival curve is read straight from the data. Each estimator is
exported as a ready-made instance -- ``KaplanMeier``, ``NelsonAalen``,
``FlemingHarrington`` and ``Turnbull`` -- whose ``fit(x, c, n, t)``
returns a :doc:`NonParametric <univariate/non_parametric_class>` model.
That model carries the survival, hazard and quantile functions,
confidence bounds and bands, the restricted mean and plotting.

The theory is in :doc:`Non-Parametric Estimation` and worked examples
are in :doc:`Non-Parametric SurPyval Modelling`. The log-rank test and
the two-group restricted-mean difference, which compare non-parametric
estimates between groups, are documented in
:doc:`comparison_and_validation`.

Non-Parametric Class
--------------------

The fitted model every estimator returns.

.. toctree::
   :maxdepth: 1

   univariate/non_parametric_class


Non-Parametric Estimators
-------------------------

Kaplan-Meier, Nelson-Aalen and Fleming-Harrington handle observed,
right-censored and truncated data, and raise a ``ValueError`` on left-
or interval-censored data; Turnbull handles every combination of
censoring and truncation.

.. toctree::
   :maxdepth: 1

   univariate/kaplan_meier
   univariate/nelson_aalen
   univariate/fleming_harrington
   univariate/turnbull

Other Non-Parametric Functions
------------------------------

Zero-failure (success-run) testing, and the plotting positions used by
probability plots and by the ``MPP`` parametric fitting method.

.. toctree::
   :maxdepth: 1

   univariate/success_run
   univariate/plotting_positions
