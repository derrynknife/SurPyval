Regression Modelling
=====================

Models in which covariates change the survival of an individual unit.
The families differ in *how* a covariate acts: multiplying the hazard,
adding to it, scaling time, or shifting the odds.

For a narrative introduction with worked examples, see
:doc:`Regression Modelling with SurPyval`.

Semi-Parametric Models
----------------------

No assumed baseline distribution: the shape of the baseline is left to
the data, and only the covariate effect is parameterised.

.. toctree::
    :maxdepth: 1

    regression/cox_ph
    regression/additive_hazards
    regression/buckley_james

Parametric Models
-----------------

A fitted baseline distribution combined with a covariate function. The
page below covers the proportional-hazards, accelerated-failure-time,
proportional-odds and accelerated-life families, together with the
time-varying covariate schedules the first three can be evaluated
along.

.. toctree::
    :maxdepth: 1

    regression/parametric

Correlated Observations
-----------------------

For data arriving in groups that share an unobserved effect.

.. toctree::
    :maxdepth: 1

    regression/frailty
