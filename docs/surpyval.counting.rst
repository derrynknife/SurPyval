Recurrent Event Models
======================

Models for items that can have the same event many times -- repairable
systems that fail and are repaired, patients with repeated
hospitalisations. The data are each item's event times, measured from
the start of its life, with a right-censored (``c=1``) row marking the
end of its observation. Import everything on these pages from
``surpyval.recurrent``. The theory is in
:doc:`Recurrent Event Analysis` and
:doc:`Recurrent Event Regression Analysis`; worked examples are in
:doc:`Recurrent Event Modelling with SurPyval` and
:doc:`Recurrent Event Regression Modelling with SurPyval`.

Each fitter below returns one of three fitted-model classes: a
:doc:`ParametricRecurrenceModel <counting/parametric_recurrence_model>`
(the Poisson processes), a :doc:`RenewalModel <counting/renewal_model>`
(the renewal and imperfect-repair models) or a
:doc:`ProportionalIntensityModel <counting/proportional_intensity_models>`
(the regression models). The non-parametric and cause-specific fitters
return a fitted instance of their own class.

Recurrent Event (Poisson Process) Models
-----------------------------------------

Processes whose events arrive at a rate that depends only on the
system's age (or not at all, for the HPP), so a repair leaves the
system as it was just before the failure ("as bad as old").

.. toctree::
    :maxdepth: 1

    counting/parametric_recurrence_model
    counting/hpp
    counting/duane
    counting/cox_lewis
    counting/crow_amsaa

Non-Parametric Models
---------------------

The mean cumulative function estimated without assuming a process.

.. toctree::
    :maxdepth: 1

    counting/nonparametric_mcf

Renewal Models
--------------

Processes in which a repair restores the system part of the way to new
(or all the way, for an ordinary renewal process).

.. toctree::
    :maxdepth: 1

    counting/renewal_model
    counting/grp
    counting/g1_rp
    counting/ara
    counting/ari

Competing Risks (Marked) Models
-------------------------------

Recurrent processes with several kinds of event, each with its own
intensity.

.. toctree::
    :maxdepth: 1

    counting/cause_specific_mcf
    counting/cause_specific_nhpp

Recurrent Event Regression Models
----------------------------------

Intensities that depend on covariates.

.. toctree::
    :maxdepth: 1

    counting/proportional_intensity_models
    counting/hpp_pi
    counting/nhpp_pi

Trend Tests and Diagnostics
---------------------------

Tests of whether the event rate is changing, and the result classes of
the tests and of the models' goodness-of-fit methods.

.. toctree::
    :maxdepth: 1

    counting/trend_tests
