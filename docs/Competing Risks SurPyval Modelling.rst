
Competing Risks SurPyval Modelling
===================================

This page shows how to use SurPyval's competing risks classes. For the
theoretical background see :doc:`Competing Risks Analysis`.

.. note::

    The competing risks subsystem is under active development. The
    ``CompetingRisks`` example on this page is executed when the documentation
    is built; the ``FineGray`` and ``CompetingRisksProportionalHazards``
    snippets show the (working) regression API.

Standard imports used throughout this page:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt


Fitting a Competing Risks Model
-------------------------------

The ``CompetingRisks`` class estimates a non-parametric cumulative incidence
function (CIF) for each failure cause, using the Nelson-Aalen or
Kaplan-Meier estimate of the overall survival function. Pass the observed
times, a cause indicator, and optional censoring flags:

.. jupyter-execute::

    from surpyval.univariate.competing_risks import CompetingRisks
    from surpyval import Weibull, Exponential

    # Simulated data: two competing causes
    np.random.seed(42)
    t1 = Weibull.random(100, 50, 2.5)   # cause 1 latent times
    t2 = Exponential.random(100, 1./80)             # cause 2 latent times

    # Observed time is the minimum; cause is which latent time was smaller
    x = np.minimum(t1, t2)
    cause = (t2 < t1).astype(int)  # 0 = cause 1, 1 = cause 2
    c = np.zeros_like(x, dtype=int)  # all observed (no censoring)

    model = CompetingRisks.fit(
        x=x,
        c=c,
        e=cause,
    )
    print(model)

Once fitted, you can query the CIF for each cause:

.. jupyter-execute::

    t_plot = np.linspace(0, 150, 300)
    for k, label in enumerate(['Cause 1 (Weibull)', 'Cause 2 (Exponential)']):
        plt.plot(t_plot, model.cif(t_plot, event=k), label=label)
    plt.xlabel('Time')
    plt.ylabel('Cumulative Incidence')
    plt.legend()
    plt.title('Competing Risks CIF by Cause')


Fine-Gray Sub-distribution Hazards
----------------------------------

The Fine-Gray model estimates the effect of covariates directly on the
cumulative incidence function of a chosen cause, using an
inverse-probability-of-censoring-weighted subdistribution risk set. ``e`` is
the per-observation cause label (``None`` for a censored row) and ``cause``
selects the cause of interest:

.. code-block:: python

    from surpyval.univariate.competing_risks import FineGray

    # x times, Z covariates, e cause labels, c censoring flags
    model = FineGray.fit(x, Z, e, c=c, cause=0)
    print(model)                 # coefficients, standard errors, p-values
    model.cif(t, Z=[0.5, 1.2])   # cumulative incidence of cause 0

Cause-Specific Proportional Hazards
-----------------------------------

``CompetingRisksProportionalHazards`` fits a proportional-hazards model per
cause. With ``how="Cox"`` (the default) each cause is a Cox model with the
other causes treated as censored; ``how="Fine-Gray"`` fits the subdistribution
model above for every cause:

.. code-block:: python

    from surpyval.univariate.competing_risks import (
        CompetingRisksProportionalHazards,
    )

    model = CompetingRisksProportionalHazards.fit(x, Z, e, c=c, how="Cox")
    model.cif(t, Z=[0.5, 1.2], event=0)
