
Competing Risks SurPyval Modelling
===================================

This page shows how to use SurPyval's competing risks classes. For the
theoretical background see :doc:`Competing Risks Analysis`.

.. warning::

    The competing risks subsystem is under active development. The
    ``CompetingRisks`` examples on this page are executed when the
    documentation is built and work as shown. The ``FineGray`` and
    ``CompetingRiskProportionalHazard`` fitters, however, currently crash on
    first call; their sections below show the intended API only. These
    issues are tracked in ``DEVELOPMENT.md``.

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

    from surpyval.competing_risks import CompetingRisks
    from surpyval import Weibull, Exponential

    # Simulated data: two competing causes
    np.random.seed(42)
    t1 = Weibull.random(100, alpha=50, beta=2.5)   # cause 1 latent times
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


Fine-Gray Sub-distribution Hazards (Planned)
---------------------------------------------

The Fine-Gray model estimates the effect of covariates directly on the
cumulative incidence function. The intended API mirrors the single-event
regression fitters:

.. code-block:: python

    # Planned API — not yet working (see DEVELOPMENT.md)
    from surpyval.competing_risks import FineGray

    model = FineGray.fit(x=x, c=c, Z=Z, cause=cause, cause_of_interest=0)
    print(model.summary())

Cause-Specific Proportional Hazards (Planned)
----------------------------------------------

Fit a separate proportional hazards model per cause, censoring all other
events:

.. code-block:: python

    # Planned API — not yet working (see DEVELOPMENT.md)
    from surpyval.competing_risks import CompetingRiskProportionalHazard

    model = CompetingRiskProportionalHazard.fit(x=x, c=c, Z=Z, cause=cause)
    print(model.summary())
