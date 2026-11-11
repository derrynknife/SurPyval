Multi-State Models
==================

.. note::

    Every example on this page is executed when the documentation is built, so
    the outputs shown are produced by the installed version of surpyval.

What a multi-state model is
---------------------------

Ordinary survival analysis follows each unit from one state ("alive",
"working") until a single event moves it into another ("dead", "failed").
Many histories have more than two states. A patient can fall ill, recover,
fall ill again and die; a pump can run, run degraded, be repaired and fail;
a borrower can pay, fall behind, catch up or default. A *multi-state model*
describes such a history as moves between a fixed set of *states*. A move
straight from one state into another is a *transition*; a state with no way
out (death, scrapped, default) is *absorbing*.

The questions it answers are the ones people ask about such histories:

- How likely is a unit to be in each state at a given time? These are the
  *state occupation probabilities*. In an illness-death model, they say
  what share of patients are healthy, ill and dead five years on.
- Starting from one state at time :math:`s`, how likely is a unit to be in
  each state at a later time :math:`t`? These are the *transition
  probabilities* :math:`P(s, t)`, a matrix with a row for each starting
  state. "Given that a pump is degraded today, how likely is it to have
  failed within a year?" is one of its elements.
- How do covariates change the hazard of each transition, and so the
  probabilities?

Competing risks are the simplest multi-state model (one starting state and
an absorbing state for each cause), and ordinary survival is simpler still
(two states, one transition). With a single transient state SurPyval's
multi-state estimate is exactly the competing-risks cumulative incidence of
:doc:`Competing Risks SurPyval Modelling`; with one transition, exactly one
minus the Kaplan-Meier estimate.

When to use one
---------------

Use a multi-state model when units pass through intermediate states that
matter: a disease stage, a degraded mode, a repair, a relapse. Ignoring the
intermediate state either loses information (fitting only the time to the
final state) or biases the answer (treating the move into the intermediate
state as censoring, which it is not).

The estimates on this page assume the process is *Markov*: where a unit goes
next, and how soon, depends on the state it is in and the time (on the
model's clock), not on how long it has been in that state or how it got
there. The state occupation probabilities from the starting distribution
remain valid estimates without that assumption (Datta and Satten, 2001),
but the transition probabilities from a later time :math:`s` need it. If
the time spent in the current state drives the hazard (a "clock-reset" or
semi-Markov process), these estimates do not describe it.

How the data are laid out
-------------------------

Each row is a *spell*: the time a unit spent in one state. A spell has

- ``state``: the state the unit was in;
- ``tl``: when the spell began (the unit's entry into the state, or into
  the study);
- ``x``: when it ended;
- ``e``: the state the unit moved into at ``x``, or missing (``None``)
  where the spell ended because observation stopped (censoring);
- optionally ``id``, the unit, and ``n``, a count of identical units.

A unit's spells follow one another: the second starts where the first
ended. A unit can also enter part-way through, in any state (delayed entry,
or left truncation): it counts as at risk only from its ``tl``.

This "long" form is the one R's ``survival`` package uses for multi-state
data. R's ``mstate`` uses a "stacked" form instead, with one row for every
transition a spell could have ended in; ``surpyval.multistate.to_stacked``
and ``from_stacked`` convert between the two.

How the estimate works, in words
--------------------------------

At each time a transition is seen, the Aalen-Johansen estimate counts how
many units were in each state just before, and how many of them made each
move. The share of the units in state :math:`g` that moved to :math:`h`,
:math:`d_{gh} / r_g`, is the estimated chance of that move at that time.
Chaining these chances from :math:`s` to :math:`t` (multiplying the
matrices of them) gives the transition probabilities:

.. math::
    P(s, t) = \prod_{s < t_j \le t} \left(I + dA(t_j)\right),

where :math:`dA(t_j)` holds the shares :math:`d_{gh} / r_g` off the diagonal
and minus their row sums on it. Censored units simply stop counting in
:math:`r` after they leave, as in the Kaplan-Meier estimate.

The uncertainty comes in three versions, all standard: the Greenwood-type
and Aalen-type variances R's ``mstate`` computes, which build up the
variance of :math:`P` step by step from the variance of each step's shares,
and the infinitesimal jackknife (R ``survival``'s ``survfit``), which adds
up how much each unit moves the estimate. They agree closely on data of a
reasonable size.

A worked example: progression of MGUS
-------------------------------------

The MGUS data (Kyle et al., 2002; R's ``survival::mgus2``) follow 1,384
patients with monoclonal gammopathy of undetermined significance, a
condition that can progress to a plasma-cell malignancy (PCM). Patients
die with or without progressing: an illness-death model with the states
``mgus``, ``pcm`` and ``death``. Times are in months.

.. jupyter-execute::

    import numpy as np
    import surpyval as surv
    from surpyval.datasets import load_mgus2
    from surpyval.multistate import (
        MultiState,
        MultiStateProportionalHazards,
        TransitionMatrix,
    )

    spells = load_mgus2(spells=True)
    spells.head()

The patient with ``id`` 1 died at 30 months without progressing, so has one
spell. A patient who progressed has two: one in ``mgus`` ending in ``pcm``,
and one in ``pcm`` ending in ``death`` or censored.

First the states and the moves between them:

.. jupyter-execute::

    tm = TransitionMatrix.illness_death(("mgus", "pcm", "death"))
    tm

Then the Aalen-Johansen estimate:

.. jupyter-execute::

    model = MultiState.fit_from_df(
        spells, "x", "e", "state", tl_col="tl", id_col="id", transitions=tm
    )
    model

The share of patients in each state 10 and 20 years (120 and 240 months)
after diagnosis:

.. jupyter-execute::

    for state in model.states:
        print(state, model.occupancy([120, 240], state).round(3))

About 1% of patients are alive with PCM at any time: progression is rare
and PCM is quickly fatal, while most deaths come without progression. The
occupation probabilities over time, stacked so that they add up to 1:

.. jupyter-execute::

    ax = model.plot()

Bounds on the probability of being alive with PCM, with each variance:

.. jupyter-execute::

    for variance in ("greenwood", "aalen", "robust"):
        print(variance, model.cb(240, "pcm", variance=variance).round(4))

The transition probabilities from a later time answer questions about
patients who have already reached a state. For a patient alive with PCM at
five years (60 months), the chance of being dead by ten years:

.. jupyter-execute::

    model.occupancy(120, "death", s=60, from_state="pcm").round(3)

Covariates: a Cox model for each transition
-------------------------------------------

``MultiStateProportionalHazards`` fits a proportional-hazards model to each
transition: each transition has its own baseline hazard and, by default,
its own coefficients. It is fitted as one Cox model, stratified by
transition, to the stacked data, which is how R's ``mstate`` does it. Here,
the effect of sex and of age (in decades from 70) on each transition:

.. jupyter-execute::

    spells["age70"] = (spells["age"] - 70) / 10
    cox = MultiStateProportionalHazards.fit_from_df(
        spells, "x", ["age70", "male"], "e", "state", tl_col="tl",
        id_col="id", transitions=tm,
    )
    cox.summary()[["coef", "exp(coef)", "se(coef)", "p"]].round(3)

Each decade of age raises the hazard of dying without PCM by about 90% and
of dying with it by about 50%, but the hazard of progressing by only about
14%, an effect within its uncertainty; men die sooner without PCM. Where an effect is thought to be the
same on every transition, ``shared`` gives it one coefficient:

.. jupyter-execute::

    shared = MultiStateProportionalHazards.fit_from_df(
        spells, "x", ["age70", "male"], "e", "state", tl_col="tl",
        id_col="id", transitions=tm, shared=["male"],
    )
    shared.parameter_names

The predictions combine the transitions for one covariate row. For a man of
75 and a woman of 65 at diagnosis, the probability of being alive with PCM
and of being dead ten years on, with bounds that include the uncertainty of
the coefficients:

.. jupyter-execute::

    for z in ([0.5, 1.0], [-0.5, 0.0]):
        print(
            z,
            cox.occupancy(120, z, "pcm", from_state="mgus").round(4),
            cox.occupancy(120, z, "death", from_state="mgus").round(3),
            cox.cb(120, z, "death", from_state="mgus").round(3),
        )

The older man is less likely to be alive with PCM, because he is far more
likely to have died first: the occupation probabilities show the
transitions acting together, which their hazards alone do not.

Where few units are left at risk the predicted hazards of leaving a state
can add up to more than 1 at one time, and the predicted probabilities can
then leave [0, 1]; the model warns when a prediction reaches such a time.

How it was checked
------------------

The estimates and every variance agree with R to rounding: the
occupation probabilities and their jackknife standard errors with
``survival``'s ``survfit`` (from the start, and from a later time, with
delayed entry and tied times); the transition probabilities and their
Greenwood-type and Aalen-type standard errors with ``mstate``'s
``probtrans``; and the Cox models' coefficients and predictions, with their
standard errors, with ``mstate``'s ``msfit`` and ``probtrans`` (Breslow's
and Efron's ties, transition-specific and shared effects). With one
transient state the estimate is SurPyval's competing-risks cumulative
incidence, and with one transition its Kaplan-Meier estimate with
Greenwood's variance. Simulated Markov models are recovered within their
bounds. The script that produced R's numbers is
``scripts/reference/reference_r_multistate.R``.

What is not here yet
--------------------

Parametric models of each transition, semi-Markov (clock-reset) models,
and states seen only at inspection times (panel data) are not built. A
fit refuses interval-censored changes of state.

References
----------

Aalen, O. O. and Johansen, S. (1978), "An empirical transition matrix for
non-homogeneous Markov chains based on censored observations",
*Scandinavian Journal of Statistics*, 5(3), 141-150.

Andersen, P. K. and Keiding, N. (2002), "Multi-state models for event
history analysis", *Statistical Methods in Medical Research*, 11(2), 91-115.

Datta, S. and Satten, G. A. (2001), "Validity of the Aalen-Johansen
estimators of stage occupation probabilities and Nelson-Aalen estimators of
integrated transition hazards for non-Markov models", *Statistics &
Probability Letters*, 55(4), 403-411.

de Wreede, L. C., Fiocco, M. and Putter, H. (2010), "The mstate package for
estimation and prediction in non- and semi-parametric multi-state and
competing risks models", *Computer Methods and Programs in Biomedicine*,
99(3), 261-274.

Kyle, R. A., Therneau, T. M., Rajkumar, S. V., et al. (2002), "A long-term
study of prognosis in monoclonal gammopathy of undetermined
significance", *New England Journal of Medicine*, 346(8), 564-569.

Putter, H., Fiocco, M. and Geskus, R. B. (2007), "Tutorial in
biostatistics: competing risks and multi-state models", *Statistics in
Medicine*, 26(11), 2389-2430.

API
---

Import these from ``surpyval.multistate`` (``MultiState``,
``MultiStateProportionalHazards`` and ``TransitionMatrix`` are also at the
top level, ``surpyval.MultiState`` and so on).

.. autoclass:: surpyval.multistate.TransitionMatrix
   :members:

.. autoclass:: surpyval.multistate.MultiState
   :members:

.. autoclass:: surpyval.multistate.MultiStateProportionalHazards
   :members:

.. autofunction:: surpyval.multistate.to_stacked

.. autofunction:: surpyval.multistate.from_stacked
