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
- optionally ``i``, the unit (``i_col`` for a DataFrame), and ``n``, a
  count of identical units.

A unit's spells follow one another: the second starts where the first
ended. A unit can also enter part-way through, in any state (delayed entry,
or left truncation): it counts as at risk only from its ``tl``.

This "long" form is the one R's ``survival`` package uses for multi-state
data. R's ``mstate`` uses a "stacked" form instead, with one row for every
transition a spell could have ended in; ``surpyval.multistate.to_stacked``
and ``from_stacked`` convert between the two.

.. _multistate-from-data:

From a SurpyvalData
-------------------

The spells can also be held in a :class:`~surpyval.SurpyvalData`, the
package's one container for data: ``s`` is the state during each row (the
spell), ``e`` the state moved into at ``x`` (blank where the unit was still
in ``s`` when last seen), ``i`` the unit and ``tl`` the entry. Every
multi-state model fits it directly, ``fit(data, transitions=...)``, with
the same result as from the arrays; the other data arguments are then left
out, and the regressions take the data's covariates ``Z`` with their
names. Five patients in an illness-death model:

.. jupyter-execute::

    import surpyval as sp
    from surpyval.multistate import MultiState, TransitionMatrix

    data = sp.SurpyvalData(
        x=[2, 6, 3, 8, 4, 7, 9],
        s=["healthy", "ill", "healthy", "ill", "healthy", "healthy",
           "healthy"],
        e=["ill", "dead", "ill", None, "dead", None, None],
        i=[1, 1, 2, 2, 3, 4, 5],
        tl=[0, 2, 0, 3, 0, 0, 0],
        unit="months",
    )
    model = MultiState.fit(data, transitions=TransitionMatrix.illness_death())
    model.occupancy(5, "ill")

Two in five patients are ill at 5 months. ``SurpyvalData.from_dataframe``
builds the same data from a table (``s_col``, ``e_col``, ``i_col``,
``tl_col``).

Data without ``s`` is the case of one starting state: every row starts
from it, so competing-risks data (the cause in ``e``) fits as it is. The
starting state is the one state of ``transitions`` that nothing moves
into, or ``"event-free"`` (the name ``TransitionMatrix.competing_risks``
gives it) where no ``transitions`` are given; where the transitions have
several such states, give ``s``. A row that failed with no state entered
(a masked cause), data with no ``e``, and data without ``s`` whose units
have several rows (stretches of a unit's life, or repeated events) are
refused, saying what fits them.

The terminal flag ``a`` of a ``SurpyvalData`` is not needed here: a
terminal event is the same idea as entering an absorbing state, which the
transitions already say, so the fits ignore it.

A row with ``c = 2`` is a move known only to lie between two times (see
:ref:`multistate-interval-moves`): ``ParametricMultiState`` and
``ParametricMultiStateRegression`` read its ``[xl, xr]`` as the two
columns of their ``x``, with the same fit as from the arrays; the
non-parametric and Cox models refuse it. A left-censored row (``c =
-1``, a move at some time before ``x``) is refused by every model: with
no last time the unit was seen in its state, it has no stay it ended;
give that time as the lower end of an interval. The models take neither
weights, right truncation, time-varying covariates nor measurements
(``m``); data holding any of them is refused, saying what to do. ``s`` is also another name for ``state`` in
``fit``, and ``s_col`` for ``state_col`` in ``fit_from_df``.
The panel models below take arrays of visits, not a ``SurpyvalData``:
their ``s`` is the state seen at a visit, not the state during a row, so
``fit(data)`` is refused, saying so.

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
        spells, "x", "e", "state", tl_col="tl", i_col="id", transitions=tm
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
        i_col="id", transitions=tm,
    )
    cox.summary()[["coef", "exp(coef)", "se(coef)", "p"]].round(3)

Each decade of age raises the hazard of dying without PCM by about 90% and
of dying with it by about 50%, but the hazard of progressing by only about
14%, an effect within its uncertainty; men die sooner without PCM. Where an effect is thought to be the
same on every transition, ``shared`` gives it one coefficient:

.. jupyter-execute::

    shared = MultiStateProportionalHazards.fit_from_df(
        spells, "x", ["age70", "male"], "e", "state", tl_col="tl",
        i_col="id", transitions=tm, shared=["male"],
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

Effects shared by some transitions, and proportional baselines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An effect can also be shared by *some* transitions only. Here sex is
thought to act the same way on the two ways of dying (with or without
PCM), but not necessarily on progression: ``shared`` takes a dictionary
from the covariate to the transitions that share its coefficient, each a
``(from, to)`` pair (or its name, ``"mgus -> death"``). The other
transitions keep their own coefficient.

.. jupyter-execute::

    into_death = [("mgus", "death"), ("pcm", "death")]
    grouped = MultiStateProportionalHazards.fit_from_df(
        spells, "x", ["age70", "male"], "e", "state", tl_col="tl",
        i_col="id", transitions=tm, shared={"male": into_death},
    )
    grouped.summary()[["coef", "se(coef)", "p"]].round(3)

A covariate can have several such groups (a list of lists of
transitions). Sharing is a constraint, so it is worth checking against
the model without it: twice the difference in log partial likelihood is
a likelihood-ratio statistic, here on one degree of freedom.

.. jupyter-execute::

    2 * (cox.log_likelihood - grouped.log_likelihood)

At about 2.2 (a p-value of about 0.14) the data do not argue against one
effect of sex on both ways of dying.

Two transitions can also be thought to have the *same shape* of hazard
over time, one a constant multiple of the other. For death with and
without PCM, for example, PCM may simply multiply the hazard of dying.
``proportional_baselines`` gives the transitions listed one baseline
hazard, the first transition's, and estimates the log of each other's
ratio to it as one more coefficient. Its exponential is the ratio of the
two hazards for units with the same covariates:

.. jupyter-execute::

    prop = MultiStateProportionalHazards.fit_from_df(
        spells, "x", ["age70", "male"], "e", "state", tl_col="tl",
        i_col="id", transitions=tm, shared={"male": into_death},
        proportional_baselines=into_death,
    )
    prop.summary()[["coef", "exp(coef)", "se(coef)"]].round(3).tail(1)

The hazard of dying is about five and a half times as high with PCM as
without it, at the same time since diagnosis, age and sex. Pooling the two baselines
borrows strength from the larger transition (there are far more deaths
without PCM), which helps where one transition has few events; the
price is the assumption, which a plot of the two cumulative baseline
hazards (``cumulative_hazard`` at zero covariates; their ratio should be
roughly constant in the separate model) can check.

This is R ``mstate``'s general design (Putter, Fiocco and Geskus, 2007):
one Cox model of the stacked data with a stratum for each transition, or
for each group with proportional baselines (with the indicator of each
transition but the first as a covariate), and each covariate expanded
into a column for each transition or group of transitions. The
predictions and their standard errors account for the covariance of the
shared coefficients across the transitions, as ``msfit`` and
``probtrans`` do. One difference from ``mstate``: two transitions that
share a baseline have the covariance :math:`\phi_h \phi_g \,
\widehat{\mathrm{var}}\, d\hat{\Lambda}_0` of their predicted
increments (:math:`\phi` each one's relative risk), whereas ``msfit``
uses :math:`\phi_h^2` (``h`` the lower-numbered transition), which
depends on the order the transitions are listed in. The estimates are the
same; the standard errors of the predictions differ (by about 7% to 13%
in the checks below).

Parametric models: a distribution for each transition
-----------------------------------------------------

The Aalen-Johansen estimate and the Cox models make no assumption about
the shape of each transition's hazard over time, and so they stop at the
last observed time and move in steps. A *parametric* multi-state model
instead gives each transition a distribution (a Weibull, say) on the
time since the start, fitted to the spells that could have ended in it.
Choose it when you need to

- **extrapolate**: predict past the last observation (the share of
  patients dead at 50 years, the share of pumps failed over a design
  life longer than the trial), as far as the distributions can be
  trusted there;
- **draw smooth curves**, which do not jump at each observed event, or
  summarise each transition in a few numbers (a shape and a scale);
- work with **few events**, where a step estimate is ragged.

The price is the assumption: each transition's hazard must follow its
distribution. Compare the fit with the Aalen-Johansen estimate where the
data are, as below, and with other distributions (``dist`` takes one for
every transition or a dictionary with one for each).

Each transition is fitted on its own: its events are the spells that
ended in it; the spells in the same state that ended in another
transition, or were censored, are censored; and a spell that began at
``tl`` (an entry into the state, or a late entry into the study) is left
truncated there. That is the maximum-likelihood fit of the whole model,
because its likelihood is a product of one factor per transition.

The probabilities then come from the *Kolmogorov forward equations*, which
say how :math:`P(s, t)` changes as :math:`t` moves on:
:math:`\partial P(s, t) / \partial t = P(s, t)\,Q(t)`, where
:math:`Q(t)` holds the transition hazards. SurPyval solves them by
chaining many short steps, each the exact answer for hazards that are
nearly constant over it (the matrix exponential, with a fourth-order
correction for hazards that change), on a grid that is refined until each
row of :math:`P(s, t)` is accurate to about :math:`10^{-8}`. With
exponential transitions (constant hazards) the answer is the matrix
exponential :math:`\exp\{Q\,(t - s)\}` exactly.

.. jupyter-execute::

    from surpyval.multistate import ParametricMultiState

    weibull = ParametricMultiState.fit_from_df(
        spells, "x", "e", "state", tl_col="tl", i_col="id", transitions=tm
    )
    weibull

The Weibull and Aalen-Johansen occupation probabilities agree where the
data are, and the Weibull model goes on past the last observation
(about 35 years):

.. jupyter-execute::

    times = [120, 240, 360]
    for state in weibull.states:
        print(
            state,
            weibull.occupancy(times, state).round(3),
            model.occupancy(times, state).round(3),
        )
    print("death at 50 years:", weibull.occupancy(600, "death").round(3))

Bounds come from the delta method over every transition's parameters
(their covariance is block diagonal, since each transition is fitted on
its own), on the log(-log) scale by default so that they stay in (0, 1):

.. jupyter-execute::

    weibull.cb([120, 240], "pcm")

In a simulation study of illness-death and progressive models with
constant and Weibull hazards (1000 replicates of each design at 100, 300
and 1000 units; ``scripts/studies/multistate_occupancy`` in the
repository), these bounds held their level at 300 and 1000 units for every
state, including one occupied 1% to 5% of the time: 95% bounds covered
93.3% to 97.2% cell by cell, 95.0% on average. At 100 units, where the
hazard out of the rare state rested on some 10 to 35 observed moves, the
bounds that involve the rare state were two to four points short (as low
as 91.2% for 95% and 86.0% for 90%), the band lying below the truth more
often than above it; ``bound_type="plain"`` was further short (85.0%).
With few moves between two states, read the bounds as somewhat
optimistic.

``ParametricMultiStateRegression`` puts a parametric regression on each
transition (``WeibullPH`` by default; any of the proportional-hazards,
accelerated-failure-time and other parametric regressions, one for every
transition or one each), and predicts for given covariates:

.. jupyter-execute::

    from surpyval.multistate import ParametricMultiStateRegression

    reg = ParametricMultiStateRegression.fit_from_df(
        spells, "x", ["age70", "male"], "e", "state", tl_col="tl",
        i_col="id", transitions=tm,
    )
    for z in ([0.5, 1.0], [-0.5, 0.0]):
        print(z, reg.occupancy(120, z, "death").round(3),
              reg.cb(120, z, "death").round(3))

Expected length of stay
~~~~~~~~~~~~~~~~~~~~~~~

How long, on average, does a unit spend in each state over a period?
That is the area under its occupation probability curve,
:math:`L_b(s, t) = \int_s^t P(\text{in } b \text{ at } u)\, du`, the
multi-state version of the restricted mean survival time; over all the
states it adds up to :math:`t - s`. ``length_of_stay`` gives it, and
``length_of_stay_cb`` bounds it (the delta method on the logit of the
fraction of the time spent in the state, so the bounds stay between 0 and
:math:`t - s`):

.. jupyter-execute::

    for state in weibull.states:
        print(state, weibull.length_of_stay(240, state).round(1))
    weibull.length_of_stay_cb(240, "pcm").round(2)

Over the first 20 years an average patient spends about 9 of them with
MGUS and only a few months with myeloma.

Semi-Markov models: the clock restarts in each state
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The models above run every hazard on the time since the start (the
clock goes forward: a Markov model). Often the time that matters is the
time spent in the current state instead: the time since a repair, since
a relapse, since a machine was moved to standby. With ``clock="reset"``
each transition's hazard is a function of the time since the unit
entered its 'from' state (a semi-Markov model). Each transition is then
fitted on the times spent in the state, ``x - tl``, with no truncation;
``tl`` must therefore be the time the unit entered the state.

.. jupyter-execute::

    reset = ParametricMultiState.fit_from_df(
        spells, "x", "e", "state", tl_col="tl", i_col="id",
        transitions=tm, clock="reset",
    )
    reset.models[("pcm", "death")]

Here every patient starts with MGUS at diagnosis, so the transitions out
of it are the same on both clocks; only the hazard of death after
myeloma changes, which now falls with the time since the myeloma was
found (a shape below 1).

Where a unit is at time :math:`t` now depends on when it made each
earlier move, so the probabilities are no longer the solution of the
forward equations. SurPyval solves the *renewal equations* instead: the
rate at which units enter a state is the sum, over the states they come
from, of the rate they entered those states times the chance of leaving
for this one after the time since. It counts the entries into each state
on a fine grid (evenly spread within each cell), by convolution, and
halves the grid until the probabilities settle (to about
:math:`10^{-6}`). It is exact in the limit, deterministic, and works
with recovery (a cycle in the transitions) as well as without; R's
``flexsurv`` answers the same question by simulating paths.

.. jupyter-execute::

    for state in reset.states:
        print(state, reset.occupancy([120, 240], state).round(4),
              reset.cb(240, state).round(4))
    print("time with myeloma by 20 years:",
          reset.length_of_stay(240, "pcm").round(2))

``transition_probabilities(x, s)`` gives, row by row, the probabilities
for a unit that *entered* each state at ``s``; since the hazards depend
only on the time in the state, they depend only on ``x - s``. A
prediction from a later ``s`` needs ``from_state``: units under
observation at ``s`` have been in their states for different times.

.. _multistate-interval-moves:

Moves known only to lie between two times
-----------------------------------------

Sometimes a move is not seen when it happens but is found at an
inspection: a pump last seen working at one inspection is found degraded
at the next; a patient well at one visit is ill at the next. The move
happened at some unknown time between the two looks. Give such a spell's
``x`` as two times, ``[xl, xr]``: the last time the unit was seen in its
state and the first time it was seen in the next (``xl = xr`` for a
spell whose end is known, and for a censored one). The spell that the
move began starts somewhere in that interval; give its ``tl`` as either
end of it (the time the unit was first seen there, say), and the unit
``i``, which ties the two together.

.. jupyter-execute::

    from surpyval import Exponential

    inspected = ParametricMultiState.fit(
        x=[[2, 4], [7, 7], [3, 3], [1, 2], [6, 6]],
        e=["degraded", "failed", "failed", "degraded", None],
        state=["working", "degraded", "working", "working", "degraded"],
        tl=[0, 4, 0, 0, 2],
        i=[1, 1, 2, 3, 3],
        transitions=[
            ("working", "degraded"), ("working", "failed"),
            ("degraded", "failed"),
        ],
        dist=Exponential,
    )
    inspected.params.round(4)

Unit 1 was found degraded at 4 after working at 2, and failed at 7;
unit 3 was found degraded at 2 and was still degraded at 6.

The same spells in a :class:`~surpyval.SurpyvalData` mark each such move
with ``c = 2`` (and ``x`` as ``[xl, xr]``, or the ends in ``xl`` and
``xr``); the fit is the same:

.. jupyter-execute::

    import surpyval as sp

    data = sp.SurpyvalData(
        x=[[2, 4], [7, 7], [3, 3], [1, 2], [6, 6]],
        c=[2, 0, 0, 2, 1],
        e=["degraded", "failed", "failed", "degraded", None],
        s=["working", "degraded", "working", "working", "degraded"],
        tl=[0, 4, 0, 0, 2],
        i=[1, 1, 2, 3, 3],
    )
    ParametricMultiState.fit(
        data,
        transitions=[
            ("working", "degraded"), ("working", "failed"),
            ("degraded", "failed"),
        ],
        dist=Exponential,
    ).params.round(4)

The likelihood then averages over the unknown time of the move: the
chance of staying in the old state until then, times the rate of the
move then, times the chance of what was seen of the unit afterwards had
it entered the new state then, added up over every time in the interval.
That ties the transitions together (the moves out of the old state
share the integral, and so do the old and new states' hazards), so they
are no longer fitted one at a time but all at once, from their one-at-a-
time fits to the middles of the intervals, and the covariance is the
inverse of the whole likelihood's curvature, with terms between
transitions. On the clock reset the next spell's hazard runs from the
unknown entry, so its density is averaged over the interval too (and
over the next interval, where that move was found at an inspection as
well). The integrals use tanh-sinh quadrature, which stays accurate
where a hazard is infinite at the start of an interval (a Weibull with a
shape below 1).

The model takes each move to be direct: in the interval the unit went
from one state to the next with nothing in between. Where it might have
passed through another state unseen (fallen ill and recovered between
visits), the data are panel data, below. With a single transition the
fit is the univariate interval-censored fit (``Weibull.fit`` with ``c =
2``), and as an interval narrows to a point the fit becomes the one with
the move's time known; both are tested. With constant hazards and no way
back, the model's likelihood is R ``msm``'s for the same visits with
exactly timed deaths (``obstype = 2``), and the fit agrees with msm's
to :math:`10^{-6}` (``scripts/reference/reference_r_multistate_interval.R``);
simulated inspections recover Weibull hazards, and their bounds cover at
their nominal level (``surpyval/tests/calibration/test_coverage_multistate_interval.py``).

States seen only at visits (panel data)
---------------------------------------

Often nobody sees a unit change state. A patient's condition is graded at
clinic visits, a component is found worn at an inspection, a survey asks
each wave who is employed. All that is known is the state at each visit,
so a move happened at some unknown time between two visits, and a unit
may even pass through a state between visits that is never seen (ill and
recovered again). Spells cannot be written down, and the models above do
not apply.

``PanelMultiState`` fits the standard model for such *panel data*
(Kalbfleisch and Lawless 1985, and R's ``msm``): a Markov chain in
continuous time with a constant *transition intensity* for each allowed
move, the rate of that move while a unit is in its 'from' state. Over a
gap of length :math:`t` the probability of going from state :math:`r` to
state :math:`s`, by any route, is element :math:`(r, s)` of the matrix
exponential :math:`P(t) = \exp(Q t)`, :math:`Q` the matrix of
intensities. Each unit contributes the probability of the states it was
seen in, given its first, multiplied over the gaps between its visits;
the intensities that make the data most likely are the estimates. The
mean stay in a state is one over the total rate of leaving it.

The data are one row per visit: the time (``x``), the state seen
(``s``) and the unit (``i``). Three kinds of row go beyond a plain
visit:

- a move into an absorbing state timed exactly (a death, a scrapping):
  name the state in ``exact_entry``, and the row means "in some other
  state just before, entered this one at this time";
- a visit that saw only a set of states ("alive, condition not
  recorded"): give it a code and say what the code means in
  ``censored``, ``{"alive": ["healthy", "ill"]}``;
- a move seen as it happened, the unit having stayed in its previous
  state until then: ``obstype="exact"`` on that row.

Unlike the spell models, the allowed transitions must be given: they
cannot be read off the visits.

.. jupyter-execute::

    import numpy as np
    from scipy.linalg import expm
    from surpyval.multistate import PanelMultiState

    rng = np.random.default_rng(7)
    states = ["healthy", "ill", "dead"]
    Q = np.array([[-0.35, 0.3, 0.05], [0.2, -0.45, 0.25], [0, 0, 0]])
    x, s, i = [], [], []
    for unit in range(150):
        state, t = 0, 0.0
        for visit in np.arange(0.0, 6.0, 1.0):
            if visit > 0:
                p = expm(Q * 1.0)[state]
                state = rng.choice(3, p=p / p.sum())
            x.append(visit)
            s.append(states[state])
            i.append(unit)
            if state == 2:
                break

    panel = PanelMultiState.fit(
        x, s, i, transitions=[("healthy", "ill"), ("healthy", "dead"),
                              ("ill", "healthy"), ("ill", "dead")],
    )
    panel

The fitted intensities, with Wald bounds on the log scale, the
probabilities of each state later on for a unit healthy now (with
delta-method bounds), and the mean stays:

.. jupyter-execute::

    print(panel.summary()[["estimate", "lower", "upper"]].round(3))
    print(panel.transition_probabilities(2.0).round(3))
    print(panel.occupancy(2.0, "dead", from_state="healthy").round(3),
          panel.cb(2.0, "dead", from_state="healthy").round(3))
    print(round(panel.mean_sojourn("healthy"), 2),
          panel.sojourn_cb("healthy").round(2))

``PanelMultiStateRegression`` puts covariates on the log intensities,
each transition with its own coefficients or one ``shared`` by every
transition; a row's covariates hold over the gap that starts at it, so
they may change from visit to visit. The coefficients are log intensity
ratios, and ``summary`` shows the ratios with their bounds. ``pci``
(a list of times) lets every intensity change at those times, each later
period multiplying each transition's intensity by its own factor.

The model assumes the future depends only on the current state (Markov)
and, apart from ``pci`` and covariates, that the intensities stay the
same over time. The standard errors come from the observed information
(the exact gradient of the likelihood, differenced once), and the bounds
on probabilities from the delta method through the exact derivatives of
the matrix exponential. Misclassified states (a hidden Markov model,
where a visit can record the wrong state) are not built.

Intensities that change with time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A constant intensity says the chance of a move in the next month is the
same however long the unit has been running, or in its state. Wear,
ageing and recovery rarely work like that. ``ParametricPanelMultiState``
takes the same visits and fits a distribution's hazard to each
transition instead (a Weibull by default, so each move has a scale and
a shape: a shape above 1 is a rate that rises, below 1 one that falls).
Two time scales are offered:

- ``clock="forward"`` (the default): each rate depends on the time
  since the start, such as age or hours since new (a *non-homogeneous*
  Markov model). The chance of going from one state to another between
  two visits, by any route, is still an element of a matrix
  :math:`P(t_{j-1}, t_j)`, now the solution of the Kolmogorov forward
  equations with rates that change along the way, found by the product
  integration ``ParametricMultiState`` uses.
- ``clock="reset"``: each rate depends on the time since the unit
  entered its current state (a *semi-Markov* model: a repaired item
  starting afresh, a disease stage with its own course). The time of
  entering a state is itself not seen, so the likelihood averages over
  it, and over the entry into every state the unit may have passed
  through between two visits unseen. Each unit is taken to have entered
  its first state at its first visit. In a progressive model (no going
  back to a state once left) those routes can be listed and every one
  is counted. With a way back (recovery) a unit could make any number
  of moves between two visits, so the routes of at most ``max_moves``
  moves (default 3) are counted; see below.

The fitted model is a ``ParametricMultiState``: the same occupation
probabilities, bounds and lengths of stay. With exponential
distributions either clock gives ``PanelMultiState``'s fit. Here the
data above were simulated with constant intensities: each fitted shape
is within two standard errors of 1, and the AIC barely moves (by less
than one, for four more parameters), so they give no reason to leave
the constant model:

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval.multistate import ParametricPanelMultiState

    weibull = ParametricPanelMultiState.fit(
        x, s, i, transitions=[("healthy", "ill"), ("healthy", "dead"),
                              ("ill", "healthy"), ("ill", "dead")],
        dist=Weibull,
    )
    for name, value, se in zip(weibull.parameter_names, weibull.params,
                               weibull.standard_errors()):
        print(f"{name:24s} {value:8.3f} ({se:.3f})")
    print(round(panel.aic(), 1), round(weibull.aic(), 1))

Every transition is fitted at once (the moves share each gap's
probability), by BFGS on the log of each positive parameter, starting
from the constant-intensity fit; the covariance is the inverse of the
whole likelihood's numerical Hessian. On the forward clock the grid of
the product integral is fixed for the search (every visit time, and the
points between that the accuracy needs, chosen again at the estimate)
so that the likelihood is smooth in the parameters. With exponential
distributions it matches ``PanelMultiState`` and R ``msm`` to
:math:`10^{-7}`; with Weibull hazards on the forward clock it matches R
``nhm`` (Titman 2011); on the reset clock it is checked against
adaptive quadrature and, where no state can be passed unseen, against
``ParametricMultiState`` with the moves interval censored. Simulated
visits recover the Weibull shapes, with bounds at their nominal
coverage (``surpyval/tests/calibration/test_coverage_multistate_panel_parametric.py``).

A way back on the reset clock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With recovery, a unit seen healthy at two visits may have stayed
healthy, or fallen ill and recovered, or done that twice, and so on: on
the reset clock each of those routes enters a state at an unknown time,
and each needs its own integral. ``ParametricPanelMultiState`` (and the
regression) count the routes of at most ``max_moves`` moves in each gap
between visits (default 3) and leave out the rest. What is left out is
worked out at the estimate, gap by gap: the chance, given what was seen
before the gap, of more moves in it than are counted. That is how far
the gap's factor of the likelihood can be short. The largest is
``model.truncation_error`` and every gap's is in
``model.truncation_errors``; a warning names the worst gap when any is
above ``truncation_tol`` (default :math:`10^{-8}`). Raise ``max_moves``
until the warning goes, or until the estimates stop moving.

How many moves are needed depends on how many happen in a gap: about
the rate of leaving a state times the time between visits. With rates
of 0.2 to 0.45 a unit of time and visits a unit apart, bound 3 leaves
out up to about :math:`2 \times 10^{-4}` of a gap and the intensities
are within about 1% of the Markov fit's; each extra move cuts what is
left out about tenfold. Every extra move nests one more integral, so the
fit slows quickly as the bound rises. On 40 simulated units seen at 0 to
5, with constant rates of 0.05 to 0.3 (fitted with exponential
distributions, so the Markov fit is the exact answer):

.. list-table::
   :header-rows: 1

   * - ``max_moves``
     - largest share left out of a gap
     - log-likelihood short by
     - intensities off by up to
     - time to fit
   * - 1
     - 0.029
     - 5.6
     - 77%
     - 2 s
   * - 2
     - 0.0024
     - 0.38
     - 8.5%
     - 75 s
   * - 3
     - 0.0002
     - 0.030
     - 1.0%
     - 3 min

A Weibull fit (eight parameters) of 60 such units at bound 3 had not
finished after 40 minutes. For
rare moves or visits close together the default bound is ample; where
visits are far apart compared with the time a unit stays in a state,
the forward clock (``clock="forward"``, which takes cycles exactly) or
``PanelMultiState`` is the practical choice.

With exponential hazards the reset clock is the Markov model, so the
truncated fit approaches ``PanelMultiState``'s as ``max_moves`` rises
(checked in the tests, along with the exact chance of each number of
moves from one block matrix exponential, and adaptive quadrature for a
Weibull unit that falls ill and recovers between two visits).

Covariates with intensities that change with time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``ParametricPanelMultiStateRegression`` is the same model with covariates:
a parametric regression for each transition (``WeibullPH`` by default, so
each move has a Weibull baseline and its own coefficients; any of the
package's parametric regressions can be given, ``WeibullAFT`` say). The
covariates are given per visit, as for ``PanelMultiStateRegression``: the
values recorded at a visit hold until the unit's next visit, so a
treatment started at a visit counts from then on.

In words: between two visits the unit's rates of moving are those of a
unit with that gap's covariates, so the chance of going from the state
seen at one visit to the state seen at the next is the same product
integral as above, worked out for that gap's covariates. On the forward
clock each distinct row of covariates gets its own grid (two rows that
differ in any covariate are two grids, so a covariate with many values,
such as a measurement taken at every visit, makes many grids and a slower
fit; see :ref:`multistate-coarsen`). On the reset clock the unit is
carried over its unknown entry times as before, each gap's hazards taken
at that gap's covariates.

With ``ExponentialPH`` regressions the model is
``PanelMultiStateRegression``'s, on either clock:

.. jupyter-execute::

    import numpy as np
    from surpyval import ExponentialPH
    from surpyval.multistate import (
        PanelMultiStateRegression, ParametricPanelMultiStateRegression,
    )

    z = (np.asarray(i) % 2).astype(float)  # a covariate per unit
    tm = [("healthy", "ill"), ("healthy", "dead"), ("ill", "healthy"),
          ("ill", "dead")]
    markov = PanelMultiStateRegression.fit(x, z, s, i, transitions=tm)
    ph = ParametricPanelMultiStateRegression.fit(
        x, z, s, i, transitions=tm, model=ExponentialPH,
    )
    print(markov.betas.ravel().round(4))
    print(ph.betas.ravel().round(4))
    print(round(markov.log_likelihood, 6), round(ph.log_likelihood, 6))

.. _multistate-coarsen:

Grouping a covariate measured at every visit
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A covariate measured afresh at every visit (a blood pressure, a vibration
level) has a different value in almost every gap, and on the forward
clock every distinct row of covariates is a product integral of its own.
``coarsen`` groups such a covariate's values before the fit, so that only
a few distinct rows are left. **It is an approximation**: the model is
then fitted to the grouped values, not the measured ones. It is off by
default.

``coarsen=k`` cuts every covariate with more than ``k`` distinct values
into ``k`` bins of about equal counts (at its quantiles, as
``pandas.qcut``) and replaces each value by the mean of the values in its
bin; a covariate with at most ``k`` values (an indicator) is left alone.
A function is applied to every covariate instead (``coarsen=np.round``,
or ``lambda z: z.round(1)``), and a dict coarsens only the covariates it
names (``coarsen={"w": 10}``). The fit is exactly the fit to the
covariates grouped that way by hand, and ``model.coarsened`` gives each
grouped covariate's number of distinct values before and after. The
model holds and saves (``to_dict``) the grouped covariates; its
predictions take the covariate values you give them, measured or not.
At least 2 bins are needed (one would make the covariate a constant):

.. code-block:: python

    model = ParametricPanelMultiStateRegression.fit(
        x, Z, s, i, transitions=tm, coarsen={"w": 10},
    )
    model.coarsened  # {'w': (740, 10)}

How much it changes the answer depends on the data. On a simulated
illness-death model (100 units seen every unit of time for 5 to 10
units, Weibull proportional hazards, a covariate fixed for each unit and
one, ``w``, uniform on (0, 2) and drawn at every visit: 785 visits and
740 distinct values of ``w``; ``visits_covariates(100, 5, ...,
continuous=True)`` in ``surpyval/tests/multistate/_inspected.py``, with
the parameters of ``test_parametric_panel_regression.py``) the full fit
took about 230 s on one core of the machine it was run on. With
``coarsen`` it took about 15 s (5 bins), 25 s (10), 38 s (20) and 58 s
(40).
The coefficient of ``w`` on healthy to ill, 0.639 (standard error 0.184)
in the full fit, was 0.664, 0.658, 0.657 and 0.652, moving by 0.14 to
0.07 of its standard error; the chance of being ill at time 5 with
``w = 1`` (0.5453) moved by at most 0.0005. The coefficient of ``w`` on
ill to dead, the least well estimated, moved most: 0.284 in the full
fit, 0.231 with 5 bins and 0.280 with 40. Check the effect on your own
data by fitting with two numbers of bins: if the answers you need do not
move between them, the grouping is fine enough.

The fit is joint, as above, with each coefficient searched in its
covariate's units (a covariate in thousands is searched as well as one in
ones). A covariate that cannot be told from the others over the gaps (or
from a constant) is reported, with ``nan`` for its coefficient, as in the
other regressions.

How it was checked
------------------

The estimates and every variance agree with R to rounding: the
occupation probabilities and their jackknife standard errors with
``survival``'s ``survfit`` (from the start, and from a later time, with
delayed entry and tied times); the transition probabilities and their
Greenwood-type and Aalen-type standard errors with ``mstate``'s
``probtrans``; and the Cox models' coefficients and predictions, with their
standard errors, with ``mstate``'s ``msfit`` and ``probtrans`` (Breslow's
and Efron's ties, transition-specific effects, effects shared by every
transition or by some, and proportional baselines; for the last, with
``msfit``'s covariance of two transitions sharing a baseline, as above,
and its log partial likelihood). The same was checked, outside the test
suite, on ``mstate``'s own ``ebmt3`` data (an illness-death model of
transplant patients, with transition-specific, partly shared and
proportional-baseline models): the coefficients and log partial
likelihood agree to about :math:`10^{-10}`, the probabilities to
:math:`10^{-12}`, and the standard errors to :math:`10^{-11}` (with
``msfit``'s covariance term; 7% with the one above). With one
transient state the estimate is SurPyval's competing-risks cumulative
incidence, and with one transition its Kaplan-Meier estimate with
Greenwood's variance. Simulated Markov models are recovered within their
bounds. The script that produced R's numbers is
``scripts/reference/reference_r_multistate.R``.

The parametric models were checked against R's ``flexsurv`` (fitting
each transition with ``flexsurvreg`` and solving the forward equations
with ``pmatrix.fs``), on the same data: the fits reach the same maximum,
and with flexsurv's parameters the transition probabilities agree to
about :math:`10^{-11}`. The forward equations were also checked against
a closed form (Weibull hazards of different shapes, one infinite at 0),
and with one transient state the occupation probabilities equal
``ParametricCompetingRisks``' cumulative incidences. The delta-method
standard errors match the spread of the probabilities over simulated
parameters, and simulated Markov models are recovered within their
bounds. The script is
``scripts/reference/reference_r_multistate_parametric.R``.

The semi-Markov models were checked against a closed form (an
illness-death model with Weibull hazards of the time in each state, one
infinite at 0, where the probabilities and lengths of stay are
one-dimensional integrals) to :math:`10^{-6}`; with exponential hazards
(where the clock does not matter) against the forward equations, with
recovery; against a seeded simulation of paths with recovery; and
against R's ``flexsurv``: the clock-reset fits reach flexsurv's maximum,
and with flexsurv's parameters the probabilities agree with its
simulation (``pmatrix.simfs``, a million paths) within its Monte Carlo
error, as do the lengths of stay (``totlos.simfs``). The clock-forward
lengths of stay agree with ``totlos.fs`` to about :math:`10^{-8}` of the
time. The delta-method standard errors match the spread over simulated
parameters, and a calibration study checks the coverage of the bounds.
The script is ``scripts/reference/reference_r_multistate_semimarkov.R``.

The panel-data models were checked against R's ``msm`` 1.9 on a
simulated four-state model with exactly timed deaths and visits that saw
only "alive": the log-likelihoods agree to :math:`10^{-12}` and the
intensities, standard errors, bounds, transition probabilities, mean
stays and intensity ratios (with covariates that change between visits,
an effect shared by every transition, and piecewise-constant intensities)
to about :math:`10^{-6}`. With every move seen when it happens the
estimates are the numbers of moves over the time at risk, and with two
states the fit is the exponential fit to interval-censored lifetimes.
Simulated panel data show the bounds on the intensities, the occupation
probabilities and the mean stays covering at their nominal level. The
script is ``scripts/reference/reference_r_multistate_panel.R``.

The parametric panel regression was checked on a simulated
illness-death model with Weibull proportional hazards, one covariate
fixed for each unit and one drawn at every visit
(``scripts/reference/reference_r_multistate_panel_regression.R``):
with exponential hazards against R ``msm`` with covariates (the
log-likelihood to :math:`10^{-14}`, the intensities and coefficients to
about :math:`10^{-6}`), and with Weibull hazards against R ``nhm``, whose
likelihood SurPyval's equals at nhm's estimates to :math:`10^{-9}`. Each
gap's factor is checked against an ODE solver with the gap's covariates,
and on the reset clock a stay spanning three rows of covariates against
adaptive quadrature. Simulated visits recover the coefficients, with
bounds at their nominal coverage
(``surpyval/tests/calibration/test_coverage_multistate_panel_regression.py``).

What is not here yet
--------------------

The non-parametric and Cox spell models refuse interval-censored
changes of state (the parametric ones take them); for states seen only
at visits use ``PanelMultiState``, which has no hidden Markov
(misclassification) model yet.
On the clock reset with a way back, ``ParametricPanelMultiState`` and
``ParametricPanelMultiStateRegression`` count at most ``max_moves`` moves
between two visits, and the nested integrals make large bounds slow; no
phase-type approximation is offered. The parametric models have no
likelihood-ratio or bootstrap bounds. The clock reset is offered for the
parametric spell models only: a Cox model per transition on the time in
the state would need its probabilities simulated over step hazards, and
the Aalen-Johansen estimate is Markov by construction (for a non-Markov
process its occupation probabilities are still consistent, Datta and
Satten, 2001, but its transition probabilities from a later time are
not). Covariates that change over a stay are not taken.

References
----------

Aalen, O. O. and Johansen, S. (1978), "An empirical transition matrix for
non-homogeneous Markov chains based on censored observations",
*Scandinavian Journal of Statistics*, 5(3), 141-150.

Andersen, P. K. and Keiding, N. (2002), "Multi-state models for event
history analysis", *Statistical Methods in Medical Research*, 11(2), 91-115.

Blanes, S., Casas, F., Oteo, J. A. and Ros, J. (2009), "The Magnus
expansion and some of its applications", *Physics Reports*, 470(5-6),
151-238.

Datta, S. and Satten, G. A. (2001), "Validity of the Aalen-Johansen
estimators of stage occupation probabilities and Nelson-Aalen estimators of
integrated transition hazards for non-Markov models", *Statistics &
Probability Letters*, 55(4), 403-411.

de Wreede, L. C., Fiocco, M. and Putter, H. (2010), "The mstate package for
estimation and prediction in non- and semi-parametric multi-state and
competing risks models", *Computer Methods and Programs in Biomedicine*,
99(3), 261-274.

Jackson, C. H. (2011), "Multi-state models for panel data: the msm
package for R", *Journal of Statistical Software*, 38(8), 1-29.

Jackson, C. H. (2016), "flexsurv: a platform for parametric survival
modeling in R", *Journal of Statistical Software*, 70(8), 1-33.

Kalbfleisch, J. D. and Lawless, J. F. (1985), "The analysis of panel
data under a Markov assumption", *Journal of the American Statistical
Association*, 80(392), 863-871.

Kyle, R. A., Therneau, T. M., Rajkumar, S. V., et al. (2002), "A long-term
study of prognosis in monoclonal gammopathy of undetermined
significance", *New England Journal of Medicine*, 346(8), 564-569.

Putter, H., Fiocco, M. and Geskus, R. B. (2007), "Tutorial in
biostatistics: competing risks and multi-state models", *Statistics in
Medicine*, 26(11), 2389-2430.

Titman, A. C. (2011), "Flexible nonhomogeneous Markov models for panel
observed data", *Biometrics*, 67(3), 780-787.

API
---

Import these from ``surpyval.multistate`` (``MultiState``,
``MultiStateProportionalHazards``, ``ParametricMultiState``,
``ParametricMultiStateRegression``, ``PanelMultiState``,
``PanelMultiStateRegression``, ``ParametricPanelMultiState``,
``ParametricPanelMultiStateRegression`` and ``TransitionMatrix`` are also
at
the top level, ``surpyval.MultiState`` and so on).

.. autoclass:: surpyval.multistate.TransitionMatrix
   :members:

.. autoclass:: surpyval.multistate.MultiState
   :members:

.. autoclass:: surpyval.multistate.MultiStateProportionalHazards
   :members:

.. autoclass:: surpyval.multistate.ParametricMultiState
   :members:

.. autoclass:: surpyval.multistate.ParametricMultiStateRegression
   :members:

.. autoclass:: surpyval.multistate.PanelMultiState
   :members:

.. autoclass:: surpyval.multistate.PanelMultiStateRegression
   :members:

.. autoclass:: surpyval.multistate.ParametricPanelMultiState
   :members:

.. autoclass:: surpyval.multistate.ParametricPanelMultiStateRegression
   :members:

.. autofunction:: surpyval.multistate.to_stacked

.. autofunction:: surpyval.multistate.from_stacked
