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

The models assume the hazards run on the time since the start (the clock
goes forward: a Markov model). A model whose hazards run on the time in
the current state (``clock="reset"``, semi-Markov) is refused: its
probabilities are not the solution of the forward equations.

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

What is not here yet
--------------------

Semi-Markov (clock-reset) models are not built. The spell models refuse
interval-censored changes of state: for states seen only at visits use
``PanelMultiState``, which has no hidden Markov (misclassification)
model yet. The parametric models have no expected length of stay in each
state yet, nor likelihood-ratio or bootstrap bounds.

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

API
---

Import these from ``surpyval.multistate`` (``MultiState``,
``MultiStateProportionalHazards``, ``ParametricMultiState``,
``ParametricMultiStateRegression``, ``PanelMultiState``,
``PanelMultiStateRegression`` and ``TransitionMatrix`` are also at
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

.. autofunction:: surpyval.multistate.to_stacked

.. autofunction:: surpyval.multistate.from_stacked
