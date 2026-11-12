Recurrent Event Regression Analysis
====================================

In the same way that we want to understand the relationship between a
single event outcome and a set of covariates, we may want to understand the
relationship between a recurrent event process and a set of covariates. For example,
we may want to understand the relationship between the number of times an
item needs repairing based on the environment, the duty cycle, and the
types of inputs it receives. For example, an electric motor in a hot and humid
environment may need more repairs than one in a cool and dry environment.
Additionally, a motor that is used more often may need more repairs than one
that is used less often. Finally, a motor that is used in a factory that
produces a lot of dust may need more repairs than one that is used in a
cleaner environment. We can use recurrent event regression to understand
the relationship between these covariates and the number of repairs that
a motor needs.

This page builds on the counting-process ideas — the intensity
:math:`\lambda(t)`, the cumulative intensity :math:`\Lambda(t)` and the
Poisson-process likelihood — explained on the :doc:`Recurrent Event Analysis`
page. Runnable examples of everything here are on the
:doc:`Recurrent Event Regression Modelling with SurPyval` page, and the API
is documented under :doc:`surpyval.counting`.

.. rubric:: From a rate to a rate that depends on covariates

We can start with the simplest version of recurrent event regression, a
Homogeneous Poisson Process Regression. In this case we have a regular
Homogeneous Poisson Process, whose expected number of events by time
:math:`t` is given by:

.. math::

    \Lambda(t) = \mathbb{E}\left[N(t)\right] = \lambda t

But we extend this to include the impact that additional factors have on the
cumulative count. So the model becomes:

.. math::

    \Lambda(t \mid Z) = \phi\left( Z \right) \lambda t

Here :math:`Z` is the row vector of an item's covariates (for example
:math:`Z = [\text{humid}, \text{duty cycle}]`) and :math:`\phi` is a positive
function of them. In this case, just as was the case with single event
proportional hazard models, there is a factor that relates the covariates to
the counting function. In doing so we can now model jointly the cumulative
event process with factors that are likely to impact the rate at which the
events occur. Again, repeating the lessons from single event survival
analysis, a logical choice for the phi function would be the exponential
function. This is because it ensures there is never a negative number and so
will always provide a valid rate even during optimisation. That is, our model
will be:

.. math::

    \Lambda(t \mid Z) = e^{Z \beta} \lambda t


This is the proportional intensity HPP model. The log-linear (exponential)
link function :math:`e^{Z\beta}` is the standard choice because it guarantees
a positive rate regardless of the sign of :math:`\beta`, mirrors the Cox model
for single events, and gives regression coefficients a direct multiplicative
interpretation on the rate. Here :math:`\beta` is the column vector of
regression coefficients, one per covariate, so
:math:`Z\beta = \beta_0 z_0 + \beta_1 z_1 + \dots`. (SurPyval names each
coefficient by its column of :math:`Z` where the fit has the columns'
names -- ``fit_from_df`` or a DataFrame ``Z`` -- and ``coef_0``,
``coef_1``, ... in the order of the columns otherwise.)

.. rubric:: Time-varying baselines: the Duane example

A constant rate is often too simple: a system under development gets more
reliable as its faults are found and fixed, and an ageing one fails more and
more often. The Duane process, particularly relevant in reliability
engineering, is the classic model of such a trend. It is a non-homogeneous
Poisson process (NHPP) whose expected number of events is a power of time;
the parameterisation SurPyval uses is

.. math::

    \Lambda(t) = b\, t^{\alpha},
    \qquad
    \lambda(t) = \frac{d\Lambda(t)}{dt} = \alpha\, b\, t^{\alpha - 1} .

The exponent :math:`\alpha` controls the trend — below one the rate of events
falls over time (reliability growth, the case Duane studied), above one it
rises (wear-out) — and :math:`b` is the expected number of events by
:math:`t = 1`. (Many textbooks swap the letters, writing
:math:`\alpha t^{\beta}`; the parameters that SurPyval prints are ``alpha``
for the exponent and ``b`` for the scale.)

To add covariates we keep this trend as a *baseline* intensity
:math:`\lambda_0(t)` and scale it by the same factor as before, exactly as the
proportional hazards model scales a baseline hazard:

.. math::

    \lambda(t \mid Z) = \lambda_0(t)\, e^{Z\beta}
    = \alpha\, b\, t^{\alpha - 1} e^{Z\beta} .

Every item shares the power-law trend; its covariates only make its events
more or less frequent at every age.

.. rubric:: The general proportional-intensity model

More generally, any of SurPyval's counting-process baselines can play the role
of :math:`\lambda_0(t)`. The proportional-intensity model multiplies that
baseline by the covariate factor :math:`e^{Z\beta}`:

.. math::

    \lambda(t \mid Z) = \lambda_0(t)\, e^{Z\beta},
    \qquad
    \Lambda(t \mid Z) = \Lambda_0(t)\, e^{Z\beta}.

With a constant baseline this is the proportional-intensity HPP; with a
power-law (Duane / Crow-AMSAA) or log-linear (Cox-Lewis) baseline it is the
proportional-intensity NHPP. Because the covariate factor scales the whole
cumulative intensity, the ratio of expected event counts between two covariate
settings is the constant :math:`e^{(Z_2 - Z_1)\beta}` at every time — the
recurrent-event analogue of a hazard ratio [Cook2007r]_, and the reason the
coefficients :math:`\beta` read directly as multiplicative effects on the event
rate.

The baseline :math:`\lambda_0(t)` is the intensity of an item whose covariates
are all zero. The covariates change *how many* events an item has, never the
*shape* of its intensity over time: every item shares the same trend, scaled up
or down. Each item's events still form a Poisson process — given its
covariates, an item's intensity does not depend on its own past events.

.. rubric:: Estimation

The parameters are estimated jointly by maximum likelihood [Lawless1987]_,
maximising the NHPP log-likelihood with the covariate-scaled intensity
substituted in. For item :math:`i` with covariates :math:`Z_i`, observed from
:math:`s_i` to :math:`\tau_i` with events at :math:`t_{i1}, t_{i2}, \dots`,

.. math::

    \ell(\theta, \beta) = \sum_i \left\{ \sum_j \left[\ln \lambda_0(t_{ij};
    \theta) + Z_i\beta\right] - e^{Z_i\beta}\left[\Lambda_0(\tau_i; \theta)
    - \Lambda_0(s_i; \theta)\right] \right\},

where :math:`\theta` are the baseline parameters. Exactly as for the models
without covariates, the window end :math:`\tau_i` is the item's
right-censoring time (time-truncated) or its last event (failure-truncated),
the start :math:`s_i` is its left-truncation time or zero, and counts of
events over inspection intervals enter as Poisson terms. The static, per-item
covariates enter through :math:`e^{Z\beta}`, so an item in a harsher
environment simply accumulates events faster. This proportional-intensity
framing is standard in the reliability-growth literature [Rigdon2000r]_.

The baseline parameters and the coefficients are estimated together, and
their joint covariance (the inverse observed information) gives standard
errors for every parameter and delta-method confidence bounds on
:math:`\Lambda(t \mid Z)` at any covariate setting.

.. rubric:: Covariates that change over time

So far a covariate has been a property of the item. Often it is a property
of a *period* of the item's life instead: a pump runs at high load in some
months and at low load in others, a machine is overhauled part of the way
through its life, a vehicle moves to a harsher route. Using one number per
item for such a covariate (its first value, or its average) gives the wrong
answer: it credits failures to conditions that were not in force when they
happened, and it throws away the timing that carries most of the
information about the covariate's effect.

The proportional-intensity model extends to this case with one assumption:
the value the covariate has *now* multiplies the intensity *now*. In words,
while the pump is on high load its failure rate is the baseline rate for
its present age times the high-load factor, and when it returns to low load
the rate returns to the baseline. The model has no memory of earlier
values; if accumulated exposure matters, it has to be supplied as a
covariate of its own. With :math:`Z_i(t)` the covariates of item :math:`i`
in force at time :math:`t`,

.. math::

    \lambda_i(t) = \lambda_0(t)\, e^{Z_i(t)\beta} .

SurPyval takes covariates that are piecewise constant: item :math:`i`'s
history is cut into intervals :math:`(a_{ik}, b_{ik}]`, on each of which
its covariates hold one value :math:`Z_{ik}`. Two facts then give the
likelihood. The expected number of failures in an interval is the
baseline's expected number over that interval times the factor for the
covariates holding then, :math:`e^{Z_{ik}\beta}\left[\Lambda_0(b_{ik}) -
\Lambda_0(a_{ik})\right]`. And each failure contributes the intensity at
its time, with the covariates holding then. The log-likelihood is the sum
over failures of the log intensity minus the sum over intervals of the
expected counts:

.. math::

    \ell(\theta, \beta) = \sum_i \left\{ \sum_j \left[\ln
    \lambda_0(t_{ij}; \theta) + Z_i(t_{ij})\beta\right] - \sum_k
    e^{Z_{ik}\beta}\left[\Lambda_0(b_{ik}; \theta) - \Lambda_0(a_{ik};
    \theta)\right] \right\} .

When every item has a single covariate value this is exactly the
likelihood of the previous section, so the two fits agree; and cutting an
interval in two with the same covariates on both pieces changes nothing,
because the two expected counts add up to the original one. An item need
not be watched from time zero (its first interval can start later: delayed
entry) or without a break (a gap between intervals is time it was not at
risk). Failures that were only counted over an interval, :math:`n` of them
somewhere in :math:`(a, b]`, enter as the Poisson probability of that count,
whose mean is the interval's expected count.

A prediction now depends on the whole *schedule* of covariates, not on one
value. The expected number of failures by time :math:`t` for an item that
follows a schedule with segments :math:`(a_k, b_k]` and values :math:`z_k`
adds up the same pieces,

.. math::

    \Lambda\bigl(t \mid Z(\cdot)\bigr) = \sum_k e^{z_k\beta}
    \left[\Lambda_0(\min(t, b_k)) - \Lambda_0(a_k)\right]
    \quad \text{over the segments with } a_k < t,

and the intensity at :math:`t` is :math:`\lambda_0(t)\, e^{Z(t)\beta}`. Its
confidence bounds come from the same delta method, on the log scale, as
those at a fixed covariate setting. The number of failures an item actually
has over a period is Poisson distributed about the expected number, as
before. The worked example is on the
:doc:`Recurrent Event Regression Modelling with SurPyval` page.

.. rubric:: Several event types, each with its own effects

When the events come in several mutually exclusive types (the failure modes
of a machine), each type can be given its own baseline and its own
coefficients. In words: the rate of type :math:`k` events on an item is that
type's baseline rate at the item's age, times that type's own factor for
the item's covariates. A covariate can then raise one type's rate and leave
another's alone. For :math:`K` types,

.. math::

    \lambda_k(t \mid Z) = \lambda_{0k}(t; \theta_k)\, e^{Z\beta_k},
    \qquad k = 1, \dots, K .

If the types arrive as independent Poisson processes (an event of one type
neither hastens nor delays an event of another), the log-likelihood is a sum
of :math:`K` terms, the :math:`k`-th involving only :math:`\theta_k` and
:math:`\beta_k`:

.. math::

    \ell = \sum_{k=1}^{K} \ell_k(\theta_k, \beta_k),

where :math:`\ell_k` is the proportional-intensity log-likelihood above for
the events of type :math:`k` alone, over the whole time each item was
observed. So each type is estimated by fitting the ordinary model to its own
events, with the other types' events left out (not censored: the item stays
under observation through them), and the estimates of different types are
independent. :math:`e^{\beta_{kj}}` is the rate ratio of covariate :math:`j`
for type :math:`k`.

The rate of events of any type is the sum of the types' rates, and the
expected number of events of any type is the sum of their expected numbers:

.. math::

    \lambda_\bullet(t \mid Z) = \sum_k \lambda_k(t \mid Z), \qquad
    \Lambda_\bullet(t \mid Z) = \sum_k \Lambda_k(t \mid Z) .

Because the types' estimates are independent, the variance of an estimated
total is the sum of the types' delta-method variances,
:math:`\operatorname{Var}\hat\Lambda_\bullet = \sum_k
\operatorname{Var}\hat\Lambda_k`, and the confidence bounds are formed from
it on the log scale, as each type's are. Note that the total does *not*
follow a proportional-intensity model of its own unless every type has the
same coefficients: its rate ratio for a covariate is a mixture of the types'
ratios, weighted by their shares of the events, which is why one model
fitted to all the events can report a small effect for a covariate that is
strong for one type. The information criteria of the whole model use the
summed log-likelihood, the parameters of every type and the events of every
type. In SurPyval this is ``CauseSpecificProportionalIntensity``; with
covariates that change over time each :math:`\ell_k` is the time-varying
likelihood of the previous section. The worked example is on the
:doc:`Recurrent Event Regression Modelling with SurPyval` page.

.. rubric:: Interpreting the coefficients

- :math:`e^{\beta_k}` is the **rate ratio** for a one-unit increase in
  covariate :math:`k`, holding the others fixed: :math:`\beta_k = 0.7` means
  about twice as many events (:math:`e^{0.7} \approx 2.0`) at every age.
  Because the ratio is the same at every time, it is also the ratio of the
  expected number of events by any time :math:`t`.
- A confidence interval for :math:`\beta_k` becomes one for the rate ratio by
  exponentiating its end points.
- The baseline describes :math:`Z = 0`. Coding a two-level factor as 0/1 makes
  the baseline one of the levels; *centring* a continuous covariate (subtracting
  its mean) makes the baseline a typical item and usually makes the
  optimisation better behaved too.
- A factor with :math:`L` levels needs :math:`L - 1` indicator columns; a
  column of ones would duplicate the baseline's scale parameter, and the model
  would not be identifiable.

.. rubric:: Choosing the baseline

The choice of baseline is the choice of the *shared* trend over time, and the
same reasoning applies as without covariates: use a constant baseline (the
HPP) if there is no trend, and an NHPP baseline — a power law (``Duane``, the
default, or ``CrowAMSAA``) or the log-linear ``CoxLewis`` — if there is.
Duane and Crow-AMSAA are the same power-law process in two parameterisations,
so as baselines they describe the same model, and their fits should agree.
(Likewise ``ProportionalIntensityNHPP`` with an ``HPP`` baseline is the same
model as ``ProportionalIntensityHPP``.)
Compare candidate baselines with the information criteria and the diagnostics
below.

A useful fact: when every item is observed over the *same* window, the
coefficient estimates do not depend on the shape of the baseline at all —
only the total expected count over the window enters the part of the
likelihood that involves :math:`\beta`. The baseline then matters for
prediction over time but not for the covariate effects. When the windows
differ (items entered service at different times, say), the two are
estimated jointly and a wrong baseline can bias the coefficients.

.. rubric:: Assumptions and limitations

- **Static covariates in** ``fit``. There the covariates are properties of
  the item, constant over its observation. Supply them one row per event
  row (repeating the item's values) or as a dictionary keyed by item. The
  residuals and the goodness-of-fit test take each item's covariates from
  its first row.
- **Covariates that change over time** (``fit_tvc``) must be piecewise
  constant, and act only through their current value. The residuals, the
  trend and goodness-of-fit tests, ``diagnose``, ``plot`` and simulation
  along a schedule are not available for such a fit yet; each refuses,
  saying so.
- **Proportionality.** Every item follows the same trend, scaled by its
  covariates. If harsh-environment motors *wear out faster* (a different
  shape), rather than simply failing more often, a single proportional model
  is the wrong tool; fit the groups separately and compare their shapes.
- **Poisson behaviour and heterogeneity.** The model assumes the covariates
  explain all systematic differences between items. If items differ in ways
  the covariates do not capture (extra-Poisson variation), the coefficient
  estimates remain sensible but the model-based standard errors tend to be
  too small, and the goodness-of-fit test and martingale residuals are the
  place to look for it.
- **Scope.** Covariates are available for the HPP and NHPP intensity models,
  and for the models with an unspecified baseline in the next section.
  The imperfect-repair (renewal) models and gapped (multi-window)
  observation of the parametric models do not take covariates.

Without a baseline shape: four ways to treat a repeat event
-----------------------------------------------------------

The models above give the baseline intensity a formula. When you only care
how the covariates change the rate, and would rather not say what the
baseline looks like, the baseline can be left unspecified, exactly as a Cox
model leaves the baseline hazard of a single event unspecified. The
coefficients are then estimated from the *order* in which the items had
their events: at each event the model asks which of the items that could
have had it did.

"Could have had it" is the whole question, because an item that has had an
event is still there and can have another. There are four standard answers.
They are four different statements about what a repeat event is, so they
answer different questions, and choosing between them is a matter of the
subject, not of fit.

.. list-table::
   :header-rows: 1
   :widths: 18 41 41

   * - Model
     - What it says about repeat events
     - The question it answers
   * - Andersen-Gill [AndersenGill1982]_
     - An event does not change the item. Its rate at any age depends on
       its covariates and its age, not on how many events it has had or
       when. One baseline for every event.
     - How do the covariates change the *rate of events*?
   * - Prentice-Williams-Peterson, total time
     - An event changes the item. The rate of a second event is its own
       function of age, different from that of a first; an item cannot have
       a second before its first.
     - Among items that have had :math:`k - 1` events, how do the covariates
       change the rate of the :math:`k`-th, at a given age?
   * - Prentice-Williams-Peterson, gap time
     - The same, but the clock restarts at each event: what matters is the
       time since the last event, not the age.
     - How do the covariates change the *time between* one event and the
       next?
   * - Wei-Lin-Weissfeld
     - Nothing about how one event leads to the next. Each of "the time
       from the start to the first event", "... to the second event", and
       so on, is modelled on its own.
     - How do the covariates change the time, *from the start*, until the
       :math:`k`-th event?

Two examples make the difference concrete.

*A fleet of pumps.* Pumps trip and are restarted. If a restart leaves a
pump as it was, the number of earlier trips tells you nothing more than
its age and duty do, and the Andersen-Gill model is the natural one: "an
abrasive duty multiplies the trip rate by 1.8". If instead each trip is
followed by a repair that changes the pump, so that a pump on its third
repair is not the machine it was when new, each repair number deserves a
baseline of its own: the gap-time Prentice-Williams-Peterson model says
"an abrasive duty shortens the time from one repair to the next", with a
different typical gap after the first, second and third repair.

*A trial of a treatment for a relapsing illness.* Patients are randomised
at time 0 and may relapse several times. The Andersen-Gill model answers
"does the treatment lower the relapse rate?". The conditional models answer
"among patients who have relapsed once, does the treatment delay the
second relapse?", which is a question about a selected group: patients who
relapsed early despite treatment are not like those who relapsed on
placebo. The Wei-Lin-Weissfeld model keeps the comparison the trial
randomised: every patient counts towards "time from randomisation to the
second relapse", whether or not they have had the first.

In formulas, with :math:`Y_i(t)` equal to 1 while item :math:`i` is "at
risk" and 0 otherwise, each model is

.. math::

    \lambda_{ik}(t \mid Z_i) = Y_{ik}(t)\, \lambda_{0k}(t)\, e^{\beta' Z_i}

and they differ in the baseline and in the risk indicator:

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Model
     - Baseline
     - Item :math:`i` is at risk of event :math:`k` ...
   * - Andersen-Gill
     - :math:`\lambda_0(t)`, shared
     - whenever it is under observation.
   * - PWP, total time
     - :math:`\lambda_{0k}(t)`, :math:`t` the age
     - from its :math:`(k-1)`-th event to its :math:`k`-th.
   * - PWP, gap time
     - :math:`\lambda_{0k}(u)`, :math:`u` the time since the last event
     - from its :math:`(k-1)`-th event to its :math:`k`-th.
   * - Wei-Lin-Weissfeld
     - :math:`\lambda_{0k}(t)`, :math:`t` from the start
     - from the start until its :math:`k`-th event.

Each is fitted by the Cox partial likelihood on rows laid out to match:
one row per interval at risk, in the stratum of its event number where the
baseline depends on it [Therneau2000]_. So the tie handling, the treatment
of a coefficient the data cannot determine and the check that a maximum
was reached are those of the Cox model.

.. rubric:: Why the standard errors are "robust"

One item contributes several rows, and they are not independent: a pump
that trips often does so for reasons the covariates do not record (its
installation, its operator), and a patient who relapses early tends to
relapse again. The partial likelihood's own standard errors take every row
as a separate piece of evidence, so when items differ in such unrecorded
ways they are too small, often by a wide margin. The remedy keeps the
estimates and replaces the standard errors: add up, item by item, how much
each item pulls the estimate (its score residuals), and measure the spread
of those pulls *between items*. Items are independent of each other even
when an item's events are not.

This is the sandwich estimator with each item a cluster [LinWei1989]_. If
:math:`A` is the information of the partial likelihood and :math:`s_i` is
item :math:`i`'s summed score residual,

.. math::

    V_{robust} = A^{-1} \Big( \sum_i s_i s_i^{\top} \Big) A^{-1},

where the model-based covariance is :math:`A^{-1}` alone. With it the
Andersen-Gill coefficients remain valid as effects on the *mean number of
events* even when events are not Poisson [LinWeiYangYing2000]_, and it is
what makes the Wei-Lin-Weissfeld model usable at all, since there every
item appears in every stratum [WeiLinWeissfeld1989]_. SurPyval reports the
robust standard errors by default for all four models, and the model-based
ones beside them; the two being close is itself a finding (the covariates
explain the differences between items).

.. rubric:: What these models do not give

- **A formula for the baseline.** The cumulative baseline is a step
  function known only up to the last event, so it cannot be extrapolated.
  To forecast beyond the data, use the parametric proportional-intensity
  models above.
- **A simple expected count from the conditional and marginal models.**
  The Andersen-Gill cumulative baseline, scaled by the covariates, is the
  expected number of events by an age. The others give the cumulative
  hazard of one event number, not a count.
- **Later events for free.** A baseline per event number is estimated from
  the items that reached that number, which soon become few. It is usual
  to model the first few events only [PrenticeWilliamsPeterson1981]_.

Model checking
--------------

The fitted regression model carries the same diagnostics as the unconditional
intensity models, applied *per item* with each item's intensity scaled by its
own covariate factor. The time-rescaling residuals pool the rescaled
inter-arrival increments across items (i.i.d. Exp(1) under a well-specified
model), the trend test checks whether a time-varying baseline was warranted at
all, and the Cramér–von Mises test provides a bootstrapped goodness-of-fit
p-value. Confidence bounds on the fitted cumulative intensity at a covariate
setting come from the delta method via ``cif_cb``.

A few points are specific to the regression setting:

- The **martingale residuals** (observed minus expected events per item) are
  the most direct check of the covariate model. Plotted against a covariate,
  a trend in them suggests the covariate's effect is not log-linear, and
  against a covariate left *out* of the model, a trend suggests it should be
  in.
- The cumulative-hazard residuals only include the gaps that ended in an
  event; each item's final, right-censored gap is left out. With only a few
  events per item that selection pulls their average below one even when the
  model is right, so judge their *pattern* rather than insisting on a mean of
  exactly one.
- The **trend test** uses only the event times and observation windows, not
  the covariates. It answers "is there a common trend over time?", which is
  the question that separates the HPP and NHPP baselines.
- The **Cramér–von Mises** bootstrap resimulates every item from its own
  covariate-scaled intensity and refits the whole regression each time, so it
  checks the baseline shape and the covariate effects together.

These methods form a comprehensive toolkit for researchers and practitioners
working with recurrent event data, enabling detailed analysis and prediction of
event occurrences.

For worked examples — fitting the proportional-intensity HPP and NHPP models,
reading off the event-count ratio between covariate settings, and plotting the
covariate-conditional cumulative intensity with confidence bounds — see the
:doc:`Recurrent Event Regression Modelling with SurPyval` page.

A terminal event that depends on the recurrences: the joint frailty model
-------------------------------------------------------------------------

Every model above ends an item's history with a right-censored row, and
takes that end to say nothing about the item's events: the observation
stopped because the study did. Often it stopped because of an event
that ends the history for good -- a machine scrapped after one repair
too many, a patient who dies after repeated hospital admissions -- and
the items with many recurrences are the ones that reach it sooner. That
end is then informative, and a model of the recurrences alone is biased:
the items with the highest rates leave the data early, so the rate looks
lower than it is.

The **joint frailty model** [LiuWolfeHuang2004]_ [Rondeau2007]_ models
the two kinds of event together. Each item has its own unobserved
*frailty* :math:`u` -- how prone it is to events, beyond what its
covariates say -- which multiplies its rate of recurrent events, and,
raised to a power :math:`\alpha`, its hazard of the terminal event:

.. math::

    r_i(t \mid u) = u\, r_0(t)\, e^{\beta' Z_i}, \qquad
    \lambda_i(t \mid u) = u^{\alpha}\, \lambda_0(t)\, e^{\gamma' Z_i}.

In words:

- :math:`\theta`, the variance of the frailty, says how much the items
  differ: 0 is no difference beyond the covariates, and the two kinds of
  event are then independent.
- :math:`\alpha` says how the terminal event depends on that difference:
  positive when the items with more recurrences reach it sooner, 0 when
  the terminal event does not depend on the recurrences (given the
  covariates), negative when the frail items last longer.
- :math:`\beta` and :math:`\gamma` are the covariates' effects on each
  kind of event, as log rate ratios given the frailty.

The frailty is gamma with mean 1 (``family="gamma"``, the default) or
log-normal with :math:`\log u \sim N(0, \theta)`
(``family="lognormal"``). The baselines are Weibull:
:math:`R_0(t) = (t / \eta_r)^{k_r}`, the power-law (Crow-AMSAA)
intensity, and :math:`\Lambda_0(t) = (t / \eta_t)^{k_t}`. The recurrent
intensity runs on the item's age (``time_scale="calendar"``, the
Andersen-Gill form) or on the time since its last event
(``time_scale="gap"``: each event renews the baseline); the terminal
hazard is always on the age. An item seen only from an age :math:`e`
(delayed entry, ``tl``) is one that had not reached the terminal event
by then, and the likelihood divides by the chance of that, which
depends on the frailty.

The fit maximises the marginal likelihood, the frailty integrated out of
each item's likelihood,

.. math::

    L_i = \frac{\int_0^\infty \prod_j u\, r_0(t_{ij}) e^{\beta' Z_i}
    \, e^{-u R_i e^{\beta' Z_i}} \,
    \big(u^\alpha \lambda_0(X_i) e^{\gamma' Z_i}\big)^{\delta_i}
    e^{-u^\alpha \Lambda_0(X_i) e^{\gamma' Z_i}} \, dG_\theta(u)}
    {\int_0^\infty e^{-u^\alpha \Lambda_0(e_i) e^{\gamma' Z_i}}
    \, dG_\theta(u)},

with :math:`R_i` the item's cumulative baseline intensity over its
observation and :math:`\delta_i` 1 if the terminal event ended it. The
integral has no closed form unless :math:`\alpha = 1`, and is computed
in :math:`w = \log u` over each item's own range, from the integrand's
peak out to where it has fallen by :math:`e^{-36}`, by a trapezoid rule,
which on an integrand this smooth is exact to rounding. The parameters'
covariance is the inverse of the exact (automatic-derivative) Hessian;
``param_cb`` gives Wald or profile-likelihood bounds.

What the model gives:

- ``frailty_tests()``: the likelihood-ratio test of :math:`\theta = 0`
  (no frailty: the two parts fitted separately), whose p-value is that of
  an even mixture of a point mass at 0 and :math:`\chi^2_1`, since 0 is
  the edge of :math:`\theta`'s range; and of :math:`\alpha = 0` (the
  terminal event independent of the recurrences), against
  :math:`\chi^2_1`.
- ``frailties``: each item's predicted frailty, its mean given its own
  history.
- ``cif`` and ``iif``: the recurrent events' cumulative intensity and
  intensity with no terminal event to stop them; ``mcf``: the expected
  number of recurrences by an age, the terminal event stopping them,
  :math:`e^{\beta' Z} \int_0^t r_0(s)\, E[u\, S(s \mid u)]\, ds`.
- ``sf``, ``ff``, ``hf``, ``Hf``, ``df``: the terminal event's functions.
  Each prediction is for the population of items with the covariates
  (``frailty=None``, averaged over the frailty) or for one item of known
  frailty (``frailty=u``).
- ``random``: items simulated from the model, as rows ``fit`` takes.

R's ``frailtypack`` (``frailtyPenal`` with ``terminal()``) fits the same
model; SurPyval's agrees with its log-normal fits to three or four figures. Its
gamma fits use Gauss-Laguerre quadrature with fixed nodes, whose answer
moves with the number of nodes; SurPyval's are the maximum of the exact
likelihood. Not built: spline (penalised) baselines, covariates that
change over time, several terminal events, and frailties shared by
groups of items (a fleet, a hospital).

References
----------

.. [AndersenGill1982] Andersen, P.K. and Gill, R.D., 1982. Cox's regression
   model for counting processes: a large sample study. *The Annals of
   Statistics*, 10(4), pp.1100-1120.

.. [Cook2007r] Cook, R.J. and Lawless, J.F., 2007. *The Statistical Analysis of
   Recurrent Events*. Springer.

.. [Lawless1987] Lawless, J.F., 1987. Regression methods for Poisson process
   data. *Journal of the American Statistical Association*, 82(399),
   pp.808-815.

.. [Rigdon2000r] Rigdon, S.E. and Basu, A.P., 2000. *Statistical Methods for the
   Reliability of Repairable Systems*. John Wiley & Sons.

.. [LinWei1989] Lin, D.Y. and Wei, L.J., 1989. The robust inference for the
   Cox proportional hazards model. *Journal of the American Statistical
   Association*, 84(408), pp.1074-1078.

.. [LinWeiYangYing2000] Lin, D.Y., Wei, L.J., Yang, I. and Ying, Z., 2000.
   Semiparametric regression for the mean and rate functions of recurrent
   events. *Journal of the Royal Statistical Society: Series B*, 62(4),
   pp.711-730.

.. [PrenticeWilliamsPeterson1981] Prentice, R.L., Williams, B.J. and
   Peterson, A.V., 1981. On the regression analysis of multivariate failure
   time data. *Biometrika*, 68(2), pp.373-379.

.. [Therneau2000] Therneau, T.M. and Grambsch, P.M., 2000. *Modeling
   Survival Data: Extending the Cox Model*. Springer. (Chapter 8.)

.. [WeiLinWeissfeld1989] Wei, L.J., Lin, D.Y. and Weissfeld, L., 1989.
   Regression analysis of multivariate incomplete failure time data by
   modeling marginal distributions. *Journal of the American Statistical
   Association*, 84(408), pp.1065-1073.

.. [LiuWolfeHuang2004] Liu, L., Wolfe, R.A. and Huang, X., 2004. Shared
   frailty models for recurrent events and a terminal event.
   *Biometrics*, 60(3), pp.747-756.

.. [Rondeau2007] Rondeau, V., Mathoulin-Pelissier, S., Jacqmin-Gadda, H.,
   Brouste, V. and Soubeyran, P., 2007. Joint frailty models for recurring
   events and death using maximum penalized likelihood estimation:
   application on cancer events. *Biostatistics*, 8(4), pp.708-721.
