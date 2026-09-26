Recurrent Event Analysis
========================

This section is about those events that occur but do not "kill" the subject.
For example, in engineering an item may fail and be repaired instead of replaced.
In medicine, a patient may have multiple heart attacks which are not fatal.
In single-event (univariate) survival analysis one is interested in only the
time to the first, and only, event: once it happens the subject leaves the
study. In recurrent event analysis the subject stays under observation after
each event, so one is interested in the time to the first, second, third, etc.
event, and in how the *rate* of events changes as the subject ages and is
repaired or treated. We therefore need a way to describe how many events have
occurred by time :math:`t`. For a comprehensive, book-length treatment of
recurrent-event analysis, see [Cook2007]_; for the repairable-systems view
used throughout reliability engineering, see [Rigdon2000]_.

Recurrent events are modelled using counting processes and point processes.
The underlying mathematical framework — martingale theory and stochastic
integrals — is known as the counting-process formulation, but that is an
implementation detail. Users navigate by their data type: "do my subjects
experience repeated events?" If yes, this section applies.

The two broad families of recurrent-event model, which differ in their methods,
are:

    1. Recurrent Event (counting-process) Modelling
    2. Renewal Modelling

In a recurrent event model the events arrive at some underlying *rate* (an
intensity) that is a function of calendar time, whereas in a renewal model
there is an underlying *lifetime distribution* and, after each event, the
apparent age of the subject is changed by the repair or treatment. The two
families meet at their extremes: as the sections below show, "repair that
leaves the item exactly as it was" is a Poisson-process model, and "repair
that makes the item as good as new" is an ordinary renewal process.

This page explains the ideas and the mathematics. Every model described here
has a worked, runnable example on the :doc:`Recurrent Event Modelling with
SurPyval` page, and the full API is documented under
:doc:`surpyval.counting`.

Recurrent Event Modelling
-------------------------

Recurrent event models aim to find a rate at which events occur. This is done by
estimating the intensity (the recurrent-event analogue of the hazard rate) of
the process. This can be done parametrically or non-parametrically. Before
either, we need three definitions.

Counting processes, the intensity and the MCF
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**The counting process.** For a single item, let :math:`N(t)` be the number of
events it has experienced in :math:`(0, t]`. As a function of time
:math:`N(t)` is a staircase: it starts at zero and steps up by one at each
event time :math:`t_1 < t_2 < \dots`. Everything in recurrent-event analysis
is a statement about how this staircase grows.

**The intensity.** The intensity function answers the question "how likely is
an event in the next instant, given everything that has happened so far?":

.. math::

    \lambda(t \mid \mathcal{H}_{t}) = \lim_{\Delta t \to 0}
    \frac{\Pr\left(\text{event in } [t, t + \Delta t) \mid
    \mathcal{H}_{t}\right)}{\Delta t}

where :math:`\mathcal{H}_{t}` is the history of the process up to (but not
including) :math:`t` — the times of all earlier events. In reliability this is
also called the *rate of occurrence of failures* (ROCOF). It plays exactly the
role the hazard rate plays for a single event: over a short interval of length
:math:`\Delta t` the chance of an event is approximately
:math:`\lambda(t)\,\Delta t`. The difference is that the hazard rate of a
lifetime distribution describes the *first* (and only) event, whereas the
intensity keeps going after each event.

**Poisson processes.** The simplest, and most important, recurrent-event
models are those in which the intensity does *not* depend on the history —
only on the current time, :math:`\lambda(t \mid \mathcal{H}_t) = \lambda(t)`.
These are the Poisson processes. Their defining property is that the numbers
of events in disjoint time intervals are independent, and the number of
events in any interval is Poisson distributed:

.. math::

    N(b) - N(a) \sim \text{Poisson}\left(\Lambda(b) - \Lambda(a)\right),
    \qquad
    \Lambda(t) = \int_0^t \lambda(u)\, du .

:math:`\Lambda(t)` is the **cumulative intensity function** (CIF), which
SurPyval exposes as ``cif`` (and :math:`\lambda(t)` as ``iif``, the
*instantaneous* intensity function). For a Poisson process the CIF is the
expected number of events by :math:`t`: :math:`\mathbb{E}[N(t)] = \Lambda(t)`.

**The mean cumulative function.** When we observe many items we are usually
interested in the population average,

.. math::

    M(t) = \mathbb{E}\left[N(t)\right],

the **mean cumulative function** (MCF) [Nelson2003]_ — "how many repairs will
the average unit have needed by age :math:`t`?". The MCF is defined for *any*
recurrent process, not just Poisson ones, which is why the non-parametric
estimator below makes no model assumptions. For a Poisson process
:math:`M(t) = \Lambda(t)`, so the MCF and the CIF coincide and SurPyval's
parametric models answer ``mcf`` and ``cif`` with the same numbers.

.. note::

    A common pitfall is to treat the slope of the MCF as the hazard rate of a
    lifetime distribution. It is not: it is the rate at which *repeat* events
    accumulate across the population. An increasing MCF slope means events
    are arriving more often as the units age (deterioration, a "sad" system);
    a decreasing slope means they are arriving less often (reliability
    growth, a "happy" system); a straight line means the rate is constant.

Non-Parametric - Mean Cumulative Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean cumulative function is the average number of events that have occurred
by time t. It is estimated by:

.. math::

    \hat{M}(x) = \sum_{i:x_{i} \leq x}^{} \frac{d_{i} }{r_{i}}

where the :math:`x_i` are the distinct event times (pooled over all items),
:math:`d_i` is the number of events observed at :math:`x_i` and :math:`r_i`
is the number of items *at risk* (under observation) at :math:`x_i`. Each
term :math:`d_i / r_i` is the average number of events per observed item at
that instant, and the MCF simply adds these up.

This equation looks the same as the cumulative hazard estimate of the
Nelson-Aalen estimator. That's because it is, the only difference is that when
we do this for single events the number of items at risk decreases after each
failure. Whereas with recurrent events the item continues to be observed until
more events occur. Therefore the number of items in the risk set does not
always decrease after each observed event. When doing recurrent event analysis
an item will remain in the at risk set since it will be healed/repaired and
returned to service/health to continue with life. In SurPyval an item is in
the risk set from its entry time (zero, or its left-truncation time) up to and
including the end of its observation — its right-truncation time ``tr`` if it
has one, otherwise its last recorded time (its right-censoring time, or its
last event if it has no censoring row). An item therefore only leaves the risk
set when it leaves *observation*, never because it had an event.

The MCF function is the non-parametric estimator of the number of events that
will occur up to x. So once we fit a model we can then estimate the expected
number of events that will occur up to time x by using the MCF function.

SurPyval attaches the Lawless-Nadeau robust variance to :math:`\hat{M}`
[LawlessNadeau1995]_. Write :math:`\delta_k(t)` for whether item :math:`k` is
at risk at :math:`t`, :math:`n_k(t)` for its events there and
:math:`\hat{m}(t) = d(t)/r(t)` for the MCF's jump. Then

.. math::

    \widehat{\text{Var}}\,\hat{M}(t) = \sum_k \Big[ \sum_{t_j \le t}
    \frac{\delta_k(t_j)}{r(t_j)} \big( n_k(t_j) - \hat{m}(t_j) \big)
    \Big]^2 .

Each item's deviations from the average are added up over time *before* they
are squared. An item that fails more often than average does so at every
step, and those deviations reinforce each other; the variance includes that
within-item covariance, so it stays honest when items differ in their rates
and does not assume the events form a Poisson process. (A per-step variance
that squares each step on its own misses this and can be several times too
small.) With a single item there is nothing to compare the item with, and the
variance is zero.

By default the confidence bounds are computed on the log scale,
:math:`\hat{M} \exp(\pm z \sqrt{\widehat{\text{Var}}} / \hat{M})`, so that
the bounds cannot go below zero; ``bound_type="normal"`` gives the plain
:math:`\hat{M} \pm z \sqrt{\widehat{\text{Var}}}`. These are *pointwise*
bounds: each one covers the MCF at a single time, not the whole curve at once.

Non-Parametric estimation for recurrent events has the same limitations as does
single event survival analysis. The main one being that it is not possible to
extrapolate an estimate of the MCF function beyond the last observed event. This
is because when doing non-parametric analysis we make no assumptions about the
shape of the curve and cannot therefore extrapolate beyond the last observed event.
SurPyval returns ``nan`` for times beyond the last observed time (the latest
event, end-of-observation row or right-truncation time of any item). Two
further assumptions are worth stating:

- **Independent end of observation.** Items must not leave observation *because*
  they were about to have an event (or because they had many). If units with a
  bad record are withdrawn early, the remaining risk set is healthier than the
  population and the MCF is biased downwards.
- **Few items at the end.** As the risk set shrinks towards the end of
  follow-up, each event moves the MCF by :math:`1/r_i`, so the right-hand tail
  of the curve is noisy. The widening confidence bounds show this.

The non-parametric MCF currently accepts exact event times, right-censored
end-of-observation rows, left truncation (delayed entry), right truncation
and gapped observation windows. A right-truncation time ``tr`` ends the
item's observation window exactly as an end-of-observation row at ``tr``
would, the same window-close the parametric intensity models integrate to.
Interval-counted data and left-censored counts are rejected rather than
silently mishandled; use a parametric intensity model for those.

The risk-set counts :math:`d_i` and :math:`r_i` alone only support a simpler,
per-step variance, which squares each step's deviations on its own and so
treats every step as independent of every other. Assuming the :math:`d_i`
events at a time all happened to different items, a step's deviations sum to
:math:`d_i (r_i - d_i) / r_i^3`. That is the right answer for a Poisson
process but understates the uncertainty whenever items differ in their rates.
SurPyval uses it only for an MCF built directly from ``(x, r, d)`` arrays
(``NonParametricCounting.from_xrd``), which do not record which item had each
event.

Parametric Recurrent Event Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SurPyval has several parametric models for recurrent event analysis. These can
be categorised as either:

    - Homogeneous Poisson Process, or
    - Non-Homogeneous Poisson Process

The Homogeneous Poisson Process is one where the rate of events is constant over time.
This is the simplest model. A Non-Homogeneous Poisson Process is one where the
rate of events is not constant over time.

SurPyval has a series of available Non-Homogeneous Poisson Process models available
for use. These are:

    - Duane
    - Cox-Lewis
    - Crow-AMSAA

The key point is that each of these is a parametric representation of the
intensity of the process — a formula for :math:`\lambda(t)` with a small number
of parameters — and because they are Poisson processes, the intensity depends
only on time and not on the history. The formulas below use SurPyval's own
parameter names (the names printed when you fit the model), which in one case
differ from the textbook letters. Each model's API page is linked from
:doc:`surpyval.counting`.

Homogeneous Poisson Process (HPP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. math::

    \lambda(t) = \lambda, \qquad \Lambda(t) = \lambda t .

One parameter, ``lambda``, the constant event rate (events per unit time). The
times between events are independent exponential random variables with mean
:math:`1/\lambda`. The HPP is the "no trend" model: the system is neither
improving nor deteriorating. Its maximum-likelihood estimate is the intuitive
one, the total number of events divided by the total time under observation.

Crow-AMSAA (power law)
^^^^^^^^^^^^^^^^^^^^^^

.. math::

    \Lambda(t) = \left(\frac{t}{\alpha}\right)^{\beta}, \qquad
    \lambda(t) = \frac{\beta}{\alpha^{\beta}}\, t^{\beta - 1} .

This is the power-law process popularised for reliability growth by Crow
[Crow1974]_ (AMSAA is the US Army Materiel Systems Analysis Activity).

- ``beta`` is the *shape*: :math:`\beta > 1` means the intensity increases with
  time (deterioration), :math:`\beta < 1` means it decreases (reliability
  growth), and :math:`\beta = 1` is an HPP with rate :math:`1/\alpha`.
- ``alpha`` is the *scale*: it is the time by which one event is expected,
  since :math:`\Lambda(\alpha) = 1`.

The power law is the workhorse NHPP because :math:`\log \Lambda(t)` is a
straight line in :math:`\log t`, so it describes any trend that looks linear
on a log-log MCF plot. It also has a neat physical reading: its intensity is
exactly the hazard function of a Weibull distribution with the same
:math:`\alpha` and :math:`\beta`, so a Crow-AMSAA process is what you get when
a Weibull-lifetime unit is *minimally* repaired after every failure (see
`Renewal Modelling`_ below).

Duane
^^^^^

.. math::

    \Lambda(t) = b\, t^{\alpha}, \qquad
    \lambda(t) = \alpha\, b\, t^{\alpha - 1} .

Duane [Duane1964]_ observed that, during reliability-growth programmes, the
*cumulative* mean time between failures :math:`t / \Lambda(t) = t^{1-\alpha}/b`
plots as a straight line on log-log paper. The Duane model is the same
power-law process as Crow-AMSAA written differently. Note SurPyval's naming:

- ``alpha`` is the *exponent* (the role played by Crow-AMSAA's ``beta``):
  :math:`\alpha < 1` is reliability growth, :math:`\alpha > 1` deterioration;
- ``b`` is the expected number of events by :math:`t = 1`.

The two parameterisations are related by :math:`\alpha_{\text{Duane}} =
\beta_{\text{CA}}` and :math:`b = \alpha_{\text{CA}}^{-\beta_{\text{CA}}}`, so
fitting either to the same data gives the same fitted curve. Use whichever
parameters you find easier to communicate.

Cox-Lewis (log-linear)
^^^^^^^^^^^^^^^^^^^^^^

.. math::

    \lambda(t) = e^{\alpha + \beta t}, \qquad
    \Lambda(t) = \frac{e^{\alpha}}{\beta}\left(e^{\beta t} - 1\right) .

The Cox-Lewis model [CoxLewis1966]_ makes the *logarithm* of the intensity a
straight line in time.

- ``alpha`` is the log of the intensity at :math:`t = 0` (so it is negative
  whenever the starting rate is below one event per time unit);
- ``beta`` is the proportional change in the intensity per unit time:
  :math:`\beta > 0` deteriorating, :math:`\beta < 0` improving.

Unlike the power law, the Cox-Lewis intensity is finite and non-zero at
:math:`t = 0`, which suits systems that start with a definite event rate.
When :math:`\beta < 0` the cumulative intensity levels off at
:math:`e^{\alpha} / (-\beta)`: the model predicts a *finite* expected number of
events in total, and ``inv_cif`` returns ``inf`` for counts beyond that
asymptote.

Parametric recurrent event models can be estimated in similar ways to single event
survival analysis. That is, we can use a simple mean square error estimation
or, more powerfully, we can use maximum likelihood estimation. The latter is
the default in SurPyval (see `Parameter Estimation`_).

MLE for recurrent event models is done by understanding that the likelihood
between events is actually just the conditional likelihood of the next event
given the previous event. That is the equivalent to a left truncated
observation: having reached :math:`t_{k-1}`, the density of the next event at
:math:`t_k` is :math:`\lambda(t_k)\exp\{-[\Lambda(t_k) - \Lambda(t_{k-1})]\}`,
exactly the density of a lifetime with hazard :math:`\lambda` that is
left-truncated at :math:`t_{k-1}`.

The benefit of parametric models is that they can be extrapolated beyond the
last observed event. This is because we have made assumptions about the shape
of the curve and can therefore extrapolate beyond the last observed event. The
price is that the extrapolation is only as good as that assumption: a power
law and a log-linear intensity can fit the observed period equally well and
then diverge sharply. Always look at the fitted curve over the non-parametric
MCF before trusting a forecast.

Renewal Modelling
-----------------

A renewal model is one where recurrent events occur after some intervention has
restored some of the life of the subject. This is different to a recurrent event
model where the events occur at some underlying rate. In a renewal model the
events occur at some underlying distribution. After each event the apparent age
of the subject is changed.

In engineering applications a renewal process is one where the item is repaired
and not replaced. A renewal model captures the effectiveness of the repair
process. In medicine, a renewal process is one where the patient is treated to
lessen the impact of some disease, in this case a renewal model captures the
effectiveness of the treatment.

In SurPyval there are four such imperfect-repair models, which the sections
below build up in turn:

    - Generalised Renewal Process (Kijima virtual age)
    - G1 Renewal Process
    - Arithmetic Reduction of Age (ARA)
    - Arithmetic Reduction of Intensity (ARI)

Repair assumptions: perfect, minimal and imperfect
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The whole of this section rests on one question: *what state is the item in
straight after a repair?* Suppose a new item has a lifetime distribution with
survival function :math:`S(t)`, density :math:`f(t)` and hazard
:math:`h(t) = f(t)/S(t)`. There are two limiting answers.

**Perfect repair (as-good-as-new).** The repair returns the item to the state
of a brand-new one — for example, the failed unit is replaced. The times
between events are then independent draws from the same lifetime distribution:
an ordinary *renewal process*. If the lifetime is exponential (constant hazard)
this is the HPP.

**Minimal repair (as-bad-as-old).** The repair fixes only the failed part and
leaves the item exactly as it was just before the failure — same age, same
wear. The chance of the next failure then depends only on the item's total age,
so the intensity is the lifetime hazard evaluated at calendar time,
:math:`\lambda(t) = h(t)`. This is precisely a *non-homogeneous Poisson
process*: minimal repair of a Weibull unit is the Crow-AMSAA process described
above.

**Imperfect repair** sits between the two (or, occasionally, outside them): the
repair helps, but does not make the item new. The imperfect-repair models make
this precise with one or two extra parameters that measure how effective the
repair is. Fitting them answers practical questions such as "is our overhaul
worth doing?" and "how much of the benefit of a new unit does a repair give
us?". Two ideas are used to formalise "partly restored":

- reduce the item's **virtual age** — its effective age as seen by the lifetime
  distribution (Kijima models and ARA), or rescale its remaining life (G1);
- reduce the **intensity** directly (ARI).

All of these models share some practical requirements, because the state after
each repair depends on the full history of the item:

- each item must be observed **from new** (time zero), so that its virtual age
  or intensity reduction can be tracked from the start. Left truncation and
  gapped observation windows are rejected;
- the event times must be **exact**, with an optional right-censored row at the
  end of observation (``c=1``); interval and left censoring are not supported;
- the models have no closed-form mean cumulative function, so SurPyval computes
  the MCF, and the curve shown by ``plot``, by Monte Carlo simulation.

Generalised Renewal Process
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Generalised Renewal Process (GRP) is one where the intervention can cause the
item to be:

    - as-good-as-new
    - as-bad-as-old
    - better-than-old-but-worse-than-new
    - worse-than-old

This is possible since by using a notion of "virtual age." The virtual age can
be understood to be the apparent age of the subject after the nth intervention.
An engineering example is the best to illustrate this point. Consider a pump
that has been operating for 1000 hours and then has a failure. After a technician
fixes the issue, the pump is returned to service.

If the pump was repaired
perfectly it could be considered as-good-as-new and the virtual age would be
zero. That is, the apparent age of the pump would effectively be as if were straight
out of the factory.

If the pump repair was only
minimal then the virtual age would be exactly 1000 since the pump can be treated
as if it were exactly the same as it was prior to the repair, just without the
fault that was fixed.

If the pump was well repaired, but not perfectly, then the apparent age of the
pump would be somewhere between 0 and 1000.

And finally, if the intervention
repaired the fault while doing some other damage, the age of the pump could be
above 1000.

Given a virtual age :math:`V` after a repair, the next failure behaves like a
unit of age :math:`V` that has not yet failed: the time :math:`x` to the next
event has survival function :math:`S(V + x)/S(V)`. The likelihood contribution
of an observed inter-arrival time is therefore that of a left-truncated
lifetime,

.. math::

    \frac{f(V_{n-1} + x_n)}{S(V_{n-1})}
    \quad\text{(event)}, \qquad
    \frac{S(V_{n-1} + x_n)}{S(V_{n-1})}
    \quad\text{(right-censored at the end of observation)},

which is how SurPyval fits these models by maximum likelihood, using any of
its parametric lifetime distributions (Weibull by default).

A GRP has two ways that the virtual age can be calculated. These are known as
the Kijima-I and Kijima-II models [Kijima1989]_.

The Kijima-I model assumes that the repair can only repair damage incurred
since the last repair. Mathematically, this is defined as:

.. math::

    V_{n} = V_{n-1} + qt_{n}

Where :math:`V_{n}` is the virtual age after the intervention, :math:`V_{n-1}`
is the virtual age just prior to the intervention, :math:`t_{n}` is the time
since the previous event, i.e. the inter-arrival time, and finally :math:`q` is
the effectiveness of the repair. (Unrolling the recursion, Kijima-I gives
:math:`V_n = q\,T_n`, a fixed fraction of the item's total age :math:`T_n`.)

The Kijima-II model assumes that the repair can repair all damage incurred
since the item was new. Mathematically, this is defined as:

.. math::

    V_{n} = q(V_{n-1} + t_{n})

Each term here has the same meaning as the Kijima-I model.

Note that if :math:`q = 0` then the Kijima-I and Kijima-II models are
equivalent. That is the virtual age is always zero. This means that the
intervention is always perfect. If :math:`q = 1` then the Kijima-I and Kijima-II
models are also equivalent. That is the virtual age is always the same as the
actual age. This means that the intervention is always as-bad-as-old.

If :math:`q` is anything other than 0 or 1 then the models are different.
However, if :math:`0 < q < 1` then the intervention according to both models
is better-than-old-but-worse-than-new. Finally if :math:`q > 1` then the
intervention make the item worse-than-old, that is "older" than it was when
it failed.

The practical difference between the two is how much the history matters. Under
Kijima-I the virtual age keeps growing (in proportion to the total age), so an
item never escapes its past; under Kijima-II every repair discounts *all* of
the accumulated age, so with :math:`q < 1` old damage is progressively
forgotten and the virtual age does not keep growing with the item's total
age.

Both these options are available in SurPyval (``kijima="i"``, the default, or
``kijima="ii"``), which estimates :math:`q \ge 0` together with the lifetime
distribution's parameters. But both have a shortcoming which is that they cannot
handle cases where the repair makes the item better-than-new, since a virtual
age cannot be negative. To do this you will need to use the G1 Renewal Process.

G1 Renewal Process
~~~~~~~~~~~~~~~~~~

A G1 Renewal Process [Kaminskiy2010]_ is one where the intervention can be
better-than-new. The G1 renewal process is able to do this by changing the life
parameter of the underlying distribution. This is in contrast to the GRP which
alters the age whereas the G1 process alters the remaining life.

The G1 Renewal process is able to do this by changing the life parameter of the
underlying distribution. This is done by using a transformation of the life
parameter after an event. The transformation is defined as:

.. math::

    \alpha_{i} = \alpha(1 + q)^{i - 1}

Where :math:`\alpha` is the scale (life) parameter of the lifetime
distribution, :math:`\alpha_i` the one that applies to the :math:`i`-th
inter-arrival time, and :math:`q` is the effectiveness of the intervention.
Unlike what is possible with the G-Renewal process, if :math:`q` is greater
than zero the model captures an intervention that improves the life of the
subject beyond new. If
:math:`q = 0` then the repair is as-good-as-new. If :math:`q < 0` then each
repair leaves the item *worse* than before — a partial or harmful repair that
shortens the subsequent life (a deteriorating system). Note that :math:`q`
cannot be less than -1.

SurPyval implements this by scaling the whole inter-arrival time rather than a
named parameter: the :math:`i`-th inter-arrival time is the base lifetime
multiplied by :math:`c_i = (1 + q)^{i-1}`, so its density and survival
function are :math:`f(x / c_i)/c_i` and :math:`S(x / c_i)`. For a distribution
with a scale parameter (Weibull, Exponential, Gamma, ...) this is exactly the
scaling of the life parameter above, but it means any non-negative lifetime
distribution in SurPyval can be used. Distributions whose support includes
negative values (such as the Normal or Gumbel) are rejected, because a scaled
inter-arrival time must stay positive.

As an example, if the life parameter of an items was 100 hours and the repair
effectiveness was -0.2, then after the first repair the next time to an event
would have a life parameter of 80 hours. After the second repair the life
parameter would be 64 hours. After the third repair the life parameter would be
51.2 hours. And so on.

If the repair effectiveness was 0.2 then after the first repair the next time
to an event would have a life parameter of 120 hours. After the second repair
the life parameter would be 144 hours. After the third repair the life would be
172.8 hours. And so on.

The ability of the G1 Renewal Process to capture the behaviour of when the
intervention can improve the life of the subject is the reason why it is
a useful model to have available. Note, though, what it does *not* model: the
change in life is fixed by the *number* of repairs, not by how long the item
has been running, so G1 suits processes that improve or degrade step by step
with each intervention.

Arithmetic Reduction of Age and Intensity (ARA/ARI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Kijima virtual-age models above are the special cases of a more general
family due to Doyen and Gaudoin [Doyen2004]_. Rather than fold every past
repair into a single running age, these models make the *memory* of the repair
explicit: how many previous failures does an intervention act on?

Both models are parameterised by a **repair efficiency**
:math:`\rho \in (0, 1)` and an integer **memory** :math:`m \ge 1` (or
:math:`m = \infty`). Throughout, :math:`T_1 < T_2 < \dots` are the item's
failure times.

The **Arithmetic Reduction of Age** (ARA) model reduces the virtual age.
Immediately after the :math:`n`-th repair the virtual age is

.. math::

    V_n = T_n - \rho \sum_{j=0}^{\min(m, n) - 1} (1 - \rho)^{j}\, T_{n-j},

and until the next failure the item ages from there, so its intensity is the
lifetime hazard evaluated at the virtual age,
:math:`\lambda(t) = h\left(V_n + t - T_n\right)` for
:math:`T_n < t \le T_{n+1}`. The likelihood is the same left-truncated form as
the Kijima models.

- :math:`\rho = 1` gives :math:`V_n = 0`: as-good-as-new (perfect repair).
- :math:`\rho = 0` gives :math:`V_n = T_n`: as-bad-as-old (minimal repair), i.e.
  the NHPP whose intensity is the lifetime hazard.
- With memory :math:`m = 1` (ARA\ :sub:`1`) each repair removes a fraction
  :math:`\rho` of the wear accumulated *since the previous repair*; this is
  exactly Kijima-I with :math:`q = 1 - \rho`.
- With infinite memory (ARA\ :sub:`∞`) each repair removes a fraction
  :math:`\rho` of the *whole* current virtual age; this is exactly Kijima-II
  with :math:`q = 1 - \rho`.
- Finite memories :math:`m \ge 2` are the genuinely new cases: the repair
  reaches back over the last :math:`m` inter-arrival periods and no further.

So the repair efficiency :math:`\rho` plays the role of :math:`1 - q`. Because
:math:`\rho` is restricted to :math:`(0, 1)`, the ARA models cover the range
from as-bad-as-old to as-good-as-new; unlike Kijima with :math:`q > 1` they do
not represent worse-than-old repair.

The **Arithmetic Reduction of Intensity** (ARI) model instead reduces the
*intensity* directly. It starts from a *baseline intensity*
:math:`\lambda_0(t)` — the intensity the item would have under minimal repair,
which in SurPyval is one of the NHPP models (Crow-AMSAA by default, or Duane or
Cox-Lewis) — and after the :math:`n`-th repair subtracts a memory-weighted
fraction of the intensity at the recent failures:

.. math::

    \lambda(t) = \lambda_0(t) - \rho \sum_{j=0}^{\min(m, n) - 1}
    (1 - \rho)^{j}\, \lambda_0(T_{n-j}),
    \qquad T_n < t \le T_{n+1}.

So the process acts on the rate of events rather than on an effective age.
The log-likelihood is the general point-process one,
:math:`\sum_k \ln \lambda(T_k) - \int \lambda(u)\,du`, evaluated with this
reduced intensity. :math:`\rho = 0` recovers the baseline NHPP (minimal
repair). As with ARA, :math:`m = 1` makes each repair act only on the most
recent failure and :math:`m = \infty` on the whole history.

ARA and ARI therefore answer the same question in different currencies —
"how much younger does the repair make the item?" versus "how much lower does
the repair make the failure rate?" — and they generally give different
predictions. Both are designed for *deteriorating* systems, where there is
wear for the repair to remove. With an increasing baseline intensity the
reduced ARI intensity stays positive; with a decreasing baseline the
subtraction can drive it to zero or below, which is outside the model (the
likelihood is infinite there), so ARI is not a natural choice for
reliability-growth data.

The memory :math:`m` is not estimated — it is a modelling choice you pass in,
which lets you dial the model from "the last repair undid only the most recent
wear" (:math:`m = 1`) up to "every repair reaches back over the whole history"
(:math:`m = \infty`). A practical approach is to fit a few values and compare
their information criteria, as shown on the how-to page.

The Geometric Process
~~~~~~~~~~~~~~~~~~~~~

A closely related idea is Lam's **geometric process** [Lam1988]_. Here the
successive inter-arrival times :math:`X_1, X_2, \ldots` are scaled so that
:math:`a^{\,k-1} X_k` are independent and identically distributed for a single
ratio :math:`a > 0`. When :math:`a > 1` the inter-arrivals shrink
geometrically (a deteriorating system), when :math:`a < 1` they grow
(reliability growth), and :math:`a = 1` is an ordinary renewal process. The
mean inter-arrival time is then the geometric sequence
:math:`\mathbb{E}[X_k] = \mu / a^{\,k-1}`.

This is exactly the G1 Renewal Process in a different parameterisation:
matching the G1 scaling :math:`(1 + q)^{j}` to :math:`a^{-j}` gives
:math:`a = 1 / (1 + q)`. A deteriorating system (:math:`a > 1`) therefore
corresponds to a negative restoration factor, and reliability growth
(:math:`a < 1`) to a positive one. Fitting a
:doc:`GeneralizedOneRenewal <counting/g1_rp>` gives the geometric process
with a parametric lifetime distribution.


Parameter Estimation
--------------------

As with regular survival analysis there are several ways one can estimate the
parameters of the models. Mean Square Error (MSE) is quite straight forward
for the intensity (Poisson-process) models. For the NHPP models (Crow-AMSAA,
Duane, Cox-Lewis) SurPyval's ``how="MSE"`` option chooses the parameters that
make :math:`\Lambda(t)` pass as closely as possible (in the least-squares
sense) through the non-parametric MCF at the observed times,

.. math::

    \hat{\theta}_{\text{MSE}} = \arg\min_{\theta} \sum_i
    \left[\Lambda(x_i \mid \theta) - \hat{M}(x_i)\right]^2 .

This is fast and needs no distributional assumptions about the counts, and
SurPyval uses it as the starting point for the likelihood search of these
models. It does not produce a likelihood, so the likelihood-based inference
(AIC/BIC, standard errors, confidence bounds) is not available for an MSE
fit. The HPP has no MSE option: its maximum-likelihood estimate is already
the simple ratio of events to exposure. For the
renewal models the complication is that the MCF has no closed form, so an MSE
fit would need a Monte Carlo simulation for each set of parameters. This can
get quite time consuming and expensive; SurPyval fits the renewal models by
maximum likelihood only.

Maximum Likelihood Estimation also provides an excellent way to estimate
parameters, and it is the default. It is a relatively straight forward logical
step from single event survival analysis to multiple events. For the first
event, the likelihood of that particular event is the same as for regular
survival analysis. For recurrent events we tend to use the intensity, as this
captures the full nature of the circumstances, to define the model.

The intensity-model likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The log-likelihood for a non-homogeneous Poisson process observed over
:math:`[0, T]` with events at times :math:`x_1, x_2, \ldots, x_n` is:

.. math::

    \ell(\theta) = \sum_{i=1}^{n} \ln \lambda(x_i \mid \theta) - \int_0^T \lambda(u \mid \theta)\, du

where :math:`\lambda(t \mid \theta)` is the intensity function of the
process and the integral, :math:`\Lambda(T \mid \theta)`, is the expected
number of events over the observation window. For a homogeneous Poisson
process :math:`\lambda` is constant and the integral reduces to
:math:`\lambda T`.

The two terms pull in opposite directions, which is what makes the estimate
sensible: the first rewards a high intensity *where events happened*, the
second (the *compensator*) penalises expecting many events over the whole
window. It is the sum, over the inter-event intervals, of the left-truncated
contributions described in `Parametric Recurrent Event Models`_. With several
items the log-likelihood is the sum of each item's contribution over its own
window.

Time- versus failure-truncated observation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The upper limit :math:`T` of the integral is *when observation stopped*, and it
matters how it was decided:

- **Time-truncated** (time-terminated) data: each item is watched for a fixed
  period and observation ends at :math:`T` whether or not an event happens
  then. In SurPyval you record this with a right-censored row (``c=1``) at
  :math:`T` as the last row of the item (or with a right-truncation time
  ``tr``, see below).
- **Failure-truncated** (failure-terminated) data: observation stops at an
  event — for example, a test that runs until the :math:`n`-th failure. Then
  :math:`T = x_n`, and in SurPyval you simply provide the events with no
  censoring row: the integral stops at the last event.

The distinction changes the estimates. For the Crow-AMSAA process the MLEs are
available in closed form,

.. math::

    \hat{\beta} = \frac{n}{\sum_{i} \ln (T / x_i)}, \qquad
    \hat{\alpha} = \frac{T}{n^{1/\hat{\beta}}},

where for time-truncated data the sum runs over all :math:`n` events, and for
failure-truncated data :math:`T = x_n` and the sum effectively runs over the
first :math:`n-1` events (the :math:`n`-th term is :math:`\ln 1 = 0`).
Forgetting the censoring row when the observation actually ran on past the
last event therefore shortens the window, making the process look more
intense (and more strongly increasing) than it is — the most common data
preparation error in repairable-systems analysis.

Grouped (interval) counts
~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes the exact event times are unknown and only counts are recorded:
":math:`n_k` failures were found at the inspection covering
:math:`(a_k, b_k]`". Because a Poisson process has independent Poisson
counts, each such record contributes

.. math::

    n_k \ln \left[\Lambda(b_k) - \Lambda(a_k)\right]
    - \left[\Lambda(b_k) - \Lambda(a_k)\right] - \ln n_k! ,

the Poisson log-probability of the observed count. SurPyval's HPP and NHPP
models accept these as interval-censored rows (``c=2`` with ``x`` given as
``[a, b]`` and the count in ``n``), and a count of events in
:math:`(0, b]` as a left-censored row (``c=-1``). Mixed exact and grouped
records are combined in the same likelihood. The renewal models and the
non-parametric MCF do not accept grouped counts.

Imperfect-repair likelihoods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The renewal models are fitted in the same spirit, but each inter-arrival time
is conditioned on the state left by the previous repair: the left-truncated
lifetime contributions shown in `Generalised Renewal Process`_ for Kijima and
ARA (with the virtual age computed from the item's history and the current
value of the repair parameter), the rescaled lifetime for G1, and the reduced
intensity for ARI. SurPyval maximises these jointly over the repair parameter
and the lifetime (or baseline intensity) parameters, restarting the optimiser
from several values of the repair parameter because the likelihood surface can
have more than one local optimum. The repair parameter is reported as ``q``
(Kijima, G1) or ``rho`` (ARA, ARI).

Uncertainty: standard errors and confidence bounds
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every model fitted by maximum likelihood carries the usual large-sample
machinery. The covariance of the estimates is approximated by the inverse of
the observed information (the Hessian of the negative log-likelihood at the
optimum), which gives standard errors. Confidence bounds on a parameter are
Wald intervals computed on a scale chosen to respect the parameter's range —
log scale for a positive parameter, logit scale for one confined to an
interval such as :math:`\rho \in (0, 1)` — so the bounds never leave the
allowed range. Confidence bounds on the fitted cumulative intensity are
obtained by the delta method: with :math:`\hat{\Sigma}` the parameter
covariance and :math:`\nabla_\theta \Lambda` the gradient of the cumulative
intensity with respect to the parameters,

.. math::

    \widehat{\text{SE}}\big[\hat{\Lambda}(t)\big] \approx
    \sqrt{\nabla_\theta \Lambda(t)^{\top}\, \hat{\Sigma}\,
    \nabla_\theta \Lambda(t)},
    \qquad
    \hat{\Lambda}(t) \exp\left(\pm z\,
    \widehat{\text{SE}} / \hat{\Lambda}(t)\right),

again on the log scale so the bounds stay positive (the same construction as
the default MCF bounds).

Two warnings apply. When an estimate sits on the edge of its range — a repair
parameter estimated as exactly perfect or minimal repair is the usual case —
the normal approximation does not hold; SurPyval warns when it cannot
compute a variance and reports the affected standard errors as NaN, but even
a finite standard error at a boundary estimate should not be trusted. And with few events the approximations
are rough in any case: treat the bounds as a guide.

Prediction: expected counts and future counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two different questions are easily confused when forecasting:

1. *How many events do we expect?* This is the cumulative intensity (or MCF),
   and its uncertainty is the confidence band on :math:`\Lambda(t)` described
   above. With enough data this band shrinks towards zero width.
2. *How many events will actually occur?* Even if the parameters were known
   exactly, the count is random. For a Poisson process the number of events
   in a future window :math:`(t_1, t_2]` is
   :math:`\text{Poisson}(\Lambda(t_2) - \Lambda(t_1))` whatever happened
   before :math:`t_1`, so a *prediction interval* for the count comes from
   the Poisson quantiles. This interval does not shrink to zero width, however
   much data you have. Plugging in the fitted :math:`\hat{\Lambda}` ignores
   the parameter uncertainty, so with little data it is somewhat too narrow.

SurPyval does not have a dedicated prediction-interval method; the how-to page
shows how to compute the Poisson interval from a fitted model's ``cif`` in a
couple of lines. For the renewal models the future count depends on the
item's history, so there is no simple formula; simulate from the fitted model
instead.

Truncation and Delayed Entry
----------------------------

Observation of a recurrent process rarely starts at the origin. A machine may
already have been in service before monitoring began, or a study may only
record events after a subject enrols. This is **left truncation** (delayed
entry): the item is only under observation from an entry time :math:`t_L`, and
its first observed interval is integrated from :math:`t_L` rather than from
zero. **Right truncation** closes the observation window at a time :math:`t_R`,
extending the compensator integral out to :math:`t_R` even if no event or
censoring row sits exactly there. (Closing the window with ``tr`` and closing
it with a right-censored row at the same time give the same likelihood.)

The intensity (Poisson) models handle delayed entry directly, because the
likelihood over any interval depends only on the intensity over that interval.
The virtual-age and history-dependent models (Kijima, G1, ARA, ARI) cannot:
the virtual age at entry depends on the unobserved failures before entry, so
those models require the process to be observed from the start.

The non-parametric MCF handles both through the risk set — an item is only
counted at risk from its entry time, so early event times are averaged over
the items actually being watched, and it stays at risk up to its
right-truncation time, just as it would up to a right-censored row there.
The trend tests (below) assume every item is observed from time zero.

Model Checking: Residuals, Trend Tests, and Goodness of Fit
-----------------------------------------------------------

Having a fitted model is not the same as having a *good* model, and recurrent
processes admit the same kind of residual analysis as ordinary regression.

The key tool is the **time-rescaling theorem** [Ogata1988]_ [DaleyVereJones2003]_.
If events follow a process with cumulative intensity :math:`\Lambda`, then
transforming each event time by its own compensator turns the observed events
into a *unit-rate* Poisson process. Concretely, the rescaled inter-arrival
increments

.. math::

    e_k = \Lambda(x_k) - \Lambda(x_{k-1})

are independent Exp(1) random variables when the fitted model is correct. So a
simple check is whether the residuals :math:`e_k` look like an i.i.d. Exp(1)
sample (mean one); the probability-integral transform :math:`1 - e^{-e_k}`
turns them into U(0, 1) values for a QQ-style check, and a per-item
**martingale residual** — observed event count minus the compensator over the
item's window — flags items the model over- or under-predicts. For the
virtual-age and renewal models the same construction is applied to the
*conditional* intensity (the cumulative hazard accumulated over each interval
given the model's virtual age or intensity reduction), so residuals extend to
those families too.

One practical caution: the residuals :math:`e_k` are only computed for gaps
that *ended in an event*. For time-truncated data each item's final gap, from
its last event to the end of observation, is censored and left out, and the
gaps that are kept are (by selection) the shorter ones. When items have many
events this hardly matters; with only a handful per item it pulls the mean of
the residuals noticeably below one even for a correct model. Look for
patterns — residuals drifting with time, or a few very large values — rather
than insisting on a mean of exactly one, and use the goodness-of-fit test
below for a calibrated verdict.

Trend tests
~~~~~~~~~~~

A **trend test** asks a more basic question: was a time-varying intensity
warranted at all? The null hypothesis is a homogeneous Poisson process (no
trend). The Laplace test and the Military-Handbook (MIL-HDBK-189C) test
[Rigdon2000]_ both use only the event times and observation windows, not the
fitted parameters, so they are a useful sanity check before committing to a
particular parametric form.

The intuition behind the **Laplace test** is that, under an HPP and given the
number of events, the event times are scattered uniformly over the observation
window, so their average should sit near the middle of the window. For systems
:math:`q` observed on :math:`(0, T_q]` with :math:`n_q` events at
:math:`t_{qj}`,

.. math::

    U = \frac{\sum_q \sum_j t_{qj} - \sum_q n_q T_q / 2}
             {\sqrt{\sum_q n_q T_q^2 / 12}}

is approximately standard normal under the null. Events bunched late
(:math:`U > 0`) point to an increasing intensity, events bunched early
(:math:`U < 0`) to a decreasing one.

The **MIL-HDBK-189C test** is built from the power-law process:

.. math::

    \chi^2 = 2 \sum_q \sum_j \ln\left(\frac{T_q}{t_{qj}}\right)

is chi-squared with :math:`2N` degrees of freedom under the HPP (:math:`N`
events in total). It equals :math:`2N/\hat{\beta}` for the Crow-AMSAA shape
estimate, so a *small* statistic (:math:`\hat{\beta} > 1`) indicates an
increasing intensity and a large one a decreasing intensity. It is the more
powerful test when the true trend is a power law; the Laplace test is the more
powerful when it is log-linear (Cox-Lewis-like).

For failure-truncated data the last event of each system *is* the end of its
window rather than a random event, so it is dropped from both statistics. Both
tests can be run one-sided (``"increasing"`` or ``"decreasing"``) when you
only care about one direction. Two cautions: the tests assume all systems are
observed from time zero (SurPyval refuses delayed-entry and gapped data), and
the direction a test reports is simply which side of its null value the
statistic fell (the sign of :math:`U`; :math:`\chi^2` below or above its
:math:`2N` degrees of freedom) — only the p-value tells you whether that
direction is distinguishable from noise. Both are available as standalone
functions and as a ``trend_test`` method on every fitted model; see
:doc:`counting/trend_tests`.

Goodness of fit
~~~~~~~~~~~~~~~

Finally, a **Cramér–von Mises** goodness-of-fit test [DaleyVereJones2003]_
measures how far the conditionally-uniform transforms fall from uniformity.
For a Poisson process observed over a fixed window
:math:`(s, \tau]`, and conditional on the number of events the item shows
there, the normalised compensators

.. math::

    u_k = \frac{\Lambda(x_k) - \Lambda(s)}{\Lambda(\tau) - \Lambda(s)}

are i.i.d. U(0, 1) under the true model. For a failure-truncated item the
window ends at its last event, which is left out (the remaining events,
normalised by the compensator there, are uniform in the same way). The
statistic pools the :math:`u_k` of all items, sorts them into
:math:`u_{(1)} \le \dots \le u_{(M)}`, and measures their departure from
uniformity,

.. math::

    C_M^2 = \frac{1}{12M} + \sum_{j=1}^{M}
    \left(u_{(j)} - \frac{2j - 1}{2M}\right)^2 .

A large value means the events are not spread over time the way the fitted
intensity says they should be. Because the parameters were estimated from the
same data, the p-value is obtained by a parametric bootstrap — resimulating
every item over its own window from the fitted model, refitting, and
recomputing the statistic — so it accounts for the estimation. For the
power-law (Crow-AMSAA) process this is the construction behind Crow's
classical goodness-of-fit test.

For the imperfect-repair models the compensator is the running sum of the
conditional (virtual-age or reduced-intensity) increments described above.
Uniformity is then exact for failure-truncated items and only approximate for
time-truncated ones, whose window close is itself history-dependent. The
bootstrap resimulates each item the way it was observed — a failure-truncated
item with its observed number of events, a time-truncated one over its window
to the same end-of-observation time, with however many events the model gives
it there — so the bootstrap statistics share that approximation and the
p-value accounts for it. Every bootstrap replicate is a full refit, so the
test is slow for the imperfect-repair models; a small number of replicates
gives only a coarse p-value (with :math:`B` replicates the smallest possible
p-value is :math:`1/(B + 1)`).

Choosing a model
~~~~~~~~~~~~~~~~

There is no single right model, but a sensible order of work is:

1. **Plot the non-parametric MCF.** Its shape is the evidence: a straight line
   suggests a constant rate, a curve bending up suggests deterioration, a curve
   bending down reliability growth. Items that behave very differently from the
   rest show up here too.
2. **Test for a trend.** If neither trend test rejects the HPP, the HPP is the
   simplest defensible model, and extra parameters are likely to be fitting
   noise.
3. **If there is a trend, fit an NHPP.** Start with the power law (Crow-AMSAA
   or Duane). Try Cox-Lewis if the rate is clearly non-zero at the start or the
   MCF looks exponential rather than power-like. Compare the fits with AIC/BIC
   and, above all, by plotting them over the MCF.
4. **If the repair process itself is the question** — does a repair restore
   the unit, and by how much? — and each item's history is known from new, fit
   the imperfect-repair models. Compare the Kijima types, the G1 process and
   several ARA/ARI memories by their information criteria; all are fitted to
   the same event times by maximum likelihood, so their AIC values are on a
   common footing, and so are their BIC values: every recurrent model takes
   BIC's sample size to be the number of observed events -- exact ones plus
   the events in left- and interval-censored counts, with
   end-of-observation rows not counting -- the rule of every SurPyval model
   (see :ref:`information-criteria`). Check the winner with residuals.
5. **If events come from several distinct mechanisms,** analyse them per
   cause (next section); **if items differ systematically** (environment, duty
   cycle, design version), move to the regression models on the
   :doc:`Recurrent Event Regression Analysis` page.

Remember that AIC only ranks the models you tried: it cannot tell you that all
of them are poor. That is the job of the residuals and the goodness-of-fit
test.

Competing Risks: Marked Recurrent Events
----------------------------------------

An item can experience events of several *mutually exclusive types* — a pump
that suffers seal failures, bearing failures and impeller failures, say. Each
event carries a **mark** identifying its type, and we usually want a separate
picture per type.

Non-parametrically, the **cause-specific mean cumulative function** is the MCF
restricted to one cause. The at-risk set is shared across causes (an item is at
risk for every cause until it leaves observation); only the event counts are
split by type:

.. math::

    \hat{M}_k(x) = \sum_{i: x_i \le x} \frac{d_{ik}}{r_i},

where :math:`d_{ik}` counts the events of cause :math:`k` at :math:`x_i`. When
every event carries a cause, the cause-specific MCFs add up to the overall MCF.
This is the recurrent-process analogue of the cause-specific cumulative
incidence in single-event competing risks. There is one simplification:
because an event of one type does not remove the item from observation, an
event of another type can still follow it, so no adjustment for the other
causes is needed.

Each cause-specific MCF carries the Lawless-Nadeau robust variance (see
`Non-Parametric - Mean Cumulative Function`_) of that cause's events: an event
of another cause counts as a non-event, while the item stays in the shared
risk set. So its confidence bounds, like the overall MCF's, allow for items
differing in their rates of that cause.

Parametrically, a marked Poisson process has an elegant structure: the
cause-specific processes are **independent thinned Poisson processes**. An
event of one cause neither advances nor interrupts another cause's intensity,
so the joint likelihood factorises over causes. Each cause's intensity is
therefore just the ordinary NHPP fit to that cause's events over the full
observation window of every item, treating other-cause events exactly as a
censored (unobserved) period. The total intensity is the sum of the
cause-specific intensities.

This structure relies on the process being a Poisson process. If a repair
after one cause changes the risk of another (a seal replacement that also
disturbs the bearing, say), the causes are no longer independent and the
cause-specific fits describe the marginal rates only.

Gapped (Multi-Window) Observation
---------------------------------

Sometimes an item is observed over several *disjoint* windows with unobserved
gaps in between — a fleet vehicle tracked only while it is in the depot, or a
system monitored during business hours. Events may occur during a gap but are
never recorded, so the analysis must not assume the item was under observation
throughout.

For a Poisson (intensity) process this has a clean solution. Because event
counts over disjoint windows are independent, a gapped item's likelihood
factorises over its windows, and each window can be treated as its own
observation period with its own entry and exit. The intensity likelihood and
the non-parametric MCF at-risk set then handle the gaps with no special
machinery: an item is simply absent from the risk set while it is unobserved.
The virtual-age and renewal models cannot accommodate gaps, because the virtual
age at the start of a later window depends on the unobserved failures during
the gap.

Because SurPyval implements the gaps by treating each window as its own
observation period, anything computed per item is computed per *window* for
gapped data: the martingale residuals come one per window, and the trend
tests, which need every item watched from time zero, refuse gapped data. The
covariate (regression) and cause-specific models do not accept windows.

For worked examples — fitting the NHPP, HPP and renewal models, estimating and
plotting the mean cumulative function, and handling gapped observation — see the
:doc:`Recurrent Event Modelling with SurPyval` page. For covariate models see
the :doc:`Recurrent Event Regression Analysis` and
:doc:`Recurrent Event Regression Modelling with SurPyval` pages.

References
----------

.. [Doyen2004] Doyen, L. and Gaudoin, O., 2004. Classes of imperfect repair
   models based on reduction of failure intensity or virtual age. *Reliability
   Engineering & System Safety*, 84(1), pp.45-56.

.. [Lam1988] Lam, Y., 1988. Geometric processes and replacement problem. *Acta
   Mathematicae Applicatae Sinica*, 4(4), pp.366-377.

.. [Ogata1988] Ogata, Y., 1988. Statistical models for earthquake occurrences
   and residual analysis for point processes. *Journal of the American
   Statistical Association*, 83(401), pp.9-27.

.. [DaleyVereJones2003] Daley, D.J. and Vere-Jones, D., 2003. *An Introduction
   to the Theory of Point Processes, Volume I: Elementary Theory and Methods*,
   2nd ed. Springer.

.. [Rigdon2000] Rigdon, S.E. and Basu, A.P., 2000. *Statistical Methods for the
   Reliability of Repairable Systems*. John Wiley & Sons.

.. [Cook2007] Cook, R.J. and Lawless, J.F., 2007. *The Statistical Analysis of
   Recurrent Events*. Springer.

.. [LawlessNadeau1995] Lawless, J.F. and Nadeau, C., 1995. Some simple robust
   methods for the analysis of recurrent events. *Technometrics*, 37(2),
   pp.158-168.

.. [Nelson2003] Nelson, W.B., 2003. *Recurrent Events Data Analysis for Product
   Repairs, Disease Recurrences, and Other Applications*. ASA-SIAM Series on
   Statistics and Applied Probability. SIAM.

.. [Crow1974] Crow, L.H., 1974. Reliability analysis for complex, repairable
   systems. In Proschan, F. and Serfling, R.J. (eds), *Reliability and
   Biometry: Statistical Analysis of Lifelength*, pp.379-410. SIAM.

.. [Duane1964] Duane, J.T., 1964. Learning curve approach to reliability
   monitoring. *IEEE Transactions on Aerospace*, 2(2), pp.563-566.

.. [CoxLewis1966] Cox, D.R. and Lewis, P.A.W., 1966. *The Statistical Analysis
   of Series of Events*. Methuen.

.. [Kijima1989] Kijima, M., 1989. Some results for repairable systems with
   general repair. *Journal of Applied Probability*, 26(1), pp.89-102.

.. [Kaminskiy2010] Kaminskiy, M.P. and Krivtsov, V.V., 2010. G1-renewal process
   as repairable system model. *Reliability: Theory & Applications*, 1(3)
   (issue 18), pp.7-14. arXiv:1006.3718.
