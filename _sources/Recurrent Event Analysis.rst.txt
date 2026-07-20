Recurrent Event Analysis
========================

This section is about those events that occur but do not "kill" the subject.
For example, in engineering an item may fail and be repaired instead of replaced.
In medicine, a patient may have multiple heart attacks which are not fatal.
In these cases, the event of interest is the time between events, which is the
subject of single-event survival analysis. In single-event (univariate) analysis
one is interested in only the time to the first, and only, event. In recurrent
event analysis one is interested in the time to the first, second, third, etc.
event. We therefore need a way to describe how many events have occurred by time t.
For a comprehensive, book-length treatment of recurrent-event analysis, see
[Cook2007]_.

Recurrent events are modelled using counting processes and point processes.
The underlying mathematical framework — martingale theory and stochastic
integrals — is known as the counting-process formulation, but that is an
implementation detail. Users navigate by their data type: "do my subjects
experience repeated events?" If yes, this section applies.

This section is split into two subsections:

    1. Recurrent Event Modelling
    2. Renewal Modelling

These are both types of recurrent event models but they differ in their methods.
In a recurrent event model the inter-arrival time is based on some underlying
hazard rate, whereas in a renewal model there is an underlying distribution where,
after each event, the apparent age of the subject is changed.

Recurrent Event Modelling
-------------------------

Recurrent event models aim to find a rate at which events occur. This is done by
estimating the hazard rate of the process. This can be done parametrically or
non-parametrically.

Non-Parametric - Mean Cumulative Function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mean cumulative function is the average number of events that have occurred
by time t. It is defined as:

.. math::

    MCF(x) = \sum_{i:x_{i} \leq x}^{} \frac{d_{i} }{r_{i}}

This equation looks the same as the cumulative hazard estimate of the Nelson-Aalen estimator. 
That's because it is, the only difference is that when we do this for single events
the number of items at risk decreases after each failure. Whereas with recurrent
events the item continues to be observed until more events occur. Therefore the 
number of items in the risk set does not always decrease after each observed event.
Whereas when doing recurrent event analysis an item will remain in the at risk
set since it will be healed/repaired and returned to service/health to continue with life.

The MCF function is the non-parametric estimator of the number of events that
will occur up to x. So once we fit a model we can then estimate the expected
number of events that will occur up to time x by using the MCF function.

Non-Parametric estimation for recurrent events has the same limitations as does
single event survival analysis. The main one being that it is not possible to
extrapolate an estimate of the MCF function beyond the last observed event. This
is because when doing non-parametric analysis we make no assumptions about the
shape of the curve and cannot therefore extrapolate beyond the last observed event.


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
    - AMSAA
    - Crow-AMSAA

See the API documentation to see the details of each of these models. The key
point is that each of these are a parametric representation of the hazard rate
of the process.

Parametric recurrent event models can be estimated in similar ways to single event
survival analysis. That is, we can use a simple mean square error estimation
or, more powerfully, we can use maximum likelihood estimation. The latter is
the default in SurPyval.

MLE for recurrent event models is done by understanding that the likelihood
between events is actually just the conditional likelihood of the next event
given the previous event. That is the equivalent to a left truncated
observation.

The benefit of parametric models is that they can be extrapolated beyond the
last observed event. This is because we have made assumptions about the shape
of the curve and can therefore extrapolate beyond the last observed event.

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

In surpyval there are two types of renewal models, these are:

    - Generalised Renewal Process
    - G1 Renewal Process

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

A GRP has two ways that the virtual age can be calculated. These are known as
the Kijima-I and Kijima-II models.

The Kijima-I model assumes that the repair can only repair damage incurred
since the last repair. Mathematically, this is defined as:

.. math::

    V_{n} = V_{n-1} + qt_{n}

Where :math:`V_{n}` is the virtual age after the intervention, :math:`V_{n-1}`
is the virtual age just prior to the intervention, :math:`t_{n}` is the time
since the previous event, i.e. the inter-arrival time, and finally :math:`q` is
the effectiveness of the repair.

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
However, if :math:`0 < q < 1` then the interventionm according to both models
is better-than-old-but-worse-than-new. Finally if :math:`q > 1` then the
intervention make the item worse-than-old, that is "older" than it was when
it failed.

Both these options are available in SurPyval. But both have a shortcoming which
is that they cannot handle cases where the repair makes the item
better-than-new. To do this you will need to use the G1 Renewal Process.

G1 Renewal Process
~~~~~~~~~~~~~~~~~~

A G1 Renewal Process is one where the intervention can be better-than-new. The
G1 renewal process is able to do this by changing the life parameter of the
underlying distribution. This is in contrast to the GRP which alters the age
whereas the G1 process alters the remaining life.

The G1 Renewal process is able to do this by changing the life parameter of the
underlying distribution. This is done by using a transformation of the life
parameter after an event. The transformation is defined as:

.. math::

    \alpha_{i} = \alpha(1 + q)^{i - 1}

Where :math:`\alpha` is the life parameter of a location-scale distribution and
:math:`q` is the effectiveness of the intervention. Unlike what is possible 
with the G-Renewal process, if q is greater than zero the model captures the
behaviour of when the intervention can improves the life of the subject. If 
:math:`q = 0` then the repair is as-good-as-new. If :math:`q < 0` then the
intervention is restoring some life but not to make it as good as new. Note
that :math:`q` cannot be less than -1.

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
a useful model to have available.

Arithmetic Reduction of Age and Intensity (ARA/ARI)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Kijima virtual-age models above are the special cases of a more general
family due to Doyen and Gaudoin [Doyen2004]_. Rather than fold every past
repair into a single running age, these models make the *memory* of the repair
explicit: how many previous failures does an intervention act on?

The **Arithmetic Reduction of Age** (ARA) model reduces the virtual age by a
fixed fraction :math:`\rho` of the age accumulated over the last :math:`m`
inter-arrival times. With infinite memory (:math:`m = \infty`) and one prior
failure this recovers Kijima-I; with :math:`m = 1` it acts only on the most
recent interval. The repair efficiency :math:`\rho \in (0, 1)` plays the role
of :math:`1 - q`: :math:`\rho = 1` is as-good-as-new (perfect repair) and
:math:`\rho = 0` is as-bad-as-old (minimal repair).

The **Arithmetic Reduction of Intensity** (ARI) model instead reduces the
*intensity* directly. After each repair the failure intensity is lowered by a
fraction :math:`\rho` of the intensity built up over the last :math:`m`
intervals, so the process acts on the rate of events rather than on an
effective age. ARA and ARI coincide for a constant (homogeneous) baseline but
differ whenever the baseline intensity varies with time.

Both models take an integer memory :math:`m` (with :math:`m = \infty` the
infinite-memory limit), which lets you dial the model from "the last repair
undid only the most recent wear" (:math:`m = 1`) up to "every repair reaches
back over the whole history" (:math:`m = \infty`).

The Geometric Process
~~~~~~~~~~~~~~~~~~~~~~~

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
:class:`~surpyval.recurrent.GeneralizedOneRenewal` gives the geometric process
with a parametric lifetime distribution.


Parameter Estimation
--------------------

As with regular survival analysis there are several ways one can estimate the
parameters of the models. Mean Square Error (MSE) is quite straight forward
for both renewal and recurrence models. The complication with recurrence models
is that you have to do a monte carlo simulation for each set of parameters. This
can get quite time consuming and expensive. But it does work.

Maximum Likelihood Estimation also provides an excellent way to estimate
parameters. It is a relatively straight forward logical step from single
event survival analysis to multiple events. For the first event, the likelihood
of that particular event is the same as for regular survival analysis. For
recurrent events we tend to use the hazard rate, as this captures the full
nature of the circumstances, to define the model. The log-likelihood for a
non-homogeneous Poisson process observed over :math:`[0, T]` with events at
times :math:`x_1, x_2, \ldots, x_n` is:

.. math::

    \ell(\theta) = \sum_{i=1}^{n} \ln h(x_i \mid \theta) - \int_0^T h(u \mid \theta)\, du

where :math:`h(t \mid \theta)` is the intensity (hazard) function of the
process and the integral is the expected number of events over the observation
window. For a homogeneous Poisson process :math:`h` is constant and the
integral reduces to :math:`\lambda T`.

.. note::

    Full derivations for each parametric model (Crow-AMSAA, Duane, Cox-Lewis)
    are in the API docstrings.

Truncation and Delayed Entry
----------------------------

Observation of a recurrent process rarely starts at the origin. A machine may
already have been in service before monitoring began, or a study may only
record events after a subject enrols. This is **left truncation** (delayed
entry): the item is only under observation from an entry time :math:`t_L`, and
its first observed interval is integrated from :math:`t_L` rather than from
zero. **Right truncation** closes the observation window at a time :math:`t_R`,
extending the compensator integral out to :math:`t_R` even if no event or
censoring row sits exactly there.

The intensity (Poisson) models handle delayed entry directly, because the
likelihood over any interval depends only on the intensity over that interval.
The virtual-age and history-dependent models (Kijima, G1, ARA, ARI) cannot:
the virtual age at entry depends on the unobserved failures before entry, so
those models require the process to be observed from the start.

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

A **trend test** asks a more basic question: was a time-varying intensity
warranted at all? The null hypothesis is a homogeneous Poisson process (no
trend). The Laplace test and the Military-Handbook (MIL-HDBK-189C) test
[Rigdon2000]_ both use only the event times and observation windows, not the
fitted parameters, so they are a useful sanity check before committing to a
particular parametric form.

Finally, a **Cramér–von Mises** goodness-of-fit test [DaleyVereJones2003]_
measures how far the conditionally-uniform transforms fall from uniformity.
Conditional on the number of events an item shows, the normalised compensators
:math:`\Lambda(x_k) / \Lambda(\text{close})` are i.i.d. U(0, 1) under the true
model; the statistic aggregates their departure from uniformity. Because the
parameters were estimated from the same data, the p-value is obtained by a
parametric bootstrap — resimulating from the fitted model, refitting, and
recomputing the statistic — so it accounts for the estimation. For the
power-law (Crow-AMSAA) process this is the construction behind Crow's
classical goodness-of-fit test.

Competing Risks: Marked Recurrent Events
----------------------------------------

An item can experience events of several *mutually exclusive types* — a pump
that suffers seal failures, bearing failures and impeller failures, say. Each
event carries a **mark** identifying its type, and we usually want a separate
picture per type.

Non-parametrically, the **cause-specific mean cumulative function** is the MCF
restricted to one cause. The at-risk set is shared across causes (an item is at
risk for every cause until it leaves observation); only the event counts are
split by type. This is the recurrent-process analogue of the cause-specific
cumulative incidence in single-event competing risks.

Parametrically, a marked Poisson process has an elegant structure: the
cause-specific processes are **independent thinned Poisson processes**. An
event of one cause neither advances nor interrupts another cause's intensity,
so the joint likelihood factorises over causes. Each cause's intensity is
therefore just the ordinary NHPP fit to that cause's events over the full
observation window of every item, treating other-cause events exactly as a
censored (unobserved) period. The total intensity is the sum of the
cause-specific intensities.

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