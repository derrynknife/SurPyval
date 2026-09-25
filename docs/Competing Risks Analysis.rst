Competing Risks Analysis
========================

Competing risks analysis addresses situations where a subject is at risk of
experiencing more than one type of event, but only one event can occur first
and doing so removes the subject from further observation. A subject "competes"
among several possible failure causes.

Classic examples:

- A patient may die from cancer, heart disease, or another cause; the first
  to occur ends their observation period.
- A mechanical component may fail by fatigue, corrosion, or overload; which
  failure mode occurs first determines both the failure time and its cause.
- A customer may churn, upgrade, or downgrade; the event that happens first
  changes the analysis for the remaining outcomes.

Competing risks require special treatment because treating the competing
events as ordinary independent censoring gives a quantity that cannot be
interpreted as a real-world probability. A deeper subtlety is the
**identifiability problem**: from competing-risks data alone the *marginal*
(net, latent) distribution of each cause — the distribution that would be seen
if the other causes were removed — cannot be identified without an untestable
assumption about the dependence between causes [Tsiatis1975cr]_. This is why
the observable, well-defined target is the cumulative incidence function
rather than a marginal cause-specific survival.

This page builds the theory from first principles: what the data look like,
the two quantities that *can* be estimated (the cause-specific hazard and the
cumulative incidence function), why the tempting shortcut of "one minus
Kaplan-Meier with the other causes censored" is wrong, and then the
non-parametric, parametric and regression estimators SurPyval provides. Every
method described here has a runnable example on the
:doc:`Competing Risks SurPyval Modelling` page, and the full API is on the
:doc:`surpyval.competing_risks` reference page.

Relationship to Univariate Analysis
-------------------------------------

Standard survival methods (Kaplan-Meier, parametric MLE) applied to a
single cause — ignoring others — estimate the *cause-specific* survival
function. Naïvely applying KM while censoring the competing events yields a
quantity that cannot be interpreted as the probability of experiencing the
event in the real world because the competing events are not truly independent
censoring mechanisms. The subsections below make this precise.

The data: a time and a cause
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In ordinary (univariate) survival analysis each unit contributes a time
:math:`x_i` and a flag saying whether that time is an observed failure or a
censoring time. Competing-risks data adds one more piece of information to each
*observed* failure: its **cause** (also called the failure mode or event type).
Write

- :math:`T` for the time of the *first* event of any kind,
- :math:`K \in \{1, \dots, m\}` for the cause of that event, where :math:`m`
  is the number of distinct causes,
- :math:`C` for the censoring time, so that we observe
  :math:`x = \min(T, C)` and, if :math:`T \leq C`, the cause :math:`K`.

A censored unit therefore has *no* cause: we only know that none of the
:math:`m` events had happened by :math:`x`. In SurPyval the cause is passed
as the array ``e`` (any hashable labels: integers, strings, ...), and a
missing cause (``None`` or ``NaN``) marks a censored row.

Throughout, censoring is assumed to be **independent**: :math:`C` carries no
information about :math:`(T, K)` (for the regression models, conditionally on
the covariates). This is the same assumption the Kaplan-Meier estimator makes;
it is about the *censoring*, not about the causes, and it says nothing about
whether the causes are independent of each other.

The cause-specific hazard
~~~~~~~~~~~~~~~~~~~~~~~~~

The natural building block is the rate at which cause :math:`k` strikes units
that are still event-free. The **cause-specific hazard** is

.. math::

    h_k(t) = \lim_{\Delta t \to 0}
             \frac{P(t \leq T < t + \Delta t,\; K = k \mid T \geq t)}{\Delta t},
    \qquad k = 1, \dots, m.

Intuitively :math:`h_k(t)\,\Delta t` is the probability that a unit which has
survived everything up to :math:`t` fails from cause :math:`k` in the next
small interval. Because a unit can fail from only one cause at a time, the
all-cause hazard is the sum of the cause-specific hazards, and so the
all-cause survival function (the probability of being free of *every* event)
is

.. math::

    h(t) = \sum_{k=1}^{m} h_k(t), \qquad
    S(t) = P(T > t) = \exp\!\Big(-\sum_{k=1}^{m} H_k(t)\Big),

where :math:`H_k(t) = \int_0^t h_k(u)\,du` is the cumulative cause-specific
hazard. The cause-specific hazards are *identifiable*: they can be estimated
from competing-risks data without any assumption about how the causes depend
on one another, because each one only involves units that are observed to be
still at risk.

The cumulative incidence function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The correct marginal quantity of interest is the **Cumulative Incidence
Function (CIF)**, also called the sub-distribution function. For
cause :math:`k`:

.. math::

    F_k(t) = P(T \leq t,\; K = k)

where :math:`T` is the event time and :math:`K` is the event type. In words:
the probability that a unit has failed *from cause* :math:`k` by time
:math:`t`, in the real world where all the other causes are also operating.
It is built from the cause-specific hazard by asking, at each instant, "is the
unit still event-free, and does cause :math:`k` strike now?":

.. math::

    F_k(t) = \int_0^t h_k(u)\, S(u^-)\, du .

The integrand :math:`f_k^{\text{sub}}(t) = h_k(t)S(t^-)` is the
*sub-distribution density* (SurPyval calls it the instantaneous incidence
function, ``iif``). Two properties follow immediately:

- The CIFs sum to the overall failure probability:

  .. math::

      \sum_{k=1}^{m} F_k(t) = F(t) = 1 - S(t).

- Each CIF is **improper**: it plateaus at the eventual probability of that
  cause, :math:`F_k(\infty) = P(K = k) < 1`, rather than rising to one. A
  unit that fails from cause 2 can never go on to fail from cause 1.

Notice that :math:`F_k` depends on *all* the cause-specific hazards through
:math:`S`. Raising the hazard of cause 2 lowers the incidence of cause 1,
even if :math:`h_1` is untouched, simply because fewer units survive long
enough to fail from cause 1. This is the single most important idea in
competing risks, and it is why the regression section below distinguishes
models for the hazard from models for the incidence.

Why one minus Kaplan-Meier of one cause is wrong
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The tempting shortcut is to analyse cause :math:`k` on its own: treat failures
from every other cause as censored and compute a Kaplan-Meier curve
:math:`\hat{S}_k^{\text{KM}}`. This estimates

.. math::

    1 - S_k(t) = 1 - \exp\{-H_k(t)\},

which is the probability of failing from cause :math:`k` by :math:`t` in a
*hypothetical* world where cause :math:`k` is the only cause in operation (the
"net" probability). Two things are wrong with reading it as a real-world
probability:

1. **It overstates the incidence.** Since :math:`S(u) \leq S_k(u)`,
   :math:`F_k(t) = \int_0^t h_k S \, du \leq \int_0^t h_k S_k\, du
   = 1 - S_k(t)`. Treating competing failures as censored pretends those
   units could still go on to fail from cause :math:`k`, when in reality they
   never can. Summed over the causes, the "one minus KM" curves can exceed one.
2. **It is only meaningful under an untestable assumption.** The hypothetical
   world "with the other causes removed" corresponds to the distribution of a
   *latent* failure time :math:`T_k`, and it equals :math:`S_k` only if the
   latent times of the different causes are independent. Competing-risks data
   cannot confirm or refute that independence [Tsiatis1975cr]_.

A small example makes the first point concrete. Six units are followed; the
fourth is censored and the rest fail from cause A or B:

.. list-table::
   :header-rows: 1

   * - time :math:`x`
     - 1
     - 2
     - 3
     - 4
     - 5
     - 6
   * - cause
     - A
     - B
     - A
     - censored
     - B
     - A
   * - at risk :math:`r`
     - 6
     - 5
     - 4
     - 3
     - 2
     - 1
   * - all-cause KM :math:`\hat{S}(x)`
     - 5/6
     - 4/6
     - 3/6
     - 3/6
     - 1/4
     - 0

By time 6 every unit has either failed or been censored. The Aalen-Johansen
estimator described next gives :math:`\hat{F}_A(6) = 7/12` and
:math:`\hat{F}_B(6) = 5/12`, which sum to :math:`1 - \hat{S}(6) = 1`. The
naive "1 - KM with B censored" for cause A instead reaches **1** at time 6
(the last unit at risk fails from A), and the naive curve for B reaches 0.6:
together they claim a total failure probability of 1.6. The naive curve for A
says that every unit eventually fails from A, when in fact 5/12 of the
incidence was due to B. The same numbers are reproduced with code on
the :doc:`Competing Risks SurPyval Modelling` page.

Net and crude probabilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two quantities above have traditional names. The CIF :math:`F_k` is the
**crude** probability: it is what actually happens, and it is always
identifiable. :math:`1 - S_k` is the **net** probability: what would happen if
cause :math:`k` acted alone. In reliability engineering the net quantity is
sometimes exactly what is wanted — "if we eliminate the corrosion failure mode
by a design change, what will the fatigue life look like?" — but that is a
counterfactual question and its answer relies on the causes acting
independently. SurPyval exposes both; the CIF (``cif``) should be the default
answer to "how likely is failure from this cause?", and the net quantities
(``ff``/``sf`` with an ``event``) should be used knowingly.

Non-Parametric CIF Estimation
------------------------------

The Aalen-Johansen estimator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The non-parametric estimate replaces every quantity in
:math:`F_k(t) = \int_0^t h_k(u) S(u^-) du` with its empirical counterpart, in
exactly the way the Nelson-Aalen and Kaplan-Meier estimators do for
single-cause data (see :doc:`Non-Parametric Estimation`). Order the distinct
observed times :math:`x_1 < x_2 < \dots < x_J` and at each :math:`x_j` let

- :math:`r_j` be the number at risk (units with an observed or censored time
  :math:`\geq x_j`),
- :math:`d_{k,j}` be the number of cause-:math:`k` events at :math:`x_j`, and
  :math:`d_j = \sum_k d_{k,j}` the number of events of any cause.

The empirical CIF for cause :math:`k` is then estimated from the
cause-specific hazard rates:

.. math::

    \hat{F}_k(t) = \sum_{x_j \leq t} \hat{h}_k(x_j)\, \hat{S}(x_{j}^-)
                 = \sum_{x_j \leq t} \frac{d_{k,j}}{r_j}\,\hat{S}(x_{j-1})

where :math:`\hat{h}_k(x_j) = d_{k,j} / r_j` is the cause-specific hazard
increment at event time :math:`x_j`, and
:math:`\hat{S}(x_j^-) = \hat{S}(x_{j-1})` is the all-cause Kaplan-Meier
survival just before :math:`x_j`,

.. math::

    \hat{S}(t) = \prod_{x_j \leq t}\left(1 - \frac{d_j}{r_j}\right),
    \qquad \hat{S}(x_0) = 1 .

This is the **Aalen-Johansen** estimator [AalenJohansen1978cr]_. Read each term
as "the fraction of the population still event-free just before
:math:`x_j`" times "the fraction of those at risk that fail from cause
:math:`k` at :math:`x_j`". With a single cause and no censoring it reduces to
the empirical CDF, and with a single cause and censoring to
:math:`1 - \text{KM}`.

Two details of the formula are easy to get wrong, and both matter:

- **Weight by** :math:`\hat{S}(x_j^-)`, **not** :math:`\hat{S}(x_j)`. The
  hazard at :math:`x_j` acts on the units alive just *before* :math:`x_j`.
  Using the survival after the jump removes the units failing at
  :math:`x_j` before they have been counted, and makes the CIFs
  systematically too small (with one cause and
  no censoring the total incidence stops well short of one).
- **Use the product-limit (Kaplan-Meier) survival as the weight.** Only it
  satisfies the telescoping identity
  :math:`\hat{S}(x_{j-1}) - \hat{S}(x_j) = \hat{S}(x_{j-1})\,d_j/r_j`, which
  is what makes the estimated CIFs sum *exactly* to :math:`1 - \hat{S}(t)`.
  Pairing the discrete increments :math:`d_{k,j}/r_j` with the exponential
  (Nelson-Aalen) survival :math:`e^{-\hat{H}}` instead inflates the CIFs and
  can push the total incidence above one in small samples.

The worked example above is this formula applied by hand: for cause A,
:math:`\hat{F}_A(6) = 1 \cdot \tfrac{1}{6} + \tfrac{4}{6} \cdot \tfrac{1}{4}
+ \tfrac{1}{4} \cdot \tfrac{1}{1} = \tfrac{7}{12}`.

What SurPyval computes
~~~~~~~~~~~~~~~~~~~~~~

SurPyval estimates the non-parametric CIF for each cause with the
:class:`~surpyval.univariate.competing_risks.nonparametric.competing_risks.CompetingRisks`
class, which uses this cause-specific-hazard construction directly. The
incidence increments are always weighted by the Kaplan-Meier :math:`\hat{S}(x_j^-)`,
whichever ``method`` (``"Nelson-Aalen"``, the default, or ``"Kaplan-Meier"``)
is requested; the shared helper
:func:`~surpyval.univariate.competing_risks.aalen_johansen.aalen_johansen_iif`
implements the weighting once for the non-parametric CIF, the cause-specific
Cox CIF and the pooled CIF inside Gray's test. So ``cif`` and ``iif`` do not
depend on ``method``.

Alongside the CIFs, the fitted model also carries the cause-specific
hazard quantities. ``hf`` returns the hazard increment
:math:`d_{k,j}/r_j` at the most recent observed time (a jump size, not a
rate; zero if no cause-:math:`k` failure occurred there). The survival
functions follow ``method``:

.. list-table::
   :header-rows: 1

   * - ``method``
     - ``sf(t, event=k)``
     - ``Hf(t, event=k)``
   * - ``"Nelson-Aalen"`` (default)
     - :math:`\exp\{-\hat{H}_k(t)\}`, with
       :math:`\hat{H}_k(t) = \sum_{x_j \leq t} d_{k,j}/r_j`
     - :math:`\hat{H}_k(t)`
   * - ``"Kaplan-Meier"``
     - :math:`\prod_{x_j \leq t} (1 - d_{k,j}/r_j)`
     - :math:`-\log` of that product

so that ``sf == exp(-Hf)`` and ``ff == 1 - sf`` for either method. With an
``event`` these are the *net* quantities discussed above (cause :math:`k`
acting alone, the other causes treated as censoring); without one they refer
to all causes combined, using :math:`d_j` in place of :math:`d_{k,j}`. The
two methods differ little while the risk sets are large: since
:math:`e^{-h} \geq 1 - h`, the Nelson-Aalen survival is never below the
product limit, and the gap grows in the tail, where the risk sets are small
and each increment is large. All of these are step
functions, equal to zero (or one, for survival) before the first observed
time. The fitted all-cause survival at the distinct times is also stored as
the attribute ``S``.

Censoring and truncation
~~~~~~~~~~~~~~~~~~~~~~~~

Censoring enters the Aalen-Johansen estimator only through the risk sets
:math:`r_j`: a unit censored at :math:`x_i` counts as at risk at every event
time up to and including :math:`x_i` and then leaves, exactly as in the
Kaplan-Meier estimator. The competing-risks classes in SurPyval support
**right censoring** only. Left- and interval-censored rows (``c`` of ``-1`` or
``2``) are rejected with a ``ValueError``: with an interval-censored failure
of known cause the unit's contribution to :math:`F_k` would have to be spread
across the interval using all of the cause-specific hazards, which these
estimators do not do.

Left truncation (delayed entry) is not an argument of the non-parametric or
regression competing-risks classes. For parametric models it can be handled
exactly by fitting each cause separately and assembling the result, as the
next section explains.

Parametric Competing Risks
---------------------------

The latent-failure-time picture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A parametric competing risks model specifies a separate parametric
distribution for each cause. The most intuitive way to picture it is the
*latent failure time* model: each cause :math:`k` has its own clock
:math:`T_k` with distribution :math:`F_k^{\text{net}} = 1 - S_k`, the unit
fails at :math:`T = \min_k T_k`, and the cause is whichever clock ran out
first. The overall survival function is the product of the cause-specific
survival functions (assuming independent latent failure times):

.. math::

    S(t) = \prod_{k=1}^{m} S_k(t)

The overall density is:

.. math::

    f(t) = \sum_{k=1}^{m} f_k(t) \prod_{j \neq k} S_j(t)

The term inside the sum is
the sub-distribution density of cause :math:`k`: the density of a cause-:math:`k`
failure at :math:`t` multiplied by the probability that no other cause has
struck first. Integrating it gives the model's cumulative incidence,

.. math::

    F_k(t) = \int_0^t f_k(u) \prod_{j \neq k} S_j(u)\, du
           = \int_0^t h_k(u)\, S(u)\, du ,

and the eventual probability of each cause is :math:`F_k(\infty)`.

A more careful — and assumption-free — reading of the same model is that it
specifies the *cause-specific hazards* parametrically: :math:`h_k(t)` is the
hazard of the chosen distribution for cause :math:`k`, and
:math:`S_k = e^{-H_k}` is just a convenient way of writing
:math:`\exp(-\int h_k)`. Under that reading the fitted CIFs, the all-cause
survival and the simulated ``(time, cause)`` pairs are all valid whether or not
the latent times are independent. Only the *net* interpretation of each
:math:`S_k` ("the life if the other causes were removed") needs independence.

Why the fit separates by cause
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SurPyval provides the
:class:`~surpyval.univariate.competing_risks.parametric.parametric_competing_risks.ParametricCompetingRisks`
class for this model. Write :math:`\delta_{ik} = 1` if unit :math:`i` was
observed to fail from cause :math:`k` and 0 otherwise (a censored unit has
:math:`\delta_{ik} = 0` for every :math:`k`). The likelihood of a unit is its
cause-specific hazard at the failure time (if it failed) times the probability
of surviving all causes until :math:`x_i`:

.. math::

    L = \prod_i \Big[\prod_k h_k(x_i)^{\delta_{ik}}\Big]\, S(x_i)
      = \prod_{k=1}^{m} \Big[\prod_i h_k(x_i)^{\delta_{ik}}\, S_k(x_i)\Big].

Each bracket on the right is an ordinary right-censored likelihood for
cause :math:`k`'s distribution in which failures from cause :math:`k` are the
observed events and *everything else* — failures from the other causes and
genuinely censored units — is right-censored. Under the independent-latent-times
assumption (or, equivalently, with the cause-specific hazards parametrised
separately) the joint likelihood therefore separates, so each cause's
distribution is fitted independently by MLE with the *other* causes' events
treated as right-censored. This is not an approximation: it is the exact
maximum-likelihood estimate of the joint model, and its log-likelihood, AIC and
BIC are the sums of the per-cause values.

It is worth being clear about how this squares with the warning against
"1 - KM of one cause". Treating the other causes as censored is the correct
way to *estimate* each cause-specific hazard; it is only wrong to read
:math:`1 - S_k` as the probability of failing from cause :math:`k`. The
parametric model gets its CIFs right by recombining the fitted hazards through
the formula for :math:`F_k` above.

The same factorisation extends to **left truncation**. A unit that enters
observation at age :math:`\tau_i` contributes
:math:`\prod_k h_k(x_i)^{\delta_{ik}}\, S(x_i)/S(\tau_i)`, and because
:math:`S(x)/S(\tau) = \prod_k S_k(x)/S_k(\tau)` this also splits into one
left-truncated, right-censored likelihood per cause. Delayed-entry data can
therefore be handled by fitting each cause's distribution with its truncation
bounds and combining the fits with ``ParametricCompetingRisks.from_fitted``;
the how-to page shows this. (Interval censoring does *not* factorise in this
way, because the probability of a cause-:math:`k` failure inside an interval
involves every cause's survival across the interval.)

Computing the model's quantities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Most model quantities are closed-form combinations of the per-cause models:
the all-cause survival is :math:`\prod_k S_k(t)`, the all-cause hazard is
:math:`\sum_k h_k(t)`, and the instantaneous incidence of cause :math:`k` is
:math:`f_k(t)\prod_{j\neq k} S_j(t)`. The CIF integral generally has no closed
form, so SurPyval evaluates it numerically (trapezoidal integration on a fine
grid from 0 to the largest requested time), and :math:`F_k(\infty)` is
evaluated at a time by which the causes have essentially played out.

Because the parametric model is a full generative model, it can also be
**simulated**: draw a latent time from every cause's distribution and keep the
earliest, together with its cause. The per-cause models need not be from the
same family, and a cause may carry a cure (limited-failure-population)
fraction, in which case some units never fail from that cause; if every cause
has a cure fraction some units never fail at all and the probabilities of the
causes sum to less than one.

When to prefer a parametric model: when you need smooth CIFs, extrapolation
beyond the last observed failure, a compact description of each failure mode
(e.g. a Weibull shape telling wear-out from random failures), or simulation for
Monte-Carlo studies. The non-parametric estimator makes no shape assumption and
is the right first look — and a good check of the parametric fit.

Regression: Fine-Gray and Cause-Specific PH
--------------------------------------------

When covariates :math:`Z` (a row vector per unit: treatment, load, material,
...) are available there are two fundamentally different things one can
model, mirroring the two quantities defined above: the cause-specific hazards,
or the cumulative incidence directly. Two main regression approaches are used
in competing risks:

**Cause-specific proportional hazards** — fits a separate Cox or parametric
PH model for each cause, with all other cause events treated as censored:

.. math::

    h_k(t \mid Z) = h_{k,0}(t)\, e^{Z \beta_k}

This estimates the effect of covariates on the hazard of each cause
independently.

**Fine-Gray sub-distribution hazards** — models the effect of covariates
directly on the CIF via a proportional hazards model on the sub-distribution
hazard:

.. math::

    h_k^*(t \mid Z) = h_{k,0}^*(t)\, e^{Z \gamma_k}

This is the natural choice when the scientific question is about the
probability of a cause occurring in the presence of competing risks (e.g.
clinical risk scores).

SurPyval provides the ``FineGray`` and
``CompetingRisksProportionalHazards`` classes. The subsections below explain
each model, how it is estimated and how to read its coefficients.

Cause-specific proportional hazards in detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here :math:`h_{k,0}(t)` is an unspecified baseline hazard for cause :math:`k`
and :math:`\beta_k` a vector of coefficients for that cause (one per column of
:math:`Z`). By the same factorisation as in the parametric case, the partial
likelihood separates by cause, so each :math:`\beta_k` is estimated by an
ordinary Cox model in which cause-:math:`k` failures are events and all other
rows are censored. :math:`e^{\beta_{k,p}}` is a cause-specific hazard ratio:
the multiplicative change in the *rate* of cause :math:`k` among units still
event-free, per unit increase of covariate :math:`p`.

``CompetingRisksProportionalHazards`` with ``how="Cox"`` fits one ``CoxPH``
model per cause (see :doc:`regression/cox_ph`) and keeps each cause's Breslow
baseline cumulative hazard :math:`\hat{\Lambda}_{k,0}`. Its ``tie_method``
argument is passed on as the Cox tie-handling ``method``; note that its
default is ``"efron"``, whereas ``CoxPH.fit`` on its own defaults to
``"breslow"``. The two agree when no failure times are tied. The CIF at a covariate vector :math:`Z` is then
assembled exactly as in the Aalen-Johansen formula, but with covariate-specific
hazards:

.. math::

    \hat{F}_k(t \mid Z) = \sum_{x_j \leq t}
        \Delta\hat{\Lambda}_{k,0}(x_j)\, e^{Z\hat{\beta}_k}\,
        \hat{S}(x_{j-1} \mid Z),
    \qquad
    \hat{S}(t \mid Z) = \prod_{x_j \leq t}\Big(1 - \sum_{l=1}^{m}
        \Delta\hat{\Lambda}_{l,0}(x_j)\, e^{Z\hat{\beta}_l}\Big).

The formula shows the catch in interpreting cause-specific coefficients: the
incidence of cause :math:`k` depends on *every* cause's coefficients through
:math:`\hat{S}(t \mid Z)`. A covariate can raise the hazard of cause :math:`k`
(:math:`\beta_k > 0`) and yet lower its incidence, if it raises a competing
cause's hazard even more.

The survival weight is a *product limit*, for the same reason as in the
Aalen-Johansen estimator: only then do the increments telescope, so that the
cause-specific CIFs sum to exactly :math:`1 - \hat{S}(t \mid Z)` and never
exceed one. At a covariate value far from the data a step's total hazard
increment :math:`\sum_l \Delta\hat{\Lambda}_{l,0}\, e^{Z\hat{\beta}_l}` can
exceed one (a small risk set times a large multiplier); such a step exhausts
the survivors, and each cause takes its proportional share of them. The model's
``sf`` is the Cox survival :math:`\exp(-\sum_l \hat{\Lambda}_{l,0}(t)
e^{Z\hat{\beta}_l})`, which is very close to the product limit when the
increments are small.

The Fine-Gray model in detail
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Fine and Gray [FineGray1999cr]_ asked a different question: is there a hazard
whose proportional-hazards model acts *directly* on the CIF, the way the
ordinary hazard acts on the survival function? The answer is the
**sub-distribution hazard**

.. math::

    h_k^*(t) = \lim_{\Delta t \to 0}
        \frac{P\big(t \leq T < t + \Delta t,\, K = k \;\big|\;
              T \geq t \;\text{or}\; (T < t,\, K \neq k)\big)}{\Delta t}
      = \frac{f_k^{\text{sub}}(t)}{1 - F_k(t)} .

The conditioning event is the peculiar part: units that have *already failed
from a competing cause* are kept "at risk" for cause :math:`k` — they will
never fail from :math:`k`, which is exactly what makes :math:`1 - F_k(t)` the
right denominator. Because :math:`h_k^*` is the hazard of the improper
distribution :math:`F_k`, we get the familiar relation
:math:`F_k(t) = 1 - \exp\{-\Lambda_k^*(t)\}` with
:math:`\Lambda_k^* = \int h_k^*`. The proportional sub-distribution hazards
model :math:`h_k^*(t \mid Z) = h_{k,0}^*(t) e^{Z\gamma_k}` therefore gives

.. math::

    F_k(t \mid Z) = 1 - \exp\big\{-\Lambda_{k,0}^*(t)\, e^{Z\gamma_k}\big\},

so a positive coefficient *always* raises the incidence of cause :math:`k`,
and :math:`e^{\gamma_{k,p}}` is a sub-distribution hazard ratio. (In the code
and on the how-to page the Fine-Gray coefficients are called ``beta``.)

**Estimation with censoring.** Without censoring the sub-distribution risk set
at time :math:`t` is simply "everyone who has not failed from cause :math:`k`
by :math:`t`". Under right censoring we do not know whether a unit that failed
from a competing cause at :math:`x_i < t` *would* still have been under
observation at :math:`t`. Fine and Gray keep it in the risk set with the
inverse-probability-of-censoring weight

.. math::

    w_i(t) = \begin{cases}
       1 & x_i \geq t \quad \text{(still under observation)},\\[2pt]
       \hat{G}(t)/\hat{G}(x_i) & x_i < t \text{ and failed from a competing cause},\\[2pt]
       0 & \text{otherwise (censored, or failed from cause } k \text{, before } t),
    \end{cases}

where :math:`\hat{G}(t)` is the Kaplan-Meier estimate of the *censoring*
survival function :math:`P(C > t)` (the censored rows play the role of
"events"). :math:`\hat{G}(t)/\hat{G}(x_i)` is the estimated probability that
the unit would have remained uncensored from :math:`x_i` to :math:`t`. The
coefficients maximise the weighted partial log-likelihood

.. math::

    \ell(\gamma) = \sum_{i:\,K_i = k} n_i \Big[ Z_i\gamma
        - \log \sum_j w_j(x_i)\, n_j\, e^{Z_j\gamma} \Big],

with :math:`n_i` the count of each row. (Tied event times are handled by
this Breslow form: each tied event sees the same weighted risk set.) SurPyval's
``FineGray`` maximises this with BFGS (exact gradients by automatic
differentiation), reports standard errors from the inverse of the Hessian of
:math:`\ell` at the optimum, and estimates the baseline by the Breslow-type
increments :math:`\Delta\hat{\Lambda}_{k,0}^*(t) = d_k(t) / \sum_j w_j(t)\,
n_j\, e^{Z_j\hat{\gamma}}`. The fitted CIF is a step function, flat before
the first and after the last cause-:math:`k` event time. Those standard errors
are the model-based (inverse information) ones; the robust sandwich variance
that Fine and Gray derived to account for the estimated weights is not
implemented, so treat the reported ``se`` and ``p_values`` as approximate.

Choosing between them
~~~~~~~~~~~~~~~~~~~~~

The two models answer different questions and are complementary rather than
competing [Latouche2013cr]_:

- **"What drives the rate of this failure mode?"** — aetiology, physics of
  failure, mechanism. Use cause-specific hazards. The coefficients have the
  usual hazard-ratio meaning among units still at risk.
- **"Who is most likely to end up failing from this mode?"** — prognosis,
  risk scores, warranty or spares forecasting. Use Fine-Gray. The coefficients
  map monotonically onto the incidence.

The two coincide only in special cases (for instance, when the covariate does
not affect the competing causes at all). Two practical caveats of Fine-Gray:
the model is fitted one cause at a time, and separate Fine-Gray fits for
different causes are not constrained to be mutually consistent (their CIFs can
sum to more than one); and the sub-distribution hazard itself has no direct
physical meaning, so interpret the model through its CIFs. In SurPyval,
``CompetingRisksProportionalHazards`` with ``how="Fine-Gray"`` fits the
Fine-Gray model for every cause at once.

Comparing incidence across groups: Gray's test
----------------------------------------------

The log-rank test compares survival curves; its competing-risks analogue is
**Gray's test** [Gray1988cr]_, which compares the *cumulative incidence
functions* of a chosen cause across groups. The distinction matters. A
cause-specific log-rank compares the cause-specific *hazards* — the
instantaneous rate of the cause among those still at risk — whereas Gray's
test compares the CIFs themselves, i.e. the actual *incidence* of the cause in
a population that is also being depleted by the competing causes.

Gray's test achieves this by modifying the risk set. Instead of removing
subjects who fail from a competing cause (as a cause-specific analysis would),
it keeps them in the **subdistribution risk set** with an
inverse-probability-of-censoring weight

.. math::

    w_j(t) = \frac{\hat{G}(t)}{\hat{G}(x_j)},

where :math:`\hat{G}` is the Kaplan-Meier estimate of the censoring
distribution. Subjects who have already failed from a competing cause therefore
continue to count — with a decaying weight — which is precisely what makes the
comparison one of incidence rather than of instantaneous rate. Under the null
hypothesis of equal CIFs the resulting statistic is approximately
:math:`\chi^2` distributed with :math:`G - 1` degrees of freedom for :math:`G`
groups. Reach for it when the question is "how many fail of this
cause", and for the cause-specific log-rank when the question is "how fast".

The statistic
~~~~~~~~~~~~~

SurPyval computes the test as a weighted log-rank comparison on the weighted
sub-distribution risk set (the same weights :math:`w_i(t)` as in the Fine-Gray
model above). At each distinct time :math:`\tau` at which the cause of
interest occurs, let :math:`R_g(\tau) = \sum_{i \in g} n_i w_i(\tau)` be the
weighted risk set of group :math:`g`, :math:`R = \sum_g R_g`,
:math:`d_g(\tau)` the number of cause-of-interest events in group :math:`g`
and :math:`d = \sum_g d_g`. Under the null hypothesis that every group has the
same CIF, the events should be shared out in proportion to the weighted risk
sets, so the test accumulates observed-minus-expected counts and a
hypergeometric variance:

.. math::

    U_g = \sum_\tau W(\tau)\Big[d_g(\tau) - d(\tau)\frac{R_g(\tau)}{R(\tau)}\Big],
    \qquad
    V = \sum_\tau W(\tau)^2\, \frac{d\,(R - d)}{R - 1}
        \big[\operatorname{diag}(p) - p\,p^{\top}\big],
    \quad p_g = R_g / R .

Dropping one group to make :math:`V` invertible, the statistic
:math:`U^{\top} V^{-1} U` is referred to a :math:`\chi^2` distribution with
(number of groups :math:`- 1`) degrees of freedom. The weight
:math:`W(\tau) = \{1 - \hat{F}(\tau^-)\}^{\rho}` uses the pooled
Aalen-Johansen CIF of the cause; the default :math:`\rho = 0` (every event time
weighted equally) is the standard test, and :math:`\rho > 0` down-weights late
event times, making the test more sensitive to differences in early
incidence.

This construction is SurPyval's own, and it is close to, but not identical
with, Gray's original statistic. Gray [Gray1988cr]_ estimates the censoring
distribution separately within each group and derives a variance from the
asymptotic theory of the CIF estimators; SurPyval uses one censoring
Kaplan-Meier :math:`\hat{G}` for the pooled sample and the hypergeometric
variance of an ordinary log-rank test computed on the weighted risk sets. The
two give very similar answers when the groups are censored in a similar way,
and the test is calibrated in simulation (the how-to page checks this), but
p-values will not match R's ``cmprsk::cuminc`` to many digits, and they can
differ more when the groups' censoring patterns are very different.

A useful way to see the difference from a cause-specific log-rank: imagine two
groups with *identical* cause-1 hazards but a much larger cause-2 hazard in the
second group. The cause-specific log-rank for cause 1 sees no difference (the
rates are equal), while Gray's test correctly reports that far fewer units in
the second group ever fail from cause 1. Both answers are right; they answer
different questions. The how-to page runs exactly this experiment.

For worked examples of estimating cumulative incidence functions, fitting the
Fine-Gray and cause-specific proportional hazards models, and comparing groups
with Gray's test, see the :doc:`Competing Risks SurPyval Modelling` page. For
*repeated* events with several failure modes (the recurrent-events analogue of
competing risks) see :doc:`Recurrent Event Analysis`.


Further Reading
---------------

- Prentice, R. L., Kalbfleisch, J. D., Peterson, A. V., Flournoy, N.,
  Farewell, V. T., & Breslow, N. E. (1978). The analysis of failure times in
  the presence of competing risks. *Biometrics*, 34(4), 541–554.
- Gray, R. J. (1988). A class of K-sample tests for comparing the cumulative
  incidence of a competing risk. *The Annals of Statistics*, 16(3), 1141–1154.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the
  subdistribution of a competing risk. *JASA*, 94(446), 496–509.
- Pintilie, M. (2006). *Competing Risks: A Practical Perspective*. Wiley.
- Putter, H., Fiocco, M., & Geskus, R. B. (2007). Tutorial in biostatistics:
  competing risks and multi-state models. *Statistics in Medicine*, 26(11),
  2389–2430. An accessible introduction to cause-specific hazards, the CIF and
  the Aalen-Johansen estimator.

.. [Tsiatis1975cr] Tsiatis, A. (1975). A nonidentifiability aspect of the
   problem of competing risks. *Proceedings of the National Academy of
   Sciences*, 72(1), 20–22.

.. [AalenJohansen1978cr] Aalen, O. O., & Johansen, S. (1978). An empirical
   transition matrix for non-homogeneous Markov chains based on censored
   observations. *Scandinavian Journal of Statistics*, 5(3), 141–150.

.. [FineGray1999cr] Fine, J. P., & Gray, R. J. (1999). A proportional hazards
   model for the subdistribution of a competing risk. *Journal of the American
   Statistical Association*, 94(446), 496–509.

.. [Gray1988cr] Gray, R. J. (1988). A class of K-sample tests for comparing the
   cumulative incidence of a competing risk. *The Annals of Statistics*,
   16(3), 1141–1154.

.. [Latouche2013cr] Latouche, A., Allignol, A., Beyersmann, J., Labopin, M., &
   Fine, J. P. (2013). A competing risks analysis should report results on all
   cause-specific hazards and cumulative incidence functions. *Journal of
   Clinical Epidemiology*, 66(6), 648–653.
