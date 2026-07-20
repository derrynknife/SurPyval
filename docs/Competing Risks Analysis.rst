
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

Competing risks require special treatment because the causes are not
independent; accounting for the wrong event type leads to biased estimates.
This is sometimes called the "identifiability problem" in competing risks.

Relationship to Univariate Analysis
-------------------------------------

Standard survival methods (Kaplan-Meier, parametric MLE) applied to a
single cause — ignoring others — estimate the *cause-specific* survival
function. Naïvely applying KM while censoring the competing events yields a
quantity that cannot be interpreted as the probability of experiencing the
event in the real world because the competing events are not truly independent
censoring mechanisms.

The correct marginal quantity of interest is the **Cumulative Incidence
Function (CIF)**, also called the sub-distribution survival function. For
cause :math:`k`:

.. math::

    F_k(t) = P(T \leq t,\; K = k)

where :math:`T` is the event time and :math:`K` is the event type. The CIFs
sum to the overall failure probability:

.. math::

    \sum_{k=1}^{K} F_k(t) = F(t) = 1 - S(t)

Non-Parametric CIF Estimation
------------------------------

The empirical CIF for cause :math:`k` is estimated from the cause-specific
hazard rates:

.. math::

    \hat{F}_k(t) = \sum_{x_i \leq t} \hat{h}_k(x_i)\, \hat{S}(x_i^-)

where :math:`\hat{h}_k(x_i) = d_{k,i} / r_i` is the cause-specific hazard at
event time :math:`x_i`, :math:`d_{k,i}` is the number of cause-:math:`k`
events at :math:`x_i`, :math:`r_i` is the total at-risk count, and
:math:`\hat{S}(x_i^-)` is the overall survival estimate just before
:math:`x_i`.

SurPyval estimates the non-parametric CIF for each cause with the
``CompetingRisks`` class, which uses this cause-specific-hazard construction
directly (the Nelson-Aalen or Kaplan-Meier estimate of the overall survival
supplies :math:`\hat{S}(x_i^-)`).


Parametric Competing Risks
---------------------------

A parametric competing risks model specifies a separate parametric
distribution for each cause. The overall survival function is the product of
the cause-specific survival functions (assuming independent latent failure
times):

.. math::

    S(t) = \prod_{k=1}^{K} S_k(t)

The overall density is:

.. math::

    f(t) = \sum_{k=1}^{K} f_k(t) \prod_{j \neq k} S_j(t)

SurPyval provides the ``CompetingRisks`` class for this model. Parameters for
each cause distribution are estimated jointly by MLE.


Regression: Fine-Gray and Cause-Specific PH
--------------------------------------------

Two main regression approaches are used in competing risks:

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
``CompetingRisksProportionalHazards`` classes.


Comparing incidence across groups: Gray's test
----------------------------------------------

The log-rank test compares survival curves; its competing-risks analogue is
**Gray's test** (1988), which compares the *cumulative incidence functions* of
a chosen cause across groups. The distinction matters. A cause-specific
log-rank compares the cause-specific *hazards* — the instantaneous rate of the
cause among those still at risk — whereas Gray's test compares the CIFs
themselves, i.e. the actual *incidence* of the cause in a population that is
also being depleted by the competing causes.

Gray's test achieves this by modifying the risk set. Instead of removing
subjects who fail from a competing cause (as a cause-specific analysis would),
it keeps them in the **subdistribution risk set** with an
inverse-probability-of-censoring weight

.. math::

    w_j(t) = \frac{\hat{G}(t)}{\hat{G}(x_j)},

where :math:`\hat{G}` is the Kaplan-Meier estimate of the censoring
distribution. Subjects who have already failed from a competing cause therefore
continue to count — with a decaying weight — which is precisely what makes the
comparison one of incidence rather than of instantaneous rate. The resulting
statistic is :math:`\chi^2` distributed with :math:`k - 1` degrees of freedom
for :math:`k` groups. Reach for it when the question is "how many fail of this
cause", and for the cause-specific log-rank when the question is "how fast".

For examples of estimating cumulative incidence functions and comparing them
with Gray's test, see the Competing Risks entry in the SurPyval Modelling
section of the docs.


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
