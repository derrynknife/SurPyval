
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

.. note::

    Non-parametric CIF estimation is not yet implemented in SurPyval. Use
    the parametric ``CompetingRisks`` model or an external library such as
    ``lifelines`` for this calculation currently.


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


Further Reading
---------------

- Prentice, R. L., Kalbfleisch, J. D., Peterson, A. V., Flournoy, N.,
  Farewell, V. T., & Breslow, N. E. (1978). The analysis of failure times in
  the presence of competing risks. *Biometrics*, 34(4), 541–554.
- Fine, J. P., & Gray, R. J. (1999). A proportional hazards model for the
  subdistribution of a competing risk. *JASA*, 94(446), 496–509.
- Pintilie, M. (2006). *Competing Risks: A Practical Perspective*. Wiley.
