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
:math:`Z\beta = \beta_0 z_0 + \beta_1 z_1 + \dots`. (SurPyval labels the
coefficients ``beta_0``, ``beta_1``, ... in the order of the columns of
:math:`Z`.)

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

- **Static covariates.** The covariates are properties of the item, constant
  over its observation. Supply them one row per event row (repeating the
  item's values) or as a dictionary keyed by item. The residuals and the
  goodness-of-fit test take each item's covariates from its first row.
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
- **Scope.** Covariates are available for the HPP and NHPP intensity models.
  The imperfect-repair (renewal) models, the cause-specific (marked) models
  and gapped (multi-window) observation do not take covariates.

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

References
----------

.. [Cook2007r] Cook, R.J. and Lawless, J.F., 2007. *The Statistical Analysis of
   Recurrent Events*. Springer.

.. [Lawless1987] Lawless, J.F., 1987. Regression methods for Poisson process
   data. *Journal of the American Statistical Association*, 82(399),
   pp.808-815.

.. [Rigdon2000r] Rigdon, S.E. and Basu, A.P., 2000. *Statistical Methods for the
   Reliability of Repairable Systems*. John Wiley & Sons.
