Degradation Analysis
====================

Sometimes failure data is scarce or takes too long to collect: items are
highly reliable, test time is limited, and few (or no) units fail during the
observation window. But failure is often the end point of a gradual,
measurable process — a crack grows, a resistance drifts, a lumen output fades,
a material wears. Degradation analysis exploits this: instead of waiting for
units to fail, we track a *degradation measurement* over time on each unit,
define failure as the measurement crossing a *threshold*, and infer a
failure-time distribution from the way the measurements evolve — even for units
that never actually failed on test.

There are two broad strategies, and they suit different physics.

The general-path (pseudo-failure-time) approach
-----------------------------------------------

The classic approach treats each unit's degradation as a smooth deterministic
curve observed with measurement error. It proceeds in three steps:

1. A degradation *path model* — a parametric curve such as a straight line — is
   fitted, by least squares, to each unit's measurements.
2. Each unit's fitted path is extrapolated to the failure threshold. The
   crossing time is that unit's *pseudo failure time*.
3. A lifetime distribution (Weibull, LogNormal, ...) is fitted to the pseudo
   failure times, and can then be used like any other parametric survival
   model.

If a unit's fitted path never reaches the threshold — it is not degrading, or
is trending away — the unit is treated as right censored at its last observed
time, and that censoring is carried through to the lifetime-distribution fit.

The path model is the load-bearing assumption: it is extrapolated well beyond
the observed data, so a shape with physical justification is always preferable
to one chosen on measurement fit alone. The available shapes, and the
threshold-crossing (pseudo failure) time each implies for a threshold
:math:`y_t`, are:

.. list-table::
    :header-rows: 1

    * - Name
      - Path
      - Threshold crossing time
    * - linear
      - :math:`y = a + b x`
      - :math:`(y_{t} - a) / b`
    * - quadratic
      - :math:`y = a + b x + c x^{2}`
      - first positive root of :math:`c t^{2} + b t + (a - y_{t})`
    * - exponential
      - :math:`y = a e^{b x}`
      - :math:`\ln(y_{t} / a) / b`
    * - offset-exponential
      - :math:`y = a + b e^{c x}`
      - :math:`\ln((y_{t} - a) / b) / c`
    * - power
      - :math:`y = a x^{b}`
      - :math:`(y_{t} / a)^{1/b}`
    * - logarithmic
      - :math:`y = a + b \ln(x)`
      - :math:`e^{(y_{t} - a) / b}`
    * - lloyd-lipow
      - :math:`y = a - b / x`
      - :math:`b / (a - y_{t})`
    * - gompertz
      - :math:`y = a e^{-b e^{-c x}}`
      - :math:`-\ln(-\ln(y_{t}/a)/b) / c`
    * - michaelis-menten
      - :math:`y = a x / (b + x)`
      - :math:`b y_{t} / (a - y_{t})`

Degradation can be increasing (crack length) or decreasing (luminous flux);
the direction is captured by the sign of the fitted parameters and needs no
special configuration. Models linear in their parameters are fitted in closed
form; the others by nonlinear least squares. When the physics does not dictate
a shape, one can be selected by information criterion (AICc pooled over all
units) — but only ever as a fallback, because the winner is extrapolated far
beyond the data.

Predicting a new unit, and remaining useful life
------------------------------------------------

A fitted model can forecast the failure time of a *new*, partially observed
unit from its own trajectory, by fitting the path shape to its measurements and
extrapolating to the same threshold. Trusting that single least-squares
extrapolation is dangerous when the trajectory is short or noisy, so a more
robust forecast blends the unit's own trend with the population. In this
Bayesian view the population path-parameter distribution is the *prior*, the
unit's measurements are the *likelihood*, and the posterior of the unit's path
parameters — pushed through the threshold crossing — gives the remaining useful
life (RUL) with a credible interval. A short or noisy trajectory is shrunk
toward the population's typical path, and as measurements accumulate the
forecast converges to the plain extrapolation.

The population path-parameter distribution
------------------------------------------

Beyond each individual unit, degradation analysis estimates the *population*
distribution of the path parameters — what a random-effects treatment (and any
blending of a new unit with the population) needs. The subtlety, identified by
Lu and Meeker [LuMeeker1993]_, is that each unit's fitted parameters are
least-squares *estimates*, so their scatter across units mixes two sources:
genuine unit-to-unit variability :math:`\Sigma` and per-unit estimation noise
:math:`V_i`,

.. math::

    \mathrm{Cov}(\hat{\theta}_i) = \Sigma + V_i.

The raw sample covariance therefore *overstates* the between-unit variability.
The Lu-Meeker two-stage correction pools the measurement variance from the
per-unit residuals, forms each unit's estimation covariance
:math:`V_i = \sigma^2 (J_i^{\top} J_i)^{-1}` from the path Jacobian, and
subtracts the average, recovering an estimate of :math:`\Sigma`. When the
estimation noise rivals the between-unit scatter (few units, or few
measurements per unit) the correction can go rank-deficient; the robust
alternative is to fit the random-effects model directly by **REML**, treating
each unit's parameters as draws :math:`\theta_i \sim \mathrm{MVN}(\mu, \Sigma)`.
For nonlinear paths REML uses the Lindstrom-Bates linearisation
[LindstromBates1990]_, which reduces to the exact fit on a linear path.

Induced failure-time distribution (Lu-Meeker)
---------------------------------------------

The pseudo-failure-time route extrapolates each unit to one (noisy) failure
time and fits a distribution to those times — simple and robust, but it discards
the fact that we have estimated the whole *population* of paths. The Lu-Meeker
**induced** failure-time distribution uses that population directly: once the
population path-parameter distribution :math:`\theta \sim N(\mu, \Sigma)` is
fitted, the failure time of a unit with parameters :math:`\theta` is
deterministic — the time its path crosses the threshold,
:math:`T(\theta) = \mathrm{inv\_path}(D; \theta)` — so the population
failure-time distribution is simply the distribution of :math:`T(\theta)`.
There is rarely a closed form, so it is evaluated by Monte Carlo. Its chief use
is as a **diagnostic**: the pseudo-failure fit and the induced distribution
reach the population life two different ways, and close agreement is reassuring
while a large gap warns that the path model or the Gaussian population
assumption is off.

Accelerated degradation: stress-dependent path parameters
---------------------------------------------------------

In an **accelerated degradation test** units are run at elevated stress so
they degrade fast enough to measure, and life is extrapolated back to use
conditions. The simplest treatment keeps the path fits as they are and lets
stress act only on the pseudo failure times, through a regression life model
(an accelerated-failure-time fit, say). That predicts life at a stress, but it
never models *why* life changes: the population of path parameters is pooled
across the stress levels, so it describes no unit actually tested.

The mechanistic alternative models the degradation **rate itself** as a
function of stress [Meeker1998]_. Each path parameter is placed on a *link
scale* — the identity, or the log for a parameter that must stay positive and
whose stress effect is multiplicative — and the link-scale parameters of unit
:math:`i`, tested at stress :math:`z_i`, are

.. math::

    \eta_i = D(z_i)\,\gamma + u_i, \qquad u_i \sim \mathrm{MVN}(0, \Sigma),
    \qquad \theta_i = h(\eta_i).

The design :math:`D(z)` gives each stress-dependent parameter an intercept and a
coefficient per covariate, and every other parameter an intercept only;
:math:`\gamma` holds those fixed effects and :math:`\Sigma` the unit-to-unit
scatter that remains *after* the stress effect is removed. A log-linked rate
with the covariate :math:`z = 1/T` is exactly the **Arrhenius** relationship,
:math:`\log b = \gamma_0 + \gamma_1 / T`. For a path that is linear in its
parameters with identity links this is still a linear mixed model, so the
two-stage (Lu-Meeker) and REML estimators above apply with a wider
fixed-effects design; a log link makes the path nonlinear in :math:`\eta`, and
the Lindstrom-Bates linearisation handles it as for any nonlinear path.

Modelling the mechanism buys two things the pooled population cannot give.
First, a **stress-conditional prior** for remaining-useful-life prediction: a
new unit running at stress :math:`z` is updated against
:math:`N(D(z)\gamma, \Sigma)`, the population of units at *its* stress, rather
than against a mixture of every stress tested. Second, a **stress-conditional
induced life**: drawing :math:`\eta \sim N(D(z)\gamma, \Sigma)` and pushing
each draw through the threshold crossing gives the failure-time distribution
at any stress — including stresses outside the tested range, where the
mechanism, not a curve fitted to the pseudo failure times, carries the
extrapolation.

Step-stress degradation: an accelerated clock
---------------------------------------------

Both treatments above assume each unit is tested at one stress. In a
**step-stress** test the stress is raised part way through, on the same units,
so a unit's path runs at several stresses in turn. The path then has to say how
it carries on at a step, and the natural answer is the **cumulative-exposure**
principle [Nelson1980]_: stress speeds up the unit's clock. A unit at stress
:math:`z` ages

.. math::

    \mathrm{AF}(z) = \exp\!\bigl(\gamma^\top (z - z_{\text{ref}})\bigr)

times faster than at the reference stress :math:`z_{\text{ref}}`, so under a
stress history :math:`z(s)` it has aged :math:`\tau(t) = \int_0^t
\mathrm{AF}(z(s))\,ds` of reference-stress time by calendar time :math:`t`, and
its path is the ordinary path model on that clock,

.. math::

    y_{ij} = g\bigl(\tau_i(t_{ij}); \theta_i\bigr) + \varepsilon_{ij},
    \qquad \theta_i \sim \mathrm{MVN}(\mu, \Sigma).

The path parameters and their population describe degradation at the reference
stress, and :math:`\gamma` how strongly stress speeds it up (with :math:`z =
1/T`, the Arrhenius relationship). It is the same clock the stochastic-process
models use for time-varying stress (below).

Why a clock rather than some other rule for the step? For a path that rises
with time, :math:`y = g(\tau)` means :math:`dy/dt = \mathrm{AF}(z)\,f(y)` with
:math:`f(y) = g'(g^{-1}(y))`: at every moment the unit degrades at a rate set by
its *current damage* and multiplied by the stress's acceleration factor. So the
clock model is also the natural *rate-based* model — the damage is carried over
at a step, and the unit continues from it at the new stress's rate. The two
rules only part company if stress changes the *shape* of the path rather than
its speed, which a step-stress profile cannot identify without a model for the
damage rate itself.

Estimating :math:`\gamma` needs care, because a unit held at a single stress can
absorb any acceleration into its own path parameters (every standard path family
is closed under rescaling time). The information comes from two places:

* **Units whose stress steps during the test.** The change of slope at a step
  fixes the acceleration factor between the two stresses. The two-stage
  estimate uses only this: for a trial :math:`\gamma` every unit's path is
  refitted on its clock, and :math:`\gamma` minimises the pooled residual sum
  of squares (profile least squares).
* **Units at different stresses, through the population.** Units share one
  population of path parameters, so the systematic difference between units
  run at different stresses identifies :math:`\gamma` too, even with no steps.
  The mixed-model estimate uses both sources: it maximises the approximate
  marginal likelihood (the Lindstrom-Bates first-order linearisation) over
  :math:`\gamma`, with the population refitted at each trial value.

Once :math:`\gamma` is estimated, everything else is the ordinary general-path
analysis on the clock. The pseudo failure times :math:`\tau_i^* =
g^{-1}(D; \theta_i)` are reference-stress lifetimes, a lifetime distribution
:math:`F_0` is fitted to them, and life under *any* stress history follows from
the clock, :math:`F(t) = F_0(\tau(t))` — at a constant stress simply
:math:`F_0(\mathrm{AF}(z)\,t)`, the accelerated-failure-time form.

A unit being monitored is predicted on its own clock too. Its measured
stress history gives the reference-stress time at each measurement, the
population :math:`N(\mu, \Sigma)` is the prior for its path parameters, and
each posterior draw's reference-stress failure time :math:`g^{-1}(D;
\theta)` is carried back to calendar time along the history and the planned
future stress. The rest of its life therefore depends on the stress plan,
not only on how degraded it is.

Stochastic-process degradation models
--------------------------------------

The general-path approach assumes each unit follows a smooth curve observed with
error. Sometimes the degradation is instead *random over time* — a crack that
jumps ahead in fits and starts, a wear signal that wanders. Then it is more
honest to model the **increments** of the degradation as a stochastic process,
and read the failure-time distribution off the process directly, as the
distribution of the **first time the process crosses the threshold** (the
*first-passage time*) [Meeker1998]_. Two processes cover the common cases, and
the choice between them is dictated by whether the degradation can *decrease*.

The Wiener process
~~~~~~~~~~~~~~~~~~~

A **Wiener process with drift** (Brownian motion with drift) models the
degradation as :math:`W(t) = \mu t + \sigma B(t)`, with :math:`B(t)` standard
Brownian motion. The **drift** :math:`\mu` is the average rate at which
degradation accumulates; the **diffusion** :math:`\sigma` is the size of the
random wobble around that trend. Over an interval :math:`\Delta t` the change is
Gaussian, :math:`\Delta W \sim N(\mu \Delta t, \sigma^2 \Delta t)`, which can be
negative — so the path may go down as well as up, making the Wiener process the
right model for noisy, non-monotone signals. Its first-passage time to a
threshold :math:`D` is **Inverse Gaussian** with

.. math::

    \text{mean} = \frac{D}{\mu}, \qquad \text{shape} = \frac{D^2}{\sigma^2}.

The mean life :math:`D/\mu` — distance to failure over average speed — is
intuitive, and the shape controls how tightly failure times cluster around it.
Because the increments are independent, the drift must be positive for the life
to be well defined. Wiener degradation models, including random-effect
extensions, are surveyed in [Wang2010]_.

The Gamma process
~~~~~~~~~~~~~~~~~

The Wiener process allows the signal to decrease, which is wrong for damage that
only ever *accumulates* — wear, corrosion, crack growth. The **Gamma process**
models such monotone degradation: its increments are Gamma distributed and
therefore always non-negative, so the path is non-decreasing by construction.
Like the Wiener process, its failure-time distribution is the first-passage time
to the threshold, obtained from the process parameters. Gamma processes are a
standard tool in maintenance modelling; see [vanNoortwijk2009]_ for a survey.

Stress and time-varying stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In an accelerated degradation test the stress — temperature, voltage, load — is
raised to make degradation happen faster, and in a **step-stress** test it is
raised part way through the test on the same units. Both process models handle
this through an **acceleration of the clock** (the cumulative-exposure idea of
[WhitmoreSchenkelberg1997]_): a unit held at stress :math:`z` ages at

.. math::

    \mathrm{AF}(z) = \exp\!\bigl(\gamma^\top (z - z_{\text{ref}})\bigr)

times the rate it would at the reference (use) stress :math:`z_{\text{ref}}`,
and under a stress profile :math:`z(s)` its **operational time** is

.. math::

    \tau(t) = \int_0^t \mathrm{AF}\bigl(z(s)\bigr)\, ds.

The process runs on :math:`\tau` instead of :math:`t`. Over a measurement interval
the Wiener increment becomes :math:`N(\mu\,\Delta\tau, \sigma^2\,\Delta\tau)` and
the Gamma increment :math:`\mathrm{Gamma}(\alpha\,\Delta\tau, \beta)`, so the
parameters :math:`(\mu, \sigma)` or :math:`(\alpha, \beta)` describe degradation at
the reference stress and :math:`\gamma` how strongly stress speeds it up. With
:math:`z = 1/T` (absolute temperature) the acceleration factor is Arrhenius, with
:math:`\gamma = -E_a / k`; with :math:`z = \log V` it is an inverse power law.

Because stress only changes the speed of the clock, the life under any stress
history is the reference life read at the operational time,
:math:`F(t) = F_0\bigl(\tau(t)\bigr)` — closed form for both processes, and a
simple rescaling of time, :math:`F(t) = F_0(\mathrm{AF}(z)\,t)`, at a constant
stress. For the Wiener process this assumes the stress scales the diffusion
along with the drift (the ratio :math:`\mu/\sigma^2` is stress-free), which is
what makes the time-scale model identifiable and the life closed form.

For worked examples of all of the above — fitting general-path and
stochastic-process models, predicting remaining useful life, the Lu-Meeker
diagnostic, and serialising a fitted model — see the
:doc:`Degradation Modelling with SurPyval` page.

Destructive degradation
-----------------------

Both approaches so far assume each unit is measured *repeatedly* over time. In a
**destructive** test the act of measuring destroys the specimen — a coupon must
be broken to read its strength, insulation driven to breakdown — so each unit
yields exactly **one** :math:`(t, y)` observation. With one point per unit there
are no paths to fit and no increments to accumulate, so neither the general-path
nor the stochastic-process machinery applies [Meeker1998]_.

Instead the **distribution of the degradation as a function of time** is modelled
directly. The measurement follows a location-scale distribution whose location
moves with a transform of time,

.. math::

    Y \mid t \ \sim\ \mathrm{dist}\bigl(\text{loc} = \beta_0 + \beta_1\,\varphi(t),
    \ \text{scale} = \sigma\bigr),

with :math:`\varphi` a time transform (identity, log, ...). This is an ordinary
censored location-scale regression of the response on :math:`\varphi(t)`, so
destructive measurements that are only bounded — a strength below the test floor,
a specimen that survived the maximum load — enter as left- or right-censored
observations in the usual way.

A unit fails when its degradation crosses the threshold :math:`D_f`. Because only
the location moves with time, the population ordering is preserved, and the
failure-time distribution is read straight off the fitted degradation
distribution: for **increasing** degradation (wear, crack growth)
:math:`F_T(t) = P(Y(t) > D_f)`, and for **decreasing** degradation (strength
loss) :math:`F_T(t) = P(Y(t) < D_f)`. This is the standard destructive-degradation
(degradation-distribution) model; see [Meeker1998]_.

References
----------

.. [Meeker1998] Meeker, W.Q. and Escobar, L.A., 1998. *Statistical Methods for
   Reliability Data*. John Wiley & Sons.

.. [LuMeeker1993] Lu, C.J. and Meeker, W.Q., 1993. Using degradation measures to
   estimate a time-to-failure distribution. *Technometrics*, 35(2), pp.161-174.

.. [Wang2010] Wang, X., 2010. Wiener processes with random effects for
   degradation data. *Journal of Multivariate Analysis*, 101(2), pp.340-351.

.. [vanNoortwijk2009] van Noortwijk, J.M., 2009. A survey of the application of
   gamma processes in maintenance. *Reliability Engineering & System Safety*,
   94(1), pp.2-21.

.. [Nelson1980] Nelson, W., 1980. Accelerated life testing — step-stress
   models and data analyses. *IEEE Transactions on Reliability*, R-29(2),
   pp.103-108.

.. [WhitmoreSchenkelberg1997] Whitmore, G.A. and Schenkelberg, F., 1997.
   Modelling accelerated degradation data using Wiener diffusion with a time
   scale transformation. *Lifetime Data Analysis*, 3(1), pp.27-45.

.. [LindstromBates1990] Lindstrom, M.J. and Bates, D.M., 1990. Nonlinear
   mixed effects models for repeated measures data. *Biometrics*, 46(3),
   pp.673-687.
