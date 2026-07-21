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

For worked examples of all of the above — fitting general-path and
stochastic-process models, predicting remaining useful life, the Lu-Meeker
diagnostic, and serialising a fitted model — see the
:doc:`Degradation Modelling with SurPyval` page.

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

.. [LindstromBates1990] Lindstrom, M.J. and Bates, D.M., 1990. Nonlinear
   mixed effects models for repeated measures data. *Biometrics*, 46(3),
   pp.673-687.
