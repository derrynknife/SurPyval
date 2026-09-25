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

The idea that makes this work is simple. If failure *is* "the degradation
reached :math:`D`", then anything that tells you how degradation evolves also
tells you when it will reach :math:`D`. A unit that has drifted halfway to the
threshold in 500 hours, at a steady rate, will probably fail at around 1000
hours — no failure needed. Everything on this page is a more careful version of
that sentence: how to describe the evolution (a curve, or a random process),
how to account for units differing from each other, how to account for
measurement noise, and how to carry all of it through to a failure-time
distribution with honest uncertainty.

This page is the theory. Every model here is demonstrated, with runnable code,
on the :doc:`Degradation Modelling with SurPyval` page, which follows the same
order.

Choosing an approach
~~~~~~~~~~~~~~~~~~~~

Three families of model are available, and the choice is dictated mostly by
*how* you measure and *what the degradation physically does*:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Your data look like…
     - Use
     - Because
   * - each unit measured repeatedly, and each unit's trace is a smooth trend
       plus measurement scatter (units differ in their *rate*)
     - the **general-path** model (``DegradationAnalysis``)
     - the randomness is *between* units (each has its own curve), not along
       each unit's path
   * - each unit measured repeatedly, and each unit's trace wanders or jumps
       on its own
     - a **stochastic process** (``WienerProcess`` if the signal can go down,
       ``GammaProcess`` if it only ever increases)
     - the randomness is *along* the path: every interval adds an independent
       random amount of damage
   * - each unit measured **once**, because measuring destroys it
     - **destructive degradation** (``DestructiveDegradation``)
     - there are no paths, only the distribution of the measurement at each age

All three can be combined with stress: units run at elevated or changing stress
(temperature, voltage, load) to make degradation happen faster, and the model
then predicts life at the (lower) use stress. How stress enters is its own
topic, covered in `Accelerated degradation: stress-dependent path
parameters`_ and `Step-stress degradation: an accelerated clock`_.

Notation
~~~~~~~~

Used throughout the page:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Symbol
     - Meaning
   * - :math:`i`, :math:`j`
     - unit :math:`i = 1..n`, and its :math:`j`-th measurement
   * - :math:`t_{ij}`, :math:`y_{ij}`
     - the time and value of that measurement
   * - :math:`D`
     - the failure threshold: a unit has failed once its degradation reaches
       :math:`D`
   * - :math:`g(t; \theta)`
     - a *path model*: the degradation curve at time :math:`t` for a unit
       with path parameters :math:`\theta` (e.g. :math:`a + b t`)
   * - :math:`\theta_i`
     - unit :math:`i`'s path parameters; :math:`\hat\theta_i` its
       least-squares estimate
   * - :math:`\mu`, :math:`\Sigma`
     - the population mean and between-unit covariance of the path parameters
   * - :math:`\sigma^2`
     - the measurement-error variance around a unit's path
   * - :math:`z`, :math:`\gamma`
     - a stress (covariate) row, and the coefficients describing its effect
   * - :math:`\mathrm{AF}(z)`, :math:`\tau(t)`
     - the acceleration factor of stress :math:`z`, and the reference-stress
       time a unit has aged by calendar time :math:`t`

The general-path (pseudo-failure-time) approach
-----------------------------------------------

The classic approach treats each unit's degradation as a smooth deterministic
curve observed with measurement error:

.. math::

    y_{ij} = g(t_{ij}; \theta_i) + \varepsilon_{ij},
    \qquad \varepsilon_{ij} \sim N(0, \sigma^2).

Units differ because each has its own :math:`\theta_i` — one resistor drifts at
0.3 Ω per 1000 h, another at 0.4 — and the measurements scatter around each
unit's curve because instruments are noisy. The randomness that makes lifetimes
differ is therefore *between* units: once you know a unit's :math:`\theta_i`,
its failure time is fixed,

.. math::

    T_i = g^{-1}(D; \theta_i),

the time its curve reaches the threshold. The analysis proceeds in three steps:

1. **Fit a path to each unit.** The path model is fitted to each unit's
   measurements by least squares, giving :math:`\hat\theta_i`.
2. **Extrapolate to the threshold.** Each fitted path is solved for the time it
   reaches :math:`D`: the unit's *pseudo failure time*
   :math:`\hat T_i = g^{-1}(D; \hat\theta_i)`. "Pseudo" because it is a
   prediction, not an observed failure — usually well beyond the last
   measurement.
3. **Fit a lifetime distribution.** A lifetime distribution (Weibull by
   default; any SurPyval parametric distribution) is fitted to the pseudo
   failure times, and from then on is used like any other survival model.

If a unit's fitted path never reaches the threshold at a positive time — it is
not degrading, or is trending away from :math:`D` — it has no pseudo failure
time. Such a unit is treated as **right censored at its last observed time**
(it had not failed by then, as far as we know), and that censoring is carried
into the step-3 fit, with a warning so that you know it happened.

The path model
~~~~~~~~~~~~~~

The path model is the load-bearing assumption: it is extrapolated well beyond
the observed data, so a shape with physical justification (a diffusion-limited
process grows like :math:`\sqrt t`, a first-order reaction decays
exponentially, …) is always preferable to one chosen on how well it fits.
The available shapes, and the threshold-crossing (pseudo failure) time each
implies for a threshold :math:`y_t`, are:

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
the direction is carried by the sign of the fitted parameters and needs no
configuration.

Two properties of a path model matter later on:

* **Linear in its parameters.** The linear, quadratic, logarithmic and
  Lloyd-Lipow paths are linear in :math:`\theta` (for example
  :math:`a + b\ln x` is linear in :math:`(a, b)` even though it is not linear
  in :math:`x`). They are fitted by ordinary least squares in closed form, and
  the population models below are then *exact* linear mixed models. The
  others (exponential, power, Gompertz, …) are fitted by nonlinear least
  squares started from a linearised fit, and the population models use a
  linearisation.
* **Closed under rescaling time.** Every built-in family has the property that
  replacing :math:`t` by :math:`c\,t` gives another member of the same family
  (:math:`a + b\,(ct) = a + (bc)\,t`). This is what lets a unit held at one
  stress "absorb" an acceleration factor into its own parameters, which becomes
  important for step-stress tests.

When the physics does not dictate a shape, one can be selected by information
criterion: every registered path is fitted to every unit, the residual sums of
squares are pooled (assuming one common measurement variance), and the path with
the smallest AICc,

.. math::

    \mathrm{AICc} = N \ln(\mathrm{RSS}/N) + 2k + \frac{2k(k+1)}{N - k - 1},
    \qquad k = n \times (\text{parameters per path}) + 1,

wins (:math:`N` is the total number of measurements, and the ``+ 1`` counts the
common variance). Treat this only as a fallback: the winner is extrapolated far
beyond the data, and a better fit inside the data says little about behaviour
outside it.

Predicting a new unit, and remaining useful life
------------------------------------------------

A fitted model can forecast the failure time of a *new*, partially observed
unit from its own trajectory. The simplest forecast repeats steps 1–2 for the
new unit — fit the path to its measurements, extrapolate to :math:`D` — and the
**remaining useful life** (RUL) is that failure time minus the unit's current
age.

Trusting that single least-squares extrapolation is dangerous when the
trajectory is short or noisy: two measurements a little too close together can
imply a slope of nearly zero and a failure time of centuries. The robust
forecast blends the unit's own trend with what the population says is
plausible. In Bayesian terms:

* the **prior** for the new unit's path parameters is the population,
  :math:`\theta \sim N(\mu, \Sigma)` (estimated as described in the next
  section);
* the **likelihood** is its measurements,
  :math:`y_j \sim N(g(t_j; \theta), \sigma^2)`;
* the **posterior** of :math:`\theta` combines the two.

For a path that is linear in its parameters, :math:`g(t; \theta) = J\theta`
with a fixed design matrix :math:`J`, the posterior is exactly Gaussian (the
conjugate update):

.. math::

    \Lambda = \Sigma^{-1} + \frac{J^\top J}{\sigma^2},
    \qquad
    \theta \mid y \sim N\!\left(\Lambda^{-1}\Bigl(\Sigma^{-1}\mu
        + \frac{J^\top y}{\sigma^2}\Bigr),\ \Lambda^{-1}\right).

The posterior precision :math:`\Lambda` adds the population's precision to the
data's; the posterior mean is the precision-weighted compromise between the
population mean :math:`\mu` and the unit's own least-squares fit. With one or
two noisy points the population term dominates and the unit is *shrunk* toward
the typical path; as measurements accumulate :math:`J^\top J/\sigma^2` grows
without bound and the posterior converges on the plain extrapolation. For a
nonlinear path the same update is iterated, re-linearising :math:`g` about the
current estimate until it settles (a Laplace approximation to the posterior).

The posterior is then pushed through the threshold crossing by Monte Carlo:
draw many :math:`\theta` from the posterior, compute :math:`g^{-1}(D; \theta)`
for each, and read the median and a credible interval off the draws. Two
probabilities come along for free: the fraction of draws whose failure time is
already in the past (the unit has *probably* already crossed :math:`D`), and the
fraction whose path never reaches :math:`D` at all.

The population path-parameter distribution
------------------------------------------

Both the Bayesian forecast and the induced life (next section) need the
**population** of path parameters: the model that each unit's
:math:`\theta_i` is a draw from a common distribution,

.. math::

    \theta_i \sim \mathrm{MVN}(\mu, \Sigma).

:math:`\mu` is the typical unit and :math:`\Sigma` how much units differ. Put
together with the measurement model above this is a **random-effects** (mixed)
model: fixed population parameters :math:`(\mu, \Sigma, \sigma^2)`, and one
random :math:`\theta_i` per unit. Two ways of estimating it are available.

The two-stage (moments) estimate
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The obvious estimate — the sample mean and covariance of the fitted
:math:`\hat\theta_i` — has a flaw identified by Lu and Meeker
[LuMeeker1993]_. Each :math:`\hat\theta_i` is a least-squares *estimate*, so
its scatter across units mixes two sources: genuine unit-to-unit variability
:math:`\Sigma` and the unit's own estimation noise :math:`V_i`,

.. math::

    \mathrm{Cov}(\hat{\theta}_i) = \Sigma + V_i.

The raw sample covariance therefore *overstates* how much units really differ —
badly so when each unit has few, noisy measurements. The two-stage correction
estimates and removes the noise part:

1. pool the measurement variance from every unit's residuals,
   :math:`\hat\sigma^2 = \sum \mathrm{RSS}_i / \sum (n_i - p)`;
2. form each unit's least-squares estimation covariance from its path Jacobian
   :math:`J_i` (the matrix of derivatives :math:`\partial g/\partial\theta` at
   its measurement times), :math:`V_i = \hat\sigma^2 (J_i^{\top} J_i)^{-1}`;
3. subtract the average: :math:`\hat\Sigma = S - \bar V`, with :math:`S` the
   sample covariance of the :math:`\hat\theta_i`.

A difference of covariance matrices need not be a covariance matrix: when the
estimation noise rivals the between-unit scatter (few units, or few
measurements per unit) :math:`S - \bar V` can have negative eigenvalues. These
are clipped to zero and a warning is raised, because the estimate is then
unreliable.

The REML estimate
~~~~~~~~~~~~~~~~~

The robust alternative fits the random-effects model directly. For a path that
is linear in its parameters, :math:`y_i = X_i\theta_i + \varepsilon_i`, and the
random :math:`\theta_i` can be integrated out: each unit's measurement vector is
marginally

.. math::

    y_i \sim N\bigl(X_i \mu,\ V_i\bigr),
    \qquad V_i = X_i \Sigma X_i^{\top} + \sigma^2 I.

That is a likelihood for :math:`(\mu, \Sigma, \sigma^2)` from all the raw
measurements at once — no per-unit fits, no subtraction, and a :math:`\Sigma`
that is positive definite by construction (it is parameterised by its Cholesky
factor). Plain maximum likelihood for variance components is biased low in
small samples, for the same reason the sample variance divides by
:math:`n - 1`: it ignores that :math:`\mu` was estimated from the same data.
**REML** (restricted maximum likelihood) fixes this by maximising the
likelihood of the part of the data that does not depend on :math:`\mu`; in
practice it adds a :math:`\log\det(\sum X_i^\top V_i^{-1} X_i)` term to the
objective, with :math:`\mu` profiled out by generalised least squares. On a
balanced design (every unit measured at the same times) REML and the corrected
moments estimate coincide; they differ, and REML is the better estimate, on
unbalanced data and with few units.

A **nonlinear** path has no fixed :math:`X_i`, and the marginal likelihood has
no closed form. The Lindstrom-Bates algorithm [LindstromBates1990]_ (known as
FOCE, first-order conditional estimation) alternates two steps until nothing
changes:

1. with the population fixed, find each unit's most probable parameters — its
   *conditional mode*, the penalised least-squares fit that balances its data
   against the population prior;
2. linearise the path about those modes (a first-order Taylor expansion), which
   turns the problem into a linear mixed model with pseudo-data, and take one
   REML step on it to update :math:`(\mu, \Sigma, \sigma^2)`.

On a path that is already linear the linearisation is exact and the loop stops
after one pass, so the two routes agree. Computationally each REML evaluation
uses the Woodbury identity, which turns every unit's
:math:`n_i \times n_i` covariance :math:`V_i` into a
:math:`p \times p` problem (:math:`p` the number of path parameters), so the
cost does not grow with the number of measurements per unit.

Induced failure-time distribution (Lu-Meeker)
---------------------------------------------

The pseudo-failure-time route extrapolates each unit to one (noisy) failure
time and fits a distribution to those times — simple and robust, but it discards
the fact that we have estimated the whole *population* of paths. The Lu-Meeker
**induced** failure-time distribution uses that population directly: a unit
with parameters :math:`\theta` fails at :math:`T(\theta) = g^{-1}(D; \theta)`,
so the population failure-time distribution is the distribution of
:math:`T(\theta)` when :math:`\theta \sim N(\mu, \Sigma)`,

.. math::

    F(t) = P\bigl(g^{-1}(D; \theta) \le t\bigr),
    \qquad \theta \sim N(\mu, \Sigma).

There is rarely a closed form, so it is evaluated by Monte Carlo: draw many
:math:`\theta`, compute each :math:`T(\theta)`, and use the empirical
distribution of the draws.

Some draws may describe a path that never reaches :math:`D` (a non-increasing
slope, say). They contribute an infinite failure time — a *defective* "never
fails" mass. That is a real prediction, not a numerical accident: if a fraction
of the population never fails, the population has no finite mean life, and
quantiles that reach into that fraction are infinite too.

The induced distribution's chief use is as a **diagnostic**. The
pseudo-failure fit and the induced distribution reach the population life two
different ways — one through a parametric family fitted to extrapolated times,
the other through the path model and a Gaussian population. Close agreement is
reassuring; a large gap warns that the path model, the Gaussian population
assumption, or the covariance estimate is off.

Uncertainty: two-stage confidence bounds
----------------------------------------

The pseudo-failure-time approach is a **two-stage** estimator: stage one
estimates each unit's path, stage two fits a lifetime distribution to the
extrapolated failure times *as if they had been observed exactly*. The
ordinary confidence bounds of the stage-two fit therefore only reflect the
scatter between the pseudo failure times, and miss that each one is itself an
uncertain extrapolation. They are too narrow — sometimes much too narrow when
the extrapolation is long.

Two corrections are available.

* **Analytic (delta method / generated regressor).** Each pseudo failure time
  :math:`\hat T_i` carries a variance :math:`v_i`, propagated from the unit's
  path-fit covariance through :math:`g^{-1}` by the delta method. If
  :math:`\hat T_i` moved a little, the life-model estimate :math:`\hat\phi`
  would move by :math:`\partial\hat\phi/\partial T_i` (by the implicit function
  theorem, :math:`H^{-1}` times the derivative of the score, with :math:`H` the
  life model's observed information). The corrected covariance adds the
  first-stage contribution to the usual one,

  .. math::

      \mathrm{Cov}(\hat\phi) = H^{-1}
          + \sum_i v_i
          \frac{\partial\hat\phi}{\partial T_i}
          \frac{\partial\hat\phi}{\partial T_i}^{\!\top},

  and bounds on survival (or anything else) follow by one more delta step,
  taken on the logit scale so they stay inside :math:`(0, 1)`. It is fast — no
  refitting.
* **Bootstrap.** Resample whole units with replacement, rerun the entire
  pipeline (path fits, pseudo failure times, life fit) on each resample, and
  take percentiles of the resulting curves. Slower, but it makes no
  first-order approximation, so it is the better choice with few units or long
  extrapolations, and a useful check on the analytic bounds.

For the accelerated models further down, the analytic correction is not
derived (the regression life fit, or the estimated clock, adds uncertainty the
formula above does not capture), and the bootstrap is the method: units are
resampled together with their stresses and the whole accelerated fit is rerun
each time.

Accelerated degradation: stress-dependent path parameters
---------------------------------------------------------

In an **accelerated degradation test** (ADT) units run at elevated stress —
temperature, voltage, humidity, load — so that they degrade fast enough to
measure in a test of reasonable length, and the life at the (lower) *use*
stress is then extrapolated from a model of how stress speeds things up.
SurPyval offers three ways for stress to enter a general-path model. They
answer increasingly detailed questions, and they need increasingly specific
data:

.. list-table::
   :header-rows: 1
   :widths: 18 30 26 26

   * - Stress acts on…
     - Model
     - Data needed
     - Gives you
   * - the **pseudo failure times** (``Z``)
     - a regression life model (AFT) fitted to the pseudo failure times
     - each unit at one constant stress; at least two stress levels
     - life at any stress
   * - the **path parameters** (``Z`` and ``links``)
     - each unit's path parameters depend on its stress through a link
     - each unit at one constant stress; at least two levels
     - life at any stress *and* the degradation mechanism: how the rate
       changes with stress, a population of paths at each stress, and
       stress-aware remaining-life prediction
   * - the **clock** (``Z`` and ``acceleration="clock"``)
     - stress speeds up time for the whole path
     - stress may change *during* a unit's test (step-stress), as well as
       between units
     - all of the above, under any stress history, including remaining life
       on a planned future stress profile

The first two are described here and the clock in the next section.

Stress on the life: a regression on the pseudo failure times
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The simplest treatment keeps the path fits exactly as before and lets stress act
only in step three: instead of a plain distribution, an accelerated failure time
(AFT) regression is fitted to the pseudo failure times, with each unit's stress
row :math:`z_i` as its covariate. In SurPyval's AFT parameterisation the
cumulative hazard is

.. math::

    H(t \mid z) = H_0\bigl(e^{\beta^\top z}\, t\bigr),

so a unit at stress :math:`z` lives :math:`e^{\beta^\top z}` times *shorter* than
one at :math:`z = 0` (a positive coefficient means higher stress, shorter life).
Any SurPyval regression fitter can be used instead of AFT.

This predicts life at any stress, but it never models *why* life changes. The
population of path parameters is pooled across the stress levels — it mixes
fast units at high stress with slow ones at low stress — so it describes no
unit actually tested, and it cannot be used as a prior for a new unit or to
induce the life at a stress.

Stress on the mechanism: path parameters that depend on stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The mechanistic alternative models the degradation **rate itself** as a
function of stress [Meeker1998]_ — the way a physicist would: "the reaction rate
follows Arrhenius in temperature, the initial state does not depend on
temperature". Each path parameter is placed on a **link scale** — the identity,
or the log for a parameter that must stay positive and whose stress effect is
multiplicative — and the link-scale parameters of unit :math:`i`, tested at
stress :math:`z_i`, are

.. math::

    \eta_i = D(z_i)\,\gamma + u_i, \qquad u_i \sim \mathrm{MVN}(0, \Sigma),
    \qquad \theta_i = h(\eta_i),

with :math:`h` the inverse links (identity, or :math:`\exp`). Read it as:

* :math:`D(z)\gamma` is the **typical** unit at stress :math:`z`. The design
  :math:`D(z)` gives each stress-dependent parameter an intercept and a
  coefficient per covariate, and every other parameter an intercept only.
* :math:`u_i` is unit :math:`i`'s own deviation from that typical unit, and
  :math:`\Sigma` the unit-to-unit scatter that remains *after* the stress
  effect is removed — the same at every stress.

A concrete case makes :math:`D` less abstract. For the linear path
:math:`a + b\,t` with the rate log-linked on one covariate,
``links={"b": "log"}``, the link-scale parameters are :math:`(a, \log b)` and

.. math::

    \begin{pmatrix} a_i \\ \log b_i \end{pmatrix}
    = \underbrace{\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & z_i \end{pmatrix}}_{D(z_i)}
      \begin{pmatrix} \gamma_a \\ \gamma_{b,0} \\ \gamma_{b,1} \end{pmatrix}
      + u_i,

so :math:`\gamma` holds three fixed effects — the intercept :math:`a`, the
log-rate intercept, and the log-rate's slope in stress. SurPyval labels them
``a``, ``log(b)`` and ``log(b):Z0`` (``:Zj`` for the coefficient on covariate
:math:`j`).

With the covariate :math:`z = 1/T` (absolute temperature) a log-linked rate is
exactly the **Arrhenius** relationship,

.. math::

    \log b = \gamma_{b,0} + \gamma_{b,1}\,\frac{1}{T},
    \qquad \gamma_{b,1} = -\frac{E_a}{k},

with :math:`E_a` the activation energy and :math:`k` Boltzmann's constant
(:math:`8.617\times10^{-5}` eV/K). A log-linked rate with :math:`z = \log V`
is an inverse power law in voltage.

**Estimation** follows the two population routes above, with the wider
fixed-effects design. The two-stage estimate fits each unit's path on the link
scale, regresses those per-unit estimates on their designs :math:`D(z_i)` by
least squares to get :math:`\gamma`, and takes :math:`\Sigma` as the covariance
of the regression residuals less the average link-scale estimation covariance.
REML fits the mixed model directly: with identity links on a path linear in its
parameters it is still an exact linear mixed model (fixed-effects design
:math:`X_i D(z_i)`), and a log link makes the path nonlinear in :math:`\eta`, so
the Lindstrom-Bates linearisation handles it as for any nonlinear path.

The life model is still the regression of the previous subsection, so life at a
stress is available exactly as before. What the mechanism adds is a **population
of paths at each stress**, which buys two things the pooled population cannot
give:

* a **stress-conditional prior** for remaining-useful-life prediction: a new
  unit running at stress :math:`z` is updated against
  :math:`N(D(z)\gamma, \Sigma)`, the population of units at *its* stress,
  rather than against a mixture of every stress tested. The update is done on
  the link scale, so a log-linked rate stays positive;
* a **stress-conditional induced life**: drawing
  :math:`\eta \sim N(D(z)\gamma, \Sigma)` and pushing each draw through the
  threshold crossing gives the failure-time distribution at any stress —
  including stresses outside the tested range, where the mechanism, not a
  curve fitted to the pseudo failure times, carries the extrapolation.

Because each link is monotone and each link-scale parameter normal, the
population *median* of each parameter at stress :math:`z` is exactly
:math:`h(D(z)\gamma)`: for a log-linked rate that is the geometric-mean rate (the
rate's mean is larger, by the log-normal factor).

Step-stress degradation: an accelerated clock
---------------------------------------------

Both treatments above assume each unit is tested at one stress. In a
**step-stress** test the stress is raised part way through, on the same units —
50 °C for 100 hours, then 75 °C, then 100 °C — so each unit's path runs at
several stresses in turn. That is efficient (every unit contributes information
at every level, and failures come sooner) but it raises a question the
constant-stress models never had to answer: *how does a path carry on at a
step?*

The clock
~~~~~~~~~

The natural answer is the **cumulative-exposure** principle [Nelson1980]_:
stress does not change what degradation looks like, it changes how *fast* it
happens — it speeds up the unit's clock. A unit at stress :math:`z` ages

.. math::

    \mathrm{AF}(z) = \exp\!\bigl(\gamma^\top (z - z_{\text{ref}})\bigr)

times faster than at the reference stress :math:`z_{\text{ref}}` (usually the
use condition, where :math:`\mathrm{AF} = 1`). Under a stress history
:math:`z(s)` it has therefore aged

.. math::

    \tau(t) = \int_0^t \mathrm{AF}\bigl(z(s)\bigr)\,ds

of reference-stress time by calendar time :math:`t` — an hour at 100 °C might
count as eight hours at 50 °C — and its path is the ordinary path model on that
clock,

.. math::

    y_{ij} = g\bigl(\tau_i(t_{ij}); \theta_i\bigr) + \varepsilon_{ij},
    \qquad \theta_i \sim \mathrm{MVN}(\mu, \Sigma).

Everything about the path — :math:`\theta_i`, its population, the pseudo failure
times — now describes degradation at the reference stress, and :math:`\gamma`
describes how strongly stress speeds it up. With :math:`z = 1/T` this is again
Arrhenius, :math:`\mathrm{AF} = \exp\bigl(\tfrac{E_a}{k}(\tfrac{1}{T_{\text{ref}}}
- \tfrac{1}{T})\bigr)` with :math:`\gamma = -E_a/k`: a coefficient of
:math:`-5000` is an activation energy of about 0.43 eV. It is the same clock the
stochastic-process models use for time-varying stress (below), and at a constant
stress it is exactly the AFT form of the first accelerated model, now estimated
from the paths rather than from pseudo failure times.

In data, stress is recorded per measurement: the stress row attached to a
measurement is the stress applied over the interval that **ends** at it, the
first interval starting at time zero. For a unit inspected every 10 hours with
the chamber stepped up just after the 100-hour inspection, the rows for
:math:`t = 10, \dots, 100` carry 50 °C and those from :math:`t = 110` carry
75 °C. The unit's clock at its :math:`j`-th measurement is then the sum
:math:`\tau_{ij} = \sum_{k \le j} \mathrm{AF}(z_{ik})\,(t_{ik} - t_{i,k-1})`.

Why a clock rather than some other rule for the step? For a path that rises
with time, :math:`y = g(\tau)` means

.. math::

    \frac{dy}{dt} = \mathrm{AF}(z)\,f(y), \qquad f(y) = g'\bigl(g^{-1}(y)\bigr):

at every moment the unit degrades at a rate set by its *current damage* and
multiplied by the stress's acceleration factor. So the clock model is also the
natural *rate-based* model — the damage is carried over at a step, and the unit
continues from it at the new stress's rate, without a jump. The two rules would
only part company if stress changed the *shape* of the path rather than its
speed, which a step-stress profile cannot identify without a separate model of
the damage rate; so stress-dependent *links* are not combined with the clock.

Estimating the acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Estimating :math:`\gamma` needs care. Because every built-in path family is
closed under rescaling time, a unit held at a single stress can absorb *any*
acceleration into its own path parameters: at constant stress
:math:`a + b\,(\mathrm{AF}\cdot t)` is just a line with slope
:math:`b\,\mathrm{AF}`, and on its own that unit cannot tell a fast unit from an
accelerated one. The information about :math:`\gamma` comes from two places:

* **Units whose stress steps during the test.** A unit's own slope changes by
  exactly :math:`\mathrm{AF}(z_2)/\mathrm{AF}(z_1)` at a step from :math:`z_1` to
  :math:`z_2` — its own path parameters cannot absorb that, because they are
  shared by both segments. The two-stage (``moments``) estimate uses only this:
  for a trial :math:`\gamma` every unit's path is refitted on its clock, and
  :math:`\gamma` minimises the pooled residual sum of squares (*profile least
  squares*). With no stepped units it has nothing to go on, and refuses.
* **Units at different stresses, through the population.** Units share one
  population of path parameters, so if the units at 100 °C all look eight times
  faster than the units at 50 °C, that is not unit-to-unit variation — it is
  the stress. The mixed-model (``reml``) estimate uses both sources, so it also
  works for a classic constant-stress test with no steps at all. It maximises
  the approximate (Lindstrom-Bates) marginal likelihood of all the
  measurements over :math:`\gamma`, refitting the population at each trial
  value. A fast joint iteration, with :math:`\gamma` treated as one more fixed
  effect, first finds the neighbourhood; it would then crawl along the ridge on
  which a unit's rate and :math:`\gamma` nearly trade off, so the profile
  likelihood of :math:`\gamma` is maximised directly. That profile uses the
  plain likelihood rather than REML, because :math:`\gamma` changes the
  fixed-effects design and REML likelihoods for different designs are not
  comparable; the population is then estimated by REML at the chosen
  :math:`\gamma`.

Life and prediction on the clock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once :math:`\gamma` is estimated, everything else is the ordinary general-path
analysis on the clock. The pseudo failure times
:math:`\hat\tau_i^* = g^{-1}(D; \hat\theta_i)` are *reference-stress*
lifetimes, and a lifetime distribution :math:`F_0` is fitted to them. Life under
**any** stress history follows from the clock,

.. math::

    F(t) = F_0\bigl(\tau(t)\bigr),

because a unit that has aged :math:`\tau(t)` reference hours has exactly the
reference-stress chance of having failed by then. At a constant stress this is
:math:`F_0(\mathrm{AF}(z)\,t)`: every quantile divides by :math:`\mathrm{AF}`.
Under a profile, the density picks up the clock's rate,
:math:`f(t) = f_0(\tau(t))\,\mathrm{AF}(z(t))`, and the mean life is the
integral of :math:`1 - F(t)`.

A unit being **monitored** is predicted on its own clock. Its measured stress
history gives the reference-stress time at each of its measurements; the
population :math:`N(\mu, \Sigma)` is the prior for its path parameters, updated
by its measurements exactly as in the Bayesian forecast above; and each
posterior draw's reference-stress failure time :math:`g^{-1}(D; \theta)` is
carried back to calendar time — through its history if it has already been
reached, and otherwise through the *planned* future stress from now on. Its
remaining life therefore depends on the stress plan, not only on how degraded it
is: the same unit has a shorter remaining life if you intend to run it hotter.

Stochastic-process degradation models
--------------------------------------

The general-path approach assumes each unit follows a smooth curve observed with
error, with all the randomness *between* units. Sometimes the degradation is
instead *random over time* — a crack that jumps ahead in fits and starts, a wear
signal that wanders — and two units that are identical in every respect still
fail at different times because of the luck of the draw along the way. Then it
is more honest to model the **increments** of the degradation as a stochastic
process, and read the failure-time distribution off the process directly, as the
distribution of the **first time the process crosses the threshold** (the
*first-passage time*) [Meeker1998]_.

Both processes below have *independent increments*: the change over an
interval does not depend on what happened before it. That has three practical
consequences. Fitting uses only the increments
:math:`\Delta y = y_{j} - y_{j-1}` over :math:`\Delta t = t_j - t_{j-1}`, so
units measured at different, irregular times need no special handling. The
failure-time distribution comes from the fitted process in closed form, with no
pseudo failure times. And a unit's **remaining life** from its current level
:math:`y` is simply a fresh first passage over the remaining distance
:math:`D - y`. Two processes cover the common cases, and the choice between them
is dictated by whether the degradation can *decrease*.

The Wiener process
~~~~~~~~~~~~~~~~~~~

A **Wiener process with drift** (Brownian motion with drift) models the
degradation as :math:`W(t) = \mu t + \sigma B(t)`, with :math:`B(t)` standard
Brownian motion. The **drift** :math:`\mu` is the average rate at which
degradation accumulates; the **diffusion** :math:`\sigma` is the size of the
random wobble around that trend. Over an interval :math:`\Delta t` the change is
Gaussian, :math:`\Delta W \sim N(\mu \Delta t, \sigma^2 \Delta t)`, which can be
negative — so the path may go down as well as up, making the Wiener process the
right model for noisy, non-monotone signals.

**Fitting.** The increments are independent Gaussians, so the maximum
likelihood estimates have closed forms:

.. math::

    \hat\mu = \frac{\sum \Delta y}{\sum \Delta t},
    \qquad
    \hat\sigma^2 = \frac{1}{m}\sum \frac{(\Delta y - \hat\mu\,\Delta t)^2}{\Delta t},

over all :math:`m` increments of all units: the drift is total degradation over
total time, and the diffusion measures the scatter of the increments around
that trend, each scaled by its own interval length.

**Life.** The first-passage time to a threshold :math:`D` is **Inverse
Gaussian** with

.. math::

    \text{mean} = \frac{D}{\mu}, \qquad \text{shape} = \frac{D^2}{\sigma^2}.

The mean life :math:`D/\mu` — distance to failure over average speed — is
intuitive, and the shape controls how tightly failure times cluster around it
(more diffusion, more scatter). The drift must be positive for the life to be
well defined: with :math:`\mu \le 0` the process is not reliably heading toward
the threshold, so the fit refuses. Wiener degradation models, including
random-effect extensions, are surveyed in [Wang2010]_.

The Gamma process
~~~~~~~~~~~~~~~~~

The Wiener process allows the signal to decrease, which is wrong for damage that
only ever *accumulates* — wear, corrosion, crack growth. The **Gamma process**
models such monotone degradation: its increment over :math:`\Delta t` is Gamma
distributed,

.. math::

    \Delta W \sim \mathrm{Gamma}\bigl(\text{shape} = \alpha\,\Delta t,\
        \text{rate} = \beta\bigr),

and therefore never negative, so the path is non-decreasing by construction.
Its mean degradation per unit time is :math:`\alpha/\beta` (the analogue of the
Wiener drift) and its variance per unit time :math:`\alpha/\beta^2`: a large
:math:`\alpha` gives many small, regular increments, a small one fewer, larger,
jumpier ones. Gamma processes are a standard tool in maintenance modelling; see
[vanNoortwijk2009]_ for a survey.

**Fitting.** The likelihood of the increments is a product of Gamma densities.
For a given :math:`\alpha` the best :math:`\beta` has a closed form,
:math:`\hat\beta = \alpha \sum\Delta t / \sum \Delta y` (it matches the mean
rate), so the fit is a one-dimensional search over :math:`\alpha` with
:math:`\beta` profiled out. A decrease between two measurements is impossible
under this model, so data with one is refused, with a pointer to the Wiener
process.

**Life.** Because the path only goes up, it has crossed :math:`D` by time
:math:`t` exactly when its level at :math:`t` is at least :math:`D`:

.. math::

    F(t) = P\bigl(W(t) \ge D\bigr) = Q(\alpha t,\ \beta D),

the regularised upper incomplete gamma function. The density, quantiles and mean
follow from it numerically.

Stress and time-varying stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In an accelerated degradation test the stress — temperature, voltage, load — is
raised to make degradation happen faster, and in a **step-stress** test it is
raised part way through the test on the same units. Both process models handle
this through the same **acceleration of the clock** as the general-path model
(for the Wiener process this is the time-scale transformation of
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

Fitting maximises the same increment likelihood with :math:`\Delta\tau` in place
of :math:`\Delta t`: given :math:`\gamma` the Wiener parameters still have the
closed forms above and the Gamma rate is still profiled out, so only
:math:`\gamma` (and the Gamma shape) needs a numerical search. As for the
general-path clock, :math:`\gamma` is identified by stress that differs across
the measurement intervals — between units or within them — and at least two
distinct stress levels are required.

Because stress only changes the speed of the clock, the life under any stress
history is the reference life read at the operational time,
:math:`F(t) = F_0\bigl(\tau(t)\bigr)` — closed form for both processes, and a
simple rescaling of time, :math:`F(t) = F_0(\mathrm{AF}(z)\,t)`, at a constant
stress. For the Wiener process this assumes the stress scales the diffusion
along with the drift (the ratio :math:`\mu/\sigma^2` is stress-free), which is
what makes the time-scale model identifiable and the life closed form.

Choosing between a path and a process
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The two approaches put the randomness in different places, and the question to
ask of your data is *where the spread in lifetimes comes from*. If each unit's
trace is a clean trend with small scatter and the traces fan out at different
slopes, the spread comes from differences between units: use a general path. If
each trace wanders so much that a smooth curve through it would be a fiction,
the spread comes from the process itself: use a Wiener process if it can go down
and a Gamma process if it cannot. The process models handle irregular
measurement times without special treatment and give the life distribution
directly from the fitted process, rather than through noisy pseudo failure
times; the general-path models capture unit-to-unit differences in rate and
curve shape explicitly, which the (homogeneous) process models do not.

Destructive degradation
-----------------------

All the approaches so far assume each unit is measured *repeatedly* over time.
In a **destructive** test the act of measuring destroys the specimen — a coupon
must be broken to read its strength, insulation driven to breakdown — so each
unit yields exactly **one** :math:`(t, y)` observation. With one point per unit
there are no paths to fit and no increments to accumulate, so neither the
general-path nor the stochastic-process machinery applies [Meeker1998]_.

Instead the **distribution of the degradation as a function of time** is modelled
directly. The measurement follows a location-scale distribution whose location
moves with a transform of time,

.. math::

    Y \mid t \ \sim\ \mathrm{dist}\bigl(\text{loc} = \beta_0 + \beta_1\,\varphi(t),
    \ \text{scale} = \sigma\bigr),

with :math:`\varphi` a time transform (identity, log, square root, reciprocal)
chosen to make the trend in location linear. "Location-scale" covers the Normal
(location = mean) and, through the log of the measurement, the LogNormal — the
usual choice for a positive quantity such as a strength, where the scatter grows
with the level. This is an ordinary censored location-scale regression of the
response on :math:`\varphi(t)`, fitted by maximum likelihood, so destructive
measurements that are only bounded — a strength below the test floor, a
specimen that survived the maximum load — enter as left- or right-censored
observations in the usual way.

A unit fails when its degradation crosses the threshold :math:`D_f`. Because only
the location moves with time, the population ordering is preserved: a specimen
in the weakest 10 % at one age is in the weakest 10 % at every age. So the
failure-time distribution is read straight off the fitted degradation
distribution: for **increasing** degradation (wear, crack growth)
:math:`F_T(t) = P(Y(t) > D_f)`, and for **decreasing** degradation (strength
loss) :math:`F_T(t) = P(Y(t) < D_f)`. The direction is inferred from the sign of
the fitted trend unless it is given. This is the standard destructive-degradation
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
