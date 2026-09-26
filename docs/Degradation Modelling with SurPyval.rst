Degradation Modelling with SurPyval
===================================

Sometimes failure data is scarce or takes too long to collect: items are
highly reliable, test time is limited, and few (or no) units fail during
the observation window. But failure is often the end point of a gradual,
measurable process — a crack grows, a resistance drifts, a lumen output
fades, a material wears. Degradation analysis exploits this: instead of
waiting for units to fail, we track a *degradation measurement* over time
on each unit, define failure as the measurement crossing a *threshold*,
and work out the failure-time distribution from how the measurements
evolve — even for units that never actually failed on test.

This page shows how to do all of it with SurPyval, with runnable examples. The
concepts and the mathematics behind each model — why it works, what it
assumes, how it is estimated — are on the :doc:`Degradation Analysis` page,
which follows the same order; it is worth reading the matching section there
alongside each section here.

What is on this page
~~~~~~~~~~~~~~~~~~~~

There are three families of model, each a fitter whose ``fit`` returns a fitted
model object:

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Fitter
     - Returns
     - Use when
   * - :class:`DegradationAnalysis <surpyval.degradation.degradation_analysis.DegradationAnalysis_>`
     - :class:`~surpyval.degradation.degradation_analysis.DegradationModel`
     - each unit is measured repeatedly and follows a smooth trend (the
       *general-path* model) — with optional stress effects
   * - :class:`~surpyval.degradation.process_models.WienerProcess`,
       :class:`~surpyval.degradation.process_models.GammaProcess`
     - :class:`~surpyval.degradation.process_models.WienerProcessModel`,
       :class:`~surpyval.degradation.process_models.GammaProcessModel`
     - each unit is measured repeatedly and its degradation wanders randomly
       (a *stochastic process*)
   * - :class:`DestructiveDegradation <surpyval.degradation.destructive.DestructiveDegradation_>`
     - :class:`~surpyval.degradation.destructive.DestructiveDegradationModel`
     - each unit can be measured only once

The page covers, in order: the general-path model (fitting, predicting a new
unit, the population of paths, the induced life, confidence bounds),
accelerated tests for it in three steps of increasing detail — stress on the
life, stress on the path parameters, and stress on the clock (which handles
step-stress tests) — with a summary of which to use, the stochastic-process
models (with their own stress support), destructive degradation, and saving a
fitted model.

The data
~~~~~~~~

Repeated-measures degradation data comes in *long* format: three arrays of the
same length, one entry per measurement,

* ``x`` — the time of the measurement (hours, cycles, days, …),
* ``y`` — the degradation measured,
* ``i`` — which unit it belongs to (any hashable labels).

Units need not be measured at the same times or the same number of times. An
accelerated test adds a fourth array, ``Z``, holding the stress for each
measurement; that is described in `Accelerated degradation testing
(covariates)`_.

Here is the data used for the next several sections: twelve units inspected
every 100 hours, each drifting upward at its own rate from its own starting
level, with measurement noise. A unit fails when its measurement reaches 450.

.. jupyter-execute::

    import warnings

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from surpyval.degradation import DegradationAnalysis

    rng = np.random.default_rng(1)
    times = np.arange(100.0, 1100.0, 100.0)       # an inspection every 100 h
    xs, ys, ids = [], [], []
    for unit in range(12):
        a = rng.normal(10.0, 3.0)                 # this unit's starting level
        b = rng.normal(0.30, 0.06)                # this unit's rate per hour
        xs.append(times)
        ys.append(a + b * times + rng.normal(0, 3.0, times.size))  # + noise
        ids.append(np.full(times.size, unit))
    x, y, i = (np.concatenate(v) for v in (xs, ys, ids))

    pd.DataFrame({"x": x, "y": y, "i": i}).head(12)

Degradation path models
-----------------------

The path model is the shape fitted to each unit's measurements. The path
models available, and the pseudo failure time each implies for a threshold
:math:`y_{t}`, are:

.. list-table::
    :header-rows: 1

    * - Name
      - Path
      - Threshold crossing time
    * - ``"linear"``
      - :math:`y = a + b x`
      - :math:`(y_{t} - a) / b`
    * - ``"quadratic"``
      - :math:`y = a + b x + c x^{2}`
      - first positive root of :math:`c t^{2} + b t + (a - y_{t})`
    * - ``"exponential"``
      - :math:`y = a e^{b x}`
      - :math:`\ln(y_{t} / a) / b`
    * - ``"offset-exponential"``
      - :math:`y = a + b e^{c x}`
      - :math:`\ln((y_{t} - a) / b) / c`
    * - ``"power"``
      - :math:`y = a x^{b}`
      - :math:`(y_{t} / a)^{1/b}`
    * - ``"logarithmic"``
      - :math:`y = a + b \ln(x)`
      - :math:`e^{(y_{t} - a) / b}`
    * - ``"lloyd-lipow"``
      - :math:`y = a - b / x`
      - :math:`b / (a - y_{t})`
    * - ``"gompertz"``
      - :math:`y = a e^{-b e^{-c x}}`
      - :math:`-\ln(-\ln(y_{t}/a)/b) / c`
    * - ``"michaelis-menten"``
      - :math:`y = a x / (b + x)`
      - :math:`b y_{t} / (a - y_{t})`

Degradation can be increasing (crack length) or decreasing (luminous
flux); the direction is captured by the sign of the fitted parameters and
needs no configuration. Models that are linear in their parameters
(linear, quadratic, logarithmic, Lloyd-Lipow) are fitted in closed form;
the others are fitted by nonlinear least squares started from a
linearised fit. The offset-exponential covers growth or decay toward an
asymptote (``a = 0`` reduces it to the exponential); Gompertz is
S-shaped; Michaelis-Menten saturates from zero toward ``a``. Some paths
need positive data — the exponential, power, Gompertz and Michaelis-Menten
need positive measurements, and the power, logarithmic, Lloyd-Lipow and
Michaelis-Menten need positive times — and say so if they do not get it.

Pass the name as ``path`` (``"linear"`` is the default). Not sure which shape
fits? Pass ``path="best"``: every registered path model is fitted to every
unit and the one with the smallest AICc (pooled over all units, penalising the
per-unit parameter count) is selected:

.. jupyter-execute::

    best = DegradationAnalysis.fit(x, y, i, threshold=450.0, path="best")
    print("selected:", best.path_model.name)
    {name: round(score, 1) for name, score in best.path_selection.items()}

Candidates that cannot be fitted to every unit — domain violations such as
negative measurements for the exponential, too few distinct measurement times
for their parameter count, or non-convergence — are excluded and score
``nan``. The selection is by measurement fit only; as always, prefer a shape
with physical justification when one is known, since the winner is
extrapolated well beyond the data.

**A custom path.** When the physics suggests a shape that is not in the list,
subclass :class:`~surpyval.degradation.path_models.PathModel`: give it a ``name``, its
``param_names``, the ``path`` itself and its inverse ``inv_path`` (the time the
path reaches a level, ``nan`` or non-positive if it never does). ``fit``
defaults to nonlinear least squares from an ``_initial_guess`` you supply; a
path that is linear in its parameters can instead set
``linear_in_parameters = True`` and provide a closed-form ``fit`` and its
(constant) ``jacobian``, which also makes the population estimates below
exact. The ``jacobian`` (the derivatives of the path with respect to its
parameters, used for the estimation covariances, the Bayesian update and REML)
otherwise defaults to central finite differences, and an optional
``check_data(x, y)`` can reject data outside the path's domain with a clear
message. The built-in shapes are importable objects too (``LinearPath``,
``GompertzPath``, …, and the name-to-object mapping ``PATH_MODELS`` in
``surpyval.degradation``), so ``path=GompertzPath`` is the same as
``path="gompertz"``. Here is a diffusion-limited, square-root path:

.. jupyter-execute::

    from surpyval.degradation import PathModel

    class SquareRootPath(PathModel):
        """y = a + b * sqrt(x): diffusion-limited growth."""

        name = "Square-root"
        param_names = ["a", "b"]
        linear_in_parameters = True

        def path(self, x, a, b):
            return a + b * np.sqrt(np.asarray(x, dtype=float))

        def inv_path(self, y, a, b):
            with np.errstate(divide="ignore", invalid="ignore"):
                return ((np.asarray(y, dtype=float) - a) / b) ** 2

        def jacobian(self, x, *params):
            x = np.asarray(x, dtype=float)
            return np.column_stack([np.ones_like(x), np.sqrt(x)])

        def fit(self, x, y):
            design = np.column_stack([np.ones(len(x)), np.sqrt(x)])
            return np.linalg.lstsq(design, np.asarray(y, dtype=float),
                                   rcond=None)[0]

    # each unit with its own start and rate, as before
    rng_sq = np.random.default_rng(3)
    start = np.repeat(rng_sq.normal(10.0, 5.0, 12), times.size)
    rate = np.repeat(rng_sq.normal(9.0, 1.0, 12), times.size)
    y_sqrt = start + rate * np.sqrt(x) + rng_sq.normal(0, 2.0, x.size)
    sqrt_model = DegradationAnalysis.fit(x, y_sqrt, i, threshold=450.0,
                                         path=SquareRootPath())
    sqrt_model.pseudo_failure_times[:4].round(0)

Example
-------

Fit the linear path to the twelve units above:

.. jupyter-execute::

    model = DegradationAnalysis.fit(x, y, i, threshold=450.0)
    model

The summary names the path model and the lifetime distribution fitted to the
pseudo failure times (Weibull by default) with its parameters. Everything the
three steps produced is on the model:

.. jupyter-execute::

    print("pseudo failure times:", model.pseudo_failure_times.round(0))
    print("censored (1) or not :", model.c)
    print("unit 0's fitted a, b:", model.path_params[0].round(4))

``model.path(t, unit)`` evaluates a unit's fitted path, and ``model.plot()``
draws every unit's data and fitted path, extended to its pseudo failure time,
against the threshold:

.. jupyter-execute::

    model.plot()

The usual lifetime functions — ``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf``,
``mean``, ``random`` — are forwarded to the fitted life model, which is also
available directly as ``model.life_model`` (an ordinary SurPyval parametric
model):

.. jupyter-execute::

    print("reliability at 1000, 1500, 2000 h:", model.sf([1000.0, 1500.0, 2000.0]).round(3))
    print("B10 and median life              :", model.qf([0.1, 0.5]).round(0))
    print("mean life                        :", round(float(model.mean()), 0))

**Units that never reach the threshold.** Add a thirteenth unit that is not
degrading — its reading drifts slightly *down*. Its fitted path never reaches
450, so it has no pseudo failure time;
it is treated as right censored at its last measurement, which the life fit
takes into account, and a warning says which unit it was:

.. jupyter-execute::

    x_flat = np.concatenate([x, times])
    y_flat = np.concatenate([y, 12 - 0.005 * times + rng.normal(0, 1.0, times.size)])
    i_flat = np.concatenate([i, np.full(times.size, 99)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with_flat = DegradationAnalysis.fit(x_flat, y_flat, i_flat, threshold=450.0)
    print(caught[0].message)
    print("censored flags:", with_flat.c)

The opposite case is a unit that is **already past the threshold** at its first
measurement — its fitted path crossed 450 at or before time zero. It has
failed, only we do not know when, so it is left censored at its first
measurement time (flag ``-1``), again with a warning; the summary counts it as
"Failed Before Start". Which side of the threshold counts as failed is read
from the units that do cross it, so the flat unit above, trending away on the
good side, still never reaches it:

.. jupyter-execute::

    x_early = np.concatenate([x, times])
    y_early = np.concatenate([y, 470 + 0.3 * times + rng.normal(0, 1.0, times.size)])
    i_early = np.concatenate([i, np.full(times.size, 98)])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with_early = DegradationAnalysis.fit(x_early, y_early, i_early, threshold=450.0)
    print(caught[0].message)
    print("censored flags:", with_early.c)

**Decreasing degradation.** Nothing changes when the measurement falls toward
the threshold instead of rising to it. Here eight LEDs lose light output
exponentially from about 100 %, and a lamp has failed once it is below 70 % of
its initial output (the "L70" life). The exponential path is
:math:`a e^{bt}` with a *negative* rate :math:`b`, and its crossing time
:math:`\ln(70/a)/b` is positive because numerator and rate are both negative:

.. jupyter-execute::

    rng_led = np.random.default_rng(7)
    hours = np.arange(1000.0, 7000.0, 1000.0)
    xs, ys, ids = [], [], []
    for lamp in range(8):
        a = rng_led.normal(100.0, 1.0)                  # initial output, %
        b = -abs(rng_led.normal(2.5e-5, 0.5e-5))        # decay rate per hour
        xs.append(hours)
        ys.append(a * np.exp(b * hours) + rng_led.normal(0, 0.3, hours.size))
        ids.append(np.full(hours.size, lamp))
    x_led, y_led, i_led = (np.concatenate(v) for v in (xs, ys, ids))

    led = DegradationAnalysis.fit(x_led, y_led, i_led, threshold=70.0,
                                  path="exponential")
    print("fitted decay rates b     :", led.path_params[:, 1].round(7))
    print("pseudo L70 lives (hours) :", led.pseudo_failure_times.round(-2))
    print("median L70 life (hours)  :", round(float(led.qf(0.5)), -2))

The test ran for 6000 hours and no lamp got near 70 %, yet every lamp has an
L70 estimate — around 14 000 hours, well beyond the data, which is exactly why
the choice of path shape matters so much.

**Other inputs and options.** Data can come straight from a DataFrame with
``fit_from_df``, naming the columns (``Z_cols`` names the stress column(s) for
the accelerated models below, and every other ``fit`` argument passes
through); the life distribution fitted to the pseudo failure times, and its
fitting method, can be changed with ``distribution`` and ``how`` (``"MLE"`` by
default, passed on to the distribution's ``fit``):

.. jupyter-execute::

    from surpyval import LogNormal

    df = pd.DataFrame({"hours": x, "resistance": y, "unit": i})
    DegradationAnalysis.fit_from_df(df, x="hours", y="resistance", i="unit",
                                    threshold=450.0, distribution=LogNormal)

Predicting a new unit's failure time
------------------------------------

A fitted model can estimate the failure time of a *new*, partially observed
unit from its degradation trajectory. ``predict_failure_time`` fits the model's
path shape to the new measurements and extrapolates to the same threshold;
``predict_remaining_life`` subtracts the unit's age (its last measurement
time):

.. jupyter-execute::

    x_new = np.array([100.0, 200.0, 300.0])     # observed for 300 h so far
    y_new = np.array([45.0, 70.0, 103.0])

    print("failure time  :", round(model.predict_failure_time(x_new, y_new), 0))
    print("remaining life:", round(model.predict_remaining_life(x_new, y_new), 0))

If the fitted path crossed the threshold between time zero and the last
measurement, the predicted failure time is in the past and the remaining life
is negative. A trajectory already past the threshold at its first measurement
returns the non-positive time at which its fitted path crossed. If the new
unit's fitted path
never reaches the threshold (it is not degrading), both return ``nan`` with a
warning. The trajectory needs at least as many measurements as the path has
parameters, at two or more distinct times. For a population-level view instead
of a per-unit extrapolation, use the fitted life model — e.g. the survival of a
unit that has already survived to time ``a``: ``model.life_model.cs(t, a)``.

Bayesian remaining-life prediction
----------------------------------

``predict_failure_time`` trusts the new unit's least-squares fit completely —
dangerous when the trajectory is short or noisy. ``predict_rul`` instead blends
the unit's own trend with the population: the population path-parameter
distribution (next section) is the prior, the unit's measurements are the
likelihood, and the Gaussian posterior of the unit's path parameters is pushed
through the threshold crossing by Monte Carlo:

.. jupyter-execute::

    pred = model.predict_rul(x_new, y_new, alpha_ci=0.05, random_state=0)

    print("failure time (median)  :", round(pred.failure_time, 0))
    print("95% credible interval  :", tuple(round(v, 0) for v in pred.failure_time_interval))
    print("remaining life (median):", round(pred.rul, 0))
    print("95% credible interval  :", tuple(round(v, 0) for v in pred.rul_interval))
    print("P(already failed)      :", pred.prob_failed)
    print("P(never fails)         :", pred.prob_never_fails)
    print("posterior mean a, b    :", pred.posterior_mean.round(4))

``pred`` is a :class:`~surpyval.degradation.degradation_analysis.RULPrediction`; ``pred.samples``
holds the Monte Carlo failure times (``inf`` for draws whose path never reaches
the threshold, so the median or an interval end is ``inf`` when that many draws
never fail; ``0`` for draws already past the threshold at the first
measurement, which count as failed) and ``posterior_cov`` the posterior covariance. ``alpha_ci`` sets
the interval level, ``n_samples`` the number of draws, and ``random_state``
makes the draws reproducible.

The point of the prior shows when the trajectory is short. Here is the same unit
seen after one, two and three measurements, next to the plain least-squares
extrapolation (which needs at least two points):

.. jupyter-execute::

    for k in (1, 2, 3):
        p = model.predict_rul(x_new[:k], y_new[:k], random_state=0)
        lo, hi = p.failure_time_interval
        plain = (model.predict_failure_time(x_new[:k], y_new[:k])
                 if k >= 2 else float("nan"))
        print(f"{k} measurement(s): Bayesian {p.failure_time:6.0f} "
              f"({lo:5.0f} to {hi:5.0f})   least squares {plain:6.0f}")

With one measurement the forecast leans on the population, and its interval is
wide; with two, the least-squares line through two noisy points overshoots,
while the Bayesian forecast moves only part of the way toward it; with three it
has moved most of the way to the unit's own trend, and its interval has
narrowed. The posterior mean is a
precision-weighted compromise, so as measurements accumulate the prediction
converges to the plain least-squares extrapolation. The posterior is exact
(conjugate) for path models that are linear in their parameters and an
iterated-linearisation (Laplace) approximation for the others. It requires a
positive ``measurement_var``: if every training unit's path fitted its
measurements exactly there is no noise model to blend with, and
``predict_rul`` says so.

The population path-parameter distribution
------------------------------------------

The fitted model also estimates the *population* distribution of the path
parameters, :math:`\theta_i \sim N(\mu, \Sigma)` — what a random-effects
treatment, and the Bayesian prior above, needs:

.. jupyter-execute::

    print("mean a, b (mu)          :", model.path_param_mean.round(4))
    print("between-unit sd (Sigma) :", np.sqrt(np.diag(model.path_param_cov)).round(4))
    print("raw sample sd           :", np.sqrt(np.diag(model.path_param_sample_cov)).round(4))
    print("measurement sd (sigma)  :", round(float(np.sqrt(model.measurement_var)), 3))

The data were simulated with a mean start of 10 and rate of 0.30, starting
levels spread by 3, rates by 0.06 and measurement noise 3, and the estimates are
close to all of them. Notice that the raw sample
standard deviation of the fitted intercepts is larger than the corrected one:
because each unit's fitted parameters are least-squares *estimates*, their
scatter across units mixes two sources — real unit-to-unit variability and
per-unit estimation noise (:math:`\mathrm{Cov}(\hat{\theta}_i) = \Sigma + V_i`).
``path_param_cov`` applies the Lu-Meeker two-stage correction: the measurement
variance is pooled from the per-unit residuals, each unit's estimation
covariance :math:`V_i = \sigma^2 (J_i^T J_i)^{-1}` is computed from the path
Jacobian, and the average is subtracted from the sample covariance.

The result is projected onto the positive semi-definite cone. If material
clipping was needed — the estimation noise is comparable to the between-unit
scatter, typically with few units or few measurements per unit — a warning is
raised and the corrected covariance should be treated as unreliable. When every
unit has only as many measurements as path parameters, the measurement variance
cannot be estimated and no correction is applied.

REML estimation of the population
---------------------------------

The moments correction can go rank-deficient when the estimation noise
rivals the between-unit scatter. The robust alternative is to fit the
random-effects (Lu-Meeker) formulation directly as a linear mixed
model — each unit's parameters are draws
:math:`\theta_i \sim MVN(\mu, \Sigma)`, so with the random effects
integrated out each unit's measurement vector is marginally

.. math::

    y_i \sim N(X_i \mu, \; X_i \Sigma X_i^T + \sigma^2 I)

and :math:`(\mu, \Sigma, \sigma^2)` are estimated by maximising the
restricted (REML) marginal likelihood — REML rather than plain ML so
the variance components do not inherit the small-sample downward bias
from estimating :math:`\mu`. Select it with ``population_method="reml"``.

Here is where it matters: six units, each measured only three to six times at
irregular moments, with noisy measurements. The moments correction subtracts
more estimation noise from the intercepts than their raw scatter, clips the
intercept spread to zero and warns; REML estimates it directly and gets a
sensible answer (the truth is 3):

.. jupyter-execute::

    rng5 = np.random.default_rng(5)
    xs, ys, ids = [], [], []
    for unit in range(6):
        t = np.sort(rng5.uniform(50, 1000, rng5.integers(3, 7)))
        a, b = rng5.normal(10, 3.0), rng5.normal(0.3, 0.06)
        xs.append(t)
        ys.append(a + b * t + rng5.normal(0, 8.0, t.size))
        ids.append(np.full(t.size, unit))
    x_few, y_few, i_few = (np.concatenate(v) for v in (xs, ys, ids))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        moments = DegradationAnalysis.fit(x_few, y_few, i_few, threshold=450.0)
    print("warning:", str(caught[0].message)[:70], "...")
    reml = DegradationAnalysis.fit(x_few, y_few, i_few, threshold=450.0,
                                   population_method="reml")
    print("moments between-unit sd:", np.sqrt(np.diag(moments.path_param_cov)).round(3))
    print("REML between-unit sd   :", np.sqrt(np.diag(reml.path_param_cov)).round(3))

The estimates land in the same attributes (``path_param_mean``,
``path_param_cov``, ``measurement_var``), so ``predict_rul`` and everything else
work unchanged. :math:`\Sigma` is parameterised by its Cholesky factor, so it is
positive definite by construction — no clipping. On a balanced design (every
unit measured at the same times, with a path linear in its parameters) REML
coincides with the corrected moments estimate whenever that needed no clipping.
The twelve units at the top of the page are such a design, and the two methods
agree to every printed digit:

.. jupyter-execute::

    reml12 = DegradationAnalysis.fit(x, y, i, threshold=450.0,
                                     population_method="reml")
    print("moments between-unit sd:", np.sqrt(np.diag(model.path_param_cov)).round(4))
    print("REML between-unit sd   :", np.sqrt(np.diag(reml12.path_param_cov)).round(4))
    print("measurement sd         :", round(float(np.sqrt(model.measurement_var)), 4),
          round(float(np.sqrt(reml12.measurement_var)), 4))

They differ on unbalanced data and when the unit count is small, where REML is
preferable. The method used is recorded as ``population_method`` on the fitted
model. REML requires a positive measurement variance.

For path models that are **linear in their parameters** (linear,
quadratic, logarithmic, Lloyd-Lipow) the design matrix :math:`X_i` is
fixed and the marginal model above is exact. For a **nonlinear** path
(exponential, power, Gompertz, …) the mean :math:`f(x_i, \theta_i)` is
no longer :math:`X_i \theta_i`, so REML uses the Lindstrom–Bates FOCE
linearisation [LindstromBates1990]_: each unit's parameters are
estimated at their conditional (penalised-least-squares) mode, the path
is linearised about that mode to give a working linear mixed model, and
the linear REML step is iterated to convergence. On a linear path this
reduces to the exact fit in a single pass. Select it the same way,
``DegradationAnalysis.fit(..., path="exponential", population_method="reml")``.

Induced failure-time distribution (Lu-Meeker)
---------------------------------------------

Everything so far follows the **pseudo-failure-time** route: extrapolate each
unit's fitted path to the threshold, get one (noisy) failure time per unit, and
fit a lifetime distribution to those times. That is simple and robust, but it
throws away structure — it treats each extrapolated time as a plain observation
and never uses the fact that we have *estimated the whole population of paths*.

The **Lu-Meeker** approach uses that population directly. Once the model has
fitted the population path-parameter distribution
:math:`\theta \sim N(\mu, \Sigma)` (the ``path_param_mean`` and
``path_param_cov`` above), the failure-time distribution of the population
follows *deterministically*: a unit with path parameters :math:`\theta` fails
at the time its path crosses the threshold, i.e. at
:math:`T(\theta) = \text{inv\_path}(D; \theta)`. So the population failure-time
distribution is simply the distribution of :math:`T(\theta)` as
:math:`\theta` ranges over its population. There is rarely a closed form, so
``induced_life`` evaluates it by **Monte Carlo**: draw many
:math:`\theta \sim N(\mu, \Sigma)`, push each through ``inv_path``, and collect
the resulting failure times.

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from surpyval.degradation import DegradationAnalysis

    rng = np.random.default_rng(0)
    threshold = 30.0

    xs, ys, ids = [], [], []
    for unit in range(50):
        a = rng.normal(0.0, 0.2)      # per-unit intercept
        b = rng.normal(1.0, 0.25)     # per-unit slope (degradation rate)
        t = np.arange(0, 20, 2.0)
        y = a + b * t + rng.normal(0, 0.4, t.size)
        xs.append(t)
        ys.append(y)
        ids.append(np.full(t.size, unit))
    x, y, i = (np.concatenate(z) for z in (xs, ys, ids))

    model = DegradationAnalysis.fit(x, y, i, threshold=threshold, path="linear")
    induced = model.induced_life(n_samples=20000, random_state=1)
    induced

The returned ``InducedFailureDistribution`` behaves
like any other life object — ``sf``, ``ff``, ``qf``, ``mean``, ``median`` and
``random`` are all there — so you can read life quantiles straight off it:

.. jupyter-execute::

    print("pseudo-failure median :", round(float(model.qf(0.5)), 2))
    print("induced median        :", round(induced.median(), 2))
    print("induced B10 life      :", round(induced.qf(0.10), 2))

The real purpose is as a **diagnostic**. The pseudo-failure fit and the induced
distribution come at the population life two different ways; when the path model
and its population summary are trustworthy, they should agree. Overlaying the
two CDFs is the check:

.. jupyter-execute::

    t = np.linspace(0, 60, 200)
    plt.plot(t, model.ff(t), label="pseudo-failure fit")
    plt.plot(t, induced.ff(t), "--", label="induced (Lu-Meeker)")
    plt.xlabel("Time")
    plt.ylabel("Probability of failure  F(t)")
    plt.legend()

Close agreement is reassuring; a large gap is a warning that the path model,
the Gaussian population assumption, the covariance estimate — or the lifetime
distribution fitted to the pseudo failure times — is off. Here the medians,
30.3 and 28.7, differ by about 5 %, but the shapes differ more: the induced
curve starts later and has the longer right tail. That is the path model
talking. A failure time :math:`(D - a)/b` with a normally distributed rate
:math:`b` is right-skewed, and the default Weibull (shape near 5) is not. A
right-skewed life distribution for the pseudo failure times should then agree
better, and it does — the LogNormal matches the induced quantiles far more
closely than the Weibull, and has the lower AIC:

.. jupyter-execute::

    from surpyval import LogNormal

    lognormal_fit = DegradationAnalysis.fit(x, y, i, threshold=threshold,
                                            distribution=LogNormal)
    p = [0.05, 0.1, 0.5, 0.9, 0.95]
    print("quantiles at", p)
    print("Weibull   :", np.round(model.qf(p), 1))
    print("LogNormal :", np.round(lognormal_fit.qf(p), 1))
    print("induced   :", np.round(induced.qf(p), 1))
    print("AIC Weibull, LogNormal:", round(model.life_model.aic(), 1),
          round(lognormal_fit.life_model.aic(), 1))

A subtlety the induced distribution surfaces honestly: some draws of
:math:`\theta` describe paths that **never reach the threshold** (a
non-increasing slope, say). Those contribute an ``inf`` failure time — a
defective *"never fails"* mass reported as ``prob_never_fails`` — and once the
quantiles reach into that mass they, and the ``mean``, become ``inf``. This is
the correct behaviour: if a fraction of the population genuinely never fails,
the population has no finite mean life. At the other end, a draw whose path is
already past the threshold at the earliest measurement time crossed it at or
before time zero; it counts as a failure at time zero. Finally, ``induced_life`` needs a
single population of paths: an accelerated (covariate) model pools every
stress level, so it is refused there unless the path parameters are modelled
against stress (``links``, below), in which case it takes the stress ``Z`` to
induce the life at.

Confidence bounds
-----------------

The pseudo-failure-time approach is a *two-stage* estimator: the life
distribution is fitted to *extrapolated* failure times as if they had been
observed exactly. The plain life-model bounds therefore treat the pseudo
failure times as certain and are too narrow. ``DegradationModel.cb`` corrects
this, folding the first-stage (path-fit and extrapolation) uncertainty back
into the life-model covariance with an analytic delta-method /
generated-regressor correction:

.. jupyter-execute::

    import numpy as np
    from matplotlib import pyplot as plt
    from surpyval.degradation import DegradationAnalysis

    rng = np.random.default_rng(0)
    times = np.arange(1, 9) * 100.0
    n_units = 60
    x = np.tile(times, n_units)
    unit = np.repeat(np.arange(n_units), times.size)
    slopes = rng.normal(0.22, 0.05, size=n_units)     # between-unit spread
    y = 10 + np.repeat(slopes, times.size) * x + rng.normal(0, 2.0, size=x.size)

    model = DegradationAnalysis.fit(x, y, unit, threshold=150)

    t = np.linspace(400, 800, 200)
    band = model.cb(t, on='sf')                # (n, 2): two-stage [lower, upper]
    plt.plot(t, model.sf(t), 'b', label='S(t)')
    plt.fill_between(t, band[:, 0], band[:, 1], alpha=0.2,
                     label='95% two-stage band')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('S(t)')

The correction adds a positive term to the life-model information inverse, so
the two-stage parameter covariance (``model.life_parameter_covariance()``) is
the ordinary MLE covariance *plus* the propagated first-stage variance — the
bounds widen to their correct coverage. A slower, assumption-light cross-check
resamples whole units and reruns the whole pipeline:

.. jupyter-execute::

    model.cb(np.array([500.0, 600.0]), on='sf', method='bootstrap',
             n_boot=100, seed=0)

Both methods take ``on`` (``"sf"``, ``"ff"`` or ``"Hf"``), ``alpha_ci`` (the
total tail probability: a two-sided band has ``alpha_ci / 2`` in each tail, so
the default is a 95 % band) and ``bound`` (``"two-sided"``, ``"lower"`` or ``"upper"``). The bounds describe
the *life model*; the stochastic-process and destructive models further down
have their own uncertainty story (the destructive model offers bootstrap bounds;
the process models do not yet report parameter uncertainty).


Accelerated degradation testing (covariates)
--------------------------------------------

In accelerated degradation testing (ADT) units are run at *elevated stress*
(temperature, voltage, load) so they degrade fast enough to measure, and life
is then extrapolated back to use conditions. There are three ways to let stress
into the general-path model — on the life, on the path parameters, or on the
clock — compared side by side on the :doc:`Degradation Analysis` page; this
section takes them in that order.

The simplest: pass the stress as ``Z`` to :meth:`DegradationAnalysis.fit <surpyval.degradation.degradation_analysis.DegradationAnalysis_.fit>`.
``Z`` is aligned to ``x`` (one row per measurement, one column per stress
variable) and must be constant within each unit — a unit is tested at a single
stress — while the units must span at least two stress levels (the fit refuses
a single one: the stress effect could not be told from the baseline). The paths are fitted exactly as before, and step three fits a
*regression* life model to the pseudo failure times instead of a plain
distribution, with each unit's stress as its covariate, so life can be
predicted at any stress. A plain distribution is wrapped automatically in an
accelerated-failure-time model, :math:`H(t \mid z) = H_0(e^{\beta^\top z} t)`
(a positive coefficient means higher stress, shorter life); an explicit
regression fitter (``AFT(LogNormal)``, ``WeibullPH``, …) is used as given.

.. jupyter-execute::

    rng = np.random.default_rng(0)
    times = np.arange(1, 11) * 5.0
    xs, ys, ids, Zs = [], [], [], []
    uid = 0
    for stress in [0.0, 0.5, 1.0, 1.5]:        # four stress levels
        for _ in range(12):
            rate = 0.5 * np.exp(0.8 * stress) * np.exp(rng.normal(0, 0.1))
            path = 10 + rng.normal(0, 1) + rate * times
            xs.append(times)
            ys.append(path + rng.normal(0, 0.5, times.size))
            ids.append(np.full(times.size, uid))
            Zs.append(np.full(times.size, stress))
            uid += 1
    xd, yd, idd, Zd = (np.concatenate(a) for a in (xs, ys, ids, Zs))

    model = DegradationAnalysis.fit(xd, yd, idd, threshold=100.0, Z=Zd)
    model

The last fitted coefficient is the stress effect (higher stress ⇒ faster
degradation ⇒ shorter life); the regression itself is ``model.life_model``,
``model.Z`` holds one stress row per unit (aligned to ``model.units``), and
``model.is_accelerated`` is ``True``. The prediction methods now take the
stress vector ``Z`` at which to evaluate life — and refuse to predict without
it — so life at use conditions is one call:

.. jupyter-execute::

    for stress in [0.0, 0.5, 1.0]:
        print(f'stress {stress}: mean life = {model.mean(Z=[stress]):.1f}')

    t = np.linspace(0, 300, 200)
    for stress in [0.0, 0.5, 1.0]:
        plt.plot(t, model.sf(t, Z=[stress]), label=f'stress {stress}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Reliability at stress')

``qf`` and ``mean`` invert / integrate the regression survival function, and
``random`` draws from it.

Two-stage confidence bounds at a stress are available by bootstrap: units
are resampled (each carrying its stress), the whole accelerated pipeline is
rerun, and the reliability at ``Z`` is read off each refit, so the first-stage
path/extrapolation uncertainty is folded in — just as for the plain model, but
evaluated at a chosen stress:

.. jupyter-execute::

    t = np.array([50.0, 100.0, 150.0])
    band = model.cb(t, on='sf', method='bootstrap', Z=[0.0],
                    n_boot=50, seed=0)
    band                                        # (n, 2): [lower, upper] at Z=0

The analytic (generated-regressor) delta-method correction used for the plain
model is not derived for the regression life fit, so ``method='bootstrap'`` is
required for a covariate model (and ``model.cb`` needs the stress ``Z``). The
first-stage-only regression bounds — which ignore the extrapolation
uncertainty — remain available directly through ``model.life_model.cb(x, Z,
...)``.

Modelling the degradation mechanism against stress
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The fit above lets stress act only on the pseudo failure times. It never says
*why* life shortens, and its population of path parameters pools every stress
level, so it describes no unit actually tested. Passing ``links`` alongside
``Z`` models the degradation **mechanism** instead: the named path parameters
depend on stress, on an ``"identity"`` link (the parameter itself is linear in
``Z``) or a ``"log"`` link (its log is, so it stays positive and stress acts
multiplicatively); the others do not depend on stress, though they still vary
from unit to unit. Here the degradation rate ``b`` is log-linear in stress
(with ``Z = 1/T`` that is the Arrhenius relationship) and the starting level
``a`` is not:

.. jupyter-execute::

    mech = DegradationAnalysis.fit(xd, yd, idd, threshold=100.0, Z=Zd,
                                   links={'b': 'log'})
    dict(zip(mech.path_param_fixed_names,
             mech.path_param_fixed.round(3).tolist()))

The data were simulated with ``log b = log 0.5 + 0.8 * stress``, and the fixed
effects recover it: ``log(b)`` is the log-rate intercept and ``log(b):Z0`` the
stress coefficient. ``path_param_link_cov`` holds the unit-to-unit scatter left
once the stress effect is removed, estimated by the same two-stage or REML
route as the plain population. The life model is still the covariate
regression on the pseudo failure times, so ``sf``, ``qf``, ``mean`` and the
bootstrap bounds all work exactly as above.

What the mechanism adds is a population of paths *at each stress*: on the link
scale, :math:`\eta \sim N(D(z)\gamma, \Sigma)`. ``path_param_link_mean(Z)``
is its mean :math:`D(z)\gamma` (here ``a`` and ``log(b)``), and
``path_param_link_cov`` its covariance :math:`\Sigma`, the same at every
stress. ``path_param_median(Z)`` maps the mean through the links to give the
typical (median) unit's natural-scale path parameters at that stress:

.. jupyter-execute::

    print("link-scale mean at stress 1:", mech.path_param_link_mean([1.0]).round(3))
    print("median a, b at stress 1    :", mech.path_param_median([1.0]).round(3))
    print("between-unit sd (link)     :",
          np.sqrt(np.diag(mech.path_param_link_cov)).round(3))

(The pooled ``path_param_mean`` and ``path_param_cov`` are still computed, but
they mix every stress level.) ``induced_life(Z=...)`` pushes the whole
stress-conditional population through the threshold crossing. Inside the
tested range it agrees with the regression
life fit; outside it — here at ``-0.5``, below every tested level, as use
conditions usually are — it is the mechanism rather than a curve through the
pseudo failure times that carries the extrapolation:

.. jupyter-execute::

    for stress in [-0.5, 0.0, 1.0]:
        rate = mech.path_param_median([stress])[1]
        induced = mech.induced_life(Z=[stress], random_state=0)
        regression = float(np.ravel(model.qf(0.5, Z=[stress]))[0])
        print(f'stress {stress:+.1f}: median rate {rate:.3f}, '
              f'median life induced {induced.median():6.1f} '
              f'/ regression {regression:6.1f}')

Remaining useful life becomes stress-aware in the same way.
``predict_rul(x, y, Z=...)`` updates a new unit's trajectory against the
population of units at *its* stress, rather than against a mixture of every
stress tested; the posterior is taken on the link scale, so a log-linked rate
stays positive. With only a couple of measurements the stress matters a great
deal, and as measurements accumulate the prediction converges on the unit's
own trend whatever the stress:

.. jupyter-execute::

    new_x, new_y = [5.0, 10.0], [13.0, 16.0]
    for stress in [0.0, 1.5]:
        pred = mech.predict_rul(new_x, new_y, Z=[stress], random_state=0)
        lower, upper = pred.rul_interval
        print(f'stress {stress}: RUL {pred.rul:5.1f}  '
              f'(95% interval {lower:5.1f} to {upper:5.1f})')

A model fitted with ``Z`` alone (no ``links``) refuses ``Z`` in
``predict_rul`` and ``induced_life``, since it has no stress-conditional
population to condition on (its ``predict_rul`` uses the pooled population
and its ``induced_life`` is refused altogether); a model fitted with
``links`` requires it. The step-stress clock below takes ``Z`` in both, in its
own way.

Step-stress tests: an accelerated clock
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything above assumes each unit is tested at one stress — ``Z`` must be
constant within a unit. In a **step-stress** test the same units are stepped up
in stress during the test, so each unit's path runs at several stresses in
turn. ``acceleration='clock'`` handles this. Stress speeds up the clock of every
unit's path: a unit at stress ``z`` ages
:math:`\mathrm{AF}(z) = \exp(\gamma^\top (z - z_{\text{ref}}))` times faster than
at the reference stress, and its path is the ordinary path model evaluated on
the time it has aged at the reference stress. ``Z`` is now one row per
measurement giving the stress over the interval that *ends* at that
measurement, and ``stress_ref`` is the use condition.

Here 25 units run for 300 hours — 100 at 50 °C, 100 at 75 °C, then 100 at
100 °C — with ``z = 1/T`` (so the acceleration is Arrhenius) and a true
:math:`\gamma = -5000`, inspected every 10 hours. At 50 °C each unit's path is
``a + b * t`` with a unit-to-unit rate ``b`` around ``0.02``, and failure is at
``15``:

.. jupyter-execute::

    from surpyval import StepSchedule

    rng = np.random.default_rng(3)
    temps = np.array([323.0, 348.0, 373.0])        # 50, 75 and 100 C, in kelvin
    z_levels = 1 / temps
    gamma_true = -5000.0
    times = np.arange(10.0, 300.0 + 1e-9, 10.0)
    z = np.select([times <= 100, times <= 200], z_levels[:2], z_levels[2])
    af = np.exp(gamma_true * (z - z_levels[0]))    # speed-up relative to 50 C

    xs, ys, ids, Zs = [], [], [], []
    for unit in range(25):
        a, b = rng.normal([1.0, 0.02], [0.2, 0.003])  # 50 C start and rate
        tau = np.cumsum(10.0 * af)                     # hours aged at 50 C
        xs.append(times)
        ys.append(a + b * tau + rng.normal(0, 0.3, times.size))
        ids.append(np.full(times.size, unit))
        Zs.append(z)
    xs_, ys_, ids_, Zs_ = (np.concatenate(v) for v in (xs, ys, ids, Zs))

    step = DegradationAnalysis.fit(xs_, ys_, ids_, threshold=15.0, Z=Zs_,
                                   acceleration='clock',
                                   stress_ref=[z_levels[0]])
    step

The stress coefficient is close to the ``-5000`` simulated (an activation
energy of :math:`5000 \times 8.617\times10^{-5} \approx 0.43` eV). It is
stored as ``step.gamma``, with ``step.stress_ref`` the reference stress (the
mean stress over the measurement intervals if ``stress_ref`` is not given),
``step.acceleration == 'clock'``, and ``step.Z`` the stress rows as given. The
path parameters, their population (``path_param_mean``, ``path_param_cov``) and
the pseudo failure times are all on the 50 °C clock, so the life distribution
listed is the life *at the reference stress*.

It is worth looking at how the stress rows line up with the measurements around
the first step. The chamber goes from 50 °C to 75 °C just after the 100-hour
inspection, so the row for the 100-hour measurement still says 50 °C (that
interval ran at 50 °C) and the row for the 110-hour measurement says 75 °C.
Each unit's clock adds up ``AF`` times the interval length, which is what the
model sees instead of calendar time:

.. jupyter-execute::

    unit0 = ids_ == 0
    rows = slice(8, 12)
    af_rows = step.acceleration_factor
    tau0 = np.cumsum(np.diff(np.concatenate([[0.0], xs_[unit0]]))
                     * np.array([af_rows([zz]) for zz in Zs_[unit0]]))
    pd.DataFrame({
        "time (h)": xs_[unit0][rows],
        "stress row (C)": (1 / Zs_[unit0][rows] - 273.0).round(0),
        "AF over interval": [round(af_rows([zz]), 2) for zz in Zs_[unit0][rows]],
        "clock (50 C hours)": tau0[rows].round(1),
    })

``model.path(t, unit)`` evaluates
a unit's fitted path in calendar time, along its own stress history (holding
its last stress beyond its last measurement), and bends at each step;
``model.plot()`` draws every unit this way:

.. jupyter-execute::

    for unit in range(4):
        m = ids_ == unit
        line, = plt.plot(xs_[m], ys_[m], '.', alpha=0.6)
        plt.plot(times, step.path(times, unit), color=line.get_color())
    for edge in (100, 200):
        plt.axvline(edge, color='grey', linestyle=':')
    plt.xlabel('Time (h)')
    plt.ylabel('Degradation')

Every life method takes the stress as ``Z``: one row for a constant stress, or a
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` for a stress that changes over time. Because
stress only changes the speed of the clock, life under any history is the
reference-stress life at the clock time, :math:`F(t) = F_0(\tau(t))`:

.. jupyter-execute::

    profile = StepSchedule.from_changepoints([0, 100, 200], z_levels)
    print('AF at 100 C                 :',
          round(step.acceleration_factor([z_levels[2]]), 2))
    print('mean life at 50 C           :', round(step.mean(Z=[z_levels[0]]), 1))
    print('mean life at 100 C          :', round(step.mean(Z=[z_levels[2]]), 1))
    print('mean life on the test profile:', round(step.mean(Z=profile), 1))

    t = np.linspace(0, 400, 401)
    plt.plot(t, step.ff(t, Z=[z_levels[2]]), label='constant 100 C')
    plt.plot(t, step.ff(t, Z=profile), label='step profile')
    plt.plot(t, step.ff(t, Z=[z_levels[1]]), label='constant 75 C')
    plt.xlabel('Time (h)')
    plt.ylabel('Probability of failure  F(t)')
    plt.legend()

Where the information about :math:`\gamma` comes from decides the estimation
method. With the default ``population_method='moments'`` it comes from the
units whose stress *steps* — the change of slope at a step fixes the
acceleration — by profile least squares: for each trial :math:`\gamma` every
unit's path is refitted on its clock, and :math:`\gamma` minimises the pooled
residual sum of squares. A unit held at one stress can absorb any acceleration
into its own rate, so with no steps at all this raises an error.
``population_method='reml'`` fits the mixed model instead: units share one
population of path parameters, so differences between units run at different
constant stresses identify :math:`\gamma` as well, and it works for a classic
constant-stress test too. It maximises the (Lindstrom-Bates) approximate
marginal likelihood over :math:`\gamma`, then estimates the population by REML
at that :math:`\gamma` — the details are on the :doc:`Degradation Analysis`
page. It is the slower of the two (seconds rather than a fraction of a second
here). On this stepped test the two agree:

.. jupyter-execute::

    step_reml = DegradationAnalysis.fit(xs_, ys_, ids_, threshold=15.0, Z=Zs_,
                                        acceleration='clock',
                                        stress_ref=[z_levels[0]],
                                        population_method='reml')
    print('gamma, moments:', step.gamma.round(0), '  REML:', step_reml.gamma.round(0))

``reml`` is also how a clock is fitted to a classic **constant-stress** test,
where ``moments`` refuses. Here it is on the four-level test from the start of
this section, next to the two earlier treatments of the same data. The life
regression's stress coefficient, the log-rate's stress coefficient under
``links``, and the clock's :math:`\gamma` all estimate the same quantity (the
data were simulated with a rate proportional to :math:`e^{0.8 z}`, which for
the linear path is a clock with :math:`\gamma = 0.8`):

.. jupyter-execute::

    clock_adt = DegradationAnalysis.fit(xd, yd, idd, threshold=100.0, Z=Zd,
                                        acceleration='clock',
                                        population_method='reml',
                                        stress_ref=[0.0])
    print('life regression coefficient:', round(model.life_model.params[-1], 3))
    print('links log(b):Z0            :', round(mech.path_param_fixed[-1], 3))
    print('clock gamma                :', clock_adt.gamma.round(3))
    for name, fitted in [('regression', model), ('clock', clock_adt)]:
        print(f'median life at stress -0.5, {name:10s}:',
              round(float(np.ravel(fitted.qf(0.5, Z=[-0.5]))[0]), 1))

The three coefficients are 0.851, 0.827 and 0.824, and the median life at
:math:`-0.5`, below every tested level, is 283 from the regression, 273 from
the clock and 271 from the ``links`` model's induced life above: three routes
to the same extrapolation, differing by a few percent. That agreement is the
check to make; a clear disagreement would mean the stress acts on the paths in
a way one of the models cannot represent. For the linear path the clock and
``links={'b': 'log'}`` are nearly the same model (the theory page explains
why), so their closeness is expected.

**Remaining life on a stress plan.** For a unit you are watching, the stress
matters twice: its *history* sets how far along its reference-stress clock it
already is, and the *plan* for the rest of its life sets how fast it gets
through the remainder. ``predict_rul`` takes both: ``Z`` is the unit's history,
one row per measurement as at fit (or one row for a constant stress), and
``Z_future`` the stress from its last measurement on — a row, or a
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` whose time zero is *now* (by default the last
stress is held). Here is a unit run on the test profile for 150 hours, with two
plans for what comes next:

.. jupyter-execute::

    new_t = np.arange(10.0, 150.0 + 1e-9, 10.0)
    new_z = np.select([new_t <= 100], [z_levels[0]], z_levels[1])
    new_y = 0.9 + 0.022 * np.cumsum(
        10.0 * np.exp(gamma_true * (new_z - z_levels[0]))
    ) + rng.normal(0, 0.3, new_t.size)

    plans = {
        'stay at 75 C': [z_levels[1]],
        '50 h at 75 C, then 100 C': StepSchedule.from_changepoints(
            [0, 50], [[z_levels[1]], [z_levels[2]]]),
    }
    for name, plan in plans.items():
        pred = step.predict_rul(new_t, new_y, Z=new_z, Z_future=plan,
                                random_state=0)
        lower, upper = pred.rul_interval
        print(f'{name:26s}: RUL {pred.rul:5.1f} h '
              f'(95% interval {lower:5.1f} to {upper:5.1f})')

The posterior is taken on the unit's clock against the reference-stress
population, and each sampled failure time is mapped back to calendar time
along the history and the plan. ``predict_failure_time`` and
``predict_remaining_life`` take the same ``Z`` and ``Z_future``.
``induced_life(Z=...)`` pushes the whole population of paths through the
threshold under any stress or profile, and the two-stage confidence bounds are
available by bootstrap: units are resampled with their stress histories and
the clock is re-estimated on every resample.

.. jupyter-execute::

    induced = step.induced_life(Z=profile, random_state=0)
    print('induced median life on the profile:', round(induced.median(), 1))
    step.cb([200.0, 240.0], Z=profile, method='bootstrap', n_boot=50, seed=0)

``acceleration='clock'`` cannot be combined with ``links`` or ``path='best'``,
and the life model must be a plain distribution, since stress enters through
the clock. The analytic confidence-bound correction is not derived for a clock
model, whose pseudo failure times also depend on the estimated clock, so
``cb`` needs ``method='bootstrap'``. With ``population_method='reml'`` every
bootstrap resample re-runs the mixed-model estimate of the clock, which takes
seconds per resample; budget ``n_boot`` accordingly.

Which stress model?
~~~~~~~~~~~~~~~~~~~

The three accelerated general-path models side by side (the reasoning is in
:ref:`Choosing how stress enters a general-path model <deg-choosing-stress>`
on the :doc:`Degradation Analysis` page):

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - ``Z``
     - ``Z`` + ``links``
     - ``Z`` + ``acceleration='clock'``
   * - stress within a unit
     - constant
     - constant
     - may change (step-stress)
   * - stress acts on
     - pseudo failure times (regression)
     - chosen path parameters
     - the time scale of the whole path
   * - life methods (``sf``, ``qf``, ``mean``, …) take
     - one stress row
     - one stress row
     - a row or a ``StepSchedule``
   * - ``predict_rul`` prior
     - pooled population (no ``Z``)
     - population at the unit's stress (``Z``)
     - reference population on the unit's clock (``Z``, ``Z_future``)
   * - ``induced_life``
     - refused
     - at a stress row
     - at a row or a ``StepSchedule``
   * - ``population_method``
     - either
     - either
     - ``moments`` needs stepped units; ``reml`` works for any test
   * - ``cb``
     - bootstrap
     - bootstrap
     - bootstrap

Stochastic-process degradation models
-------------------------------------

Everything above is the **general-path** approach: fit a deterministic curve to
each unit, extrapolate it to the threshold to get a *pseudo failure time*, then
fit a lifetime distribution to those times. It works well when each unit really
does follow a smooth trend plus measurement noise.

But sometimes the degradation itself is *random over time*, not a smooth curve
observed with error. A crack does not grow along a tidy exponential; it jumps
ahead in fits and starts. A wear signal wanders up and down from measurement to
measurement. In those cases it is more honest to model the **increments** of
the degradation as a stochastic process, and read the failure-time distribution
off the process directly — as the distribution of the **first time the process
crosses the threshold** (the "first-passage time"). This is a standard part of
the reliability toolkit; see [Meeker1998]_ for a textbook treatment and
[LuMeeker1993]_ for the idea of deriving a failure-time distribution from
degradation measurements.

SurPyval provides two such processes. They are not competitors; they describe
different physics, and the right one is dictated by whether your degradation can
*decrease*:

* :class:`~surpyval.degradation.process_models.WienerProcess` — for signals that **fluctuate**
  up and down (noisy sensors, measurements that wobble).
* :class:`~surpyval.degradation.process_models.GammaProcess` — for damage that only ever
  **accumulates** (wear, corrosion, crack growth).

Both are fitted from the same three arrays you have used throughout this
section: ``x`` (measurement times), ``y`` (degradation measurements) and ``i``
(the unit each measurement belongs to). Internally each model looks only at the
*increments* between consecutive measurements of a unit, so units can be
measured at different, irregular times without any special handling — a two-week
gap simply contributes a larger ``dt``.

One consequence to keep in mind: the fit never sees a unit's starting level,
but the life distribution assumes every unit starts from degradation ``0`` at
time ``0``, so ``threshold`` is the *distance* a new unit travels to failure.
If your measurements start from a baseline (a resistance of 100 Ω that fails at
110 Ω), subtract it, or pass the distance (``threshold=10``). The examples
below all start at zero.

The Wiener process
------------------

A **Wiener process with drift** (Brownian motion with drift) models the
degradation as

.. math::

    W(t) = \mu\, t + \sigma\, B(t),

where :math:`B(t)` is standard Brownian motion. There are just two parameters,
and it is worth being clear about what each one *means*:

* :math:`\mu` — the **drift**. This is the average rate at which degradation
  accumulates: on average the signal climbs by :math:`\mu` per unit time. A
  larger drift means faster wear-out and a shorter life.
* :math:`\sigma` — the **diffusion** (or volatility). This is the size of the
  random wobble around that average trend. With :math:`\sigma = 0` the process
  would be a perfectly straight line :math:`\mu t` (noise-free data, which the
  fit refuses: that is not a Wiener process); the bigger :math:`\sigma`,
  the more the path jitters up and down and the more spread-out the failure
  times become. (Wiener degradation models, including extensions with
  unit-to-unit random effects, are surveyed in [Wang2010]_.)

Over any interval of length :math:`\Delta t`, the change in degradation is
**Gaussian**:

.. math::

    \Delta W \sim \mathrm{Normal}\!\left(\mu\, \Delta t,\; \sigma^2\, \Delta t\right).

Two things follow from this. First, because a Normal can be negative, the path
can go *down* as well as up — which is exactly why the Wiener process is the
right model for noisy, non-monotone signals. Second, the increments are
independent, so fitting is easy: the drift is just the total degradation divided
by the total time, and the diffusion is estimated from how much the increments
scatter around that trend. SurPyval does this by maximum likelihood.

**Failure = first passage.** A unit fails the first time :math:`W(t)` reaches
the threshold :math:`D`. For a Wiener process this first-passage time has a
famous closed form — the **Inverse Gaussian** distribution — with

.. math::

    \text{mean} = \frac{D}{\mu}, \qquad \text{shape} = \frac{D^2}{\sigma^2}.

The mean life :math:`D/\mu` is beautifully intuitive: distance to failure
divided by the average speed. The shape controls how tightly the failure times
cluster around that mean (more diffusion → more scatter). Because the increments
are independent, the model also needs the drift to be **positive** — a
non-positive drift would mean the process is not reliably heading toward the
threshold at all, so SurPyval raises an error rather than return a "life" that
may never end.

Let's fit one. We simulate 30 units, each measured every half–time-unit, with a
true drift of ``0.5`` and diffusion ``0.4``, failing at a degradation of ``10``:

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    from surpyval.degradation import WienerProcess

    rng = np.random.default_rng(0)
    mu_true, sigma_true, threshold = 0.5, 0.4, 10.0

    xs, ys, ids = [], [], []
    for unit in range(30):
        t = np.arange(0, 15.5, 0.5)
        increments = rng.normal(
            mu_true * 0.5, sigma_true * np.sqrt(0.5), size=t.size - 1
        )
        y = np.concatenate([[0.0], np.cumsum(increments)])
        xs.append(t)
        ys.append(y)
        ids.append(np.full(t.size, unit))
    x, y, i = (np.concatenate(a) for a in (xs, ys, ids))

    model = WienerProcess.fit(x, y, i, threshold=threshold)
    model

Read the summary line by line: the fitted **drift** and **diffusion** are close
to the ``0.5`` and ``0.4`` we simulated, and the **mean time to failure** is
``threshold / drift`` — roughly ``10 / 0.5 = 20`` time units. Notice the paths
below are jagged and occasionally dip downward; that non-monotone wobble is the
Wiener process's defining feature.

.. jupyter-execute::

    for unit in range(8):
        m = i == unit
        plt.plot(x[m], y[m], alpha=0.6)
    plt.axhline(threshold, color="k", linestyle="--", label="threshold")
    plt.xlabel("Time")
    plt.ylabel("Degradation")
    plt.legend()

Now the payoff: a full **failure-time distribution**, derived from the process,
that you can query like any other SurPyval model. ``mean()`` is the average
life, ``ff(t)`` is the probability of having failed by time ``t`` (the CDF),
``sf(t)`` is the reliability, and ``qf(p)`` is the quantile (e.g. the median
life, or the time by which 10 % have failed):

.. jupyter-execute::

    print("mean life           :", round(model.mean(), 2))
    print("P(fail by t = 25)   :", round(model.ff(25.0), 3))
    print("median life         :", round(model.qf(0.5), 2))
    print("B10 life (10% fail) :", round(model.qf(0.10), 2))

    t = np.linspace(0, 40, 200)
    plt.plot(t, model.ff(t))
    plt.xlabel("Time to failure")
    plt.ylabel("Probability of failure  F(t)")

**Remaining useful life.** The real power of a degradation model is that it can
update its forecast for a unit you have been *watching*. If a unit is currently
at degradation level ``7`` (out of a threshold of ``10``), only the remaining
distance of ``3`` matters, and — because Wiener increments are independent of
the past — the remaining life is itself an Inverse Gaussian over that shorter
distance. ``predict_rul`` returns its median and an interval:

.. jupyter-execute::

    rul = model.predict_rul(current_degradation=7.0)
    print("median remaining life :", round(rul.rul, 2))
    print("95% interval          :", tuple(round(v, 2) for v in rul.rul_interval))

The interval widths tell you how much uncertainty remains: a unit close to the
threshold has a short, tight remaining-life estimate; a fresh unit has a long,
uncertain one. If the current degradation is already at or beyond the threshold,
``prob_already_failed`` is ``1`` and the remaining life is ``0``. ``rul`` is a
``ProcessRUL`` holding ``rul`` (the median), ``rul_interval`` (equal-tailed, at
level ``alpha_ci``, default ``0.05``) and ``prob_already_failed``. Unlike the
general-path ``predict_rul`` it needs only the current level, not the unit's
history — the independent increments make the past irrelevant — and the
interval reflects the randomness of the process, not uncertainty in the fitted
``mu`` and ``sigma`` (the process models do not report parameter
uncertainty). The fitted parameters are ``model.mu`` and ``model.sigma``
(together, ``model.params``), and ``hf``, ``Hf``, ``df`` and
``random(size, random_state=...)`` complete the set of life methods.

The Gamma process
-----------------

The Wiener process allows the signal to decrease, which is wrong for damage that
is physically **irreversible** — a crack never heals, corrosion never reverses,
wear never un-wears. For those, use a **Gamma process** (whose use in
maintenance and reliability is surveyed in [vanNoortwijk2009]_), whose
increments are
strictly non-negative, so the path is **monotone increasing**.

Over an interval of length :math:`\Delta t`, the Gamma-process increment is
Gamma-distributed:

.. math::

    \Delta W \sim \mathrm{Gamma}\!\left(\text{shape} = \alpha\, \Delta t,\;
    \text{rate} = \beta\right).

Again, two parameters, and again it is worth knowing what they mean:

* :math:`\alpha` — the **shape rate**. It controls how *quickly and how
  steadily* damage accrues. The amount of shape accumulated by time :math:`t` is
  :math:`\alpha t`; a large :math:`\alpha` gives many small, regular increments
  (a smooth-looking climb), a small :math:`\alpha` gives fewer, larger, jumpier
  increments.
* :math:`\beta` — the **rate** parameter of those increments. It sets their
  scale: the mean degradation accumulated per unit time is
  :math:`\alpha / \beta`, and the variance per unit time is
  :math:`\alpha / \beta^2`.

So :math:`\alpha/\beta` is the Gamma process's analogue of the Wiener drift —
the average degradation speed — while :math:`\alpha` alone governs how *regular*
versus *jumpy* the accumulation is.

**Failure = first passage, again**, but now the monotonicity makes it especially
clean: the process has crossed the threshold :math:`D` by time :math:`t` exactly
when its level :math:`W(t)` is at or above :math:`D`. So the probability of
having failed by :math:`t` is

.. math::

    F(t) = \Pr\!\left(W(t) \ge D\right),

which SurPyval evaluates with the (regularised) incomplete gamma function. There
is no simpler closed form than that, but every method you need — ``sf``, ``ff``,
``df``, ``qf``, ``mean``, ``random`` — is computed from it.

Here we simulate 40 units of monotone wear with shape rate ``3`` and rate
``1.5`` (so mean degradation speed :math:`\alpha/\beta = 2` per unit time),
failing at ``30``:

.. jupyter-execute::

    from surpyval.degradation import GammaProcess

    rng = np.random.default_rng(1)
    alpha_true, beta_true, threshold = 3.0, 1.5, 30.0

    xs, ys, ids = [], [], []
    for unit in range(40):
        t = np.arange(0, 12.5, 0.5)
        increments = rng.gamma(alpha_true * 0.5, 1.0 / beta_true, size=t.size - 1)
        y = np.concatenate([[0.0], np.cumsum(increments)])
        xs.append(t)
        ys.append(y)
        ids.append(np.full(t.size, unit))
    x, y, i = (np.concatenate(a) for a in (xs, ys, ids))

    model = GammaProcess.fit(x, y, i, threshold=threshold)
    model

The fitted ``alpha`` and ``beta`` recover the ``3`` and ``1.5`` we used, and the
mean time to failure is about ``threshold / (alpha/beta) = 30 / 2 = 15``. The
paths this time only ever climb — no downward wobble is possible:

.. jupyter-execute::

    for unit in range(8):
        m = i == unit
        plt.plot(x[m], y[m], alpha=0.6)
    plt.axhline(threshold, color="k", linestyle="--", label="threshold")
    plt.xlabel("Time")
    plt.ylabel("Degradation")
    plt.legend()

The failure-time distribution and remaining-life prediction work exactly as they
did for the Wiener model — same method names, same meaning:

.. jupyter-execute::

    print("mean life      :", round(model.mean(), 2))
    print("median life    :", round(model.qf(0.5), 2))

    rul = model.predict_rul(current_degradation=20.0)
    print("RUL at y = 20  :", round(rul.rul, 2),
          "  interval", tuple(round(v, 2) for v in rul.rul_interval))

Because a Gamma process cannot go down, passing it degradation that *decreases*
over an interval is a modelling error, and SurPyval says so rather than fitting
something meaningless — the message points you at the Wiener process instead:

.. jupyter-execute::
    :raises: ValueError

    # this data dips from 5 back to 3 -- not allowed for a monotone process
    GammaProcess.fit([0, 1, 2], [0.0, 5.0, 3.0], [1, 1, 1], threshold=10.0)

An increment of exactly **zero** is allowed, but a gamma increment is never
exactly zero, so it means the change was too small to register: it enters the
likelihood as censored below the measurement resolution,
:math:`P(\Delta W \le \delta)`. ``resolution`` sets :math:`\delta`; by
default it is the smallest positive increment in the data, which for readings
rounded to a grid is the grid step. Here the wear readings are rounded to the
nearest 0.5, as a coarse gauge would give them:

.. jupyter-execute::

    y_gauge = np.round(y / 0.5) * 0.5
    dy = np.diff(y_gauge)[np.diff(i) == 0]
    print("zero increments:", int((dy == 0).sum()), "of", dy.size)
    GammaProcess.fit(x, y_gauge, i, threshold=threshold)

The mean life, 14.7, stays close to the 15.3 of the unrounded readings; the
shape rate moves further (4.2 against 3.0), because rounding also coarsens
every non-zero increment, which the censoring does not undo. Treating the
zeros as tiny positive increments instead would put the shape rate at 0.37.

Choosing between Wiener and Gamma
---------------------------------

The decision is almost always settled by one question: **can your degradation
physically decrease?**

.. list-table::
   :header-rows: 1
   :widths: 40 20 40

   * - Your degradation paths…
     - Use
     - Because
   * - fluctuate up and down (noisy sensor, wandering signal)
     - ``WienerProcess``
     - Gaussian increments allow decreases and absorb measurement noise; failure
       time is a closed-form Inverse Gaussian.
   * - only ever increase (wear, corrosion, crack growth, fatigue)
     - ``GammaProcess``
     - non-negative increments enforce monotone, irreversible damage; forcing a
       Wiener fit would misread real jumps as noise.

A quick practical test: **plot your paths.** If they visibly dip, the Wiener
process models that directly. If they are monotone, the Gamma process is the
physically honest choice — and if you try to give monotone-only data that
happens to dip (usually a measurement glitch) to the Gamma process, the error it
raises is a useful signal in itself.

Compared with the general-path approach at the top of this page, both process
models share two advantages: they handle **irregular measurement spacing**
without any special treatment (each increment simply carries its own ``dt``),
and they give the failure-time distribution **directly from the fitted process**
rather than through noisy per-unit pseudo failure times. The general-path models
remain the better choice when each unit truly follows a smooth deterministic
trend observed with error, or when you need a specific parametric path shape.

Accelerated and step-stress tests
---------------------------------

Degradation at use conditions is often too slow to watch, so tests raise the
stress. In a **step-stress** test the *same* units are held at one stress for a
while, then stepped up, and up again — every unit sees several stresses, and
each measurement interval ran at whatever stress was applied over it. Both
``WienerProcess`` and ``GammaProcess`` take that stress through ``Z``: one row
per measurement, giving the stress applied over the interval that *ends* at that
measurement.

The model is an **accelerated clock** [WhitmoreSchenkelberg1997]_. Stress
``z`` makes a unit age :math:`\mathrm{AF}(z) = \exp(\gamma^\top (z - z_{\text{ref}}))`
times faster than at the reference stress, so an interval ``dt`` at stress ``z``
contributes :math:`\mathrm{AF}(z)\,dt` of operational time and the process runs
on that clock. The process parameters describe degradation at the reference
stress ``stress_ref`` (pass the use condition), and the new ``gamma``
coefficients describe how strongly stress speeds it up. With ``z = 1/T`` in
kelvin this is Arrhenius, :math:`\gamma = -E_a/k`.

Here 40 units run for 300 hours: 100 hours at 50 °C, 100 at 75 °C, then 100 at
100 °C, with a true :math:`\gamma = -5000` (an activation energy of about
0.43 eV), inspected every 5 hours. The use condition is 50 °C.

.. jupyter-execute::

    from surpyval import StepSchedule

    rng = np.random.default_rng(2)
    temps = np.array([323.0, 348.0, 373.0])  # kelvin
    z_levels = 1 / temps
    z_use = z_levels[0]
    g_true, mu_true, sigma_true, threshold = -5000.0, 0.05, 0.12, 12.0

    times = np.arange(5.0, 300.0 + 1e-9, 5.0)
    z = np.select([times <= 100, times <= 200], z_levels[:2], z_levels[2])
    af = np.exp(g_true * (z - z_use))

    xs, ys, ids, Zs = [], [], [], []
    for unit in range(40):
        dtau = af * 5.0  # operational time in each interval
        increments = rng.normal(mu_true * dtau, sigma_true * np.sqrt(dtau))
        xs.append(np.r_[0.0, times])
        ys.append(np.r_[0.0, np.cumsum(increments)])
        ids.append(np.full(times.size + 1, unit))
        Zs.append(np.r_[z_levels[0], z])
    x, y, i, Z = (np.concatenate(a) for a in (xs, ys, ids, Zs))

    model = WienerProcess.fit(x, y, i, threshold=threshold, Z=Z, stress_ref=[z_use])
    model

The drift and diffusion are close to the 50 °C values (``0.05`` and ``0.12``),
and the stress coefficient is close to ``-5000``. The ``Mean life (ref.)`` line is the mean
life at the reference stress. Every life method now needs a stress, passed as
``Z``: a single row for a constant stress,

.. jupyter-execute::

    print("acceleration factor at 100 C :", round(model.acceleration_factor([z_levels[2]]), 1))
    print("mean life at 50 C            :", round(model.mean(Z=[z_use]), 1))
    print("mean life at 100 C           :", round(model.mean(Z=[z_levels[2]]), 1))

or a :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` for a stress that changes over time. Because
stress only changes the speed of the clock, the life under a profile is still
closed form: :math:`F(t) = F_0(\tau(t))` with :math:`\tau` the operational time.
Here is the life of a fresh unit on the test profile itself, next to the life at
the two ends of it:

.. jupyter-execute::

    profile = StepSchedule.from_changepoints([0, 100, 200], z_levels)

    t = np.linspace(0, 300, 301)
    plt.plot(t, model.ff(t, Z=[z_use]), label="constant 50 C")
    plt.plot(t, model.ff(t, Z=profile), label="step profile")
    plt.plot(t, model.ff(t, Z=[z_levels[2]]), label="constant 100 C")
    plt.xlabel("Time (h)")
    plt.ylabel("Probability of failure  F(t)")
    plt.legend()

    print("median life on the profile:", round(model.qf(0.5, Z=profile), 1), "h")

``ff``, ``sf``, ``df``, ``hf``, ``qf``, ``mean`` and ``random`` all take ``Z``
this way. For remaining life, ``Z`` is the stress *from now on*, and a schedule
starts at time zero = now. A unit currently at degradation ``6`` that will run
another 20 hours at 75 °C before going to 100 °C:

.. jupyter-execute::

    plan = StepSchedule.from_changepoints([0, 20], [[z_levels[1]], [z_levels[2]]])
    rul = model.predict_rul(6.0, Z=plan)
    print("median remaining life :", round(rul.rul, 1), "h")
    print("95% interval          :", tuple(round(v, 1) for v in rul.rul_interval))

A few practical points:

* The same ``Z`` argument covers a **constant-stress** accelerated test, where
  each unit stays at one stress and different units run at different stresses.
  Either way, ``Z`` needs at least two distinct stress levels, or the stress
  coefficients cannot be estimated and the fit says so.
* ``Z`` can carry several stresses (e.g. ``[1/T, log V]``); ``gamma`` then has one
  coefficient per column.
* ``GammaProcess`` works identically. Its ``alpha`` is the shape accrual at
  the reference stress (at stress ``z`` the shape accrues at
  ``alpha * AF(z)``), and ``beta`` is the same at every stress.
* Unlike the general-path ``moments`` fit, no steps are needed: every unit
  shares the process parameters, so units at different constant stresses
  identify ``gamma`` on their own.
* For the Wiener process the stress scales the diffusion along with the drift.
  That is the assumption that makes the life closed form under any profile.
* A model fitted without ``Z`` is unchanged and refuses a ``Z`` argument. A
  model fitted with ``Z`` refuses to predict without one, because its life
  depends on the stress.

Destructive degradation
-----------------------

Everything so far assumes each unit is measured *repeatedly* over time. In a
**destructive** test the measurement destroys the specimen — you break a coupon
to read its strength, or drive insulation to breakdown — so each unit yields
exactly **one** ``(time, degradation)`` point. There are no per-unit paths to
fit and extrapolate, so the pseudo-failure-time machinery above does not apply.

:class:`DestructiveDegradation <surpyval.degradation.destructive.DestructiveDegradation_>` instead models the
*population* degradation distribution directly, as a location-scale regression
whose location moves with a transform of time,
:math:`Y \mid t \sim \mathrm{dist}(\text{loc} = \beta_0 + \beta_1\,\varphi(t),
\ \sigma)`, and induces the lifetime distribution by crossing the failure
threshold. In the example below strength *decreases* with age and a unit fails
once it drops below ``D_f`` — the direction is inferred from the trend:

.. jupyter-execute::

    from surpyval import Normal
    from surpyval.degradation import DestructiveDegradation

    rng = np.random.default_rng(0)
    n = 300
    age = rng.uniform(0, 40, n)                      # one specimen per point
    strength = 100.0 - 1.5 * age + rng.normal(0, 5.0, n)
    D_f = 40.0                                       # failed below this strength

    model = DestructiveDegradation.fit(
        age, strength, threshold=D_f, distribution=Normal,
    )
    model

The fitted model exposes the induced *lifetime* distribution at the threshold
(``sf`` / ``ff`` / ``Hf`` / ``df``) and the fitted *degradation* distribution
over time (``degradation_quantile``):

.. jupyter-execute::

    t = np.linspace(1, 50, 100)
    plt.plot(t, model.sf(t), label='reliability S(t)')
    plt.plot(t, model.degradation_quantile(0.5, t) / 100.0,
             '--', label='median strength (scaled)')
    plt.axhline(D_f / 100.0, color='0.7', lw=1)
    plt.legend()
    plt.xlabel('Age')

The fitted location is ``model.beta`` (intercept and slope on
:math:`\varphi(t)`) and the scale ``model.sigma``; ``model.direction`` records
the direction used (``"auto"``, the default, inferred ``"decreasing"`` here
from the downward trend; pass ``direction=`` to set it).

The response distribution is ``LogNormal`` by default (a positive-valued
measurement, whose scatter grows with its level); use ``Normal`` when the
response can be negative. The time transform :math:`\varphi` is ``"linear"`` by
default — ``"log"``, ``"sqrt"`` and ``"reciprocal"`` are also available, and
``"best"`` fits all four and keeps the one with the lowest AICc, reporting
every score in ``transform_scores``. Because the fit goes through the
distribution's own likelihood, censored measurements — a strength below the
test floor (left censored, ``c = -1``), a specimen that did not break at the
maximum load (right censored, ``c = 1``) — are passed through the ordinary
``c`` argument, recorded at the bound. Here the strength decays exponentially
(so its log is linear in age: the ``LogNormal`` default with the linear
transform), the rig cannot read below 45, and a unit has failed below 50;
``cb`` gives bootstrap confidence bounds on the induced lifetime by
resampling specimens and refitting:

.. jupyter-execute::

    rng = np.random.default_rng(4)
    age = rng.uniform(1, 40, 200)
    strength = 100.0 * np.exp(-0.02 * age + rng.normal(0, 0.08, age.size))
    floor = 45.0                                  # the rig cannot read below 45
    c = np.where(strength < floor, -1, 0)         # -1: left censored at the floor
    reading = np.maximum(strength, floor)

    best = DestructiveDegradation.fit(age, reading, threshold=50.0, c=c,
                                      transform="best")
    print({name: round(score, 1) for name, score in best.transform_scores.items()})
    print("selected transform        :", best.transform, "  censored:", (c == -1).sum())
    print("location, scale           :", best.beta.round(4), round(best.sigma, 4))
    print("median strength at 10, 40 :", best.median_degradation([10.0, 40.0]).round(1))
    print("reliability at 25, 35, 45 :", best.sf([25.0, 35.0, 45.0]).round(3))
    best.cb([25.0, 35.0, 45.0], n_boot=50, seed=0)

The linear transform wins, as simulated; the location recovers
:math:`\log 100 \approx 4.61` and the slope :math:`-0.02` closely, and the
median life, where the median strength falls to 50, is at about
:math:`\ln 2 / 0.02 \approx 35`. ``median_degradation(t)`` is the
``degradation_quantile(0.5, t)`` of the fitted measurement distribution.

Saving and loading a fitted model
---------------------------------

Every fitted degradation model can be serialised to a plain, JSON-safe
dictionary with ``to_dict`` and rebuilt with its class's ``from_dict`` — the
general-path ``DegradationModel``, the stochastic-process
``WienerProcessModel`` / ``GammaProcessModel``, the destructive
``DestructiveDegradationModel``, and the ``InducedFailureDistribution`` — or
with the package-level ``surpyval.from_dict``, which dispatches on the stored
model type:

.. jupyter-execute::

    import surpyval
    from surpyval.degradation import DegradationAnalysis, DegradationModel

    saveable = DegradationAnalysis.fit(
        np.tile(np.arange(100, 1100, 100), 4),
        10 + np.repeat([0.31, 0.28, 0.44, 0.37], 10)
        * np.tile(np.arange(100, 1100, 100), 4),
        np.repeat([1, 2, 3, 4], 10),
        threshold=150,
    )
    reloaded = DegradationModel.from_dict(saveable.to_dict())
    grid = np.array([300.0, 450.0, 600.0])
    print("match:", np.allclose(saveable.sf(grid), reloaded.sf(grid)))
    print("dispatched:", type(surpyval.from_dict(saveable.to_dict())).__name__)

``DegradationModel`` stores its raw data and everything fitted from it — the
path parameters, the population, the pseudo failure times, the life model
(plain or accelerated, through its own serialisation), and for accelerated
models the stresses, ``links`` fixed effects or the clock's ``gamma`` and
``stress_ref`` — so every prediction method works on the reloaded model,
including the *bootstrap* confidence bound. That bound reruns the whole fit on
resampled units; the fitter it reruns (the lifetime distribution, or for an
accelerated model the regression fitter such as ``WeibullPH`` or
``AFT(Weibull)``) is recovered from the restored life model, so with the same
seed the reloaded model reproduces the original's band exactly:

.. jupyter-execute::

    print(saveable.cb(grid, method="bootstrap", n_boot=50, seed=1).round(3))
    print(reloaded.cb(grid, method="bootstrap", n_boot=50, seed=1).round(3))

Every one of these models also has ``to_json(path)`` and ``from_json(path)``
for writing a file directly (``surpyval.from_json`` reads any of them). The
destructive model stores its specimens as well, so a reloaded one reproduces
its bootstrap bounds in the same way:

.. jupyter-execute::

    from surpyval.degradation import DestructiveDegradationModel

    reloaded_destructive = DestructiveDegradationModel.from_dict(best.to_dict())
    print(np.allclose(best.cb([25.0, 35.0], n_boot=20, seed=2),
                      reloaded_destructive.cb([25.0, 35.0], n_boot=20, seed=2)))

One limit: the path model is stored by its *name* (the ``path=`` string, such
as ``"offset-exponential"``) and resolved among the built-in ones, so a model
fitted with a custom ``PathModel`` subclass (like the square-root path above)
cannot be rebuilt.
