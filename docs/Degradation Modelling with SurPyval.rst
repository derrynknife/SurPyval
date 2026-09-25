Degradation Modelling with SurPyval
===================================

Sometimes failure data is scarce or takes too long to collect: items are
highly reliable, test time is limited, and few (or no) units fail during
the observation window. But failure is often the end point of a gradual,
measurable process — a crack grows, a resistance drifts, a lumen output
fades, a material wears. Degradation analysis exploits this: instead of
waiting for units to fail, we track a *degradation measurement* over time
on each unit, define failure as the measurement crossing a *threshold*,
and extrapolate each unit's degradation trend to that threshold to obtain
a failure time — even for units that never actually failed on test.

For the concepts and the underlying theory — the general-path model, the
Lu-Meeker two-stage correction, the induced failure-time distribution, and the
stochastic-process (Wiener/Gamma) first-passage models — see the
:doc:`Degradation Analysis` page.

SurPyval implements the classic *pseudo-failure-time* approach:

1. A degradation *path model* (e.g. linear) is fitted, by least squares,
   to each unit's measurements.
2. Each unit's fitted path is extrapolated to the failure threshold. The
   crossing time is that unit's *pseudo failure time*.
3. A lifetime distribution (Weibull by default) is fitted to the pseudo
   failure times, and can then be used like any other SurPyval parametric
   model.

If a unit's fitted path never reaches the threshold (for example, the
unit is not degrading, or is trending away from the threshold), the unit
is treated as right censored at its last observed time, and the censoring
is passed through to the lifetime distribution fit.

This page also covers two alternatives to the pseudo-failure-time approach:
the `Stochastic-process degradation models`_ (Wiener and Gamma processes,
which model the increments directly), and `Destructive degradation`_ for tests
where each specimen can be measured only once.

Degradation path models
-----------------------

The path models available, and the pseudo failure time each implies for a
threshold :math:`y_{t}`, are:

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
S-shaped; Michaelis-Menten saturates from zero toward ``a``.

Not sure which shape fits? Pass ``path="best"``:

.. code:: python

    model = DegradationAnalysis.fit(x, y, i, threshold=150, path="best")
    model.path_model.name   # the selected model
    model.path_selection    # AICc score per candidate

Every registered path model is fitted to every unit and the model with
the smallest AICc (pooled over all units, penalising the per-unit
parameter count) is selected; candidates that cannot be fitted to every
unit — domain violations such as negative measurements for the
exponential, too few distinct measurement times for their parameter
count, or non-convergence — are excluded and score ``nan``. The
selection is by measurement fit only; as always, prefer a shape with
physical justification when one is known, since the winner is
extrapolated well beyond the data.

Example
-------

Consider four units whose degradation is measured every 100 hours, with
failure defined as the measurement reaching 150:

.. code:: python

    import numpy as np
    from surpyval.degradation import DegradationAnalysis

    x = np.tile(np.arange(100, 1100, 100), 4)
    i = np.repeat([1, 2, 3, 4], 10)
    slopes = np.repeat([0.31, 0.28, 0.44, 0.37], 10)
    y = 10 + slopes * x

    model = DegradationAnalysis.fit(x, y, i, threshold=150)
    print(model)

.. code:: text

    Degradation Analysis SurPyval Model
    ===================================
    Path Model          : Linear
    Threshold           : 150.0
    Number of Units     : 4
    Censored Units      : 0
    Life Distribution   : Weibull
    Parameters          :
         alpha: 441.4780882117898
          beta: 6.987078993008337

The fitted model exposes the per-unit results and forwards the usual
lifetime functions to the fitted life model:

.. code:: python

    model.pseudo_failure_times
    # array([451.61290323, 500.        , 318.18181818, 378.37837838])

    model.sf([300, 400, 500])
    # array([0.93496716, 0.60538471, 0.09196813])

    model.life_model    # the underlying Parametric Weibull model
    model.plot()        # data, fitted paths, and the threshold

Data can also come straight from a DataFrame with
``DegradationAnalysis.fit_from_df(df, x="time", y="measurement",
i="unit", threshold=150)``, the life distribution and fitting method can
be changed with the ``distribution`` and ``how`` arguments (e.g.
``distribution=LogNormal, how="MPP"``), and a custom path shape can be
used by passing a ``surpyval.degradation.PathModel`` subclass instance as
``path``.

Predicting a new unit's failure time
------------------------------------

A fitted model can estimate the failure time of a *new*, partially
observed unit from its degradation trajectory. The model fits its path
shape to the new measurements and extrapolates to the same threshold:

.. code:: python

    # a new unit observed for only 300 hours, degrading at ~0.35/hour
    x_new = [100, 200, 300]
    y_new = [45, 80, 115]

    model.predict_failure_time(x_new, y_new)   # ~400: when y reaches 150
    model.predict_remaining_life(x_new, y_new) # ~100: minus its age (300)

If the trajectory has already crossed the threshold, the predicted
failure time is in the past and the remaining life is negative. If the
new unit's fitted path never reaches the threshold (it is not
degrading), both return ``nan`` with a warning. For a population-level
view instead of a per-unit extrapolation, the fitted life model can be
used directly — e.g. the survival of a unit that has already survived
to time ``a``: ``model.life_model.cs(t, a)``.

Bayesian remaining-life prediction
----------------------------------

``predict_failure_time`` trusts the new unit's least-squares fit
completely — dangerous when the trajectory is short or noisy.
``predict_rul`` instead blends the unit's own trend with the
population: the population path-parameter distribution (below) is the
prior, the unit's measurements are the likelihood, and the Gaussian
posterior of the unit's path parameters is pushed through the
threshold crossing by Monte Carlo:

.. code:: python

    pred = model.predict_rul(x_new, y_new, alpha_ci=0.05)

    pred.failure_time           # posterior median failure time
    pred.failure_time_interval  # 95% credible interval
    pred.rul                    # median remaining useful life
    pred.rul_interval           # 95% credible interval
    pred.prob_failed            # P(already crossed the threshold)
    pred.prob_never_fails       # P(path never reaches the threshold)
    pred.posterior_mean         # the unit's posterior path parameters
    pred.posterior_cov

The posterior mean is a precision-weighted compromise: a short or
noisy trajectory is shrunk toward the population's typical path, and
as measurements accumulate the prediction converges to the plain
least-squares extrapolation. Unlike ``predict_failure_time``, it works
from a single measurement, and a not-yet-degrading trajectory yields a
long-but-finite prediction with wide bounds rather than ``nan``. The
posterior is exact (conjugate) for path models that are linear in
their parameters (linear, quadratic, logarithmic, Lloyd-Lipow) and an
iterated-linearisation (Laplace) approximation for the others. It
requires a positive ``measurement_var`` — with noiseless training
paths there is nothing to blend.

The population path-parameter distribution
------------------------------------------

The fitted model also estimates the *population* distribution of the
path parameters, which is what a random-effects treatment (and any
Bayesian blending of a new unit's trajectory with the population) needs
as its prior:

.. code:: python

    model.path_param_mean        # mean path parameters across units
    model.path_param_cov         # between-unit covariance (corrected)
    model.path_param_sample_cov  # raw sample covariance (uncorrected)
    model.measurement_var        # pooled measurement-error variance

Because each unit's fitted parameters are least-squares *estimates*,
their scatter across units mixes two sources: real unit-to-unit
variability and per-unit estimation noise
(:math:`\mathrm{Cov}(\hat{\theta}_i) = \Sigma + V_i`). The raw sample
covariance therefore overstates the between-unit variability.
``path_param_cov`` applies the Lu-Meeker two-stage correction: the
measurement variance is pooled from the per-unit residuals, each
unit's estimation covariance :math:`V_i = \sigma^2 (J_i^T J_i)^{-1}`
is computed from the path Jacobian, and the average is subtracted from
the sample covariance. The result is projected onto the positive
semi-definite cone; if material clipping was needed (estimation noise
comparable to the between-unit scatter — few units or few measurements
per unit), a warning is raised and the corrected covariance should be
treated as unreliable. When every unit has only as many measurements
as path parameters, the measurement variance cannot be estimated and
no correction is applied.

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
from estimating :math:`\mu`. Select it with:

.. code:: python

    model = DegradationAnalysis.fit(
        x, y, i, threshold=150, population_method="reml"
    )

The estimates land in the same attributes (``path_param_mean``,
``path_param_cov``, ``measurement_var``), so ``predict_rul`` and
everything else work unchanged. :math:`\Sigma` is parameterised by its
Cholesky factor, so it is positive definite by construction — no
clipping. On balanced designs (every unit measured at the same times)
REML coincides with the corrected moments estimate; they differ on
unbalanced data and when the unit count is small, where REML is
preferable. REML requires a positive measurement variance.

For path models that are **linear in their parameters** (linear,
quadratic, logarithmic, Lloyd-Lipow) the design matrix :math:`X_i` is
fixed and the marginal model above is exact. For a **nonlinear** path
(exponential, power, Gompertz, …) the mean :math:`f(x_i, \theta_i)` is
no longer :math:`X_i \theta_i`, so REML uses the Lindstrom–Bates FOCE
linearisation [LindstromBates1990]_: each unit's parameters are
estimated at their conditional (penalised-least-squares) mode, the path
is linearised about that mode to give a working linear mixed model, and
the linear REML step is iterated to convergence. On a linear path this
reduces to the exact fit in a single pass. Select it the same way:

.. code:: python

    model = DegradationAnalysis.fit(
        x, y, i, threshold=20, path="exponential",
        population_method="reml",
    )

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

Close agreement (as here) is reassuring; a large gap is a warning that the
path model, the Gaussian population assumption, or the covariance estimate is
off.

A subtlety the induced distribution surfaces honestly: some draws of
:math:`\theta` describe paths that **never reach the threshold** (a
non-increasing slope, say). Those contribute an ``inf`` failure time — a
defective *"never fails"* mass reported as ``prob_never_fails`` — and once the
quantiles reach into that mass they, and the ``mean``, become ``inf``. This is
the correct behaviour: if a fraction of the population genuinely never fails,
the population has no finite mean life. Finally, ``induced_life`` needs a
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

Random-effects (Lu-Meeker, fitted by REML) and stochastic-process (Wiener,
gamma-process) degradation models, which propagate this uncertainty through a
single likelihood rather than a two-stage correction, are candidates for future
work.


Accelerated degradation testing (covariates)
--------------------------------------------

In accelerated degradation testing (ADT) units are run at *elevated stress*
(temperature, voltage, load) so they degrade fast enough to measure, and life
is then extrapolated back to use conditions. Passing a per-unit stress
covariate ``Z`` to :meth:`DegradationAnalysis.fit` fits a *regression* life
model on the pseudo failure times instead of a plain distribution, so life can
be predicted at any stress. A plain distribution is wrapped automatically in an
accelerated-failure-time model; an explicit regression fitter
(``AFT(LogNormal)``, ``WeibullPH``, …) is used as given.

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
degradation ⇒ shorter life). The prediction methods now take the stress vector
``Z`` at which to evaluate life, so life at use conditions is one call:

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
                    n_boot=100, seed=0)
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
depend on stress, on an ``"identity"`` or ``"log"`` link, and the others are
common to every unit. Here the degradation rate ``b`` is log-linear in stress
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

What the mechanism adds is a population of paths *at each stress*.
``path_param_median(Z)`` gives the typical unit's path parameters there, and
``induced_life(Z=...)`` pushes the whole stress-conditional population through
the threshold crossing. Inside the tested range it agrees with the regression
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

A model fitted without ``links`` refuses ``Z`` in ``predict_rul`` and
``induced_life``, since it has no stress-conditional population to condition
on; a model fitted with ``links`` requires it.

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

The stress coefficient is close to the ``-5000`` simulated. The path
parameters, their population (``path_param_mean``, ``path_param_cov``) and the
pseudo failure times are all on the 50 °C clock, so the life distribution
listed is the life *at the reference stress*. ``model.path(t, unit)`` evaluates
a unit's fitted path in calendar time, along its own stress history, and bends
at each step:

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
:class:`~surpyval.StepSchedule` for a stress that changes over time. Because
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
acceleration — by profile least squares. A unit held at one stress can absorb
any acceleration into its own rate, so with no steps at all this raises an
error. ``population_method='reml'`` fits the mixed model instead: units share
one population of path parameters, so differences between units run at
different constant stresses identify :math:`\gamma` as well, and it works for a
classic constant-stress test too.

A few limits of this first version: ``acceleration='clock'`` cannot be combined
with ``links`` or ``path='best'``, and the life model must be a plain
distribution, since stress enters through the clock. Remaining-life prediction
(``predict_rul``), the induced life and the confidence bounds are not yet
available for a clock model, and each raises an error saying so.

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

* :class:`~surpyval.degradation.WienerProcess` — for signals that **fluctuate**
  up and down (noisy sensors, measurements that wobble).
* :class:`~surpyval.degradation.GammaProcess` — for damage that only ever
  **accumulates** (wear, corrosion, crack growth).

Both are fitted from the same three arrays you have used throughout this
section: ``x`` (measurement times), ``y`` (degradation measurements) and ``i``
(the unit each measurement belongs to). Internally each model looks only at the
*increments* between consecutive measurements of a unit, so units can be
measured at different, irregular times without any special handling — a two-week
gap simply contributes a larger ``dt``.

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
  would be a perfectly straight line :math:`\mu t`; the bigger :math:`\sigma`,
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
``prob_already_failed`` is ``1`` and the remaining life is ``0``.

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

or a :class:`~surpyval.StepSchedule` for a stress that changes over time. Because
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
* ``GammaProcess`` works identically; its ``alpha`` (the shape accrual) and
  ``beta`` are the reference-stress values.
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

:class:`~surpyval.degradation.DestructiveDegradation` instead models the
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

The response distribution is ``LogNormal`` by default (a positive-valued
measurement); use ``Normal`` when the response can be negative. The time
transform :math:`\varphi` is ``"linear"`` by default — ``"log"``, ``"sqrt"``,
``"reciprocal"`` or ``"best"`` (lowest AICc) are also available. Because the
fit goes through the distribution's own likelihood, censored measurements — a
strength below the test floor (left-censored), a specimen that did not break at
the maximum load (right-censored) — are passed through the ordinary ``c``
argument, and ``cb`` gives bootstrap confidence bounds on the induced lifetime.

Saving and loading a fitted model
---------------------------------

Every fitted degradation model can be serialised to a dictionary or JSON file
and rebuilt later — the general-path ``DegradationModel``, the stochastic-
process ``WienerProcessModel`` / ``GammaProcessModel``, the destructive
``DestructiveDegradationModel``, and the ``InducedFailureDistribution``:

.. jupyter-execute::

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

Use ``to_json`` / ``from_json`` for a file directly. ``DegradationModel`` stores
its raw data, so the reloaded model can also produce bootstrap confidence
bounds; its fitted life model (plain or accelerated) round-trips through its own
serialisation.
