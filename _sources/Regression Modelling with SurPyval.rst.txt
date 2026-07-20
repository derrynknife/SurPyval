
Regression Modelling with SurPyval
==================================

The time until an event — a failure, death, recovery — will almost always depend
on external factors. A bearing may last longer in a cool, clean environment than
in a hot, dirty one. A patient's survival may depend on age, dosage, and
comorbidities. The question regression modelling answers is: *how much* do these
factors matter, and in what direction?

Regression survival modelling is fundamentally about capturing the relationship
between covariates :math:`Z` and the survival distribution. Unlike ordinary
regression, we must handle censored observations — items that had not yet failed
when we stopped watching them — and we want our model to remain valid as a
probability distribution (survival functions must start at 1 and decay to 0).

For the rest of this page we assume the following imports:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt


Choosing a regression model family
------------------------------------

There are four fundamentally different ways a covariate can affect a survival
distribution. Each gives rise to a distinct model family:

**Proportional Hazards (PH)** — the covariate multiplies the *rate of dying*:

.. math::

    h(x \mid Z) = h_0(x) \cdot \phi(Z)

If :math:`\phi(Z) = 2`, an individual with that covariate value fails at twice
the rate at every instant in time. The *shape* of the hazard curve is unchanged;
only its level shifts. This is the most common choice in medical research.

**Accelerated Failure Time (AFT)** — the covariate stretches or compresses the
*time axis*:

.. math::

    H(x \mid Z) = H_0\!\left(\phi(Z) \cdot x\right)

If :math:`\phi(Z) = 2`, an individual "ages" at twice the normal rate — reaching
at age 10 the same cumulative risk that a baseline individual has at age 20. The
*entire* survival curve shifts left or right on the time axis. This is often a
more natural framing in engineering and materials science.

**Proportional Odds (PO)** — the covariate scales the *odds of having failed*:

.. math::

    \frac{F(x \mid Z)}{S(x \mid Z)} = \frac{F_0(x)}{S_0(x)} \cdot \phi(Z)

The PO formulation is natural when you think about the problem in terms of *odds
ratios* rather than hazard ratios. A key practical difference from PH: the
covariate effect attenuates over time. Early in life, hazard ratios and odds
ratios behave similarly; at long follow-up times the PO effect fades as everyone
converges toward failure regardless of their covariates. When you believe the
PH assumption ("constant hazard ratio for all time") is too strong, PO is often
a better default.

**Additive Hazards (AH)** — the covariate *adds* to the hazard instead of
multiplying it:

.. math::

    h(x \mid Z) = h_0(x) + \beta' Z

Here :math:`\beta_j` is a *risk difference* — the change in the absolute hazard
per unit of covariate, constant over time — rather than a hazard *ratio*. This
is the natural scale when you care about *how many extra failures per unit time*
a factor causes: excess risk in epidemiology, or reliability settings where
hazards from independent mechanisms genuinely add. Because the effect is
additive rather than exponential it is not constrained to be positive, which is
both its interpretive appeal and its main caveat (discussed below).

**Accelerated Life (AL)** — the covariate substitutes the distribution's *life
parameter*:

.. math::

    \theta(Z) = \phi(Z), \quad F(x \mid Z) = F\!\left(x;\,\theta(Z),\,\text{other params}\right)

This is the standard approach in *accelerated life testing* (ALT) — reliability
testing under elevated stress (high temperature, voltage, humidity) to extract
failure data quickly, then extrapolating back to use conditions. The stress
relationship :math:`\phi(Z)` is chosen from domain knowledge: Arrhenius for
thermally-activated failure, Eyring for quantum-mechanical processes, Power Law
for voltage or mechanical loading.

For PH, AFT, and PO the covariate function is always the log-linear form,

.. math::

    \phi(Z) = e^{\beta_1 z_1 + \beta_2 z_2 + \cdots}

because the exponential guarantees :math:`\phi > 0` for any covariate value and
any β — no parameter constraints needed. A positive β makes failure faster;
negative makes it slower.

SurPyval supports all of these families, each available as pre-built instances
for every standard distribution and as factory functions for custom
combinations (with additive hazards provided as a single semi-parametric
model):

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Family
     - Effect of covariates
     - Ready-to-use examples
   * - Proportional Hazards (PH)
     - Multiplies the hazard rate :math:`h(x|Z) = h_0(x)\,\phi(Z)`
     - ``WeibullPH``, ``ExponentialPH``, …
   * - Accelerated Failure Time (AFT)
     - Scales the time axis :math:`H(x|Z) = H_0(\phi(Z)\,x)`
     - ``WeibullAFT``, ``LogNormalAFT``, …
   * - Proportional Odds (PO)
     - Scales the survival odds :math:`O(x|Z) = O_0(x)\,\phi(Z)`
     - ``WeibullPO``, ``LogisticPO``, …
   * - Additive Hazards (AH)
     - Adds to the hazard rate :math:`h(x|Z) = h_0(x) + \beta'Z`
     - ``AdditiveHazards``
   * - Accelerated Life (AL)
     - Substitutes the life parameter with a physics-motivated function
     - ``AcceleratedLife(Weibull, Power)``, ``AcceleratedLife(Weibull, Eyring)``


Semi-Parametric — Cox Proportional Hazards
------------------------------------------

The Cox PH model is the most widely used survival regression model in any field.
Its central insight is that :math:`\beta` can be estimated *without* specifying
the shape of the baseline hazard :math:`h_0(x)`. The baseline cancels out of a
*partial likelihood*, leaving only the relative ordering of event times. This
means you can detect and quantify covariate effects even when you have no idea
what the baseline distribution looks like.

The price you pay is that the model is *semi-parametric*: once you have β, the
baseline is estimated non-parametrically (a step function with jumps only at
observed event times). This means predictions can only be made within the
observed time range, and extrapolation is not possible. If you need to predict
far beyond your observed data, a parametric PH model is more appropriate.

In this example we use data from Krivtsov et al., testing tires to failure with
seven measured characteristics. We want to know which characteristics
significantly affect tire life.

.. jupyter-execute::

    from surpyval.datasets import load_tires_data
    from surpyval import CoxPH

    tires = load_tires_data()
    x = tires['Survival']
    c = tires['Censoring']
    Z = tires[['Tire age', 'Wedge gauge', 'Interbelt gauge', 'EB2B', 'Peel force',
               'Carbon black (%)', 'Wedge gauge×peel force']]
    model = CoxPH.fit(x=x, Z=Z, c=c)
    model

We can immediately check which coefficients are statistically significant:

.. jupyter-execute::

    print(model.p_values)

Several covariates are not significant. We can re-fit with only the significant
ones, which also improves numerical stability:

.. jupyter-execute::

    Z = tires[['Wedge gauge', 'Interbelt gauge', 'Peel force',
               'Wedge gauge×peel force']]
    model = CoxPH.fit(x=x, Z=Z, c=c)
    print(model.p_values)
    model

All four coefficients are negative, meaning higher gauge and peel force values
*reduce* the hazard rate (improve life) — except the interaction term, which
captures a counteracting combined effect.

Survival curves can be evaluated at any covariate value. Here we compare the
mean tire against 10% above and below average:

.. jupyter-execute::

    Z_mean = Z.mean().values
    plot_x = np.linspace(x.min(), x.max())
    for f in [0.9, 1.0, 1.1]:
        plt.step(plot_x, model.sf(plot_x, Z=Z_mean * f), label=f'{f:.0%}')
    plt.legend(title='Covariate scale')
    plt.xlabel('Survival time')
    plt.ylabel('S(x)')

The step-function shape is the signature of the non-parametric baseline —
the model makes no smoothness assumptions about :math:`h_0(x)`.


Time-Varying Covariates
~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes a covariate changes *during* a subject's follow-up — a dose is
increased, a treatment begins, a machine is moved to a harsher environment.
Cox handles this through the **counting-process (start-stop) format**: each
subject contributes one row per interval :math:`(\text{start}, \text{stop}]`
on which its covariates are constant, and ``event`` is 1 only on the interval
that ends at the subject's failure. Because each interval is exactly a
delayed-entry (left-truncated) observation, the partial likelihood fits this
format directly — splitting a subject into intervals with the same covariates
leaves the fit unchanged.

Use :meth:`CoxPH.fit_tvc` (arrays) or :meth:`CoxPH.fit_tvc_from_df` (a
start-stop ``DataFrame``). In the example below a covariate ``stress`` switches
from 0 to 1 at a random time for each unit and genuinely raises the hazard once
it turns on; units that fail before the switch contribute a single interval,
those that survive it contribute two:

.. jupyter-execute::

    import pandas as pd
    from surpyval import CoxPH

    rng = np.random.default_rng(0)
    n, lam, beta = 2000, 0.5, 1.0
    switch = rng.uniform(0.3, 1.5, size=n)
    t_low = rng.exponential(1 / lam, size=n)
    t_high = switch + rng.exponential(1 / (lam * np.exp(beta)), size=n)
    T = np.where(t_low > switch, t_high, t_low)

    rows = []
    for i in range(n):
        if T[i] <= switch[i]:                    # failed before the switch
            rows.append((i, 0.0, T[i], 1, 0.0))
        else:                                    # survived it: two intervals
            rows.append((i, 0.0, switch[i], 0, 0.0))
            rows.append((i, switch[i], T[i], 1, 1.0))
    df = pd.DataFrame(rows, columns=['id', 'start', 'stop', 'event', 'stress'])

    model = CoxPH.fit_tvc_from_df(
        df, id_col='id', start_col='start', stop_col='stop',
        event_col='event', Z_cols='stress',
    )
    model

The fitted coefficient recovers the simulated log-hazard-ratio of the stress
(:math:`\beta \approx 1`) — a plain Cox fit that ignored the timing of the
switch could not.

Because survival now depends on the *whole* covariate path, evaluate it with
:meth:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel.predict_tvc`,
passing a subject's intervals. It returns the times, survival and cumulative
hazard along that path; with a single constant interval it reduces exactly to
``sf``. Here a unit stressed from ``t = 1`` onward has visibly lower survival
than one never stressed:

.. jupyter-execute::

    t_never, s_never, _ = model.predict_tvc(start=[0.0], stop=[3.0], Z=[[0.0]])
    t_late, s_late, _ = model.predict_tvc(
        start=[0.0, 1.0], stop=[1.0, 3.0], Z=[[0.0], [1.0]]
    )
    plt.step(t_never, s_never, where='post', label='never stressed')
    plt.step(t_late, s_late, where='post', label='stressed after t=1')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('S(t)')

Only left-truncation / delayed entry is available for Cox (supply a 1-D ``tl``
to :meth:`CoxPH.fit`); right or interval truncation is rejected, because the
forward partial likelihood cannot express it.


Semi-Parametric — Buckley-James (AFT)
-------------------------------------

Cox leaves the baseline *hazard* unspecified; Buckley-James is its
accelerated-time counterpart, leaving the error *distribution* unspecified:

.. math::

    \log T = \beta' Z + \varepsilon,

with :math:`\varepsilon` drawn from an arbitrary distribution estimated from the
data. It is fitted by the Buckley-James iteration — repeatedly imputing each
censored log-time by its conditional expectation under the Kaplan-Meier of the
current residuals, then re-fitting by least squares, until the coefficients stop
moving. As with Cox, coefficients are in surpyval's accelerated-failure sign: a
*positive* coefficient shortens life.

.. jupyter-execute::

    from surpyval import BuckleyJames

    # log T = 3 - 0.8 Z + noise, right-censored: higher Z shortens life, so
    # the accelerated-failure coefficient is +0.8.
    rng = np.random.default_rng(0)
    Z_bj = rng.normal(size=(400, 1))
    T_bj = np.exp(3.0 - 0.8 * Z_bj[:, 0] + rng.normal(0, 0.5, size=400))
    cens = np.exp(3.6)
    c_bj = (T_bj > cens).astype(int)
    x_bj = np.minimum(T_bj, cens)

    model = BuckleyJames.fit(x=x_bj, Z=Z_bj, c=c_bj)
    model

Buckley-James has no simple closed-form standard error, so uncertainty comes
from a percentile bootstrap — resampling, refitting, and taking coefficient
percentiles:

.. jupyter-execute::

    model.bootstrap_ci(n_boot=200, seed=1)

Predictions use the fitted residual distribution directly,
:math:`S(t \mid Z) = S_\varepsilon(\log t + \beta' Z)`, so the survival curves
shift with the covariate:

.. jupyter-execute::

    t = np.linspace(1, 60, 200)
    for z in [-1.0, 0.0, 1.0]:
        plt.step(t, model.sf(t, Z=[z]), where='post', label=f'Z = {z:g}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('S(t)')


Semi-Parametric — Additive Hazards
----------------------------------

The Lin & Ying additive hazards model is the additive-scale companion to Cox.
Like Cox it leaves the baseline hazard :math:`h_0(x)` completely unspecified,
but the covariate effect is a *risk difference* rather than a hazard *ratio*:

.. math::

    h(x \mid Z) = h_0(x) + \beta' Z

Its practical convenience is that, unlike Cox's iterative partial likelihood,
the coefficient estimator is *closed form* — a ratio of sums accumulated over
the risk sets — so there is no optimisation and nothing to converge. Standard
errors come from the Lin-Ying sandwich estimator. We can reuse the tire data
and the significant covariates from the Cox fit above:

.. jupyter-execute::

    from surpyval import AdditiveHazards

    model = AdditiveHazards.fit(x=x, Z=Z, c=c)
    model

.. jupyter-execute::

    print(model.p_values)

The coefficients read as risk differences: a one-unit change in a covariate
shifts the absolute hazard by :math:`\beta` at every time. As in the Cox fit,
higher gauge and peel-force values reduce the hazard (improving life) while the
interaction term counteracts — the same story, told on the additive scale.

.. note::

   An additive hazard can go **negative** when :math:`\beta' Z` is sufficiently
   negative — nothing constrains :math:`h_0(x) + \beta' Z > 0`. When that
   happens the fitted cumulative hazard is no longer monotone and the implied
   survival can rise above 1. SurPyval returns the raw estimate without
   clamping; a survival above 1 is a signal that the additive model is a poor
   description at that covariate value (or that you are outside the range where
   it is well behaved), and is best read as a caution rather than a prediction.
   This is an inherent property of additive-hazards models, not a defect of the
   fit. When covariate effects are strongly protective, a proportional-hazards
   model — whose exponential form keeps the hazard positive — is often the
   safer choice.

Just as Cox has parametric proportional-hazards counterparts (the next
section), there is also a *parametric* additive-hazards model — a parametric
baseline hazard with the same additive covariate term, ``h(x|Z) = h_0(x;θ) +
β'Z``, fit by maximum likelihood. It is available as the ``AH(distribution)``
factory and as pre-built ``WeibullAH``, ``ExponentialAH``, … instances, and
gives a smooth, extrapolatable version of what ``AdditiveHazards`` estimates
non-parametrically. The positivity caveat above applies more sharply here:
because the likelihood needs ``log(h)`` at every event, a fit whose optimum
requires a negative hazard raises rather than returning an invalid model.


Parametric Proportional Hazards (PH)
--------------------------------------

When you are confident about the shape of the baseline distribution — or when
you need to extrapolate beyond the observed time range — a fully parametric PH
model is preferable. It estimates the same β as the Cox model, but also
estimates the baseline distribution parameters, giving a smooth, continuous
survival function.

SurPyval provides pre-built parametric PH instances for every standard
distribution: ``ExponentialPH``, ``NormalPH``, ``WeibullPH``, ``GumbelPH``,
``LogisticPH``, ``LogNormalPH``, and ``GammaPH``. The Weibull is the most
common choice in reliability engineering — its shape parameter lets it capture
increasing, constant, or decreasing hazard rates.

.. jupyter-execute::

    from surpyval import WeibullPH

    model = WeibullPH.fit(x=x, Z=Z, c=c)
    model

Notice the coefficients are very close to the Cox model — this is expected when
the Weibull is a reasonable fit to the baseline.

If none of the pre-built distributions suit your data, the ``PH`` factory creates
a parametric PH model for any surpyval distribution:

.. jupyter-execute::

    from surpyval import LogNormal
    from surpyval import PH

    model = PH(LogNormal).fit(x=x, Z=Z, c=c)
    model

The log-normal is a natural choice when the log of the survival time is expected
to be normally distributed — common in medical and biological data.


Accelerated Failure Time (AFT)
--------------------------------

The AFT model has a different and often more interpretable structure than PH.
Rather than saying "this covariate increases your hazard rate by X%", it says
"this covariate makes you age X% faster". Formally, if the baseline survival
time is :math:`T_0`, then the survival time given covariates is:

.. math::

    T \mid Z \;=\; \frac{T_0}{\phi(Z)} \;=\; T_0 \cdot e^{-\beta' Z}

A positive :math:`\beta_j` means covariate :math:`z_j` compresses time
(accelerates failure). A negative :math:`\beta_j` stretches time (prolongs
life). The median survival time simply scales by :math:`e^{-\beta' Z}` —
a direct and intuitive interpretation.

The relationship to the cumulative hazard is:

.. math::

    H(x \mid Z) = H_0\!\left(e^{\beta'Z} \cdot x\right)

An important practical note: for scale-family distributions (Weibull,
Log-Normal, Exponential), the Weibull-AFT and Weibull-PH are different
parameterisations of the same model family — they will produce equivalent
fits with re-parameterised coefficients. For other distributions they are
genuinely distinct.

.. jupyter-execute::

    from surpyval import WeibullAFT

    model = WeibullAFT.fit(x=x, Z=Z, c=c)
    model

The ``AFT`` factory works with any distribution. Log-Normal AFT is a
particularly common choice — it corresponds to ordinary linear regression on
:math:`\log T` with censored observations, and is closely related to the
accelerated failure time model of Buckley & James:

.. jupyter-execute::

    from surpyval import LogNormalAFT

    model_ln = LogNormalAFT.fit(x=x, Z=Z, c=c)
    model_ln

For distributions not in the pre-built list:

.. jupyter-execute::

    from surpyval import Gamma
    from surpyval import AFT

    model = AFT(Gamma).fit(x=x, Z=Z, c=c)
    model

We can visualise the "time shift" interpretation by plotting survival curves.
The entire curve moves left (shorter life) or right (longer life) on the time
axis as the covariates change — a hallmark of the AFT model:

.. jupyter-execute::

    model = WeibullAFT.fit(x=x, Z=Z, c=c)
    Z_mean = Z.mean().values
    plot_x = np.linspace(x.min(), x.max())
    for f in [0.9, 1.0, 1.1]:
        plt.plot(plot_x, model.sf(plot_x, Z=Z_mean * f), label=f'{f:.0%}')
    plt.legend(title='Covariate scale')
    plt.xlabel('Survival time')
    plt.ylabel('S(x)')

Compare this with the Cox PH survival curves above: the PH curves cross or
converge at long times; the AFT curves remain parallel on the log-time scale.


Proportional Odds (PO)
-----------------------

The proportional odds model is less common than PH or AFT but has an important
niche: it is the right model when you believe the *relative odds* of failure are
constant across covariate values, rather than the relative hazard rates.

The survival odds at time :math:`x` are :math:`O(x) = S(x) / F(x)`. A PO model
assumes these are scaled by :math:`\phi(Z)`:

.. math::

    \frac{S(x \mid Z)}{F(x \mid Z)} = \frac{S_0(x)}{F_0(x)} \cdot e^{\beta' Z}

Rearranging, the survival function is:

.. math::

    S(x \mid Z) = \frac{e^{\beta' Z} \cdot S_0(x)}{F_0(x) + e^{\beta' Z} \cdot S_0(x)}

The key difference from PH becomes clear at long follow-up: as :math:`x \to
\infty`, :math:`S_0(x) \to 0`, and the ratio :math:`F_0 + \phi S_0 \to 1`, so
the covariate effect *fades away*. Everyone eventually fails, and the PO model
respects that by letting the hazard ratio converge to 1 over time. Under PH,
the hazard ratio is constant forever — a stronger and often unrealistic
assumption for long studies.

PO is the natural companion to the Logistic and Log-Logistic distributions
because it corresponds to logistic regression at each time point.

.. jupyter-execute::

    from surpyval import LogisticPO

    model = LogisticPO.fit(x=x, Z=Z, c=c)
    model

The ``PO`` factory accepts any distribution:

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval import PO

    model = PO(Weibull).fit(x=x, Z=Z, c=c)
    model

A practical rule of thumb: if the Kaplan-Meier curves for different covariate
groups converge at long times (rather than remaining parallel on the log-hazard
scale), PO is likely a better fit than PH.


Confidence Bounds
-----------------

A point estimate is only half the story. The parametric regression models (PH,
AFT, PO, AL) carry the full parameter covariance, so every coefficient and every
predicted curve comes with an interval. After a fit, the parameter covariance
and standard errors are available directly:

.. jupyter-execute::

    from surpyval import WeibullPH

    rng = np.random.default_rng(0)
    Z_cb = rng.normal(size=(300, 1))
    x_cb = 10.0 * (
        -np.log(rng.uniform(size=300)) / np.exp(Z_cb[:, 0] * 0.8)
    ) ** (1 / 2.0)
    m_cb = WeibullPH.fit(x=x_cb, Z=Z_cb, c=np.zeros(300, dtype=int))

    print(m_cb.parameter_names())
    print(m_cb.standard_errors())

``param_cb`` gives a Wald confidence bound on a single parameter, computed on a
scale chosen from the parameter's support (log for a positive scale, natural for
an unbounded coefficient) so the interval always stays valid:

.. jupyter-execute::

    m_cb.param_cb('beta_0')      # 95% CI for the covariate coefficient

``cb`` propagates the parameter covariance through a predicted function by the
delta method, returning a confidence *band*. Here is the survival at a covariate
value with its 95% band:

.. jupyter-execute::

    x_grid = np.linspace(1, 40, 200)
    band = m_cb.cb(x_grid, Z=[0.5], on='sf')      # (n, 2): [lower, upper]
    sf = m_cb.sf(x_grid, Z=[0.5])
    plt.plot(x_grid, sf, 'b', label='S(x | Z=0.5)')
    plt.fill_between(x_grid, band[:, 0], band[:, 1], alpha=0.2,
                     label='95% confidence band')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('S(x)')

``cb`` accepts ``on='ff'``, ``'Hf'``, ``'hf'`` or ``'df'``, one- or two-sided
bounds via ``bound=``, and any ``alpha_ci``. The convenience method
``model.plot()`` draws the fitted survival at the mean covariate against the
Kaplan-Meier with this band already applied.


.. _accelerated-life:

Accelerated Life (AL)
----------------------

Accelerated life testing (ALT) is a branch of reliability engineering where
products are tested under *elevated stress conditions* — higher temperature,
voltage, humidity, or load — to generate failure data faster than would be
possible at normal operating conditions. The failures observed at high stress
are then extrapolated back to normal conditions using a physical model for how
the stress affects the life of the product.

This is fundamentally different from the regression models above. In PH, AFT,
and PO, the covariates are measured characteristics of each unit (e.g. tire
gauge, patient age). In AL, the covariate is a controlled experimental condition
(stress level), and there are typically only two or three distinct levels. The
relationship between stress and life is not statistical but physical, and the
choice of life model reflects domain knowledge about the failure mechanism.

The AL model substitutes the life parameter :math:`\theta` of a distribution
with a stress function :math:`\phi(Z)`:

.. math::

    F(x \mid Z) = F\!\left(x;\; \phi(Z),\; \text{other params}\right)

For example, in a Weibull AL model the scale parameter :math:`\alpha` becomes
:math:`\phi(Z)`, while the shape parameter :math:`\beta` is estimated globally
across all stress levels (the assumption being that the failure mechanism is the
same at all stresses, just faster or slower).

Available life models
~~~~~~~~~~~~~~~~~~~~~~

The choice of life model depends on the physical failure mechanism:

.. list-table::
   :header-rows: 1
   :widths: 26 38 36

   * - Life model
     - Formula :math:`\phi(Z)`
     - Typical use
   * - ``ExponentialLifeModel``
     - :math:`b \cdot e^{a/Z}` (Arrhenius)
     - Thermally-activated (chemical, diffusion, electromigration)
   * - ``Eyring``
     - :math:`Z^{-1} e^{-(c - a/Z)}`
     - Quantum-mechanical processes; more accurate than Arrhenius at
       extreme temperatures
   * - ``InversePower``
     - :math:`1 / (a \cdot Z^n)`
     - Voltage, electrical field, mechanical fatigue
   * - ``Power``
     - :math:`a \cdot Z^n`
     - Same as InversePower but with life increasing with stress
   * - ``Linear``
     - :math:`a + b \cdot Z`
     - Simple first-order approximation; valid over narrow stress ranges
   * - ``DualExponential``
     - :math:`c \cdot e^{a/Z_1} e^{b/Z_2}`
     - Two thermal stresses
   * - ``DualPower``
     - :math:`c \cdot Z_1^m Z_2^n`
     - Two non-thermal stresses
   * - ``PowerExponential``
     - :math:`c \cdot e^{a/Z_1} Z_2^n`
     - One thermal + one non-thermal
   * - ``InverseEyring``
     - Reciprocal of Eyring
     - Inverse Eyring relationship
   * - ``InverseExponential``
     - Reciprocal of Arrhenius
     - Inverse Arrhenius relationship

A note on units: the stress variable :math:`Z` for Arrhenius and Eyring should
be in Kelvin (absolute temperature), not Celsius.

Using the factory
~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval import AcceleratedLife, Power, ExponentialLifeModel

    # Discrete stress levels — three temperatures in Kelvin
    stress = np.array([358.]*20 + [378.]*20 + [398.]*20)  # 85°C, 105°C, 125°C
    np.random.seed(42)
    # Simulate Arrhenius-like failure times (life halves every ~20°C)
    Ea, k = 0.7, 8.617e-5   # activation energy eV, Boltzmann constant eV/K
    scale = np.exp(Ea / (k * stress))
    x_al = np.random.weibull(2.5, 60) * scale / scale.max() * 5000
    c_al = np.ones(60, dtype=int)
    c_al[::4] = 1   # every 4th observation right-censored

    Z_al = stress.reshape(-1, 1)

    # Weibull + Arrhenius (ExponentialLifeModel) — the most common ALT model
    model_arr = AcceleratedLife(Weibull, ExponentialLifeModel).fit(x_al, Z=Z_al)
    model_arr

Notice that the Weibull shape parameter :math:`\beta` is estimated globally —
it is the same for all stress levels — while the scale parameter :math:`\alpha`
varies with stress via the Arrhenius relationship. This is the key assumption of
ALT: the failure mechanism does not change with stress, only the rate.

.. jupyter-execute::

    # Power law — a common choice for voltage or load acceleration
    model_power = AcceleratedLife(Weibull, Power).fit(x_al, Z=Z_al)
    model_power

To use the fitted model for extrapolation, pass the operating stress to any
of the survival functions:

.. jupyter-execute::

    # Predict life at operating temperature 85°C = 358K
    x_pred = np.linspace(0, 10000, 500)
    Z_use = np.array([[358.]])   # operating condition

    plt.plot(x_pred, model_arr.sf(x_pred, Z=Z_use), label='Predicted at 85°C (358K)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Reliability')
    plt.legend()
    plt.title('Extrapolated life at operating conditions')

Creating a custom life model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If none of the built-in life models matches your failure physics, you can define
your own by subclassing ``LifeModel``. The two methods you must implement are:

- ``phi(Z, *params)`` — the stress relationship. Use ``autograd.numpy`` so that
  gradients are available for the optimiser.
- ``phi_init(life, Z)`` — a closed-form or least-squares initialiser for the
  model parameters. ``life`` is a vector of estimated life parameters at each
  unique stress level; ``Z`` is the corresponding stress values. Good
  initialisation is important for convergence.

.. jupyter-execute::

    from surpyval import LifeModel, AcceleratedLife
    from surpyval import Weibull
    import autograd.numpy as anp

    class InverseSquareRoot(LifeModel):
        """Life proportional to 1/sqrt(Z) — a simple custom example."""
        def __init__(self):
            super().__init__(
                name="InverseSquareRoot",
                phi_param_map={"a": 0},
                phi_bounds=((0, None),),
            )

        def phi(self, Z, *params):
            a = params[0]
            return a / anp.sqrt(Z)

        def phi_init(self, life, Z):
            # life ~ a / sqrt(Z) => a ~ life * sqrt(Z)
            a_est = float(anp.mean(life * anp.sqrt(Z.flatten())))
            return [a_est]

    model_custom = AcceleratedLife(Weibull, InverseSquareRoot()).fit(x_al, Z=Z_al)
    model_custom


Model Selection
---------------

With several competing models it is useful to compare them on information
criteria. AIC penalises log-likelihood by the number of parameters (favouring
simpler models); BIC additionally penalises by sample size (favouring even
simpler models with larger datasets). Lower is better for both.

For the tires data, we can compare all three statistical regression families
using the same Weibull baseline:

.. jupyter-execute::

    from surpyval import WeibullAFT, WeibullPH
    from surpyval import PO
    from surpyval import Weibull

    models = {
        'WeibullPH':  WeibullPH.fit(x=x, Z=Z, c=c),
        'WeibullAFT': WeibullAFT.fit(x=x, Z=Z, c=c),
        'WeibullPO':  PO(Weibull).fit(x=x, Z=Z, c=c),
    }

    for name, m in models.items():
        print(f'{name:12s}  AIC={m.aic():.2f}  BIC={m.bic():.2f}')

A note of caution: AIC and BIC compare how well a model fits the *observed
data*, not whether the model's assumptions are correct. A PH model with a lower
AIC than a PO model does not mean PH is the "true" model — it means PH uses its
parameters more efficiently on this dataset. If the proportional hazards
assumption is violated (e.g. survival curves cross), a lower-AIC PH model can
still give misleading predictions. Goodness-of-fit diagnostics like
Schoenfeld residuals (for PH) or log-log survival plots should accompany any
model comparison.

Saving and loading a fitted model
---------------------------------

A fitted parametric regression model can be serialised to a plain dictionary
or a JSON file and rebuilt later — so you can fit once and reuse the model
without the training data on hand. This works for the fixed-form parametric
families: Accelerated Failure Time, Proportional Hazards, Proportional Odds
and (parametric) Additive Hazards.

.. jupyter-execute::

    import tempfile, os
    from surpyval import WeibullAFT
    from surpyval.univariate.regression import ParametricRegressionModel

    model = WeibullAFT.fit(x=x, Z=Z, c=c)

    # to a dictionary (JSON-serialisable) ...
    blob = model.to_dict()

    # ... and back
    restored = ParametricRegressionModel.from_dict(blob)

    # the restored model predicts identically
    import numpy as np
    t = np.array([1.0, 5.0, 20.0])
    Z_use = np.asarray(Z)[0]
    print("match:", np.allclose(model.sf(t, Z_use), restored.sf(t, Z_use)))

Use ``to_json`` / ``from_json`` for a file directly:

.. jupyter-execute::

    path = os.path.join(tempfile.mkdtemp(), "aft.json")
    model.to_json(path)
    reloaded = ParametricRegressionModel.from_json(path)
    print(reloaded)

If you don't know (or don't want to hard-code) which class wrote a file, the
package-level readers ``surpyval.from_json`` / ``surpyval.from_dict`` dispatch
on the serialised content itself and work for every serialisable SurPyval
model:

.. jupyter-execute::

    import surpyval

    print(surpyval.from_json(path))

Storing models in MongoDB
~~~~~~~~~~~~~~~~~~~~~~~~~

``to_dict`` emits documents of native Python types only — string keys, lists,
floats, ints — so its output is BSON-safe and every serialisable model can be
stored in MongoDB directly. On the way back, ``surpyval.from_dict`` ignores
the ``_id`` field MongoDB adds and restores the right class from the document
itself:

.. code:: python

    collection.insert_one(model.to_dict())

    doc = collection.find_one({"distribution": "Weibull"})
    model = surpyval.from_dict(doc)

Every document also carries a ``"schema"`` version stamped by ``to_dict``.
It changes only when a document's shape changes incompatibly, so models
stored today stay recognisable to future SurPyval versions; a document
written by a *newer* schema than the installed SurPyval understands is
refused with a clear error rather than misread.

If the fitted model carried a computable parameter covariance, it is stored in
the dictionary, so the reloaded model can also produce confidence bounds
(``cb``, ``param_cb``, ``standard_errors``) without the original data. Only the
prediction/inference state is serialised — the empirical overlay in ``plot``
needs the fitted data, so re-fit if you need that. Models with a bespoke
covariate link (an Accelerated Life parameter-substitution model, whose link is
an arbitrary life-model) cannot be rebuilt from a name and raise
``NotImplementedError`` when serialised.

The **semi-parametric** regression models save and load the same way, each on
its own result class: Cox proportional hazards
(``SemiParametricRegressionModel``), the Lin-Ying additive-hazards model
(``AdditiveHazardsModel``), and the Buckley-James AFT (``BuckleyJamesModel``).
Because their baseline is nonparametric, what is stored is the fitted
coefficients plus the baseline step arrays (or, for Buckley-James, the residual
survival), so the reloaded model predicts identically:

.. jupyter-execute::

    from surpyval import CoxPH
    from surpyval.univariate.regression import SemiParametricRegressionModel

    cox = CoxPH.fit(x=x, Z=Z, c=c)
    cox_reloaded = SemiParametricRegressionModel.from_dict(cox.to_dict())
    print("match:", np.allclose(cox.sf(t, Z_use), cox_reloaded.sf(t, Z_use)))

Cox's time-varying-covariate prediction (``predict_tvc``), the additive model's
covariance, and Buckley-James's ``bootstrap_ci`` (which keeps a copy of the fit
data) all survive the round-trip.
