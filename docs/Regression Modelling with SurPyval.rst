
Regression Modelling with SurPyval
==================================

The time until an event — a failure, death, recovery — will almost always depend
on external factors. A bearing may last longer in a cool, clean environment than
in a hot, dirty one. A patient's survival may depend on age, dosage, and
comorbidities. The question regression modelling answers is: *how much* do these
factors matter, and in what direction?

For the concepts and mathematics behind the regression families used here — the
proportional-hazards, accelerated-failure-time, accelerated-life, proportional-odds
and additive-hazards models, the Cox partial likelihood and its tie corrections,
the Cox diagnostics, robust, stratified and frailty fits, time-varying
covariates, survival trees, and prediction-validation metrics — see the
:doc:`regression analysis` page. This page is the practical companion: every
example runs, and each section links back to the theory it relies on. The full
API reference is under :doc:`surpyval.regression`.

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

**How this page is organised.** It is long, and it builds up in four parts.
First, how the families differ and the conventions every fitter shares (the
data arguments and how predictions pair times with covariates). Second, the
*semi-parametric* models, which leave the baseline to the data: the Cox model
with its whole toolkit — tied times, delayed entry, DataFrames and formulas,
time-varying covariates, the proportional-hazards check, robust errors and
strata — then its accelerated-time counterpart Buckley-James and the
additive-hazards model. Third, the *parametric* families — proportional
hazards, accelerated failure time, proportional odds, their confidence bounds,
accelerated life testing, time-varying covariates across families and shared
frailty. Fourth, choosing between fitted models, validating their predictions,
and saving them. If you already know the question you are asking, the table
below points straight to the section that answers it.

.. list-table:: Which model, when
   :header-rows: 1
   :widths: 44 28 28

   * - If you want to ...
     - use
     - see
   * - estimate hazard ratios without assuming a lifetime distribution
     - ``CoxPH``
     - `Semi-Parametric — Cox Proportional Hazards`_
   * - predict the whole lifetime distribution, extrapolate beyond the data,
       or use left- or interval-censored data
     - ``WeibullPH``, ``PH(dist)``, and the other parametric families
     - `Parametric Proportional Hazards (PH)`_
   * - say "this factor costs a fraction of the life" (a time ratio)
     - ``WeibullAFT``, ``LogNormalAFT``; ``BuckleyJames`` with no
       distribution assumed
     - `Accelerated Failure Time (AFT)`_,
       `Semi-Parametric — Buckley-James (AFT)`_
   * - carry accelerated-test results to use conditions through a physical
       stress-life law
     - ``AcceleratedLife(Weibull, ExponentialLifeModel)``, ...
     - `Accelerated Life (AL)`_
   * - model an effect that fades as time goes on
     - ``LogisticPO``, ``PO(dist)``
     - `Proportional Odds (PO)`_
   * - report an excess risk (extra failures per unit time)
     - ``AdditiveHazards``, ``WeibullAH``
     - `Semi-Parametric — Additive Hazards`_
   * - use covariates that change during follow-up, or forecast along a
       planned covariate path
     - ``fit_tvc`` / ``sf_tvc`` (Cox, PH, AH, AFT)
     - `Time-Varying Covariates`_, `Time-varying covariates across families`_
   * - check that a hazard ratio really is constant
     - ``model.check_ph()``
     - `Checking the proportional-hazards assumption`_
   * - get honest standard errors for grouped or repeated data
     - ``model.robust_summary(cluster=...)``
     - `Cluster-robust standard errors`_
   * - model, and predict for, the variation between groups
     - ``WeibullFrailty``, ``Frailty(dist)``
     - `Shared-frailty models`_
   * - remove a nuisance factor that breaks proportional hazards
     - ``CoxPH.fit(..., strata=...)``
     - `Stratified Cox models`_
   * - find structure you cannot specify (thresholds, interactions)
     - ``RandomSurvivalForest`` (beta)
     - `Survival trees and random survival forests (beta)`_
   * - compare, validate or store fitted models
     - ``aic()``, ``surpyval.metrics``, ``to_dict()``
     - `Model Selection`_, `Validating a survival predictor`_,
       `Saving and loading a fitted model`_

.. contents:: On this page
   :local:
   :depth: 1


Choosing a regression model family
------------------------------------

There are four fundamentally different ways a covariate can affect a survival
distribution through a simple link, plus the physics-driven accelerated life
model. Each gives rise to a distinct model family:

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
*entire* survival curve shifts left or right on the (log) time axis. This is often a
more natural framing in engineering and materials science.

**Proportional Odds (PO)** — the covariate scales the *odds of surviving*:

.. math::

    \frac{S(x \mid Z)}{F(x \mid Z)} = \frac{S_0(x)}{F_0(x)} \cdot \phi(Z)

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
any β — no parameter constraints needed. For PH and AFT a positive β makes
failure faster and a negative one slower. **Proportional odds is the exception**:
its :math:`\phi` multiplies the odds of *survival*, so a positive β makes
failure *slower*. Expect a PO fit to report coefficients of the opposite sign to
a PH fit on the same data.

SurPyval supports all of these families, each available as pre-built instances
for every standard distribution and as factory functions for custom
combinations, together with semi-parametric versions that leave the baseline
unspecified, a frailty version for grouped data, and tree-based predictors:

.. list-table::
   :header-rows: 1
   :widths: 22 42 36

   * - Family
     - Effect of covariates
     - Ready-to-use examples
   * - Proportional Hazards (PH)
     - Multiplies the hazard rate :math:`h(x|Z) = h_0(x)\,\phi(Z)`
     - ``WeibullPH``, ``ExponentialPH``, …, ``PH(dist)``, and the
       semi-parametric ``CoxPH``
   * - Accelerated Failure Time (AFT)
     - Scales the time axis :math:`H(x|Z) = H_0(\phi(Z)\,x)`
     - ``WeibullAFT``, ``LogNormalAFT``, …, ``AFT(dist)``, and the
       semi-parametric ``BuckleyJames``
   * - Proportional Odds (PO)
     - Scales the survival odds :math:`O(x|Z) = O_0(x)\,\phi(Z)`
     - ``WeibullPO``, ``LogisticPO``, …, ``PO(dist)``
   * - Additive Hazards (AH)
     - Adds to the hazard rate :math:`h(x|Z) = h_0(x) + \beta'Z`
     - ``AdditiveHazards`` (semi-parametric), ``WeibullAH``, …, ``AH(dist)``
   * - Accelerated Life (AL)
     - Substitutes the life parameter with a physics-motivated function
     - ``AcceleratedLife(Weibull, Power)``, ``AcceleratedLife(Weibull, Eyring)``
   * - Shared frailty PH
     - PH with a random multiplier shared within a group
     - ``WeibullFrailty``, …, ``Frailty(dist)``
   * - Survival trees and forests (beta)
     - No link: recursive splits on the covariates
     - ``SurvivalTree``, ``RandomSurvivalForest`` in ``surpyval.beta.ml``

Data, covariates and predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every regression fitter takes the observed times ``x`` and a covariate matrix
``Z`` with one row per observation and one column per covariate (a
one-dimensional ``Z`` is read as a single covariate, one value per row) —
plus surpyval's usual optional arrays: the censoring flag ``c`` (``0`` observed,
``1`` right, ``-1`` left, ``2`` interval censored) and counts ``n``. What each
fitter accepts:

- the parametric families (PH, AFT, PO, AH and AL) accept every censoring
  type, and truncation through ``t`` (a two-column ``[tl, tr]`` array);
- ``CoxPH`` takes observed and right-censored data, with left truncation
  through a 1-D ``tl``, and refuses left- or interval-censored rows (the
  partial likelihood has no term for them);
- the Lin-Ying (``AdditiveHazards``), Buckley-James and frailty fitters take
  observed and right-censored data only, and say so if given anything else.

A row with a missing (``NaN``) or infinite covariate cannot enter any of
these likelihoods, so every fitter drops it — from the times, flags, counts
and truncation too — and warns with the number of rows dropped. The same
goes for ``fit_from_df``, with named columns or a ``formula``. (The
time-varying-covariate fits are the exception: dropping one interval would
change a subject's history, so they refuse a missing covariate instead.)
Predicting from a DataFrame row with a missing covariate gives ``nan`` for
that row, in its place.

Each family also has a ``fit_from_df`` that names DataFrame columns instead
(see `Fitting from a DataFrame: formulas and categorical covariates`_).

Predictions — ``sf``, ``ff``, ``df``, ``hf`` and ``Hf`` — take times and
covariates. Given **one** covariate row they return the curve over all the
times; given ``n`` rows and ``n`` times they pair them **element-wise**, one
time per row, which is what you want for scoring a data set but not for drawing
several curves. To draw curves for several covariate values, call once per
value. (Buckley-James predictions take a single covariate row only, and the
random survival forest returns a full grid; both are noted in their sections.)
A small simulated data set shows both forms:

.. jupyter-execute::

    from surpyval import WeibullPH

    rng = np.random.default_rng(42)
    Z_demo = rng.binomial(1, 0.5, size=(300, 1)).astype(float)
    # Weibull(10, 2) baseline; exposure multiplies the hazard by e^0.7 ~ 2
    x_demo = 10 * rng.weibull(2.0, 300) * np.exp(-0.7 * Z_demo[:, 0] / 2.0)
    demo = WeibullPH.fit(x=x_demo, Z=Z_demo)
    print('alpha, beta, beta_0 :', demo.params.round(3))

    print('one row, three times :', demo.sf([5.0, 10.0, 15.0], Z=[1.0]).round(3))
    print('two rows, paired     :', demo.sf([5.0, 5.0], Z=[[0.0], [1.0]]).round(3))

The regression models do not have a quantile function (``qf``). A quantile at a
given covariate value is the root of :math:`S(x \mid Z) = 1 - p`, which a
bracketing root-finder finds reliably because ``sf`` is monotone:

.. jupyter-execute::

    from scipy.optimize import brentq

    for z in [0.0, 1.0]:
        median = brentq(lambda t: demo.sf([t], Z=[z])[0] - 0.5, 1e-6, 100.0)
        print(f'median life at Z = {z:g}: {median:.2f}')

With a hazard ratio of about 2 and a Weibull shape of 2, the exposed median is
shorter by a factor of about :math:`2^{1/2}` — exactly what the PH/AFT
equivalence for the Weibull (see `Accelerated Failure Time (AFT)`_) predicts.


Semi-Parametric — Cox Proportional Hazards
------------------------------------------

The Cox PH model is the most widely used survival regression model in any field.
Its central insight is that :math:`\beta` can be estimated *without* specifying
the shape of the baseline hazard :math:`h_0(x)`. The baseline cancels out of a
*partial likelihood*, leaving only the relative ordering of event times. This
means you can detect and quantify covariate effects even when you have no idea
what the baseline distribution looks like (the derivation is in the
:doc:`regression analysis` page).

The price you pay is that the model is *semi-parametric*: once you have β, the
baseline is estimated non-parametrically (a step function with jumps only at
observed times). This means predictions can only be made within the
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

Several covariates are not significant at the 5% level (the first, fourth and
sixth). We can re-fit with only the significant ones, which also improves
numerical stability — there are only 34 tires, 11 of them failures, so every
extra coefficient is expensive:

.. jupyter-execute::

    Z = tires[['Wedge gauge', 'Interbelt gauge', 'Peel force',
               'Wedge gauge×peel force']]
    model = CoxPH.fit(x=x, Z=Z, c=c)
    print(model.p_values)
    model

The first three coefficients are negative, meaning higher gauge and peel force
values *reduce* the hazard rate (improve life); the positive interaction term
captures a counteracting combined effect.

A coefficient is a log hazard ratio per unit of its covariate. The fitted model
keeps the score and information closures of its partial likelihood
(``model.jac(beta)`` returns the pair), so the model-based standard errors are
the square roots of the diagonal of the inverse information:

.. jupyter-execute::

    info = model.jac(model.beta)[1]                 # observed information
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    for name, b, s, p in zip(Z.columns, model.beta, se, model.p_values):
        print(f'{name:24s} beta = {b:7.2f}  se = {s:5.2f}  p = {p:.3f}')

The ``p_values`` are exactly the Wald tests :math:`2(1 - \Phi(|\beta/\text{se}|))`
built from these standard errors. Because the model contains an interaction,
no single coefficient can be changed on its own — raising peel force also
raises the interaction column — so a hazard ratio is best computed between two
concrete tires. ``model.phi(Z)`` returns the multiplier :math:`e^{\beta'Z}`, and
the ratio of two multipliers is their hazard ratio at every time:

.. jupyter-execute::

    Z_mean = Z.mean().values
    for f in [0.9, 1.1]:
        hr = model.phi(Z_mean * f) / model.phi(Z_mean)
        print(f'every covariate x {f}: hazard ratio against the mean tire = '
              f'{hr:.2f}')

Survival curves can be evaluated at any covariate value. Here we compare the
mean tire against 10% above and below average:

.. jupyter-execute::

    plot_x = np.linspace(x.min(), x.max())
    for f in [0.9, 1.0, 1.1]:
        plt.step(plot_x, model.sf(plot_x, Z=Z_mean * f), label=f'{f:.0%}')
    plt.legend(title='Covariate scale')
    plt.xlabel('Survival time')
    plt.ylabel('S(x)')
    plt.show()

The step-function shape is the signature of the non-parametric baseline —
the model makes no smoothness assumptions about :math:`h_0(x)`. Keep the
evaluation times inside the observed range: the baseline is only estimated
there, and a Cox model's ``hf`` returns the size of the baseline *step* at the
latest observed time, not a smooth hazard rate.

Tied event times
~~~~~~~~~~~~~~~~

When failure times are recorded coarsely — to the day, the shift, the
inspection — several units share a time and the partial likelihood needs a tie
convention, chosen with ``method=``: ``'breslow'``, ``'efron'``, ``'exact'`` or
``'kalbfleisch-prentice'`` (``'kp'``). ``CoxPH.fit`` defaults to Breslow;
``CoxPH.fit_from_df`` and the time-varying-covariate fits default to Efron,
which is also the default of R and lifelines. Below, fifty units have
continuous lifetimes that were recorded only to the whole day, so up to six
share a day; each method is compared with the fit to the unrounded times,
which is the answer rounding took away:

.. jupyter-execute::

    rng = np.random.default_rng(0)
    z_tie = rng.binomial(1, 0.5, 50).astype(float)
    t_true = 10 * rng.weibull(2, 50) * np.exp(-0.7 * z_tie / 2)  # true beta 0.7
    t_day = np.ceil(t_true)                        # recorded to the whole day

    counts = np.unique(t_day, return_counts=True)[1]
    print('distinct days:', counts.size, '  largest tie:', counts.max())
    print('unrounded times        : beta = %.3f'
          % CoxPH.fit(x=t_true, Z=z_tie).beta[0])
    for method in ['breslow', 'efron', 'exact', 'kalbfleisch-prentice']:
        m = CoxPH.fit(x=t_day, Z=z_tie, method=method)
        print(f'{method:22s} : beta = {m.beta[0]:.3f}')

Breslow's approximation pulls the coefficient towards zero; Efron's recovers
almost all of what rounding lost, and ``'exact'`` — which averages the
likelihood over every order in which the tied failures could have happened —
lands closest to the unrounded fit, as it should for rounded continuous time.
``'kalbfleisch-prentice'`` (alias ``'kp'``) answers a different question: it
treats time as genuinely discrete, so its coefficient is a log *odds* ratio of
failing within a day rather than a log hazard ratio, and is larger here for
that reason, not because it is more accurate. Use it when time really is
discrete (a unit can only fail at an inspection), and compare it only with
other discrete-time fits. The two exact methods cost more than Efron — each
tie group is a recursion (``'kalbfleisch-prentice'``) or a numerical integral
(``'exact'``) rather than a closed form — but both grow only polynomially with
the size of a tie group, so they remain practical on heavily tied data. With
no ties all four agree.

Delayed entry (left truncation)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A unit that only came under observation after it had already survived for a
while — equipment bought second-hand, patients enrolled some time after
diagnosis — must not be treated as if it had been watched from new: that would
credit it with survival it was never at risk of failing to show. Pass the entry
ages as ``tl`` to ``CoxPH.fit`` (for the parametric families, as the first
column of ``t``, or ``tl_col`` in ``fit_from_df``). Below, units whose failure
came before their entry age were never seen at all, as happens in practice:

.. jupyter-execute::

    rng = np.random.default_rng(1)
    z_all = rng.normal(size=900)
    T_all = 10 * rng.weibull(1.5, 900) * np.exp(-0.8 * z_all / 1.5)  # beta = 0.8
    entry_all = rng.uniform(0, 8, 900)
    seen = T_all > entry_all                   # the rest failed before entry
    x_le, z_le, tl_le = T_all[seen], z_all[seen].reshape(-1, 1), entry_all[seen]
    t_le = np.column_stack([tl_le, np.full(seen.sum(), np.inf)])

    naive = WeibullPH.fit(x=x_le, Z=z_le)
    trunc = WeibullPH.fit(x=x_le, Z=z_le, t=t_le)
    print('truth (alpha, beta, beta_0) : [10.    1.5   0.8]')
    print('WeibullPH ignoring entry    :', naive.params.round(3))
    print('WeibullPH with truncation   :', trunc.params.round(3))
    print('Cox ignoring / with tl      : %.3f / %.3f' % (
        CoxPH.fit(x=x_le, Z=z_le).beta[0],
        CoxPH.fit(x=x_le, Z=z_le, tl=tl_le).beta[0]))

Ignoring the entry ages badly distorts the *baseline* — the fitted life is too
long and the wear-out too steep, because the sample has been filtered towards
survivors — while the coefficient is affected less here because entry age is
unrelated to the covariate. If entry were related to the covariate the
coefficient would be biased too. Only left truncation is available for Cox; right or
interval truncation cannot be expressed in the forward partial likelihood and is
rejected, so use a parametric family (``t=[tl, tr]``) for those.
``CoxPH.fit_from_df`` takes the entry ages as a column, ``tl_col``:

.. jupyter-execute::

    import pandas as pd

    entry_df = pd.DataFrame({'age': x_le, 'z': z_le[:, 0], 'entry': tl_le})
    cox_df = CoxPH.fit_from_df(entry_df, x_col='age', Z_cols='z',
                               tl_col='entry', method='breslow')
    print('Cox with tl_col : %.3f' % cox_df.beta[0])

Fitting from a DataFrame: formulas and categorical covariates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every family has ``fit_from_df``, which names the columns of a pandas
DataFrame instead of passing arrays: ``x_col``, ``c_col``, ``n_col`` and the
covariates either as ``Z_cols`` (a list of numeric columns) or as a
``formula`` (a `formulaic <https://matthewwardrop.github.io/formulaic/>`__
formula such as ``"age + site"`` or ``"age * site"``). The fitted model
remembers its ``feature_names`` — and the formula's encoding — so it can
predict directly from a DataFrame of raw covariates. Beyond those, the
parametric families take ``tl_col`` / ``tr_col`` (truncation) and ``init`` /
``fixed``; ``CoxPH.fit_from_df`` takes ``tl_col`` (delayed entry), ``method``
and ``strata_col``; and the
frailty fitter requires a ``group_col``. There is a single time column, so
interval-censored data (two time columns) go through ``fit``.

A **categorical** covariate (a string or categorical column) is expanded with
reference-level (treatment) coding: its first level is the baseline and each
other level gets a coefficient measuring its effect *relative to that
reference*. One level must be left out because the baseline hazard (or
baseline distribution) already plays the role of the intercept — a column for
every level would sum to one and be perfectly collinear with it, leaving the
coefficients non-identified. Here three sites have different risks:

.. jupyter-execute::

    import pandas as pd

    rng = np.random.default_rng(8)
    n_df = 300
    site = rng.choice(['A', 'B', 'C'], n_df)
    age = rng.uniform(20, 60, n_df)
    site_effect = np.select([site == 'A', site == 'B', site == 'C'],
                            [0.0, 0.5, -0.5])            # log-HR against site A
    T_df = 50 * rng.weibull(2.0, n_df) * np.exp(-(0.03 * (age - 40)
                                                  + site_effect) / 2.0)
    cens_df = rng.uniform(20, 90, n_df)
    patients = pd.DataFrame({'time': np.minimum(T_df, cens_df),
                             'censored': (T_df > cens_df).astype(int),
                             'age': age, 'site': site})

    cox_df = CoxPH.fit_from_df(patients, x_col='time', c_col='censored',
                               formula='age + site')
    for name, b in zip(cox_df.feature_names, cox_df.beta):
        print(f'{name:10s} {b:6.3f}')

``site[T.B]`` and ``site[T.C]`` are the log hazard ratios of sites B and C
against site A (true values 0.5 and -0.5), and ``age`` the log hazard ratio per
year (true 0.03). The same call works for the parametric families, and the
fitted model predicts from a DataFrame of raw covariates — the new rows are
encoded exactly as at fit time:

.. jupyter-execute::

    weib_df = WeibullPH.fit_from_df(patients, x_col='time', c_col='censored',
                                    formula='age + site')
    new = pd.DataFrame({'age': [40, 40, 40], 'site': ['A', 'B', 'C']})
    print(weib_df.feature_names)
    print('S(40) at age 40, sites A, B, C:',
          weib_df.sf(np.full(3, 40.0), new).round(3))

A formula beginning with ``0 +`` asks for the full one-hot coding instead; with
a baseline distribution in the model that brings back the collinearity above,
so it is rarely what you want. Data-dependent transforms inside a formula
(``scale(x)``, ``center(x)``) fit, but such a model cannot be serialised (see
`Saving and loading a fitted model`_).


Time-Varying Covariates
~~~~~~~~~~~~~~~~~~~~~~~~~

Sometimes a covariate changes *during* a subject's follow-up — a dose is
increased, a treatment begins, a machine is moved to a harsher environment.
Cox handles this through the **counting-process (start-stop) format**: each
subject contributes one row per interval :math:`(\text{start}, \text{stop}]`
on which its covariates are constant, and ``c`` is 0 only on the interval
that ends at the subject's failure. Because each interval is exactly a
delayed-entry (left-truncated) observation, the partial likelihood fits this
format directly — splitting a subject into intervals with the same covariates
leaves the fit unchanged.

Use :meth:`CoxPH.fit_tvc <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit_tvc>` (arrays) or :meth:`CoxPH.fit_tvc_from_df <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit_tvc_from_df>` (a
start-stop ``DataFrame``). The interval bounds follow surpyval's ``xl`` / ``xr``
naming and the status column ``c`` follows surpyval's censoring convention —
``c = 0`` for the terminal event, ``c = 1`` for a right-censored interval end
(a covariate change or administrative end). A subject may have at most one
``c = 0`` row, it must be its last interval, and its intervals must not overlap.
In the example below a covariate
``stress`` switches from 0 to 1 at a random time for each unit and genuinely
raises the hazard once it turns on; units that fail before the switch
contribute a single interval, those that survive it contribute two:

.. jupyter-execute::

    from surpyval import CoxPH
    from surpyval.univariate.regression import StepSchedule

    rng = np.random.default_rng(0)
    n, lam, beta = 2000, 0.5, 1.0
    switch = rng.uniform(0.3, 1.5, size=n)
    t_low = rng.exponential(1 / lam, size=n)
    t_high = switch + rng.exponential(1 / (lam * np.exp(beta)), size=n)
    T = np.where(t_low > switch, t_high, t_low)

    rows = []
    for i in range(n):
        if T[i] <= switch[i]:                    # failed before the switch
            rows.append((i, 0.0, T[i], 0, 0.0))     # c=0: event on one interval
        else:                                    # survived it: two intervals
            rows.append((i, 0.0, switch[i], 1, 0.0))  # c=1: covariate change
            rows.append((i, switch[i], T[i], 0, 1.0)) # c=0: terminal event
    df = pd.DataFrame(rows, columns=['id', 'xl', 'xr', 'c', 'stress'])

    model = CoxPH.fit_tvc_from_df(
        df, id_col='id', xl_col='xl', xr_col='xr', c_col='c', Z_cols='stress',
    )
    model

The fitted coefficient recovers the simulated log-hazard-ratio of the stress
(:math:`\beta \approx 1`) — a plain Cox fit that ignored the timing of the
switch could not.

Writing intervals by hand is error-prone. A covariate *timeline* — one row per
covariate change per subject, each value holding until the subject's next row,
the first row's time being the entry and the last row carrying the exit time
and status — can be given instead with :meth:`CoxPH.fit_tvc_timeline <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit_tvc_timeline>` or
:meth:`CoxPH.fit_tvc_timeline_from_df <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit_tvc_timeline_from_df>`. The covariate on a subject's last row
is ignored, as is ``c`` on every row but the last. It is expanded to the same
intervals, so the fit is identical:

.. jupyter-execute::

    timeline = []
    for i in range(n):
        timeline.append((i, 0.0, 0.0, 1))                 # enters with stress 0
        if T[i] > switch[i]:
            timeline.append((i, switch[i], 1.0, 1))       # stress turns on
        timeline.append((i, T[i], np.nan, 0))             # fails (Z ignored)
    tl_df = pd.DataFrame(timeline, columns=['id', 'time', 'stress', 'c'])
    print(tl_df.head(5))

    model_tl = CoxPH.fit_tvc_timeline_from_df(
        tl_df, id_col='id', time_col='time', Z_cols='stress', c_col='c',
    )
    print('same fit:', np.allclose(model_tl.beta, model.beta))

Because survival now depends on the *whole* covariate path, evaluate it with
``sf_tvc``, describing the path as a
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` (or
``(xl, Z)`` arrays). This is the same ``sf_tvc`` interface every regression
family exposes (see `Time-varying covariates across families`_); with a single
constant segment it reduces exactly to ``sf``. Here a unit stressed from
``t = 1`` onward has visibly lower survival than one never stressed:

.. jupyter-execute::

    t = np.linspace(0.01, 3.0, 100)
    never = StepSchedule.constant([0.0])
    late = StepSchedule.from_changepoints([0.0, 1.0], [[0.0], [1.0]])
    plt.plot(t, model.sf_tvc(t, never), label='never stressed')
    plt.plot(t, model.sf_tvc(t, late), label='stressed after t=1')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('S(t)')
    plt.show()

The older interval-oriented
:meth:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel.predict_tvc`
— which returns the survival and cumulative hazard at the baseline jump times
along a subject's ``(xl, xr]`` intervals — remains available and agrees with
``sf_tvc`` exactly.

.. note::

   **Predicting along a future covariate path.** Some packages (lifelines, for
   one) deliberately *refuse* to produce a survival curve from a
   time-varying-covariate Cox model, reasoning that a subject's future
   covariate values are unknown: you cannot know :math:`Z(u)` for :math:`u` up
   to a future time :math:`t`, so an *observed* subject's survival past its
   last record is undefined. SurPyval takes a different view because ``sf_tvc``
   answers a different question. You *supply* the covariate path as a plan or a
   hypothesis — mission phases, a periodic duty cycle, a scheduled load
   increase — and given that stated :math:`Z(\cdot)` the survival
   :math:`S(t \mid Z(\cdot))` is exactly defined. This is scenario / what-if
   evaluation under an assumed trajectory, not a claim to know the future; the
   answer is only ever as good as the covariate plan you feed it. When the
   future schedule is genuinely unknown, that is a reason not to specify one —
   not a reason for the model to refuse an otherwise well-posed question.

Delayed entry is also supported here: a subject whose first interval starts
after 0 simply enters the risk sets late, and gaps between a subject's
intervals are allowed (it is not at risk in the gap). The cluster-robust
standard errors of a start-stop fit (below) cluster the rows by subject
automatically.


Checking the proportional-hazards assumption
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A Cox fit is only trustworthy if its central assumption holds: that each
covariate multiplies the baseline hazard by a *constant* factor over time. When
a coefficient is really drifting with time — a treatment that helps early but
not late, say — the single number Cox reports is a time-average that can hide
the effect entirely.

The standard check is the **Grambsch-Therneau test**, built on the scaled
Schoenfeld residuals. A fitted model exposes it through
:meth:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel.check_ph`.
It returns a joint ``global`` test and a ``per_covariate`` breakdown; a *small*
``p``-value is evidence *against* proportional hazards. We fit the tires model
with :meth:`CoxPH.fit_from_df <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit_from_df>` so the report carries the covariate names:

.. jupyter-execute::

    cols = ['Wedge gauge', 'Interbelt gauge', 'Peel force',
            'Wedge gauge×peel force']
    model = CoxPH.fit_from_df(tires, x_col='Survival', Z_cols=cols,
                              c_col='Censoring')

    ph = model.check_ph()
    print('global p-value:', round(ph['global']['p_value'], 3))
    for row in ph['per_covariate']:
        print(f"  {row['covariate']:24s} p = {row['p_value']:.3f}")

Here every ``p``-value is large, so there is no evidence against proportional
hazards — the Cox coefficients can be read as constant hazard ratios. (With 11
failures the test has little power, so "no evidence" is not strong evidence of
proportionality either.) The statistics match R's ``cox.zph`` and lifelines,
including under Efron ties.

To see what a violation looks like, simulate a covariate whose effect
*reverses*: exposed units have three times the baseline hazard before
:math:`t = 0.5` and a third of it afterwards. The Cox coefficient averages the
two into something unremarkable, but the test is emphatic, and the scaled
Schoenfeld residuals — which estimate :math:`\beta(t)` at each failure — show
the effect falling over time:

.. jupyter-execute::

    rng = np.random.default_rng(11)
    z_rev = rng.binomial(1, 0.5, 400).astype(float)
    e = rng.exponential(size=400)       # unit-exponential "cumulative hazard"
    # exposed: H(t) = 3t before t = 0.5, then 1.5 + (t - 0.5) / 3
    t_exposed = np.where(e < 1.5, e / 3, 0.5 + 3 * (e - 1.5))
    t_rev = np.where(z_rev == 1, t_exposed, e)
    rev = pd.DataFrame({'x': np.minimum(t_rev, 3.0),
                        'c': (t_rev > 3.0).astype(int), 'z': z_rev})

    m_rev = CoxPH.fit_from_df(rev, x_col='x', Z_cols='z', c_col='c')
    print('averaged beta :', m_rev.beta.round(3))
    for transform in ['km', 'rank', 'identity', 'log']:
        p = m_rev.check_ph(transform=transform)['global']['p_value']
        print(f'check_ph(transform={transform!r:10s}) p = {p:.1e}')

    scaled = m_rev.compute_residuals('scaled_schoenfeld')[:, 0]
    event_times = rev['x'][rev['c'] == 0].to_numpy()   # same order as residuals
    plt.plot(event_times, scaled, '.', alpha=0.4)
    plt.axhline(m_rev.beta[0], color='k', label='fitted constant beta')
    plt.xlabel('failure time'); plt.ylabel('scaled Schoenfeld residual')
    plt.legend()
    plt.show()

The residuals sit high early and low late, straddling the constant fit. The
``transform`` argument chooses the function of time the residuals are tested
against: ``"km"`` (the default, :math:`1 -` Kaplan-Meier) spreads the failures
evenly and is the usual choice; ``"rank"``, ``"identity"`` and ``"log"`` are the
alternatives from ``cox.zph``. A violation like this one calls for
stratification (below), a time-varying covariate, or a different family.

The residuals underlying the test (and several others) are available directly
through
:meth:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel.compute_residuals`,
with ``kind`` one of ``"schoenfeld"``, ``"scaled_schoenfeld"``,
``"martingale"``, ``"deviance"``, ``"score"`` or ``"dfbeta"``. Schoenfeld
residuals come one row per failure, in the order the failures appear in the
fitted data (as in the plot above); the others one row per observation, in
input order (for a start-stop fit, in the fit's internal order, sorted by
subject and entry time). They follow the tie method of a Breslow or Efron
fit; after an ``'exact'`` or ``'kalbfleisch-prentice'`` fit the Breslow forms
are used. Martingale
residuals plotted against a covariate reveal non-linear functional form;
deviance residuals highlight poorly-predicted individuals; dfbeta residuals
show how far each observation moves each coefficient:

.. jupyter-execute::

    martingale = model.compute_residuals('martingale')
    print('martingale residuals sum to zero:',
          np.isclose(martingale.sum(), 0.0))
    dfbeta = model.compute_residuals('dfbeta')
    most = np.abs(dfbeta).argmax(axis=0)
    print('most influential tire per coefficient:', most)

Schoenfeld, score and martingale residuals all sum to zero at the maximum of
the partial likelihood — a useful sanity check that the fit has converged.


Cluster-robust standard errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model-based standard errors assume every observation is independent. When
the data are *clustered* — several failures from the same machine, repeated
events on the same subject, items drawn in grouped batches — that assumption is
wrong and the naive errors are too small. The **Lin-Wei sandwich** (or
"robust") variance corrects for it, using the dfbeta residuals grouped by
cluster.

Pass a ``cluster`` label per observation to
:meth:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel.robust_summary`
(or ``robust_covariance`` for the matrix);
with no cluster argument each observation is its own cluster (the ordinary
robust variance):

.. jupyter-execute::

    summary = model.robust_summary()
    for name, se, p in zip(summary['covariate'], summary['se'],
                           summary['p_value']):
        print(f"  {name:24s} robust SE = {se:8.3f}   p = {p:.3f}")

To see why clustering matters, imagine every tire had been measured *twice* and
both rows entered the fit as if independent. Ignoring that would understate the
standard errors by exactly :math:`\sqrt{2}`; passing the shared ``cluster``
label recovers the correct value. (We use Breslow ties so that duplicating the
data leaves the coefficients exactly unchanged; Efron's correction treats the
duplicates as extra ties.)

.. jupyter-execute::

    twice = pd.concat([tires, tires], ignore_index=True)
    tire_id = np.tile(np.arange(len(tires)), 2)
    once = CoxPH.fit_from_df(tires, x_col='Survival', Z_cols=cols,
                             c_col='Censoring', method='breslow')
    dup = CoxPH.fit_from_df(twice, x_col='Survival', Z_cols=cols,
                            c_col='Censoring', method='breslow')

    naive_se = lambda m: np.sqrt(np.diag(np.linalg.inv(m.jac(m.beta)[1])))
    print('naive SE, once / twice :', (naive_se(once) / naive_se(dup)).round(3))
    print('robust SE, once        :', once.robust_summary()['se'].round(3))
    print('robust SE, twice, clustered by tire:',
          dup.robust_summary(cluster=tire_id)['se'].round(3))


Stratified Cox models
~~~~~~~~~~~~~~~~~~~~~~~

When proportional hazards fails for a *nuisance* covariate — a study site, a
batch, a device generation you do not want to model explicitly — the standard
remedy is **stratification**: fit a separate baseline hazard for each stratum
while sharing the coefficients :math:`\beta`. Risk sets never cross a stratum
boundary, so the comparison is always within-stratum.

Pass ``strata`` (a label per observation) to :meth:`CoxPH.fit <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit>`, or
``strata_col`` to :meth:`CoxPH.fit_from_df <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit_from_df>`. The example below is deliberately
adversarial: the baseline hazard differs by an order of magnitude across three
sites *and* the covariate is correlated with the site. An ordinary Cox fit is
badly confounded; the stratified fit recovers the true coefficient:

.. jupyter-execute::

    st_rng = np.random.default_rng(0)
    n_st = 600
    site = st_rng.integers(0, 3, n_st)
    Z_st = st_rng.normal(site.astype(float), 1.0).reshape(-1, 1)   # confounded
    baseline = np.array([1.0, 6.0, 30.0])[site]
    x_st = st_rng.exponential(baseline / np.exp(0.8 * Z_st[:, 0]))  # true 0.8
    c_st = (st_rng.random(n_st) < 0.15).astype(int)

    pooled = CoxPH.fit(x=x_st, Z=Z_st, c=c_st)
    stratified = CoxPH.fit(x=x_st, Z=Z_st, c=c_st, strata=site)
    print(f"true beta   = 0.80")
    print(f"pooled      = {pooled.beta[0]:.3f}   (confounded by site)")
    print(f"stratified  = {stratified.beta[0]:.3f}")

Prediction on a stratified model needs a ``stratum`` argument to pick the right
baseline — one of ``stratified.strata_labels``, here the site codes 0, 1 and
2; ``sf``, ``Hf``, ``hf``, ``ff`` and ``df`` all accept it, and refuse to
guess if it is left out:

.. jupyter-execute::

    stratified.sf(x=1.0, Z=[[0.0]], stratum=0)

The residual diagnostics and robust errors above assume a single baseline, so
they are not available on a stratified model; the coefficients and their
model-based ``p``-values are. A stratified model can also not be serialised.
Along a covariate path, ``sf_tvc``, ``Hf_tvc`` and ``predict_tvc`` take the
same ``stratum`` argument.


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
moving. Coefficients are reported in surpyval's accelerated-failure sign, the
same as ``WeibullAFT``: a *positive* coefficient shortens life (it is the
negative of the slope in the equation above).

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

The ``Converged`` line reports whether the iteration reached a fixed point
(``model.converged`` and ``model.n_iter``); the estimator can settle into a
two-point cycle, which surpyval detects and averages, and a fit that has not
converged within ``max_iter`` iterations (default 100, with step tolerance
``tol=1e-5``) warns. The coefficients are ``model.beta`` (also ``model.coef``).
Only observed and right-censored data with positive times are accepted.

Buckley-James has no simple closed-form standard error, so uncertainty comes
from a percentile bootstrap — resampling, refitting, and taking coefficient
percentiles:

.. jupyter-execute::

    model.bootstrap_ci(n_boot=200, seed=1)

Predictions use the fitted residual distribution directly,
:math:`S(t \mid Z) = S_\varepsilon(\log t + \beta' Z)`, so the survival curves
shift with the covariate (``sf``, ``ff`` and ``Hf`` are available; being a
step function, the model has no density or hazard rate). Each call takes a
single covariate row and returns the curve at every time given:

.. jupyter-execute::

    t = np.linspace(1, 60, 200)
    for z in [-1.0, 0.0, 1.0]:
        plt.step(t, model.sf(t, Z=[z]), where='post', label=f'Z = {z:g}')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('S(t)')
    plt.show()

``BuckleyJames.fit_from_df`` accepts ``Z_cols`` or a ``formula``, exactly as the
other families do.


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
    print(model.standard_errors())

The coefficients read as risk differences: a one-unit change in a covariate
shifts the absolute hazard by :math:`\beta` at every time. As in the Cox fit,
higher gauge and peel-force values reduce the hazard (improving life) while the
interaction term counteracts — the same story, told on the additive scale.

The model's ``Hf``, ``sf`` and ``ff`` use the step baseline
:math:`\hat H_0(t) + t\,\beta'Z`. A hazard *rate* needs a smooth baseline, so
``hf`` (and ``df``) kernel-smooth the baseline increments; the ``bandwidth``
argument of ``hf`` controls the smoothing (by default a normal-reference rule on
the event times), and estimates near the ends of the observed range are
attenuated. The additive model has no multiplier, so ``phi()`` is not defined
for it.

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
non-parametrically. Below, a Weibull wear-out hazard has an exposure that adds
0.05 failures per unit time on top of it:

.. jupyter-execute::

    from surpyval import WeibullAH

    rng = np.random.default_rng(4)
    z_ah = rng.binomial(1, 0.5, 500).astype(float)
    # H(x) = (x / 10)^2 + 0.05 z x  -> solve H(x) = E for a unit exponential E
    E = rng.exponential(size=500)
    x_ah = (-0.05 * z_ah + np.sqrt((0.05 * z_ah) ** 2 + 0.04 * E)) / 0.02
    c_ah = (x_ah > 25).astype(int)
    x_ah = np.minimum(x_ah, 25)

    wah = WeibullAH.fit(x=x_ah, Z=z_ah.reshape(-1, 1), c=c_ah)
    print('WeibullAH (alpha, beta, beta_0):', wah.params.round(3))
    print('standard errors                :', wah.standard_errors().round(3))
    ly = AdditiveHazards.fit(x=x_ah, Z=z_ah.reshape(-1, 1), c=c_ah)
    print('Lin-Ying beta_0 = %.3f (se %.3f)' % (ly.beta[0], ly.se[0]))

Both estimators recover the risk difference to within about two standard
errors (0.035 and 0.046 against a true 0.05), and the Weibull baseline
(scale 10, shape 2) is recovered too. The parametric fit is more efficient
when its baseline is right; Lin-Ying makes no assumption about the baseline.

The positivity caveat above bites differently here. The likelihood needs
``log(h)`` at every failure, so the optimiser only accepts parameter values
that keep :math:`h_0(x) + \beta'Z` positive at every observed failure. When
the data would prefer a negative hazard — a strongly protective covariate —
the fit returns the best model that stays positive, pressed against that
boundary (the fitted hazard of the protected units is then close to zero at
their earliest failures, and the baseline is bent to compensate), and warns
that it has done so; it raises only if the optimiser cannot end at a
positive-hazard point. Treat that warning as a verdict on the model, not the
optimiser, and prefer a proportional-hazards model when effects are strongly
protective.


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
the Weibull is a reasonable fit to the baseline. The parameters are listed in
the order ``model.parameter_names()`` gives: the distribution's own parameters
first, then one ``beta_j`` per covariate column.

If none of the pre-built distributions suit your data, the ``PH`` factory creates
a parametric PH model for any surpyval distribution:

.. jupyter-execute::

    from surpyval import LogNormal
    from surpyval import PH

    model = PH(LogNormal).fit(x=x, Z=Z, c=c)
    model

The log-normal is a natural choice when the log of the survival time is expected
to be normally distributed — common in medical and biological data. (A
log-normal *PH* model multiplies a log-normal hazard; it is not the same as the
log-normal *AFT* model below, which is linear regression on log time.)

Fixed parameters, censoring and simulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PH, AFT, PO, AH and accelerated-life fitters accept ``fixed={name: value}``
to hold any parameter —
a distribution parameter or a coefficient — at a known value, for instance a
Weibull shape known from experience with the failure mode. The fixed parameter
is excluded from the covariance (its standard error is zero):

.. jupyter-execute::

    fixed_shape = WeibullPH.fit(x=x, Z=Z, c=c, fixed={'beta': 15})
    print(fixed_shape.parameter_names())
    print(fixed_shape.params.round(3))
    print(fixed_shape.standard_errors().round(3))

The parametric fitters use surpyval's full likelihood, so every observation
type can be mixed in one fit: ``c = -1`` (left censored), ``c = 2`` (interval
censored, with ``x`` given as ``[left, right]`` pairs) and truncation through
``t``. Here the demo data set from the top of the page is re-recorded as an
inspection study — each unit checked every 2 time units, so a failure is only
known to lie in the interval between inspections — and the interval-censored fit
recovers the same parameters as the exact times:

.. jupyter-execute::

    left_edge = np.floor(x_demo / 2) * 2
    x_int = np.column_stack([left_edge, left_edge + 2])
    x_int[left_edge == 0, 0] = 1e-6          # first interval starts at ~0
    c_int = np.full(len(x_demo), 2)
    interval_fit = WeibullPH.fit(x=x_int, Z=Z_demo, c=c_int)
    print('exact times         :', demo.params.round(3))
    print('inspection intervals:', interval_fit.params.round(3))

``phi(Z)`` returns the fitted hazard multiplier :math:`e^{\beta'Z}` (for AFT
it is the acceleration factor, for PO the odds multiplier, for accelerated
life the modelled life; an additive model has none). ``random(size, Z)``
draws lifetimes from the fitted model — useful for simulation studies and for
checking a fit against its own simulated data. It exists for the PH, parametric
AH and accelerated-life families (not AFT or PO). A PH or parametric AH model
returns ``size`` draws for each covariate row, in the order given, together
with the matching covariate rows; an accelerated life model does the same for
each *distinct* stress, in sorted order:

.. jupyter-execute::

    print('hazard multipliers at Z = 0, 1:', demo.phi([[0.0], [1.0]]).round(3))
    np.random.seed(0)
    sim_x, sim_Z = demo.random(5, [[0.0], [1.0]])
    print(sim_x.round(2))
    print(sim_Z.ravel())

A custom covariate function
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The log-linear :math:`e^{\beta'Z}` is a choice, not a requirement. Some fields
use other forms — radiation epidemiology, for example, often models an
*excess relative risk* that grows linearly with dose,
:math:`\phi(z) = 1 + \beta z`, so that :math:`\beta` is the extra risk per unit
dose. ``ProportionalHazardsFitter`` builds a PH fitter around any
:math:`\phi(Z, *\text{params})` written with ``autograd.numpy``. Its arguments
are a name, the baseline distribution, ``phi``, a display name for it, the
parameter bounds and the parameter-name map (each either fixed or a function of
``Z``), and optionally a starting value. The bounds are how you keep
:math:`\phi` positive — here :math:`\beta > 0`:

.. jupyter-execute::

    import autograd.numpy as anp
    from surpyval import ProportionalHazardsFitter, Weibull

    def linear_rr(Z, *params):                  # phi(Z) = 1 + beta'Z
        return 1.0 + anp.dot(Z, anp.array(params))

    WeibullLinearRR = ProportionalHazardsFitter(
        'WeibullLinearRR', Weibull, linear_rr, "Linear [1 + beta'Z]",
        phi_bounds=lambda Z: ((0, None),) * Z.shape[1],
        phi_param_map=lambda Z: {f'beta_{i}': i for i in range(Z.shape[1])},
        phi_init=lambda Z: np.full(Z.shape[1], 0.5),
    )

    rng = np.random.default_rng(0)
    dose = rng.uniform(0, 4, 400)
    # Weibull(10, 2) baseline; each unit of dose adds 50% to the hazard
    x_rr = 10 * (-np.log(rng.uniform(size=400)) / (1 + 0.5 * dose)) ** (1 / 2)
    rr = WeibullLinearRR.fit(x=x_rr, Z=dose)
    print('alpha, beta, beta_0:', rr.params.round(3))
    print('standard errors    :', rr.standard_errors().round(3))

The excess relative risk per unit dose, 0.60 (standard error 0.17), is within
one standard error of the true 0.5. Everything else — predictions, bounds,
``fit_from_df`` — works as for the pre-built models, but a custom covariate
function cannot be rebuilt from a name, so such a model cannot be serialised.


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

An important practical note: the **Weibull** (and its special case the
Exponential) is the one distribution for which AFT and PH are the same model.
A Weibull-AFT and a Weibull-PH fit to the same data have identical likelihoods,
and their coefficients differ only by the Weibull shape,
:math:`\beta_{PH} = \text{shape} \times \beta_{AFT}`. For every other
distribution — log-normal included — AFT and PH are genuinely distinct models.

.. jupyter-execute::

    from surpyval import WeibullAFT

    model = WeibullAFT.fit(x=x, Z=Z, c=c)
    model

Checking the Weibull equivalence numerically against the ``WeibullPH`` fit:

.. jupyter-execute::

    ph_fit = WeibullPH.fit(x=x, Z=Z, c=c)
    shape = model.params[1]
    print('shape x AFT coefficients:', (shape * model.params[2:]).round(3))
    print('PH coefficients         :', ph_fit.params[2:].round(3))
    print('neg log-likelihoods     :', round(model.neg_ll(), 4),
          round(ph_fit.neg_ll(), 4))

The ``AFT`` factory works with any distribution. Log-Normal AFT is a
particularly common choice — it corresponds to ordinary linear regression on
:math:`\log T` with censored observations, and is the parametric counterpart
of the Buckley-James model above:

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
    plt.show()

Compare this with the Cox PH survival curves above. Under PH the curves are
powers of one another, :math:`S(x \mid Z) = S_0(x)^{\phi(Z)}`; under AFT they
are the same curve shifted along the log-time axis, so on a log-time plot they
are parallel. Neither family lets two curves cross. (For these Weibull fits the
two descriptions coincide, as shown above.)


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

Because :math:`e^{\beta'Z}` multiplies the odds of *surviving*, a positive
coefficient here means a *longer* life — the opposite of the PH and AFT sign
convention. To compare with a PH fit, negate the PO coefficients.

The key difference from PH becomes clear at long follow-up: as :math:`x \to
\infty`, :math:`S_0(x) \to 0`, and the ratio :math:`F_0 + \phi S_0 \to 1`, so
the covariate effect *fades away*. Everyone eventually fails, and the PO model
respects that by letting the hazard ratio converge to 1 over time. Under PH,
the hazard ratio is constant forever — a stronger and often unrealistic
assumption for long studies.

PO is the natural companion to the Logistic and Log-Logistic distributions:
with a logistic baseline the odds multiplier is a shift of the location, and
with a log-logistic baseline it is a rescaling of time, so the model is also an
AFT model (see the :doc:`regression analysis` page).

.. jupyter-execute::

    from surpyval import LogisticPO

    model = LogisticPO.fit(x=x, Z=Z, c=c)
    model

The tire coefficients have the opposite signs to the PH and AFT fits — the same
conclusions, in the survival-odds convention. The ``PO`` factory accepts any
distribution:

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval import PO

    model = PO(Weibull).fit(x=x, Z=Z, c=c)
    model

The Weibull baseline tells the same story as the logistic one: every
coefficient has the opposite sign to the PH fit, as the survival-odds
convention requires. With 11 failures and four covariates, though, none of
these fits is well determined, and `Model Selection`_ below shows the data
cannot separate the PO description from the PH/AFT one.

The fading effect is easiest to see on data simulated from a PO model. Below,
a log-logistic baseline has its survival odds multiplied by :math:`e^{1}` for
exposed units. The hazard ratio of exposed to unexposed units starts near
:math:`e^{-1} \approx 0.37` and climbs towards 1:

.. jupyter-execute::

    from surpyval import LogLogistic

    rng = np.random.default_rng(2)
    z_po = rng.binomial(1, 0.5, 400).astype(float)
    U = rng.uniform(size=400)
    # invert F(x|z) = U for survival odds (x/10)^-3 * e^z
    x_po = 10 * (np.exp(z_po) * U / (1 - U)) ** (1 / 3)
    po = PO(LogLogistic).fit(x=x_po, Z=z_po.reshape(-1, 1))
    print('alpha, beta, beta_0:', po.params.round(3))

    times = np.array([1.0, 5.0, 10.0, 20.0, 40.0])
    print('hazard ratio at', times, ':',
          (po.hf(times, [1.0]) / po.hf(times, [0.0])).round(3))

A practical rule of thumb: if the Kaplan-Meier curves for different covariate
groups converge at long times (rather than remaining parallel on the log-hazard
scale), PO is likely a better fit than PH. Proportional odds has no
time-varying-covariate support and no ``random``.


Confidence Bounds
-----------------

A point estimate is only half the story. The parametric regression models (PH,
AFT, PO, AH and AL) carry the full parameter covariance — the inverse of the
numerical Hessian of the negative log-likelihood — so every coefficient and every
predicted curve comes with an interval. After a fit, the parameter covariance
(``covariance()``) and standard errors are available directly:

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
    plt.show()

``cb`` takes a single covariate vector ``Z`` and accepts ``on='ff'``,
``'Hf'``, ``'hf'`` or ``'df'``, one- or two-sided bounds via ``bound=``, and
any ``alpha_ci``. A one-sided bound is what a reliability demonstration
usually needs — for example the lower 90% bound on the reliability at time 5:

.. jupyter-execute::

    m_cb.cb([5.0], Z=[0.5], on='sf', bound='lower', alpha_ci=0.1)

The convenience method ``model.plot()`` draws the fitted survival at the mean
covariate, with this band, against a non-parametric estimate of the pooled
data (the exponentiated Nelson-Aalen estimate) — a quick visual check, though
the pooled curve ignores the covariates. The bounds here are Wald /
delta-method bounds; the likelihood-ratio bounds available for univariate
parametric fits are not implemented for the regression models.

The other families quantify uncertainty their own way: Cox through the
information matrix (``p_values``, and ``jac`` as shown earlier) and the robust
sandwich; Lin-Ying through its sandwich ``standard_errors()``; Buckley-James by
``bootstrap_ci``; and the frailty model (below) through ``standard_errors()``
and ``param_cb``.


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
same at all stresses, just faster or slower). Which parameter is the "life"
for each distribution — and how the Exponential, Log-Normal and Gamma convert
between a life and their own parameter — is tabulated on the
:doc:`regression analysis` page. Weibull, Exponential, Normal, Log-Normal,
Gamma, Gumbel and Logistic are supported.

Available life models
~~~~~~~~~~~~~~~~~~~~~~

The choice of life model depends on the physical failure mechanism. The
letters in each formula are the parameter names the fitted model reports, and
:math:`Z_1, Z_2` are the two columns of ``Z`` for the two-stress models:

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
     - :math:`Z^{-1} e^{-(b - a/Z)}`
     - Temperature, from reaction-rate (transition-state) theory: Arrhenius
       with a :math:`1/Z` pre-factor
   * - ``InversePower``
     - :math:`1 / (a \cdot Z^n)`
     - Voltage, electrical field, mechanical fatigue
   * - ``Power``
     - :math:`a \cdot Z^n`
     - The same law written directly as a life; with :math:`n < 0` life
       falls as stress rises
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
     - :math:`Z e^{c - a/Z}`, the reciprocal of Eyring
     - Inverse Eyring relationship
   * - ``InverseExponential``
     - :math:`1 / (b \cdot e^{a/Z})`, the reciprocal of Arrhenius
     - Inverse Arrhenius relationship

A note on units: the stress variable :math:`Z` for Arrhenius and Eyring should
be in Kelvin (absolute temperature), not Celsius. The accelerated life fitter
takes the same ``c``, ``n``, ``t``, ``init`` and ``fixed`` arguments as the
other parametric families (the life-model parameters can be held with
``fixed`` too), and has a ``fit_from_df``.

Using the factory
~~~~~~~~~~~~~~~~~

The example simulates a classic temperature test: twenty units at each of
85 °C, 105 °C and 125 °C, with an activation energy of 0.7 eV, and a test that
is stopped at 6,000 hours so that most of the coolest units are still running
(right censored):

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval import AcceleratedLife, Power, ExponentialLifeModel

    # Discrete stress levels — three temperatures in Kelvin
    stress = np.repeat([358., 378., 398.], 20)   # 85°C, 105°C, 125°C
    Ea, k = 0.7, 8.617e-5   # activation energy eV, Boltzmann constant eV/K
    rng = np.random.default_rng(42)
    true_life = 1.4e-6 * np.exp(Ea / (k * stress))   # Arrhenius, in hours
    T_al = true_life * rng.weibull(2.5, stress.size)
    test_end = 6000.0
    x_al = np.minimum(T_al, test_end)
    c_al = (T_al > test_end).astype(int)            # still running at the end
    print('censored at each stress:',
          [int(c_al[stress == s].sum()) for s in (358., 378., 398.)])

    # Weibull + Arrhenius (ExponentialLifeModel) — the most common ALT model
    model_arr = AcceleratedLife(Weibull, ExponentialLifeModel).fit(
        x_al, Z=stress, c=c_al)
    model_arr

Notice that the Weibull shape parameter :math:`\beta` is estimated globally —
it is the same for all stress levels — while the scale parameter :math:`\alpha`
varies with stress via the Arrhenius relationship. This is the key assumption of
ALT: the failure mechanism does not change with stress, only the rate. The
``alpha: 1.0`` in the report is a placeholder: the life parameter is replaced
by :math:`\phi(Z)`, so it is held fixed and carries no information (it is listed
in ``model_arr.fixed``, and is not counted as a parameter in the AIC). The Arrhenius parameter ``a`` is
:math:`E_a / k_B`, so the fit estimates the activation energy directly:

.. jupyter-execute::

    print('activation energy (eV)  : %.3f' % (model_arr.params[2] * k))
    print('95% CI on a, in eV      :', (model_arr.param_cb('a') * k).round(3))

.. jupyter-execute::

    # Power law — a common choice for voltage or load acceleration
    model_power = AcceleratedLife(Weibull, Power).fit(x_al, Z=stress, c=c_al)
    model_power

Over a narrow range of temperatures a steep power law mimics Arrhenius (hence
the extreme exponent), and the two fit the test data about equally well. They
part company as soon as they extrapolate — here by about 20% at a use
temperature only 30 °C below the coolest test — which is why the life model
should come from the physics rather than from the fit statistics alone:

.. jupyter-execute::

    use = 328.0                                  # 55°C use condition
    for name, m in [('Arrhenius', model_arr), ('Power', model_power)]:
        print(f'{name:10s} AIC = {m.aic():7.2f}'
              f'   characteristic life at 55°C = {m.phi([use])[0]:7.0f} h')
    print('true characteristic life at 55°C = %7.0f h'
          % (1.4e-6 * np.exp(Ea / (k * use))))

Both models under-predict the true use life. The fitted activation energy,
0.67 eV against a true 0.70 eV, is well within its confidence interval, but an
error in the slope of the stress-life line is multiplied by the distance of the
extrapolation — a reminder to report the uncertainty of an extrapolated life
(for example ``model_arr.cb`` at the use stress), not just its point estimate.
The power law, whose form is wrong for this mechanism, is further off still.

To use the fitted model for extrapolation, pass the operating stress to any
of the survival functions:

.. jupyter-execute::

    # Predict life at the 55°C use condition, outside the tested range
    x_pred = np.linspace(0, 200000, 500)
    Z_use = np.array([[use]])   # operating condition

    band = model_arr.cb(x_pred, Z=Z_use, on='sf')        # 95% delta-method band
    plt.plot(x_pred, model_arr.sf(x_pred, Z=Z_use), label='Predicted at 55°C (328K)')
    plt.fill_between(x_pred, band[:, 0], band[:, 1], alpha=0.2, label='95% band')
    plt.plot(x_pred, Weibull.sf(x_pred, 1.4e-6 * np.exp(Ea / (k * use)), 2.5),
             'k--', label='true reliability')
    plt.xlabel('Time (hours)')
    plt.ylabel('Reliability')
    plt.legend()
    plt.title('Extrapolated life at operating conditions')
    plt.show()

The band is wide — sixty units tested for at most 6,000 hours say only so much
about lives of tens of thousands of hours — and it is the band, which here
contains the true curve, rather than the point estimate that should drive a
decision.

Two stresses at once
~~~~~~~~~~~~~~~~~~~~

The dual life models take a two-column ``Z``, one column per stress. A
common design tests every combination of two temperatures and two voltages;
``PowerExponential`` then combines an Arrhenius term in temperature with a
power law in voltage, :math:`c\, e^{a/Z_1} Z_2^{n}`:

.. jupyter-execute::

    from surpyval import PowerExponential

    rng = np.random.default_rng(0)
    temp = np.repeat([358., 378., 358., 378.], 25)       # kelvin
    volts = np.repeat([10., 10., 20., 20.], 25)
    true_life2 = 2e-4 * np.exp(Ea / (k * temp)) * volts ** -1.5
    x_2s = true_life2 * rng.weibull(2.5, 100)

    model_2s = AcceleratedLife(Weibull, PowerExponential).fit(
        x_2s, Z=np.column_stack([temp, volts]))
    for name, value in zip(model_2s.parameter_names(), model_2s.params):
        print(f'{name:5s} = {value:.4g}')
    print('activation energy (eV): %.3f' % (model_2s.params[3] * k))

The fit separates the two effects — an activation energy of 0.67 eV against
the true 0.7, and a voltage exponent ``n`` of -1.44 against the true -1.5 —
because the design varies each stress while the other is held fixed. Had voltage been raised only
together with temperature, the two columns would be collinear and no fit
could tell their effects apart.

Creating a custom life model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If none of the built-in life models matches your failure physics, you can define
your own by subclassing ``LifeModel``. Its constructor takes a ``name``, a
``phi_param_map`` naming the parameters in order, and their ``phi_bounds``. The
two methods you must implement are:

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

    model_custom = AcceleratedLife(Weibull, InverseSquareRoot()).fit(
        x_al, Z=stress, c=c_al)
    model_custom

This life model is deliberately wrong for Arrhenius data — life cannot fall
steeply enough with temperature — and the fit shows it: to reconcile the
three stress levels it inflates the scatter within each (the Weibull shape
drops well below the true 2.5), and its AIC is far worse:

.. jupyter-execute::

    print('AIC, Arrhenius       : %.1f' % model_arr.aic())
    print('AIC, InverseSquareRoot: %.1f' % model_custom.aic())


.. _tvc-parametric:

Time-varying covariates across families
---------------------------------------

The counting-process machinery shown for Cox is not unique to it. Wherever the
cumulative hazard is *additive over disjoint time intervals*, a
time-varying-covariate subject factorises exactly into one left-truncated
observation per constant-covariate interval — so the parametric proportional
hazards (``PH``) and additive hazards (``AH``) families fit start-stop data
with the same ``fit_tvc`` / ``fit_tvc_timeline`` (and ``_from_df``) methods and
the same ``i`` / ``xl`` / ``xr`` / ``c`` convention as Cox. (Keyword arguments
such as ``fixed=`` and ``init=`` are passed through to the ordinary ``fit``.)
Fitting the truncated likelihood takes a few seconds for these 2,000 subjects,
noticeably longer than Cox:

.. jupyter-execute::

    from surpyval import WeibullPH

    ph = WeibullPH.fit_tvc_from_df(
        df, id_col='id', xl_col='xl', xr_col='xr', c_col='c', Z_cols='stress',
    )
    ph.params

The data were simulated with an exponential baseline of rate 0.5 (a Weibull with
scale 2 and shape 1) and :math:`\beta = 1`, which the fit recovers.

**Accelerated failure time** also fits start-stop data through the same
``fit_tvc`` interface. AFT rescales the *time axis* rather than the hazard, so a
subject's likelihood depends on its accumulated *accelerated age*
:math:`\psi = \sum e^{\beta'z}\,(b - a)` across intervals and cannot be
reshaped into independent left-truncated rows the way PH/AH can; ``WeibullAFT``
fits it with a dedicated accumulated-age likelihood instead, but the call is
identical (it accepts ``fixed=`` but not ``init=``):

.. jupyter-execute::

    from surpyval import WeibullAFT

    aft = WeibullAFT.fit_tvc_from_df(
        df, id_col='id', xl_col='xl', xr_col='xr', c_col='c', Z_cols='stress',
    )
    aft.params

The true baseline is a Weibull of shape 1, for which accelerated failure time
and proportional hazards are the same model with
:math:`\beta_{PH} = \text{shape} \times \beta_{AFT}`, so the AFT coefficient
is about 1 as well.

Because the accelerated age is integrated from time zero, the AFT fit needs each
subject's whole covariate history: every subject's first interval must start
at 0 and its intervals must be contiguous. Delayed entry or a gap would mean
ageing at an unknown rate over the unobserved stretch, so instead of guessing,
the fit refuses and points to Cox, which handles both exactly:

.. jupyter-execute::

    late_entry = df.copy()
    late_entry.loc[late_entry.index[0], 'xl'] = 0.1   # subject 0 enters at 0.1
    try:
        WeibullAFT.fit_tvc_from_df(late_entry, id_col='id', xl_col='xl',
                                   xr_col='xr', c_col='c', Z_cols='stress')
    except ValueError as err:
        print(err)

Proportional odds is the one family without time-varying-covariate fitting.

**Evaluating a covariate path.** Every family that has a closed form along a
step path — Cox, parametric ``PH`` and ``AH``, and accelerated failure time
(``AFT``) — exposes the same ``sf_tvc(x, Z, xl=None, given=None)`` (plus the
matching ``Hf_tvc``). Pass either ``(xl, Z)`` arrays or a
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`, and
``given=`` for conditional survival :math:`S(x \mid \text{survived to } g)`.
Proportional and additive hazards accumulate a cumulative hazard over the
segments; AFT instead accumulates an *accelerated age*
:math:`\psi(x) = \sum e^{\beta'z}\,(b - a)` and evaluates the baseline once at
:math:`\psi`. Proportional odds has no such closed form yet, and raises.

Describing the covariate path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` is a
piecewise-constant covariate path. The covariate
*must* be a step function — that is precisely what keeps each family's
cumulative form exact — so a schedule can be built structurally (a constant,
change-points, explicit intervals, or a repeating duty cycle) or from a
step-valued expression in ``t``. Every schedule's covariate rows have one
column per covariate, so a multivariate path is just a wider ``Z``:

.. jupyter-execute::

    # structural
    StepSchedule.constant([1.0])                                    # never changes
    StepSchedule.from_changepoints([0.0, 500.0], [[0.0], [1.0]])   # a switch
    StepSchedule.from_intervals([0, 1, 2], [1, 2, 3], [[0.], [1.], [0.]])
    StepSchedule.cyclic([0, 8], [[1.0], [0.0]], period=24)         # 8-on/16-off

    # expression in t, materialised to a horizon
    duty = StepSchedule.from_expression("1.0 if t % 24 < 8 else 0.0",
                                        horizon=96)
    trend = StepSchedule.from_expression("0.3 * 2 ** floor(t / 1000)",
                                         horizon=5000)
    trend

An expression is proved piecewise-constant *statically*, from its syntax tree,
before it is ever evaluated: ``t`` may reach the value only through a quantizer
(``floor``, ``ceil``, ``round``, ``trunc``, ``//``) or a comparison. A genuinely continuous covariate
(``0.3 + 1e-4 * t``, ``sin(t)``) is rejected with ``StepValuedError`` rather
than silently returning a wrong answer — a covariate that varies continuously
would break the exactness of the segment sum. (surpyval owns only this
step-valued guarantee; sandboxing an *untrusted* expression string is the
calling application's responsibility.) The expression is sampled on a grid of
spacing ``resolution`` (default 1) up to ``horizon``, so the resolution must be
no coarser than the narrowest step; beyond the horizon the last value is held.
For several covariates pass a list of expressions, one per covariate
(``StepSchedule.from_expression(["...", "..."], horizon=...)``), and ``t0``
starts the path somewhere other than 0. The expressions may use ``t``, numbers,
arithmetic, comparisons, ``a if cond else b``, the constants ``pi``, ``e``,
``tau`` and ``inf``, and the functions ``floor``, ``ceil``, ``round``,
``trunc``, ``abs``, ``min`` and ``max``; anything else is refused.

.. jupyter-execute::

    try:
        StepSchedule.from_expression("0.3 + 1e-4 * t", horizon=100)
    except surv.StepValuedError as err:
        print(err)

Evaluating the fitted parametric model along a path is then identical to the
Cox case:

.. jupyter-execute::

    t = np.linspace(0.1, 3.0, 100)
    never = ph.sf_tvc(t, StepSchedule.constant([0.0]))
    late = ph.sf_tvc(t, StepSchedule.from_changepoints([0.0, 1.0],
                                                       [[0.0], [1.0]]))
    plt.plot(t, never, label='never stressed')
    plt.plot(t, late, label='stressed after t=1')
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('S(t)')
    plt.show()

The array form gives the segment start times as ``xl`` and one covariate row per
segment, and ``given=`` conditions on survival to an age along the same path.
Conditional survival is only meaningful at times at or after ``given``:

.. jupyter-execute::

    at = np.array([1.5, 2.5, 3.5])
    pulse = dict(Z=[[0.0], [1.0], [0.0]], xl=[0.0, 1.0, 2.0])   # on for 1 < t < 2
    print('S(t)             :', ph.sf_tvc(at, **pulse).round(3))
    print('S(t | T > 1)     :', ph.sf_tvc(at, **pulse, given=1.0).round(3))
    print('same as a ratio  :', (ph.sf_tvc(at, **pulse)
                                 / ph.sf_tvc([1.0], **pulse)).round(3))


Worked example: forecasting equipment on a duty cycle
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Putting the two halves together — fitting on a time-varying covariate and then
evaluating a *future* path — answers a question that comes up constantly in
reliability but that a semi-parametric time-varying Cox fit (as in lifelines)
cannot: *my equipment runs on a duty cycle; how long will it last, and how much
longer if I change that cycle?* The fit tells you how much the load matters; the
forecast turns that into a survival curve for any schedule you might run next,
including ones the fleet has never yet seen.

Consider a fleet of pumps that alternate week-long **high-load** and
**low-load** shifts. High load wears a pump faster. Pumps enter service in
different weeks, so their cycles are offset — and that phase spread is exactly
what lets the load effect be identified. We observe each pump in start-stop
format, one row per week, with the load on that week as the covariate:

.. note::

   **When a time-varying covariate earns its keep.** The load coefficient is
   estimated from *contrast within the risk set*: at each failure, does the
   pump that failed carry a heavier load than the others still running? If every
   pump ran the identical cycle locked to one calendar — all high-load in the
   same weeks — there is no such contrast, load is perfectly confounded with
   time, and the coefficient is not estimable (a Cox fit goes singular; a
   parametric fit returns a silently biased number). Operational spread across
   the fleet — staggered commissioning, different shift patterns, idle periods —
   is what makes the effect measurable. When your equipment really is all
   operated the same way there is nothing for a time-varying covariate to
   latch onto, and the right tool is an ordinary ``Weibull`` fit to the
   failure times: it characterises reliability under that one
   operating regime (it just cannot forecast a *different* cycle, because the
   data never varied the load).

.. jupyter-execute::

    import pandas as pd

    rng = np.random.default_rng(1)
    n_pumps, horizon = 40, 60
    alpha, shape, load_effect = 30.0, 2.0, 0.9   # ground truth for the demo

    rows = []
    for pump in range(n_pumps):
        phase = pump % 2                    # starts on a high or a low shift
        energy = rng.exponential(1.0)       # latent failure threshold
        H = 0.0
        for week in range(horizon):
            load = 1.0 if (week + phase) % 2 == 0 else 0.0
            dH = (((week + 1) / alpha) ** shape - (week / alpha) ** shape) \
                * np.exp(load_effect * load)
            failed = (H + dH) >= energy
            rows.append((pump, week, week + 1, 0 if failed else 1, load))
            if failed:
                break
            H += dH

    pumps = pd.DataFrame(rows, columns=['pump', 'xl', 'xr', 'c', 'load'])
    pumps.head()

Fitting is the ordinary ``fit_tvc_from_df`` call. A ``WeibullPH`` recovers both
the baseline wear-out (the Weibull ``alpha`` and ``beta``) and the load effect
as a coefficient; ``exp(beta_load)`` is the hazard ratio of a high-load week
against a low-load one:

.. jupyter-execute::

    from surpyval import WeibullPH

    model = WeibullPH.fit_tvc_from_df(
        pumps, id_col='pump', xl_col='xl', xr_col='xr', c_col='c', Z_cols='load',
    )
    print('parameters :', np.round(model.params, 3))
    print('load hazard ratio exp(beta) : %.2f' % np.exp(model.params[-1]))

Now the part that motivates the whole exercise. Because the fit is fully
parametric, ``sf_tvc`` will evaluate the survival curve along *any* step
schedule you hand it — not just paths the pumps actually ran. That makes it a
planning tool: pose each candidate duty cycle as a
:class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule` and
read off the survival it implies. Here we compare the current 50/50 cycle
against an *eased* cycle (one high week in four) and a punishing always-high
regime:

.. jupyter-execute::

    from surpyval.univariate.regression import StepSchedule

    horizon_f = 60
    cycles = {
        'current: 1 week on / 1 week off': '1.0 if t % 2 < 1 else 0.0',
        'eased: 1 week on / 3 weeks off':  '1.0 if t % 4 < 1 else 0.0',
        'always high load':                '1.0',
    }

    t = np.linspace(0.5, horizon_f, 200)
    for label, expr in cycles.items():
        sched = StepSchedule.from_expression(expr, horizon=horizon_f)
        sf = model.sf_tvc(t, sched)
        median = t[np.argmax(sf < 0.5)] if np.any(sf < 0.5) else np.nan
        print('%-34s median life ~ %4.1f weeks' % (label, median))
        plt.plot(t, sf, label=label)

    plt.legend()
    plt.xlabel('Weeks in service')
    plt.ylabel('S(t)')
    plt.title('Forecast pump survival under alternative duty cycles')
    plt.show()

The eased cycle buys a few weeks of median life over the current one, and the
always-high regime costs several — a quantitative answer to "should we throttle
back?" that falls straight out of the fitted model. None of these three curves
required running a pump on that schedule; they are the model's forecast under a
covariate plan you supply. A semi-parametric time-varying Cox model has no
baseline hazard beyond the last observed event time, so it cannot produce a
survival curve into the future at all — which is why this scenario forecasting
needs the parametric ``fit_tvc`` / ``sf_tvc`` pair.


Shared-frailty models
---------------------

When your data come in **groups** — lots from the same supplier, units at the
same site, repeated failures of one repairable machine — the members of a group
tend to fail more alike than units from different groups, because they share
something you did not measure. A **shared-frailty** model captures that with a
random hazard multiplier ``u`` shared within each group, on top of a
proportional-hazards baseline. Fit it with the ``Frailty(distribution)``
factory (or a pre-built instance: ``WeibullFrailty``, ``ExponentialFrailty``,
``LogNormalFrailty``, ``GammaFrailty``), passing a ``groups``
label per observation (see :doc:`regression/frailty`):

.. jupyter-execute::

    from surpyval import WeibullFrailty

    rng = np.random.default_rng(1)
    n_groups, per = 60, 6
    rows_x, rows_c, rows_z, rows_g = [], [], [], []
    for g in range(n_groups):
        u = rng.gamma(1 / 0.6, 0.6)              # group frailty, mean 1
        for _ in range(per):
            z = rng.normal()
            t = 20 * (-np.log(rng.uniform()) / (np.exp(0.8 * z) * u)) ** (1 / 1.8)
            rows_x.append(min(t, 40)); rows_c.append(0 if t <= 40 else 1)
            rows_z.append(z); rows_g.append(g)

    model = WeibullFrailty.fit(
        x=np.array(rows_x), c=np.array(rows_c),
        Z=np.array(rows_z).reshape(-1, 1), groups=np.array(rows_g),
    )
    print(model.summary())
    print("theta 95% CI:", np.round(model.param_cb("theta"), 3))
    print("theta standard error: %.3f" % model.standard_errors()["theta"])

The frailty variance ``theta`` (also ``model.frailty_variance``) quantifies
the between-group spread. Its interval is built on the log scale, so it can
never include zero and says how precisely :math:`\theta` is known rather than
whether it is positive; the evidence of real heterogeneity is an estimate
well clear of zero relative to its standard error — here about three standard
errors. (The data were simulated with
:math:`\theta = 0.6`. A variance of a random effect is hard to pin down — with
60 groups of six, this sample happens to land low, and the interval only just
misses the truth — while the coefficient, true value 0.8, is recovered well.)
The per-group posterior frailties — an empirical-Bayes estimate for each
observed group, shrunk toward 1 — are on ``model.frailties``, keyed by group
label (as a string), and ``model.standard_errors()`` gives the Wald standard
errors of every parameter as a dictionary keyed by name.

Prediction comes in two flavours. The default is **marginal** (population
averaged), the right curve for a *new* unit from an *unknown* group; passing
``group=`` conditions on an observed group's posterior frailty, for another unit
from a group you have already seen, and ``frailty=`` conditions on any frailty
value you supply:

.. jupyter-execute::

    t = np.linspace(1, 40, 100)
    z = np.array([0.0])
    plt.plot(t, model.sf(t, z), 'k-', label='marginal (new group)')
    g0 = model.group_labels[0]
    plt.plot(t, model.sf(t, z, group=g0), 'b--',
             label=f'conditional on group {g0}')
    plt.plot(t, model.sf(t, z, frailty=2.0), 'r:', label='frailty = 2')
    plt.legend(); plt.xlabel('Time'); plt.ylabel('S(t)')
    plt.show()

The coefficient is a *within-group* effect: inside any one group a unit of
:math:`z` multiplies the hazard by :math:`e^{\beta}`. Across the population the
frail groups fail first, so the marginal hazard ratio starts at
:math:`e^{\beta}` and shrinks over time:

.. jupyter-execute::

    times = np.array([1.0, 5.0, 10.0, 20.0, 40.0])
    print('exp(beta)             :', np.exp(model.beta).round(3))
    print('marginal hazard ratio :',
          (model.hf(times, [1.0]) / model.hf(times, [0.0])).round(3))

``fit_from_df`` names the columns instead (``group_col`` for the groups, and
``Z_cols`` or a ``formula`` for the covariates), and the fitted model then
predicts from a DataFrame:

.. jupyter-execute::

    lots = pd.DataFrame({'x': rows_x, 'c': rows_c, 'z': rows_z,
                         'lot': [f'L{g:02d}' for g in rows_g]})
    by_lot = WeibullFrailty.fit_from_df(lots, x_col='x', group_col='lot',
                                        Z_cols='z', c_col='c')
    print(by_lot.feature_names, by_lot.group_labels[:3])
    print(by_lot.sf([10.0], pd.DataFrame({'z': [0.0]})),
          by_lot.sf([10.0], [0.0], group='L00'))

Omit ``Z`` entirely for a pure random-effects survival model (grouped data, no
covariates). Only Gamma frailty is available for now (``Frailty(dist)`` takes
any baseline distribution; ``WeibullFrailty``, ``ExponentialFrailty``,
``LogNormalFrailty`` and ``GammaFrailty`` are pre-built), on observed and
right-censored data, and at least two groups are required. When the data show
little between-group variation the estimate of ``theta`` goes to its boundary
at zero, and the frailty fit then coincides with the ordinary ``WeibullPH`` fit
(the same baseline, coefficients and likelihood). The frailty model has the
same ``neg_ll()``, ``aic()`` and ``bic()`` as the parametric families, counting
``theta`` as one more parameter, so the two fits can be compared directly —
here on grouped data with no frailty at all:

.. jupyter-execute::

    rng = np.random.default_rng(2)
    z_ff = rng.normal(size=300)
    x_ff = 20 * (-np.log(rng.uniform(size=300)) / np.exp(0.8 * z_ff)) ** (1 / 1.8)
    g_ff = np.repeat(np.arange(30), 10)          # 30 groups, but no frailty
    no_frailty = WeibullFrailty.fit(x=x_ff, Z=z_ff.reshape(-1, 1), groups=g_ff)
    ph_ff = WeibullPH.fit(x=x_ff, Z=z_ff.reshape(-1, 1))
    print('theta               : %.1g' % no_frailty.theta)
    print('neg log-likelihood  : frailty %.4f, PH %.4f'
          % (no_frailty.neg_ll(), ph_ff.neg_ll()))
    print('AIC                 : frailty %.2f, PH %.2f'
          % (no_frailty.aic(), ph_ff.aic()))

The likelihoods agree and the frailty model pays 2 AIC units for its unused
``theta``: report the proportional-hazards model, since a variance on its
boundary has no meaningful Wald interval (``param_cb('theta')`` is then
``[0, inf]``).


Model Selection
---------------

With several competing models it is useful to compare them on information
criteria. AIC penalises log-likelihood by the number of parameters (favouring
simpler models); BIC additionally penalises by sample size (favouring even
simpler models with larger datasets). Lower is better for both. In surpyval
the parameter count :math:`k` is the number of *estimated* parameters — held
(``fixed``) parameters and the accelerated-life placeholder are not counted —
and the BIC's sample size is the number of exactly observed failures.

For the tires data, we can compare the three statistical regression families
with a Weibull baseline, and try a second baseline for AFT and PO:

.. jupyter-execute::

    from surpyval import WeibullAFT, WeibullPH, LogNormalAFT, LogisticPO
    from surpyval import PO
    from surpyval import Weibull

    models = {
        'WeibullPH':    WeibullPH.fit(x=x, Z=Z, c=c),
        'WeibullAFT':   WeibullAFT.fit(x=x, Z=Z, c=c),
        'WeibullPO':    PO(Weibull).fit(x=x, Z=Z, c=c),
        'LogNormalAFT': LogNormalAFT.fit(x=x, Z=Z, c=c),
        'LogisticPO':   LogisticPO.fit(x=x, Z=Z, c=c),
    }

    for name, m in models.items():
        print(f'{name:12s}  AIC={m.aic():6.2f}  BIC={m.bic():6.2f}')

The PH and AFT rows are identical — for a Weibull baseline they are the same
model (see `Accelerated Failure Time (AFT)`_). Proportional odds comes within
one AIC unit of them with either baseline, and the log-normal AFT is about
three units behind, so the choice of baseline matters here as much as the
choice of family. Differences of a unit or two are not meaningful — with 11
failures the data cannot separate these descriptions — so compare each family
at its best baseline before ruling it out, and let the purpose and the
diagnostics decide between close contenders.

A note of caution: AIC and BIC compare how well a model fits the *observed
data*, not whether the model's assumptions are correct. A PH model with a lower
AIC than a PO model does not mean PH is the "true" model — it means PH uses its
parameters more efficiently on this dataset. If the proportional hazards
assumption is violated (e.g. survival curves cross), a lower-AIC PH model can
still give misleading predictions. Goodness-of-fit diagnostics like
Schoenfeld residuals (for PH) or log-log survival plots should accompany any
model comparison. Information criteria are also only comparable between models
fitted to the same data by full likelihood: a Cox model's partial likelihood,
the Lin-Ying estimator and Buckley-James have no comparable likelihood, so
compare those on held-out predictions instead (next section).

Validating a survival predictor
-------------------------------

Information criteria compare models on the data they were fit to. To judge how
well a model *predicts*, score it on held-out data. Two right-censored-standard
metrics live in :mod:`surpyval.metrics.validation` (importable from
``surpyval.metrics``), and both work for **any** model that exposes
``sf(x, Z)`` — the parametric families, ``CoxPH``, and the ``surpyval.beta.ml``
forest (see :doc:`comparison_and_validation`).

Both handle censoring by inverse-probability-of-censoring weighting (IPCW), so
a subject censored before the horizon does not silently bias the score.

- The **Brier score** ``BS(t)`` is the weighted mean squared error between the
  predicted survival ``S(t | Z)`` and the survival indicator; the **integrated
  Brier score** (IBS) averages it over a time grid. Lower is better; a useful
  model scores below the marginal Kaplan-Meier reference.
- The **time-dependent AUC** (Uno's cumulative/dynamic estimator) measures
  discrimination as a function of the horizon — the probability that a subject
  who has failed by ``t`` was assigned a higher risk than one still event-free.
  0.5 is chance, 1.0 is perfect.

The helper :func:`~surpyval.metrics.validation.survival_probability` builds the predicted
survival matrix ``S(times | Z_i)`` from a fitted model. We fit a Cox model on a
training set and score it on an independent test set:

.. jupyter-execute::

    from surpyval import CoxPH, KaplanMeier
    from surpyval.metrics import (
        survival_probability, brier_score, integrated_brier_score, auc_td,
    )

    def make(seed, n=600):
        r = np.random.default_rng(seed)
        Z = r.normal(0, 1, (n, 2))
        t = r.exponential(1.0 / np.exp(1.2 * Z[:, 0] - 0.8 * Z[:, 1]))
        cens = r.exponential(np.median(t) * 3)
        return np.minimum(t, cens), (cens < t).astype(int), Z

    x_tr, c_tr, Z_tr = make(1)
    x_te, c_te, Z_te = make(2)
    cox = CoxPH.fit(x=x_tr, Z=Z_tr, c=c_tr)

    times = np.quantile(x_te[c_te == 0], [0.25, 0.5, 0.75])
    S = survival_probability(cox, Z_te, times)

    _, bs = brier_score(x_te, c_te, S, times, x_train=x_tr, c_train=c_tr)
    ibs = integrated_brier_score(x_te, c_te, S, times, x_train=x_tr, c_train=c_tr)
    _, auc = auc_td(x_te, c_te, 1 - S, times)
    print('Brier score :', np.round(bs, 3))
    print('IBS         :', round(ibs, 3))
    print('AUC         :', np.round(auc, 3))

The AUC around 0.8 shows the two covariates discriminate well. To see that the
IBS is meaningful, compare it against the marginal Kaplan-Meier — a model that
ignores the covariates entirely. The Cox model should score lower:

.. jupyter-execute::

    km = KaplanMeier.fit(x_tr, c_tr)
    S_km = np.tile([km.sf([t])[0] for t in times], (len(x_te), 1))
    ibs_km = integrated_brier_score(
        x_te, c_te, S_km, times, x_train=x_tr, c_train=c_tr
    )
    print(f'IBS  Cox = {ibs:.3f}   marginal KM = {ibs_km:.3f}')

Concordance
~~~~~~~~~~~

Harrell's concordance index is the fraction of comparable pairs of subjects
that a risk score ranks in the right order (the one that failed first has the
higher score), with 0.5 for chance and 1 for perfect; the pair and tie rules
are on the :doc:`regression analysis` page. surpyval's implementation is
``surpyval.utils.score.score(x, c, scores)``, where the scores are
*mortality-like* — higher means expected to fail earlier. For a proportional
hazards model the linear predictor :math:`\beta'Z` is exactly such a score:

.. jupyter-execute::

    from surpyval.utils.score import score

    print('C, Cox on the test set  : %.3f' % score(x_te, c_te, Z_te @ cox.beta))
    print('C, a random score       : %.3f' % score(
        x_te, c_te, np.random.default_rng(0).normal(size=len(x_te))))

For proportional odds, where a higher linear predictor means a *longer* life,
negate it first; for any model, the predicted failure probability
:math:`1 - S(t \mid Z)` at a fixed time is also a valid risk score. Concordance only
measures ranking; pair it with the Brier score, which also checks that the
predicted probabilities are right.

Survival trees and random survival forests (beta)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When you do not know how the covariates act — thresholds, interactions, effects
that only appear in combination — a tree-based predictor can find the structure
itself. ``SurvivalTree`` recursively splits the data on one covariate at a time
and fits a survival model in each leaf; ``RandomSurvivalForest`` averages many
trees grown on bootstrap resamples. Both live in ``surpyval.beta.ml``: they
are tested and usable, but their interface may still change (see
:doc:`surpyval.beta`). The theory is in the :doc:`regression analysis` page.

The simulated data below have a risk that depends on an *interaction*: units
fail faster only when :math:`z_0 > 0.5` **and** :math:`z_1 < 0.5`; the third
covariate is noise. A single shallow tree, allowed to consider every covariate
at each split (``n_features_split='all'``), finds the interaction on its own.
The tree ``kind`` couples the split rule with the leaf model:
``'non-parametric'`` uses the log-rank statistic and Nelson-Aalen leaves (for
observed, right-censored and left-truncated data); ``'weibull'`` (the default)
and ``'exponential'`` use a likelihood split and parametric leaves and accept
every kind of censoring and truncation, at a higher computational cost:

.. jupyter-execute::

    from surpyval.beta.ml import SurvivalTree, RandomSurvivalForest

    def make_tree_data(seed, n=300):
        r = np.random.default_rng(seed)
        Z = r.uniform(0, 1, (n, 3))
        risky = (Z[:, 0] > 0.5) & (Z[:, 1] < 0.5)     # an interaction
        t = 10 * r.weibull(1.5, n) * np.exp(-1.5 * risky / 1.5)
        cens = r.uniform(5, 25, n)
        return np.minimum(t, cens), (cens < t).astype(int), Z

    xt_tr, ct_tr, Zt_tr = make_tree_data(1)
    xt_te, ct_te, Zt_te = make_tree_data(2)

    np.random.seed(0)          # trees draw their candidate features at random
    tree = SurvivalTree.fit(x=xt_tr, Z=Zt_tr, c=ct_tr, max_depth=2,
                            kind='non-parametric', n_features_split='all')
    root = tree._root
    print('root split     : z%d <= %.2f' % (root.split_feature_index,
                                             root.split_feature_value))
    print('right child    : z%d <= %.2f' % (
        root.right_child.split_feature_index,
        root.right_child.split_feature_value))
    print('S(5), risky unit   :', tree.sf([5.0], [0.9, 0.1, 0.5]).round(3))
    print('S(5), ordinary unit:', tree.sf([5.0], [0.1, 0.9, 0.5]).round(3))

The root splits on :math:`z_0` near 0.5, and the right-hand branch then splits
on :math:`z_1` near 0.5 — the interaction, recovered without being specified.
(The ``_root`` node structure is shown only to make the splits visible.)

A forest averages many such trees, each grown on a bootstrap sample and
considering a random subset of ``n_features_split`` covariates at each split.
Its ``sf(x, Z)`` returns a grid — one row per covariate row, one column per
time — unlike the element-wise regression models, and its ``score(x, Z, c)``
is the concordance of its mortality score. The forest reports its progress
through joblib on standard error, which we silence here. On held-out data it
is compared with a Cox model on the same metrics:

.. jupyter-execute::

    import contextlib, io

    np.random.seed(0)
    with contextlib.redirect_stderr(io.StringIO()):    # joblib progress log
        rsf = RandomSurvivalForest.fit(x=xt_tr, Z=Zt_tr, c=ct_tr, n_trees=10,
                                       max_depth=3, n_features_split=2,
                                       kind='non-parametric')
    print('forest sf grid shape:', rsf.sf([3.0, 6.0], Zt_te[:4]).shape)

    cox_t = CoxPH.fit(x=xt_tr, Z=Zt_tr, c=ct_tr)
    grid = np.array([3.0, 6.0, 9.0])
    for name, m, risk in [('forest', rsf, None),
                          ('Cox', cox_t, Zt_te @ cox_t.beta)]:
        S_m = survival_probability(m, Zt_te, grid)
        ibs_m = integrated_brier_score(xt_te, ct_te, S_m, grid,
                                       x_train=xt_tr, c_train=ct_tr)
        C = rsf.score(xt_te, Zt_te, ct_te) if risk is None else \
            score(xt_te, ct_te, risk)
        print(f'{name:6s}  IBS = {ibs_m:.3f}   C = {C:.3f}')

With ten shallow trees the forest already edges out a Cox model that cannot
represent the interaction; more and deeper trees usually widen the gap, at a
proportional cost in time. Setting ``kind='weibull'`` (the default) gives
parametric leaves and handles left and interval censoring and truncation, but
fits a likelihood at every candidate split and is much slower. Fitted trees and
forests serialise like every other model (next section).

Saving and loading a fitted model
---------------------------------

A fitted parametric regression model can be serialised to a plain dictionary
or a JSON file and rebuilt later — so you can fit once and reuse the model
without the training data on hand. This works for the fixed-form parametric
families — Accelerated Failure Time, Proportional Hazards, Proportional Odds
and (parametric) Additive Hazards — and for Accelerated Life models built on a
built-in life model.

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
model (see :doc:`surpyval.serialisation`):

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
(``cb``, ``param_cb``, ``standard_errors``) without the original data, and it
survives repeated save/load cycles. Only the
prediction/inference state is serialised — the empirical overlay in ``plot``
needs the fitted data, so re-fit if you need that. An Accelerated Life model is
rebuilt from its distribution and life-model names, so the built-in life models
round-trip but a custom ``LifeModel`` subclass (like ``InverseSquareRoot``
above) cannot be rebuilt from a name and raises ``NotImplementedError`` when
serialised:

.. jupyter-execute::

    al_restored = surpyval.from_dict(model_arr.to_dict())
    print('Arrhenius model restored:',
          np.allclose(al_restored.sf([1e4], [use]), model_arr.sf([1e4], [use])))
    try:
        model_custom.to_dict()
    except NotImplementedError as err:
        print('custom life model:', type(err).__name__)

A model fitted from a DataFrame keeps its covariate names, and a formula model
keeps its categorical levels, so the restored model still predicts from a
DataFrame of raw covariates. The exception is a formula with a data-dependent
transform (``scale()``, ``center()``), whose fitted statistics cannot be
stored; serialising one raises rather than round-tripping to a wrong encoding.

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
data) all survive the round-trip. The residual diagnostics, ``check_ph`` and
the robust variance need the training data and are not available on a restored
Cox model, and a stratified Cox model cannot be serialised at all. Be aware that
a serialised Buckley-James model contains its training data.

The frailty model, survival trees and random survival forests serialise too —
a tree as its node structure with each leaf's fitted model, a forest as its
trees — and all of them restore through ``surpyval.from_dict``:

.. jupyter-execute::

    frailty_back = surpyval.from_dict(by_lot.to_dict())
    forest_back = surpyval.from_dict(rsf.to_dict())
    print(type(frailty_back).__name__, np.allclose(
        frailty_back.sf([10.0], [0.0]), by_lot.sf([10.0], [0.0])))
    print(type(forest_back).__name__, np.allclose(
        forest_back.sf([5.0], Zt_te[0]), rsf.sf([5.0], Zt_te[0])))

A restored tree or forest is a predictor only: it keeps no training data and
cannot be re-fitted.
