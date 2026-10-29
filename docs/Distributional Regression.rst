Distributional Regression
=========================

.. jupyter-execute::
    :hide-code:

    import numpy as np
    import pandas as pd
    import surpyval as surv

What it does
------------

A regression model in SurPyval usually lets the covariates change one
thing about the life distribution. In an accelerated failure time (AFT)
model a heavier load makes every unit wear out sooner by the same
factor; in a proportional hazards (PH) model it multiplies the hazard by
the same factor at every age. Either way the *shape* of the distribution
is the same for every unit: if failures are wear-out at light load, they
are wear-out, in the same way, at heavy load.

A **distributional regression** lets the covariates change every
parameter of the distribution, each through its own link. For a Weibull
that is the scale ``alpha`` (how long units typically last) *and* the
shape ``beta`` (whether failures come early, at random or from wear-out).
It answers a question the AFT and PH models assume away: **does the way
units fail change with the conditions they run in?** It is the model of
R's ``gamlss`` ("generalised additive models for location, scale and
shape") and of ``flexsurvreg``'s ``anc`` argument, on SurPyval's survival
data: right, left and interval censoring, truncation and counts work as
in every other fit.

When to use it:

- the shape may depend on a stress (a shape near 1 at light load,
  random failures, and well above 1 at heavy load, wear-out), which
  changes the maintenance strategy, not only the interval;
- the spread of lives differs between groups (sites, suppliers,
  treatment arms) while the typical life may or may not;
- a plot of the data by group shows lines that are not parallel on a
  Weibull (or log-normal) probability plot: the AFT and PH models make
  them parallel.

A worked example: does the shape depend on load?
-------------------------------------------------

Bearings were run at four loads (1 to 4 kN) until they failed or the
test stopped at 2000 hours. (The data are simulated, with a shape that
grows with load, so we can see whether the fits find it.)

.. jupyter-execute::

    rng = np.random.default_rng(2026)
    load = rng.choice([1.0, 2.0, 3.0, 4.0], size=240)
    life = 3000 * np.exp(-0.4 * load) * rng.weibull(np.exp(0.25 * load))
    bearings = pd.DataFrame({
        "hours": np.minimum(life, 2000.0),
        "censored": (life > 2000.0).astype(int),
        "load": load,
    })

First the usual model, a Weibull AFT: load shortens life, one shape for
all. Then the distributional model, with load on both parameters (with
no parameter named, every covariate goes on every parameter):

.. jupyter-execute::

    aft = surv.WeibullAFT.fit_from_df(
        bearings, x_col="hours", c_col="censored", Z_cols=["load"]
    )
    model = surv.WeibullDistributional.fit_from_df(
        bearings, x_col="hours", c_col="censored", Z_cols=["load"]
    )
    model

Read the coefficients through their links. ``alpha_load = -0.40``: each
kN multiplies the scale by :math:`e^{-0.40} = 0.67`. ``beta_load =
0.27``: each kN multiplies the shape by :math:`e^{0.27} = 1.31`, and its
interval is well clear of 0. So the shape does change with load. The
shape at each load is one call away:

.. jupyter-execute::

    loads = pd.DataFrame({"load": [1.0, 2.0, 3.0, 4.0]})
    model.parameters_at(loads).round(2)

At 1 kN the shape is 1.25, close to random failures; at 4 kN it is 2.8,
clear wear-out. The likelihood-ratio test of "the shape does not depend
on load" compares the two fits (the AFT is the distributional model with
``beta_load = 0``), and the AIC agrees:

.. jupyter-execute::

    from scipy.stats import chi2

    lr = 2 * (aft.neg_ll() - model.neg_ll())
    print(f"LR statistic {lr:.1f}, p = {chi2.sf(lr, 1):.1e}")
    print(f"AIC: AFT {aft.aic():.1f}, distributional {model.aic():.1f}")

It matters for what the models predict. The B10 life (the age by which
10% have failed) at each load, from each model:

.. jupyter-execute::

    print("distributional:", model.qf(0.1, loads).round(0))
    print("AFT:           ", aft.qf(0.1, loads).round(0))
    print(model.quantile_cb(0.1, loads).round(0))

The AFT model, forced to one shape for all loads, puts the B10 at 4 kN
at 189 hours; the distributional model, which lets the heavy-load
failures bunch up as wear-out does, at 287 hours (95% bounds 237 to 347).
At 1 kN it is the other way round: 649 hours from the AFT, 350 from the
distributional model, because random-looking failures at light load
start early.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _p = model.params
    assert np.allclose(_p[2:], [-0.3978, 0.2695], atol=5e-5)
    assert np.allclose(model.parameters_at(loads).iloc[:, 1],
                       [1.254, 1.642, 2.151, 2.816], atol=5e-4)
    assert abs(lr - 22.34) < 0.01 and chi2.sf(lr, 1) < 1e-5
    assert np.allclose(model.qf(0.1, loads), [350.0, 359.3, 333.6, 287.0],
                       atol=0.05)
    assert np.allclose(aft.qf(0.1, loads), [648.7, 429.9, 284.9, 188.8],
                       atol=0.05)
    assert np.allclose(model.quantile_cb(0.1, loads)[3], [237.1, 347.3],
                       atol=0.05)

Choosing each parameter's covariates
------------------------------------

Each parameter takes its own covariates. With columns, ``covariates``
names them per parameter; a parameter not named has none (its intercept
only):

.. jupyter-execute::

    shape_only = surv.WeibullDistributional.fit_from_df(
        bearings, x_col="hours", c_col="censored", Z_cols=["load"],
        covariates={"beta": ["load"]},
    )
    shape_only.parameter_names

With a formula per parameter, ``formulas`` does the same, with
everything a formula can say (categorical columns, interactions,
transforms such as ``np.log(load)``):

.. jupyter-execute::

    by_formula = surv.WeibullDistributional.fit_from_df(
        bearings, x_col="hours", c_col="censored",
        formulas={"alpha": "load", "beta": "np.log(load)"},
    )
    by_formula.parameter_names

With arrays, ``fit(x, Z, c=..., covariates={"beta": [0]})`` picks columns
of ``Z`` by number. A fitted model predicts from a DataFrame with the
same columns, or from an array of every column it was fitted to, one row
per unit, as every regression does.

The model, in formulas
----------------------

For a distribution with parameters :math:`\theta_1, \dots, \theta_K`,
each parameter has a linear predictor and a link :math:`g_k`:

.. math::

    g_k\bigl(\theta_k(z)\bigr) = g_k(\theta_k) + x_k(z)' b_k,
    \qquad k = 1, \dots, K,

where :math:`\theta_k` is the parameter at :math:`z = 0` and
:math:`x_k(z)` the parameter's columns. A unit with covariates :math:`z`
has the distribution's own functions at its parameters, so for a Weibull

.. math::

    S(t \mid z) = \exp\left[-\left(\frac{t}{\alpha(z)}\right)^{\beta(z)}\right],
    \qquad
    \alpha(z) = \alpha\, e^{x_\alpha(z)' b_\alpha},
    \quad
    \beta(z) = \beta\, e^{x_\beta(z)' b_\beta}.

The likelihood is that of every SurPyval regression, each row at its own
parameters: :math:`\log f` for an observed time, :math:`\log S` for a
right censored one, :math:`\log F` for a left censored one,
:math:`\log(F(r) - F(l))` for an interval, less the log of each row's
probability of falling inside its truncation window, all weighted by the
counts. It is maximised directly, with the exact gradient and Hessian.

**Links.** The default link keeps each parameter in its range: log for a
positive parameter (a scale, a shape, a rate), identity for one on the
whole line (a location). ``links={"mu": "log"}`` puts a log link on a
location; an identity link on a positive parameter is refused, because
it could take the parameter below 0, where the likelihood is not
defined.

**Parameters.** ``params`` are the distribution's parameters at
:math:`z = 0` (at the covariate means with ``center=True``), then each
parameter's coefficients on its link's scale, named
``<parameter>_<column>``. A coefficient's sign is that of its link: a
positive ``alpha_load`` means a longer life.

**The AFT model is a special case.** Covariates on the scale alone give
an AFT model, with the coefficients' signs reversed (SurPyval's AFT
writes :math:`H_0(e^{b'z} x)`, where a positive coefficient shortens
life):

.. list-table::
   :header-rows: 1

   * - AFT model
     - The same distributional model
     - Coefficients
   * - ``WeibullAFT``
     - ``Weibull``, covariates on ``alpha``
     - ``alpha_z = -b_z``
   * - ``ExponentialAFT``
     - ``Exponential``, on ``failure_rate``
     - ``failure_rate_z = b_z``
   * - ``LogNormalAFT``
     - ``LogNormal``, on ``mu``
     - ``mu_z = -b_z``
   * - ``AFT(LogLogistic)``
     - ``LogLogistic``, on ``alpha``
     - ``alpha_z = -b_z``
   * - ``GammaAFT``
     - ``Gamma``, on its rate ``beta``
     - ``beta_z = b_z``

A Weibull PH model with covariates is also one, covariates on ``alpha``
with ``b_PH = -beta * alpha_z``; and the accelerated life models with a
log-linear life (``Power``: ``alpha`` on ``np.log(V)``) are too. The
tests check each of these gives the same fit.

**Inference.** Wald bounds come from the exact observed information;
``method="lr"`` gives likelihood-ratio bounds and ``method="bootstrap"``
parametric bootstrap bounds, for the parameters (``param_cb``), the
functions (``cb``) and the quantiles (``quantile_cb``), as for the other
regressions. ``parameters_at(Z)`` gives each row's parameters and
``distribution_at(row)`` the univariate distribution of one unit, with
every univariate function.

**Checks of the fit.** Each parameter's columns are checked on their
own: a column that is constant, or a combination of that parameter's
other columns, is aliased (reported as ``nan``, with one warning). A
search that does not reach a verified maximum says so, and data with no
finite maximum (for example a group whose failures all fall at one time,
whose shape then grows without bound) give a "No finite maximum"
warning.

Smooth terms
------------

A spline basis of fixed size in the formula, ``bs(load, df=3)`` or
``cr(load, df=3)``, lets a parameter follow a smooth curve in a
covariate rather than a straight line. The basis is fixed before the
fit, so the fit and its bounds are ordinary maximum likelihood; compare
a few sizes by AIC. ``cr()`` without ``constraints="center"`` includes
the intercept's direction, so one of its columns is aliased (the fit
says so); ``bs()`` does not. Penalised splines, whose flexibility is
chosen from the data, are not available yet.

Limits
------

- The distributions are the continuous ones whose support does not
  depend on their parameters: Exponential, Rayleigh, Weibull, Gamma,
  LogNormal, LogLogistic, Normal, Gumbel, GumbelLEV, Logistic, Beta and
  ExpoWeibull. Prebuilt fitters are ``WeibullDistributional``,
  ``ExponentialDistributional``, ``GammaDistributional``,
  ``LogNormalDistributional``, ``LogLogisticDistributional``,
  ``NormalDistributional``, ``GumbelDistributional`` and
  ``LogisticDistributional``; ``Distributional(dist)`` builds the others.
- A covariate that changes over a unit's life is not supported:
  ``sf_tvc`` and the other path functions refuse, because with the
  shape moving there is no accumulated age to carry from one value to
  the next. Use an AFT or PH model for a covariate path.
- More parameters need more data: covariates on the shape typically
  need several failures at each covariate level.

References: Rigby and Stasinopoulos (2005), *Generalized additive models
for location, scale and shape*, Applied Statistics 54(3); Jackson
(2016), *flexsurv: a platform for parametric survival modeling in R*,
Journal of Statistical Software 70(8); Meeker and Escobar (1998),
*Statistical Methods for Reliability Data*, on accelerated tests whose
shape depends on stress.
