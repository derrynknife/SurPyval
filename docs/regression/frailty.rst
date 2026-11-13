Shared Frailty Models
=====================

A frailty model adds an unobserved multiplicative random effect shared
by every observation in a group, so that units from the same group are
correlated:

.. math::

    h(x \mid Z, w) = w \, h_0(x) \, e^{\beta' Z}

with the frailty :math:`w` drawn once per group from a distribution with
mean one and variance :math:`\theta`. Setting :math:`\theta = 0`
recovers the ordinary proportional-hazards model, so the fitted
:math:`\theta` measures how much of the variation is between groups
rather than within them.

Use this when observations arrive in clusters that share something you
have not measured -- repairs on the same machine, patients at the same
hospital, components from the same batch. Treating them as independent
understates the uncertainty.

Factory::

    from surpyval import Frailty, Weibull
    model = Frailty(Weibull).fit(x, Z=Z, c=c, groups=unit_id)

Pre-built instances: ``ExponentialFrailty``, ``WeibullFrailty``,
``LogNormalFrailty``, ``GammaFrailty``.

The frailty is Gamma-distributed by default, so it integrates out of
each group's likelihood in closed form. ``Frailty(dist,
family="lognormal")`` takes a log-normal frailty instead, :math:`w =
e^{v}` with :math:`v \sim N(0, \theta)` (the parameterisation of R's
``frailtypack`` and ``coxme``: ``theta`` is then the variance of the log
frailty, and the median frailty is 1), whose group integrals are computed
by adaptive Gauss-Hermite quadrature with 30 nodes (accurate to about
:math:`10^{-10}` in each group's log-likelihood for :math:`\theta \le
1`, :math:`10^{-7}` at 2 and :math:`10^{-5}` at 5). ``frailty_variance``
(the variance of the frailty scaled to mean 1) and ``kendall_tau`` put the
two families on one scale. Only
observed and right-censored data are supported, and at least two groups
are needed; a row with a missing group label (``None``, ``NaN``) is dropped,
with a warning, and a missing ``group=`` at prediction gives ``nan``. The fitted model predicts the *marginal* (population) curve
by default, or the curve conditional on an observed group's posterior
frailty (``group=``) or on a given frailty (``frailty=``). Like the
parametric regression models it reports ``neg_ll()``, ``aic()``,
``bic()`` and ``aic_c()`` (``theta`` counted as a parameter), so a
frailty fit can be compared directly with the proportional-hazards fit it
reduces to at ``theta = 0``.

``CoxFrailty`` fits the gamma frailty with the baseline left unspecified
(a Cox baseline), by EM over the frailties with ``theta`` maximising the
profile likelihood -- the fit of R's ``coxph(... + frailty(id, dist =
"gamma"))``::

    from surpyval import CoxFrailty
    model = CoxFrailty.fit(x, Z=Z, c=c, groups=unit_id)

It predicts like the parametric models, from a step baseline as ``CoxPH``
does, and reports R's integrated likelihood (its ``loglik``) as
``log_likelihood``, which ``aic()`` and ``bic()`` penalise.

``CoxFrailty.fit(..., family="lognormal")`` fits the log-normal frailty
with a Cox baseline instead: each group's frailty is :math:`e^{v}` with
:math:`v \sim N(0, \theta)`, a random intercept on the log hazard. It is
fitted by maximum likelihood as R's ``coxme(... + (1 | id))``: for a
given :math:`\theta` the coefficients and each group's :math:`v` maximise
the penalised partial likelihood, and :math:`\theta` maximises its
Laplace approximation to the integrated partial likelihood, reported as
``log_likelihood``. ``frailties`` are :math:`e^{\hat v}`, each group's
predicted (posterior-mode) log-frailty exponentiated; ``theta``'s
standard error is from the curvature of its profile likelihood. R's
``coxph(... + frailty(id, dist = "gaussian"))`` fits the same penalised
partial likelihood but chooses :math:`\theta` by a different rule (an
approximate REML rather than maximum likelihood); at the same
:math:`\theta` (``theta=``) the two fits agree.

Mixed-effects Cox models
------------------------

When the groups come in more than one layer -- wards within hospitals,
or batches crossed with machines -- or when the effect of a covariate
itself differs from group to group, use ``CoxMixedEffects``, the
counterpart of R's ``coxme``. In plain words: every grouping factor gives
each of its groups a random multiplier on the hazard (and, if you ask, a
random change to a covariate's effect), drawn from a normal distribution
on the log scale whose variance the fit estimates. A group with few
events is pulled towards the average (shrinkage). The fixed coefficients
compare two units of the same groups.

The model is

.. math::

    h_i(t) = h_0(t) \exp\Big(\beta' Z_i + \sum_{\text{terms } t}
    v_{ti}' b_{t, g_t(i)}\Big), \qquad b_{t, g} \sim N(0, \Sigma_t),

where ``v`` holds a 1 (a random intercept) and any random-slope
covariates, and :math:`b_{t,g}` are group ``g``'s random effects for term
``t``, independent between groups and between terms. The fit is
``coxme``'s (Ripatti and Palmgren 2000; Therneau, Grambsch and Pankratz
2003): for given variances, :math:`\beta` and the :math:`b` maximise the
penalised partial likelihood :math:`PL(\beta, b) - \tfrac12 b'
\Sigma^{-1} b`; the variances maximise the Laplace approximation to the
integrated partial likelihood. Ties are Efron's or Breslow's, and counts,
late entry (``tl``) and strata are taken as ``CoxPH`` takes them.

The random effects are given with ``groups=`` (a random intercept), with
``RandomEffect`` terms, or in ``fit_from_df`` with the bar notation of
``coxme`` and ``lme4``::

    from surpyval import CoxMixedEffects, RandomEffect

    # a random intercept per centre
    model = CoxMixedEffects.fit(x, Z=Z, c=c, groups=centre)
    # a correlated random intercept and slope of column 0 per centre
    model = CoxMixedEffects.fit(
        x, Z=Z, c=c, random_effects=RandomEffect(centre, slopes=[0])
    )
    # wards nested in hospitals, crossed with batches
    model = CoxMixedEffects.fit(
        x, Z=Z, c=c,
        random_effects=[
            RandomEffect(hospital),
            RandomEffect((hospital, ward)),
            RandomEffect(batch),
        ],
    )
    # the same in a DataFrame
    model = CoxMixedEffects.fit_from_df(
        df, x_col="time", c_col="censored", Z_cols=["trt", "age"],
        random_effects="(1 | hospital/ward) + (1 | batch)",
    )

In the bar notation ``(1 | g)`` is a random intercept, ``(1 + trt | g)``
a correlated intercept and slope, ``(0 + trt | g)`` a slope alone,
``(1 + trt || g)`` an independent intercept and slope, and ``(1 | a/b)``
``b`` nested in ``a``. A slope must say whether the term has an intercept:
``(trt | g)`` is refused, because ``lme4`` and ``coxme`` read it
differently. A random slope is a covariate of the fixed part too.

The model reports what ``coxme`` reports. ``beta``, ``summary()`` and
``param_cb`` give the fixed coefficients with Wald intervals, from the
inverse of the penalised information. ``variance_components()`` gives
each term's variances, standard deviations and correlations;
``variance_cb()`` their profile-likelihood intervals (a lower bound of 0
where the data cannot rule a variance of 0 out); ``random_effects`` each
group's predicted random effects (``coxme``'s ``ranef``).
``log_likelihood`` is the integrated log-likelihood (the one ``aic()``
and ``bic()`` penalise, by the fixed coefficients and the variance
parameters), ``log_likelihood_penalised`` and ``log_likelihood_fitted``
the penalised partial likelihood at the fit with and without its penalty,
``log_likelihood_null`` its value at :math:`\beta = 0, b = 0` and
``df_penalised`` the penalised fit's effective degrees of freedom.
``lr_test()`` tests all the random effects against the Cox model: a
variance of 0 is on the edge of its range, so with one variance the
p-value is half the chi-square's (the 50:50 mixture), with several
independent variances the binomial mixture, and with correlations the
plain chi-square, which is conservative.

Predictions (``sf``, ``Hf``, ``ff``, ``hf``, ``df``, ``qf``) are marginal
by default, as ``CoxFrailty``'s: the random effects of every term not
named in ``group=`` are averaged over, the curve of a unit from a group
not yet seen. ``group=label`` (``group={"hospital": 3, "batch": 2}`` with
several terms) conditions on an observed group's predicted random
effects, and ``marginal=False`` sets the remaining random effects to 0
(a typical group, the curve ``coxme`` predicts). ``stratum=`` picks a
stratum's baseline. The baseline is a step
function, as ``CoxPH``'s.

A worked example, the lung cancer data with a random intercept per
institution:

.. code:: python

    >>> from surpyval import CoxMixedEffects
    >>> from surpyval.datasets import load_lung
    >>> df = load_lung()
    >>> df["censored"] = 1 - df["status"]
    >>> model = CoxMixedEffects.fit_from_df(
    ...     df, x_col="time", c_col="censored", Z_cols=["age", "sex"],
    ...     random_effects="(1 | inst)",
    ... )
    >>> model.lr_test()["p_value"]
    1.0
    >>> model.variance_cb().round(3)
                           estimate  lower  upper
    term parameter
    inst var((Intercept))       0.0    0.0  0.104

The institutions do not differ detectably: the variance is 0 (the fit is
``CoxPH``'s, on the 227 rows with an institution), and the data rule out a
variance above about 0.10, a standard deviation of 0.32 in the log
hazard. (One row has no institution and is dropped, with a warning.)

Not available: ``random()`` (the baseline is a step function: use a
parametric frailty model), ``diagnose()`` (test the random effects with
``lr_test()``, and check the fixed part with ``CoxPH.fit(...).diagnose()``),
weights, and interval- or left-censored data (use ``Frailty(Weibull,
family="lognormal")``). The random-effects design is held as a dense
matrix, so a model with many thousands of groups is slow.

.. autofunction:: surpyval.univariate.regression.frailty.Frailty

.. autoclass:: surpyval.univariate.regression.frailty.frailty_fitter.FrailtyFitter
    :members: fit, fit_from_df

.. autoclass:: surpyval.univariate.regression.frailty.frailty_model.FrailtyModel
    :members:

.. autoclass:: surpyval.univariate.regression.frailty.cox_frailty.CoxFrailtyFitter
    :members: fit, fit_from_df

.. autoclass:: surpyval.univariate.regression.frailty.cox_frailty.CoxFrailtyModel
    :members:

.. autoclass:: surpyval.univariate.regression.frailty.cox_mixed_effects.CoxMixedEffectsFitter
    :members: fit, fit_from_df

.. autoclass:: surpyval.univariate.regression.frailty.cox_mixed_effects.CoxMixedEffectsModel
    :members:

.. autoclass:: surpyval.univariate.regression.frailty.cox_mixed_effects.RandomEffect
