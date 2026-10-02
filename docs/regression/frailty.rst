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
does, and reports R's integrated likelihood (``loglik``) in place of the
information criteria, which a nonparametric baseline does not have.

.. autofunction:: surpyval.univariate.regression.frailty.Frailty

.. autoclass:: surpyval.univariate.regression.frailty.frailty_fitter.FrailtyFitter
    :members: fit, fit_from_df

.. autoclass:: surpyval.univariate.regression.frailty.frailty_model.FrailtyModel
    :members:

.. autoclass:: surpyval.univariate.regression.frailty.cox_frailty.CoxFrailtyFitter
    :members: fit, fit_from_df

.. autoclass:: surpyval.univariate.regression.frailty.cox_frailty.CoxFrailtyModel
    :members:
