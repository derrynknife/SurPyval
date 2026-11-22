Censored Quantile Regression
============================

A semi-parametric model of the *quantiles* of the time to an event. For each
level :math:`\tau` the :math:`\tau` quantile of the log time is linear in the
covariates,

.. math::

    \log Q_T(\tau \mid Z) = \beta_0(\tau) + \beta(\tau)' Z,

with a coefficient vector of its own at every level. A covariate can
therefore shorten the early failures (the B10 life, :math:`\tau = 0.1`) a
great deal and the median hardly at all. Neither proportional hazards nor an
accelerated failure time is assumed: an AFT model is the special case where
every slope is the same at every level. :math:`e^{\beta_k(\tau)}` is the
factor by which a unit of covariate :math:`k` multiplies the :math:`\tau`
quantile.

Right-censored data are taken by one of two estimators, chosen with
``method=``, each fitted on a grid of levels (``step=0.01`` by default) from
the bottom up, as R's ``quantreg::crq`` fits them:

- ``"peng-huang"`` (the default): Peng and Huang's (2008) martingale
  estimating equations. With no covariates it is the quantile of the
  Nelson-Aalen (Fleming-Harrington) survival curve.
- ``"portnoy"``: Portnoy's (2003) recursively reweighted quantile
  regression, which moves each censored unit's weight to the right once
  the quantile has passed it. With no censoring it is ordinary quantile
  regression; with no covariates it is the Kaplan-Meier quantile.

The process stops at the last level the data determine,
``model.tau_max``: past the last events the censoring leaves nothing to fit.
Each unit's predicted distribution (``ff``, ``sf``, ``Hf``, ``qf``) is read
from its quantiles at the fitted levels, rearranged into order where they
cross; ``ff`` rises in steps to ``tau_max`` and holds it, and ``qf`` is
``nan`` above it. There is no closed-form standard error: ``bootstrap_ci``,
``summary(n_boot=...)``, ``quantile_cb`` and ``cb`` give percentile bootstrap
intervals, resampling the units. Only observed and right-censored data with
positive times are accepted.

Usage::

    from surpyval import QuantileRegression
    model = QuantileRegression.fit(x, Z, c=c)
    model.coefficients(0.1)          # intercept and slopes at the B10
    model.qf(0.1, Z_new)             # the B10 life of each new unit
    model.quantile_cb(0.1, Z_new, alpha_ci=0.1, bound="lower")

See "Semi-Parametric — Censored Quantile Regression" in
:doc:`/Regression Modelling with SurPyval` for a worked example.

.. autoclass:: surpyval.univariate.regression.quantile.quantile_regression.QuantileRegression_
    :members:

.. autoclass:: surpyval.univariate.regression.quantile.quantile_regression.QuantileRegressionModel
    :members:
