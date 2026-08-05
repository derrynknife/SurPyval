Buckley-James
=============

A semi-parametric accelerated-failure-time estimator. It regresses
:math:`\log x` on the covariates by least squares, but a right-censored
observation contributes its *conditional expectation* given that it
survived past the censoring time rather than the censoring time itself:

.. math::

    \hat{y}_i = \delta_i \log x_i
              + (1 - \delta_i)\, E\!\left[\log T \mid \log T > \log x_i\right]

The expectation is taken under the Kaplan-Meier estimate of the
residual distribution, which depends on the coefficients, so the two
steps alternate until they stop moving.

The appeal is that it makes no parametric assumption about the baseline
-- it is the AFT counterpart of what Cox regression is for proportional
hazards. The cost is that the iteration is not guaranteed to converge,
and standard errors come from resampling rather than a likelihood.

Usage::

    from surpyval import BuckleyJames
    model = BuckleyJames.fit(x, Z=Z, c=c)

.. autoclass:: surpyval.univariate.regression.buckley_james.buckley_james.BuckleyJames_
    :members:

.. autoclass:: surpyval.univariate.regression.buckley_james.buckley_james.BuckleyJamesModel
    :members:
