
Regression Analysis
===================

The time until some event happens will, almost certainly, be impacted by factors. For example, when considering how long a machine will last before failure an engineer will want to account for the operational conditions. It may operate in a humid environment, or it may operate at a higher rate. The question is then, how do we account for these variations, or 'covariates', on the time until failure?

Regression analysis is the process of capturing the effect that covariates have on the item. That is, we use data on other factors to 'regress' onto the survival distribution. The purpose of this type of regression is so that you can ask, and answer, questions like "what effect will increasing X have on the survival time?"

Surpyval covers three types of regression models. These are:
 - Proportional Hazards,
 - Accelerated Time, and
 - Accelerated Life.

There are special cases when these are the same, however, it is important to understand the difference between them in general. I detail the differences in the following sections.

Proportional Hazards Model
--------------------------

A proportional hazards model is one in which we change the hazard rate of the distribution by some proportional amount. You may recall that every distribution can be defined by a hazard rate or a cumulative hazard rate, see the "Handy References" section which shows that the density, CDF, and survival function can all be defined in terms of the hazard rate, h(t).

So what we can do then is assume that the covariates will affect the survival time of the thing by having some effect on the hazard rate. The general definition for a proportional hazard model is:

.. math::

	h(t|X) = \phi(X) h_{0}(t)

This is to say that the hazard rate at time t is the function (of a vector) of covariates on a 'baseline' hazard rate. Let's use a simple example, a proportional hazard model with covariates that affect a constant hazard rate. Let's say that some factory produces one widget an hour. But this is only wih one machine in operation, if we add a second machine, we can produce widgets at two per hours, if we had a third, it will be three per hour. In this case the base rate is 1 and the function linking X to the base rate is to simply multiply X by the baserate.

This is to say that for this example:

.. math::

    \phi(X) = X \\
    h_{0}(t) = 1

Therefore:

.. math::
	h(t|X) = X

This is an overly simple model, but it shows how we can construct a PH model.

In this case we have a simple proportional hazard model, also, it is limited to only an increasing hazard rate, but sometimes we need to caputre a negative impact. Further, we may need a way to capture more coviariates. For these reasons a very common selection for the function of covariates is an exponential function.

.. math::
	\phi(X) = e^{X\cdot \beta }

Where

.. math::
	X\cdot \beta = X_{0}\beta_{0} + X_{1} \beta_{1} + ... + X_{n-1}\beta_{n-1} + X_{n}\beta_{n}

In this case the proportional term is the e raised to the power of the cross product of X and beta. Using this as the covairate function is a very common choice. This is because it will not ever become negative. It can capture situations where a covariate will increase the hazard rate if it's coefficient, beta, is positive, and it will decrease the hazard rate it it's coefficient is negative.

Semi-Parametric
^^^^^^^^^^^^^^^

The previous sections covered 'parametric' and 'non-parametric' survival models, so what is 'semi-parametric'? A semi-parametric model is a survival model with a non-parametric baseline and parametric function that affects that baseline. Recall that a proportional hazard model can be defined as:

.. math::

	h(t|X) = \phi(X) h_{0}(t)

It is interesting to note that the phi term must be parametric, however, the baseline hazard rate need not be parametric, it can be 'non-parametric'!! Therefore, what we have is a parametric relationship of the covariates to the baseline hazard rate, but a non-parametric baseline hazard rate, therefore, a 'semi-parametric' model.

By far the most common of any regression model of any kind (parametric, non-parametric, and semi-parametric) is the Cox Proportional Hazard model. The CoxPH model is a semi-parametric model and is used in a wide variety of fields.


Accelerated Time
----------------

Coming Soon

Accelerated Life
----------------

Coming Soon