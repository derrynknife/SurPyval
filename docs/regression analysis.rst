
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

In this case the proportional term is the e raised to the power of the cross product of X and beta. Using this as the covairate function is a very common choice. This is because it will not ever become negative. It can capture situations where a covariate will increase the hazard rate if it's coefficient, beta, is positive, and it will decrease the hazard rate it it's coefficient is negative. Also, the dot product can capture a varying number of covariates with ease. For these reasons the Cox model is a widely used. Although you can choose any function for your covariates there is already likely literature about your problem which might indicate which function to use.

Surpyval uses MLE to estimate the parameters for proportional hazards models. This is a simple conversion from regulare MLE since we know the relationship between a baseline distribution and the proportional hazards version. These relationships are:

.. math::

	f(t|X) = \phi(X) h_{0}(t) e^{-\phi(X) H(t)} \\ 
	\\
	F(t|X) = 1 - e^{-\phi(X) H(t)} \\
	\\
	S(t|X) = e^{-\phi(X) H(t)}

It is therefore relatively simple to adjust the MLE methods to accommodate propotional hazard models.

The details on fitting proportional hazards model is detailed more in the surpyval analysis section.

Semi-Parametric
^^^^^^^^^^^^^^^

The previous sections covered 'parametric' and 'non-parametric' survival models, so what is 'semi-parametric'? A semi-parametric model is a survival model with a non-parametric baseline and parametric function that affects that baseline. Recall that a proportional hazard model can be defined as:

.. math::

	h(t|X) = \phi(X) h_{0}(t)

It is interesting to note that the phi term must be parametric, however, the baseline hazard rate need not be parametric, it can be non-parametric! Therefore, what we have is a parametric relationship of the covariates to the baseline hazard rate, but a non-parametric baseline hazard rate, therefore, a 'semi-parametric' model.

By far the most common of any regression model of any kind (parametric, non-parametric, and semi-parametric of all the accelerated life, proportinal hazard, and accelerated time) is the Cox Proportional Hazard model, it is a semi-parametric model.

The Cox model is used in a wide variety of fields. It has been used in criminology to study the recidivism of parolees, in engineering to understand the factors affecting tire reliability, and in medical science to understand factors affecting cancer and other diseases, among many many other applications. The wide use of the model shows the utility the model has and the broad applicability to solve problems.


Accelerated Time
----------------

An accelerated time model is very similar to a proportional hazards model. The difference is where the function is applied; instead of multiplying the hazard function, and accelerated time model multiplies the time by the function of covariates. The general definition is:

.. math::

	f(t|X) = f(\phi(X)t)

It is called an accelerated time since the time term is transformed by the covariates, i.e. time is 'accelerated' by the covariates.

.. math::

	t_{a} = \phi(X)t


Just like proportinal hazards, there are simple transofmations that apply 


.. math::

	f(t|X) = f(\phi(X)t) \\
	\\
	F(t|X) = F(\phi(X)t) \\
	\\
	S(t|X) = S(\phi(X)t)

Given the simple transofmation of the time term the MLE is feasible with an additional transformation step. This is how surpyval estimates the parameters.

Accelerated Life
----------------

An accelerated life model is, in many cases, simply the inverse of an accelerated time model. However, there are some cases where they are different. Consider an accelerated life model with a normal distribution:

.. math::

	F(t|X) = \Phi\left(\frac{\phi(X)t - \mu}{\sigma}\right) \\

Where :math:`\Phi` is the CDF of the standard normal distribution. In this case :math:`\mu` is the expected life of the model, however, we may isntead be interested in determining what effect covariates have on the expected life of an item. In this case we can simply substitute the expected life:

.. math::

	F(t|X) = \Phi\left(\frac{t - \phi(X)}{\sigma}\right) \\

An accelerated life model is, therefore, simply a model where the life parameter of a distribution is substituted with a function of the covariates, that is, it 'accelerates' the expected life, as opposed to accelerating time as per an accelerated time model. For each of the distributions in Surpyval their life parameter that varies is as per the following table:

+--------------+------------+
| Distribution | Life Param |
+--------------+------------+
| Weibull      | alpha      |
+--------------+------------+
| Exponential  | 1./lambda  |
+--------------+------------+
| Normal       | mu         |
+--------------+------------+
| LogNormal    | mu         |
+--------------+------------+
| Gamma        | alpha      |
+--------------+------------+
| Gumbel       | mu         |
+--------------+------------+
| Logistic     | mu         |
+--------------+------------+
| LogLogistic  | alpha      |
+--------------+------------+
| ExpoWeibull  | Not Avail  |
+--------------+------------+
| Uniform      | Not Avail  |
+--------------+------------+
| Beta         | Not Avail  |
+--------------+------------+

Given the simple substitution into the life parameter, surpyval uses MLE to calculate the parameters.

For examples on how to do regression analysis, see the entry in the 'SurPyval Analysis' section of the docs.