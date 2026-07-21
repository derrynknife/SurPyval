
Regression Analysis
===================

The time until some event happens will, almost certainly, be impacted by factors. For example, when considering how long a machine will last before failure an engineer will want to account for the operational conditions. It may operate in a humid environment, or it may operate at a higher rate. The question is then, how do we account for these variations, or 'covariates', on the time until failure?

Regression analysis is the process of capturing the effect that covariates have on the item. That is, we use data on other factors to 'regress' onto the survival distribution. The purpose of this type of regression is so that you can ask, and answer, questions like "what effect will increasing X have on the survival time?"

Surpyval covers five families of regression model, distinguished by *how* the covariates act on the distribution:

 - Proportional Hazards (they multiply the hazard),
 - Accelerated Failure Time (they scale the time axis),
 - Accelerated Life (they scale the life parameter),
 - Proportional Odds (they multiply the odds of failure), and
 - Additive Hazards (they add to the hazard).

There are special cases when several of these coincide, however, it is important to understand the difference between them in general. I detail the differences in the following sections.

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

For a *parametric* proportional hazards model (a known baseline such as the Weibull) surpyval uses MLE to estimate the parameters. This is a simple conversion from regular MLE since we know the relationship between a baseline distribution and the proportional hazards version. (The Cox model in the next section is *semi-parametric* — its baseline is left unspecified and its coefficients are estimated by *partial* likelihood, not full MLE.) These relationships are:

.. math::

	f(t|X) = \phi(X) h_{0}(t) e^{-\phi(X) H(t)} \\ 
	\\
	F(t|X) = 1 - e^{-\phi(X) H(t)} \\
	\\
	S(t|X) = e^{-\phi(X) H(t)}

It is therefore relatively simple to adjust the MLE methods to accommodate propotional hazard models.

The details on fitting proportional hazards model is detailed more in the SurPyval Modelling section.

Semi-Parametric
^^^^^^^^^^^^^^^

The previous sections covered 'parametric' and 'non-parametric' survival models, so what is 'semi-parametric'? A semi-parametric model is a survival model with a non-parametric baseline and parametric function that affects that baseline. Recall that a proportional hazard model can be defined as:

.. math::

	h(t|X) = \phi(X) h_{0}(t)

It is interesting to note that the phi term must be parametric, however, the baseline hazard rate need not be parametric, it can be non-parametric! Therefore, what we have is a parametric relationship of the covariates to the baseline hazard rate, but a non-parametric baseline hazard rate, therefore, a 'semi-parametric' model.

By far the most common of any regression model of any kind (parametric, non-parametric, and semi-parametric of all the accelerated life, proportinal hazard, and accelerated time) is the Cox Proportional Hazard model, it is a semi-parametric model.

The Cox model is used in a wide variety of fields. It has been used in criminology to study the recidivism of parolees, in engineering to understand the factors affecting tire reliability, and in medical science to understand factors affecting cancer and other diseases, among many many other applications. The wide use of the model shows the utility the model has and the broad applicability to solve problems.


Accelerated Failure Time
------------------------

An accelerated failure time (AFT) model is very similar to a proportional hazards model. The difference is where the function is applied; instead of multiplying the hazard function, an accelerated failure time model multiplies the time by the function of covariates. The general definition is:

.. math::

	f(t|X) = \phi(X)\, f_{0}(\phi(X)t)

It is called accelerated failure time since the time term is transformed by the covariates, i.e. time is 'accelerated' by the covariates.

.. math::

	t_{a} = \phi(X)t


Just like proportional hazards, there are simple transformations that apply. Note the density carries an extra :math:`\phi(X)` factor — the Jacobian of the time change of variables — while the survival and CDF do not:


.. math::

	f(t|X) = \phi(X)\, f_{0}(\phi(X)t) \\
	\\
	F(t|X) = F_{0}(\phi(X)t) \\
	\\
	S(t|X) = S_{0}(\phi(X)t)

Given the simple transformation of the time term the MLE is feasible with an additional transformation step. This is how surpyval estimates the parameters.

Accelerated Life
----------------

An accelerated life model is, in many cases, simply the inverse of an accelerated time model. However, there are some cases where they are different. Consider an accelerated life model with a normal distribution:

.. math::

	F(t|X) = \Phi\left(\frac{\phi(X)t - \mu}{\sigma}\right) \\

Where :math:`\Phi` is the CDF of the standard normal distribution. In this case :math:`\mu` is the expected life of the model, however, we may instead be interested in determining what effect covariates have on the expected life of an item. In this case we can simply substitute the expected life:

.. math::

	F(t|X) = \Phi\left(\frac{t - \phi(X)}{\sigma}\right) \\

An accelerated life model is, therefore, simply a model where the life parameter of a distribution is substituted with a function of the covariates, that is, it 'accelerates' the expected life, as opposed to accelerating time as per an accelerated time model. For each of the distributions in Surpyval their life parameter that varies is as per the following table:

+------------------+----------------+
| **Distribution** | **Life Param** |
+------------------+----------------+
| Weibull          | alpha          |
+------------------+----------------+
| Exponential      | 1./lambda      |
+------------------+----------------+
| Normal           | mu             |
+------------------+----------------+
| LogNormal        | mu             |
+------------------+----------------+
| Gamma            | alpha          |
+------------------+----------------+
| Gumbel           | mu             |
+------------------+----------------+
| Logistic         | mu             |
+------------------+----------------+
| LogLogistic      | alpha          |
+------------------+----------------+
| ExpoWeibull      | Not Avail      |
+------------------+----------------+
| Uniform          | Not Avail      |
+------------------+----------------+
| Beta             | Not Avail      |
+------------------+----------------+

Given the simple substitution into the life parameter, surpyval uses MLE to calculate the parameters.

Proportional Odds
-----------------

A proportional odds model acts on the *odds* of having failed rather than on the hazard. Writing the odds of failure by time :math:`t` as :math:`F(t) / S(t)`, the model multiplies the baseline odds by a function of the covariates:

.. math::

    \frac{F(t \mid X)}{S(t \mid X)} = \phi(X)\, \frac{F_{0}(t)}{S_{0}(t)}.

Its defining feature is that the covariate effect *decays* over time — two survival curves under a proportional odds model converge as :math:`t \to \infty` rather than staying a constant multiple apart. This makes it the natural choice when a treatment or covariate matters early but its influence fades, a pattern proportional hazards cannot represent.

Additive Hazards
----------------

Where proportional hazards *multiplies* the baseline hazard, an additive hazards model *adds* to it:

.. math::

    h(t \mid X) = h_{0}(t) + \beta \cdot X.

The covariate shifts the absolute hazard by a constant amount at every time, rather than scaling it. This is often the more natural scale for risk-difference questions (excess deaths per unit time attributable to an exposure), and, like Cox, the Lin-Ying form leaves the baseline hazard unspecified and admits a closed-form estimator for :math:`\beta`.

Semi-Parametric — Buckley-James
-------------------------------

Cox leaves the baseline *hazard* unspecified; Buckley-James is the accelerated-failure-time counterpart that leaves the *error distribution* unspecified. It fits an AFT model by iterating between imputing the censored failure times from the current fit (using the Kaplan-Meier residual distribution) and re-estimating the coefficients by least squares on the completed data. The result is a semi-parametric AFT: covariate effects on the log-time scale without committing to a parametric family for the baseline.

Checking a proportional hazards fit
-----------------------------------

Every proportional hazards model rests on one assumption: that a covariate multiplies the baseline hazard by a *constant* factor for all time. If that is false — a treatment that helps early but not late, a covariate whose effect drifts — the single coefficient the model reports is a time-average that can be misleading.

The assumption is checked with the **Schoenfeld residuals**. At each event time the Schoenfeld residual for a covariate is the observed covariate value of the subject who failed minus the risk-weighted mean covariate value over everyone still at risk. If proportional hazards holds, these residuals have no trend in time; if the effect is drifting, they trend. The **Grambsch-Therneau test** formalises this by regressing the *scaled* Schoenfeld residuals on a transform of time and testing for a non-zero slope, both per covariate and jointly. A small :math:`p`-value is evidence *against* proportional hazards.

SurPyval exposes several other residuals for a fitted Cox model, each answering a different question: **martingale** residuals (observed minus expected events) reveal non-linear covariate functional form; **deviance** residuals highlight poorly-predicted individuals; **score** and **dfbeta** residuals measure each observation's influence on the coefficients. The Schoenfeld, score and martingale residuals all sum to zero at the maximum of the partial likelihood.

Cluster-robust standard errors
------------------------------

The model-based standard errors assume every observation is independent. When the data are *clustered* — repeated events on the same subject, several failures from one machine, grouped sampling — that assumption is wrong and the naive errors are too small. The **Lin-Wei sandwich** (or "robust") variance corrects for it. Writing :math:`H` for the information matrix and :math:`s_c` for the sum of a cluster's score (dfbeta) contributions, the robust covariance is

.. math::

    V_{\text{robust}} = H^{-1} \left( \sum_{c} s_c s_c^{\top} \right) H^{-1},

which reduces to the usual variance when there is one observation per cluster and there is no within-cluster correlation.

Stratification
--------------

When proportional hazards fails for a *nuisance* covariate — a study site, a batch, a device generation you would rather not model — the standard remedy is **stratification**: allow a separate baseline hazard :math:`h_{0,g}(t)` for each stratum :math:`g` while sharing the coefficients :math:`\beta`. Because the partial likelihood is summed *within* strata, risk sets never cross a stratum boundary and the nuisance factor is removed from the comparison without ever estimating its effect. The Cox partial likelihood factorises across strata, so this is a small change to the estimation with a large gain in robustness.

Validating a survival predictor
-------------------------------

Information criteria (AIC, BIC) compare how well models fit the data they were trained on. To judge how well a model *predicts*, it must be scored on held-out data, and the metrics must account for censoring. Two right-censored-standard measures are used, both handling censoring by inverse-probability-of-censoring weighting (IPCW):

- The **Brier score** :math:`BS(t)` is the weighted mean squared error between the predicted survival :math:`S(t \mid Z)` and the survival indicator :math:`\mathbb{1}(T > t)`; the **integrated Brier score** averages it over a time grid. Lower is better, and a useful model scores below the marginal Kaplan-Meier reference.
- The **time-dependent AUC** (Uno's cumulative/dynamic estimator) measures discrimination as a function of the horizon — the probability that a subject who has failed by :math:`t` was assigned a higher risk than one still event-free. 0.5 is chance, 1.0 is perfect.

For worked examples on how to do regression analysis — including checking the proportional hazards assumption, robust and stratified fits, and validating predictions — see the :doc:`Regression Modelling with SurPyval` page.