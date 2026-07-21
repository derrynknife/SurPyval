
Non-Parametric SurPyval Modelling
=================================

This page is the how-to companion to the :doc:`Non-Parametric Estimation` page,
which covers the concepts and mathematics behind the Kaplan-Meier,
Nelson-Aalen, Fleming-Harrington and Turnbull estimators.

To get started, let's import some useful packages, as such, for the rest of this page we will assume the following imports have occurred:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt


Survival modelling with *surpyval* is very easy. This page will take you through a series of scenarios that can show you how to use the features of *surpyval* to get you the answers you need. The first example is if you simply have a list of event times and need to find the distribution of best fit.


In each of the examples below, each of the ``KaplanMeier``, ``NelsonAalen``, or ``FlemingHarrington`` can be substituted with any of the others. It is the choice of the analyst which should be used. The 
``Turnbull`` estimator has additional capabilities that can be used when you have right truncated, left censored, or interval censored data.

Complete Data
-------------

Using data of the stress of Bofors steel from Weibull's original paper we can esimtate the reliability, that is, the probability that a sample of steel will survive up to a given applied stress. So what does that mean?

We can find when the steel will break. This is particularly useful when we know the application.

For this example, lets say that the maximum tensile stress our design will see during use is 34 units. Lets try and estimate the proportion that will fail during operation.

For this we can use the Nelson-Aalen estimator of the hazard rate, then convert it to the reliability. This is all done with one easy call.

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt

    x = np.array([32, 33, 34, 35, 36, 37, 38, 39, 40, 42])
    n = np.array([10, 33, 81, 161, 224, 289, 336, 369, 383, 389])

    # Weibull's measurements are cumulative so we need to tranasform them
    n = np.concatenate([[n[0]], np.diff(n)])

    bofors_steel_na = surv.NelsonAalen.fit(x, n=n)

    plt.figure(figsize=(10, 7));
    plt.ylabel('Survival Probability')
    plt.xlabel('Stress [1.275kg/mm2]')
    plt.ylim([0, 1])
    plt.xlim([31, 42])
    plt.step(bofors_steel_na.x, bofors_steel_na.R)
    plt.title('Survival Prob vs Stress of Bofors Steel');

So what purpose is this?

With our non-parametric model of the Bofors steel. We can use this model to estimate the reliability in our application. Let's say that our application uses Bofors steel up to 34. What is our estimate of the number of failures?

.. jupyter-execute::

    print(str(bofors_steel_na.sf(34).round(4).item() * 100) + "%")

The above shows that approximately 80% will survive up to a stress of 34. Therefore we will have an approximately 20% chance of our component failing in the design. 

It is up to the designer to determine whether this is acceptable.

What if we want to take into account our uncertainty about the reliability. The non-parametric class automatically computes the variance of the estimate using the formula appropriate to the estimator (Greenwood's formula for Kaplan-Meier, Aalen's variance for Nelson-Aalen, and the tie-corrected variance for Fleming-Harrington) and uses that to compute the upper and lower confidence intervals. Let's plot the intervals to see.

.. jupyter-execute::

    plt.figure(figsize=(10, 7))
    bofors_steel_na.plot(interp='linear')
    plt.xlabel('Stress [1.275kg/mm2]')
    plt.ylabel('Survival Probability')
    plt.ylim([0, 1])
    plt.xlim([32, 42])
    plt.title('Surv Prob vs Stress of Bofors Steel')


The confidence bounds can also be used to estimate the probability of survival up to some point with some degree of confidence. For example:

.. jupyter-execute::

    print(str(bofors_steel_na.R_cb(34, bound='lower', interp='linear', alpha_ci=0.05).round(4).item() * 100) + "%")

Therefore we can be 95% confident that the reliability at 34 is above 76%. For a Kaplan-Meier
model with no right censoring the variance at the final value is undefined with Greenwood's
formula, so the bounds at the last observation are filled with the last finite upper bound and
zero for the lower bound. The Nelson-Aalen and Fleming-Harrington variances remain finite at
the final value so their bounds are defined all the way to the last observation.


Right Censored Data
-------------------

Non-Parametric estimation can handle right censored, this is possible because at the point of censoring the item is removed from the at risk group without couting a death/failure.

.. jupyter-execute::

    import numpy as np
    from surpyval import KaplanMeier as KM

    x = np.array([3, 4, 5, 6, 10])
    c = np.array([0, 0, 0, 0, 1])
    n = np.array([1, 1, 1, 1, 5])

    model = KM.fit(x=x, c=c, n=n)
    model.R

.. jupyter-execute::

    model.plot()

In this example, we have included right censored data. This example can be done for the Nelson-Aalen,
Fleming-Harrington, and Turnbull estimators as well.

Left Truncated Data
-------------------

In some instances you will need to account for left truncated data. These data can be passed
stright to the same KM, NA, and FH fitters. A common source of left truncation is delayed
entry into a study: each subject's clock starts before they enrol, so anyone who failed
before they could enrol is never observed, biasing the sample towards longer survivors.
We can simulate such a cohort:

.. jupyter-execute::

    from surpyval import KaplanMeier as KM

    np.random.seed(10)
    lifetimes = surv.Weibull.random(1_000, 10, 2.5)
    entry = np.random.uniform(0, 10, 1_000)

    # Only subjects still alive at their entry time are ever enrolled
    enrolled = lifetimes > entry
    x = lifetimes[enrolled]
    tl = entry[enrolled]

    model = KM.fit(x=x, tl=tl)
    model_no_trunc = KM.fit(x=x)

    model.plot(plot_bounds=False)
    model_no_trunc.plot(plot_bounds=False)
    plt.legend(['Truncation', 'No Truncation'])


The image above shows that if you fail to take into account the left truncation (using the ``tl`` keyword)
you will overstate the survival probability. This can be used with any of the other non-parametric fitters.

Arbitrarily Truncated and Censored Data
---------------------------------------

In the event you have data that has interval, left, or right censoring with no, left, or right truncation, the previous estimators will not work. Enter the ``Turnbull`` estimator. First an interval
estimation example:


.. jupyter-execute::

    from surpyval import Turnbull as TB

    low = np.array([0, 0, 0, 4, 5, 5, 6, 7, 7, 11, 11, 15, 17, 17,
                    17, 18, 19, 18, 22, 24, 24, 25, 26, 27, 32, 33,
                    34, 36, 36, 36, 36, 37, 37, 37, 37, 38, 40, 45,
                    46, 46, 46, 46, 46, 46, 46, 46])
    upp = np.array([7, 8, 5, 11, 12, 11, 10, 16, 14, 15, 18, np.inf,
                    np.inf, 25, 25, np.inf, 35, 26, np.inf, np.inf,
                    np.inf, 37, 40, 34, np.inf, np.inf, np.inf, 44,
                    48, np.inf, np.inf, 44, np.inf, np.inf, np.inf,
                    np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                    np.inf, np.inf, np.inf, np.inf, np.inf])

    x = np.array([low, upp]).T
    model = TB.fit(x)
    model.plot()

And finally, an example with completely arbitrary censoring:


.. jupyter-execute::

    from surpyval import Turnbull as TB

    x = [1, 2, [3, 6], 7, 8, 9, [5, 9], [4, 10], [7, 10], 11, 12]
    c = [1, 1, 2, 0, 0, 0, 2, 2, 2, -1, 0]
    n = [1, 2, 1, 3, 2, 2, 1, 1, 2, 1, 1]

    model = TB.fit(x=x, c=c, n=n)
    model.plot()

With a completely arbitrary set of data we have created a non-parametric estimate of the survival
curve that can be used to estimate probabilities.

The Turnbull fitter also accepts truncation (``tl``/``tr``). Under
truncation the EM iterates with the Kaplan-Meier self-consistency update
(the canonical Turnbull M-step), and the requested hazard-form estimator
(Fleming-Harrington / Nelson-Aalen) is applied to the converged step
function -- so on well-sized samples the truncated estimate converges and
is well behaved.

Be aware, though, that the truncated NPMLE is a delicate object: on small
or heavily truncated samples it can be *non-identifiable* -- classically,
probability mass escapes below every observation's entry time and the
survival estimate collapses. SurPyval detects this degenerate fixed point,
raises a warning, and sets a ``degenerate`` flag on the model, rather than
silently returning an all-zero curve. It also warns if the EM stops before
converging. Treat the estimate with suspicion whenever either flag is
raised:

.. jupyter-execute::

    import warnings
    from surpyval import Turnbull as TB

    # A small, heavily truncated, mixed-censoring sample.
    x = [1, 2, [3, 6], 7, 8, 9, [5, 9], [4, 10], [7, 10], 11, 12]
    c = [1, 1, 2, 0, 0, 0, 2, 2, 2, -1, 0]
    n = [1, 2, 1, 3, 2, 2, 1, 1, 2, 1, 1]
    tl = [0, 0, 0, 0, 0, 2, 3, 3, 1, 1, 5]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = TB.fit(x=x, c=c, n=n, tl=tl)
    print('degenerate:', model.degenerate)

What is interesting about the Turbull estimate is that it first finds the data in the 'xrd' format.
This is done even though we might not have a complete failure occur in an interval. This can be seen by looking at the number of deaths/failures occur at each value.

.. jupyter-execute::

    model.d

You can see that some values are 0 (or essentially 0) or that there is an interval where there were
4.1639025 failures. But because the Turbull estimate finds the x, r, d format we can actually elect to use the Nelson-Aalen or Kaplan-Meier estimate with the Turnbull estimates of x, r, and d.

.. jupyter-execute::

    model = TB.fit(x=x, c=c, n=n, turnbull_estimator='Nelson-Aalen')
    model.plot()

The Greenwood confidence intervals do give us a strange set of bounds. But you can see that 
using the Nelson-Aalen estimator instead of the Kaplan-Meier gives us a better approximation 
for the tail end of the distribution.

Some Issues with the Turnbull Estimate
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Caution must be given when using the Turnbull estimate when all values are truncated by some left and/or
right value. This will be shown below in the methods for estimating parameters with truncated values. But
essentially the Turnbull method cannot make any assumptions about the probability by which the smallest
value if left truncated should be adjusted. This is because there is no information available with the
non-parametric method below this smallest value. The same is true for the largest value if it is also
right truncated, there is no information available about the probability of its observation. Therefore
the Turnbull method makes an implicit assumption that the first value, if left truncated has 100% chance
of observation, and the highest value, if right truncated also has 100% chance of being observed. 

The implications of this are detailed in the Parametric section, because the only way to gain an understanding of these situations is by assuming a shape of the distribution. That is, by doing parametric analysis. This is possible since if the distribution within the truncated ends has a shape that matches to a particular distribution you can then extrapolate beyond the observed values. Parametric analysis is therefore incredibly powerful for prediction / extrapolation.


Comparing two groups: the log-rank test
---------------------------------------

Having estimated a survival curve for each of several groups, the natural next
question is whether they *differ*. The **log-rank test** is the standard answer:
at every event time it compares the observed number of failures in each group
with the number expected if all groups shared one survival curve, and combines
those differences into a chi-squared statistic with ``k - 1`` degrees of freedom
(``k`` groups). A small ``p``-value is evidence the groups differ.

.. jupyter-execute::

    import numpy as np
    import surpyval as surv
    from surpyval import logrank

    np.random.seed(1)
    control = surv.Weibull.random(200, 10, 1.2)
    treatment = surv.Weibull.random(200, 16, 1.2)   # longer-lived
    x = np.concatenate([control, treatment])
    group = np.array(['control'] * 200 + ['treatment'] * 200)

    result = logrank(x, group)
    print(result)

The test accepts right-censored data through the ``c`` argument, and offers the
Gehan, Tarone-Ware and Fleming-Harrington weightings (via ``weighting=``) when
you want to emphasise early or late differences instead of the equal-weight
log-rank.

Stratified log-rank
~~~~~~~~~~~~~~~~~~~~~

When a *nuisance* factor influences survival — a study site, a batch — comparing
groups while ignoring it can be badly misleading if the groups are unevenly
distributed across its levels. The **stratified** log-rank accumulates the
observed-minus-expected counts *within* each stratum before forming the
statistic, so groups are only ever compared against others in the same stratum.
Pass a ``strata`` label per observation.

The example below is confounded on purpose: the baseline hazard differs sharply
by site, and the group is unevenly allocated across sites, but there is no true
group effect. The pooled test is fooled; the stratified test is not:

.. jupyter-execute::

    np.random.seed(2)
    n = 600
    site = np.random.randint(0, 2, n)
    group = np.where(site == 0, np.random.random(n) < 0.8,
                     np.random.random(n) < 0.2).astype(int)
    baseline = np.where(site == 0, 4.0, 20.0)
    x = np.random.exponential(baseline)              # no group effect

    print('pooled     p = %.4g' % logrank(x, group).p_value)
    print('stratified p = %.4g' % logrank(x, group, strata=site).p_value)

Restricted mean survival time
-----------------------------

A hazard ratio (from Cox or the log-rank test) is only interpretable when the
proportional-hazards assumption holds. When it does not — survival curves that
cross, treatments that help early but not late — the **restricted mean survival
time** (RMST) is an assumption-light alternative. It is simply the area under
the survival curve up to a horizon :math:`\tau`, i.e. the average event-free
time over the first :math:`\tau` units, and it is always well defined.

Any fitted non-parametric model exposes ``rmst(tau)``, returning the point
estimate with its standard error and confidence interval:

.. jupyter-execute::

    from surpyval import KaplanMeier

    control_model = KaplanMeier.fit(control)
    rmst = control_model.rmst(tau=20)
    print('RMST(20) = %.2f  (95%% CI %.2f - %.2f)'
          % (rmst['rmst'], rmst['lower'], rmst['upper']))

To compare two groups, ``surpyval.rmst_diff`` gives the difference in RMST with
a standard error, confidence interval and two-sided ``p``-value. The horizon
defaults to the smaller of the two groups' largest observed times (their common
support):

.. jupyter-execute::

    from surpyval import rmst_diff

    treatment_model = KaplanMeier.fit(treatment)
    diff = rmst_diff(treatment_model, control_model, tau=20)
    print('RMST difference = %.2f  (p = %.4g)'
          % (diff['difference'], diff['p_value']))

The treatment group spends about four more time units event-free over the first
twenty — a difference on the natural time scale, with no proportional-hazards
assumption required.
