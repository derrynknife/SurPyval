
Types of Data
=============

Survival analysis is the statistics about durations. To understand durations, or time to events, we must have data that captures how long something lasts. This is the start of survival analysis where we have data in the form of a list of durations of some time to event. This time to event can be engineering failure data, health data on time to death from a given disease, economic data on the duration of a recession or time between recessions, or it could be race times for a group of athletes in a triathlon.

Survival analysis is unique in statistics because of the types of data that we encounter, specifically censoring and truncation. The purpose of this section is to explain these types of data and the scenarios under which they are generated so that you can understand when you might need to use the different flags in surpyval in your analysis.

Throughout, the quantity being measured is called :math:`x` (and passed to SurPyval as ``x``). It is usually a time, but it does not have to be: it can be a distance, a number of cycles, a stress at which something broke, or a dollar value. What matters is that we are interested in the distribution of :math:`x`, and that the way we collected the data may have hidden some of the values from us. Censoring and truncation are two different ways in which values can be hidden, and they need different treatment:

- **Censoring**: we know an item exists, but we only know a *range* in which its value lies.
- **Truncation**: some items are missing from the data altogether, because their values fell outside the range in which we could have seen them.

Exactly Observed
----------------

The first type of data is exactly observed data. This is the type of data where we know exactly when the death or failure occurred. For example, if I run a test on how long some light bulbs will last, I get 5 and turn them on and watch them continuously. Then as each fail I record their failure times 983, 1321, 1889, 1923, and 2932 hours. Each of these times is exact because I saw the exact moment at which they failed. So the task is then understanding the distribution of these exact failures.

Exactly observed data is the default in SurPyval: if you pass only ``x``, every value is taken to be an exact observation.

.. jupyter-execute::

    import numpy as np
    import surpyval

    bulbs = [983, 1321, 1889, 1923, 2932]
    exact_model = surpyval.Weibull.fit(bulbs)
    print(exact_model.params)

Censored Data
-------------

Say that I got bored of sitting and looking at light bulbs. And because of this boredom I stopped looking at the light bulbs at 1900 hours. I would therefore not have seen two of the light bulbs fail, the failures that would have occurred at 1923 and 2932 hours. All we would know about these two light bulbs is that they failed sometime *after* 1,900 hours. That is, we know that these two light bulbs would have failed if they continued the test but that this failure time is greater than 1,900. This is to say that the failure time has been *censored*. Specifically the failure has been right censored. 'Right' is used because if we consider a (horizontal) timeline with time progressing along the line from left to right, we know that the failure would have occurred to the right of the time at which we stopped our observation. Hence, the observation is right censored.

If on the other hand, I also knew it would take some time for the bulbs to start failing so instead of waiting from the very start of the test I did not sit there for the first 1,000 hours. That is, the test continues to run but is not being observed for the first 1,000 hours, then after the first 1,000 hours I return to the test and start my observations. When I return I find that a bulb has failed. From the original data, I see that there was a bulb that failed at 983 hours. But if I was not observing for the first 1,000 hours all I would know about this failure is that it occurred sometime *before* 1,000 hours. Using the timeline concept again, I know that the failure would have occurred to the left of the 1,000 hour mark. Therefore, we say that the failure is *left* censored.

Finally, had I not been patient enough to sit down for any extended period of time and instead inspected the light bulbs at different times to see if any had failed. So say I inspect the bulbs every 100 hours from 1000 hours till 2,000 hours. The first and last failures would be left and right censored. But the middle failures would be known to fail between inspections. So the second failure would have occurred between the 1300 and 1400 hours inspections, the third between 1800 and 1900, and the second last failure would have happened between the 1900 and 2000 hours inspections. These failures are said to be *intervally* censored. That is because they are known to have happened in a given interval on a timeline.

In each case the item is *in* the data set: we know it exists and that it failed (or will fail), we just do not know exactly when. Censoring replaces an exact value with a range:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Type
     - What we know about the value :math:`X`
     - Typical cause
   * - Exactly observed
     - :math:`X = x`
     - continuous observation
   * - Right censored
     - :math:`X > x`
     - the study ended, or the item was withdrawn, before it failed
   * - Left censored
     - :math:`X \leq x`
     - the failure had already happened at the first look
   * - Interval censored
     - :math:`x_l < X \leq x_r`
     - failures found at periodic inspections

Survival analysis has several methods for handling censored data in the parametric and non-parametric analysis. Surpyval is able to handle an input that has an arbitrary combination of observed and left, right, and intervally censored failure data, although not all methods can handle all types of data. This is covered in the sections on each of the estimation and fitting methods, and summarised at the end of this page.

Flagging censoring in SurPyval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Surpyval uses a convention regarding censoring. Specifically, surpyval takes as input, with a list of failure times 'x', an optional censoring flag array 'c'. If no flagging array is provided, it is assumed that all the data are exact observations, i.e. that they are not censored. But if the 'c' array is provided, it must have a value for each value in the x input. That is, they must be the same length. The possible values of c are -1, 0, 1, and 2. The convention tries to illustrate the concept of left, right, and interval censoring on the timeline. An observed failure is 0, the centre of the timeline. -1 is the flag for left censoring because the failure lies to the *left* of the recorded value. 1 is used to flag a value as right censored, because the failure lies to the *right*. Finally, 2 is used to flag a value as being intervally censored because it has 2 data points, a left and right point. In practice this will therefore look like:

.. jupyter-execute::

    x = [3, 3, 3, 4, 4, [4, 6], [6, 8], 8]
    c = [-1, -1, -1, 0, 0, 2, 2, 1]

    model = surpyval.Weibull.fit(x=x, c=c)

Applying the flags to the inspected light bulbs from above: the bulb found failed at the first inspection is left censored at 1000, the three bulbs found failed between inspections are interval censored, and the bulb still working at the last inspection is right censored at 2000.

.. jupyter-execute::

    x = [1000, [1300, 1400], [1800, 1900], [1900, 2000], 2000]
    c = [-1, 2, 2, 2, 1]

    inspected_model = surpyval.Weibull.fit(x=x, c=c)
    print("exact failure times :", exact_model.params)
    print("inspection data     :", inspected_model.params)

The two fits differ, and they should: the inspection data carries less information than the exact failure times (with five bulbs, a lot less). The point is that the censored fit uses exactly the information that was collected, no more and no less.

A few rules make the flags unambiguous:

- An interval censored row (``c = 2``) must have two values, ``[lower, upper]``, with ``lower < upper``. Every other row has a single value. If ``x`` is given as two columns, a row that is not an interval repeats its value, ``[v, v]``, and its flag says whether it is observed, left censored or right censored (flagging such a row ``2`` is an error).
- An interval with an infinite end is really a one-sided censoring, and SurPyval converts it: ``[v, inf]`` becomes right censored at ``v`` and ``[-inf, v]`` becomes left censored at ``v``. This is convenient when data comes as "last seen working" and "first seen failed" columns (see :doc:`Data Wrangler Examples`).
- If ``c`` is not given but ``x`` has two columns, each row with different values is flagged as interval censored and each row with equal values as observed.

.. jupyter-execute::

    x, c, n, t = surpyval.xcnt_handler(
        x=[[1000, np.inf], [-np.inf, 1000], [1300, 1400], [1200, 1200]]
    )
    print(x)
    print(c)

The rows come back sorted, which is how SurPyval stores data internally: the first two rows are the converted one-sided intervals (left censored, ``-1``, and right censored, ``1``, both at 1000), followed by the exact observation at 1200 (``0``) and the interval (``2``).

The same data can be given as two separate arrays with ``xl`` and ``xr`` in place of ``x``, which is often more natural when every row is an interval.

Condensing repeated values with counts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example shows the flexibility surpyval offers. It allows users to analyse data that has any arbitrary combination of the different types of censoring. The surpyval format is even more powerful, because the first example above can be condensed even further through using the 'n' value, the number of items that share the same value and censoring flag.

.. jupyter-execute::

    x = [3, 4, [4, 6], [6, 8], 8]
    c = [-1, 0, 2, 2, 1]
    n = [3, 2, 1, 1, 1]

    model = surpyval.Weibull.fit(x=x, c=c, n=n)

The first step of the fit method actually wrangles the input data into the densest form possible. So internally, the example without the n value, will be condensed to be the second example without you seeing it. But it shows the capability of how data can be input to surpyval if you have different formats. Counts must be positive integers: ``n`` is a number of items, not a weight.

Pitfalls with censored data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Censoring must not carry information about the failure time.** Every method in SurPyval assumes that the reason an item was censored is unrelated to when it would have failed (*non-informative* censoring). Stopping a test at a fixed time, or at a fixed number of failures, is fine. Withdrawing units *because* they look about to fail is not: those units are systematically the weak ones, and treating them as ordinary survivors makes the population look more reliable than it is.
- **Do not drop or "fill in" censored values.** Dropping the survivors, or treating a censoring time as if it were a failure, biases life estimates downwards (the :doc:`Quickstart` shows an example). Pass the censored values with their flags instead.
- **There must be some failures.** A data set in which every value is right censored (or every value is left censored) cannot identify a distribution, and SurPyval raises an error. More generally, a parametric fit needs at least as many distinct non-right-censored values as it has free parameters to estimate, otherwise the answer would be wherever the optimiser happened to stop; SurPyval rejects such fits with an explanation. Fixing a parameter (``fixed={...}``) buys back a degree of freedom.


Truncated Data
--------------

For my light bulb test, let's say I test a different manufacturers bulbs. This time, I know that the bulbs from this manufacturer have been tested for 500 hours prior to shipping them. This situation needs to be treated differently because we know that in this circumstance we only have the bulbs because they survived more than 500 hours. If there were any failures prior to 500 hours the bulb would not have been shipped and therefore would not be being tested by me. This is to say, that my observation of the distribution of the light bulb failures has been *truncated*. In this regime there is no way I can have any observation below 500 hours because of the testing then discarding done by the manufacturer. The astute reader might have observed that this data is in fact *left* truncated. This is because the truncation occurs to the left of the observation on a timeline. In this example, all the bulbs are left truncated at the 500 hour mark.

The difference from censoring is worth dwelling on. A censored bulb is in my data set, with partial information about its life. A bulb that failed in the manufacturer's burn-in is simply *not in my data set at all*; I do not even know how many there were. So truncation does not change what we know about any single value, it changes which values could have reached us. Mathematically, an observation left truncated at :math:`t_l` does not follow the density :math:`f(x)` but the *conditional* density given that it survived past :math:`t_l`:

.. math::

    f(x \mid X > t_l) = \frac{f(x)}{R(t_l)}, \qquad x > t_l

and, in general, an observation that could only have been seen inside a window :math:`(t_l, t_r]` follows :math:`f(x) / \left(F(t_r) - F(t_l)\right)`. Dividing by the probability of the window is what corrects for the values we never had the chance to see.

In biostatistics left truncation is known as 'late-entry', this is because in clinical trials a participant can enter a trial later than other participants. Therefore this participant was at risk of not being present in the trial. This is because they could have died prior to entering the trial. Morbid, yes, but the estimate of the distribution needs to account for this risk otherwise the estimate will overestimate the true survival: everyone who enters late has, by construction, already survived to their entry time.

Right truncated data is when you only observe a value because it happened below some time. For example, in the light bulb experiment, I received some of the bulbs that passed the burn in test. That is, I received some of the bulbs that survived the original 500 hours of testing. But if the failed bulbs were then given to an engineering team to investigate possible design changes that will improve reliability; they will have a series of failure times that must be below 500 hours. That is, from their perspective, they have data that is right truncated. There is one condition to this situation, they must not know how many other bulbs were tested. If they knew how many other bulbs were tested, they would know how many would fail after 500 hours. That is, they would know that all the other bulbs are right censored. So for our engineers investigating the failed bulbs, they must be ignorant of how many other bulbs were actually tested for the right truncation to work for them. In many applications we do know how many were under test and therefore right truncation becomes right censoring, but from our engineers' circumstance, we can see that their data is right truncated.

Parametric and non-parametric analysis can both handle left truncated data. This is explained further in the estimation methods for both these methods. Right truncation can be handled in surpyval with parametric analysis, with Maximum Likelihood Estimation, with Maximum Product Spacing when every observation shares the same truncation value, and with probability plotting using the Turnbull heuristic; non-parametrically it is handled by the Turnbull estimator. This is also explained in their respective sections of these notes.

In surpyval, passing truncated data to the fitting method looks like:

.. jupyter-execute::

    x  = [674, 792, 1153, 1450, 1555, 1923, 2019]
    tl = 500

    model = surpyval.Weibull.fit(x=x, tl=tl)

Truncation values can be given in three ways:

- ``tl`` and/or ``tr`` as a **scalar**: every observation shares the same bound, as with the burn-in above.
- ``tl`` and/or ``tr`` as an **array** the same length as ``x``: each observation has its own bound. This is the late-entry case, where each person or unit came under observation at a different age.
- ``t`` as a **two-column array** of ``[tl, tr]`` rows. ``t`` cannot be combined with ``tl`` or ``tr``.

A missing bound means no truncation on that side: internally it is stored as :math:`-\infty` (left) or :math:`+\infty` (right).

What goes wrong if truncation is ignored
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To see why truncation matters, simulate bulbs whose true life is Weibull with :math:`\alpha = 1500` and :math:`\beta = 2`, and keep only those that survived the 500 hour burn-in:

.. jupyter-execute::

    rng = np.random.default_rng(4)
    life = 1500 * rng.weibull(2.0, 400)
    shipped = life[life > 500]              # the burn-in failures never reach us

    print("ignoring truncation :", surpyval.Weibull.fit(shipped).params)
    print("with tl=500         :", surpyval.Weibull.fit(shipped, tl=500).params)

Ignoring the truncation, the fit only ever sees bulbs that were strong enough to pass burn-in, so it overestimates the characteristic life and makes the failures look more tightly bunched (a larger :math:`\beta`) than they really are. Telling SurPyval about the burn-in with ``tl=500`` recovers estimates close to the true values.

Rules and conventions for truncation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **A value must lie strictly above its left truncation.** An item that entered observation at :math:`t_l` can only be seen to fail *after* :math:`t_l`, so SurPyval rejects a value equal to its own left truncation bound (it would have a zero-length observation window). Right truncation must be at or above the value.
- **Risk sets use the (entry, exit] convention.** For the non-parametric estimators, an item with left truncation :math:`t_l` and value :math:`x` is counted in the risk set at every time :math:`s` with :math:`t_l < s \le x`. An item entering at exactly the time of someone else's failure is therefore *not* at risk for that failure. This is the convention used by R's ``survival`` package and by lifelines.
- **Censoring and truncation combine.** A right censored item that was also right truncated at :math:`t_r` is known to have failed in :math:`(x, t_r]`, and a left censored item that was also left truncated at :math:`t_l` failed in :math:`(t_l, x]`. SurPyval uses these intervals in the likelihood automatically, so no special handling is needed on your part.

.. jupyter-execute::

    try:
        surpyval.Weibull.fit(x=[500, 674, 792], tl=500)
    except ValueError as err:
        print(err)


Concluding Points
-----------------

Having read through the above explanation you might be thinking how often these scenarios appear in real data, if ever. The vast majority of data used in survival analysis is observed or right censored. This is what happens when you observe a whole population but finish the observation before the event happens on all the items being observed.

Right truncation is extremely rare because it only happens if you do not know the size of the whole population under test. It can happen with scientific instruments where say, a camera is limited in the frequencies of light it can capture. So if we were to try capture a distribution of light of an object, say a star, this distribution could be truncated above and below certain frequencies. Meeker and Escobar provide an example in their book on reliability statistics for warranty analysis, similar to the contrived example provided above. If you have some returns of products from the field, these are right-truncated because you do not know what has been bought and used in the field. A more realistic example could be the estimation of race finish times at a triathlon or marathon. If I arrive at the finish line of a race and record the times of participants as they cross the line during that window I will have truncated data. I do not know how many people started the race (presumably) and I only stay and watch for a given period of time, therefore all the observations I make are truncated within the window of my observation time. In conclusion though, right truncation in survival analysis is rare.

Left truncation is common in insurance studies. If an insurance company wants to estimate the distribution of losses due to property crime based on policy payouts they need to consider the impact of 'excess'. Excess is the cost of making a claim on an insurance policy. So if I have an insurance policy with an excess of $500, if I lose $20,000 worth of property in a robbery I will have to pay $500 to be paid $20,000. Because of this, it is clear that if I lost $400 in a robbery I would not pay the $500 excess to make a claim. Therefore the distribution of property crime will be truncated by the value of the excesses on the policies. Actuaries need to consider this in their calculations of policy fees.

Insurance is also a good example of right censoring. An insurance policy will also have a maximum payout. So if calculating the distribution of the value of property crime an analyst will need to consider that those payouts that are at the maximum of that policy value are in fact censored. That is, the value of the loss or damage was greater than the actual payout and therefore the payout is a censored value. In the classic Boston housing pricing data there is censored data! A histogram of the values of houses shows that there is a large number of houses at the highest price. This can be understood because a limit was set on the highest possible value, therefore these house prices are actually censored, not exact observations. This example is worked through in :doc:`applications`.

A checklist for identifying the type of each value:

1. Do I know the value exactly? Then it is observed (``c = 0``).
2. Do I know only a range for it? Then it is censored: right (``c = 1``) if I only know a lower limit, left (``c = -1``) if I only know an upper limit, interval (``c = 2``) if I know both.
3. Could items with values outside some range have failed to reach my data set at all, without my knowing how many? Then the data is truncated at the limits of that range (``tl``, ``tr``). A value can be both censored and truncated.

Which methods handle which data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Not every estimator can use every type of data. For the univariate estimators:

.. list-table::
   :header-rows: 1

   * - Method
     - Right censored
     - Left censored
     - Interval censored
     - Left truncated
     - Right truncated
   * - Kaplan-Meier, Nelson-Aalen, Fleming-Harrington
     - Yes
     - No
     - No
     - Yes
     - No
   * - Turnbull
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Maximum Likelihood (MLE)
     - Yes
     - Yes
     - Yes
     - Yes
     - Yes
   * - Maximum Product Spacing (MPS)
     - Yes
     - Yes
     - No
     - One value for all
     - One value for all
   * - Probability Plotting (MPP)
     - Yes
     - With ``heuristic="Turnbull"``
     - With ``heuristic="Turnbull"``
     - Yes, with the (default) Nelson-Aalen, Kaplan-Meier, Fleming-Harrington or Turnbull heuristic
     - With ``heuristic="Turnbull"``
   * - Mean Square Error (MSE)
     - Yes
     - Yes
     - Yes
     - No
     - No
   * - Method of Moments (MOM)
     - No
     - No
     - No
     - No
     - No

When in doubt, use Turnbull for a non-parametric view and MLE (the default ``how``) for a parametric fit: between them they handle every combination. Where a method cannot handle your data, SurPyval raises an error saying so rather than silently ignoring part of the data.
