
Non-Parametric Estimation
=========================

Non-parametric survival analysis is the attempt to capture the distribution of survival data without making any assumptions about the shape of the distribution. That is, non-parametric analysis, unlike parametric analysis, does not assume that the survival data was Weibull distributed or that it was Normally distributed etc. Concretely, non-parametric estimation does not attempt to estimate the parameters of a distribution, therefore "non-parametric." Parametric analysis is covered in more detail in the section covering parametric estimation but it is important to contrast non-parametric estimation against what it is not. So what exactly is non-parametric analysis?

Survival analysis is using statistics to answer the question 'what is the probability that the thing survived to a particular time?' Non-parametric analysis answers this by estimating the probability from the proportion failed upto a given time. This can be done by either estimating the probability of surviving a particular segment or estimating the hazard rate.

Non-parametric estimation is well understood by appreciating the data format used to estimate the CDF. Specifically, the 'xrd' format and particularly understanding the r and d sets of that format.

The number of components at risk, r, at a given time, x, is the number of things at risk just prior to time x. The number of deaths, d, is the number of the at risk items that died (or failed) at time x. So for completely observed data the number at risk counts down for every death. So r would count down, e.g. 6, 5, 4, 3... for each death, 1, 1, 1, 1, ... So in this example there were 6 items at risk at one death at the first time. Then, because there was 1 death at the first time the number of items at risk has decreased to 5, therefore for the next death there are only 5 at risk. This continues further until there are no more items at risk because they have all died, i.e. there is 1 at risk and 1 death.

This can be extended to more than one death. For example, the risk set could be 8, 6, 5, 3, 2, 1. with an accompanying death set of 2, 1, 2, 1, 1, 1. In this example there were times where there were 2 deaths and therefore the number at risk decreased by 2 after that number of deaths.

So a complete example of this format is:

.. code:: python

	x = [1, 2, 3, 4, 5, 6]
	r = [7, 5, 4, 3, 2, 1]
	d = [2, 1, 1, 1, 1, 1]

This format for data is not how survival data is usually provided in text books or papers. Survival data is usually displayed with the simple list of failure times such as "1, 3, 6, 7, 10, 16". The first step surpyval does for non-parametric analysis is to transform data into the xrd format. All the :code:`fit()` methods for surpyval take as input the xcnt format, see more at the data types docs. So if you provide surpyval with the data "1, 2, 3, 4, 5, 6" it will assume that each of them are one death, and then create the risk set from the death counts resulting in the xrd format from above.

Given we now understand the format of the data we can estimate the probability of survival to some time with non-parametric methods. The first method we will visit is the Kaplan-Meier.

Kaplan-Meier Estimation
-----------------------

Kaplan-Meier [KM]_ is a very popular method for estimating survival curves for populations. The insight for this method is that for each time there is a death, we can estimate the probability of having survived since the previous deaths. Using the data from above as an example, at time 1, there are 7 items at risk and there are 2 deaths. We can therefore say that the probability of surviving this period was (7 - 2)/7, i.e. 5/7. Then the next time there is a death, the probability of having survived that extra time is (5 - 1)/5, i.e. 4/5.

To be clear, this is the chance of survival between each death. Therefore the chance of surviving up to a given time is the chance of surviving each segment. Therefore the probability of surviving up to any given time is the probability of surviving through all the previous segments. The probability of surviving multiple outcomes is the multiplication of each of the survival probabilities. Surviving through three sections is equal to the probability that I survive the first, then multiply this by the probability of surviving the second, then multiplying this result with the probability of surviving the third. So continuing our example from above, the probability of surviving the first two segments is (5/7) x (4/5) = 4/7.

Therefore using the at risk count, r, and the death count, d, can be used to estimate the segment survival probabilities and the survival probability to any point can be found by multiplying these probabilities. Formally, this has the following formula:

.. math::

   R(x) = \prod_{i:x_{i} \leq x}^{} \left ( 1 - \frac{d_{i} }{r_{i}}  \right )


Nelson-Aalen Estimation
-----------------------

The Nelson-Aalen estimator [NA]_ (also known as the Breslow estimator), instead of finding the probability, estimates the cumulative hazard function, and given that we know the relationship between the cumulative hazard function and the reliability function, the Nelson-Aalen cumulative hazard estimate can be converted to a survival curve.

The first step in computing the NA estimate is to convert your data to the x, r, d format. Once in this format the instantaneous hazard rate is found by:

.. math::

   h(x) = \frac{d_{x} }{r_{x}}

This estimate of the instantaneous hazard rate is the proportion of deaths/failures at a value, x. Then to find the cumulative hazard rate for any x we simply take the sum of the instantaneous hazard rates for all the values below x. Mathematically:

.. math::
   H(x) = \sum_{i:x_{i} \leq x}^{} \frac{d_{i} }{r_{i}}

Then, since we know that the reliability, or survival function, is related to the cumulative hazard function, we can easily compute it.

.. math::
   R(x) = e^{-H(x)}


So we now have the survival/reliability function. One benefit of the Nelson-Aalen estimator is that it does not estimate a probability of 0 for the highest value (in a completely observed data set). This means that for a completely observed data set the whole estimation can be plotted on a transformed y-axis. For this reason SurPyval uses the Nelson-Aalen as the default plotting position.


Fleming-Harrington Estimation
-----------------------------

The Fleming-Harrington estimator [FH]_, uses the same principal as the Nelson-Aalen estimator. That is, it finds the cumulative hazard function and then converts that to the reliability/survival estimate. However, the NA estimate assumes, for any given step that the number of items at risk is equal for each death, the FH estimate changes this. Mathematically, the hazard rate is calculated with:

.. math::

   h(x) = \frac{1}{r_{x}} + \frac{1}{r_{x} - 1} + + \frac{1}{r_{x} - 2} + ... + \frac{1}{r_{x} - d_{x}}

You can see that the cumulative hazard rate will be slightly higher than the NA estimate since:

.. math::

   \frac{1}{r_{x}} + ... + \frac{1}{r_{x}} \leq \frac{1}{r_{x}} + ... + \frac{1}{r_{x} - d_{x}}

The above is less than or equal for the case where there is one death/failure. The Fleming-Harrington and Nelson-Aalen estimates are particularly useful for small samples, see [FH]_.


Turnbull Estimation
-------------------

The Turnbull estimator is a remarkable non-parametric estimation method for data that can handle arbitrary censoring and truncation [TB]_. The Turnbull estimator can be found with a procedure of finding the most likely survival curve from the data, for that reason it is also known as the Non-Parametric Maximum Likelihood Estimator. The Kaplan-Meier is also known as the Maximum Likelihood estimator, so is there a contradiciton? No, the Turnbull estimator is the same as the Kaplan-Meier for fully observed data.

The Turnbull estimate

Confidence Intervals
--------------------

Right Censoring
---------------

Left Truncation
---------------

References
----------

.. [KM] Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American statistical association, 53(282), 457-481.

.. [NA] Nelson, Wayne (1969). Hazard plotting for incomplete failure data. Journal of Quality Technology, 1(1), 27-52.

.. [FH] Fleming, Thomas R and Harrington, David P (1984). Nonparametric estimation of the survival distribution in censored data. Communications in Statistics-Theory and Methods, 13(20), 2469-2486.

.. [TB] Turnbull, Bruce W (1976). The empirical distribution function with arbitrarily grouped, censored and truncated data. Journal of the Royal Statistical Society: Series B (Methodological), 38(3), 290-295.
