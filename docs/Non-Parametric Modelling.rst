
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

   R(t) = \prod_{i:t_{i} \leq t}^{} \left ( 1 - \frac{d_{i} }{r_{i}}  \right )



Nelson-Aalen Estimation
-----------------------

The Nelson-Aalen estimator [NA]_, isntead of finding the probability, it estimates the cumulative hazard function, and given that we know the relationship between the cumulative hazard function and the reliability function, the Nelson-Aalen estiamte can be converted to a survival curve. The equations are:

.. math::

   H(t) = \sum_{i:t_{i} \leq t}^{} \frac{d_{i} }{r_{i}} \\
   \\
   R(t) = e^{-H(t)}










Fleming-Harrington Estimation
-----------------------------

Turnbull Estimation
-------------------

References
----------

.. [KM] Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American statistical association, 53(282), 457-481.

.. [NA] Nelson, Wayne (1969). Hazard plotting for incomplete failure data. Journal of Quality Technology, 1(1), 27-52.