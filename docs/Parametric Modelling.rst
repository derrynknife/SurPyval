
Parametric Modelling
====================

Parametric modelling is the process of estimating the parameters of a particular distribution from a set of data. This is distinct from non-parametric modelling where we make no assumptions about the shape of the distribution. In parametric modelling we make some assumptions, explicit or implied, about the shape of the data that we have.

For this segment I will use the Weibull distribution as the example distribution. The Weibull distribution is a very useful distribution for one interesting reason. It is the distribution for the 'weakest link.' As the normal distribution is the limiting distribution of averages, the Weibull distribution is the limiting distribution for minimums. What does that mean? If we have a large number of sets of samples from something that is normally distributed the average of these sets will also be normally distributed but the minimums of these sets of samples will be Weibull distributed. This is analagous to a chain. It is common wisdom that a chain is only as strong as it's weakest link. The Weibull distribution enables us to model the strength of a chain based on the strength of the links. This is extremely (lol) powerful.

The Weibull distribution can then be used in scenarios where we assume that the shape of the distribution will be due to a weakest link effect. This assumption holds in many scenarios, the strength of materials, the fielded life of equipment, the lifetime of animals, the time until another recession, or the time until germination of seeds. This example makes clear the assumption that we can make when using the Weibull distribution. Other distributions have differing processes that can result in their generation. If we know and understand these processes we can check them against the scenario we are analysing and chose a distribution from them. For example, a lognormal distribution can arise due to the combined effect of the product of random variables so in petroleum engineering the total recoverable oil is a product of the height, width, depth, features of the rock and an infinitude of other variables of the field. Therefore fields can be lognormally distributed. Similar considerations can be applied for many other types of distributions. Finally, If we don't know, or mind, what distribution we have, we can simply find the best fit amongst a set of distributions.

SurPyval offers users several methods for estimating parameters, these are:

- Maximum Likelihood Estimation (MLE)
- Minimum Product Spacing (MPS)
- Method of Probability Plotting (MPP)
- Mean Square Error (MSE)
- Method of Moments (MOM)

There are other methods that can be used, e.g. L-moments or generalised method of moments. These are interesting, and may be added in future, but for now surpyval offers the above estimation methods. Surpyval is unique in the capability to provide the estimation technique. Most other survival analysis methods do not allow for specifying different methods. The advantage of this flexibility will become apparent in the following discussion.

Maximum Likelihood Estimation (MLE)
-----------------------------------

Maximum Likelihood Estimation (MLE) is the most widely used, and most flexible of all the estimation methods. It's relative simplicity (because of modern computing power) makes it the reasonable first choice for parametric estimation. What does it do? Essentially MLE asks what parameters of a distribution are 'most likely' given the data that we have seen. Consider the following data and distributions:

.. image:: images/mle_1.png
	:align: center

The dashed lines represent the data we have observed, the red and blue lines are the density functions for two alternate distributions. Given the data and the two distributions, which one seems to explain the dsitribution of the data better? That is, which distribution is more likely to produce, if sampled, the black lines? It should be fairly obvious that the red distribution is more likely to do so. This is because the density function of the red distribution is higher around the black lines than is the blue distribution. MLE formalises this concept by saying that the most likely distribution is the one that has the highest (geometric) mean of the 'height' of the density function for each sample of data. The 'height' of the density at a particular observation is known as the likelihood. Mathematically, (for uncensored data) MLE then maximises the following:


.. math::

	L = {\left ( \prod_{i=1}^{n}f(x; \theta ) \right )}^{1/n} 

F is the pdf of the distribution being estimated, x is the observed value, theta is the parameters, and L is the geometric mean of all the values. This is complicated. But a simplification is available by taking the log of this product. Doing so yields:

.. math::

	l = { \frac{1}{n}}log\left ( \sum_{i=1}^{n}f(x; \theta ) \right )  

Therefore MLE simply finds the parameters of the distribution that maximise the average of the log of the density for each point... One final transform that is used in modern optimisers is that we take the negative of the above equation so that we find the minimum of the negative log-likelihood.

SurPyval defaults to using MLE because it is so powerful and so flexible. But it can be explicitly called using the :code:`how` argument with the three letter reference to the method:

.. code:: python

	import surpyval

	x = [4, 6, 8, 12, 15, 16, 17, 18]

	model = surpyval.Weibull.fit(x, how='MLE')


Given that MLE is so powerful and flexible, why would we bother with any other method? There are some cases where MLE does not succeed as well as other methods. In particular, with offset distributions.

.. code:: python

	import surpyval

	# Sample 100 values with alpha = 20, and beta = 0.8
	x = surv.Weibull.random(100, 20, 0.8)

	model = surpyval.Weibull.fit(x, how='MLE')
	print(model)

