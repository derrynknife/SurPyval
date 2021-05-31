
SurPyval Modelling
==================

Survival modelling with *surpyval* is very easy. This page will take you through a series of scenarios that can show you how to use the features of *surpyval* to get you the answers you need. The first example is if you simply have a list of event times and need to find the distribution of best fit.

.. code:: python

	import surpyval as surv
	import numpy as np
	from matplotlib import pyplot as plt

	np.random.seed(1)
	N = 100
	alpha = 10
	beta = 3
	x = surv.Weibull.random(50, 30., 9.)
	model = surv.Weibull.fit(x)
	print(model)
.. code:: raw

	Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters (29.805137406871953, 10.296037991991037)

To visualise the outcome of this fit we can inspect the results on a probability plot:

.. code:: python

	model.plot()

.. image:: images/surpyval-modelling-1.png
	:align: center

The :code:`model` object from the above example can be used to calculate the density of the distribution with the parameters found with the best fit from above. This is very easy to do:

.. code:: python

	x = np.linspace(10, 50, 1000)
	f = model.df(x)

	plt.plot(x, f)


.. image:: images/surpyval-modelling-2.png
	:align: center

The CDF :code:`ff()`, Survival (or Reliability) :code:`sf()`, hazard rate :code:`hf()`, or cumulative hazard rate :code:`Hf()` can be computed as well. This functionality makes it very easy to work with surpyval models to determine risks or to pass the function to other libraries to find optimal trade-offs. 

Censored Data
-------------

A common complication in survival analysis is that all the data is not observed up to the point of failure (or death). In this case the data is right censored, see the types of data section for a more detailed discussion, surpyval offers a very clean and easy way to model this. First, let's create a simulated data set:

.. code:: python

	np.random.seed(10)
	x = surv.Weibull.random(50, 30, 2.)

	observation_limit = 40
	# Censoring flag
	c = (x >= observation_limit).astype(int)
	x[x >= observation_limit] = observation_limit

In this example, we created 50 random Weibull distributed values with alpha = 30 and beta = 2. For this example the observation window has been set to 40. This value is where we stopped observing the events. For all the randomly generated values that are above this limit we create the censoring flag array c. This array has zeros where the event time was observed, and a 1 where the value is above the recorded value. For all the values in the data that are above 40 we set them to 40. This is a common occurence in survival analysis and surpyval is designed to accept this input with a simple call:


.. code:: python

	model = surv.Weibull.fit(x, c)
	print(model)
	model.plot()

.. code:: raw

	Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters (29.249243175047084, 2.2291485877428756)

The plot for this can be seen to be:

.. image:: images/surpyval-modelling-3.png
	:align: center

The results from this model are very close to the data we input, and with only 50 samples. This example can be extended to another kind of censoring; left censored data. This is the case where the values are known to fall below a particular value. We can change our example data set to have a start observation time for which we will left censor all the data below that:

.. code:: python

	observation_start = 10
	# Censoring flag
	c[x <= observation_start] = -1
	x[x <= observation_start] = observation_start

That is, we set the start of the observations at 10 and flag that all the values at or below this are left censored. We can then use the updated values of x and c:

.. code:: python

	model = surv.Weibull.fit(x, c)
	print(model)
	model.plot()

.. code:: raw

	Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters (29.347097662381277, 2.304902790957594)

The values did not substantially change, although the plot does look different as there are no values below 10.

.. image:: images/surpyval-modelling-4.png
	:align: center

The next type of censoring that is naturally handled by surpyval is interval censoring. Creating another example data set:

.. code:: python

	np.random.seed(30)
	x = surv.Weibull.random(50, 30, 10.)
	n, xx = np.histogram(x, bins=[20, 23, 26, 29, 32, 35, 38])
	x = np.vstack([xx[0:-1], xx[1::]]).T

In this example we have created the varable x with a matrix of the intervals within which each of the obervations have failed. That is each exact observation has been binned into a window and the x array has an entry [left, right] within which the event failed. We also have the n array that has the count of the failures within the window. With these two values we can make the simple surpyval call:


.. code:: python

	model = surv.Weibull.fit(x, n=n)
	print(model)

.. code:: raw

	Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters (30.074154903683105, 9.637405285678366)

Again, we have a result that is very close to the original parameters. SurPyval can take as input an arbitrary combination of censored data. If we plot the data we will see:

.. image:: images/surpyval-modelling-5.png
	:align: center

This does not look to be such a good fit. This is because the Turbull estimator finds the probability of failing in a window, not at a given point. So if we align the model plot to the end of the window instead of start with:

.. code:: python

	np_model = surv.Turnbull.fit(x, n=n)
	plt.step(np_model.x, np_model.R, where='post')
	x_plot = np.linspace(20, 37.5, 1000)
	plt.plot(x_plot, model.sf(x_plot), color='k', linestyle='dashed')

We get:

.. image:: images/surpyval-modelling-6.png
	:align: center


Which is, visually, clearly a better fit. You need to be careful when using the Turnbull plotting points to estimate the parameters of a distribution. This is because it is not known where in the intervals a death has actually occurred. However it is good to check the start and end of the window (changing 'where' betweek 'pre' and 'post' or 'mid') to see the goodness-of-fit.


Truncated Data
--------------

Surpyval has the capacity to handle arbitrary truncated data. A common occurence of this is in the insurance industry data. When customers make a claim on their policies they have to pay an 'excess' which is a charge to submit a claim for processing. If say, the excess on a set of policies in an area is $250, then it would not be logical for a customer to submit a claim for a loss of less than that number. Therefore there will be no claims under $250. This can also happen in engineering where a part may be tested up to some limit prior to be sold, therefore, as a customer you need to make sure you take into account the fact that some parts would have been rejected at the end of the line which you may not have seen. So a washing machine may run through 25 cycles prior to shipping. This is similar to, but distinct from censoring. When something is left censored, we know there was a failure or event below the threshold.  Whereas with truncation, we do not see any variables below the threshold. A simulated example may explain this better:

.. code:: python

	np.random.seed(10)
	x = surv.Weibull.random(100, alpha=100, beta=0.6)
	# Keep only those values greater than 250
	threshold = 25
	x = x[x > threshold]

We have therefore simulated a scenario where we have taken 100 random samples from a fat tailed Weibull distribution. We then filter to keep only those records that are above the threshold. In this case we assume we haven't seen the data for the washing machines with less than 25 cycles. To understand what could go wrong if we ignore this, what do we get if we assume all the data are failures and there is no truncation?

.. code:: python

	model = surv.Weibull.fit(x=x)
	print(model.params)

.. code:: raw

	(218.39245675499225, 1.050718601374874)

With a plot that looks like:

.. image:: images/surpyval-modelling-7.png
	:align: center


Looking at the parameters of the distribution, you can see that the beta value is greater than 1. Although only slightly, this implies that this distribution has an increasing hazard rate. If you were the operator of the washing machines (e.g. a hotel or a laundromat) and any downtime had a cost, you would conclude from this that replacing the machines after a fixed time would be a good policy.

But if you take the truncation into account:

.. code:: python

	model = surv.Weibull.fit(x=x, tl=threshold)
	print(model.params)

.. code:: raw

	(127.32704868357536, 0.7105357186212391)

With the plot:

.. image:: images/surpyval-modelling-8.png
	:align: center

You can see now that 


