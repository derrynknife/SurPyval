
Data Wrangling Examples
=======================


Lets just say we have a list of right censored data and a list of failures. How can we wrangle these into data for the :code:`fit()` method to accept?

.. code:: python

	import surpyval as surv

	# Failure data
	f = [2, 3, 4, 5, 6, 7, 8, 8, 9]
	# 'suspended' or right censored data
	s = [1, 2, 10]

	# convert to xcn format!
	x, c, n = surv.fs_to_xcn(f, s)
	print(x, c, n)

	model = surv.Weibull.fit(x, c, n)
	print(model)

.. code:: text

	[ 1  2  2  3  4  5  6  7  8  9 10] [1 0 1 0 0 0 0 0 0 0 1] [1 1 1 1 1 1 1 1 2 1 1]
	Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters (7.200723109183674, 2.474773882227539)


You can even bring in your left censored data as well:

.. code:: python

	# Failure data
	f = [2, 3, 4, 5, 6, 7, 8, 8, 9]
	# 'suspended' or right censored data
	s = [1, 2, 10]
	# left censored data
	l = [7, 8, 9]

	# convert to xcn format!
	x, c, n = surv.fsl_to_xcn(f, s, l)
	print(x, c, n)

	model = surv.Weibull.fit(x, c, n)
	print(model)

.. code:: text

	[ 1  2  2  3  4  5  6  7  7  8  8  9  9 10] [ 1  0  1  0  0  0  0 -1  0 -1  0 -1  0  1] [1 1 1 1 1 1 1 1 1 1 2 1 1 1]
	Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters (6.814750943874994, 2.4708983791967163)


Another common type of data that is provided is in a simple text list with "+" indicating that the observation was censored at that point. Using some simple python list comprehensions can help.

.. code:: python

	# Example provided data
	data = "1, 2, 3+, 5, 6, 8, 10, 3+, 5+"

	f = [float(x) for x in data.split(',') if "+" not in x]
	s = [float(x[0:-1]) for x in data.split(',') if "+" in x]

	data = surv.fs_to_xcn(f, s)

	model = surv.Weibull.fit(*data)


.. code:: text

	Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters (6.737537377506333, 1.9245506420162473)

Again, this can be extended to left censored data as well:

.. code:: python

	data = "1, 2, 3+, 5, 6, 8, 10, 3+, 5+, 15-, 16-, 17-"
	split_data = data.split(',')

	f = [float(x) for x in split_data if ("+" not in x) & ("-" not in x)]
	s = [float(x[0:-1]) for x in split_data if "+" in x]
	l = [float(x[0:-1]) for x in split_data if "-" in x]

	# Create the x, c, n data
	data = surv.fsl_to_xcn(f, s, l)

	model = surv.Weibull.fit(*data)

Surpyval also offers the ability to use a pandas DataFrame as an input. All you need to do is tell it which columns to look at for x, c, n, and t. Columns for c, n, and t are optional. Further, if you have interval censored data you can use the 'xl' and 'xr' column names instead. If you have mixed interval and observed or censored data, just make sure the value in the 'xl' column is the value of the observation or left or right censoring.


.. code:: python

	xr = [2, 4, 6, 8, 10]
	xl = [1, 2, 3, 4, 5]
	df = pd.DataFrame({'xl' : xl, 'xr' : xr})

	model = surv.Weibull.fit_from_df(df)
	print(model)


.. code:: text

	Parametric Surpyval model with Weibull distribution fitted by MLE yielding parameters (4.694329418712716, 2.4106930022962714)


