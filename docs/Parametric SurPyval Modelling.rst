Parametric SurPyval Modelling
=============================

The parametric API is essentially the exact same as the non-parametric API. All models are fit by a
call to the ``fit()`` method. However, the parametric models have more options that are only applicable to parametric modelling. The inputs of ``x`` for the random variable, ``c`` for the censoring flag, ``n``
for count of each ``x``, ``xl`` and ``xr`` for intervally censored data (can't be used with ``x``) ``t``
for the truncation matrix, ``tl`` for the left truncation scalar or array, and ``tr`` for the right truncation scalar or array all remain.

These ingredients compose freely — any mix of censoring, counts and truncation
in a single ``fit`` call — and the same convention is used by every model in
the package (non-parametric, regression, recurrent, competing-risks and
copula). See :doc:`Data Wrangler Examples` for worked examples that combine
them and convert between input formats.

On top of the data, ``fit`` takes the options that describe the *model* and
how to estimate it. Each is demonstrated below:

- ``how``: the estimation method, one of ``'MLE'`` (the default), ``'MPS'``,
  ``'MSE'``, ``'MPP'`` or ``'MOM'``;
- ``offset=True``: add a threshold (shift) parameter ``gamma``;
- ``lfp=True``: a limited failure population, where only a proportion ``p``
  can ever fail;
- ``zi=True``: zero inflation, where a proportion ``f0`` fails at time zero;
- ``fixed``: a dictionary of parameters to hold at known values;
- ``init``: a starting point for the optimiser;
- ``heuristic``, ``rr``, ``on_d_is_0`` and ``turnbull_estimator``: options for
  probability plotting.

What each estimation method optimises, and why each accepts the data it does,
is explained in :doc:`Parametric Estimation`. The API reference for every
distribution is listed in :doc:`surpyval.parametric`, and the fitted model
object is documented in :doc:`Parametric model API <univariate/parametric_class>`.

Available distributions
-----------------------

Every distribution is an object in the ``surpyval`` namespace
(``surv.Weibull``, ``surv.Gamma``, ...) with the same ``fit``,
``from_params`` and distribution functions. The continuous ones are:

.. list-table::
   :header-rows: 1
   :widths: 18 22 14 16 10 20

   * - Distribution
     - Parameters (``param_names``)
     - Support
     - ``offset`` / ``zi``
     - ``how='MPP'``
     - Solved in closed form
   * - ``Exponential``
     - ``failure_rate``
     - :math:`(0, \infty)`
     - yes / yes
     - yes
     - MLE (exact and right-censored data, left truncation)
   * - ``Weibull``
     - ``alpha``, ``beta``
     - :math:`(0, \infty)`
     - yes / yes
     - yes
     -
   * - ``ExpoWeibull``
     - ``alpha``, ``beta``, ``mu``
     - :math:`(0, \infty)`
     - yes / yes
     - no
     -
   * - ``Gamma``
     - ``alpha``, ``beta``
     - :math:`(0, \infty)`
     - yes / yes
     - no
     -
   * - ``LogNormal`` (also ``Galton``)
     - ``mu``, ``sigma``
     - :math:`(0, \infty)`
     - yes / yes
     - yes
     - MLE (complete data); MOM
   * - ``LogLogistic``
     - ``alpha``, ``beta``
     - :math:`(0, \infty)`
     - yes / yes
     - yes
     -
   * - ``Rayleigh``
     - ``sigma``
     - :math:`(0, \infty)`
     - yes / yes
     - yes
     -
   * - ``Normal`` (also ``Gauss``)
     - ``mu``, ``sigma``
     - :math:`(-\infty, \infty)`
     - no / no
     - yes
     - MLE (complete data)
   * - ``Gumbel`` (smallest extreme value)
     - ``mu``, ``sigma``
     - :math:`(-\infty, \infty)`
     - no / no
     - yes
     -
   * - ``GumbelLEV`` (largest extreme value)
     - ``mu``, ``sigma``
     - :math:`(-\infty, \infty)`
     - no / no
     - yes
     -
   * - ``Logistic``
     - ``mu``, ``sigma``
     - :math:`(-\infty, \infty)`
     - no / no
     - yes
     -
   * - ``Uniform``
     - ``a``, ``b``
     - :math:`[a, b]`
     - no / no
     - yes
     - MLE; MOM
   * - ``Beta``
     - ``alpha``, ``beta``
     - :math:`[0, 1]`
     - no / yes
     - no
     - MOM
   * - ``Beta4``
     - ``alpha``, ``beta``, ``a``, ``b``
     - :math:`[a, b]`
     - no / no
     - no
     -
   * - ``Hypoexponential``
     - ``lambda_1``, ..., ``lambda_m``
     - :math:`(0, \infty)`
     - built with ``from_params`` only
     -
     -
   * - ``CustomDistribution``
     - your own
     - your own
     - if the support is :math:`(0, \infty)`
     - no
     -

``Galton`` and ``Gauss`` are the same distributions as ``LogNormal`` and
``Normal`` under their other names. Every distribution in the table except the
``Hypoexponential`` also accepts ``lfp=True``. The discrete lifetimes
(``Geometric``, ``DiscreteWeibull``, ``NegativeBinomial``, ``BetaGeometric``,
``Discretize`` and ``Poisson``) and the per-demand and degenerate models
(``Bernoulli``, ``FixedEventProbability``, ``Binomial``, ``ExactEventTime``,
``InstantlyOccurs`` and ``NeverOccurs``) have their own sections below, and the
flexible Royston-Parmar model and mixtures of any of these have theirs.

Complete Data
-------------

The easiest and simplest case is that when you have a dataset of exactly observed data. that is,
you have one array of data with the values at which they failed. Fitting a parametric distribution
to the data can be done with a simple call to the ``fit()`` method:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(10)
    x = surv.Weibull.random(50, 30., 9.)
    model = surv.Weibull.fit(x)
    model

To visualise the outcome of this fit we can inspect the results on a probability plot:

.. jupyter-execute::

    model.plot()

The :code:`model` object from the above example can be used to calculate the density of the distribution with the parameters found with the best fit from above. This is very easy to do:

.. jupyter-execute::

    from matplotlib import pyplot as plt

    x_plot = np.linspace(10, 50, 1000)
    f = model.df(x_plot)
    plt.plot(x_plot, f)

The CDF :code:`ff()`, Survival (or Reliability) :code:`sf()`, hazard
rate :code:`hf()`, or cumulative hazard rate :code:`Hf()` can be computed as
well. This functionality makes it very easy to work with surpyval models to
determine risks or to pass the function to other libraries to find optimal
trade-offs.

Working with a fitted model
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The fitted parameters are in ``model.params``, in the order given by the
distribution's ``param_names``, and each is also available by name:

.. jupyter-execute::

    print(model.dist.param_names, model.params)
    print("alpha =", model.alpha, " beta =", model.beta)

Every function of the distribution is a method of the model. As well as the
five functions above there is the quantile function ``qf`` (the inverse of
the CDF, so ``model.qf(0.1)`` is the "B10 life" by which 10% have failed), the
conditional survival ``cs(x, X)`` (the probability of surviving a further
``x`` given survival to ``X``), and the summary statistics:

.. jupyter-execute::

    t = np.array([20., 30., 40.])
    print("R(t)   :", model.sf(t))
    print("F(t)   :", model.ff(t))
    print("h(t)   :", model.hf(t))
    print("H(t)   :", model.Hf(t))
    print("B10    :", model.qf(0.1))
    print("median :", model.qf(0.5))
    print("P(survive 5 more | survived 25):", model.cs(5, 25))
    print("mean, variance :", model.mean(), model.var())
    print("E[X^2], entropy:", model.moment(2), model.entropy())

The model also records how it was made: the estimation method, the optimiser
that converged (see :doc:`Parametric Estimation`), the support, and, for a
maximum likelihood fit, the parameter covariance ``hess_inv`` whose diagonal
holds the squared standard errors:

.. jupyter-execute::

    print("fitted by :", model.method, "using", model.optimizer)
    print("support   :", model.support)
    print("std errors:", np.sqrt(np.diag(model.hess_inv)))

The distributions can also be used directly, without a model, by passing
the parameters after ``x``. This is handy for a quick calculation:

.. jupyter-execute::

    surv.Weibull.sf(t, 30., 9.)

Models from known parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not every model comes from data. A supplier's datasheet, a handbook value or
an earlier analysis may give you the parameters, and ``from_params`` builds a
full model from them, with all of the methods above. It also takes the
structural options as values: ``gamma`` for an offset, ``p`` for a limited
failure population and ``f0`` for zero inflation (each explained below).

.. jupyter-execute::

    known = surv.Weibull.from_params([30., 9.])
    print("R(25) =", known.sf(25.))

    shifted = surv.Weibull.from_params([10., 2.], gamma=5.)
    print(shifted)
    print("R at 4, 5 and 10:", shifted.sf([4., 5., 10.]))

Nothing can fail before the offset, so the survival of the shifted model is
exactly one up to ``gamma = 5``.

Some distributions exist *only* in this form. The ``Hypoexponential`` is the
lifetime of something that must pass through several independent,
memoryless stages in turn: a load-sharing group whose failure rate changes as
members fail, or a warm-standby system. Each stage lasts an exponential time
with its own rate, and the lifetime is their sum. Because the number of
stages is up to you, it takes any number of rates, and there is no ``fit``:
you construct it from rates you know (see :doc:`univariate/hypoexponential`).

.. jupyter-execute::

    from surpyval import Hypoexponential

    # Three stages with rates 0.5, 1.5 and 3.0 per unit time
    standby = Hypoexponential.from_params([0.5, 1.5, 3.0])
    print(standby)
    print("mean:", standby.mean())
    print("R at 1, 2 and 5:", standby.sf([1., 2., 5.]))

The mean is the sum of the stage means, :math:`1/0.5 + 1/1.5 + 1/3 = 3`, as
it should be. The rates must be distinct: as two rates approach each other the
closed form becomes numerically unstable, and SurPyval raises an error. When
every stage has the same rate the sum is an Erlang distribution, which is a
``Gamma`` with an integer shape:

.. jupyter-execute::

    erlang = surv.Gamma.from_params([3, 1.0])   # three stages, each with rate 1
    print("R at 1, 2 and 5:", erlang.sf([1., 2., 5.]))

Random samples
^^^^^^^^^^^^^^

Random samples are drawn with ``random``, either from a distribution with
given parameters or from a model. They are useful for simulation studies,
for testing an analysis on data where you know the answer (as the examples
on this page do), and for Monte Carlo propagation of risk. Seed numpy's
generator to make them repeatable. A model can also be sampled *truncated*,
between ``a`` and ``b``:

.. jupyter-execute::

    np.random.seed(1)
    print(surv.Weibull.random(5, 30., 9.))   # from the distribution
    print(model.random(5))                   # from a fitted model
    print(model.random(5, a=25, b=30))       # only values between 25 and 30

A model with a limited failure population returns its sample as
``(x, c, n, t)`` arrays instead of a single array, because some of the drawn
units never fail and have to be recorded as right censored. The limited
failure population section below uses this.

Saving and loading a model
^^^^^^^^^^^^^^^^^^^^^^^^^^

A fitted model can be stored and restored with ``to_dict`` and
``surpyval.from_dict``, or written to a JSON file with ``to_json`` and read
back with ``surpyval.from_json``. The package-level readers work out which
kind of model wrote the file, so the same call restores a Weibull, a mixture,
a Royston-Parmar model or any other SurPyval model (see
:doc:`surpyval.serialisation`).

.. jupyter-execute::

    import os
    import tempfile

    restored = surv.from_dict(model.to_dict())
    print(restored.sf(25.), model.sf(25.))

    path = os.path.join(tempfile.mkdtemp(), "weibull.json")
    model.to_json(path)
    print(surv.from_json(path).params)

The dictionary holds the parameters, their covariance and the fitted
negative log-likelihood but, by default, not the data. So a restored model can
still give Wald confidence bounds, ``neg_ll()`` and ``aic()``, but not the
criteria that need the sample size (``bic()``, ``aic_c()``), ``plot()``, or
likelihood-ratio bounds -- each says so if asked. Pass ``with_data=True`` to
``to_dict`` to keep the data, which restores ``plot()`` and every information
criterion; for likelihood-ratio bounds, refit.

Using censored data
-------------------

Right Censored
^^^^^^^^^^^^^^

A common complication in survival analysis is that all the data is not
observed up to the point of failure (or death). In this case the data is
right censored, see the :doc:`Types of Data` section for a more detailed discussion,
surpyval offers a very clean and easy way to model this. First, let's create
a simulated data set:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(10)
    x = surv.Weibull.random(50, 30, 2.)
    observation_limit = 40
    # Censoring flag
    c = (x >= observation_limit).astype(int)
    x[x >= observation_limit] = observation_limit

In this example, we created 50 random Weibull distributed values with
alpha = 30 and beta = 2. For this example the observation window has been
set to 40. This value is where we stopped observing the events. For all the
randomly generated values that are above this limit we create the censoring
flag array c. This array has zeros where the event time was observed, and a 1
where the value is above the recorded value. For all the values in the data
that are above 40 we set them to 40. This is a common occurrence in survival
analysis and surpyval is designed to accept this input with a simple call:

.. jupyter-execute::

    model = surv.Weibull.fit(x, c)
    model

The plot for this can be seen to be:

.. jupyter-execute::

    model.plot()

The results from this model are very close to the data we input, and with only 50 samples.

Left Censored
^^^^^^^^^^^^^

The above example can be extended to another kind of censoring; left censored data. This is the case where the values are known to fall below a particular value. We can change our example data set to have a start observation time for which we will left censor all the data below that:

.. jupyter-execute::

    observation_start = 10
    # Censoring flag
    c[x <= observation_start] = -1
    x[x <= observation_start] = observation_start

That is, we set the start of the observations at 10 and flag that all the values at or below this are left censored. We can then use the updated values of x and c:

.. jupyter-execute::

    model = surv.Weibull.fit(x, c)
    model

.. jupyter-execute::

    model.plot(heuristic="Turnbull")

The values did not substantially change, although the plot does look different as there are no values below 10. Note the ``heuristic="Turnbull"``: left-censored units have no rank, so the plotting positions have to come from the Turnbull estimator.


Intervally Censored
^^^^^^^^^^^^^^^^^^^

The next type of censoring that is naturally handled by surpyval is interval censoring. Creating another example data set:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(30)
    x = surv.Weibull.random(100, 30, 10.)
    n, xx = np.histogram(x, bins=[20, 23, 26, 29, 32, 35, 38])
    x = np.vstack([xx[0:-1], xx[1:]]).T

In this example we have created the variable x with a matrix of the intervals within which each of the observations have failed. That is each exact observation has been binned into a window and the x array has an entry [left, right] within which the event failed. We also have the n array that has the count of the failures within the window. With these two values we can make the simple surpyval call:

.. jupyter-execute::

    model = surv.Weibull.fit(x, n=n)
    model

.. jupyter-execute::

    model.plot(heuristic="Turnbull")

Again, we have a result that is very close to the original parameters.
SurPyval can take as input an arbitrary combination of censored data. This
plot also looks to be a great fit! The data at the tails are a little bit
off, but the data are binned into only six intervals and the core of the model matches the data
quite well.

The same intervals can be given as two separate arrays, ``xl`` for the left
ends and ``xr`` for the right ends, which is often how inspection data arrive.
It is the same data, so it is the same fit:

.. jupyter-execute::

    surv.Weibull.fit(xl=xx[:-1], xr=xx[1:], n=n).params

A row with ``xl == xr`` is treated as an exact observation, a row with
``xr = np.inf`` as right censored and a row with ``xl = -np.inf`` as left
censored, so these two arrays can describe any mix of censoring.

Mixed Censoring
^^^^^^^^^^^^^^^

Mixed censoring, or arbitrary censoring is easily handled by SurPyval. So no matter the combination
of the data that you have, SurPyval will be able to fit a distribution to it.

.. jupyter-execute::

    import surpyval as surv

    x  = [0, 1, 2, [3, 4], [6, 10], [4, 8], 5, 19, 10, 13, 15]
    c  = [0, 0, 1, 2, 2, 2, 0, -1, 0, 1, 0]
    surv.Gumbel.fit(x, c=c)

Using truncated data
--------------------

Left truncated
^^^^^^^^^^^^^^

Surpyval has the capacity to handle arbitrary truncated data. A common occurrence of this is in the insurance industry data. When customers make a claim on their policies they have to pay an 'excess' which is a charge to submit a claim for processing. If say, the excess on a set of policies in an area is $250, then it would not be logical for a customer to submit a claim for a loss of less than that number. Therefore there will be no claims under $250. This can also happen in engineering where a part may be tested up to some limit prior to be sold, therefore, as a customer you need to make sure you take into account the fact that some parts would have been rejected at the end of the line which you may not have seen. So a washing machine may run through 25 cycles prior to shipping. This is similar to, but distinct from censoring. When something is left censored, we know there was a failure or event below the threshold.  Whereas with truncation, we do not see any variables below the threshold. A simulated example may explain this better:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    np.random.seed(10)
    x = surv.Weibull.random(100, 100, 0.6)
    # Keep only those values greater than 25
    threshold = 25
    x = x[x > threshold]

We have therefore simulated a scenario where we have taken 100 random samples from a fat tailed Weibull distribution. We then filter to keep only those records that are above the threshold. In this case we assume we haven't seen the data for the washing machines with less than 25 cycles. To understand what could go wrong if we ignore this, what do we get if we assume all the data are failures and there is no truncation?

.. jupyter-execute::

    model = surv.Weibull.fit(x=x)
    print(model.params)

With a plot that looks like:

.. jupyter-execute::

    model.plot()

Looking at the parameters of the distribution, you can see that the beta value is greater than 1. Although only slightly, this implies that this distribution has an increasing hazard rate. If you were the operator of the washing machines (e.g. a hotel or a laundromat) and any downtime had a cost, you would conclude from this that replacing the machines after a fixed time would be a good policy.

But if you take the truncation into account:

.. jupyter-execute::

    model = surv.Weibull.fit(x=x, tl=threshold)
    print(model.params)

With the plot:

.. jupyter-execute::

    model.plot(heuristic="Turnbull")

You can see now that the model fits the data much better, but also that the beta parameter is actually below 1. This shows that ignoring the left-truncated data in parametric estimation can lead to errors in prediction.

Right truncated
^^^^^^^^^^^^^^^

The example from above can be continued for right-truncated data as well. Here the data come from a Normal distribution with a mean of 100 and a standard deviation of 10, but only values between 85 and 115 could be recorded:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    np.random.seed(10)
    x = surv.Normal.random(100, 100, 10)
    tl = 85
    tr = 115
    # Truncate the data
    x = x[(x > tl) & (x < tr)]
    print(len(x), "values were recorded")

    naive = surv.Normal.fit(x)
    model = surv.Normal.fit(x=x, tl=tl, tr=tr)
    print("ignoring the truncation :", naive.params)
    print("with the truncation     :", model.params)

When plotted we get:

.. jupyter-execute::

    model.plot(heuristic="Turnbull")

From the output above, the number of data points we have has been reduced from the simulated 100, down to 87. Both fits find the centre, but the naive fit badly underestimates the spread: the truncation removed the tails, so the recorded values look less variable than the population really is. Accounting for the truncation moves the estimate of :math:`\sigma` back towards the true value of 10. It cannot recover it completely -- the tails that carry most of the information about the spread were never recorded -- which is a useful reminder that truncation costs information even when it is modelled correctly.

In the cases above we used a scalar value for the truncation values. But some data has individual values for left truncation. This is seen in trials where someone may join the trial as a late entry. Therefore each data point as an entry time. For example:

.. jupyter-execute::

    import surpyval as surv

    x  = [3, 4, 6, 7, 9, 10]
    tl = [0, 0, 0, 0, 5, 2]

    model = surv.Weibull.fit(x, tl=tl)
    print(model.params)


Intervally and Arbitrarily truncated
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Surpyval can even work with arbitrary left and right truncation:

.. jupyter-execute::

    import surpyval as surv

    x  = [3, 4, 6, 7, 9, 10]
    tl = [0, 0, 0, 0, 5, 2]
    tr = [10, 9, 8, 10, 15, 15]

    model = surv.Weibull.fit(x, tl=tl, tr=tr)
    print(model.params)

In the above example we used both the tl and tr. However, surpyval has a flexible API where it can take the truncation data as a two dimensional array:

.. jupyter-execute::

    import surpyval as surv

    x  = [3, 4, 6, 7, 9, 10]
    t =  [[0, 10], [0, 9], [0, 8], [0, 10], [5, 15], [2, 15]]

    model = surv.Weibull.fit(x, t=t)
    print(model.params)

Which, obviously, gives the same result. This shows the flexibility of the surpyval API, you can use scalar, array, or matrix values for the truncations using the t, tl, and tr keywords with the fit method and surpyval does the rest.

Truncation needs a method that models it. Maximum likelihood handles any
truncation; MPS handles a single window shared by every observation (scalar
``tl`` and ``tr``); probability plotting handles it only through the
non-parametric estimate, with the limitation shown at the end of the section
on alternate estimation methods below; and MSE and MOM do not accept
truncated data at all.

Offsets
-------

Another common feature in survival analysis is a requirement to fit a distribution with an offset. These distributions are sometimes referred to as the two-parameter (e.g. two parameter exponential) three parameter, (e.g., the three parameter Weibull), or four parameter (e.g four parameter Exponentiated Weibull distribution). SurPyval however just uses an ``offset`` to increase the numbers of parameters and allow the distribution to be shifted.

Using data from Weibull's original paper for the strength of Bofors steel shows when this might be necessary.

.. jupyter-execute::

    import surpyval as surv
    from surpyval.datasets import load_bofors_steel

    df = load_bofors_steel()
    x = df['x']
    n = df['n']

    model = surv.Weibull.fit(x=x, n=n)
    print(model.params)

.. jupyter-execute::

    model.plot()

The above plot does not look to be a good fit. However, if we use an offset we can use the three parameter Weibull distribution to attempt to get a better fit. Using offset values with surpyval is very easy:

.. jupyter-execute::

    import surpyval as surv
    from surpyval.datasets import load_bofors_steel

    df = load_bofors_steel()
    x = df['x']
    n = df['n']

    model = surv.Weibull.fit(x=x, n=n, offset=True)
    print(model)

.. jupyter-execute::

    model.plot()

This is evidently a much better fit! The offset value for an offset distribution is saved as :code:`gamma` in the model object, and every method of the model -- ``sf``, ``qf``, ``mean`` and the rest -- includes the shift. Offsets can be used for any continuous distribution whose support is the half real line :math:`(0, \infty)`: the Weibull, Gamma, LogNormal, LogLogistic, Exponential, Rayleigh and exponentiated Weibull, and a custom distribution declared on that support. For example:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(10)
    x = surv.LogLogistic.random(100, 10, 3) + 10
    model = surv.LogLogistic.fit(x, offset=True, how='MLE')
    print(model)

.. jupyter-execute::

    model.plot()

A four parameter exponentiated Weibull can also be found:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(10)
    x = surv.ExpoWeibull.random(100, 10, 1.2, 4) + 10
    model = surv.ExpoWeibull.fit(x, offset=True)
    print(model)

.. jupyter-execute::

    model.plot()

Offsets only make sense for distributions supported on the half real line ``[0, inf)`` - the offset ``gamma`` simply slides the lower bound of the support. A distribution on the whole real line, such as the Normal, can already sit anywhere, so ``surv.Normal.fit(x, offset=True)`` raises a ``ValueError``. A distribution with a finite upper bound, such as the Beta distribution on ``[0, 1]``, cannot be offset either, and ``surv.Beta.fit(x, offset=True)`` will also raise a ``ValueError``. Sliding the lower bound while pinning the upper bound at 1 does not produce another member of the Beta family. If your data are bounded on both sides and you need to estimate where those bounds are, use the four parameter Beta distribution (``Beta4``) instead, which estimates the lower bound ``a`` and upper bound ``b`` along with the two shape parameters:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(10)
    x = surv.Beta4.random(10000, 3., 4., 10., 20.)
    model = surv.Beta4.fit(x)
    print(model)

A caution: offset parameters can be unidentifiable
--------------------------------------------------

The offset ``gamma`` is a *threshold* parameter, and threshold parameters are statistically awkward. ``gamma`` trades off against the shape and scale parameters, so two very different parameter tuples can describe almost the same distribution. A high-shape Gamma sitting near the origin is, by the central limit theorem, nearly the same bell-shaped curve as a moderate Gamma shifted out to 10. The likelihood surface is correspondingly flat along that trade-off, which makes a threshold fit far more sensitive to its starting point than an ordinary two-parameter fit.

In practice that sensitivity is the estimator's problem to solve, not yours. Shifted Gamma data is recovered by every fit method that the Gamma supports:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(0)
    x = surv.Gamma.random(10_000, 3.0, 2.0) + 10.0

    print('Truth : gamma=10.000, alpha=3.000, beta=2.000')
    for how in ['MOM', 'MSE', 'MPS', 'MLE']:
        m = surv.Gamma.fit(x, offset=True, how=how)
        print('{:6s}: gamma={:.3f}, alpha={:.3f}, beta={:.3f}'.format(
            how, m.gamma, *m.params))

``MPP`` is absent from that list because the Gamma does not offer it. A
probability plot needs a straight-line y-axis that can be drawn *before*
the parameters are known; the Gamma's CDF is the regularised incomplete
gamma function, with the shape inside the special function rather than
outside as an exponent, so the only such axis is the inverse incomplete
gamma — which needs the shape. To draw the axis you need the answer.
``Gamma.fit(x, how="MPP")`` raises rather than guessing a shape to draw
the axis with; ``Gamma.plot()`` is unaffected, since by then the fitted
parameters are in hand.

What makes this work is the starting point. Every optimised offset fit (all but MPP, which needs no start) begins with ``gamma`` just below the data, at ``min(x) - 1``, and the initialisers read the remaining parameters off ``x - gamma``. Moments taken from the *unshifted* data would be dominated by the offset, and a shape read from them explodes (a Gamma shape of 649 for a true shape of 3); from such a start the optimiser can stop on an absurd tuple that is nonetheless an acceptable *distribution*, precisely because of the flat trade-off described above.

The underlying caution still stands, though, and it is worth keeping in mind for your own data:

- **Judge an offset fit by what it predicts, not only by the printed parameters.** Plot it against the non-parametric estimate, or compare the survival function, quantiles, mean and variance. Two parameter tuples that look very different can imply nearly the same distribution.
- **If you need ``gamma`` itself to be meaningful** - you are interpreting it as a guaranteed minimum life, say - prefer ``MLE``, which remains the most accurate on the parameters, and treat a single point estimate of a threshold with care regardless of method. Note that ``gamma`` has no standard error, so ``param_cb`` cannot give an interval for it (see :doc:`Parametric Estimation`).
- **If the MLE struggles** - a small sample, or a shape that puts an infinite density at the threshold - try ``how='MPS'``, which was designed for exactly this case (see the section on alternate estimation methods below).

``test_offset_divergence.py`` in the test suite pins this down for offset Gamma and Rayleigh fits with measured KL and Wasserstein distances alongside parameter tolerances: ``MLE`` is held to 5% on every parameter, and ``MOM`` (on the Rayleigh) to 10%, with the implied distributions essentially identical either way.

Fixing parameters
-----------------

Another useful feature of surpyval is the ability to easily fix parameters. For example:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(30)
    x = surv.Normal.random(50, 10., 2)
    model = surv.Normal.fit(x, fixed={'mu' : 10})
    print(model)

.. jupyter-execute::

    model.plot()

You can see that the mu parameter has been fixed at 10. This can work for distributions with many more parameters, including the offset.

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(30)
    x = surv.ExpoWeibull.random(50, 10., 2, 4) + 10
    model = surv.ExpoWeibull.fit(x, offset=True, fixed={'mu' : 4, 'gamma' : 10, 'alpha' : 10})
    print(model)

.. jupyter-execute::

    model.plot()

We have fit only one of the four parameters of an offset exponentiated-Weibull distribution, holding the other three at known values!

Parameters are fixed by name, using the names in the distribution's
``param_names`` plus ``gamma`` for the offset. Fixing works with ``MLE``,
``MPS``, ``MSE`` and ``MOM``, but not with probability plotting, which fits
all of the parameters of its line at once. With ``MOM`` a fixed parameter
needs no equation of its own, so the method matches one moment per *free*
parameter (see :doc:`Parametric Estimation`). A fixed parameter is known rather
than estimated, so it has no standard error, the confidence bounds of the other
parameters are conditional on it, and it does not count in the parameter number
:math:`k` of ``aic()``, ``aic_c()`` and ``bic()``.

Fixing a parameter also reduces how much data a fit needs. SurPyval refuses to
fit when there are fewer distinct (non-right-censored) values than free
parameters, because the answer would be arbitrary; a Weibull cannot be fitted
to a single value. With the shape fixed -- a common practice in reliability,
where a "Weibayes" analysis assumes a shape from experience -- one value is
enough:

.. jupyter-execute::

    surv.Weibull.fit([10.], fixed={'beta': 2.}).params

Finally, the optimiser can be given a starting point with ``init``: the
values in the order of ``param_names``, with ``gamma`` first if there is an
offset and ``p`` then ``f0`` last for a limited failure population or zero
inflation. With ``fixed``, ``init`` may list just the free parameters. You
rarely need it, but if a fit fails, a starting point near the answer -- a
shape of 1 and a scale near the mean of the data, say -- is the first thing
to try. An explicit ``init`` is used as the only start: the extra starting
points SurPyval otherwise tries for limited-failure and custom-distribution
fits (see :doc:`Parametric Estimation`) are skipped.

.. jupyter-execute::

    np.random.seed(30)
    x = surv.Weibull.random(50, 30., 2.)
    surv.Weibull.fit(x, init=[30., 1.]).params

Modelling with arbitrary input
------------------------------

The surpyval API is extremely flexible. All the unique examples provided above can all be used at once. That is, data can be censored, truncated, and directly observed with offsets and fixing parameters. The API is completely flexible. This makes surpyval an extremely useful tool for analysts where the data is gathered in a manner where its cleanliness is not guaranteed.

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    x  = [0, 1, 2, [3, 4], [6, 10], [4, 8], 5, 19, 10, 13, 15]
    c  = [0, 0, 1, 2, 2, 2, 0, -1, 0, 1, 0]
    tl = [-1, 0, 0, 0, 0, 0, 2, 2, -np.inf, 0, 0]
    tr = 25
    model = surv.Normal.fit(x, c=c, tl=tl, tr=tr, fixed={'mu' : 1.})
    print(model)

Data often live in a table. ``fit_from_df`` takes a pandas ``DataFrame`` and
the names of its columns; ``tl`` and ``tr`` may be a column name or a single
value, and any other ``fit`` option is passed straight through:

.. jupyter-execute::

    import pandas as pd

    df = pd.DataFrame({
        'hours':    [3, 4, 6, 7, 9, 10, 12],
        'censored': [0, 0, 1, 0, 0, 1, 0],
        'entry':    [0, 0, 0, 1, 2, 2, 0],
    })
    model = surv.Weibull.fit_from_df(df, x='hours', c='censored', tl='entry')
    print(model.params)

Sometimes there are no unit-level data at all, only a curve: a failure
curve read off a supplier's report, say. ``fit_from_ecdf(x, F)`` fits the
distribution to the points of such a curve by probability plotting -- the
same straight line ``how='MPP'`` draws, but through the CDF values you
give. ``fit_from_non_parametric`` does the same with a fitted
non-parametric model, so it matches ``how='MPP'`` with that estimator as
the heuristic:

.. jupyter-execute::

    t = np.array([2., 5., 8., 12., 16.])
    F = np.array([0.04, 0.22, 0.47, 0.76, 0.92])   # read off a published curve
    from_curve = surv.Weibull.fit_from_ecdf(t, F)
    print(from_curve.params, "R(10) =", from_curve.sf(10.))

    np.random.seed(1)
    x = surv.Weibull.random(60, 10., 2.)
    km = surv.KaplanMeier.fit(x)
    print(surv.Weibull.fit_from_non_parametric(km).params)
    print(surv.Weibull.fit(x, how='MPP', heuristic='Kaplan-Meier').params)

A model made this way has all the distribution functions, but it holds no
data, so it has no likelihood, information criteria or confidence bounds.
Only distributions with a probability plot (``how='MPP'``) can be fitted
from a curve.

Using alternate estimation methods
----------------------------------

Surpyval's API is very flexible because you can change which method is used to estimate parameters. This is useful when a more appropriate method is needed or the method you are using fails. The five methods, what they optimise and what data each accepts are explained in :doc:`Parametric Estimation`; here we see them at work.

When MLE is not the best estimator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default parametric method for surpyval is the maximum likelihood estimation (MLE), this is because it can take any arbitrary input. However, the MLE is not always the best estimator. Consider an example with the uniform distribution:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(5)
    x = surv.Uniform.random(20, 5, 10)
    print(x.min(), x.max())

    mle_model = surv.Uniform.fit(x)
    print(*mle_model.params)

You can see that the results are the same. This is because the maximum likelihood estimate of the parameters of a uniform distribution are just the smallest and largest values in the sample. If however we use the 'Maximum Product Spacing' method we get:

.. jupyter-execute::

    mps_model = surv.Uniform.fit(x, how='MPS')
    print(*mps_model.params)

You can see that using the MPS method we have parameters that are closer to the real values. This is because the MPS method can 'look outside' the existing values to estimate where the real value lies. See the details of this method in the :doc:`Parametric Estimation` section. But the MPS method is useful when you need to estimate the point at which a distribution's support starts or for any distribution that has unknown support. Concretely, this includes any offset distribution or a distribution with a finite upper and lower support (such as the Uniform).

When an estimation method fails
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The other important use case is when, for some reason, an alternate estimation method just does not work. For example, fitting an offset LogLogistic to only ten points:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    import warnings

    np.random.seed(30)
    x = surv.LogLogistic.random(10, 4., 2) + 10

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = surv.LogLogistic.fit(x, how='MLE', offset=True)
    print(str(caught[0].message).splitlines()[0])
    model.plot()

This shows, that the Maximum Likelihood Estimation has failed for this data: SurPyval warns and hands back the optimiser's starting point instead. For many distributions that starting point is a probability-plot fit; for an offset LogLogistic it is only a rough guess, which is why the fitted curve misses the points. The warning is captured and printed above; in your own code it simply appears as a ``UserWarning``. However, because we have access to other methods, we can use an alternate estimation method:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(30)
    x = surv.LogLogistic.random(10, 4., 2) + 10
    model = surv.LogLogistic.fit(x, how='MPS', offset=True)
    print(model)

.. jupyter-execute::

    model.plot()

Our estimation has worked! The fit converged and follows the ten points. Do not expect it to return the parameters the data were simulated with (an offset of 10, ``alpha = 4``, ``beta = 2``): ten points say little about a three-parameter distribution, and, as the caution on offsets above explains, quite different parameter sets describe nearly the same curve. Even though we used the MPS estimate for the parameters, we can still call all the same functions with the created variable to find the density :code:`df()`, hazard :code:`hf()`, CDF :code:`ff()`, SF :code:`sf()` etc. So regardless of the estimation method, we can still use the model.

This shows the power of the flexible API that surpyval offers, because if your modelling fails using one estimation method, you can use another. In this case, the MPS method is quite good at handling offset distributions. It is therefore a good approach to use when using offset distributions.

Every method on the same data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each method has its own options and its own limits. Probability plotting
takes the plotting-position ``heuristic`` (any of those listed in
:doc:`Parametric Estimation`, or ``'Nelson-Aalen'``, ``'Kaplan-Meier'``,
``'Fleming-Harrington'``, ``'Turnbull'`` and ``'Filliben'``) and the
regression direction ``rr``. MSE, MPP and MLE accept censored data (MPP
needs ``heuristic='Turnbull'`` for left- or interval-censored data); MPS
accepts right- and left-censored but not interval-censored data; MOM needs
exact observations. Here the four that accept censoring fit the same
right-censored sample (the data were simulated with ``alpha = 10`` and
``beta = 2``), and MOM declines:

.. jupyter-execute::

    np.random.seed(2)
    x = surv.Weibull.random(100, 10., 2.)
    c = (x > 15).astype(int)
    x = np.minimum(x, 15.)

    for how in ["MLE", "MPS", "MSE", "MPP"]:
        print(f"{how:<4}", surv.Weibull.fit(x, c, how=how).params)
    print("MPP with Blom and rr='x'",
          surv.Weibull.fit(x, c, how="MPP", heuristic="Blom", rr="x").params)

    try:
        surv.Weibull.fit(x, c, how="MOM")
    except ValueError as e:
        print("MOM :", e)

Asking a method for something it cannot do raises a ``ValueError`` (or
``NotImplementedError``) that says why, rather than returning a quietly wrong
answer.

Likelihoods and information criteria for any method
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

That extends to the information criteria. A log-likelihood is a property of the
parameters and the data, not of the search that found them, so ``neg_ll()``,
``aic()``, ``bic()`` and ``aic_c()`` are available after *any* fit — not only
after MLE:

.. jupyter-execute::

    np.random.seed(1)
    x = surv.Weibull.random(500, 10., 2.)

    print(f"{'how':<6}{'neg_ll':>12}{'AIC':>12}{'BIC':>12}")
    for how in ["MLE", "MPS", "MSE", "MOM", "MPP"]:
        m = surv.Weibull.fit(x, how=how)
        print(f"{how:<6}{m.neg_ll():12.3f}{m.aic():12.3f}{m.bic():12.3f}")

Two things are worth noticing. The first is that you can compare distributions
by AIC or BIC regardless of how you fitted them, which is the usual way of
choosing between candidate models. The second is a sanity check on the
estimators themselves: MLE attains the lowest negative log-likelihood, because
that is precisely the quantity it minimises. The others land close behind, each
optimising something else — MPS the spacings, MSE the distance to the
non-parametric estimate, MPP the straightness of the probability plot, MOM the
moments.

Comparing distributions
^^^^^^^^^^^^^^^^^^^^^^^

To choose a distribution, fit the candidates to the same data and compare an
information criterion; lower is better. ``fit_best(x, c, n, t)`` does this by
maximum likelihood for thirteen continuous distributions -- ``Beta``,
``Beta4``, ``Exponential``, ``ExpoWeibull``, ``Gamma``, ``Gumbel``,
``Logistic``, ``LogLogistic``, ``LogNormal``, ``Normal``, ``Rayleigh``,
``Uniform`` and ``Weibull`` (not ``GumbelLEV``, the discrete distributions or
offset models) -- and returns the winner (see
:doc:`comparison_and_validation`). ``metric`` may be ``'aic'`` (the default),
``'aic_c'``, ``'bic'`` or ``'neg_ll'``, and ``include`` or ``exclude`` (lists
of names, not both) narrow the candidates. A candidate that cannot be fitted
-- the Beta when the data leave :math:`[0, 1]`, say -- is skipped with a
warning, and ``None`` is returned if none can.

.. jupyter-execute::

    np.random.seed(1)
    x = surv.Weibull.random(100, 10., 2.)

    for dist in [surv.Weibull, surv.Gamma, surv.LogNormal, surv.Rayleigh]:
        print(f"{dist.name:<10} AIC = {dist.fit(x).aic():8.2f}")

    best = surv.fit_best(x, include=["Weibull", "Gamma", "LogNormal", "Rayleigh"])
    print("best:", best.dist.name, best.params)

The data are Weibull with a shape of 2, yet the Rayleigh wins. That is not a
mistake: the Rayleigh *is* a Weibull with the shape fixed at 2, so it fits
just as well with one parameter fewer, and the criterion rewards the simpler
model. Information criteria choose the most economical adequate model, not the
"true" one; always look at the fit as well.

A warning about truncated data and probability plotting
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As the :doc:`Non-Parametric Estimation` notes explain, when every value is
truncated by the same window the Turnbull estimator cannot tell how much
probability lies outside it, so its estimate runs from about 0 to about 1
inside the window. A probability plot fitted to it inherits the problem. We
will now show what happens. First, some example data:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(1)
    x = surv.Normal.random(1000, 100, 10)
    tl = 90
    tr = 110
    x = x[x > tl]
    x = x[x < tr]

    mpp_model = surv.Normal.fit(x, tl=tl, tr=tr, heuristic="Turnbull", how='MPP')
    mpp_model

.. jupyter-execute::

    mpp_model.plot(heuristic="Turnbull")

You can see that there is a strange match between the Turnbull estimate of the CDF and the parametric
model. Also, you can see that the CDF at 90 is near 0% and the CDF at 110 is near 100%. This shows
that it has not taken into account the truncation. Instead, if we use MLE we get:

.. jupyter-execute::

    model = surv.Normal.fit(x, tl=tl, tr=tr, how='MLE')
    model

.. jupyter-execute::

    model.plot(heuristic="Turnbull")

We can see that the MLE method is a much better fit to this data, further, the MLE estimate of the
:math:`\sigma` parameter is much closer. The plotting points for the MLE plot
have been adjusted in accordance with the truncation that the MLE model has estimated at the first entry.
This is because it is known to be truncated and needs to be adjusted. This is not possible with the MPP
method because the Turnbull estimator cannot adjust the truncation at the first and last value as it
can make no assumptions about the truncation at those points.

This is just a word of warning for when using Truncation and the MPP method, make sure not all values
are truncated by the same value, otherwise it will give a poor fit.

Mixture Models
--------------

On occasion, it can appear as though there are one, or two different distributions in the data you are using. On these occasions it can be useful to use a different type of distribution; or really, distributions. A mixture model is a distribution made from the partial combination of several distributions. Intuitively, it can be understood as a distribution where there is a proportion that fail for each kind of distribution. So 60% may come from a Weibull(3, 4) distribution but then another 40% come from a Weibull(19, 2) distribution. With weights :math:`w_{j}` that sum to one, the mixture of :math:`m` distributions is

.. math::

    F(x) = \sum_{j=1}^{m} w_{j} F_{j}(x), \qquad f(x) = \sum_{j=1}^{m} w_{j} f_{j}(x).

SurPyval uses the Expectation-Maximisation (EM) algorithm to fit a mixture. We do not know which component each unit came from, and EM alternates between two easy problems: given the current fit, compute each unit's probability of belonging to each component (the E step), then refit every component, and the weights, with the units weighted by those probabilities (the M step). Each round cannot decrease the likelihood, and the rounds repeat until it stops changing. A mixture is created with ``MixtureModel(dist, m)`` -- the distribution to use for every component, and the number of components -- and fitted with ``fit``:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt

    x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17 ,17, 18, 19]
    x_ = np.linspace(np.min(x), np.max(x))

    model = surv.Weibull.fit(x)
    wmm = surv.MixtureModel(dist=surv.Weibull, m=2)
    wmm.fit(x)

    model.plot(plot_bounds=False)
    plt.plot(x_, wmm.ff(x_), color='red')

You can see that the mixture model, in red, tracks the data more closely than does the single model. The fitted weights and component parameters are shown by printing the mixture:

.. jupyter-execute::

    wmm

SurPyval has incredible flexibility. The number of distributions can be changed by simply changing the value of ``m``, and, the distribution passed to ``dist`` in the mixture can also be changed. Consider:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt

    np.random.seed(3)
    x1 = surv.Normal.random(40, -10, 3)
    x2 = surv.Normal.random(60, 10, 4)
    x3 = surv.Normal.random(80, 30, 5)
    x = np.concatenate([x1, x2, x3])
    np.random.shuffle(x)
    x_ = np.linspace(np.min(x), np.max(x))

    normal = surv.Normal.fit(x)
    gmm = surv.MixtureModel(dist=surv.Normal, m=3)
    gmm.fit(x)

    normal.plot(plot_bounds=False)
    plt.plot(x_, gmm.ff(x_), color='red')

It was that simple to create a gaussian mixture model using ``m=3`` and the ``dist=surv.Normal`` parameters. There is no default distribution, so ``dist`` must always be given; ``m`` defaults to 2. Any of the fittable distributions can be used as the component distribution. The components are found in the order the EM settles on, so compare them by their parameters rather than their position:

.. jupyter-execute::

    print("weights :", gmm.w.round(3))
    print("(mu, sigma) of each component:")
    print(gmm.params.round(2))

The weights recover the 40/60/80 split of the simulated data (2/9, 3/9 and 4/9), and the component means sit close to -10, 10 and 30.

Mixture models take counts, censoring flags and truncation as input (``x``, ``c``, ``n``, ``t``, ``tl``, ``tr``, ``xl``, ``xr``, as for any ``fit``). Truncation needs care: the truncation window is a property of the whole mixture, not of any one component, so a truncated mixture cannot be split up the way EM needs. For truncated data SurPyval instead maximises the truncation-corrected likelihood directly, starting from the same initial fit.

A fitted mixture is a smaller object than a fitted distribution. It has ``sf``, ``ff``, ``df``, ``Hf``, ``cs``, ``mean``, ``random`` and ``plot``, the weights ``w`` and component parameters ``params`` (one row per component), and ``loglike``, which despite its name is the *negative* log-likelihood of the fit. It has no ``hf``, ``qf``, confidence bounds or information criteria, but an AIC is easily formed by hand: a mixture of :math:`m` components with :math:`k` parameters each has :math:`mk + m - 1` free parameters (the weights sum to one). Here a two-Weibull mixture is compared with a single Weibull on right-censored data, and then saved and restored with ``to_dict`` / ``surpyval.from_dict`` like any other model:

.. jupyter-execute::

    np.random.seed(1)
    x = np.concatenate([surv.Weibull.random(60, 5, 3), surv.Weibull.random(40, 20, 4)])
    c = (x > 22).astype(int)          # right censor anything still running at 22
    x = np.minimum(x, 22)

    wmm = surv.MixtureModel(dist=surv.Weibull, m=2)
    wmm.fit(x, c=c)
    print("weights:", wmm.w.round(3))

    k_mix = wmm.m * wmm.dist.k + wmm.m - 1
    print("AIC single Weibull :", surv.Weibull.fit(x, c).aic())
    print("AIC 2-Weibull mix  :", 2 * k_mix + 2 * wmm.loglike)

    restored = surv.from_dict(wmm.to_dict())
    print(restored.sf([5, 10]), wmm.sf([5, 10]))

The mixture's AIC is lower by about 42, decisive evidence for two populations, and its weights are close to the 60/40 split that was simulated.

This makes SurPyval a truly powerful package for your survival analysis. Two cautions. A mixture has many parameters, so it needs a good amount of data: SurPyval refuses a fit with fewer than :math:`m(k + 1)` units. And the EM finds *a* maximum, which depends on where it starts. SurPyval starts by sorting the data, cutting its distinct values into :math:`m` consecutive blocks and fitting one component to each, with equal weights; with poorly separated components, check that the answer makes sense.


Limited Failure Population
--------------------------

Another kind of model that is useful in survival analysis is when a population has a limited number of items in the population that are susceptible to the failure. This is also known as a 'Defective Subpopulation' model. As such, no matter how long a test continues, it will not be possible for all items to fail (with the particular death/failure).

As an example, we can created a Defective Subpopulation Weibull, also known as a Limited Failure Population Model using a Weibull distribution:

.. jupyter-execute::

    import surpyval as surv
    import numpy as np
    from matplotlib import pyplot as plt

    lfp_weibull = surv.Weibull.from_params([10, 2], p=0.6)
    np.random.seed(10)
    # LFP Model outputs x, c, and n from `random()`
    x, c, n, _ = lfp_weibull.random(100)

    # Fit regular Weibull
    model = surv.Weibull.fit(x=x, c=c, n=n)

    # Set LFP to be `True`
    lfp_model = surv.Weibull.fit(x=x, c=c, n=n, lfp=True)
    print(lfp_model)

.. jupyter-execute::

    model.plot(plot_bounds=False)
    xx = np.linspace(np.min(x), np.max(x)*2)
    plt.plot(xx, lfp_model.ff(xx), color='red')

This API works with any distribution so simply changing ``Weibull`` to ``Exponential`` would create a Defective Subpopulation Exponential / Limited Failure Population Exponential model. Further, if it was changed to ``Gamma`` it would create a Defective Subpopulation Gamma model / Limited Failure Population Gamma.

The estimated proportion ``p`` is a parameter like any other, so it has a
confidence interval, and it changes what the model predicts far into the
future. The survival function levels off at ``1 - p`` instead of falling to
zero, and a quantile beyond ``p`` is infinite, because that proportion of the
population never fails:

.. jupyter-execute::

    print("p =", lfp_model.p, " 95% CI:", lfp_model.param_cb('p'))
    print("R(1000) =", lfp_model.sf(1000.))
    print("time by which 70% have failed:", lfp_model.qf(0.7))

``mean()`` of an LFP model is the *defective* mean, ``p`` times the mean of
the base Weibull, because a unit that never fails contributes nothing to it;
the mean life of the units that do fail is the base mean,
``surv.Weibull.mean(*lfp_model.params)``. ``var()`` and ``moment()`` follow the
same convention -- the units that never fail are scored as 0, so ``var()`` is
``moment(2) - mean()**2`` -- and the same holds for the zero-inflated mass
``f0`` below, which sits at 0 anyway.

LFP models can only be fitted with ``MLE``; the other methods raise. And ``p``
is only well determined when the data follow the units long enough to see the
failure curve level off (see :doc:`Parametric Estimation`). Real data are
rarely that kind. Meeker's integrated-circuit test put 4156 units on test for
1370 hours and saw 28 failures, most of them early:

.. jupyter-execute::

    from surpyval.datasets import load_meeker_lfp

    df = load_meeker_lfp()
    print("units:", df['n'].sum(), " failures:", df['n'][df['c'] == 0].sum())

    ic_lfp = surv.Weibull.fit(df['x'], df['c'], df['n'], lfp=True)
    ic_plain = surv.Weibull.fit(df['x'], df['c'], df['n'])
    print(ic_lfp)
    print("p 95% CI :", ic_lfp.param_cb('p'))
    print("AIC LFP  :", ic_lfp.aic(), "  plain Weibull:", ic_plain.aic())

About 0.7% of the population is susceptible to this failure mode, and the
susceptible units fail early (a shape below one, and a characteristic life of
about 28 hours). The plain Weibull, forced to explain the flattening with a
single population, needs a shape of 0.2 and a characteristic life of about
:math:`10^{14}` hours, and its AIC is 18 worse. Its fitted curve never levels
off, which is exactly the flat ridge described in
:doc:`Parametric Estimation`: along it only the combination
``p * alpha**(-beta)`` matters. A start far out on that ridge stays there:

.. jupyter-execute::

    stuck = surv.Weibull.fit(df['x'], df['c'], df['n'], lfp=True,
                             init=[1e6, 0.3, 0.1])
    print("from init  : p =", stuck.p, " alpha =", stuck.alpha,
          " neg_ll =", stuck.neg_ll())
    print("by default : p =", ic_lfp.p, " alpha =", ic_lfp.alpha,
          " neg_ll =", ic_lfp.neg_ll())

With an explicit ``init`` SurPyval uses that start alone, and the optimiser
barely moves off it. Without one, an LFP fit is also started from the failures
alone (a Weibull fitted to the 28 failures, with ``p`` at 28/4156), and the
start with the best likelihood wins, which here is almost ten log-likelihood
units better. If you do pass ``init`` to an LFP fit, compare ``neg_ll()`` with
the default fit.

Distributions that call one of their own parameters ``p`` -- the
``Geometric`` and the ``NegativeBinomial`` -- keep that name, and their
limited-failure proportion is called ``lfp_p`` instead (in ``fixed``,
``param_cb`` and the printed model); see the section on discrete distributions
below.

Zero-Inflated Modelling
-----------------------

In survival analysis you might have the scenario where many failure times are 0, known as being dead on arrival. In this case we need a model that can account for the fact that many will be failed at 0, this is a situation that cannot be handled by regular distributions, since most have a 0% chance of failing at 0. Therefore what we need is something that is symmetrical to the LFP/DS case, where a proportion of the failures occur at 0 instead of there being a proportion that will never fail.

.. jupyter-execute::

    import surpyval as surv
    from autograd import numpy as np

    dist = surv.ExpoWeibull
    model = dist.from_params([10.2, 2., 1.3], f0=0.15)
    np.random.seed(10)
    x = model.random(100)
    model

Random values from a zero-inflated model come back as a plain array, in which the dead-on-arrival units are exact zeros. Using this random data, we can make a fitted model (with the added convenience not offered in the real world of knowing exactly what parameters we are aiming toward).

.. jupyter-execute::

    fitted_model = dist.fit(x, zi=True)
    print(fitted_model)

.. jupyter-execute::

    fitted_model.plot()

We can see that we have made a good fit. The fitted ``f0`` of 0.18 is simply
the fraction of zeros in the sample -- 18 of the 100 draws happened to be
dead on arrival, against the 15% expected -- because a zero can only have come
from the zero-inflation mass. The three ExpoWeibull parameters look further
from the truth than they are: its two shape parameters trade off against each
other, so different triples draw nearly the same curve (the offset caution
above applies here too). Comparing the survival functions, rather than the
parameters, shows it:

.. jupyter-execute::

    t = np.array([5., 10., 15.])
    print("true   R(t):", model.sf(t))
    print("fitted R(t):", fitted_model.sf(t))

To showcase the SurPyval API again, and to demonstrate the flexibility, it is trivial to have Defective Subpopulation Zero Inflated (DSZI) model / Limited Failure Population and Zero Inflated model.

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    dist = surv.LogNormal
    model = dist.from_params([2.2, .2], f0=0.05, p=0.6)
    np.random.seed(10)
    # Random values from LFP models come in xcnt format
    x, c, n, _ = model.random(100)

    fitted_model = dist.fit(x, c, n, zi=True, lfp=True)
    print(fitted_model)

.. jupyter-execute::

    fitted_model.plot(plot_bounds=False)

Using a ``LogNormal`` distribution we were able to easily capture the DS/LFP and ZI behaviour of the data. With both options, ``p`` is the total proportion that ever fails, *including* the ``f0`` that fail at time zero, so here about 4% fail at once and about 58% more fail over time. Zero inflation needs a distribution whose support starts at zero (it is not available for the Normal, say), and like LFP it can only be fitted by ``MLE``.

Flexible parametric (Royston-Parmar)
------------------------------------

Sometimes no standard distribution fits: the hazard turns over, has a bathtub,
or is multi-modal. A **Royston-Parmar** model handles this by replacing the
straight line a Weibull draws for its log-cumulative-hazard against log-time
with a **restricted cubic spline** — a smooth, fully parametric baseline of
arbitrary shape. It is as flexible as a Cox baseline but, being parametric,
gives a smooth hazard and **extrapolates**, which a Cox fit cannot.

``RoystonParmar.fit`` takes a ``df`` (degrees of freedom = spline terms; ``df=1``
is exactly a Weibull) and a ``scale``: ``"hazard"`` (proportional hazards),
``"odds"`` (proportional odds), or ``"normal"`` (probit; ``df=1`` is a
log-normal). Knots default to quantiles of the event times. Here we fit data
whose hazard a single Weibull cannot capture, and pick ``df`` by AIC:

.. jupyter-execute::

    from surpyval import RoystonParmar, Weibull

    np.random.seed(2)
    x = np.concatenate([Weibull.random(400, 3, 5), Weibull.random(400, 30, 1.2)])

    for df in (1, 2, 3, 4):
        m = RoystonParmar.fit(x, df=df)
        print(f"df={df}  AIC={m.aic():8.1f}")

The AIC keeps improving past ``df=1`` (the Weibull), then stops — the usual way
to choose the number of knots. Take the best and look at the fitted survival
with its confidence band:

.. jupyter-execute::

    model = RoystonParmar.fit(x, df=3)

    t = np.linspace(0.5, 50, 200)
    plt.plot(t, model.sf(t), 'k-', label='Royston-Parmar (df=3)')
    band = model.cb(t, on='sf')
    plt.plot(t, band, 'r--')
    plt.plot(t, Weibull.fit(x).sf(t), 'b:', label='Weibull')
    plt.legend(); plt.xlabel('Time'); plt.ylabel('S(t)')

The spline follows the two-component shape the single Weibull misses. Because
the spline is linear beyond its boundary knots, the model extrapolates with a
Weibull-like tail rather than a wild cubic — which is what makes it safe to read
off a restricted-mean survival time or a far quantile. The full arbitrary
censoring/truncation surface is supported — observed, right-, left- and
interval-censored data (``c``, or ``xl`` / ``xr``), with left- and/or
right-truncation and weights (``tl`` / ``tr`` / ``t``, ``n``) — and a fitted
model serialises with ``to_dict`` / ``from_dict`` like any other.

The ``df=1`` claim is easy to check: on the hazard scale the one-knot-free
spline is :math:`\ln H(t) = \gamma_{0} + \gamma_{1}\ln t`, which is a Weibull
with :math:`\beta = \gamma_{1}` and :math:`\alpha = e^{-\gamma_{0}/\gamma_{1}}`.
The two fits reach the same likelihood. The scale is a modelling choice too,
and the same AIC comparison picks it; here the data are right censored at 40:

.. jupyter-execute::

    m1 = RoystonParmar.fit(x, df=1)
    print("RP df=1 :", m1.neg_ll(), " Weibull:", Weibull.fit(x).neg_ll())
    print("beta =", m1.params[1], " alpha =", np.exp(-m1.params[0] / m1.params[1]))

    c = (x > 40).astype(int)
    xc = np.minimum(x, 40)
    for scale in ("hazard", "odds", "normal"):
        m = RoystonParmar.fit(xc, c=c, df=3, scale=scale)
        print(f"{scale:<7} AIC={m.aic():8.1f}")

The fitted model has ``sf``, ``ff``, ``df``, ``hf``, ``Hf``, ``qf``,
``mean`` and ``random``, ``neg_ll()``, ``aic()`` and ``bic()``, and a
``summary()`` of the knots and coefficients; its confidence bands (``cb``) are formed on the
spline's linear predictor and are available on ``'sf'``, ``'ff'`` and
``'Hf'``. Explicit knots, on the log-time scale and including the two
boundary knots, can be given with ``knots``. See
:doc:`univariate/royston_parmar` for the full API.

Discrete Distributions
----------------------

Every distribution used so far is *continuous* -- a failure time can be any positive real number. But many reliability problems are naturally *discrete*: an item does not fail after "3.7 cycles", it fails **on** the 4th cycle. A switch is toggled until it breaks, a component absorbs shocks until it fractures, a system is inspected once per period until a defect appears. When the lifetime is a count -- a positive integer -- a discrete distribution is the honest model, and reaching for a continuous one can bias the answer.

*SurPyval* provides these discrete lifetime distributions, supported on the positive integers :math:`\{1, 2, 3, \dots\}`:

.. list-table::
   :header-rows: 1

   * - Distribution
     - Continuous analogue
     - Discrete hazard
   * - ``Geometric``
     - Exponential
     - constant (memoryless)
   * - ``DiscreteWeibull``
     - Weibull
     - increasing, constant, or decreasing
   * - ``NegativeBinomial``
     - Gamma
     - cycles until the ``r``-th shock
   * - ``BetaGeometric``
     - a frailty (mixture) model
     - decreasing: the frailest units fail first
   * - ``Discretize(dist)``
     - any continuous ``dist`` on :math:`[0, \infty)`
     - that of ``dist``, grouped into whole cycles

along with the ``Poisson``, a count on :math:`\{0, 1, 2, \dots\}`. They are used exactly like the continuous distributions -- the same ``fit()`` call, the same ``sf``, ``ff``, ``hf``, ``Hf`` and ``df`` methods, and the same support for censoring, truncation, and counts. Two meanings shift slightly, and are explained in :doc:`Parametric Estimation`: ``df`` is the probability *mass* :math:`P(T = k)` and ``sf`` is :math:`P(T > k)`.

The ``Geometric`` distribution is the discrete analogue of the ``Exponential``: each cycle fails independently with a constant probability ``p``, so it is *memoryless*. It models the number of cycles until the first failure.

.. jupyter-execute::

    import surpyval as surv
    import numpy as np

    np.random.seed(1)
    # 200 items; each cycle fails with probability 0.15
    x = surv.Geometric.random(200, 0.15)
    surv.Geometric.fit(x)

A pitfall hides in that parameter's name. ``p`` is also the name SurPyval
reserves for the proportion of a limited failure population, and the model's
``p`` attribute means that proportion (1 for an ordinary model). The fitted
per-cycle probability of a ``Geometric``, and the ``p`` of a
``NegativeBinomial``, are read from ``params`` instead:

.. jupyter-execute::

    geom = surv.Geometric.fit(x)
    print("per-cycle probability:", geom.params[0])
    print("geom.p               :", geom.p, "(the limited-failure proportion)")

Parameter *names*, though, always mean the distribution's own parameter
first: ``param_cb('p')`` bounds the per-cycle probability, ``fixed={'p': ...}``
fixes it, and a limited failure population fitted to these two distributions
calls its proportion ``lfp_p``:

.. jupyter-execute::

    print("per-cycle probability 95% CI:", geom.param_cb('p'))
    lfp_geom = surv.Geometric.fit(np.minimum(x, 10), c=(x > 10).astype(int),
                                  lfp=True)
    print("susceptible proportion 95% CI:", lfp_geom.param_cb('lfp_p'))

The ``DiscreteWeibull`` distribution (the Nakagawa-Osaki Type I) is the discrete analogue of the ``Weibull``, and like it has a flexible hazard: ``beta`` controls the shape, with ``beta < 1`` a decreasing (infant-mortality) hazard, ``beta = 1`` the constant-hazard Geometric, and ``beta > 1`` an increasing (wear-out) hazard. Its other parameter, ``q``, is the probability of surviving the first cycle.

.. jupyter-execute::

    np.random.seed(2)
    # beta = 2 -> a wearing-out item
    x = surv.DiscreteWeibull.random(200, 0.95, 2.0)
    model = surv.DiscreteWeibull.fit(x)
    model

Because ``beta > 1`` here, the discrete hazard rises with each cycle -- the chance of failing on the next cycle grows as the item wears:

.. jupyter-execute::

    model.hf([1, 5, 10, 15])

The ``NegativeBinomial`` distribution models the number of cycles until an item accumulates enough shocks to fail: with ``T = 1 + Y`` where ``Y`` is the number of failures before the ``r``-th success. It is overdispersed relative to a Poisson count and reduces to the ``Geometric`` when ``r = 1``.

.. jupyter-execute::

    np.random.seed(3)
    x = surv.NegativeBinomial.random(1000, 3.0, 0.4)
    surv.NegativeBinomial.fit(x)

Since ``r`` and ``p`` trade off against each other, the negative binomial usually needs more data than the single-parameter Geometric to pin both down.

The ``BetaGeometric`` is a Geometric in which every unit has its *own*
per-cycle failure probability, varying across the population as a Beta
distribution with parameters ``a`` and ``b``. The weak units fail early and
leave the strong ones behind, so the hazard of the population *falls* with
time even though each unit's hazard is constant -- a pattern no single
Geometric can produce, and a common one in customer-retention and
early-life-failure data:

.. jupyter-execute::

    np.random.seed(4)
    x = surv.BetaGeometric.random(500, 3.0, 5.0)
    model = surv.BetaGeometric.fit(x)
    print(model.params)
    print("hazard at cycles 1, 2, 5 and 10:", model.hf([1, 2, 5, 10]))

``Discretize`` turns any continuous distribution on :math:`[0, \infty)` into a
discrete one by counting the cycle in which the continuous failure happens,
:math:`K = \lceil T \rceil`. Its mass on cycle :math:`k` is the continuous
probability of failing in :math:`(k - 1, k]`, and it keeps the parameters of
the continuous distribution, so a discretised Weibull fitted to cycle counts
reports an ordinary Weibull ``alpha`` and ``beta``:

.. jupyter-execute::

    DiscretizedWeibull = surv.Discretize(surv.Weibull)

    np.random.seed(5)
    cycles = np.ceil(surv.Weibull.random(300, 10, 2))
    model = DiscretizedWeibull.fit(cycles)
    print(model.dist.name, model.params)

The ``Poisson`` distribution is the count of events in a fixed period when
events occur at a constant rate ``mu``. Unlike the lifetimes above it
includes zero:

.. jupyter-execute::

    np.random.seed(3)
    counts = surv.Poisson.random(200, 4.0)
    print(surv.Poisson.fit(counts).params, counts.mean())

The maximum likelihood estimate of ``mu`` is the sample mean, as the output
shows.

The full ``fit()`` API carries over. Censored and truncated discrete data are handled exactly as for the continuous distributions -- here every item still running after 10 cycles is right censored:

.. jupyter-execute::

    np.random.seed(4)
    x = surv.DiscreteWeibull.random(200, 0.95, 2.0)
    c = np.zeros_like(x)
    c[x > 10] = 1
    x[x > 10] = 10
    surv.DiscreteWeibull.fit(x, c=c)

A right-censored value of 10 means "still working after cycle 10", that is :math:`T > 10`.

Because the support is :math:`\{1, 2, 3, \dots\}`, the value ``0`` is left free to carry a **zero-inflation** mass -- the "dead on arrival" units from the previous section. Fitting with ``zi=True`` recovers both the lifetime parameters and the structural-zero fraction:

.. jupyter-execute::

    np.random.seed(5)
    x = surv.Geometric.random(200, 0.2)
    x = np.concatenate([x, np.zeros(40)])  # 40 dead-on-arrival units
    surv.Geometric.fit(x, zi=True)

The fraction of zeros is 40 of 240, or 0.167, exactly the fitted ``f0``. (The ``Poisson`` already has mass at zero, so it cannot be zero inflated.)

A note on estimation: probability plotting (MPP) is not defined for these discrete lifetimes, since their step-shaped CDFs cannot be drawn as a straight line, and neither is MPS, since tied integer values make the spacings degenerate; both raise a ``ValueError``. Maximum likelihood (the default), MSE and MOM all work (MOM for the ``BetaGeometric`` needs a sample more dispersed than a Geometric's, see :doc:`Parametric Estimation`). Calling ``plot()`` on a discrete model raises for the same reason as MPP, and ``offset=True`` raises a ``ValueError``: shifting a distribution on the integers by a continuous offset is not a member of the family. All the other model methods -- ``sf``, ``ff``, ``hf``, ``Hf``, ``df``, ``qf``, ``mean``, ``moment``, ``random`` and the confidence bounds ``cb`` -- work as usual (there is no ``entropy`` for these distributions).

.. jupyter-execute::

    np.random.seed(2)
    x = surv.DiscreteWeibull.random(200, 0.95, 2.0)
    for how in ["MLE", "MSE", "MOM"]:
        print(how, surv.DiscreteWeibull.fit(x, how=how).params)

    model = surv.DiscreteWeibull.fit(x)
    print("R(5) with 95% bounds:", model.sf(5), model.cb(5, on='sf'))

Per-demand and degenerate models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A handful of models describe events with little or no time dimension. They
estimate their parameters in closed form, so their ``fit`` takes only the
data it needs (no ``how``, ``offset``, ``lfp``, ``zi`` or ``fixed``):

- ``Bernoulli``: one pass/fail outcome, ``x`` is 0 or 1 and :math:`P(X = 1) = p`.
  Read as a one-shot device, ``p`` is the probability it works on demand. Its
  survival follows the convention :math:`R(x) = P(X \geq x)`, so ``R(0) = 1``
  and ``R(1) = p``. Fitted from 0/1 outcomes (and optional counts ``n``).
- ``FixedEventProbability``: a proportion ``p`` of units experience the event
  and the rest never do, with nothing said about *when*: ``F(x) = p`` at
  every ``x``.
- ``Binomial``: the number of events in ``n`` independent trials; fitted for
  a known number of trials, ``n_trials``, which is reported back as the first
  of its two parameters ``(n, p)``.
- ``ExactEventTime``: an event known to occur at one fixed time ``T``,
  estimated from "not yet" (right-censored) and "already" (left-censored)
  checks, as the midpoint between the latest "not yet" and the earliest
  "already".
- ``InstantlyOccurs`` and ``NeverOccurs``: no parameters at all; everything
  has already failed, or nothing ever will. They arise as the limits of the
  models above and as components of larger models.

.. jupyter-execute::

    # 17 of 20 demands succeeded
    print("Bernoulli p :", surv.Bernoulli.fit([0, 1], n=[3, 17]).params)
    switch = surv.Bernoulli.from_params(0.85)
    print("R(0), R(1)  :", switch.sf([0, 1]))

    # 3 of 5 units had the event at some point
    print("Fixed p     :", surv.FixedEventProbability.fit([0, 1, 1, 0, 1]).params)

    print("Binomial    :", surv.Binomial.fit([2, 3, 1, 4], n_trials=5).params)

    # checked at 2 and 3: not yet; checked at 4, 5 and 6: already happened
    event = surv.ExactEventTime.fit([2, 3, 4, 5, 6], c=[1, 1, -1, -1, -1])
    print("event time  :", event.params)

    from surpyval import NeverOccurs
    print("NeverOccurs R:", NeverOccurs.sf(np.array([1., 100.])))

See :doc:`univariate/bernoulli`, :doc:`univariate/fixed_event_probability`,
:doc:`univariate/binomial`, :doc:`univariate/exact_event_time` and
:doc:`univariate/degenerate` for their full APIs.

Confidence Intervals
--------------------

*SurPyval* can be used to compute the confidence interval for any of the functions of a distribution. That is, *SurPyval* can
compute the confidence interval for ``ff()``, ``sf()``, ``hf()``, ``Hf()``, and ``df()``.

Once you have a model, this can easily be computed with the ``cb()`` method.

.. jupyter-execute::

    from surpyval import Weibull
    import numpy as np
    from matplotlib import pyplot as plt

    np.random.seed(10)
    x = Weibull.random(100, 10, 3)

    model = Weibull.fit(x)

    x_plot = np.linspace(0, 20, 100)
    plt.plot(x_plot, model.Hf(x_plot), color='black')
    plt.plot(x_plot, model.cb(x_plot, on='Hf', alpha_ci=0.1), color='red', linestyle='--')

This shows that we can change the confidence level with ``alpha_ci`` and that we can change the function for which
we want the confidence interval. That is, the ``on`` keyword can be any of ``sf``, ``ff``, ``df``, ``hf``, or ``Hf``.
Here ``alpha_ci=0.1`` gives a 90% interval; the default is 0.05, a 95% interval. A two-sided bound returns two
columns, lower and upper; ``bound='lower'`` or ``bound='upper'`` returns a single one-sided bound. A one-sided lower
bound on reliability is the usual form of a reliability demonstration:

.. jupyter-execute::

    print("R(5)              :", model.sf(5.))
    print("two-sided 95%     :", model.cb(5., on='sf'))
    print("95% lower bound   :", model.cb(5., on='sf', bound='lower'))

This will work with models that you create as well, so even a user defined Distribution will be able to have the
confidence intervals computed. Creating these models is discussed in the section below.

Confidence bounds come from the curvature of the likelihood, so they are
available for models fitted by maximum likelihood (the default), including
limited-failure-population and zero-inflated models, whose extra parameters
widen the bounds. A model fitted with ``how='MPS'``, ``'MSE'``, ``'MPP'`` or
``'MOM'`` raises instead; so does a Uniform, whose MLE sits on the edge of its
support.

The band above is a *Wald* band: it propagates the parameter covariance through
the function by the delta method. ``cb`` also offers a **likelihood-ratio**
band via ``method='lr'``. At each time the bound is the most extreme value of
the function - here the reliability - over the parameter confidence region, so
it is transformation-invariant and does not rely on a quadratic approximation.
On small or heavily censored samples the two can differ noticeably, with the
likelihood-ratio band usually the better calibrated:

.. jupyter-execute::

    np.random.seed(10)
    x = Weibull.random(20, 10, 3)     # a small sample
    model = Weibull.fit(x)

    x_plot = np.linspace(2, 18, 60)
    plt.plot(x_plot, model.sf(x_plot), color='black', label='estimate')
    wald = model.cb(x_plot, on='sf', method='wald')
    lr = model.cb(x_plot, on='sf', method='lr')
    plt.plot(x_plot, wald, color='red', linestyle='--', label='Wald')
    plt.plot(x_plot, lr, color='blue', linestyle=':', label='likelihood ratio')
    handles, labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles[:3], ['estimate', 'Wald', 'likelihood ratio'])
    plt.xlabel('Time')
    plt.ylabel('R(t)')

The likelihood-ratio band is computed pointwise, so it is slower than the Wald
band, needs the original data (a model restored from ``from_dict`` raises), and
is not yet available for offset / limited-failure-population / zero-inflated
models.

Bounds on the *parameters* themselves come from ``param_cb``. By default it
returns a Wald interval built from the parameter's standard error. For small or
heavily censored samples the Wald interval - being symmetric on a transformed
scale - can have poor coverage, and the reliability-engineering convention is to
use a **likelihood-ratio** (profile) interval instead, via ``method='lr'``:

.. jupyter-execute::

    np.random.seed(3)
    x = Weibull.random(15, 10, 2)     # a small sample
    model = Weibull.fit(x)

    print("beta :", model.params[1])
    print("Wald :", model.param_cb('beta', method='wald'))
    print("LR   :", model.param_cb('beta', method='lr'))

The likelihood-ratio interval is the set of shape values whose profile deviance
stays within the :math:`\chi^2_1` critical value, with the scale re-optimised at
each candidate. The Wald interval for a positive parameter like ``beta`` is
symmetric on the log scale, so it is always stretched upwards by the same
factor it is stretched downwards; the likelihood-ratio interval instead follows
the actual shape of the likelihood. Both need not be symmetric about the estimate -
here the upper bound sits further from the fitted value than the lower one, as
you would expect for a shape parameter from a small sample - and the
likelihood-ratio interval is invariant to how the model is parameterised.
Because it is computed from the likelihood directly it needs the original data,
so it is only available on a model fit in-process (not one restored from
``from_dict``), and is not yet supported for offset / limited-failure-population
/ zero-inflated models; for those use ``method='wald'``. ``param_cb`` also takes
``alpha_ci`` and ``bound``, like ``cb``. How both kinds of bound are computed is
explained in :doc:`Parametric Estimation`.


Creating a custom Distribution
------------------------------

Given the implementation in SurPyval, it is possible to create a new distribution and use all the
previously listed techniques. For example, the Gompertz distribution is not implemented in the
surpyval API, this however can be quickly overcome. Its cumulative hazard is
:math:`H(x) = \nu\left(e^{b x} - 1\right)` for :math:`x \geq 0`: a hazard
:math:`h(x) = \nu b\, e^{b x}` that grows exponentially with age, which is why
it is a classic model of human mortality. First, we set up a random number
generator. Because :math:`H(X)` of a random lifetime is a unit exponential,
:math:`X = \ln(1 - \ln(U)/\nu)/b` for a uniform :math:`U`.
Because SurPyval works based on the autograd numpy implementation, it is essential that you
use the autograd numpy import to make this work.

.. jupyter-execute::

    import surpyval as surv
    # IMPORTANT - Will not work with regular numpy
    from autograd import numpy as np

    def qf(u, nu, b):
        return np.log(1 - np.log(u) / nu) / b

    # Generate random values from a Gompertz distribution
    np.random.seed(1)
    x = qf(np.random.uniform(0, 1, 100), 0.2, 1.5)

Now that we have our random data set, we can fit a Gompertz distribution to it. To do so, we need
to create a Gompertz distribution class, and to do this we need the cumulative hazard function,
the names of the parameters, the bounds of the parameters, and the distribution support.

.. jupyter-execute::

    name = 'Gompertz'

    def Hf(x, *params):
        return params[0] * (np.exp(params[1] * x) - 1)

    param_names = ['nu', 'b']
    bounds = ((0, None), (0, None))
    support = (0, np.inf)
    Gompertz = surv.CustomDistribution(name, Hf, param_names, bounds, support)

The cumulative hazard function must have the signature ``(x, *params)``, and
the names ``p``, ``gamma`` and ``f0`` are reserved for the limited failure
population, offset and zero-inflation parameters. Everything else is derived:
the hazard and the density are obtained by automatically differentiating the
cumulative hazard, and the survival function is :math:`e^{-H(x)}` (see
:doc:`CustomDistribution API <univariate/custom>`).

With this now created, it is fitted like any built-in distribution. SurPyval
knows nothing about the scale of :math:`\nu` and :math:`b`, so rather than
start the optimiser at an arbitrary point it evaluates the likelihood on a
coarse grid of magnitudes for each parameter and starts from the best
combination, and it also tries the plain default (1 for a positive parameter)
as a second start (see :doc:`Parametric Estimation`).

.. jupyter-execute::

    Gompertz.fit(x)

The fit is close to the :math:`\nu = 0.2` and :math:`b = 1.5` used to simulate the data; with only 100 values the two parameters trade off against each other a little.

The ``bounds`` are enforced during the fit, including a finite interval on
both sides. If we insisted, say, that :math:`b` cannot exceed 1.2, the fit
would respect it and settle on the edge, compensating with a larger
:math:`\nu`:

.. jupyter-execute::

    GompertzCapped = surv.CustomDistribution(
        'GompertzCapped', Hf, param_names, ((0, None), (0, 1.2)), support)
    print(GompertzCapped.fit(x).params)

If we transform the data slightly, we can show that this can be used with censored and truncated data
as well.

.. jupyter-execute::

    c = np.zeros_like(x)
    # Right censor all values above 1.5
    c[x > 1.5] = 1
    x = np.where(x > 1.5, 1.5, x)
    # Left truncate: only units that survived to 0.2 were observed
    tl = 0.2
    c = c[x > tl]
    x = x[x > tl]

    model = Gompertz.fit(x=x, c=c, tl=tl)
    model

This is extraordinary! We have created a new distribution using only the cumulative hazard function, but
are able to handle arbitrary censoring and truncation. It shows the power of the SurPyval API and
functionality.

What a cumulative hazard gives you, and what it does not:

- **Available:** fitting by ``'MLE'`` (the default), ``'MPS'``, ``'MSE'`` and
  ``'MOM'``, with ``fixed``, ``init``, ``lfp``, ``offset`` (when the support
  is :math:`(0, \infty)`) and ``zi`` (when it starts at 0); the functions
  ``sf``, ``ff``, ``df``, ``hf``, ``Hf`` and ``cs``; ``mean``, ``moment`` and
  ``var``, integrated numerically from the survival function
  (:math:`E[X^m] = \int m x^{m-1} R(x)\,dx` on a positive support); confidence
  bounds (``cb`` and ``param_cb``, Wald or likelihood ratio); ``neg_ll``,
  ``aic`` and ``bic``; and ``plot``, on linear axes.
- **Not available:** ``qf``, ``random`` and ``entropy``, since a quantile
  function does not follow from :math:`H(x)` without a numerical inversion
  SurPyval does not attempt. ``how='MPP'`` needs a linearising transform that
  a custom distribution does not have, so it is refused with a
  ``ValueError``.

Credit for this idea must be given to the creators of the *lifelines* package. *lifelines* is capable
of receiving a cumulative hazard function that can then be used as a distribution to fit parameters.
However, at the time of writing it could not handle arbitrarily censored or truncated data.

Even with a user defined ``Hf()`` we can still use the confidence bounds as well. The results of this
can be seen by simply calling the plot function:

.. jupyter-execute::

    model.plot(heuristic="Turnbull")

You can see that the distribution is not linearised. This is because the Hf is not readily convertible
into the transformation function needed to do the linearisation of the CDF. The defaults are a simple
linear scale for both the x and y axis and it shows that the confidence bounds have worked nicely.

Towards the right of the plot the band becomes wide compared with the
estimate itself. That is where the data run out: almost a quarter of the
units were censored at 1.5, so there is little direct information about the
survival there, and beyond 1.5 the model is extrapolating. The numbers show
it:

.. jupyter-execute::

    for t in [0.5, 1.0, 1.5, 2.0]:
        lower, upper = model.cb(t, on='sf')[0]
        print(f"R({t}) = {model.sf(t):.3f}   95% CI [{lower:.3f}, {upper:.3f}]")

This shows the importance of inference when working with truncated and censored data, the uncertainty can be quite wide!

.. warning::
    The confidence bounds differentiate your cumulative hazard automatically,
    and for a function that grows as fast as the Gompertz's exponential,
    parameter values far from the fit can overflow and produce implausible
    bounds. Take care when using ``cb`` with a custom distribution, and check
    the bounds against the estimate as above.
