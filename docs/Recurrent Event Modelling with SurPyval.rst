Recurrent Event Modelling with SurPyval
=======================================

This section is aims to show how you can use SurPyval to model counting
processes. For the concepts and mathematics behind these models — the HPP,
the NHPP (Duane, Cox-Lewis, Crow-AMSAA), the renewal and virtual-age models,
and the mean cumulative function — see the :doc:`Recurrent Event Analysis`
page.

Recurrent Event SurPyval Modelling
----------------------------------

First, we will start with recurrent events, and a simple non-parametric model.

Non-Parametric Counting Model with Surpyval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Creating a Non-Parametric MCF with surpyval is easy. Simply collect the data and
pass it to the ``fit`` call of the ``NonParametricCounting`` class.


.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting
    import numpy as np

    x = [1, 2, 3, 4, 5, 6, 7]

    model = NonParametricCounting.fit(x)
    model.plot()

This shows the expected number of events at any time. The model is a step
function since it is non-parametric and we have made no assumptions about the
count between observed events.

The result of this is a Non-Parametric Counting model that can be used just like
all other models in surpyval. It is important to note that the ``fit`` function
takes the values of x as the *cumulative* time to the event, not the inter-arrival
time. If you do have inter-arrival data (which is sorted in the correct order)
all you need do is take the cumulative sum of the obervations along the length
of the array. For example:

.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting
    import numpy as np

    interarrival_times = [1, 1, 2, 4, 3, 1, 2, 1]
    x = np.cumsum(interarrival_times)

    model = NonParametricCounting.fit(x)

We can then use this model to estimate the number of failures at any time. For
example, let's say we wanted to know how many failures we would expect to see
after 10 units of time. We can do this by using the ``mcf`` method of the model.

.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting
    import numpy as np

    interarrival_times = [1, 1, 2, 4, 3, 1, 2, 1]
    x = np.cumsum(interarrival_times)

    model = NonParametricCounting.fit(x)
    model.mcf(10)


The above two examples use only one item, but we can get the expected number
of events based on data from any number of items. Let's say we had three items
observed until the last event. Let's do some modelling.

.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting
    import numpy as np

    x = [1, 2, 3, 4, 5, 6, 7, 1, 4, 6, 9, 2, 7, 8, 9]
    i = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]

    model = NonParametricCounting.fit(x, i=i)
    model.plot()

Here we have the expected number of events over time based on the observations
of three different items.

These functions work with censoring as well. We need to keep in mind that the
only right censored points we can have for an item is the last. This is because
it doesn't make any sense to have a right censored point followed by another
event. The same is true for left censored and truncated data. Therefore the
"timeline" for a single item must be coherent for the model to work.

Let's look at how we can use right censoring.

.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting
    import numpy as np

    x = [1, 2, 3, 4, 5, 6, 7, 1, 4, 6, 9, 2, 7, 8, 9]
    i = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    c = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

    model = NonParametricCounting.fit(x, i=i, c=c)
    model.plot()

The ``NonParametricCounting`` model also supports **left truncation** (delayed
entry): an item that was already in service before observation began only joins
the at-risk set once its entry time is reached, so events before that entry are
estimated over a smaller risk set. Pass a per-item (or scalar) entry time with
``tl``:

.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting

    x = [2, 3, 5, 3, 4, 6]
    i = [1, 1, 1, 2, 2, 2]
    c = [0, 0, 1, 0, 0, 1]

    model = NonParametricCounting.fit(x, i=i, c=c, tl=1.0)
    model.mcf(4)

Right truncation (a finite ``tr``) is not yet supported by the non-parametric
risk-set construction; for right-truncated recurrent data use a parametric
intensity model.

Let's say this data was for the time, in years,
between repairs on home air conditioners of a specific model. We can then use
this model to estimate the number of reparis we would need on a newly installed
air conditioner. Let's say we wanted to know how many repairs we would expect to see
after 8 years. We can do this by using the ``mcf`` method of the model.

If however, we wanted to know how many reparis were needed after 10 years, we
could not do so since the data only goes up to 9 years. To address this we would
instead need to use a parametric model.

Parametric Recurrent Event Models with Surpyval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Just as is the case with single event survival analysis, non-parametric models
are not always the best choice. In the case of recurrent events, we can use
parametric models to model the number of events at any time. This is done by
assuming a hazard rate for the inter-arrival times. This also has the same
limitations as per single event survival analysis. That is, given we use a
parametric representation of the hazard rate we are making assumptions about the
shape of the cumulative intensity function. This allows us to extrapolate
above the highest observed values but may not be a good fit to the data.

Let's fit a parametric model.

.. jupyter-execute::

    from surpyval.recurrent import HPP
    import numpy as np

    x = [1, 2, 3, 4, 5, 6, 7, 1, 4, 6, 9, 2, 7, 8, 9]
    i = [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3]
    c = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1]

    model = HPP.fit(x, i=i, c=c)
    model.plot()

This model is a good fit to the data, althouhg it is just a straight line. But
we can extraplotate above the highest observed value. Let's say we wanted to
know how many events would happen up to "15", we can do this with the ``cif``
method of the model.

.. jupyter-execute::

    model.cif(15)

This means that we would expect to see 7.2 events up to "15" (in whatever units
this model is in). Let's see a different example:

.. jupyter-execute::

    x = [1, 5, 8, 10, 12, 13, 13, 14]
    HPP.fit(x).plot()


This HPP model, in this case, is not a good fit to the data. This is because
the model assumes that the accumulation of events will tend to be a straight
line whereas the data appears to be increasing over time. In this case, we have
made a poor assumption in using the HPP model. Let's try another one.


.. jupyter-execute::

    from surpyval.recurrent import Duane
    x = [1, 5, 8, 10, 12, 13, 13, 14]

    model = Duane.fit(x)
    model.plot()

This is clearly a much better fit. Have a look at the api documentation to see
what other parametric models are available in SurPyval.

Inference and model checking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fitted parametric recurrence model is more than a point estimate. Every
model fit by maximum likelihood exposes the usual likelihood quantities for
comparing models — the log-likelihood and the ``aic`` / ``bic`` information
criteria (these are attributes, not methods):

.. jupyter-execute::

    print("AIC:", model.aic, " BIC:", model.bic)

It also carries the uncertainty of the fitted parameters. ``standard_errors()``
returns the standard error of each parameter (from the observed information),
and ``param_cb`` gives a confidence interval on a named parameter — computed on
a transformed scale so the interval respects the parameter's support (a rate,
for instance, cannot go negative):

.. jupyter-execute::

    print("parameters :", model.parameter_names)
    print("std errors :", model.standard_errors())
    print("alpha 95% CI:", model.param_cb("alpha"))

That parameter uncertainty propagates to the fitted curve. ``plot()`` draws a
delta-method confidence band around the cumulative intensity function, and
``cif_cb`` returns the band directly:

.. jupyter-execute::

    model.plot()

Having a model is not the same as having a *good* model. SurPyval provides three
complementary checks. First, a **trend test** on the fitted data — the same
Laplace / Military-Handbook tests used to decide whether a time-varying
intensity was warranted in the first place:

.. jupyter-execute::

    result = model.trend_test()
    print(result.trend, "trend, p-value", round(result.p_value, 3))

Second, **residuals**. Via the time-rescaling theorem, the fitted model turns
the event times into what should be a unit-rate Poisson process, so the
cumulative-hazard residuals are (under the model) an i.i.d. Exp(1) sample with
mean 1:

.. jupyter-execute::

    print(model.residuals().round(3))

Third, a **goodness-of-fit test**. ``cramer_von_mises`` measures how far the
transformed event times fall from uniformity and calibrates the statistic with
a parametric bootstrap, so its p-value accounts for the parameters having been
estimated. A large p-value means the fitted intensity is consistent with the
data:

.. jupyter-execute::
    :stderr:

    gof = model.cramer_von_mises(n_boot=100, seed=1)
    print("statistic", round(gof.statistic, 3), " p-value", round(gof.p_value, 3))

The same inference and diagnostic methods are available on the
proportional-intensity regression models and the renewal models below.

Renewal Modelling in SurPyval
-----------------------------

In contrast to the above, where the cumulative count of events are assumed to
have an underlying rate of occurence, renewal models assume that there is an
underlying distribution of the inter-arrival times where each subsequent
inter-arrival time is affected by some restoration factor.

Generalised Renewal Process with SurPyval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generalized Renewal Process modelling is simple with SurPyval:

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval.recurrent import GeneralizedRenewal, NonParametricCounting
    import numpy as np

    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])

    model = GeneralizedRenewal.fit(x, dist=Weibull)
    model

We cannot plot the cumulative intensity function of the model since it does
not have a closed form solution. We can however plot the cumulative intensity
function of a monte carlo simulation of the model. Let's do that and compare
it to a non-parametric description of the MCF:

.. jupyter-execute::

    np_model = model.count_terminated_simulation(len(x), 5000)
    ax = np_model.plot()
    NonParametricCounting.fit(x).plot(ax=ax)

We have simulated the model we created up to the number of failures we saw in
the data with the ``count_terminated_simulation`` method. This method takes
two arguments, the first is the number of failures to simulate up to and the
second is the number of simulations to run. The more simulations you run the
more accurate the model will be. The method returns a ``NonParametricCounting``
model that can be used to plot the results.

You can see that the cumulative intensity function of the model is a very good
fit to the data. You can also see that it is "wavy." This is because the
underlying distribution is Weibull with a reasonably high shape parameter. This
means that the first inter-arrival time is going to be within a relatively
narrow period. After the first failure, and the subsequent restoration, the
next inter-arrival time is going to be in a larger range since it will be the
sum of the first inter-arrival time and the second inter-arrival time. This
process continues for each subsequent inter-arrival time. Eventually the waves
will become smaller as the mixing of previous inter-arrival times makes the
spread of the next inter-arrival time larger and larger. It looks essentially
like a smooth line at the higher values.

SurPyval uses the Kijima Type i as the default. Let's change this to
Kijima Type ii and see what happens.

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval.recurrent import GeneralizedRenewal, NonParametricCounting
    import numpy as np

    x = np.array([1, 2, 3, 4, 4.5, 5, 5.5, 5.7, 6])

    model = GeneralizedRenewal.fit(x, dist=Weibull, kijima="ii")

    np_model = model.count_terminated_simulation(len(x), 5000)
    ax = np_model.plot()
    NonParametricCounting.fit(x).plot(ax=ax)

We can see that this model is not as good a fit as the kijima type i model.
This implies that the restoration that is done only repairs damage done since
the last event. We could then use this model, via the non-parametric
simulations of it, to estimate the number of events up to a given time.

G1 Renewal Process with SurPyval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

G1 Modelling can easily be done with SurPyval:

.. jupyter-execute::

    from surpyval import Exponential
    from surpyval.recurrent import GeneralizedOneRenewal
    import numpy as np
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

    model = GeneralizedOneRenewal.fit(x, dist=Exponential)
    model


This data is from [1]_ and shows the inter-arrival times, and not the total
time to each event. We therefore need to take the cumulative sum of all the
times before passing it to the ``fit`` method. These are the same results as
achieved by Kaminskiy and Krivtsov in their paper [2]_ introducing the G1
Renewal Process.

Surpyval allows you to use any distribution in SurPyval as the underlying
distribution. Let's use the same data with a Weibull G1 Renewal Process.


.. jupyter-execute::
    :stderr:

    from surpyval import Weibull
    from surpyval.recurrent import GeneralizedOneRenewal, NonParametricCounting
    import numpy as np
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()

    model = GeneralizedOneRenewal.fit(x, dist=Weibull)
    model

We can see that the restoration factor is quite similar. What is interesting is
that the underlying Weibull distribution has a shape parameter greater than 1.
This indicates that the underlying distribution is not exponential. Since the
G1 Renewal Process does not have a closed form solution for the cif we can
create a non-parametric model from a monte carlo simulation. Let's do this and
compare it to the data MCF.

.. jupyter-execute::

    np_model = model.time_terminated_simulation(250, 1000)
    np_model.plot()
    NonParametricCounting.fit(x).plot()


In this code we created a ``NonParametricCounting`` model using the G1 Models
``time_terminated_simulation`` method. This method takes two arguments, the
first is the time to run the simulation to while the second is the number of
simulations to run. The more simulations you run the more accurate the model
will be. The method returns a ``NonParametricCounting`` model that can be
used to plot the results. We then also add the raw data to the plot for
comparison.

The image above shows that the blue line (the model from the simulation) is in
very good agreement to the data. This is a good indication that the underlying
distribution is Weibull and that the repair effectiveness has been correctly
estimated.

Arithmetic Reduction Models (ARA / ARI) with SurPyval
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``ARA`` (Arithmetic Reduction of Age) and ``ARI`` (Arithmetic Reduction of
Intensity) models make the *memory* of a repair explicit through an integer
``m``: how many prior failures an intervention acts on. ``ARA`` reduces a
virtual age; ``ARI`` reduces the intensity directly. Both are fitted with the
same API as the other renewal models, plus the ``m`` argument.

.. jupyter-execute::

    from surpyval import Weibull
    from surpyval.recurrent import ARA
    import numpy as np

    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2.2, 5, 7.5, 9, 12])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])

    model = ARA.fit(x, i, dist=Weibull, m=1)
    model

The ``Repair Efficiency`` :math:`\rho` reported here plays the role of
:math:`1 - q`: a value near 1 is close to as-good-as-new, a value near 0 is
as-bad-as-old. ``ARI`` fits the same way but with an intensity (counting
process) baseline such as ``CrowAMSAA``:

.. jupyter-execute::
    :stderr:

    from surpyval.recurrent import ARI, CrowAMSAA

    model = ARI.fit(x, i, dist=CrowAMSAA, m=1)
    model

Checking a renewal model
~~~~~~~~~~~~~~~~~~~~~~~~~~

The renewal and virtual-age models carry the same diagnostics as the intensity
models. Because they have no marginal cumulative intensity, the residuals come
from the *conditional* intensity — the cumulative hazard accumulated over each
interval given the model's virtual age — but under a well-specified model they
are still an i.i.d. Exp(1) sample:

.. jupyter-execute::
    :stderr:

    from surpyval import Weibull
    from surpyval.recurrent import GeneralizedRenewal
    import numpy as np

    x = np.array([1, 3, 6, 9, 10, 1.4, 3, 6.7, 8.9, 11, 1, 2.2, 5, 7.5, 9, 12])
    i = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3])

    model = GeneralizedRenewal.fit(x, i, dist=Weibull, kijima="i")
    print("residual mean :", round(model.residuals().mean(), 3))
    print("trend         :", model.trend_test().trend)

    gof = model.cramer_von_mises(n_boot=50, seed=1)
    print("CvM p-value   :", round(gof.p_value, 3))

The Cramér–von Mises bootstrap refits the (multi-start) imperfect-repair model
once per replicate, so it is noticeably slower than the residual checks; keep
``n_boot`` modest while exploring.

Gapped (multi-window) observation
---------------------------------

Sometimes an item is only observed over a few disjoint windows, with gaps in
between during which failures may occur but are not recorded — a vehicle seen
only while it is in the depot, for instance. Passing a ``windows`` mapping to an
intensity model tells SurPyval each item's observation windows; every event you
pass is then an observed failure (``c=0``), and the windows supply the
end-of-window censoring automatically.

.. jupyter-execute::

    from surpyval.recurrent import CrowAMSAA
    import numpy as np

    # one item, observed on [0, 12] and [20, 40] with an unobserved gap
    x = np.array([3, 7, 10, 25, 33, 38])
    i = np.array([1, 1, 1, 1, 1, 1])

    model = CrowAMSAA.fit(x, i, windows={1: [(0, 12), (20, 40)]})
    model.params

Because Poisson event counts over disjoint windows are independent, each window
is fitted as its own observation period; the intensity likelihood and the
non-parametric MCF at-risk set both account for the gaps with no extra work. The
virtual-age / renewal models reject gapped data, since the virtual age at the
start of a later window depends on the unobserved failures during the gap.

Competing risks: marked recurrent events
-----------------------------------------

When events come in several mutually exclusive types, attach a **mark** ``e`` to
each event (use ``None`` for censoring rows). The non-parametric
``CauseSpecificMCF`` gives one mean cumulative function per cause, sharing the
at-risk set across causes:

.. jupyter-execute::

    from surpyval.recurrent import CauseSpecificMCF

    x = [3, 1, 5, 2, 4, 6]
    i = [1, 1, 1, 2, 2, 2]
    c = [0, 0, 1, 0, 0, 1]
    e = ["A", "B", None, "A", "A", None]   # None marks the censoring rows

    model = CauseSpecificMCF.fit(x, i, c, e=e)
    ax = model.plot()

For a parametric picture, ``CauseSpecificNHPP`` fits one intensity model per
cause (``CrowAMSAA`` by default). A marked Poisson process decomposes into
independent thinned Poisson processes, so each cause is fitted to its own events
over the full observation window — other-cause events are ignored, exactly like
a censored period:

.. jupyter-execute::
    :stderr:

    from surpyval.recurrent import CauseSpecificNHPP

    model = CauseSpecificNHPP.fit(x, i, c, e=e)
    print("causes         :", model.event_types)
    print("cause A params :", model.models["A"].params.round(3))
    print("total cif at 6 :", round(float(model.total_cif(6.0)), 3))

Each ``model.models[cause]`` is an ordinary fitted recurrence model, so it
carries the full ``cif`` / ``iif``, inference and diagnostic behaviour shown
above; ``total_cif`` sums the causes for the overall event intensity.

Saving and loading a fitted model
---------------------------------

Every fitted recurrence model can be serialised to a plain dictionary or a
JSON file and rebuilt later. The intensity model is stateless, so only its name
and the fitted parameters are stored (for the nonparametric MCF, the step
arrays); the reloaded model reproduces every prediction exactly. This works for
the parametric intensity fits (``CrowAMSAA`` / ``Duane`` / ``Cox-Lewis`` /
``HPP``), the nonparametric MCF, the proportional-intensity regression, the two
cause-specific containers, and the renewal / imperfect-repair models
(``RenewalModel`` — generalized renewal, G1 renewal, ARA, ARI).

.. jupyter-execute::

    import numpy as np
    from surpyval.recurrent import CrowAMSAA
    from surpyval.recurrent.parametric.parametric_recurrence import (
        ParametricRecurrenceModel,
    )

    events = np.sort(np.random.default_rng(0).uniform(0, 1000, 40))
    fitted = CrowAMSAA.fit(events)

    blob = fitted.to_dict()                                   # -> dict
    restored = ParametricRecurrenceModel.from_dict(blob)      # <- dict
    t = np.array([100.0, 500.0, 900.0])
    print("match:", np.allclose(fitted.cif(t), restored.cif(t)))

Use ``to_json`` / ``from_json`` for a file directly. The likelihood-inference
state (the fitted data and the log-likelihood) is not serialised, so a reloaded
model behaves like a ``from_params`` one for confidence bounds and diagnostics
— re-fit if you need those.

References
----------

.. [1] Basu, A.P. and Rigdon, S.E., 2000. Statistical methods for the reliability of repairable systems. John Wiley & Sons.

.. [2] Kaminskiy, M.P. and Krivtsov, V.V., 2010. G1-renewal process as repairable system model. Reliability: Theory & Applications, 5(3), pp.7-14.
