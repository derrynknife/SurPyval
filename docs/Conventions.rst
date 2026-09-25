
Conventions
===========

This page is the reference for the conventions the rest of the documentation relies on: how data is laid out, what each variable name means, how censoring and truncation are flagged, which functions every model provides, and how the optional structure of a parametric model (offset, limited failure population, zero-inflation) is defined. :doc:`Types of Data` explains the ideas behind censoring and truncation; this page pins down exactly how SurPyval represents them.

Data Formats
------------

The conventional formats used in SurPyval are:

- xcnt = x variables, with c as the censoring scheme, n as the counts, and t as the truncation
- xrd  = x variables, with the risk set, r, at x and the deaths, d, also at x
- xicnt = x variables, with c as the censoring scheme, n as the counts, and t as the truncation, and i as the item number

All functions in SurPyval have default handling for c and n. That is,
if these variables aren't passed, it is assumed that there was one observation
and it was a failure (``c = 0``, ``n = 1``) for every x. Truncation defaults to none (an observation
window of :math:`(-\infty, \infty)`). For recurrent event models, if i is not passed
it is assumed that it is all from the same item.

Surpyval fit() functions use the xcnt format, because it is the most general:
every combination of observed, censored and truncated data can be written in it.
Data sometimes comes in different formats, so SurPyval has utilities available to convert
from these formats into the xcnt and xrd formats (see :doc:`Data Wrangler Examples`). These other formats are:

- fs = failure time array, f, and right censored (suspended) time array, s
- fsl = fs format plus an array, l, for left censored times
- fsli = fsl format plus an array, i, of ``[lower, upper]`` pairs for interval censored data

The xcnt format
~~~~~~~~~~~~~~~

In xcnt data, *row* :math:`j` says: ":math:`n_j` items had the value :math:`x_j`, with censoring :math:`c_j`, and could only have been observed inside the window :math:`t_j = (t_{l,j}, t_{r,j}]`". Rows are independent, and any mix of censoring and truncation is allowed.

When you call ``fit()``, SurPyval validates the arrays, converts them to numpy, groups identical ``(x, c, t)`` rows (summing their counts), and sorts them. The result is held in a ``SurpyvalData`` object, which you can also build yourself to see what SurPyval makes of your data:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    data = surv.SurpyvalData(x=[1, 2, 3, [4, 5], 2], c=[0, 1, 0, 2, 1], tl=0)
    print(data)

Notice three things. The two right censored values at 2 have been merged into one row with ``n = 2``. Because one row is an interval, ``x`` is stored as two columns, with the lower and upper values equal for the rows that are not intervals. And the scalar ``tl=0`` has been expanded to a ``[tl, tr]`` row for every observation, with no right truncation (``inf``). Parametric fitters accept a ``SurpyvalData`` object directly through ``fit_from_surpyval_data``, and ``SurpyvalData.to_json`` / ``SurpyvalData.from_json`` save and restore the data itself. ``to_json()`` returns JSON text, or writes a file when given a path; ``from_json`` parses a string as JSON *text*, so to read a file pass a ``pathlib.Path``: ``SurpyvalData.from_json(Path("data.json"))``.

The xrd format
~~~~~~~~~~~~~~

The non-parametric estimators (Kaplan-Meier, Nelson-Aalen and Fleming-Harrington) think in terms of risk sets, so internally they work in the xrd format: at each distinct time ``x``, ``r`` is the number of items at risk just before ``x`` (including those that fail at ``x``) and ``d`` is the number of deaths (failures) at ``x``.

.. jupyter-execute::

    x, r, d = surv.xcnt_to_xrd(x=[1, 2, 3, 4, 5], c=[0, 1, 1, 0, 0])
    print("x:", x)
    print("r:", r)
    print("d:", d)

Converting between the two formats loses information in both directions, so the conversions have restrictions:

- ``xcnt_to_xrd`` needs data that is observed or right censored (``c`` of 0 or 1) and not right truncated. Left censored and interval censored data have no single time at which to count a death, so they need the Turnbull estimator instead.
- ``xrd_to_xcnt`` recovers the individual observed and right censored values from ``r`` and ``d``. It cannot recover left truncation: if the risk set ever *grows* between two times (items entering late), the per-item entry times are lost, and the function raises an error rather than returning a different study.
- xrd data given directly (to ``xrd_to_xcnt``, or to ``KaplanMeier.from_xrd`` and the other non-parametric ``from_xrd`` methods) must list each time once, in increasing order, as ``xcnt_to_xrd`` returns it. This is not checked: rows out of order are read in the order given, and give a wrong estimate.

The xicnt format
~~~~~~~~~~~~~~~~

Recurrent event data (items that fail, are repaired and fail again) adds ``i``, the identifier of the item each row belongs to. The conventions are:

- ``x`` is the **cumulative** time of the event on that item (time since the item was new or first observed), not the time since the previous event. If you have inter-arrival times, take their cumulative sum per item.
- Each item may have at most one right censored row (``c = 1``), which must be its last: it marks the end of that item's observation. Likewise, at most one left censored row, which must be its first.
- Truncation (``tl``/``tr``, or ``t``) defines each item's observation window, so it must be the same for every row of an item and contain all of that item's events. An item with no left truncation is taken to have been observed from time 0. Items observed over several separate periods can be described with ``windows``.
- ``n`` greater than one is only allowed on interval or left censored rows (several events known to have occurred in the same interval).

.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting

    x = [11, 24, 40,  9, 33,  5, 18, 41]   # cumulative event times
    i = [ 1,  1,  1,  2,  2,  3,  3,  3]   # item identifiers
    c = [ 0,  0,  1,  0,  1,  0,  0,  1]   # 1 = end of observation of the item

    mcf = NonParametricCounting.fit(x=x, i=i, c=c)
    print(mcf.mcf([10, 30]))

Variable Names
--------------

The same variable names are used everywhere in SurPyval, in code and in these pages. For single event survival models, the xcnt format, they mean:

- x  = The random variable (time, stress etc.) array. For interval censored rows the entry is a pair ``[lower, upper]``.
- xl = The random variable (time, stress etc.) array for the left interval of interval censored data.
- xr = The random variable (time, stress etc.) array for the right interval of interval censored data.
- c  = An array with the censor flag for each x
- n  = The count array associated with x: the number of items that share the row's value, flag and truncation. Must be positive integers (it is a count, not a weight).
- t  = the truncation values for the left and right truncation at x (must be two dim, or use tl and tr instead)
- tl = one dimensional array or scalar value. If an array it is the value at which each value of x is left truncated. If a scalar all values of x are left truncated at the same value.
- tr = one dimensional array or scalar value. If an array it is the value at which each value of x is right truncated. If a scalar all values of x are right truncated at the same value.
- Z = the multi-dimensional array of covariates for each x, one row per observation, used by the regression models.

``x`` cannot be combined with ``xl``/``xr``, and ``t`` cannot be combined with ``tl``/``tr``. ``tl`` and ``tr`` can each be used alone.

Non-parametric models are better defined in the "xrd" format. These are taken to mean:

- x = array of random variables
- r = array with the number of items at risk for each value in x
- d = array with the number of failures/deaths at each value in x

For recurrent event data it is necessary to know which item has a subsequent event. For this we use all
the same as described above with the addition of:

- i = array with the item number/identifier for each value in x

Other areas of the package add a few more names, always with the same meaning:

- e = the event type, or cause, of each row, for competing risks (``None`` for a censored row with no attributed cause).
- y = the measured degradation value at each ``x``, for degradation models (with ``i`` identifying the unit).


Censoring Flag Conventions
--------------------------

For the censoring values, surpyval uses the convention used in Meeker and Escobar, that is:

- -1 = left
- 0 = failure / event
- 1 = right
- 2 = interval censoring. Must have left and right value in x

This convention gives an intuitive feel for the placement of the data on a timeline: the failure is at the recorded value (0), somewhere to its left (-1), somewhere to its right (1), or between two values (2).

.. list-table::
   :header-rows: 1
   :widths: 10 25 65

   * - ``c``
     - Meaning
     - What is known about the value :math:`X` of row :math:`j`
   * - -1
     - left censored
     - :math:`X \le x_j`
   * - 0
     - observed
     - :math:`X = x_j`
   * - 1
     - right censored
     - :math:`X > x_j`
   * - 2
     - interval censored
     - :math:`x_{l,j} < X \le x_{r,j}`

The same flags are used throughout the package: by the regression models, the recurrent event models (where ``c = 1`` marks the end of an item's observation), the copulas (one censoring array per dimension), and the start-stop (time-varying covariate) form of the Cox model, where ``c = 0`` is an event at the end of the interval and ``c = 1`` is right censored. The one variation is in competing risks, where ``c`` may be omitted because a missing cause (``e`` of ``None``) already says that a row is censored.

Truncation conventions
~~~~~~~~~~~~~~~~~~~~~~

- The observation window of row :math:`j` is :math:`(t_{l,j}, t_{r,j}]`. With no truncation it is :math:`(-\infty, \infty)`.
- An observed value must lie strictly above its left truncation bound, :math:`t_{l,j} < x_j`, and at or below its right truncation bound. A value equal to its own left truncation bound would have a zero-length observation window, and is rejected.
- Non-parametric risk sets use the (entry, exit] convention: an item with window starting at :math:`t_l` and value :math:`x` is at risk at time :math:`s` when :math:`t_l < s \le x`. An item entering at exactly an event time is not at risk for that event.
- A censored value is only known to lie inside its own truncation window, so a left censored row with a finite :math:`t_l` is used as the interval :math:`(t_l, x]`, and a right censored row with a finite :math:`t_r` as :math:`(x, t_r]`. This happens automatically.


Function Conventions
--------------------

The conventions for single event SurPyval models are that each object returned from a :code:`fit()` call has the ability to compute the following:

- :code:`df()` - The density function
- :code:`ff()` - The CDF
- :code:`sf()` - The survival function, or reliability function
- :code:`hf()` - The (instantaneous) hazard function
- :code:`Hf()` - The cumulative hazard function

A ``MixtureModel`` is the exception: it has no ``hf()`` or ``qf()``. These are the functions :math:`f(x)`, :math:`F(x)`, :math:`R(x)`, :math:`h(x)` and :math:`H(x)`; how each can be computed from any of the others is shown in :doc:`Handy References - Aide-mémoire`. One caution: a non-parametric estimate (Kaplan-Meier and the others) is a step function, so its ``hf()`` and ``df()`` are the sizes of the jumps between the points you ask for, not rates, and change with how finely you space them; ``smoothed_hf()`` gives a kernel-smoothed hazard rate instead. Most single event models also provide:

- :code:`qf()` - The quantile function, the inverse of the CDF. ``qf(0.1)`` is the B10 life.
- :code:`cb()` - Confidence bounds on a function (the survival function by default).
- :code:`mean()` - The mean.
- :code:`random()` - Random samples from the model.
- :code:`plot()` - A plot of the model against the data it was fitted to.

For a parametric model, ``params`` holds the fitted parameters in the order given by ``model.dist.param_names`` (each is also an attribute, e.g. ``model.alpha``), and those names are what ``fixed={...}`` refers to. Fitted parametric models also have ``neg_ll()``, ``aic()``, ``aic_c()`` and ``bic()`` for comparing fits, ``cs(x, X)`` for the conditional survival :math:`R(x + X)/R(X)`, ``var()``, ``moment()`` and ``entropy()``, and ``param_cb()`` for confidence bounds on the parameters themselves. Non-parametric models add, among others, ``rmst()`` (restricted mean survival time) and simultaneous confidence bands with ``band()``; see :doc:`Parametric SurPyval Modelling` and :doc:`Non-Parametric SurPyval Modelling`.

Models from other areas follow the same pattern with one extra argument:

- Regression models take the covariates as the second argument, ``model.sf(x, Z)``.
- Competing risks models take the cause, ``model.cif(x, event)``.

The conventions for recurrent event SurPyval models are that each object returned from a :code:`fit()` call has the ability to compute the following:

- :code:`iif()` - instantaneous intensity function
- :code:`cif()` - cumulative intensity function

The cumulative intensity is the expected number of events by time :math:`x`. It is the parametric counterpart of the mean cumulative function, which the non-parametric ``NonParametricCounting`` model provides as :code:`mcf()`.

.. jupyter-execute::

    model = surv.Weibull.fit([3, 4, 5, 6, 7, 8, 9, 10])
    print("parameter names :", model.dist.param_names)
    print("params          :", model.params)
    print("R(5), F(5)      :", model.sf(5), model.ff(5))
    print("h(5), H(5)      :", model.hf(5), model.Hf(5))
    print("median          :", model.qf(0.5))

Offset, limited failure population and zero-inflation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A parametric fit can add up to three pieces of structure to the base distribution, each switched on by a keyword of ``fit()`` (or a keyword of ``from_params()``):

- **Offset** (``offset=True``, parameter :math:`\gamma`): nothing can fail before :math:`\gamma`, a failure-free period or minimum life. This is the "three parameter Weibull". It is only available for distributions supported on the half line :math:`[0, \infty)`.
- **Limited failure population** (``lfp=True``, parameter :math:`p`): only a proportion :math:`p` of the population is susceptible to the failure mode; the remaining :math:`1 - p` never fails. This is also called a defective subpopulation or cure model.
- **Zero-inflation** (``zi=True``, parameter :math:`f_0`): a proportion :math:`f_0` fails at exactly :math:`x = 0` (dead on arrival). Zero-inflated data contains exact zeros, which an ordinary lifetime distribution cannot produce. It is only available for distributions whose support starts at 0.

With :math:`F_0` and :math:`R_0` the base distribution, the full model is

.. math::

    F(x) = f_0 + (p - f_0)\, F_0(x - \gamma), \qquad
    R(x) = 1 - p + (p - f_0)\, R_0(x - \gamma)

and the defaults :math:`\gamma = 0`, :math:`p = 1` and :math:`f_0 = 0` give back the base distribution. The conventions to remember are:

- :math:`p` is the proportion that **ever fails**, including the dead-on-arrival fraction, so :math:`F(\infty) = p` and :math:`f_0 \le p`. Writing it this way means every function (``ff``, ``sf``, ``df``, the likelihood, ``mean``, ``qf``) uses the same constant :math:`p - f_0` for the continuous part, and they are mutually consistent.
- The zero-inflation mass sits at :math:`x = 0`, even when there is an offset.
- Below the offset, :math:`F(x) = f_0` and :math:`R(x) = 1 - f_0`: the base distribution has not started.
- ``qf(q)`` is infinite for :math:`q \ge p` (that fraction of the population never fails), and 0 for :math:`q \le f_0`.
- ``mean()`` of an LFP or zero-inflated model is :math:`(p - f_0)\,E[\gamma + X_0]`, the mean over the whole population counting non-failures and zeros as contributing nothing. It is *not* the mean life of the units that fail.
- ``random()`` of an LFP or zero-inflated model returns xcnt data, ``(x, c, n, t)``, rather than a plain array, because the units that never fail come back as right censored observations.
- LFP and zero-inflated models can only be fitted by maximum likelihood (``how="MLE"``).

.. jupyter-execute::

    variants = {
        "base":            surv.Weibull.from_params([10, 2]),
        "offset gamma=5":  surv.Weibull.from_params([10, 2], gamma=5),
        "lfp p=0.3":       surv.Weibull.from_params([10, 2], p=0.3),
        "zi f0=0.1":       surv.Weibull.from_params([10, 2], f0=0.1),
        "lfp + zi":        surv.Weibull.from_params([10, 2], p=0.3, f0=0.1),
    }
    print("                  F(0)    F(5)    F(15)   F(1e6)")
    for name, m in variants.items():
        print(f"{name:16s}", np.round(m.ff([0, 5, 15, 1e6]), 4))

Reading across the rows: the offset model has not started by 5; the LFP model levels off at 0.3; the zero-inflated model starts at 0.1; and the combined model starts at 0.1 and levels off at 0.3, because the 0.1 at zero is part of the 0.3 that ever fails. See :doc:`Parametric SurPyval Modelling` for fitting these models to data, and :doc:`applications` for an LFP model used to design a burn-in test.

Saving and Loading Models
~~~~~~~~~~~~~~~~~~~~~~~~~

Almost every fitted SurPyval model can be saved and restored (the exceptions are listed at the end of this section):

- ``model.to_dict()`` returns a dictionary of plain Python types (strings, numbers, lists), so it can be written as JSON or stored directly in a document database such as MongoDB.
- ``model.to_json(path)`` writes that dictionary to a JSON file.
- ``surpyval.from_dict(d)`` and ``surpyval.from_json(path)`` restore a model **of whichever class wrote it**. You do not need to know whether the file holds a Weibull, a Kaplan-Meier estimate, a Cox model or a recurrence model; the readers work it out from the dictionary.
- Each model class also has its own ``from_dict`` / ``from_json`` for when the class is known in advance (for example ``surv.Parametric.from_dict`` or ``surv.NonParametric.from_dict``; note these are the *model* classes, not fitters such as ``surv.Weibull`` or ``surv.KaplanMeier``). They raise a ``ValueError`` if handed a dictionary written by a different class.

.. jupyter-execute::

    import os
    import tempfile

    weibull = surv.Weibull.fit([3, 4, 5, 6, 7, 8, 9, 10])
    km = surv.KaplanMeier.fit([3, 4, 5, 6, 7, 8, 9, 10], c=[0, 0, 1, 0, 0, 1, 0, 0])

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "km.json")
        km.to_json(path)
        restored_km = surv.from_json(path)

    restored_weibull = surv.from_dict(weibull.to_dict())
    print(type(restored_weibull).__name__, restored_weibull.params)
    print(type(restored_km).__name__, restored_km.sf(6), km.sf(6))

Every dictionary carries a ``"schema"`` version number. A file written by a newer version of SurPyval than the one installed is refused with an error asking you to upgrade, rather than being misread.

A restored model keeps what it needs to make predictions, but by default **not the data it was fitted to**. A univariate parametric model also keeps its covariance matrix, so ``cb`` (Wald bounds) works, and its fitted negative log-likelihood, so ``neg_ll()`` and ``aic()`` work; a parametric regression model keeps its covariance too. Anything that needs the data, such as ``plot()``, the sample-size-based criteria (``bic()``, ``aic_c()``), bootstrap or likelihood-ratio confidence bounds, residuals and diagnostics, raises an error on a restored model. What each family keeps is described on its how-to page; recurrent-event models, for example, keep no covariance, so their ``cif_cb`` needs a refit.

The univariate parametric and non-parametric models can carry their data with them: ``to_dict(with_data=True)`` adds the ``x``, ``c``, ``n`` and ``t`` arrays, and a model restored from that dictionary has ``plot()``, ``bic()`` and ``aic_c()`` (parametric) or ``bootstrap_cb()`` (non-parametric) again. ``to_json(path)`` always leaves the data out; to keep it in a file, write ``json.dump(model.to_dict(with_data=True), f)``. Likelihood-ratio bounds (``method="lr"``) are the exception: they are only available on the model as originally fitted, even when the data was saved.

.. jupyter-execute::

    try:
        restored_weibull.bic()
    except ValueError as err:
        print("without the data:", str(err)[:60], "...")

    with_data = surv.from_dict(weibull.to_dict(with_data=True))
    print("with the data    :", with_data.bic(), weibull.bic())

A few models cannot be saved, and say so when ``to_dict`` is called: a stratified Cox model, an accelerated-life model with a life model of your own, a regression fitted with a formula that uses a data-dependent transform such as ``scale()``, and a copula of a custom family. A model of a ``CustomDistribution``, or of a distribution made with ``Discretize``, is written without complaint but cannot be read back, because the reader only knows SurPyval's own distributions; keep its parameters and rebuild it with ``from_params``. Keep the data (for example with ``SurpyvalData.to_json``) whenever you may need to refit. The full API is in :doc:`surpyval.serialisation`.
