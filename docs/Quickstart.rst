Quickstart
==========


So, you know what survival analysis is and you just want to see what this can do.

Everything in *SurPyval* follows one pattern. A **fitter** (``surv.Weibull``,
``surv.KaplanMeier``, ``surv.CoxPH``, ...) has a ``fit()`` method that takes
your data and returns a fitted **model**. You then ask the model questions:
what fraction survives to time :math:`t`, what is the median life, what is the
hazard, and so on. Once you know that pattern, and the way data is passed in
(the ``x``, ``c``, ``n``, ``t`` arrays described in :doc:`Types of Data` and
:doc:`Conventions`), every family of models in the package works the same way.

Your first fit
--------------

Here are 23 failure times (the classic ball-bearing data, in millions of
revolutions). We fit a Weibull distribution to them:

.. jupyter-execute::

    import surpyval as surv

    x = [17.88, 28.92, 33, 41.52, 42.12, 45.6, 48.4, 51.84,
     51.96, 54.12, 55.56, 67.8, 68.64, 68.64, 68.88, 84.12,
     93.12, 98.64, 105.12, 105.84, 127.92, 128.04, 173.4]

    model = surv.Weibull.fit(x)
    model.plot()

This gives us the Weibull plot, which was created when the documentation
was built, so it is guaranteed to reflect the current version of *SurPyval*.
The points are a non-parametric estimate of the CDF made directly from the
data; the line is the fitted Weibull, with its confidence bounds. The axes are
scaled so that a Weibull distribution is a straight line, so points that
follow the line are evidence that the Weibull is a reasonable choice.

The fitted model can now be interrogated. ``params`` holds the parameters (for
the Weibull, the scale :math:`\alpha` and the shape :math:`\beta`), and every
model has the same set of functions of time:

.. jupyter-execute::

    print(model)
    print("Parameters                 :", model.params)
    print("Survival to 50, R(50)      :", model.sf(50))
    print("Probability failed by 50   :", model.ff(50))
    print("B10 life (10% failed)      :", model.qf(0.1))
    print("Mean life                  :", model.mean())
    print("95% bounds on R(50)        :", model.cb(50))

``sf`` is the survival (reliability) function, ``ff`` the CDF (the
probability of having failed), ``qf`` the quantile function (its inverse), and
``cb`` gives confidence bounds; by default two-sided 95% bounds on the survival
function. The shape :math:`\beta \approx 2.1` is greater than one, which says
the hazard is increasing: these bearings wear out. The full list of functions
every model shares is in :doc:`Conventions`.

Censored data: one extra argument
---------------------------------

Real data is rarely this complete. Suppose the test had been stopped at 100
million revolutions: the six bearings still running at that point have not
failed, all we know is that their lives are *longer than* 100. These are
**right censored** observations, flagged with ``c = 1`` (an observed failure
is ``c = 0``):

.. jupyter-execute::

    import numpy as np

    x = np.array(x)
    c = (x > 100).astype(int)       # 1 = still running when the test stopped
    x_test = np.minimum(x, 100)     # all we saw of those units was 100

    censored_model = surv.Weibull.fit(x_test, c)
    naive_model = surv.Weibull.fit(x_test)   # WRONG: treats 100 as a failure

    print("using the censoring flags :", censored_model.params)
    print("ignoring them             :", naive_model.params)

The censored fit recovers parameters close to the fit on the complete data.
The naive fit, which pretends the six survivors failed at exactly 100,
underestimates the scale :math:`\alpha` (the characteristic life) and
overstates the shape :math:`\beta`: it concludes that the bearings wear out
sooner and more abruptly than they do. Throwing away information about
survivors *always* biases a life estimate downwards, which is why getting
censoring right is the heart of survival analysis. :doc:`Types of Data` explains
every kind of censoring and truncation SurPyval supports.

A tour of the model families
----------------------------

The same pattern carries across every area of the package. Each example below
is deliberately small; follow the links for the theory ("Analysis" pages) and
the practical how-to ("Modelling" pages) of each area.

Non-parametric estimation
~~~~~~~~~~~~~~~~~~~~~~~~~

Non-parametric estimators make no assumption about the shape of the
distribution: they describe the data exactly as it is. They are the natural
first look at a data set and the yardstick for checking a parametric fit, but
they cannot extrapolate beyond the data. Kaplan-Meier handles observed and
right censored data (with left truncation); Turnbull handles every combination
of censoring and truncation.

.. jupyter-execute::

    km = surv.KaplanMeier.fit(x_test, c)
    print("Kaplan-Meier R(50), R(90):", km.sf([50, 90]))

Theory: :doc:`Non-Parametric Estimation`. How-to:
:doc:`Non-Parametric SurPyval Modelling`.

Parametric estimation
~~~~~~~~~~~~~~~~~~~~~

Parametric models summarise the data with a few parameters and, unlike the
non-parametric estimators, can extrapolate: a B1 life or a warranty-period
failure fraction usually sits in the tail where there is little data. If you
are unsure which distribution to use, ``fit_best`` fits a set of candidates and
returns the one with the best information criterion (AIC by default):

.. jupyter-execute::

    best = surv.fit_best(x_test, c,
                         include=["Weibull", "LogNormal", "Gamma",
                                  "LogLogistic", "Exponential"])
    print(best)

An information criterion ranks candidates; it does not prove that any of them
is right, so always look at the probability plot of the winner as well. Beyond
the choice of distribution, parametric fits can add a failure-free offset
(``offset=True``), a sub-population that never fails (``lfp=True``) or a
fraction that fails at time zero (``zi=True``); these are defined in
:doc:`Conventions`. Theory and estimation methods (MLE, MPS, MPP, MSE, MOM):
:doc:`Parametric Estimation`. How-to: :doc:`Parametric SurPyval Modelling`.
The available distributions are listed in the API reference,
:doc:`surpyval.parametric`.

Regression
~~~~~~~~~~

Regression models let the life depend on covariates ``Z`` (a stress, a
treatment, a design option). Each row of ``Z`` holds the covariates of the
matching row of ``x``. Here half the units run in condition ``Z = 1``, which
lengthens life by a factor of :math:`e^{0.7} \approx 2`:

.. jupyter-execute::

    rng = np.random.default_rng(1)
    Z = rng.integers(0, 2, (100, 1)).astype(float)
    t = 50 * rng.weibull(1.5, 100) * np.exp(0.7 * Z[:, 0])
    c_reg = (t > 80).astype(int)          # test stopped at 80
    t = np.minimum(t, 80)

    aft = surv.WeibullAFT.fit(t, Z, c_reg)
    print(aft)
    print("R(40) for Z=0 and Z=1:", aft.sf(40, np.array([[0.0], [1.0]])))

The accelerated failure time (AFT) model multiplies *time* by
:math:`e^{\beta' Z}`, so the fitted ``beta_0`` of about :math:`-0.6` says that
condition 1 runs the clock at about :math:`e^{-0.6} \approx 0.55` of the speed
of condition 0. Other families act on the hazard (``WeibullPH``,
``CoxPH``), on the odds (``WeibullPO``) or add to the hazard
(``AdditiveHazards``). Theory: :doc:`regression analysis`. How-to:
:doc:`Regression Modelling with SurPyval`.

Competing risks
~~~~~~~~~~~~~~~

When an item can fail from one of several causes and the first one ends its
life, pass the cause of each failure as ``e`` (use ``None`` for censored
rows). The cumulative incidence function (CIF) gives the probability of
having failed from each cause by time :math:`t`:

.. jupyter-execute::

    from surpyval.univariate.competing_risks import CompetingRisks

    rng = np.random.default_rng(2)
    t_wear = 60 * rng.weibull(3.0, 200)       # when wear-out would occur
    t_shock = rng.exponential(120, 200)       # when a random shock would occur
    x_cr = np.minimum(t_wear, t_shock)        # whichever comes first
    e = np.where(t_wear < t_shock, "wear", "shock")
    c_cr = (x_cr > 70).astype(int)            # test stopped at 70
    x_cr = np.minimum(x_cr, 70)
    e = np.where(c_cr == 1, None, e)          # censored rows have no cause

    cr = CompetingRisks.fit(x_cr, e=e, c=c_cr)
    print("P(failed from wear by 50)  :", cr.cif(50, event="wear"))
    print("P(failed from shock by 50) :", cr.cif(50, event="shock"))

Theory, including why a Kaplan-Meier fit to one cause (censoring the others)
does not estimate that cause's incidence: :doc:`Competing Risks Analysis`. How-to, including Fine-Gray
regression: :doc:`Competing Risks SurPyval Modelling`.

Recurrent events
~~~~~~~~~~~~~~~~

Repairable items fail, are repaired and fail again. Recurrent event data adds
an item identifier ``i`` to each row; ``x`` is the *cumulative* time of each
event on that item, and the last row of an item is usually the end of its
observation, flagged right censored (``c = 1``):

.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting, CrowAMSAA

    x_rec = [11, 24, 40,  9, 33,  5, 18, 41]
    i_rec = [ 1,  1,  1,  2,  2,  3,  3,  3]   # which item each row is from
    c_rec = [ 0,  0,  1,  0,  1,  0,  0,  1]   # 1 = end of observation

    mcf = NonParametricCounting.fit(x=x_rec, i=i_rec, c=c_rec)
    crow = CrowAMSAA.fit(x=x_rec, i=i_rec, c=c_rec)
    print("MCF at 30 (non-parametric)       :", mcf.mcf(30))
    print("Expected events by 50 (Crow-AMSAA):", crow.cif(50))

The mean cumulative function (MCF) is the expected number of events per item
by time :math:`t`; the Crow-AMSAA model is a parametric non-homogeneous
Poisson process whose ``cif`` is the same quantity. Theory:
:doc:`Recurrent Event Analysis` and :doc:`Recurrent Event Regression Analysis`.
How-to: :doc:`Recurrent Event Modelling with SurPyval` and
:doc:`Recurrent Event Regression Modelling with SurPyval`.

Degradation
~~~~~~~~~~~

Sometimes nothing has failed yet, but a measurement (wear, crack length, loss
of capacity) is drifting towards a failure threshold. Degradation analysis
fits a path to each unit (``x`` the time, ``y`` the measurement, ``i`` the
unit), turns the paths into a life distribution, and predicts the remaining
useful life of a unit from its own measurements:

.. jupyter-execute::

    from surpyval.degradation import DegradationAnalysis

    rng = np.random.default_rng(3)
    times = np.arange(0, 12, 2.0)
    xs, ys, ids = [], [], []
    for unit in range(15):
        start, rate = rng.normal(0.0, 0.5), rng.normal(1.0, 0.25)
        xs.append(times)
        ys.append(start + rate * times + rng.normal(0, 0.2, times.size))
        ids.append(np.full(times.size, unit))
    x_deg, y_deg, i_deg = (np.concatenate(v) for v in (xs, ys, ids))

    deg = DegradationAnalysis.fit(x_deg, y_deg, i_deg, threshold=15.0,
                                  path="linear")
    print("median life :", deg.qf(0.5))

    rul = deg.predict_rul([0, 2, 4, 6], [0.3, 2.9, 5.2, 8.1], random_state=0)
    print("remaining useful life of the monitored unit:", rul.rul, rul.rul_interval)

Theory: :doc:`Degradation Analysis`. How-to, including stochastic process
models and accelerated tests: :doc:`Degradation Modelling with SurPyval`.

Multivariate
~~~~~~~~~~~~

When two lifetimes are dependent (two components sharing a load, two
failure modes on the same item), a copula joins two univariate distributions
into one joint distribution, with a parameter that measures the dependence:

.. jupyter-execute::

    from surpyval.multivariate import Clayton

    truth = Clayton.from_params(2.0, margins=[surv.Weibull.from_params([10.0, 2.0]),
                                              surv.Weibull.from_params([20.0, 3.0])])
    data = truth.random(500, random_state=1)

    cop = Clayton.fit([data[:, 0], data[:, 1]],
                      margins=[surv.Weibull, surv.Weibull])
    print("theta        :", cop.params)
    print("Kendall's tau:", cop.kendall_tau())

Theory: :doc:`Multivariate Analysis`. How-to:
:doc:`Multivariate Modelling with SurPyval`.

Saving a model
--------------

Every fitted model can be written to a plain dictionary or a JSON file and
restored later, without having to remember which class wrote it:

.. jupyter-execute::

    restored = surv.from_dict(model.to_dict())
    print(restored.params)

See the "Saving and Loading Models" section of :doc:`Conventions` for the
details, including what a restored model can and cannot do.

Where to go next
----------------

- :doc:`Types of Data` explains observed, censored and truncated data, and why
  each needs different treatment.
- :doc:`Conventions` is the reference for the input arrays, the censoring flags
  and the functions every model has.
- :doc:`Handy References - Aide-mémoire` collects the identities between
  :math:`f`, :math:`F`, :math:`R`, :math:`h` and :math:`H`.
- :doc:`Data Wrangler Examples` shows how to get data from the format you have
  into the format SurPyval takes.
- :doc:`applications` works through complete examples from several fields.
- :doc:`comparison_and_validation` covers comparing groups and validating
  models.
- :doc:`surpyval` is the full API reference.
