Choosing a Model
================

This page takes you from your data to a model by asking one yes-or-no
question at a time. There are three stages:

1. **The shape of the problem** (questions 1 to 5): can a unit have the
   event more than once, can the event be of several kinds, and do you have
   readings along the way? Most data ends this stage at "one event per
   unit".
2. **Writing the data** (questions 6 to 11): what you know about each
   unit's time, written as the ``x``, ``c``, ``n``, ``tl`` and ``tr`` every
   fitter takes.
3. **The question** (questions 12 to 24): what you want to know, which
   decides between test planning, regression, a non-parametric estimate and
   a parametric distribution.

Then check the answer, and use the lookup table at the end to find the call
for a situation you recognise. Censoring, counts and truncation change how
you *write* the data (stage 2), not which model family you are in (stage 1)
or which question you are asking (stage 3).

Each question has exactly two answers. Follow "Yes" or "No" to the next
question or to a model. If the words are new, :doc:`Survival Analysis from
Scratch` explains them from the beginning, and :doc:`Glossary` maps the
engineering, medical and statistical names for the same ideas.

.. contents:: On this page
   :local:
   :depth: 1

Stage 1: the shape of the problem
---------------------------------

Most analyses have one event per unit: a failure, a death, a default. If a
unit can fail and be repaired, can fail in different ways, or has its
condition measured along the way, you are in a different family before
anything else, and a single-event model fitted to that data gives wrong
numbers without an error.

1. **Can one unit have the event more than once?** (it is repaired and goes
   back into service; a patient is admitted again)

   - Yes: **recurrent events**. Add the unit's id ``i`` to each row (``x``
     is the unit's age at each event, and its last row, flagged ``c = 1``,
     is the end of watching). The mean cumulative function is
     ``NonParametricCounting``, a non-homogeneous Poisson process such as
     ``CrowAMSAA``, and repair quality the renewal models
     (``GeneralizedRenewal``, ``GeneralizedOneRenewal``), all in
     ``surpyval.recurrent``. See :doc:`Recurrent Event Modelling with
     SurPyval`. Stop here.
   - No: go to question 2.

2. **Can the event be one of several different kinds?** (wear or shock;
   death from one of several causes)

   - Yes: go to question 3.
   - No: go to question 4.

3. **Does the first event end the unit's record?**

   - Yes: **competing risks**. Pass the kind of each event as ``e``.
     ``CompetingRisks.fit(x, e=e, c=c)`` gives the cumulative incidence of
     each kind, and ``FineGray.fit(x, Z, e, c=c)`` regression on it. A
     Kaplan-Meier fit to one kind, with the others censored, overstates
     that kind's incidence. See :doc:`Competing Risks SurPyval Modelling`.
     Stop here.
   - No (units move on between states: healthy, ill, dead): **multi-state
     models**, ``MultiState.fit``. See :doc:`Multi-State Models`. Stop
     here.

4. **Do you have condition readings over time?** (wear, crack length,
   vibration, a biomarker)

   - Yes: go to question 5.
   - No: **one event per unit**. Go to stage 2.

5. **Do you also have failure times, which the readings may drive?**

   - Yes: a **joint model** of the readings and the failures together,
     ``WeibullJointPH.fit``. See :doc:`Joint Models`. Stop here.
   - No: **degradation analysis**, the readings to a failure threshold,
     ``DegradationAnalysis.fit(x, y, i, threshold=...)``, which also
     predicts a monitored unit's remaining useful life. See
     :doc:`Degradation Modelling with SurPyval`. Stop here.

Stage 2: writing the data
-------------------------

Ask these questions of every unit. Each unit is one row: a time ``x``, a
censoring flag ``c``, and where needed a count ``n`` and a truncation
window ``tl``, ``tr``. Rows of different kinds mix freely in one data set.
:doc:`Types of Data` explains each kind with examples, and :doc:`Data
Model` lists every letter a row can carry.

6. **Did the unit have the event while you were watching?**

   - Yes: go to question 7.
   - No (it was still running when you stopped, or was removed): it is
     **right censored**, ``c = 1``, with ``x`` the last time you saw it
     working. Go to question 9.

7. **Do you know the exact time of the event?**

   - Yes: it is **observed**, ``c = 0``, with ``x`` that time. Go to
     question 9.
   - No: go to question 8.

8. **Did the event happen before you first looked?**

   - Yes: it is **left censored**, ``c = -1``, with ``x`` the time of the
     first look. Go to question 9.
   - No (it happened between two looks): it is **interval censored**,
     ``c = 2``, with ``x = [last seen working, found failed]``. Go to
     question 9.

9. **Do several units have exactly the same record?**

   - Yes: write the record once with ``n``, how many units share it. Go to
     question 10.
   - No: go to question 10 (``n`` is 1 for every row by default).

10. **Is the unit in your data only because it had already lasted to some
    age?** (installed before monitoring began; a patient enrolled some time
    after diagnosis)

    - Yes: it is **left truncated**, with ``tl`` that age. Go to
      question 11.
    - No: go to question 11.

11. **Is the unit in your data only because it failed by some time?**
    (warranty returns, where the units that did not fail are never seen)

    - Yes: it is **right truncated**, with ``tr`` that time. The record is
      done.
    - No: the record is done.

When every unit has its record, go to stage 3.

.. note::

   In SurPyval ``c = 1`` means **censored** and ``c = 0`` an observed
   event. That is the opposite of R's ``Surv(time, status)``, where
   ``status = 1`` is the event. Data brought from R needs ``c = 1 -
   status``.

Ten units, written with these questions: two observed failures, at 310
and 450; two found failed between the inspections at 400 and 520 (one row
with ``n = 2``); one already failed at the first look, at 150; and five
still running when the test stopped at 600:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    x = [310, 450, [400, 520], 150, 600]
    c = [0, 0, 2, -1, 1]
    n = [1, 1, 2, 1, 5]

    model = surv.Weibull.fit(x=x, c=c, n=n)
    print(model)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert int(np.sum(model.data["n"])) == 10, model.data

Stage 3: the question
---------------------

Every unit now has a record. What do you want to know?

12. **Are you planning a test before you have the data?** (how many units,
    for how long, to show a reliability or estimate a life)

    - Yes: **test planning**, ``surpyval.test_planning``. See :doc:`Test
      Planning with SurPyval`. Stop here.
    - No: go to question 13.

13. **Do the units differ in something that may change their life?** (a
    treatment, a stress, a design, an age at entry)

    - Yes: **regression**, with the covariates as ``Z``, one row per row of
      ``x``. Go to question 14.
    - No: go to question 17.

14. **Is the covariate a stress you set, such as temperature or voltage,
    and do you want to extrapolate to the stress in use?**

    - Yes: an **accelerated life** model, a life-stress relationship such
      as Arrhenius, Eyring or the inverse power law:
      ``AcceleratedLife(Weibull, life_models.Arrhenius).fit(x, Z, c)``.
      Stop here.
    - No: go to question 15.

15. **Do you want to assume a life distribution?**

    - Yes: go to question 16.
    - No: **Cox proportional hazards**, ``CoxPH.fit(x, Z, c=c)``: hazard
      ratios with no assumed baseline shape. Stop here.

16. **Does the covariate stretch or shrink time, rather than scale the
    risk?**

    - Yes: **accelerated failure time**, such as ``WeibullAFT.fit(x, Z,
      c)``. Stop here.
    - No: **proportional hazards** with a distribution, such as
      ``WeibullPH.fit(x, Z, c)``. Stop here.

    :doc:`Regression Modelling with SurPyval` covers these and the other
    forms (proportional odds, additive hazards, Buckley-James).

17. **Do you need answers beyond your longest observed time?**
    (extrapolation: a warranty period longer than the test, a life past
    the end of the study)

    - Yes: you need a **parametric** model. Go to question 20.
    - No: go to question 18.

18. **Do you need a mean life, a B-life below your first failure, or a
    smooth curve to predict from?**

    - Yes: you need a **parametric** model. Go to question 20.
    - No: a **non-parametric** estimate describes your data with no
      assumed shape. Go to question 19.

19. **Is any record interval or left censored, or any unit right
    truncated?**

    - Yes: **Turnbull**, ``Turnbull.fit(x, c=c, ...)``, which takes every
      kind of censoring and truncation. Stop here.
    - No: **Kaplan-Meier**, ``KaplanMeier.fit(x, c=c, ...)`` (it takes
      left truncation, ``tl``); ``NelsonAalen`` is its cumulative-hazard
      counterpart. Stop here.

20. **Has at least one unit failed?**

    - Yes: fit a **Weibull** first, ``Weibull.fit(x, c, n, ...)``, and go
      to question 21.
    - No: there is no fit to make, but there is a bound. With a Weibull
      shape you are willing to assume, ``weibayes(x, c, beta=...)`` gives
      the lower bound on the life (**Weibayes**); with none,
      ``success_run(n)`` gives the reliability a run of ``n`` successes
      demonstrates. Planning such a test is question 12. Stop here.

21. **Will some units never fail?** (a defect only some carry; a cured
    fraction)

    - Yes: add ``lfp=True`` (a limited failure population). Go to
      question 22.
    - No: go to question 22.

22. **Do some units fail at time zero?** (dead on arrival)

    - Yes: add ``zi=True`` (zero inflation). Go to question 23.
    - No: go to question 23.

23. **Is there an age before which nothing fails?**

    - Yes: add ``offset=True`` (the three-parameter Weibull). Go to
      question 24.
    - No: go to question 24.

24. **Do the failures look like two different modes mixed together?**
    (a probability plot with a kink)

    - Yes: a **mixture**, ``MixtureModel.fit(x, c=c, dist=Weibull,
      m=2)``. Then check the answer.
    - No: compare distributions. ``fit_best(x, c, ...)`` fits the plain
      candidates and returns the one with the lowest AIC; with ``lfp``,
      ``zi`` or ``offset``, fit each candidate with the same option and
      compare their ``aic()``. Then check the answer.

:doc:`Parametric SurPyval Modelling` explains each option, and
:doc:`Non-Parametric SurPyval Modelling` each estimator.

Parametric or non-parametric?
-----------------------------

Questions 17 and 18 decide this, and the reason is extrapolation.

- A **non-parametric** estimate (Kaplan-Meier, Nelson-Aalen, Turnbull)
  describes only what was seen. It stops at the longest observation: past
  it there is no estimate, and its mean is only the area up to that point
  (the restricted mean), not the mean life. ``mean()`` gives that area,
  and warns with the survival left at the longest observation when units
  are still at risk there; ask for ``rmst(tau)`` to state the restricted
  mean by name.
- A **parametric** distribution assumes a shape, and the shape is what
  lets it answer beyond the data: a mean life, a B1 life below the first
  failure, the survival at a time no unit reached.
- Check the shape against the non-parametric estimate: ``diagnose()``
  compares them, and ``plot()`` draws both.
- The further you extrapolate, the more the answer rests on the assumed
  shape rather than on the data. That judgement is the analyst's:
  SurPyval does not warn when a parametric model is asked about a time
  past the data.

The ball-bearing data of the :doc:`Quickstart`, with the test stopped at
100 million revolutions, shows the difference. Up to 100 the two agree; the
Weibull also answers at 150, which no bearing reached in the test:

.. jupyter-execute::

    x = np.array([17.88, 28.92, 33, 41.52, 42.12, 45.6, 48.4, 51.84,
                  51.96, 54.12, 55.56, 67.8, 68.64, 68.64, 68.88, 84.12,
                  93.12, 98.64, 105.12, 105.84, 127.92, 128.04, 173.4])
    c = (x > 100).astype(int)       # 1 = still running when the test stopped
    x = np.minimum(x, 100)

    km = surv.KaplanMeier.fit(x, c)
    weibull = surv.Weibull.fit(x, c)
    print("R(90), Kaplan-Meier :", km.sf(90))
    print("R(90), Weibull      :", weibull.sf(90))
    print("R(150), Weibull     :", weibull.sf(150))
    print("Mean life to 100, Kaplan-Meier:", km.rmst(100)["rmst"])
    print("Mean life to 100, Weibull     :", weibull.rmst(100)["rmst"])
    print("Mean life, Weibull            :", weibull.mean())

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert abs(km.sf(90) - weibull.sf(90)) < 0.05, (km.sf(90), weibull.sf(90))
    assert abs(km.rmst(100)["rmst"] - weibull.rmst(100)["rmst"]) < 1
    assert weibull.mean() > weibull.rmst(100)["rmst"]
    assert round(km.rmst(100)["rmst"]) == 66, km.rmst(100)
    assert round(weibull.rmst(100)["rmst"]) == 67, weibull.rmst(100)
    assert round(weibull.mean()) == 71, weibull.mean()

Both curves give a mean life to 100 of about 66 to 67; only the Weibull
gives a mean life (about 71), and it does so by assuming the Weibull's
shape past 100.

Check the answer
----------------

Every parametric and regression model has ``diagnose()``: the goodness of
fit against the non-parametric estimate, the alternatives it would suggest
(another distribution, an offset, a limited failure population) and why.
:doc:`Model Diagnostics` explains every check. Seed it, so its bootstrap
p-values are the same each time:

.. jupyter-execute::

    report = weibull.diagnose(random_state=1)
    print(report.ok)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert report.ok, report

Ask for bounds on whatever you report: ``cb`` on a function of time
(``on="sf"`` by default, or ``"ff"``, ``"Hf"``, ...), ``quantile_cb`` on a
B-life and ``param_cb`` on a parameter. SurPyval's default bounds on a
maximum-likelihood fit are two-sided 95% Wald bounds (``alpha_ci=0.05``,
``bound="two-sided"``, ``method="wald"``). **With few failures (under about
20), prefer likelihood-ratio bounds, ``method="lr"``**, which do not assume
the estimate is normally distributed, and treat a choice between
distributions as weak:

.. jupyter-execute::

    print("R(50), Wald           :", weibull.cb(50))
    print("R(50), likelihood ratio:", weibull.cb(50, method="lr"))
    print("B10, likelihood ratio  :", weibull.quantile_cb(0.1, method="lr"))

Compare models only on the same data: an AIC from one data set says nothing
about a fit to another.

A few more of the answers above, run as written. A zero-failure test of
ten units for 500 hours, assuming a Weibull shape of 2 (question 20), and
the units a demonstration with no failures needs (question 12):

.. jupyter-execute::

    from surpyval import test_planning

    bound = surv.weibayes([500] * 10, c=[1] * 10, beta=2)
    print("R(500) is at least        :", bound.sf(500))
    print("same, from the success run:", surv.success_run(10))
    print("units to show R = 0.9     :",
          test_planning.demonstration_sample_size(0.9))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(bound.sf(500), 3) == round(surv.success_run(10), 3) == 0.741
    assert test_planning.demonstration_sample_size(0.9) == 29

Covariates (questions 13 to 16): a life that doubles in one condition,
fitted with no assumed shape and with a Weibull that stretches time:

.. jupyter-execute::

    rng = np.random.default_rng(1)
    Z = rng.integers(0, 2, (100, 1)).astype(float)
    t = 50 * rng.weibull(1.5, 100) * np.exp(0.7 * Z[:, 0])
    c_z = (t > 80).astype(int)        # the study stopped at 80
    t = np.minimum(t, 80)

    cox = surv.CoxPH.fit(t, Z, c=c_z)
    aft = surv.WeibullAFT.fit(t, Z, c_z)
    print("Cox hazard ratio of condition 1:", np.exp(cox.beta[0]))
    print("Weibull AFT coefficient        :", aft.params[-1])

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.exp(cox.beta[0]) < 1, cox.beta    # condition 1 lives longer
    assert aft.params[-1] < 0, aft.params

Lookup: situations and the calls that handle them
-------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 34 18 48

   * - Situation
     - Data
     - Call
   * - Failure times, some units still running
     - ``x``, ``c``
     - ``Weibull.fit(x, c)``; ``KaplanMeier.fit(x, c)``
   * - Failures found between inspections
     - ``x = [[l, r], ...]``, ``c = 2``
     - ``Weibull.fit(x, c)``; ``Turnbull.fit(x, c)``
   * - Units installed before monitoring began
     - ``x``, ``c``, ``tl``
     - ``Weibull.fit(x, c, tl=tl)``; ``KaplanMeier.fit(x, c, tl=tl)``
   * - Grouped counts (a life table)
     - ``x``, ``c``, ``n``
     - ``Weibull.fit(x, c, n)``; ``life_table(x, c, n)``
   * - Warranty returns only
     - ``x``, ``tr``
     - ``Weibull.fit(x, tr=tr)``; ``Turnbull.fit(x, tr=tr)``
   * - Which distribution fits best
     - as above
     - ``fit_best(x, c)``
   * - Few failures, bounds on a B-life
     - as above
     - ``model.quantile_cb(0.1, method="lr")``
   * - Zero failures, a known shape
     - ``x``, ``c = 1``
     - ``weibayes(x, c, beta=...)``; ``success_run(n)``
   * - Two groups compared
     - ``x``, ``c``, a group
     - ``KaplanMeier.fit`` for each; ``logrank``
   * - Patients with covariates
     - ``x``, ``c``, ``Z``
     - ``CoxPH.fit(x, Z, c=c)``; ``WeibullPH.fit(x, Z, c)``
   * - A test at high temperature, predicting at use
     - ``x``, ``c``, ``Z`` (kelvin)
     - ``AcceleratedLife(Weibull, life_models.Arrhenius).fit(x, Z, c)``
   * - Repairs on a fleet
     - ``x``, ``i``, ``c``
     - ``surpyval.recurrent``: ``NonParametricCounting.fit(x=x, i=i,
       c=c)``; ``CrowAMSAA.fit(x=x, i=i, c=c)``
   * - Several causes, the first one ends it
     - ``x``, ``c``, ``e``
     - ``CompetingRisks.fit(x, e=e, c=c)``; ``FineGray.fit(x, Z, e, c=c)``
   * - Readings until a threshold
     - ``x``, ``y``, ``i``
     - ``surpyval.degradation``: ``DegradationAnalysis.fit(x, y, i,
       threshold=...)``
   * - Readings and failures together
     - items with ``markers``
     - ``WeibullJointPH.fit``
   * - Demonstrating a reliability before testing
     - none yet
     - ``test_planning.demonstration_sample_size(R)``;
       ``test_planning.demonstration_plan``

.. jupyter-execute::
    :hide-code:
    :hide-output:

    import inspect
    import surpyval.degradation
    import surpyval.recurrent

    # Every name this page gives is real, with the arguments it shows.
    _names = [
        "Weibull", "KaplanMeier", "NelsonAalen", "Turnbull", "life_table",
        "fit_best", "weibayes", "success_run", "logrank", "CoxPH",
        "WeibullPH", "WeibullAFT", "AcceleratedLife", "CompetingRisks",
        "FineGray", "MultiState", "MixtureModel", "WeibullJointPH",
    ]
    _missing = [_n for _n in _names if not hasattr(surv, _n)]
    assert not _missing, _missing
    assert hasattr(surv.life_models, "Arrhenius")
    for _sub, _n in [(surpyval.recurrent, "NonParametricCounting"),
                     (surpyval.recurrent, "CrowAMSAA"),
                     (surpyval.recurrent, "GeneralizedRenewal"),
                     (surpyval.recurrent, "GeneralizedOneRenewal"),
                     (surpyval.degradation, "DegradationAnalysis"),
                     (test_planning, "demonstration_plan")]:
        assert hasattr(_sub, _n), _n
    _fit = inspect.signature(surv.Weibull.fit).parameters
    assert {"lfp", "zi", "offset", "tl", "tr"} <= set(_fit), _fit
    assert {"dist", "m"} <= set(inspect.signature(surv.MixtureModel.fit).parameters)
    assert "threshold" in inspect.signature(
        surpyval.degradation.DegradationAnalysis.fit).parameters
    assert "beta" in inspect.signature(surv.weibayes).parameters
