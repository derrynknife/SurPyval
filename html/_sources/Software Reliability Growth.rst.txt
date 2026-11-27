Software Reliability Growth
===========================

While software is tested, it fails; each failure is traced to a fault, and
the fault is fixed. If the fixes work, failures come less and less often.
A *software reliability growth model* is a curve fitted to those failures,
which answers the questions a test or release manager asks:

- How many faults are left?
- How reliable is the software now?
- How much more testing would it take to reach a target?
- When is it cheapest to release?

Nothing here is special to software: the same models fit any process in
which problems are found and removed (the commissioning of a plant, the
proving of a new design). This page shows the models SurPyval has, how to
fit them to failure times or to weekly counts, how to choose between them,
the three answers with their confidence bounds, and when not to believe
them. Every class used here is documented under :doc:`surpyval.counting`.

What the models assume
----------------------

All four are Poisson processes whose failure rate (the *intensity*) changes
with the amount of testing done, ``t``. Each has a curve for the expected
number of failures by ``t``, written :math:`\Lambda(t)`:

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - Model
     - Expected failures by ``t``
     - In words
   * - ``GoelOkumoto``
     - :math:`a\,(1 - e^{-bt})`
     - There are ``a`` faults in all, and each is found at the same rate
       ``b``. The failure rate is highest at the start and falls by the
       same proportion every week.
   * - ``YamadaDelayedS``
     - :math:`a\,(1 - (1 + bt)\,e^{-bt})`
     - ``a`` faults in all, but finding one takes two steps (a failure,
       then its cause), so the count is slow to start: the failure rate
       rises to a peak and then falls. An S-shaped curve.
   * - ``InflectionS``
     - :math:`a\,\dfrac{1 - e^{-bt}}{1 + \psi e^{-bt}}`
     - ``a`` faults in all, some of which can only be found once others
       are out of the way. ``psi`` says how S-shaped the curve is;
       ``psi = 0`` is Goel-Okumoto.
   * - ``MusaOkumoto``
     - :math:`\log(1 + \lambda_0 \theta t)\,/\,\theta`
     - No fixed number of faults. Each fix lowers the failure rate by the
       same proportion ``theta``, so early fixes help most, and failures
       never quite stop.

They share three assumptions, and the answers are only as good as these:

- The testing is steady: a week of testing late on is like a week early on
  (the same people, the same kind of tests, the same software apart from
  the fixes).
- A fix removes its fault and adds no new one.
- Time is measured in whatever drives failures: running time or test
  cases executed are better clocks than the calendar, if you have them.

The first three models have a *finite* number of faults, ``a``; the fourth
does not, and neither do the general-purpose ``CrowAMSAA`` and ``Duane``
power laws (see :doc:`Recurrent Event Modelling with SurPyval`), which
can be fitted to the same data and compared.

Fitting failure times
---------------------

The data are the failures of the Naval Tactical Data System (NTDS), a
classic set in this field: the days between the 26 failures found in the
production phase of one module's development [JM1972]_ (as tabulated in
[LXZ2015]_). The models take the failure *times*, the running total:

.. jupyter-execute::

    import numpy as np
    from surpyval.recurrent import GoelOkumoto

    days_between = [9, 12, 11, 4, 7, 2, 5, 8, 5, 7, 1, 6, 1,
                    9, 4, 1, 3, 3, 6, 1, 11, 33, 7, 91, 2, 1]
    days = np.cumsum(days_between)

    model = GoelOkumoto.fit(days, unit="days")
    print(model)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert days[-1] == 250 and len(days) == 26
    assert round(model.params[0], 2) == 33.99, model.params
    assert round(model.params[1], 5) == 0.00579, model.params
    assert model.maximum == "verified"

The fit says that about 34 faults were there to be found, each found at a
rate of 0.0058 a day. The last failure was on day 250, and with no other
information the test is taken to end there. If testing went on after the
last failure with nothing found, say so: add the end of the test as a
last row with ``c=1``.

.. code-block:: python

    # the same failures, with testing to day 280 and nothing more found
    x = np.append(days, 280)
    c = [0] * 26 + [1]
    GoelOkumoto.fit(x, c=c)

Several systems under the same test are told apart with ``i``, as for the
other recurrent models; each is then expected to hold ``a`` faults.

``plot()`` draws the fitted curve over the failures actually counted, with
its confidence band:

.. jupyter-execute::

    model.plot();

Fitting weekly counts
---------------------

Often the times are not recorded, only how many failures each week (or
build, or sprint) had. Give each period as a row ``[start, end]`` with
``c=2`` and its count as ``n``. Here are the same failures counted in
25-day periods:

.. jupyter-execute::

    edges = np.arange(0, 251, 25)
    counts = np.histogram(days, edges)[0]
    periods = np.column_stack([edges[:-1], edges[1:]])
    print("failures in each period:", counts)

    grouped = GoelOkumoto.fit(periods, c=[2] * 10, n=counts)
    print(grouped.params.round(4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert counts.tolist() == [2, 4, 5, 7, 3, 1, 1, 0, 0, 3]
    assert abs(grouped.params[0] - model.params[0]) < 8
    _grouped = grouped.params.copy()

The answer is close to the one from the exact times: counts lose a
little, not much. Periods need not be the same length, and a period with
no failures is a row with ``n = 0`` (leaving it out would say it was not
tested).

Which model?
------------

Fit them all and compare. ``aic()`` is the usual yardstick (lower is
better; a difference under 2 is no real preference):

.. jupyter-execute::

    from surpyval.recurrent import (
        HPP, CrowAMSAA, InflectionS, MusaOkumoto, YamadaDelayedS,
    )

    fits = {
        f.name: f.fit(days)
        for f in (GoelOkumoto, YamadaDelayedS, InflectionS, MusaOkumoto,
                  CrowAMSAA, HPP)
    }
    for name, fit in fits.items():
        print(f"{name:28s} AIC {fit.aic():7.2f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _aic = {k: round(f.aic(), 2) for k, f in fits.items()}
    assert _aic["Goel-Okumoto"] == 169.38, _aic
    assert _aic["Yamada delayed S-shaped"] == 165.84, _aic
    assert min(_aic, key=_aic.get) == "Yamada delayed S-shaped"
    assert max(_aic, key=_aic.get) == "Crow-AMSAA"
    assert all(f.maximum == "verified" for f in fits.values())

The delayed S-shaped curve fits these failures best: they were slow to
start (2 in the first 25 days, 7 in the fourth). The constant rate
(``HPP``) and the power law are worst. Three more checks are worth a
line each:

.. jupyter-execute::

    print(model.trend_test())

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _trend = model.trend_test()
    assert _trend.trend == "decreasing" and _trend.p_value < 0.05

The trend test asks whether there is growth at all: is the failure rate
changing, or would a constant rate do? Here it is falling (p = 0.014). If
it were not, no growth model should be believed, whatever it printed.

``cramer_von_mises`` tests whether the failures follow the fitted curve (a
small p-value says they do not), and a plot of each fitted curve against
the counted failures shows where a model departs:

.. jupyter-execute::

    import matplotlib.pyplot as plt

    yamada = fits["Yamada delayed S-shaped"]
    print(yamada.cramer_von_mises(random_state=1))

    t = np.linspace(0, 400, 200)
    plt.step(days, np.arange(1, 27), where="post", color="k", label="failures")
    for name in ("Goel-Okumoto", "Yamada delayed S-shaped", "Musa-Okumoto"):
        plt.plot(t, fits[name].cif(t), label=name)
    plt.axvline(250, color="grey", linestyle=":")
    plt.xlabel("days of testing")
    plt.ylabel("failures")
    plt.legend();

The curves agree over the 250 days of data and part company after it:
that is the nature of the problem. Choosing a model is choosing how to
read the future, and the data seen so far cannot settle it alone. Give
most weight to the model whose assumptions describe how the testing was
done.

Which model predicts best?
--------------------------

A fit to the failures seen is a weak guide to the failures to come, as
the last plot shows. A stronger one is to ask how each model *would
have* predicted, had it been used all along. ``compare_predictions``
does that: at each failure it fits every model to the failures before
it, takes the model's prediction of the time to the next failure, and
scores the prediction against the time that came [ACL1986]_. Here it is
on all 31 failures of the NTDS production and test phases, starting from
the tenth:

.. jupyter-execute::

    from surpyval.recurrent import compare_predictions

    all_days = np.cumsum(days_between + [87, 47, 12, 9, 135])
    result = compare_predictions(
        all_days, [GoelOkumoto, YamadaDelayedS, CrowAMSAA, HPP], start=10
    )
    print(result)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _table = result.summary()
    assert result.best == "YamadaDelayedS", result.best
    assert _table.loc["GoelOkumoto", "failed"] == 12, _table
    assert _table.loc["YamadaDelayedS", "failed"] == 0, _table

The table has one row per model:

- ``log_likelihood`` is the *prequential likelihood*, in logs: how
  probable the model's predictions made the times that came. Higher is
  better, and a difference of 2 or 3 is a clear preference.
- ``u_distance`` and ``bias`` are the *u-plot*: whether the predictions
  were biased. *Optimistic* means failures came sooner than predicted.
  ``biased`` says whether the bias is more than chance would give.
- ``y_distance`` and ``drifting`` are the *y-plot*: whether the bias
  changed as testing went on.
- ``failed`` counts the steps at which the model could not be fitted at
  all. Goel-Okumoto fails at 12 of the 21 steps here: until the long gap
  before failure 24 the failures were not slowing down, and it refuses
  data with no growth. A model that cannot be fitted for much of the
  test cannot be relied on to predict during it.

The verdict below the table says the same in words.
``result.plot("u")``, ``result.plot("y")`` and ``result.plot("ratio")``
draw the plots:

.. jupyter-execute::

    result.plot("u");

How many faults are left?
-------------------------

``remaining_faults(t)`` is the number the model expects to be left after
testing to ``t``: the total ``a`` less those expected by then.
``remaining_faults_cb`` gives its confidence bounds.

.. jupyter-execute::

    print("Goel-Okumoto:", model.remaining_faults(250).round(1),
          model.remaining_faults_cb(250, alpha_ci=0.2).round(1))
    print("Yamada      :", yamada.remaining_faults(250).round(1),
          yamada.remaining_faults_cb(250, alpha_ci=0.2).round(1))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(float(model.remaining_faults(250)), 1) == 8.0
    assert round(float(yamada.remaining_faults(250)), 1) == 1.5
    _lo, _hi = model.remaining_faults_cb(250, alpha_ci=0.2)
    assert round(_lo, 1) == 2.4 and round(_hi, 1) == 27.2, (_lo, _hi)

Goel-Okumoto expects 8 more faults, with 80% confidence somewhere from 2
to 27; the S-shaped model, which fitted better, expects between 1 and 2.
(In the event, 8 more were found: 5 in the test phase that followed and 3
in use. The better-fitting model was the worse forecaster, which is not
rare.) Read the width of the bounds and the disagreement between
reasonable models as the honest answer: with 26 failures, the number
left is known only roughly.

A model with no fixed number of faults has no answer to this question,
and says so:

.. jupyter-execute::

    try:
        fits["Musa-Okumoto"].remaining_faults(250)
    except ValueError as err:
        print(err)

Ask it instead how many failures to expect in a stretch of further use,
``cif(t2) - cif(t1)``, which every model can answer.

How reliable is it now?
-----------------------

Two numbers describe the software as it stands at the end of the test.
The failure rate now is ``iif(250)`` (and ``mtbf(250)`` its reciprocal,
the mean time between failures if nothing more were fixed), with bounds
from ``iif_cb`` and ``mtbf_cb``:

.. jupyter-execute::

    print("failures a day now:", model.iif(250).round(4),
          model.iif_cb(250, alpha_ci=0.2).round(4))
    print("days between failures:", model.mtbf(250).round(1))

The chance of running a further stretch without any failure is
``next_failure_sf``: the reliability over a mission of that length,
starting from the end of the test (or from any ``age=``).
``next_failure_sf_cb`` gives its bounds. For 30 days of use:

.. jupyter-execute::

    print(model.next_failure_sf(30).round(3))
    print(model.next_failure_sf_cb(30, alpha_ci=0.2).round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(float(model.next_failure_sf(30)[0]), 2) == 0.28
    assert round(float(model.mtbf(250)), 1) == 21.6

A 28% chance of 30 days without a failure, and the bounds are wide. Both
numbers assume the software is used the way it was tested.

How much more testing?
----------------------

``time_to_intensity(target)`` is the testing time at which the failure
rate has fallen to a target, and ``time_to_reliability(target, x)`` the
time from which a mission of length ``x`` succeeds with the target
probability. Both are on the test's clock, so subtract the testing done
so far. For a failure rate of one in 100 days, and for an 80% chance of
30 days without a failure:

.. jupyter-execute::

    t_rate = model.time_to_intensity(0.01)
    print("rate target reached at day", round(t_rate),
          "(another", round(t_rate - 250), "days)")
    print("80% bounds:", model.time_to_intensity_cb(0.01, alpha_ci=0.2).round())

    t_rel = model.time_to_reliability(0.8, 30)
    print("reliability target reached at day", round(t_rel))
    print("80% bounds:",
          model.time_to_reliability_cb(0.8, 30, alpha_ci=0.2).round())

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(t_rate) == 515 and round(t_rel) == 551, (t_rate, t_rel)
    _lo, _hi = model.time_to_intensity_cb(0.01, alpha_ci=0.2)
    assert round(_lo) == 349 and round(_hi) == 1240, (_lo, _hi)

On this model the test is about half done. The bounds are found by
asking when the *bounds* on the failure rate (or on the reliability)
reach the target, so an upper bound of ``inf`` is possible: it says the
data cannot rule out that the target is never reached.

A target the fitted curve never reaches is refused, with the reason. A
constant failure rate, for instance, never falls to a lower target
however long the test:

.. jupyter-execute::

    try:
        fits["Homogeneous Poisson Process"].time_to_intensity(0.01)
    except ValueError as err:
        print(err)

When to release
---------------

Testing costs money, and so do faults that reach the field. If you can
put a figure on a day of testing, on a fault fixed during testing and on
a fault that reaches use, ``release_time`` finds the day on which to stop
testing for the least expected total cost [OG1980]_. Testing pays for
itself while the failure rate is above ``test_cost / (field_cost -
fix_cost)``; the cheapest release is the day the rate falls to that
level. Say a day of testing costs 1 (thousand), a fault 0.5 to fix in
testing and 40 in the field:

.. jupyter-execute::

    plan = model.release_time(test_cost=1, fix_cost=0.5, field_cost=40)
    print("release on day", round(plan["time"]),
          "at an expected cost of", round(plan["cost"]))
    print("within 1% of that cost:", [round(t) for t in plan["band"]])
    print("80% bounds on the day:",
          model.release_time_cb(1, 0.5, 40, alpha_ci=0.2).round())

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(plan["time"]) == 354 and plan["binding"] == "cost"
    assert [round(t) for t in plan["band"]] == [313, 399], plan["band"]
    assert round(plan["cost"]) == 544, plan["cost"]

The ``band`` is the stretch of days whose cost is within 1% of the best,
here about twelve weeks: the exact day matters little, and the bounds on the day (which come from
the uncertainty in the fitted model) are wider still.
``method="bootstrap"`` gives bounds by simulation instead.

Costs are in any currency and time is in the model's unit. A model with
no fixed number of faults needs ``life=``, the time at which the software
goes out of use, to add up the field's failures. A release may also have
to meet a reliability target, whatever the costs say; give it and the
release is the later of the two days:

.. jupyter-execute::

    plan = model.release_time(1, 0.5, 40, reliability=0.8, mission=30)
    print("release on day", round(plan["time"]), "set by", plan["binding"])

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(plan["time"]) == 551 and plan["binding"] == "reliability"

A whole number of faults: Jelinski-Moranda
------------------------------------------

The oldest model of all [JM1972]_ counts the faults in whole numbers:
the software starts with ``N`` faults, each causes failures at the same
rate ``phi`` until it is fixed, so the failure rate steps down by
``phi`` at every failure. ``JelinskiMoranda`` fits it to one test's
failure times (or to the times between failures, with ``gaps=True``):

.. jupyter-execute::

    from surpyval.recurrent import JelinskiMoranda

    jm = JelinskiMoranda.fit(days, unit="days")
    print(jm)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert jm.N == 31 and round(jm.phi, 6) == 0.006948, (jm.N, jm.phi)
    assert jm.remaining_faults() == 5

It estimates 31 faults, so 5 left after the 26 found. (With ``N``
allowed to be any real number, as most texts fit it, the estimate is
31.2: the published figure for these data [LXZ2015]_.) Its answers are
methods of their own, with bounds by the likelihood ratio:

.. jupyter-execute::

    print("faults left:", jm.remaining_faults(),
          jm.remaining_faults_cb(alpha_ci=0.2))
    print("failures a day now:", round(jm.current_rate(), 4),
          jm.current_rate_cb(alpha_ci=0.2).round(4))
    print("95% bounds on the number of faults:", jm.param_cb("N"))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert jm.param_cb("N").tolist() == [26.0, 94.0], jm.param_cb("N")
    _left = jm.remaining_faults_cb(alpha_ci=0.2)

The bounds are whole numbers, and they are wide and lopsided: 26 faults
found, and anything from 26 to 94 in all at 95%. That is the model
being honest. An upper bound of ``inf`` is possible too, and means the
data do not limit the number of faults at that level. As with
Goel-Okumoto, data that show no growth have no estimate at all, and the
fit is refused, saying why. Its ``aic()`` compares with that of the
other models fitted to the same failure times, and it can be one of the
models given to ``compare_predictions``.

Before testing starts: defects by phase
---------------------------------------

The models above need failures from testing. Earlier in a project there
is a different record: the defects found in each phase of development
(requirements review, design review, code inspection, unit test,
integration, system test). Across the phases these tend to rise to a
peak and fall away, and the *Rayleigh defect model* [Kan2002]_ fits that
shape: :math:`\Lambda(t) = K (1 - e^{-t^2 / (2c^2)})`, with ``K`` the
defects that will be found in all and ``c`` where the peak is. Fitted to
the phases done so far, it predicts what the later phases, and the
field, will find.

``phases_to_data`` turns a table of counts into the rows the fit takes
(phase ``j`` is the interval from ``j - 1`` to ``j``; give ``lengths=``
if the phases' durations are known and unequal). The counts here are
made up for illustration, for a project with five of its six phases
done:

.. jupyter-execute::

    from surpyval.recurrent import RayleighDefects, phases_to_data

    found = [18, 44, 52, 41, 24]
    phases = RayleighDefects.fit(**phases_to_data(found))
    print("defects in all:", phases.params[0].round(), " peak at phase",
          phases.params[1].round(2))
    print("sixth phase:", phases.expected_events(5, 6).round(1),
          phases.expected_events_cb(5, 6, alpha_ci=0.2).round(1))
    print("left for the field:", phases.remaining_faults(6).round(1),
          phases.remaining_faults_cb(6, alpha_ci=0.2).round(1))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(phases.params[0]) == 197, phases.params
    assert round(float(phases.expected_events(5, 6)), 1) == 11.6
    assert round(float(phases.remaining_faults(6)), 1) == 6.2

``expected_events(start, end)`` is the number expected in any stretch
(here the sixth phase), on every model on this page, and
``remaining_faults`` what is left after it. The bounds are on the
expected numbers. The model cannot say anything until the counts have
turned down: before the peak, any total fits, and the fit is refused.

When a fix may not help: Littlewood-Verrall
-------------------------------------------

Every model so far takes each fix to work. The Littlewood-Verrall model
[LV1973]_ does not: after each failure the failure rate is drawn afresh,
lower *on average* than the one before, but possibly higher. The rates
follow a gamma distribution with shape ``alpha`` whose scale shrinks as
failures accumulate (through ``psi(i) = beta0 + beta1 i``). There is no
number of faults in this model. What it gives is a prediction of the
next failure that allows for not knowing the rate:

.. jupyter-execute::

    from surpyval.recurrent import LittlewoodVerrall

    lv = LittlewoodVerrall.fit(all_days)
    print(lv)
    print("chance of 30 more days without a failure:",
          lv.next_failure_sf(30.0).round(3))
    print("90% interval for the failure rate now:",
          lv.current_rate_qf([0.05, 0.95]).round(4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert lv.maximum == "verified"
    _lv_sf = float(lv.next_failure_sf(30.0))
    assert 0 < _lv_sf < 1

``next_failure_qf(0.5)`` is the median time to the next failure, which
always exists; the *mean* is infinite when ``alpha`` is 1 or less, and
``next_failure_mean`` then refuses and points to the median. ``alpha``
is hard to pin down: ``param_cb("alpha")`` gives its likelihood-ratio
bounds, whose upper end is often ``inf``. If the times between failures
are no more scattered than a model with exact rates would make them,
``alpha`` has no estimate at all and the fit is refused, naming the
simpler models to use. ``LittlewoodVerrall`` can be given to
``compare_predictions`` with the others, which is the fair way to judge
whether its extra caution earns its keep on your data.

When the testing was not steady: effort and change points
---------------------------------------------------------

Every curve above supposes steady testing. A week with half the testers
then looks like a week with half the faults left, and a model fitted to
the calendar reads a holiday as growth. Two tools put the same models on
a better clock.

**Testing effort.** If you know how much testing was done in each period
(tester hours, test cases run, hours of execution), ``EffortClock``
runs a model on the effort spent instead of the time passed:
:math:`\Lambda(t) = \Lambda_0(W(t))`, with :math:`W(t)` the effort
spent by ``t``. With ``GoelOkumoto`` this is the testing-effort model of
Yamada, Ohtera and Narihisa [YON1986]_, :math:`a (1 - e^{-b W(t)})`:
``b`` is then the detection rate per unit of effort. The counts and
hours here are made up for illustration: eight weeks of testing by a
team that grew and then moved on.

.. jupyter-execute::

    from surpyval.recurrent import EffortClock

    weeks = np.arange(8.0)
    periods = np.column_stack([weeks, weeks + 1])
    hours = [40, 80, 160, 200, 200, 120, 60, 40]
    found = [3, 9, 17, 16, 11, 5, 2, 1]
    data = dict(x=periods, c=[2] * 8, n=found)

    on_hours = EffortClock(GoelOkumoto, (weeks + 1, hours)).fit(**data)
    by_week = GoelOkumoto.fit(**data)
    s_shaped = YamadaDelayedS.fit(**data)
    for name, m in (("Goel-Okumoto on hours", on_hours),
                    ("Goel-Okumoto by week", by_week),
                    ("delayed S-shaped by week", s_shaped)):
        print(f"{name:25s} aic {m.aic():5.1f}   faults left "
              f"{m.remaining_faults(8.0):4.1f}")
    print("next week, at 40 hours:", on_hours.expected_events(8, 9).round(1))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(on_hours.aic(), 1) == 34.1, on_hours.aic()
    assert round(by_week.aic(), 1) == 63.3, by_week.aic()
    assert round(s_shaped.aic(), 1) == 44.2, s_shaped.aic()
    assert round(float(on_hours.remaining_faults(8.0)), 1) == 23.6
    assert round(float(s_shaped.remaining_faults(8.0)), 1) == 5.7

The effort is a table of amounts per period, ``(ends, amounts)``, spent
evenly within each; the fit takes the same data as before, in time, and
every answer (``remaining_faults``, ``time_to_intensity``,
``release_time``, the bounds) is in time. Here the model on hours fits
far better than either calendar model, and tells a different story: the
failures fell away in the last weeks because the testing did, not
because the faults ran out. The S-shaped curve, which fits the rise and
fall of the counts well enough by week, puts a quarter as many faults
left. After the table's last period the effort is taken to carry on at
the last period's rate; a table that runs on into the plan (the hours
you intend to spend) answers "how many will next month's testing find".
Where the effort follows a smooth rise and fall, ``WeibullEffort.fit``
fits the curve :math:`W(t) = N (1 - e^{-\beta t^m})` to the table, and
can be given in its place. Effort cannot fall, and a failure recorded in
a period with no effort is refused: check the table against the
failures.

**A change point.** Where the testing changed at a known time (a new
team, a new test strategy, a new release), ``ChangePoint(model,
tau=...)`` lets the model's clock run at a different speed from ``tau``
on: every rate of the model is multiplied by ``ratio`` after it, an
extra parameter. For ``GoelOkumoto`` the detection rate ``b`` becomes
``ratio * b``, the change-point model of Zhao [Zhao1993]_; the curve is
continuous at ``tau`` and the number of faults is unchanged. Where the
time is not known, leave ``tau`` out and it is estimated, from failure
*times* of one system.

The NTDS failures above are a case. ``GoelOkumoto`` fitted them with
growth all the way; but look for a change, and the growth model with a
change point is refused:

.. jupyter-execute::

    from surpyval.recurrent import HPP, ChangePoint

    try:
        ChangePoint(GoelOkumoto).fit(days)
    except ValueError as err:
        print(err)

Either side of the best change point the failures are steady: there is
no growth left for the curve to measure, and the fit says which model to
use instead, a constant rate that changes.

.. jupyter-execute::

    from surpyval.recurrent import change_point_cb, change_point_test

    steps = ChangePoint(HPP).fit(days)
    growth = GoelOkumoto.fit(days)
    print("change at day", steps.change_point["tau"],
          " 90% bounds", change_point_cb(steps, alpha_ci=0.1))
    print("rate before:", steps.params[0].round(3), "a day;  after:",
          (steps.params[0] * steps.params[1]).round(3))
    print("aic", round(steps.aic() + 2, 1), "against", round(growth.aic(), 1),
          "for Goel-Okumoto")
    test = change_point_test(steps, n_boot=99, random_state=1)
    print("no change: p =", test["p_value"])

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert steps.change_point["tau"] == 105.0
    assert change_point_cb(steps, alpha_ci=0.1).tolist() == [104.0, 116.0]
    assert round(steps.params[0], 3) == 0.190, steps.params
    assert round(steps.params[0] * steps.params[1], 3) == 0.041
    assert round(steps.aic() + 2, 1) == 162.5, steps.aic()
    assert test["p_value"] == 0.01, test

A failure every five days until day 105, and one every twenty-four days
after: these data read as well as one step as they do as steady growth
(better, by ``aic``), and the two readings predict differently: the
growth curve expects the rate to keep falling, the step expects it to
stay where it is. Nothing in the failure times says which; the test
log might.

Three cautions. An estimated ``tau`` is searched for, so its fit looks
better than a change at a time chosen in advance would: the fitted
model's ``aic()`` and bounds take ``tau`` as given (2 is added above for
it), and ``change_point_cb`` gives likelihood-ratio bounds for ``tau``
itself, whose level is approximate. For the same reason the usual
chi-squared test of "no change" is wrong, and ``change_point_test`` uses
a parametric bootstrap instead: it simulates data with no change from
the fitted model, searches each for its best change point, and reports
how often the improvement is as large as the one seen (so ``n_boot=99``
gives a p-value in steps of 0.01, and takes about 99 times as long as
the fit). And a change point will always be found if you look: ask for
one where something did change, and compare its predictions with the
plain model's by ``compare_predictions``, which takes both.

When not to trust the answers
-----------------------------

**No growth seen.** If the failures are not slowing down, there is
nothing for a growth model to measure: the best-fitting curve is a
straight line, and "the number of faults" runs off to infinity. SurPyval
refuses such a fit rather than print a huge number, and says what it saw:

.. jupyter-execute::

    steady = np.arange(10.0, 260.0, 10.0)   # a failure every 10 days
    try:
        GoelOkumoto.fit(steady)
    except ValueError as err:
        print(err)

Keep testing (growth may show later), or report the constant rate
(``HPP``) for what it is. Beware the S-shaped models here: they can fit
steady failures by supposing the peak has just passed, and will name a
number of faults a little above the number found. If ``GoelOkumoto``
refuses the data and ``YamadaDelayedS`` does not, the data show no
growth, and the S-shaped answer rests on its assumption, not on
evidence.

**Few failures.** With a dozen failures any of these curves fits, the
bounds are wide, and they are still too narrow: they come from the
curvature of the likelihood at its peak, and the likelihood of a growth
model falls away slowly towards "many more faults". Treat an upper bound
on the faults left as a floor on your uncertainty.

**Most of the faults still to find.** The models learn the total from the
way the curve bends over. If the failure rate has not fallen to well
below where it started, the total is a guess; compare ``iif`` at the end
of the test with ``iif`` early on.

**The testing changed.** A new test team, a new kind of test, a big
release of new code or a holiday all break the assumption that testing is
steady. Fit the stretch of testing that is like what comes next, use a
clock that measures effort (test cases run, hours of execution) in
place of days, or fit the change: see "When the testing was not steady"
above.

**The use is not like the test.** Reliability "now" is the reliability
under more of the same testing. If customers use the software
differently, the failure rate in use is a different number.

**The models disagree.** That is information. Above, two well-fitting
models put the faults left at 8 and at 1.5. Report the range, not the
one you prefer.

References
----------

.. [JM1972] Jelinski, Z. and Moranda, P. B. (1972), "Software
   reliability research", in *Statistical Computer Performance
   Evaluation* (ed. W. Freiberger), Academic Press, 465-484.

.. [LXZ2015] Liu, J., Xu, M. and Zhou, Z. (2015), "Parameter estimation
   of Jelinski-Moranda model based on weighted nonlinear least squares
   and heteroscedasticity", arXiv:1503.00094 (Table 1: the NTDS data).

.. [ACL1986] Abdel-Ghaly, A. A., Chan, P. Y. and Littlewood, B. (1986),
   "Evaluation of competing software reliability predictions", *IEEE
   Transactions on Software Engineering* SE-12(9), 950-967.

.. [LV1973] Littlewood, B. and Verrall, J. L. (1973), "A Bayesian
   reliability growth model for computer software", *Journal of the Royal
   Statistical Society C* 22(3), 332-346.

.. [OG1980] Okumoto, K. and Goel, A. L. (1980), "Optimum release time
   for software systems based on reliability and cost criteria", *Journal
   of Systems and Software* 1, 315-318.

.. [Kan2002] Kan, S. H. (2002), *Metrics and Models in Software Quality
   Engineering* (2nd ed.), Addison-Wesley, chapter 7.

.. [YON1986] Yamada, S., Ohtera, H. and Narihisa, H. (1986), "Software
   reliability growth models with testing-effort", *IEEE Transactions on
   Reliability* R-35(1), 19-23.

.. [Zhao1993] Zhao, M. (1993), "Change-point problems in software and
   hardware reliability", *Communications in Statistics: Theory and
   Methods* 22(3), 757-768.

Goel, A. L. and Okumoto, K. (1979), "Time-dependent error-detection rate
model for software reliability and other performance measures", *IEEE
Transactions on Reliability* R-28(3), 206-211.

Yamada, S., Ohba, M. and Osaki, S. (1983), "S-shaped reliability growth
modeling for software error detection", *IEEE Transactions on
Reliability* R-32(5), 475-484.

Ohba, M. (1984), "Inflection S-shaped software reliability growth model",
in *Stochastic Models in Reliability Theory*, Springer, 144-162.

Musa, J. D. and Okumoto, K. (1984), "A logarithmic Poisson execution time
model for software reliability measurement", *Proceedings of the 7th
International Conference on Software Engineering*, 230-238.
