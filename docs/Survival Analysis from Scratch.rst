Survival Analysis from Scratch
==============================

This page is for someone who has never met survival analysis. It assumes you
can read a little Python and know what an average is, and nothing else. By
the end you will know what makes "how long until something happens" data
different, how to estimate a survival curve by hand, how to fit and read a
Weibull distribution, how to compare two groups, and where to go next.

Two small stories run through the page, because the same statistics serve
both:

- **Pumps** (engineering): how many running hours does a pump last before it
  fails?
- **Relapse** (medicine): after treatment, how many weeks does a patient stay
  in remission before the disease returns?

Different fields use different words for the same ideas. An engineer says
*failure*, *unit* and *reliability*; a clinician says *event*, *subject* and
*survival*. This page uses them side by side. The data sets here are made up
for teaching, and small enough to check on paper.

What is time-to-event data?
---------------------------

Time-to-event data records, for each unit (a pump, a patient, a customer),
**how long it took for some event to happen**. Three things define it:

- a **starting point**: the pump is installed; the patient finishes
  treatment;
- an **event**: the pump fails; the patient relapses;
- a **clock** that runs between them: running hours; weeks.

The clock need not be calendar time. Cycles, kilometres and start-ups are all
"time" here. SurPyval calls the measured value ``x`` whatever its unit, and
answers in the unit you gave it.

Why you can't just average the times
------------------------------------

Six pumps were installed at one site. Today, four have failed and two are
still running:

.. list-table::
   :header-rows: 1
   :widths: 20 25 55

   * - Pump
     - Hours
     - What happened
   * - A
     - 150
     - failed
   * - B
     - 340
     - still running (installed recently)
   * - C
     - 560
     - failed
   * - D
     - 800
     - failed
   * - E
     - 1130
     - still running
   * - F
     - 1720
     - failed

How long does a pump last, on average? There are two tempting answers, and
both are wrong.

1. **Average the four failures**: :math:`(150 + 560 + 800 + 1720) / 4 =
   807.5` hours. This throws away pumps B and E. But pump E has already run
   1130 hours *without* failing, which is good news about pump life, and this
   answer ignores it.
2. **Average all six numbers**: :math:`783.3` hours. This pretends pumps B
   and E failed at 340 and 1130 hours. They did not; they will fail later
   than that.

Both answers are too low, and the more units that are still going, the worse
they get. A study that stopped after one failure would report that failure's
time as "the average life". The same trap exists in the relapse story:
patients who are still in remission when the study ends are the ones doing
*best*, so leaving them out, or counting the end of the study as a relapse,
makes any treatment look worse than it is.

What we know about pump E is not nothing. We know its life is *more than*
1130 hours. Survival analysis is the set of methods that uses exactly that
much information: no more, and no less.

Censoring: knowing only part of the answer
------------------------------------------

An observation is **censored** when we know the unit exists but know only a
*range* for its time.

- **Right censored**: the event had not happened when we last looked, so the
  time is *longer than* the value recorded. Pumps B and E are right censored
  (an engineer may call them *suspensions*). So is a patient still in
  remission when the study ends, or one who moves away and is lost to
  follow-up.
- **Left censored**: the event had already happened when we first looked, so
  the time is *shorter than* the value recorded. A pump found failed at its
  first inspection; a patient who has already relapsed at the first clinic
  visit.
- **Interval censored**: the event happened *between* two looks. A pump that
  was running at the 500 hour inspection and failed by the 600 hour one; a
  relapse found at the week 12 visit that was not there at week 8.

SurPyval records this with a flag ``c`` for every value ``x``:

.. list-table::
   :header-rows: 1
   :widths: 10 30 60

   * - ``c``
     - Meaning
     - What is known
   * - 0
     - observed exactly
     - the event happened at ``x``
   * - 1
     - right censored
     - the event happens after ``x``
   * - -1
     - left censored
     - the event happened before ``x``
   * - 2
     - interval censored
     - the event happened between the two values given, ``[lower, upper]``

The codes follow a timeline: the event is at the recorded value (0), to its
left (-1) or to its right (1). **Note the direction: 0 is the event and 1 is
censored.** Some other software, and many clinical data sets, use 1 for the
event; convert such a column (``c = 1 - event``) before fitting.

The six pumps are therefore:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    x = [150, 340, 560, 800, 1130, 1720]   # running hours
    c = [  0,   1,   0,   0,    1,    0]   # 0 = failed, 1 = still running

If ``c`` is left out, every value is taken to be an observed event. Two more
arrays complete SurPyval's data layout. ``n`` is a count, for when several
units share a row ("five pumps were still running at 1500 hours" is one row
with ``n = 5``). ``t`` (or ``tl`` and ``tr``) describes truncation, which is
next. :doc:`Conventions` defines all four exactly.

Censoring is only harmless if it is *uninformative*: the reason a unit was
censored must have nothing to do with how close it was to the event. The end
of a study is uninformative. A pump pulled from service because it was
running rough, or a patient who leaves a trial because they feel worse, is
not: those units were censored *because* the event was near, and every
method on this page would then be too optimistic.

Truncation: units you never saw
-------------------------------

Censoring hides part of an answer. **Truncation** hides whole units.

Suppose the site only started keeping records two years ago. A pump that was
installed earlier and failed before the records began is not in the data at
all: not as a failure, not as a censored value. The pumps that *are* in the
data from that era are there because they lasted long enough to be recorded,
so they are a hand-picked set of survivors. This is **left truncation** (also
called *delayed entry*). The medical version: a registry that enrols patients
some weeks after treatment can only enrol those who have not yet relapsed.

The fix is to tell the method when each unit came under observation. If pump
C was already 400 hours old, and pump E 600 hours old, when the records
began:

.. jupyter-execute::

    tl = [0, 0, 400, 0, 600, 0]   # age at which each pump entered the records

A method given ``tl`` counts a unit as "at risk" only from that age onwards.
The difference between the two ideas, in one line each:

- censored: the unit is in the data, with an incomplete time;
- truncated: units like it may be missing from the data altogether.

:doc:`Types of Data` works through every combination, including right
truncation.

Three ways to describe how long things last
-------------------------------------------

Because a single average is not available, survival analysis describes the
whole pattern of "how long". Three functions do it; they are three views of
the same thing.

**The survival function**, :math:`S(t)`, is the fraction of units that have
*not yet* had the event by time :math:`t`. It starts at 1 (everything is
running; everyone is in remission) and falls over time. "70% of pumps are
still running at 500 hours" is :math:`S(500) = 0.7`. Engineers call it the
*reliability function* and write :math:`R(t)`; it is the same thing. In
SurPyval it is ``sf``.

**The hazard**, :math:`h(t)`, answers a different question: *among the units
that have made it to time* :math:`t`, *how quickly are events happening right
now?* It is a rate (events per unit of time), not a probability. Its shape
over time is the story of the process:

- a hazard that **falls** means the risky period is early: pumps with
  assembly defects fail soon and the rest settle in; the risk of relapse is
  highest just after treatment;
- a hazard that is **flat** means age does not matter: events arrive by
  chance, as often for an old unit as a new one;
- a hazard that **rises** means wear-out: bearings wear, seals harden; risk
  grows with age.

Engineers call the hazard the *failure rate*. In SurPyval it is ``hf``.

**The cumulative hazard**, :math:`H(t)`, is the hazard added up from the
start to time :math:`t`: the total amount of risk a unit has been exposed to
so far. It is less intuitive than the other two, but it is the easiest to
estimate and to plot, and its *shape* shows the hazard: a straight line means
a constant hazard, a curve bending upwards means a rising one. In SurPyval it
is ``Hf``.

Now the formulas. With :math:`T` the time of the event:

.. math::

    S(t) = P(T > t), \qquad
    h(t) = \frac{f(t)}{S(t)}, \qquad
    H(t) = \int_0^t h(u)\,du, \qquad
    S(t) = e^{-H(t)}

where :math:`f(t)` is the probability density of :math:`T`. The last identity
ties them together: know one of the three and you know the others.
:doc:`Handy References - Aide-mémoire` lists every such relationship.

Kaplan-Meier by hand
--------------------

The **Kaplan-Meier estimator** estimates the survival function directly from
censored data, with no assumption about its shape. The idea is simple: to
survive past 800 hours a pump must survive the first failure time, *and* the
second, *and* the third. So walk through the failure times in order, and at
each one ask: *of the units still under observation just before this time,
what fraction got through it?* Multiply those fractions together.

A censored unit is counted among those under observation ("at risk") for as
long as we watched it, and then quietly leaves. That is how its information
is used: pump E is in the denominator at 150, 560 and 800 hours, which is
exactly what we know about it.

For the six pumps:

.. list-table::
   :header-rows: 1
   :widths: 12 14 12 26 36

   * - Hours
     - At risk
     - Failed
     - Fraction surviving this time
     - Survival so far
   * - 150
     - 6
     - 1
     - 5/6
     - 5/6 = 0.833
   * - 340
     - 5
     - 0 (B censored)
     - 5/5
     - 0.833
   * - 560
     - 4
     - 1
     - 3/4
     - 5/6 × 3/4 = 0.625
   * - 800
     - 3
     - 1
     - 2/3
     - 0.625 × 2/3 = 0.417
   * - 1130
     - 2
     - 0 (E censored)
     - 2/2
     - 0.417
   * - 1720
     - 1
     - 1
     - 0/1
     - 0

Read the "at risk" column carefully: it falls by one at every row, whether
the pump failed or was censored, but the survival estimate only steps down at
a failure. The same calculation in SurPyval:

.. jupyter-execute::

    km = surv.KaplanMeier.fit(x, c)
    print("at risk :", km.r)
    print("failed  :", km.d)
    print("survival:", km.R.round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert km.r.tolist() == [6, 5, 4, 3, 2, 1], km.r
    assert km.d.tolist() == [1, 0, 1, 1, 0, 1], km.d
    assert np.allclose(km.R, [5 / 6, 5 / 6, 5 / 8, 5 / 12, 5 / 12, 0]), km.R

In symbols, with :math:`d_i` events among :math:`r_i` units at risk at the
:math:`i`-th event time :math:`x_i`:

.. math::

    \hat{S}(t) = \prod_{x_i \le t} \left(1 - \frac{d_i}{r_i}\right)

The fitted model answers questions at any time. The estimate is a staircase:
flat between failures, with a step down at each one.

.. jupyter-execute::

    print("S(500), S(1000):", km.sf([500, 1000]).round(3))
    print("median life    :", km.qf(0.5))

The **median** is the time by which half the units have had the event: the
first time the curve reaches 0.5 or below, here 800 hours. The median is the
usual one-number summary in survival analysis because, unlike the mean, it
can often be read off even when the longest-lived units are censored.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(km.sf([500, 1000]), [5 / 6, 5 / 12]), km.sf([500, 1000])
    assert float(km.qf(0.5)) == 800.0, km.qf(0.5)

The cumulative hazard has an equally simple estimator, the **Nelson-Aalen
estimator**: add up the fraction failing at each event time,
:math:`\hat{H}(t) = \sum_{x_i \le t} d_i / r_i`. For the pumps that is
:math:`1/6 = 0.167` after the first failure, :math:`1/6 + 1/4 = 0.417` after
the second and :math:`1/6 + 1/4 + 1/3 = 0.75` after the third:

.. jupyter-execute::

    na = surv.NelsonAalen.fit(x, c)
    print(na.Hf([150, 560, 800]).round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _H = na.Hf([150, 560, 800])
    assert np.allclose(_H, [1 / 6, 1 / 6 + 1 / 4, 1 / 6 + 1 / 4 + 1 / 3]), _H

Left truncation changes only the "at risk" column: a unit is counted from the
age at which it entered the records. With the entry ages from above, pumps C
and E are not yet at risk at 150 hours, so the first step is 3/4 rather than
5/6:

.. jupyter-execute::

    km_late = surv.KaplanMeier.fit(x, c, tl=tl)
    print("at risk :", km_late.r)
    print("survival:", km_late.R.round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert km_late.r[0] == 4 and np.isclose(km_late.R[0], 0.75), km_late.r

How sure are we? Bounds
-----------------------

Six pumps cannot pin down a survival curve. Every estimate should travel with
**confidence bounds**: a range that would contain the true value in (say) 95%
of studies like this one. ``cb`` gives them:

.. jupyter-execute::

    print("S(1000)           :", km.sf(1000).round(3))
    print("95% bounds on it  :", km.cb(1000).round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _lo, _hi = np.ravel(km.cb(1000))
    assert round(_lo, 2) == 0.06 and round(_hi, 2) == 0.77, (_lo, _hi)

The estimate is 0.42, but anything from 0.06 to 0.77 is consistent with the
data. That is not a flaw in the method. It is the honest size of what six
pumps can tell you, and it is the reason to quote bounds with every number.
More data narrows them, and it is the number of *events*, not the number of
units, that does most of the narrowing.

Fitting a Weibull distribution
------------------------------

Kaplan-Meier describes the data you have and stops where the data stops. A
**parametric model** assumes the survival function has a particular smooth
shape, described by a few numbers (parameters). In return it gives smooth
curves, a hazard, and the ability to ask about times with little or no data
(with due caution).

The **Weibull** distribution is the usual first choice because its two
parameters cover all three hazard stories:

.. math::

    S(t) = \exp\left[-\left(\frac{t}{\alpha}\right)^{\beta}\right]

- :math:`\alpha`, the **scale**, is in the units of time. It is the time by
  which 63.2% of units have had the event (engineers call it the
  *characteristic life*). A larger :math:`\alpha` stretches the whole curve
  to later times.
- :math:`\beta`, the **shape**, has no units and sets the hazard:
  :math:`\beta < 1` is a falling hazard, :math:`\beta = 1` a constant one
  (the exponential distribution) and :math:`\beta > 1` a rising one.

Here is the whole fleet of twenty pumps: eleven failures, and nine pumps
still running at 1000, 1500 or 2000 hours. The count array ``n`` saves
repeating rows:

.. jupyter-execute::

    hours = [270, 340, 390, 500, 720, 740, 840, 890, 1000,
             1140, 1280, 1460, 1500, 2000]
    c_fleet = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]
    n_fleet = [1, 1, 1, 1, 1, 1, 1, 1, 3, 1, 1, 1, 5, 1]

    weibull = surv.Weibull.fit(hours, c_fleet, n_fleet)
    print(weibull)

The fit is by *maximum likelihood*: it chooses the :math:`\alpha` and
:math:`\beta` that make the data seen the most probable, where a failure
contributes the chance of failing *at* its time and a censored pump
contributes the chance of surviving *past* its time. That is how censored
units pull the answer towards longer lives without anyone pretending to know
when they fail.

Reading the result: :math:`\alpha \approx 1620` hours, so about 63% of pumps
are expected to have failed by then; :math:`\beta \approx 1.6` is above 1,
which points to wear-out. Ask the model the questions you care about:

.. jupyter-execute::

    print("S(1000)                 :", weibull.sf(1000).round(3))
    print("  95% bounds            :", weibull.cb(1000).round(3))
    print("time when 10% failed    :", weibull.qf(0.1).round(0))
    print("  95% bounds            :", weibull.quantile_cb(0.1).round(0))
    print("95% bounds on the shape :", weibull.param_cb("beta").round(2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _a, _b = weibull.params
    assert round(_a, -1) == 1620 and round(_b, 1) == 1.6, weibull.params
    assert round(float(weibull.sf(1000)), 2) == 0.63, weibull.sf(1000)
    _lo, _hi = np.ravel(weibull.cb(1000))
    assert round(_lo, 2) == 0.42 and round(_hi, 2) == 0.78, (_lo, _hi)
    assert round(float(weibull.qf(0.1)), -1) == 400, weibull.qf(0.1)
    _blo, _bhi = np.ravel(weibull.param_cb("beta"))
    assert _blo < 1 < _bhi and round(_blo, 2) == 0.96, (_blo, _bhi)
    _km_fleet = surv.KaplanMeier.fit(hours, c_fleet, n_fleet)
    assert round(float(_km_fleet.sf(1000)), 2) == 0.60, _km_fleet.sf(1000)
    _naive = surv.Weibull.fit(hours, n=n_fleet)
    assert round(_naive.params[0], -1) == 1190, _naive.params

About 63% of pumps are expected to survive 1000 hours (between 42% and 78%),
and 10% are expected to have failed by about 400 hours (the engineer's *B10
life*; a clinician would call it the 10th percentile of the time to the
event). Two lessons sit in the last line. First, the bounds on the shape run
from just under 1 to 2.7, so although wear-out is the best reading, twenty
pumps cannot quite rule out a constant hazard. Second, compare the fit that
ignores censoring, ``surv.Weibull.fit(hours, n=n_fleet)``: its scale is about
1190 hours instead of 1620, the same "too low" error as the naive average at
the top of the page.

Always check a parametric fit against the non-parametric estimate, which
assumes nothing. Here the Kaplan-Meier estimate of surviving 1000 hours is
0.60, close to the Weibull's 0.63. ``weibull.plot()`` draws the comparison: the
points are the non-parametric estimate, the line is the fitted distribution,
and on these axes a Weibull is a straight line.

.. jupyter-execute::

    weibull.plot();

If the points bend away from the line, try another distribution;
:doc:`Parametric SurPyval Modelling` shows how, and ``surv.fit_best`` ranks a
set of candidates.

Comparing two groups
--------------------

Now the relapse story. Thirty patients were randomised to a control or a new
treatment and followed for up to 52 weeks. ``x`` is the week of relapse, or
the last week the patient was known to be in remission (``c = 1``):

.. jupyter-execute::

    x_control = [4, 5, 6, 7, 11, 12, 13, 13, 15, 18, 23, 27, 27, 41, 43]
    c_control = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    x_treated = [8, 9, 17, 17, 22, 25, 31, 35, 38, 43, 43, 46, 46, 51, 52]
    c_treated = [0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1]

    km_control = surv.KaplanMeier.fit(x_control, c_control)
    km_treated = surv.KaplanMeier.fit(x_treated, c_treated)

    for name, model in [("control", km_control), ("treated", km_treated)]:
        print(name, "| in remission at 26 weeks:", model.sf(26).round(2),
              "| median weeks to relapse:", model.qf(0.5))

.. jupyter-execute::

    import matplotlib.pyplot as plt

    for name, model in [("control", km_control), ("treated", km_treated)]:
        plt.step(np.r_[0, model.x], np.r_[1, model.R], where="post", label=name)
    plt.xlabel("weeks since treatment")
    plt.ylabel("fraction still in remission")
    plt.legend();

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(float(km_control.sf(26)), 2) == 0.32, km_control.sf(26)
    assert round(float(km_treated.sf(26)), 2) == 0.87, km_treated.sf(26)
    assert float(km_control.qf(0.5)) == 13 and float(km_treated.qf(0.5)) == 46
    assert sum(c_treated) == 8 and sum(c_control) == 1

At 26 weeks, 32% of the control group are still in remission against 87% of
the treated group, and the median time to relapse is 13 weeks against 46.
Notice that eight of the fifteen treated patients are censored, against one
in the control group: a comparison of plain averages would have discarded, or
misused, most of the treated group's good news.

Is the difference more than chance? The **log-rank test** compares whole
curves. At every event time it asks how many events each group *would* have
had if the groups were the same, given how many were at risk in each, and
adds up the gap between that and what happened:

.. jupyter-execute::

    x_all = np.array(x_control + x_treated)
    c_all = np.array(c_control + c_treated)
    treated = np.array([0] * 15 + [1] * 15)

    print(surv.logrank(x_all, treated, c_all))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _lr = surv.logrank(x_all, treated, c_all)
    assert _lr.p_value < 0.001, _lr.p_value

A p-value this small (below 0.001) says a gap this large would be very rare
if the treatment did nothing. The same test compares pumps from two
suppliers, or bearings run with two lubricants. A test says *whether* the
groups differ; it does not say by how much. For that, and for anything beyond
one yes/no grouping, use regression.

Where regression comes in
-------------------------

Groups are rarely alike in everything but the treatment. Patients differ in
age; pumps differ in duty, fluid and speed. **Regression** models how the
time to the event depends on such *covariates*, several at once, so that the
effect of one is estimated with the others held fixed. Covariates go in
``Z``, one row per row of ``x``.

The most used model is **Cox proportional hazards**. It assumes each
covariate multiplies the hazard by a constant factor, the *hazard ratio*,
and, like Kaplan-Meier, assumes nothing about the shape of the hazard over
time. Here it is with treatment and age:

.. jupyter-execute::

    age = [58, 50, 64, 74, 68, 54, 55, 62, 44, 42, 57, 66, 53, 68, 56,
           52, 57, 45, 52, 40, 37, 57, 45, 50, 55, 59, 59, 44, 46, 52]
    Z = np.column_stack([treated, age])

    cox = surv.CoxPH.fit(x_all, Z, c_all)
    print(cox)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _hr = np.exp(cox.params)
    assert round(_hr[0], 2) == 0.19 and round(_hr[1], 2) == 1.03, _hr
    assert cox.p_values[0] < 0.01 < 0.05 < cox.p_values[1], cox.p_values

Read the ``exp(coef)`` column. For treatment (``coef_0``) the hazard ratio is
0.19: at any moment, a treated patient's rate of relapse is about a fifth of
that of a control patient of the same age, and the interval around the
coefficient excludes "no effect". For age (``coef_1``) it is 1.03 per year,
with an interval that includes no effect, so these data do not show that age
matters. A hazard ratio below 1 means fewer events; above 1, more.

The engineering reading is identical: replace "treated" with "new seal
design" and "age" with "operating temperature". Engineering also has its own
regression family, *accelerated life* models, in which a stress such as
temperature or load shortens life by a known physical law, so that tests at
high stress can predict life at normal stress. :doc:`regression analysis`
explains every family and when each fits.

Where to go next
----------------

You now have the core of the subject: censoring and truncation, the three
functions, a non-parametric estimate, a parametric fit, bounds, a comparison
and a regression. From here:

- :doc:`Quickstart` shows one small example of every area of the package.
- :doc:`Types of Data` goes deeper into censoring and truncation, and how to
  recognise each in a real data set. This matters more than any choice of
  model.
- :doc:`Data Wrangler Examples` turns the data you have (a spreadsheet, a
  list of failures and survivors, a study table) into ``x``, ``c``, ``n``
  and ``t``; :doc:`Conventions` is the reference for them.
- :doc:`Non-Parametric Estimation` and :doc:`Parametric Estimation` give the
  theory behind this page; :doc:`Non-Parametric SurPyval Modelling` and
  :doc:`Parametric SurPyval Modelling` are the how-to pages.
- :doc:`regression analysis` and :doc:`Regression Modelling with SurPyval`
  cover Cox, accelerated failure time and the other regression models.
- :doc:`Model Diagnostics` shows how to check that a fitted model describes
  its data.
- More than one kind of event (relapse or death from another cause; failure
  by wear or by shock)? See :doc:`Competing Risks Analysis`.
- Events that happen again and again to the same unit (repeated repairs,
  repeated hospital admissions)? See :doc:`Recurrent Event Analysis`.
- A measurement drifting towards a threshold (wear, a biomarker)? See
  :doc:`Degradation Analysis` and :doc:`Joint Models`.
