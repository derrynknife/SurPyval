Software Reliability Growth
===========================

While software is tested, it fails; each failure is traced to a fault, and
the fault is fixed. If the fixes work, failures come less and less often.
A *software reliability growth model* is a curve fitted to those failures,
which answers the questions a test or release manager asks:

- How many faults are left?
- How reliable is the software now?
- How much more testing would it take to reach a target?

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

A whole number of faults: Jelinski-Moranda
-----------------------------------------

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
other models fitted to the same failure times.

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
steady. Fit the stretch of testing that is like what comes next, or use
a clock that measures effort (test cases run, hours of execution) in
place of days.

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
