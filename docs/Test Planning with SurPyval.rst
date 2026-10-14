Test Planning with SurPyval
===========================

Before a reliability test is run, four questions decide what it costs and
what it can show:

- **Will it demonstrate the target?** A demonstration test shows, at a
  confidence level, that a design meets a reliability or MTBF target.
- **Can it decide sooner?** A sequential test stops as soon as the
  evidence is clear.
- **How precise will the estimate be?** A life test run to estimate a
  B-life or a mean needs enough units and time for the precision wanted;
  an accelerated test also needs the right stresses.
- **Will a comparison detect the difference?** Comparing a new design with
  an old one needs enough events to detect the improvement.

``surpyval.test_planning`` answers each from planning values: the target,
and assumed values of what the test will measure. Every function here plans
a test; the fitters elsewhere in SurPyval analyse it once it is run. As
everywhere in SurPyval, a confidence level is given as ``alpha_ci``, the
significance level: for a demonstration test it is the *consumer's risk*,
the chance that a design no better than the target passes, ``0.05`` for 95%
confidence.

Demonstration tests
-------------------

A pass/fail (attribute) test runs ``n`` units for a mission each. With no
failures allowed, 59 units demonstrate 95% reliability at 95% confidence,
the success run of :func:`surpyval.success_run`; allowing one failure takes
93:

.. jupyter-execute::

    from surpyval.test_planning import (
        demonstrated_reliability,
        demonstration_pass_probability,
        demonstration_plan,
        demonstration_sample_size,
    )

    print(demonstration_sample_size(0.95), demonstration_sample_size(0.95, 1))
    print(round(demonstrated_reliability(59), 4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert demonstration_sample_size(0.95) == 59
    assert demonstration_sample_size(0.95, 1) == 93

A test has a second risk, the *producer's*: that a good design fails. The
22-unit zero-failure test that demonstrates 90% reliability at 90%
confidence passes a design of 95% reliability only about a third of the
time. :func:`~surpyval.test_planning.demonstration_plan` finds the smallest
test that keeps both risks: here 128 units, allowing 8 failures.

.. jupyter-execute::

    print(round(demonstration_pass_probability(0.95, 22), 2))
    plan = demonstration_plan(0.9, 0.95, alpha_ci=0.1, producer_risk=0.2)
    plan

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(demonstration_pass_probability(0.95, 22), 2) == 0.32
    assert (plan.n, plan.failures) == (128, 8)

When the lifetime's Weibull shape is known, testing each unit for longer
than a mission trades units for test time (the extended success run, or
Weibayes): for a wear-out shape of 2, two missions per unit need a quarter
of the units, 15 instead of 59.

.. jupyter-execute::

    demonstration_sample_size(0.95, test_multiple=2.0, shape=2.0)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert demonstration_sample_size(0.95, test_multiple=2.0, shape=2.0) == 15

At a constant failure rate an MTBF is demonstrated by total test time,
failed units repaired or replaced: about three MTBFs of testing with no
failures, at 95% confidence. :func:`~surpyval.test_planning.mtbf_demonstration_plan`
gives the fixed-length plans of MIL-HDBK-781 that keep both risks; to show
an MTBF of 1000 hours at 90% confidence, passing a 2000-hour design 80% of
the time, the test runs about 14,200 unit hours and allows 9 failures.

.. jupyter-execute::

    from surpyval.test_planning import mtbf_demonstration_plan, mtbf_test_time

    print(round(mtbf_test_time(1000.0)))
    mtbf_demonstration_plan(1000.0, 2000.0, alpha_ci=0.1)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(mtbf_test_time(1000.0)) == 2996
    fixed = mtbf_demonstration_plan(1000.0, 2000.0, alpha_ci=0.1)
    assert (round(fixed.test_time, -2), fixed.failures) == (14200, 9)

Sequential tests
----------------

A sequential test watches the failures as they come: it accepts the design
once enough test time has passed for the failures seen, rejects it if they
come too soon, and otherwise keeps testing. With the same risks it usually
stops well before the fixed-length test. Truncated at the fixed-length test,
as MIL-HDBK-781 requires, it never runs longer.

To show an MTBF of 100 hours, a 200-hour design to pass, both risks 10%,
the fixed-length test runs 2,013 hours; the sequential test about 1,080 on
average for the good design and 910 for one at the target:

.. jupyter-execute::

    from surpyval.test_planning import sequential_mtbf_test

    test = sequential_mtbf_test(100.0, 200.0, alpha_ci=0.1, producer_risk=0.1)
    print(round(test.expected_test_time(200.0)), round(test.expected_test_time(100.0)))
    test

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(test.max_time) == 2013
    assert round(test.expected_test_time(200.0), -1) == 1080
    assert round(test.expected_test_time(100.0), -1) == 910

The test's ``decide`` method says what to do at any point, and its risks
are computed exactly: Wald's boundaries and the truncation move them a
little from the 10% asked for, here to about 13% and 10%. The pass/fail
analogue is :func:`~surpyval.test_planning.sequential_demonstration_test`.

.. jupyter-execute::

    print(test.decide(failures=2, test_time=800.0))
    print(round(test.consumer_risk, 2), round(test.producer_risk, 2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert test.decide(failures=2, test_time=800.0) == "accept"
    assert round(test.consumer_risk, 2) == 0.13
    assert round(test.producer_risk, 2) == 0.10

Planning a life test for precision
----------------------------------

A life test run to *estimate* a B-life or a mean needs enough failures for
the precision wanted. The precision is the factor ``R`` of the confidence
interval ``[estimate / R, estimate * R]``; "within 20%" is ``R = 1.2``. It
depends on planning values of the distribution, above all its shape, which
earlier tests or engineering knowledge give.

Thirty units of a wear-out part (Weibull, scale 1000 hours, shape 2) on a
500-hour test, about 7 failures, estimate its B10 life of 325 hours to
within a factor of 1.62 at 95% confidence:

.. jupyter-execute::

    import surpyval
    from surpyval.test_planning import (
        life_test_length,
        life_test_precision,
        life_test_sample_size,
    )

    plan = life_test_precision(surpyval.Weibull, (1000.0, 2.0), 30,
                               censor_time=500.0)
    plan

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(plan.expected_failures) == 7
    assert round(plan.estimate) == 325
    assert round(plan.precision, 2) == 1.62

The same functions answer the question the other way round: how many units,
or how long a test, for a precision. To estimate the B10 life within 30% at
90% confidence takes 72 units on a 600-hour test; forty units need a test of
about 330 hours to estimate it within a factor of 1.5.

.. jupyter-execute::

    print(life_test_sample_size(surpyval.Weibull, (1000.0, 2.0), 1.3,
                                censor_time=600.0, alpha_ci=0.1).n)
    print(round(life_test_length(surpyval.Weibull, (1000.0, 2.0), 40, 1.5,
                                 alpha_ci=0.1).censor_time))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert life_test_sample_size(surpyval.Weibull, (1000.0, 2.0), 1.3,
                                 censor_time=600.0, alpha_ci=0.1).n == 72
    assert round(life_test_length(surpyval.Weibull, (1000.0, 2.0), 40, 1.5,
                                  alpha_ci=0.1).censor_time, -1) == 330

The precision is a large-sample approximation, from the test's expected
Fisher information. ``simulate`` checks it: it simulates the test, fits the
distribution to each simulated data set, and reports the precision the
estimates actually had, and how often they fell within the planned factor.

.. jupyter-execute::

    plan.simulate(200, random_state=1)

Accelerated life tests
----------------------

An accelerated test runs units at raised stress to get failures in time, and
estimates the life at the use stress through a life-stress model. Where to
put the stresses matters as much as how many units: the units at the lowest
stress, closest to use, carry the extrapolation, but fail least.
:func:`~surpyval.test_planning.accelerated_test_plan` chooses the lowest
stress and the share of units at it, with the highest stress fixed.

For a voltage endurance test (power law, life :math:`10^6 V^{-2}` hours,
Weibull shape 2), 40 units for 1000 hours, the highest stress 60 V, to
estimate the B10 life at 10 V: the optimum plan tests about 78% of the units
near 25 V and the rest at 60 V. A compromise plan of three equally spaced
levels (Meeker and Hahn) puts 20% of the units in the middle, giving up a
little precision for protection against a life-stress relation that bends:

.. jupyter-execute::

    from surpyval import life_models
    from surpyval.test_planning import accelerated_test_plan

    voltage = dict(use_stress=10.0, high_stress=60.0, n=40, censor_time=1000.0)
    optimum = accelerated_test_plan(
        surpyval.Weibull, life_models.Power, {"a": 1e6, "n": -2.0}, 2.0,
        **voltage)
    compromise = accelerated_test_plan(
        surpyval.Weibull, life_models.Power, {"a": 1e6, "n": -2.0}, 2.0,
        levels=3, **voltage)
    print(optimum)
    print(compromise)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(optimum.fractions[0], 2) == 0.78
    assert round(optimum.stresses[0]) == 25
    assert optimum.precision < compromise.precision

:func:`~surpyval.test_planning.accelerated_test_precision` gives the
precision of a test you propose, and the plans' ``simulate`` method checks
the approximation by fitting :class:`surpyval.AcceleratedLife` to simulated
tests.

Comparing two groups
--------------------

To show that a new design fails less often than the old, a log-rank test
needs enough *events*: detecting a halving of the hazard with 80% power at
the 5% level takes 66, with equal groups (Schoenfeld's formula). The units
follow from the chance each has of failing during the study: if 25% of the
new units and 35% of the old would fail, detecting a 30% lower hazard takes
412 units in each group.

.. jupyter-execute::

    from surpyval.test_planning import (
        logrank_events,
        logrank_power,
        logrank_sample_size,
    )

    print(logrank_events(0.5))
    print(logrank_sample_size(0.7, (0.25, 0.35)))
    print(round(logrank_power(0.7, (200, 200), (0.25, 0.35)), 2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert logrank_events(0.5) == 66
    assert logrank_sample_size(0.7, (0.25, 0.35)) == (412, 412)

Freedman's formula, as lifelines' ``sample_size_necessary_under_cph`` uses,
is the option ``method="freedman"``.

References
----------

- Meeker, W. Q. and Escobar, L. A. (1998). *Statistical Methods for
  Reliability Data*, chapters 10 and 20. Wiley.
- Nelson, W. (1990). *Accelerated Testing*, chapter 6. Wiley.
- Meeker, W. Q. and Hahn, G. J. (1985). *How to Plan an Accelerated Life
  Test*. ASQC.
- MIL-HDBK-781A (1996). *Reliability Test Methods, Plans, and Environments
  for Engineering, Development, Qualification, and Production*.
- Wald, A. (1947). *Sequential Analysis*. Wiley.
- Schoenfeld, D. (1983). Sample-size formula for the proportional-hazards
  regression model. *Biometrics* 39, 499-503.
- Freedman, L. S. (1982). Tables of the number of patients required in
  clinical trials using the logrank test. *Statistics in Medicine* 1,
  121-129.
