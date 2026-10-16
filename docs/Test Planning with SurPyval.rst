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
confidence. The other party's risk, the *supplier's* or producer's (that a
good design fails), is ``producer_risk``; the section on procurement below
works through both.

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

Supplier's and consumer's risk in procurement
---------------------------------------------

A reliability requirement in a contract is usually checked by an acceptance
test, and every acceptance test carries two risks, one for each party:

- the **consumer's risk** (the buyer's): that a design no better than the
  requirement passes the test and is accepted;
- the **supplier's risk**, or producer's risk: that a design which is good
  fails the test and is rejected.

A contract that states only a reliability "at 90% confidence" fixes the
first and leaves the second to chance. The examples below show how large it
can be, how to agree a test that bounds both, and how to check a test clause
that is already written.

**Notation.** MIL-HDBK-781 and most procurement standards call the
producer's (supplier's) risk :math:`\alpha` and the consumer's risk
:math:`\beta`. SurPyval writes every confidence level as ``alpha_ci``, and
for a demonstration test that is the *consumer's* risk; the supplier's is
``producer_risk``. So a handbook plan with :math:`\alpha = 10\%` and
:math:`\beta = 20\%` is ``producer_risk=0.1, alpha_ci=0.2``.

The supplier's risk hidden in a zero-failure test
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The familiar 59-unit, zero-failure test demonstrates 95% reliability at 95%
confidence: a design at exactly 95% passes it only 5% of the time. But the
supplier's side is far worse than most contracts realise. A design of 99%
reliability, five times fewer failures than the requirement allows, fails
the test 45% of the time; to pass it nine times in ten, a design needs about
99.8%. The test's *operating characteristic*, its chance of passing against
the design's true reliability, shows both risks at once:

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np

    reliability = np.linspace(0.9, 0.999, 200)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(reliability, demonstration_pass_probability(reliability, 59),
            label="59 units, no failures allowed")
    ax.plot(reliability, demonstration_pass_probability(reliability, 128, 8),
            label="128 units, 8 failures allowed")
    ax.axvline(0.95, color="grey", lw=0.8, ls="--")
    ax.set_xlabel("True reliability of the design")
    ax.set_ylabel("Chance of passing the test")
    ax.legend()
    plt.show()

    for r in (0.97, 0.98, 0.99, 0.995):
        print(f"{r}: supplier's risk {1 - demonstration_pass_probability(r, 59):.3f}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    from scipy import optimize

    assert round(1 - demonstration_pass_probability(0.99, 59), 2) == 0.45
    needed = optimize.brentq(
        lambda r: demonstration_pass_probability(r, 59) - 0.9, 0.95, 0.99999)
    assert round(needed, 3) == 0.998

The steeper the curve, the better the test separates good designs from bad
ones; allowing a few failures with more units is what makes it steep.

Agreeing a plan that bounds both risks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A fair acceptance test is agreed from four numbers, two from each party:
the buyer's requirement and consumer's risk, and the supplier's expected
("good") reliability and supplier's risk.
:func:`~surpyval.test_planning.demonstration_plan` gives the smallest test
that keeps both. The closer the good reliability is to the requirement, the
larger the test: for a requirement of 95% and both risks 10%, a supplier
confident of 99% needs 105 units allowing 2 failures, but one expecting only
97% needs 628 units allowing 24.

.. jupyter-execute::

    for good in (0.99, 0.98, 0.97):
        plan = demonstration_plan(0.95, good, alpha_ci=0.1, producer_risk=0.1)
        print(f"good design {good}: {plan.n} units, "
              f"accept with up to {plan.failures} failures "
              f"(consumer's risk {plan.consumer_risk:.3f}, "
              f"supplier's risk {plan.producer_risk:.3f})")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    near = demonstration_plan(0.95, 0.99, alpha_ci=0.1, producer_risk=0.1)
    far = demonstration_plan(0.95, 0.97, alpha_ci=0.1, producer_risk=0.1)
    assert (near.n, near.failures) == (105, 2)
    assert (far.n, far.failures) == (628, 24)

An MTBF acceptance test (MIL-HDBK-781)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For equipment with a constant failure rate the requirement is an MTBF, and
the test runs a total number of unit-hours, failed units repaired. The
handbook calls the requirement the *lower test MTBF* (:math:`\theta_1`), the
supplier's design goal the *upper test MTBF* (:math:`\theta_0`), and their
ratio the *discrimination ratio* :math:`d = \theta_0 / \theta_1`. For a
requirement of 500 hours, the test shrinks quickly as the ratio grows or the
risks are relaxed: with both risks 20% and a ratio of 3 it is about 2,140
unit hours, accepting with up to 2 failures (the handbook's plan XVII-D,
4.3 times the requirement).

.. jupyter-execute::

    for risk in (0.1, 0.2):
        for ratio in (1.5, 2.0, 3.0):
            plan = mtbf_demonstration_plan(
                500.0, 500.0 * ratio, alpha_ci=risk, producer_risk=risk)
            print(f"risks {risk:.0%}, d = {ratio}: {plan.test_time:,.0f} unit hours, "
                  f"accept with up to {plan.failures} failures")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    xvii = mtbf_demonstration_plan(500.0, 1500.0, alpha_ci=0.2,
                                   producer_risk=0.2)
    assert (round(xvii.test_time, -1), xvii.failures) == (2140, 2)
    assert round(xvii.test_time / 500.0, 1) == 4.3

A sequential test with the same risks
(:func:`~surpyval.test_planning.sequential_mtbf_test`) usually finishes much
sooner: for the requirement of 500 hours, a goal of 1000 and both risks 10%,
about 5,400 unit hours on average for a design at the goal, against 10,064
for the fixed-length test. Its exact risks as run are a little different
from the 10% asked for (about 13% consumer's and 10% supplier's), because of
the truncation; a contract that must not exceed a stated consumer's risk
should say which kind of test is meant.

.. jupyter-execute::

    test = sequential_mtbf_test(500.0, 1000.0, alpha_ci=0.1, producer_risk=0.1)
    print(round(test.expected_test_time(1000.0)), round(test.max_time))
    print(round(test.consumer_risk, 3), round(test.producer_risk, 3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(test.expected_test_time(1000.0), -2) == 5400
    assert round(test.max_time) == 10064
    assert round(test.consumer_risk, 2) == 0.13
    assert round(test.producer_risk, 2) == 0.10

Checking a test clause that is already written
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Contracts often state a test without its risks: *"the equipment shall be
tested for 3,000 hours and accepted with no more than 2 failures; required
MTBF 500 hours."* :func:`~surpyval.test_planning.mtbf_pass_probability`
gives both risks. Equipment at the required MTBF passes 6% of the time, so
the buyer is well protected; but equipment twice as good, at 1000 hours,
fails 58% of the time. The clause puts almost all of the risk on the
supplier, who would want either a longer test allowing more failures, or a
design goal well above 1000 hours.

.. jupyter-execute::

    from surpyval.test_planning import mtbf_pass_probability

    for mtbf in (500.0, 750.0, 1000.0, 1500.0):
        print(f"true MTBF {mtbf:,.0f} h: passes {mtbf_pass_probability(mtbf, 3000.0, 2):.1%}")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(mtbf_pass_probability(500.0, 3000.0, 2), 2) == 0.06
    assert round(1 - mtbf_pass_probability(1000.0, 3000.0, 2), 2) == 0.58

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
