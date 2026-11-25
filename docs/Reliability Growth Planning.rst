Reliability Growth Planning
===========================

A reliability growth programme tests a new design, finds what makes it
fail, fixes those things and tests again. Before it starts, someone has to
say how long it will run, how fast reliability must improve, what MTBF
each test phase should show, and how likely the programme is to show the
requirement at the end. Those are planning questions, answered before any
failure is seen; once the test is running, the growth models
(:class:`surpyval.CrowAMSAA`, :class:`surpyval.Duane` and
``CrowAMSAA.projection``) measure what it is actually achieving, and the
two go on one chart.

``surpyval.test_planning`` has the three planning models of MIL-HDBK-189C
(the US Department of Defense handbook on reliability growth management,
2011):

- **the idealised growth curve**, which assumes the MTBF grows as a power
  of test time at a chosen growth rate (Duane's pattern);
- **PM2-Continuous**, planned from what management controls instead: the
  share of the failure intensity it will fix and how effective the fixes
  are;
- **PM2-Discrete**, the same for one-shot items tested in trials.

As everywhere in SurPyval, a confidence level is given as ``alpha_ci``,
the significance level (``alpha_ci=0.2`` for 80% confidence).

The idealised growth curve
--------------------------

The system starts at an initial MTBF, ``M_I``, averaged over a first test
phase of length ``t_1``. The fixes at the end of that phase start the
growth; from then on, with every fix in, the MTBF rises as a power of the
total test time. The power is the *growth rate*, ``alpha``: the slope of
the cumulative MTBF against test time on log-log axes. Around 0.1 is slow,
0.3 to 0.4 a vigorous programme with a strong fix process, and above 0.5
is rarely seen.

Give two of the growth rate, the final MTBF and the test time, and
:func:`~surpyval.test_planning.idealised_growth_curve` solves for the
third:

.. code:: python

    from surpyval.test_planning import idealised_growth_curve

    # 68 hours over the first 500, growing at 0.23 for 2800 hours.
    curve = idealised_growth_curve(68.0, 500.0, growth_rate=0.23,
                                   test_time=2800.0)
    curve.final_mtbf                   # 131.3 hours
    curve.expected_failures(2800.0)    # 27.7 failures

    # How long to reach 130 hours at that rate?
    idealised_growth_curve(68.0, 500.0, growth_rate=0.23,
                           final_mtbf=130.0).test_time   # 2686 hours

    # What rate reaches 130 hours in 2800?
    idealised_growth_curve(68.0, 500.0, final_mtbf=130.0,
                           test_time=2800.0).growth_rate  # 0.2268

``curve.phase_mtbf(phase_ends)`` gives the MTBF each test phase should
average, the phase's test time over its expected failures.

The formula: for :math:`t > t_1`,

.. math::
    M(t) = \frac{M_I}{1 - \alpha}\left(\frac{t}{t_1}\right)^{\alpha},

and the expected number of failures by ``t`` is
:math:`(t_1 / M_I)(t / t_1)^{1-\alpha}`: a Crow-AMSAA process with shape
:math:`\beta = 1 - \alpha`, whose instantaneous MTBF is :math:`M(t)`.

Will the test demonstrate the requirement?
------------------------------------------

Reaching an MTBF is not the same as demonstrating it. At the end of the
test a Crow-AMSAA model is fitted to the failure times, and the
requirement is demonstrated if the lower confidence bound on the final
MTBF is at least the requirement. A system that grows exactly along the
plan can still fail that, from bad luck in when its failures fell.

The chance depends on just two numbers: how far the final MTBF is above
the requirement, and how many failures the test is expected to see.

.. code:: python

    curve.demonstration_probability(100.0, alpha_ci=0.2)   # 0.523

So a plan that ends at 131 hours, 31% above a 100-hour requirement, has
only about an even chance of demonstrating it at 80% confidence: the
producer's risk is 0.48. ``curve.oc_curve(100.0, growth_rates)`` gives the
chance had the system grown at other rates (the operating characteristic
of the growth test), and
:func:`~surpyval.test_planning.growth_test_time` finds the test time that
makes the chance what you want:

.. code:: python

    from surpyval.test_planning import growth_test_time

    plan = growth_test_time(48.0, 500.0, 0.3, 100.0,
                            probability=0.8, alpha_ci=0.2)
    plan.test_time    # about 5270 hours
    plan.final_mtbf   # 139.0 hours

The chance is computed exactly: given ``n`` failures, the statistic
:math:`2\beta\sum\ln(T/t_i)` is chi-squared with ``2n`` degrees of freedom,
and Crow's exact lower bound on the MTBF is :math:`n^2 \hat M(T) / z_n`,
so

.. math::
    P = \frac{1}{1 - e^{-\mu}}\sum_{n\ge1}\frac{e^{-\mu}\mu^n}{n!}
        P\left(\chi^2_{2n}\ge\frac{2z_n}{\mu d}\right),

with :math:`\mu` the expected failures and :math:`d` the final MTBF over
the requirement (Broemm, Ellner and Woodworth, 2000). At :math:`d = 1` it
is the consumer's risk, never more than ``alpha_ci``.

*Checked against the handbook.* MIL-HDBK-189C's SPLAN examples (section
5.3) read these chances from printed tables by interpolation. The curves'
final MTBFs and expected failures agree with its tables II and III to
their rounding; the chances agree within 0.035 (0.51 against its 0.48 at
the plan's growth rate, for example). The exact sum agrees with a
simulation of the test itself (fit, bound, compare; 20,000 tests) to
within 0.01. So the handbook's 5375 hours for a chance of 0.8 in example
2 is, exactly, about 5270.

PM2-Continuous
--------------

The idealised curve needs a growth rate, which is hard to know in
advance. PM2 plans from quantities management can set and track instead:

- the **management strategy**, ``MS``: the share of the initial failure
  intensity that is in failure modes the programme will fix (B modes);
  the rest (A modes) it will not;
- the **average fix effectiveness factor**, ``fef``: the share of a
  mode's intensity its fix removes, on average (0.7 to 0.8 is usual);
- the initial MTBF, the goal MTBF and the test time.

Testing finds the B modes, the common ones first. PM2 assumes new modes
are found at a rate :math:`h(t) = \lambda_B / (1 + \beta t)`, and works out
the pace :math:`\beta` that reaches the goal in the test time.

.. code:: python

    from surpyval.test_planning import pm2_continuous

    plan = pm2_continuous(25.0, 90.0, 30000.0,
                          management_strategy=0.95, fef=0.8)
    plan.growth_potential       # 104.2 hours: every B mode found and fixed
    plan.goal_to_potential      # 0.864
    plan.mtbf(15000.0)          # MTBF were every mode seen so far fixed
    plan.expected_modes(30000)  # how many fixes to expect
    plan.fraction_surfaced(30000)

The *growth potential*, :math:`M_I / (1 - MS \cdot fef)`, is the most the
programme can reach; a goal at or above it is refused. The handbook's rule
of thumb is a goal of 0.6 to 0.8 of the potential; this example's 0.86 is
aggressive.

The steps
~~~~~~~~~

A system under test does not improve continuously: fixes go in at
corrective action periods between test phases, and only for modes seen
some time (the *lag*) before. ``plan.targets(phase_ends, lags)`` gives the
MTBF of each phase: the initial MTBF for the first, and after each
period, the idealised MTBF at the period minus its lag.

.. code:: python

    plan.targets([5000.0, 15000.0, 30000.0], lags=2000.0)

Whether the goal will be demonstrated is judged, as in the handbook, by
the operational test after the growth test: the system enters it at the
goal less any degradation from the change of environment, and
``plan.demonstration_probability(65.0, 1000.0, alpha_ci=0.2,
degradation=0.1)`` is the chance a 1000-hour fixed-configuration test then
demonstrates 65 hours at 80% confidence.

The formula: with :math:`\lambda_A = (1 - MS)/M_I` and
:math:`\lambda_B = MS/M_I`,

.. math::
    \rho(t) = \lambda_A + (1 - \mu_d)(\lambda_B - h(t)) + h(t),\qquad
    \beta = \frac{1}{T}\frac{1 - M_I/M_G}{M_I/M_G - M_I/M_{GP}},

and :math:`M(t) = 1/\rho(t)`; the expected modes seen by ``t`` are
:math:`(\lambda_B/\beta)\ln(1 + \beta t)`, and the share of the B-mode
intensity in them :math:`\beta t / (1 + \beta t)`.

*Checked.* MIL-HDBK-189C prints PM2-Continuous only as a figure, so the
plan is checked against its own assumptions instead: many B modes with
gamma-distributed rates, each first seen at an exponential time. Simulated
(20,000 modes, 200 repetitions), the expected modes, the rate of new
modes and the MTBF once the seen modes are fixed agree with PM2's
formulas within 3%, 3% and 1%.

PM2-Discrete
------------

For one-shot items (a missile, an airbag, an engine start) the test is in
trials, and reliability is the chance a trial succeeds.
:func:`~surpyval.test_planning.pm2_discrete` takes the initial and goal
reliability, the trial at which the goal is to be reached, the management
strategy (now the share of :math:`-\ln R_I` in correctable modes) and the
fix effectiveness:

.. code:: python

    from surpyval.test_planning import pm2_discrete

    plan = pm2_discrete(0.8987, 0.9639, 400,
                        management_strategy=0.9, fef=0.8)
    plan.growth_potential                 # 0.9705
    plan.reliability([1, 100, 400])       # 0.8987 ... 0.9639
    plan.expected_modes(400)
    plan.new_mode_probability(400)        # chance of a new mode on trial 400
    plan.targets([100, 250, 402], lags=2)

The idealised reliability on trial ``t`` is
:math:`R(t) = R_A R_B^{1 - \mu_d (t-1)/(n+t-1)}`, with
:math:`R_A = R_I^{1-MS}`, :math:`R_B = R_I^{MS}` and the beta shape
:math:`n` set so that :math:`R(T_L) = R_G`. Checked against the exact
expectations of its assumptions (a million modes with beta-distributed
chances): agreement to 1e-5.

Planned against observed
------------------------

Each plan's ``plot`` draws the idealised curve with its phase steps, the
requirement, and, given a fitted growth model, the instantaneous MTBF the
test is actually showing:

.. code:: python

    import surpyval as surv

    model = surv.CrowAMSAA.fit(times, c=c)   # cumulative test time
    ax = curve.plot(phase_ends=[500, 1500, 2800], model=model,
                    requirement=100)

References
----------

- Department of Defense (2011). *MIL-HDBK-189C, Reliability Growth
  Management*, sections 5.2 to 5.6.
- Broemm, W. J., Ellner, P. M. and Woodworth, W. J. (2000). *AMSAA
  Reliability Growth Guide*. AMSAA TR-652.
- Crow, L. H. (1982). Confidence interval procedures for the Weibull
  process with applications to reliability growth. *Technometrics* 24,
  67-72.
