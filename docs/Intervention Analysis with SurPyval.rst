Intervention Analysis with SurPyval
===================================

A failure mode keeps coming back, so something is changed: a redesigned
seal, a new lubricant, a revised procedure, a corrective action. Months
later the question is whether it worked:

- **Did the failure rate fall?** And by how much, with what uncertainty?
- **Is the evidence enough to say?** If no change is seen, was there
  enough operating time since the change to have seen one? If not, how
  much more is needed?

:func:`surpyval.recurrent.intervention_test` answers both from the failure
records of one repairable system or a fleet, each with its own observation
window and its own date of change. It compares the rate of failures after
the intervention with the rate before it, and reports the *ratio* of the
rates (after over before: 0.5 means the rate halved), its confidence
interval, the p-value of no change, and the evidence: the power the
exposure had to detect the change hoped for, the smallest change it could
detect, and the exposure that would be needed. The functions behind the
evidence are in :mod:`surpyval.test_planning`, for planning before the
data are in. The models of the rate (homogeneous and power-law Poisson
processes) are explained in :doc:`Recurrent Event Analysis`.

One system
----------

A pump failed eight times in its first 50 weeks of service. Its seals were
changed at week 50, and it failed three times in the next 50 weeks. The
failure times are the pump's age at each failure, as for every recurrent
model in SurPyval; ``T`` is the end of its observation (today), and
``intervention`` the time of the change:

.. jupyter-execute::

    import numpy as np
    from surpyval.recurrent import intervention_test

    x = [3, 8, 15, 19, 26, 30, 37, 44, 52, 77, 95]
    result = intervention_test(x, T=100, intervention=50)
    print(result)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert (result.events_before, result.events_after) == (8, 3)
    assert abs(result.ratio - 0.375) < 1e-9
    assert abs(result.p_value - 0.2266) < 1e-4
    assert np.allclose(result.ratio_cb, [0.0641, 1.562], atol=5e-4)
    assert result.verdict == "inconclusive"
    assert abs(result.power - 0.134) < 1e-3
    assert np.isinf(result.exposure_needed)

The rate after the change is estimated at 0.375 times the rate before it
(3 failures in 50 weeks against 8 in 50), a 62% reduction. But the 95%
confidence interval of the ratio runs from 0.064 to 1.56: the data are as
consistent with no change as with a large one, and the p-value of no change
is 0.23. The verdict is not "no change": it is *inconclusive*, because the
test only had a 13% chance of detecting even a halving of the rate. The
notes say why, and it is worth knowing: with only eight failures before the
change, the earlier rate itself is too uncertain for *any* amount of
operating time after it to show a halving reliably. Only a longer history
before the change, or pooling with other pumps, would.

The ``verdict`` is one of:

- ``"decrease"`` or ``"increase"``: the test found a change
  (``p_value < alpha_ci``);
- ``"no change detected"``: it found none, and it had at least
  ``power_required`` (80%) power to detect the change ``target_ratio``
  (0.5, a halving) -- so the change most likely did not achieve that;
- ``"inconclusive"``: it found none, but the evidence was too little to
  tell;
- ``"untestable"``: the data cannot answer (no exposure after the change
  yet, no failures at all, ...), with a note saying why.

A test of "did it work?" is one-sided: ``alternative="decrease"`` asks only
whether the rate fell, and its interval is a one-sided upper bound on the
ratio, "the rate is now at most this fraction of what it was":

.. jupyter-execute::

    one_sided = intervention_test(x, T=100, intervention=50,
                                  alternative="decrease")
    print(round(one_sided.p_value, 3), one_sided.ratio_cb.round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert abs(one_sided.p_value - 0.1133) < 1e-4
    assert one_sided.ratio_cb[0] == 0
    assert abs(one_sided.ratio_cb[1] - 1.296) < 1e-3

How the test works
~~~~~~~~~~~~~~~~~~

With ``model="hpp"`` (the default) the rate is taken as constant before the
change and constant after it. Let :math:`E_b` and :math:`E_a` be the
operating time (the *exposure*) before and after, :math:`N_b` and
:math:`N_a` the failures, and :math:`\rho = \lambda_a / \lambda_b` the
ratio of the rates. Given the :math:`N = N_b + N_a` failures in all, the
number after the change is binomial,

.. math::

    N_a \mid N \sim \text{Bin}\!\left(N,\ \pi(\rho) =
    \frac{\rho E_a}{E_b + \rho E_a}\right),

whatever the rates are (Cox and Lewis 1966, section 9.4). So no change,
:math:`\rho = 1`, is tested by the binomial tail at :math:`\pi(1) = E_a /
(E_b + E_a)`, which is exact for any number of failures, however few. The
confidence interval inverts the test: it is the Clopper-Pearson interval of
:math:`\pi`, mapped to :math:`\rho = (\pi / (1 - \pi)) (E_b / E_a)`. Here
:math:`N = 11` and :math:`\pi(1) = 0.5`:

.. jupyter-execute::

    from scipy import stats

    lower_tail = stats.binom.cdf(3, 11, 0.5)
    print(round(2 * lower_tail, 4), round(one_sided.p_value, 4))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert abs(2 * lower_tail - result.p_value) < 1e-12
    assert abs(lower_tail - one_sided.p_value) < 1e-12

An exact test of a count is conservative: its real level is below the 5%
asked for (2.9% for one system expecting 10 failures before the change and
10 after), and its interval covers more often than 95%. ``method="mid-p"`` gives the mid-p test and interval
(Lancaster 1961), which count half the probability of the observed count:
their level and coverage are close to 5% and 95% on average, at the price
of no longer being guaranteed. The calibration studies in SurPyval's test
suite check both, by simulation against the exact sums.

Is the evidence enough?
-----------------------

The evidence is judged against a change worth detecting, ``target_ratio``
(a halving unless given), with the power wanted, ``power`` (80% unless
given). The result carries:

- ``power``: the chance that a test on this exposure would detect
  ``target_ratio``, taking the rate before the change as the rate of its
  failures;
- ``detectable_ratio``: the ratio nearest 1 that this exposure detects
  with 80% power (the *minimum detectable improvement* is one minus it);
- ``exposure_needed``: the exposure after the change, in all, that would
  detect ``target_ratio`` with 80% power, ``additional_exposure`` how much
  more that is, and ``events_needed`` the failures to expect by then at
  the improved rate;
- ``zero_failure_exposure``: the exposure after the change at which, with
  no further failure at all, a decrease would be shown.

A fleet of four pumps had the new seals fitted at different times, and the
records were pulled 800 hours after each change:

.. jupyter-execute::

    failures = {
        "A": [682, 1112, 1330, 1351, 1664, 1932, 2468, 2780],
        "B": [184, 648, 817, 1368, 1489, 1556, 1960, 2869, 3128, 3184,
              3269, 3751, 4064, 4077, 4835, 5532],
        "C": [721, 969, 2012, 2081, 3300],
        "D": [412, 503, 1696, 2035, 2406, 2754, 3705, 4330, 5085],
    }
    x = np.concatenate([failures[q] for q in failures])
    i = np.concatenate([[q] * len(failures[q]) for q in failures])
    change = {"A": 3000, "B": 4200, "C": 2500, "D": 3600}

    early = {q: change[q] + 800 for q in change}
    seen = np.array([t <= early[q] for t, q in zip(x, i)])
    first_look = intervention_test(x[seen], i[seen], T=early,
                                   intervention=change)
    print(first_look)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert first_look.verdict == "inconclusive"
    assert (first_look.events_before, first_look.events_after) == (32, 4)
    assert first_look.exposure_after == 3200
    assert abs(first_look.power - 0.186) < 1e-3
    assert 41000 < first_look.exposure_needed < 42000

Four failures in the 3200 pump-hours since the change, against 32 in the
13300 before it: the rate looks lower, but 3200 hours could only have
detected a halving 19% of the time. About 41000 hours after the change
would give 80% power. The full records, to the end of the study:

.. jupyter-execute::

    end = {"A": 7000, "B": 7000, "C": 6500, "D": 7000}
    final = intervention_test(x, i, T=end, intervention=change)
    print(final)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert final.verdict == "decrease"
    assert (final.events_before, final.events_after) == (32, 6)
    assert final.p_value < 1e-4
    assert np.allclose(final.reduction_cb, [0.530, 0.935], atol=1e-3)
    assert abs(final.power - 0.607) < 1e-3

The rate fell to about a fifth of its earlier value. The interval of the
reduction, ``final.reduction_cb`` (53% to 93%), says the seals removed at
least half of the failures. ``final.power`` (61%) is below 80%: the
exposure was still short of what detecting a mere *halving* needs, but the
reduction was larger than a halving, and the test found it. The note about
power matters most when nothing is found.

The same calculations plan a study before the data are in.
:func:`~surpyval.test_planning.rate_ratio_power`,
:func:`~surpyval.test_planning.rate_ratio_detectable` and
:func:`~surpyval.test_planning.rate_ratio_exposure` take the rate before
the change and the exposures. With 30 failures in 10000 hours before a
change, 5000 hours after it would detect a halving about a third of the
time, and 41000 hours would be needed for 80%:

.. jupyter-execute::

    from surpyval.test_planning import (
        rate_ratio_detectable,
        rate_ratio_exposure,
        rate_ratio_power,
    )

    rate = 30 / 10000
    print(round(rate_ratio_power(0.5, rate, 10000, 5000), 3))
    print(round(rate_ratio_detectable(rate, 10000, 5000), 3))
    print(round(rate_ratio_exposure(0.5, rate, 10000)))
    print(rate_ratio_exposure(0.5, 12 / 10000, 10000))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert abs(rate_ratio_power(0.5, rate, 10000, 5000) - 0.356) < 1e-3
    assert abs(rate_ratio_detectable(rate, 10000, 5000) - 0.263) < 1e-3
    assert 41000 < rate_ratio_exposure(0.5, rate, 10000) < 41400
    assert np.isinf(rate_ratio_exposure(0.5, 12 / 10000, 10000))
    assert np.isinf(rate_ratio_exposure(0.5, 23 / 10000, 10000))
    assert np.isfinite(rate_ratio_exposure(0.5, 24 / 10000, 10000))

The last line is the limit seen for the single pump: however long the
pumps run after the change, the rate after it becomes known exactly, but
the rate before it stays as uncertain as its 12 failures make it. Twelve
failures cannot show a halving with 80% power; the exact test needs about
24 (the normal approximation's 16 is far too few). The
power is the exact sum over the possible counts of failures, not a normal
approximation, which is poor for the handful of failures a corrective
action is usually judged on.

Fleets
------

Systems differ: some fail far more often than others. With
``baseline="per-system"`` (the default) each system keeps its own rate and
only the ratio is shared, and the test is the exact stratified test: the
failures after the change, summed over the systems, whose exact
distribution given each system's total is the sum of the systems'
binomials (Breslow and Day 1987, chapter 3). ``baseline="common"`` pools
the exposures and the failures of every system, as if they shared one rate.

Pooling can mislead badly when systems with different rates have different
shares of their exposure after the change. Here system ``a`` fails often
and was changed late, so almost all its exposure, and its failures, are
before the change; system ``b`` fails rarely and was changed early. Neither
rate changed at all:

.. jupyter-execute::

    a = np.linspace(0, 100, 42)[1:-1]            # 40 failures to 100
    b = np.array([40.0, 60.0, 100.0, 130.0, 160.0, 190.0])
    x2 = np.concatenate([a, b])
    i2 = ["a"] * a.size + ["b"] * b.size
    ends = {"a": 100.0, "b": 200.0}
    changed = {"a": 80.0, "b": 20.0}

    pooled = intervention_test(x2, i2, T=ends, intervention=changed,
                               baseline="common")
    per_system = intervention_test(x2, i2, T=ends, intervention=changed)
    print("pooled    :", pooled.verdict, round(pooled.ratio, 3))
    print("per system:", per_system.verdict, round(per_system.ratio, 3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert pooled.verdict == "decrease" and 0.2 < pooled.ratio < 0.25
    assert per_system.change == "none"

Pooled, the period before the change is mostly the busy system's and the
period after mostly the quiet one's, so the rate seems to fall by nearly
80%; each
system compared with itself shows no change. In the calibration study of a
fleet like this, the pooled test finds a "decrease" 73% of the time when
nothing changed, and the per-system test 4% of the time. Use ``"common"``
only when the systems are known to share one rate (identical units in the
same duty); it is then more powerful, and it uses systems seen on one side
of the change only, which a per-system comparison cannot.

Each system's change time is given in ``intervention`` as a dict (or an
array in the sorted order of the system ids); ``inf`` marks a system never
changed, and a time at or before a system's start one that had the change
from new. A dict ``T`` may name systems with no failure at all, so that
their exposure counts. The observation can also be given in the fitters'
form, a ``c = 1`` row ending each system (``c=``), with a later start of
observation (``tl=``, delayed entry), or as separate windows of
observation (``windows=``). The time since a system's last failure is
evidence about the change, so the end of observation is always required.

When the rate trends
--------------------

A constant rate on each side of the change is often not true: a system
wearing out fails more often as it ages, and a comparison of average rates
then takes the ageing for an effect of the change (a worsening after it, or
a hidden improvement). The HPP result checks for this: ``trend_p_value`` is
the Laplace test of a trend within the periods before and after the change,
and a note recommends the power law when it is small.

``model="power-law"`` lets the rate trend with age as the Crow-AMSAA power
law, :math:`u(t) = \lambda \beta t^{\beta - 1}`, and the change multiplies
it by :math:`\rho`, keeping its shape. The ratio is tested by the
likelihood ratio, with the scale (each system's, per system) and the shape
profiled out, and its interval is the profile-likelihood interval. These
pumps wear out (shape 3), and nothing changed at week 50:

.. jupyter-execute::

    rng = np.random.default_rng(3)
    wearing = np.sort(100 * rng.uniform(size=60) ** (1 / 3))
    hpp = intervention_test(wearing, T=100, intervention=50)
    power_law = intervention_test(wearing, T=100, intervention=50,
                                  model="power-law")
    print("HPP      :", hpp.verdict, round(hpp.ratio, 2),
          "- trend p-value", round(hpp.trend_p_value, 4))
    print("power law:", power_law.verdict, round(power_law.ratio, 2),
          "- shape", round(power_law.shape, 2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert hpp.verdict == "increase" and hpp.trend_p_value < 0.05
    assert power_law.change == "none"

The HPP comparison finds a large "increase", and flags the trend; the power
law finds the shape (about 3) and no change at the intervention. In the
calibration study of five such systems, the HPP comparison finds a change
54% of the time when there is none, and the likelihood-ratio test 5%. The
power law's evidence is judged in the scale of its cumulative intensity,
where the process has a constant rate, taking the shape as known; it is
therefore a little optimistic, and its ``exposure_needed`` (in that scale,
not in hours) is left out: ``events_needed`` gives the failures to expect
after the change instead.

Saving the result
-----------------

``to_dict`` gives the result as strict JSON (an infinite bound or a missing
value is written as ``null`` and recorded under ``"non_finite"``, as for
every SurPyval model), and ``InterventionTestResult.from_dict`` restores
it:

.. jupyter-execute::

    import json
    from surpyval.recurrent import InterventionTestResult

    text = json.dumps(final.to_dict())
    again = InterventionTestResult.from_dict(json.loads(text))
    print(again.verdict, again.ratio_cb.round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert again.verdict == final.verdict
    assert np.array_equal(again.ratio_cb, final.ratio_cb)

What the test cannot tell
-------------------------

The comparison is of the periods before and after the change, so anything
else that changed between them is attributed to the intervention:

- **Regression to the mean.** A change is usually made *because* of a run
  of failures, and a run of bad luck tends to be followed by fewer
  failures whether or not anything is done. Count the run that prompted
  the change as "before", fix the intervention time from the records of
  the work, not from the failure data, and judge the period after it on
  its own exposure.
- **Other changes.** A new duty cycle, a season, a different operator or
  a change in how failures are recorded at the same time is
  indistinguishable from the intervention. Systems never changed (``inf``)
  observed over the same calendar time show whether the rate moved for
  everyone; they enter the per-system comparison only through the
  systems that were changed, so compare them separately.
- **The failure mode.** Test the failures the change was meant to remove:
  a change that halves one mode among several can be lost among the
  others.
- **Exposure.** The exposure is in the units of ``x``. Operating hours
  are better than calendar time when the systems' use varies.

References
----------

Breslow, N. E. and Day, N. E. (1987), *Statistical Methods in Cancer
Research, Volume II: The Design and Analysis of Cohort Studies*, IARC,
chapter 3.

Cook, R. J. and Lawless, J. F. (2007), *The Statistical Analysis of
Recurrent Events*, Springer.

Cox, D. R. and Lewis, P. A. W. (1966), *The Statistical Analysis of Series
of Events*, Methuen, section 9.4.

Gu, K., Ng, H. K. T., Tang, M. L. and Schucany, W. R. (2008), "Testing the
ratio of two Poisson rates", *Biometrical Journal* 50, 283-298.

Lancaster, H. O. (1961), "Significance tests in discrete distributions",
*Journal of the American Statistical Association* 56, 223-234.

Przyborowski, J. and Wilenski, H. (1940), "Homogeneity of results in
testing samples from Poisson series", *Biometrika* 31, 313-323.

Rigdon, S. E. and Basu, A. P. (2000), *Statistical Methods for the
Reliability of Repairable Systems*, Wiley.
