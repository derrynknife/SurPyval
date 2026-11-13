Equivalent Time Scales
======================

Many things age on more than one clock at once. A gas turbine runs for
hours, and it is also started and stopped; every start heats and cools
the hot parts, and wears them as many hours of steady running would. A
pump runs for hours and is cycled; an aircraft flies hours and makes
landings; a vehicle covers kilometres and runs engine hours. Which clock
should a life be measured on?

Often neither alone. Two machines with the same running hours are not
equally worn if one was started ten times as often. An **equivalent time
scale** combines the clocks into one age,

.. math::
    T = h + k\,s ,

with :math:`h` the hours, :math:`s` the starts and :math:`k` what one
start is worth in hours. Gas-turbine makers call :math:`T` the
*equivalent operating hours* and quote a fixed :math:`k` (one start
counts as some number of hours). ``Equivalence(Weibull)`` **learns**
:math:`k` from your failure data, together with the life distribution on
:math:`T`, so you can tell how much a start really costs your fleet, with
a confidence interval, and test whether starts matter at all. The
statistics literature calls the best such combination the *ideal time
scale* (Kordonsky and Gertsbakh, 1993; Duchesne and Lawless, 2000).

The API reference is :doc:`univariate/equivalence`.

Where the information about k comes from
----------------------------------------

:math:`k` is learned by comparing units that use the two clocks at
**different rates**. If a peaking unit (many starts per hour) fails after
fewer hours than a base-load unit (few starts per hour), the difference
is what the starts cost. If every unit made the same number of starts per
hour, there would be nothing to compare: :math:`h + k s` would be the
hours times the same factor for every unit, and that factor cannot be
told apart from the life distribution's scale. The fit refuses such data
and says so.

A worked example: starts on a fleet of turbines
-----------------------------------------------

240 units run in three duties: base load (about one start per 100
hours), cycling (one per 20 hours) and peaking (about one start every
three hours). Each has been followed for between 5,000 and 25,000 hours;
40% have had the failure being studied, the rest are still running. The
data are simulated, so the truth is known: one start is worth 20 hours,
and the life on the combined clock is Weibull with scale 30,000
equivalent hours and shape 2.5.

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    import surpyval as sp

    rng = np.random.default_rng(42)
    n = 240
    duty = rng.choice(["base load", "cycling", "peaking"], n, p=[0.4, 0.35, 0.25])
    rate = pd.Series(duty).map(
        {"base load": 0.01, "cycling": 0.05, "peaking": 0.3}
    ).to_numpy()
    life_in_hours = 30000 * rng.weibull(2.5, n) / (1 + 20 * rate)
    followed = rng.uniform(5000, 25000, n)
    failed = life_in_hours < followed
    hours = np.where(failed, life_in_hours, followed)
    df = pd.DataFrame(
        {
            "hours": hours.round(0),
            "starts": np.round(rate * hours),
            "c": np.where(failed, 0, 1),
            "duty": duty,
        }
    )
    df.head()

Each row is one unit: its hours and starts at its failure (``c = 0``) or
now, if it is still running (``c = 1``). Fit the combined clock. The
first column of ``x_cols`` is the clock :math:`k` is measured in.

.. jupyter-execute::

    model = sp.WeibullEquivalence.fit_from_df(
        df, x_cols=["hours", "starts"], c_col="c"
    )
    model

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert model.maximum == "verified"
    assert round(model.k, 1) == 18.4, model.k
    _lo, _hi = model.param_cb("k")
    assert (round(_lo, 1), round(_hi, 1)) == (14.2, 23.7), (_lo, _hi)
    assert model.k_lr_test()["p_value"] < 1e-50
    assert round(model.params[1], 2) == 2.63, model.params

How to read it:

- **k = 18.4**: one start is worth about 18 hours of running (the truth
  is 20). The 95% confidence interval, 14.2 to 23.7 hours, is the range
  of values the data are consistent with. The interval's level is the
  library's ``alpha_ci`` (0.05 by default); ``model.reading(0.1)`` gives
  the 90% one.
- **Test of k = 0**: the likelihood-ratio test that starts do not matter.
  Here it is overwhelming (p about 1e-55). A large p-value would mean the
  data cannot tell starts from no effect, and hours alone may do.
- ``alpha`` and ``beta`` are the Weibull scale (in equivalent hours) and
  shape on the combined clock.

The combined age of a unit with 10,000 hours and 400 starts:

.. jupyter-execute::

    model.equivalent(10000, 400)

Fitted to hours alone, the same data give a shape of about 1.5, not 2.6:
mixing units that wear at different speeds blurs the wear-out, so the
hours-only model makes failures look more random than they are.

.. jupyter-execute::

    sp.Weibull.fit(df["hours"], df["c"]).params

Lives along a duty
------------------

The questions that matter are usually about a duty: how long will a
peaking unit last? A unit making ``rate`` starts per hour ages
:math:`1 + k \cdot rate` equivalent hours per hour, so its life in hours
is the combined life divided by that. The ``*_at_usage`` methods answer
in hours, with bounds that include the uncertainty in :math:`k`. The
B10 life (the hours by which 10% have failed) of each duty:

.. jupyter-execute::

    rates = np.array([0.01, 0.05, 0.3])
    pd.DataFrame(
        {
            "starts per hour": rates,
            "B10 (hours)": model.qf_at_usage(0.1, rates).round(0),
            "lower": model.quantile_cb_at_usage(0.1, rates)[:, 0].round(0),
            "upper": model.quantile_cb_at_usage(0.1, rates)[:, 1].round(0),
        }
    )

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _b10 = model.qf_at_usage(0.1, rates)
    assert np.allclose(_b10.round(-2), [10200, 6300, 1900]), _b10

A peaking unit reaches its B10 life after about 1,900 hours, a base-load
unit after about 10,200. The starts made in that time are ``rate`` times
the hours. And the chance a unit survives 8,000 hours in each duty:

.. jupyter-execute::

    model.sf_at_usage(8000, rates).round(3), model.cb_at_usage(8000, rates).round(3)

The usual functions (``sf``, ``ff``, ``hf``, ``qf``, ``mean`` and ``cb``)
are those of the distribution on the combined clock, in equivalent hours.

The life at one duty, as a plain life model
-------------------------------------------

``model.at_usage(rate)`` binds one rate and gives the life of a unit used
at it as an ordinary life model in hours: its ``sf``, ``ff``, ``df``,
``hf``, ``Hf``, ``qf``, ``random``, ``mean``, ``rmst``, ``cb`` and
``quantile_cb`` take hours only, each the model's ``*_at_usage`` at that
rate. ``rate`` is in the model's own units: the second clock's units per
unit of the first, here starts per hour. A unit at that rate ages
``1 + k rate`` equivalent hours per hour (``life.acceleration``). Hand it
to code that expects a univariate model, such as a block of a reliability
block diagram ("this pump, peaking"). The model's unit label goes with
it, as typed.

.. jupyter-execute::

    peaking = model.at_usage(0.3)
    peaking, peaking.qf(0.1).round(0), peaking.sf(8000).round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert peaking.qf(0.1) == model.qf_at_usage(0.1, 0.3)
    assert peaking.sf(8000) == model.sf_at_usage(8000, 0.3)

Its draws are ``qf(U)`` for uniform ``U``, so a seed gives common random
numbers across models, and its bounds carry the uncertainty in
:math:`k`:

.. jupyter-execute::

    peaking.random(3, random_state=1).round(0), peaking.rmst(2000)

It is a view on the model: save the model (``model.to_dict()``) and the
rate.

What the model assumes, and its limits
--------------------------------------

- **Constant usage per unit.** Between installation and the time it is
  seen, each unit is assumed to have used its clocks at a steady ratio
  (its starts divided by its hours). A unit that ran base load for years
  and then switched to peaking has the right totals, but the model
  places its starts evenly over its life. If duties change a lot over a
  unit's life, the estimate of :math:`k` is an average.
- **The two clocks add.** Wear from a start is the same whatever the
  unit's age, and does not interact with the running hours.
- **The clocks' order.** :math:`k` is measured in units of the first
  clock. ``fit(starts, hours)`` instead gives hours measured in starts,
  :math:`k' = 1/k`, with the same fit; choose the order that reads best
  ("one start is worth k hours").
- **k on its edge.** :math:`k` cannot be negative. Where the best fit is
  :math:`k = 0` (starts do not matter), the fit is exactly the
  distribution fitted to hours alone, the Wald interval on :math:`k` does
  not exist (it is ``nan``, with a warning), and
  ``param_cb("k", method="lr")``, the profile-likelihood interval,
  gives one that starts at 0. The test of :math:`k = 0` allows for the
  edge: its p-value is half the usual chi-square one (Self and Liang,
  1987).
- **Units with no hours.** A unit that failed with starts but no running
  hours is fine: its age is :math:`k s`. Such a failure is impossible
  if starts did not matter, so it rules out :math:`k = 0`.
- **Data.** Failures (``c = 0``), units still running (``c = 1``),
  units that failed before they were inspected (``c = -1``), counts
  ``n`` and late entry (``tl``, a pair of hours and starts at entry per
  unit) are taken. Intervals (``c = 2``) are not: an interval on two
  clocks needs both ends on each.

Why the factor (1 + k rate) matters
-----------------------------------

It is tempting to compute :math:`T = h + k s` for each unit and fit a
Weibull to the :math:`T` values, choosing the :math:`k` that fits best.
That is wrong. A larger :math:`k` stretches every :math:`T`, and a
density on a stretched clock is lower by the stretch, so that likelihood
falls as :math:`k` grows for a reason that has nothing to do with the
data; its best :math:`k` is pulled towards 0. The correct likelihood is
the one on the hours clock each unit was actually observed on. A unit
running at ``rate`` starts per hour ages :math:`1 + k \cdot rate` times
faster than one with no starts, so in hours its failure density is

.. math::
    f(h_i + k s_i;\ \theta)\,(1 + k \rho_i), \qquad
    \rho_i = s_i / h_i ,

and a unit still running contributes :math:`S(h_i + k s_i; \theta)`. The
factor :math:`1 + k\rho_i` corrects the stretch. This is an accelerated
failure time model on hours, with acceleration :math:`1 + k\rho_i`
(Duchesne and Lawless, 2000), and SurPyval's tests check that at a fixed
:math:`k` it is exactly ``WeibullAFT`` with the covariate
:math:`\log(1 + k\rho_i)` and its coefficient fixed at 1. In simulations
with :math:`k = 10`, the naive fit's estimate collapsed towards 0 while
this one's was unbiased.

The *minimum coefficient of variation* method of Kordonsky and Gertsbakh
(choose :math:`k` to make the failures' :math:`T` as alike as possible)
is not used: it has no likelihood, so it cannot take units still
running, and gives no intervals or tests.

References
----------

Duchesne, T. and Lawless, J. (2000). Alternative time scales and failure
time models. *Lifetime Data Analysis* 6, 157-179.

Kordonsky, K. B. and Gertsbakh, I. B. (1993). Choice of the best time
scale for system reliability analysis. *European Journal of Operational
Research* 65, 235-246.

Self, S. G. and Liang, K.-Y. (1987). Asymptotic properties of maximum
likelihood estimators and likelihood ratio tests under nonstandard
conditions. *Journal of the American Statistical Association* 82,
605-610.
