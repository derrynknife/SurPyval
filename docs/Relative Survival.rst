Relative Survival
=================

When units are followed after some starting point -- patients after a
diagnosis, machines after an overhaul -- many of them have the event for
reasons that have nothing to do with what is being studied: a patient
with a cancer dies of a heart attack, a vehicle is scrapped after an
accident. Often the cause of each event is not recorded, or not
reliably. *Relative survival* deals with this by comparing the units
with the population they come from, whose own hazard is known from a
table of rates.

In plain words
--------------

**Two hazards add up.** Each unit's hazard is split into the hazard it
would have anyway, as a member of its population (read from a *rate
table* for its age, the calendar year and, say, its sex), and an
*excess* hazard due to the condition studied:

.. math::

    \lambda_i(t) = \lambda_{P,i}(t) + \lambda_E(t).

The population part changes as each unit gets older and as the calendar
moves on, so an 80-year-old's background is far higher than a
50-year-old's.

**Net survival** is the survival the units would have if the excess
hazard were the only one: :math:`S_E(t) = \exp(-\Lambda_E(t))`. It is the
standard measure for comparing cancer survival between countries and
periods whose background mortality differs, and it needs no cause of
death: an event of any cause counts, and the rate table takes out what
the population would have had anyway.

**Two estimators.** ``RelativeSurvival.fit`` estimates net survival:

- ``method="pohar-perme"`` (the default): Pohar Perme's estimator
  weights each unit by the inverse of its expected survival, which puts
  back, in expectation, the units the background hazard took. It
  estimates net survival whatever the excess hazard depends on.
- ``method="ederer2"``: Ederer II, the relative survival ratio of the
  units still followed. Where the excess hazard depends on age (it
  usually does), the units still followed late on are the younger ones,
  and Ederer II no longer estimates net survival.

**A regression.** ``ExcessHazard.fit`` models how covariates act on the
excess hazard: :math:`\lambda_i(t) = \lambda_{P,i}(t) + \lambda_0(t)
e^{\beta' Z_i}`. Its coefficients are logs of *excess hazard ratios*,
and its baseline :math:`\lambda_0` is constant within intervals of
follow-up (one by default; ``breaks`` gives more).

**Net survival is not a probability of a set of units.** It is a
difference of hazards. Where fewer units fail than the population rates
expect it rises, and it can exceed 1. SurPyval reports it as it is.

The rate table
--------------

A :class:`~surpyval.RateTable` holds the population's hazard rates by
age band, and optionally by calendar-year band and stratum. Its ages,
years and the follow-up times must all be in one unit, and its rates per
that unit: per person-year with times in years. National life tables
give the probability :math:`q` of dying within a year of age; the rate
for that year is :math:`-\log(1 - q)`.

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    import surpyval as surv

    # A population whose death rates rise about 9% a year of age, the
    # men's 70% above the women's (annual bands, 0 to 110+).
    ages = np.arange(0, 111)
    women = 4e-5 * np.exp(0.092 * ages)
    table = surv.RateTable({"F": women, "M": 1.7 * women}, ages=ages)
    print(table)
    # The chance that a woman aged 70 lives another ten years
    print(table.sf(10, age=70, stratum="F").round(3))

A table by calendar year as well takes a 2-D array of rates per stratum,
one column per year band, and ``years=`` the bands' lower edges; each
unit then also needs its calendar time at the start of follow-up
(``year=2011.5``). ``RateTable.from_df`` reads a long table (one row per
age, year and stratum) from a DataFrame.

Estimating net survival
-----------------------

The units below are simulated: 2,000 people aged 50 to 85, whose excess
hazard is 0.1 a year at 65 and rises 5% a year of age, each also subject
to the population rates above, followed for up to 10 years.

.. jupyter-execute::

    rng = np.random.default_rng(1)
    size = 2000
    age = rng.uniform(50, 85, size)
    sex = rng.choice(["F", "M"], size)
    excess_rate = 0.1 * np.exp(0.05 * (age - 65))
    excess = rng.exponential(1 / excess_rate)
    # Each unit's time to death from the population's hazard: its
    # expected cumulative hazard on a grid, inverted at an exponential draw
    grid = np.linspace(0, 60, 1201)
    H = table.Hf(grid[None, :], age[:, None], stratum=sex[:, None])
    reached = (H < rng.exponential(size=size)[:, None]).sum(axis=1)
    population = grid[np.minimum(reached, grid.size - 1)]
    years = np.minimum(np.minimum(excess, population), 10.0)
    censored = (years == 10.0).astype(int)

    pohar_perme = surv.RelativeSurvival.fit(
        years, censored, rate_table=table, age=age, stratum=sex
    )
    ederer2 = surv.RelativeSurvival.fit(
        years, censored, rate_table=table, age=age, stratum=sex,
        method="ederer2",
    )
    truth = [np.mean(np.exp(-excess_rate * t)) for t in (5, 10)]
    print("true net survival      ", np.round(truth, 3))
    print("Pohar Perme            ", pohar_perme.sf([5, 10]).round(3))
    print("Ederer II              ", ederer2.sf([5, 10]).round(3))
    # 90% two-sided pointwise bounds on Pohar Perme's net survival at 10
    print("Pohar Perme, 90% bounds", pohar_perme.cb(10, alpha_ci=0.1).round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _lo, _hi = pohar_perme.cb(10, alpha_ci=0.1)
    assert _lo < truth[1] < _hi, (_lo, truth[1], _hi)
    assert ederer2.sf(10) > pohar_perme.sf(10), (ederer2.sf(10), pohar_perme.sf(10))
    assert ederer2.sf(10) > truth[1], ederer2.sf(10)

Pohar Perme's estimate at 10 years is close to the truth, inside its
90% bounds. Ederer II's is higher than both: the older units, whose
excess hazard is higher, have left the risk set faster than the younger,
and the relative survival ratio of those still followed describes the
younger ones.

``model.table`` gives, at each follow-up time, the number at risk and of
events, the increments of the observed and expected cumulative hazards,
the cumulative excess hazard ``Hf``, its standard error and net survival.
``cb`` gives pointwise bounds on ``sf``, ``ff`` or ``Hf``, on the log
scale (``bound_type="log"``, the default, as R's relsurv) or the plain
one (``"normal"``); neither is cut at 1.

.. jupyter-execute::

    print(pohar_perme.table.head())

Excess hazard regression
------------------------

``ExcessHazard`` fits the excess hazard as :math:`\lambda_0(t)
e^{\beta' Z}` beside each unit's population hazard, by maximum
likelihood (Estève et al., 1990). Here a treatment halves the excess
hazard, which is otherwise constant at 0.15 a year:

.. jupyter-execute::

    treated = rng.binomial(1, 0.5, size)
    excess = rng.exponential(1 / (0.15 * np.exp(-0.7 * treated)))
    years = np.minimum(np.minimum(excess, population), 8.0)
    df = pd.DataFrame({
        "years": years,
        "censored": (years == 8.0).astype(int),
        "treated": treated,
        "age": age,
        "sex": sex,
    })
    model = surv.ExcessHazard.fit_from_df(
        df, "years", ["treated"], "age", table, c_col="censored",
        stratum_col="sex", breaks=3,
    )
    print(model)
    columns = ["coef", "exp(coef)", "coef lower 90%", "coef upper 90%", "p"]
    print(model.summary(alpha_ci=0.1)[columns].round(3))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _lo, _hi = model.param_cb("treated", alpha_ci=0.1)
    assert _lo < -0.7 < _hi, (_lo, _hi)
    assert 0.4 < np.exp(model.coef[0]) < 0.6, model.coef
    assert model.maximum == "verified", model.maximum

The treatment's excess hazard ratio, ``exp(coef)``, is about a half, and
its 90% interval holds the true coefficient, -0.7. ``breaks=3`` gives the
baseline three intervals with equal numbers of events (an array of times
gives them directly); every interval needs events. A covariate that is
constant, or a combination of the others, is *aliased* (its coefficient
``nan``, with one warning), and a fit whose likelihood has no finite
maximum -- an interval with no more events than the population rates
expect, or a group with no events at all -- says so.

The model's ``sf``, ``ff``, ``hf``, ``Hf``, ``df`` and ``qf`` are the net
(excess-only) functions of a unit with covariates ``Z``; ``cb`` bounds
them by the delta method on the log of the cumulative excess hazard, and
``observed_sf`` is a unit's all-cause survival, its expected survival
from the rate table times its net survival:

.. jupyter-execute::

    print("net survival at 5, treated or not:",
          model.sf(5, [[1], [0]]).round(3))
    print("all-cause survival at 5 of a treated woman aged 70:",
          model.observed_sf(5, [1], age=70, stratum="F").round(3))

The maths
---------

With :math:`Y_i(t)` unit :math:`i`'s at-risk indicator,
:math:`\Lambda_{P,i}` its expected cumulative hazard along its path
through the rate table and :math:`u_k` the distinct follow-up times:

- Ederer II: :math:`\hat\Lambda_E(t) = \sum_{u_k \le t} d_k / Y(u_k) -
  \int_0^t \sum_i Y_i(s)\lambda_{P,i}(s) / Y(s)\, ds`, with variance
  :math:`\sum_{u_k \le t} d_k / Y(u_k)^2`;
- Pohar Perme: the same with each unit weighted by
  :math:`w_i(t) = 1/S_{P,i}(t)`, :math:`Y^w = \sum_i Y_i w_i`; the
  expected part over a stretch with the same units at risk is the
  increase of :math:`\log Y^w` there (as :math:`dw_i/ds =
  w_i\lambda_{P,i}`), and the variance :math:`\sum_{\text{events}} w_i^2
  / Y^w(u_k)^2`.

The expected parts are integrated exactly between the follow-up times;
between two of them a prediction takes the expected part as linear in
time. The excess hazard model's log-likelihood is

.. math::

    \ell = \sum_i n_i \Big[ \delta_i \log\big(\lambda_{P,i}(x_i) +
    \lambda_0(x_i) e^{\beta' Z_i}\big) - e^{\beta' Z_i}\Lambda_0(x_i)
    - \Lambda_{P,i}(x_i) \Big].

What is not built yet: delayed entry (left truncation) and interval
censoring, which the fits refuse; a spline or other flexible baseline
excess hazard; and covariate effects on the excess hazard that change
with time.

References
----------

Ederer, F. and Heise, H. (1959). Instructions to IBM 650 programmers in
processing survival computations. Methodological note 10, End Results
Evaluation Section, National Cancer Institute.

Estève, J., Benhamou, E., Croasdale, M. and Raymond, L. (1990).
Relative survival and the estimation of net survival: elements for
further discussion. *Statistics in Medicine* 9, 529-538.

Pohar Perme, M., Stare, J. and Estève, J. (2012). On estimation in
relative survival. *Biometrics* 68, 113-120.

Pohar, M. and Stare, J. (2006). Relative survival analysis in R.
*Computer Methods and Programs in Biomedicine* 81, 272-278.
