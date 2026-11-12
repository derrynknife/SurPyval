Zero-Inflated Regression
========================

Some units fail the moment they are used. They are *dead on arrival*:

- a power supply that does not start when it is commissioned;
- a part found broken when its box is opened;
- a seedling that never germinates, a device rejected at its first
  check, a transplant that fails on the table.

In the data these are failures at time **zero**. An ordinary lifetime
distribution cannot place a share of its failures at exactly 0, so a
fit either refuses them or bends its whole shape to accommodate them,
and gets the life of every other unit wrong.

A zero-inflated model treats the population as two groups. A unit is
dead on arrival with some probability, :math:`f_0`; a unit that works
has a time to failure with an ordinary lifetime distribution. A
zero-inflated *regression* lets covariates act on both parts, and so
answers two different questions at once:

- **Which units arrive dead?** Which suppliers, routes or batches have a
  higher chance of failing at time 0.
- **How long do the rest last?** Among the units that work, what makes
  failure come sooner.

A covariate can matter for one and not the other. Rough shipping may
break more units in transit without shortening the life of those that
survive it; a cheaper supplier may make units that both arrive dead more
often and wear out sooner. An ordinary regression, with one effect per
covariate, cannot tell these apart.

Without covariates this is the zero-inflated model that the
distributions already fit (``Weibull.fit(x, c, zi=True)``, see
:doc:`Parametric SurPyval Modelling`). ``ZeroInflated`` is that model
with covariates. It is the mirror image of the
:doc:`mixture-cure regression <Mixture-Cure Regression>`, whose extra
group never fails rather than failing at once. The API reference is
:doc:`regression/zero_inflated`.

An example: power supplies from two suppliers
---------------------------------------------

A fleet of 1,200 power supplies came from two suppliers (``supplier_b``
is 1 for the second) and travelled different distances to their sites
(``distance``, in thousands of kilometres). Some failed at commissioning
(recorded as 0 days); the rest have been in service for 500 to 1,500
days. The data are simulated, so the truth is known: the chance of
arriving dead rises with distance and is higher for supplier B, and
supplier B's working units also fail sooner. Distance does nothing to
the life of a unit that arrived working.

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    import surpyval as sp

    rng = np.random.default_rng(7)
    n = 1200
    supplier_b = rng.binomial(1, 0.5, n)
    distance = rng.uniform(0, 2, n)
    p_doa = 1 / (1 + np.exp(-(-3.0 + 0.8 * supplier_b + 0.9 * distance)))
    doa = rng.random(n) < p_doa
    life = 2000 * np.exp(-0.3 * supplier_b) * rng.weibull(1.5, n)
    in_service = rng.uniform(500, 1500, n)
    failed = ~doa & (life < in_service)
    df = pd.DataFrame(
        {
            "days": np.where(doa, 0.0, np.where(failed, life, in_service)),
            "c": np.where(doa | failed, 0, 1),
            "supplier_b": supplier_b,
            "distance": distance,
        }
    )
    print("failed at commissioning:", int(np.sum(df["days"] == 0)))

A failure at commissioning is ``days = 0`` with ``c = 0``. Fit the
model. ``WeibullZeroInflated`` is ``ZeroInflated(Weibull)``: a Weibull
life for the units that work and a logistic regression for the chance of
arriving dead.

.. jupyter-execute::

    model = sp.WeibullZeroInflated.fit_from_df(
        df, x_col="days", c_col="c", Z_cols=["supplier_b", "distance"]
    )
    model

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _p = dict(zip(model.parameter_names, model.params))
    assert model.maximum == "verified"
    assert model.fit_route == "two-stage"
    assert round(_p["f0"], 3) == 0.048, _p
    assert round(_p["f0_supplier_b"], 1) == 0.8, _p
    assert round(np.exp(_p["f0_supplier_b"]), 1) == 2.2, _p
    assert round(_p["f0_distance"], 1) == 0.8, _p
    assert round(_p["alpha_supplier_b"], 2) == 0.26, _p
    assert round(np.exp(-_p["alpha_supplier_b"]), 2) == 0.77, _p
    assert abs(_p["alpha_distance"]) < 0.05, _p

How to read it:

- ``alpha`` and ``beta`` are the Weibull scale and shape of a working
  unit's life at the reference row (supplier A, no distance), and ``f0``
  is the probability there of arriving dead: about 5%. It is the same
  name as the univariate fit's zero-inflation share.
- ``f0_supplier_b`` and ``f0_distance`` are the coefficients of the
  dead-on-arrival chance, on the logit scale, so ``exp`` of one is an
  odds ratio. A supplier B unit has ``exp(0.80) = 2.2`` times the odds of
  arriving dead, and each thousand kilometres multiplies the odds by
  about the same again.
- ``alpha_supplier_b`` and ``alpha_distance`` are the coefficients of the
  life of the units that work, with the sign of SurPyval's AFT models:
  positive means shorter. A working supplier B unit has
  ``exp(-0.26) = 0.77`` times the life of a supplier A unit. Distance
  shows no effect on the life of a unit that arrived working
  (``alpha_distance`` is near 0, as simulated): it only breaks units on
  the way.

Every coefficient has the same sign convention: **positive is worse**, a
higher chance of arriving dead or an earlier failure.

The two parts for a given unit:

.. jupyter-execute::

    rows = pd.DataFrame({"supplier_b": [0, 1], "distance": [1.0, 1.0]})
    print("dead on arrival:", model.doa_probability(rows).round(3))
    print(
        "median life of a unit that works:",
        model.qf(0.5, rows, susceptible=True).round(0),
    )

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _f0 = model.doa_probability(rows)
    assert _f0[1] > 1.8 * _f0[0], _f0

``sf``, ``ff``, ``hf`` and the rest are the *population's*: the survival
of a unit about to be commissioned, which drops to ``1 - f0`` at time 0.
With ``susceptible=True`` they are those of a unit that worked at
commissioning.

.. jupyter-execute::

    b = rows.iloc[[1]]
    t = [0, 365, 1000]
    print("a unit to be commissioned:", model.sf(t, b).round(3))
    print("a unit that started      :", model.sf(t, b, susceptible=True).round(3))
    print("95% bounds at 365 days   :", model.cb(365, b).round(3))

The difference is the spares question. To cover a site for a year, plan
for the units that will arrive dead *and* for the working ones that will
fail in service; once a unit is running, only the second applies.

Choosing what acts on what
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default every covariate acts on ``f0`` and on the latency's life
parameter (the scale of a Weibull or log-logistic, ``mu`` of a
log-normal, the rate of a gamma or exponential). ``covariates`` chooses,
by parameter; with a data frame, ``formulas`` gives each part its own
formula:

.. jupyter-execute::

    chosen = sp.WeibullZeroInflated.fit_from_df(
        df,
        x_col="days",
        c_col="c",
        formulas={"f0": "supplier_b + distance", "alpha": "supplier_b"},
    )
    print(chosen.parameter_names)
    print("AIC, distance on both parts:", round(model.aic(), 1))
    print("AIC, distance on f0 only   :", round(chosen.aic(), 1))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert chosen.aic() < model.aic(), (chosen.aic(), model.aic())

Dropping distance from the life lowers the AIC: the data agree that it
does not change how long a working unit lasts. A covariate on the shape
is possible too (``covariates={"beta": [...]}``), as in a
:doc:`distributional regression <Distributional Regression>`, of which
this model is a member: the fits, checks, bounds (``cb``, ``param_cb``
and ``parameter_cb`` with Wald, likelihood-ratio and bootstrap methods),
weights, ``fit_penalised``, ``to_dict`` and ``from_dict`` are that
family's. ``ZeroInflated(Weibull, incidence="probit")`` (or
``"cloglog"``) changes the link of ``f0``.

When a failure might have been on arrival
-----------------------------------------

Sometimes the data do not say whether a unit was dead on arrival. A site
that is only visited at day 30 finds a failed unit, but cannot tell
whether it failed at commissioning or on day 12. That unit's time is
recorded as an interval from 0 to 30 (or as left censored at 30: "failed
by day 30"). Such a row is evidence for both parts at once, and the
model uses it that way: it contributes
:math:`f_0 + (1 - f_0) F_u(30)`, the chance of arriving dead *or*
failing by day 30.

.. jupyter-execute::

    visited = rng.random(n) < 0.3
    early = visited & (df["days"] <= 30) & (df["c"] == 0)
    inspected = df.assign(
        lo=np.where(early, 0.0, df["days"]),
        hi=np.where(early, 30.0, df["days"]),
        c=np.where(early, 2, df["c"]),
    )
    joint = sp.WeibullZeroInflated.fit_from_df(
        inspected,
        xl_col="lo",
        xr_col="hi",
        c_col="c",
        Z_cols=["supplier_b", "distance"],
    )
    print(int(early.sum()), "units found failed at the day-30 visit")
    print("route:", joint.fit_route)
    joint.summary().round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert joint.fit_route == "joint"
    assert joint.maximum == "verified"
    _pj = dict(zip(joint.parameter_names, joint.params))
    assert abs(_pj["f0_distance"] - _p["f0_distance"]) < 0.05, _pj

The estimates barely move: losing the exact times of a few early
failures costs little. The fit is a different route to the same kind of
answer, as the next section explains.

How the model is fitted
-----------------------

**When every row says whether its unit was dead on arrival** (a failure
at 0 was; a failure or a censoring after 0 was not), the likelihood
splits into two pieces with no parameter in common:

- a **binary regression**, dead on arrival or not, over every unit; and
- the **latency's regression**, over the units that worked, with all
  their censoring and truncation.

Each is fitted on its own, and together they are exactly the joint
maximum, with a covariance that has no terms between the two parts.
This is the *two-stage* route, the usual one, and it is fast and
reliable: neither piece can pull the other off course.

**When some rows do not say** (left censoring, an interval that starts
at 0, or right truncation without late entry), those rows tie the two
parts together, and both are searched together from the zero-inflated
fit without covariates. This is the *joint* route.

``fit`` chooses the route (``method="auto"``); ``method="joint"`` forces
the joint search, and ``method="two-stage"`` refuses data that do not
allow it. ``model.fit_route`` says which was used. A fit with ``fixed``
parameters or an ``init`` is searched jointly.

**Late entry.** A unit that entered observation at an age above 0 (left
truncation, ``tl > 0``) was known to be working then, so it says nothing
about ``f0``: it is left out of the binary regression and enters only
the latency's, divided by its survival to the entry age. A left
truncation at 0 is no truncation.

The model
---------

A unit with covariates :math:`z` is dead on arrival with probability
:math:`f_0(z)`, and a unit that works has a life with distribution
function :math:`F_u(t \mid z)`. The population's distribution function
is

.. math::

    F(t \mid z) = f_0(z) + \bigl(1 - f_0(z)\bigr)\, F_u(t \mid z),

which jumps to :math:`f_0(z)` at time 0.

**Dead on arrival.** :math:`g(f_0(z)) = g(f_0) + x_{f_0}(z)' b_{f_0}`,
with :math:`g` the logit (the default; logistic regression), the probit
or the complementary log-log link, chosen by ``ZeroInflated(latency,
incidence=...)``. ``f0`` is :math:`f_0` at the reference row.

**Latency.** A Weibull, log-normal, log-logistic, gamma or exponential
distribution whose parameters depend on the covariates through their own
links, as in a :doc:`distributional regression <Distributional
Regression>`: a scale as :math:`\alpha(z) = \alpha\, e^{-x' b}` (the AFT
model), a location as :math:`\mu(z) = \mu - x' b`, a rate as
:math:`\lambda(z) = \lambda\, e^{x' b}`.

**Likelihood.** Each kind of observation contributes what the population
model says of it:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Observation
     - Contribution
   * - Failed at 0 (dead on arrival)
     - :math:`f_0`
   * - Failed at :math:`t > 0`
     - :math:`(1 - f_0)\, f_u(t)`
   * - Still working at :math:`t` (right censored)
     - :math:`(1 - f_0)\, S_u(t)`
   * - Failed by :math:`t` (left censored, or an interval from 0 to
       :math:`t`)
     - :math:`f_0 + (1 - f_0)\, F_u(t)`
   * - Failed between :math:`l > 0` and :math:`r`
     - :math:`(1 - f_0)\, (F_u(r) - F_u(l))`
   * - Entered observation at age :math:`a > 0` (left truncated)
     - divided by :math:`(1 - f_0)\, S_u(a)`

Counts ``n`` and weights ``w`` multiply each row's term. The covariance
is the inverse of the observed information (the robust one for a
weighted fit), and the fit reports whether it reached a verified maximum
(``model.maximum``). With no covariates on either part the fit is the
distribution's own zero-inflated fit, ``latency.fit(x, c, zi=True)``, to
the last digit of the likelihood.

One convention differs from the univariate fits: there, an interval
starting at 0 means "after 0 and by :math:`t`", which leaves out the
units dead on arrival. Here it includes them, because an inspection that
finds a unit failed cannot tell when, and the unit may never have
worked. Record a failure known to be after 0 with a lower bound above 0.

**Refusals.** If no unit failed at 0 and every row says its unit worked,
``f0`` is estimated as 0 and the model is the latency's regression
alone: the fit refuses and says to fit that instead. If every unit that
could have been dead on arrival was, there is no life left to fit, and
the fit refuses too. Where the joint route's ``f0`` runs towards 0, the
fit warns "No finite maximum", naming ``f0``.

What is not built
-----------------

- **A semi-parametric (Cox) latency**: the latency here is parametric.
- **Zero inflation and a cure fraction together** (``zi=True`` with
  ``lfp=True``).
- **Latencies on the whole line** (normal, Gumbel, logistic): a time to
  failure is positive.
- **Covariate paths** (``fit_tvc``), as for every distributional
  regression.

References
----------

- Lambert, D. (1992). Zero-inflated Poisson regression, with an
  application to defects in manufacturing. *Technometrics* 34, 1-14.
- Farewell, V. T. (1982). The use of mixture models for the analysis of
  survival data with long-term survivors. *Biometrics* 38, 1041-1046.
