Mixture-Cure Regression
=======================

Most survival models assume that every unit fails eventually. Often that
is not true:

- only the parts with a manufacturing defect fail of it; the sound ones
  never do, however long they run;
- some patients are cured, and never have the relapse the study follows;
- some customers never cancel, some borrowers never default.

The mark of such data is a survival curve that **levels off above zero**:
after a while the failures stop, with many units still under observation.

A mixture-cure model treats the population as two groups. A unit is
*susceptible* (it will fail) with some probability, and a susceptible
unit has a time to failure with an ordinary lifetime distribution. A
mixture-cure *regression* lets covariates act on both parts, and so
answers two different questions at once:

- **Who fails?** Which lots, sites or patients have a higher chance of
  ever failing. This is the *incidence*.
- **When?** Among those that do fail, what makes failure come sooner.
  This is the *latency*.

A covariate can matter for one and not the other. A bad production lot
may contain more defective parts without the defective parts failing any
faster; a treatment may cure more patients without delaying relapse in
those it does not cure. An ordinary regression, with one effect per
covariate, cannot tell these apart.

Without covariates this is the limited-failure-population model that the
distributions already fit (``Weibull.fit(x, c, lfp=True)``, see
:doc:`Parametric SurPyval Modelling`). ``MixtureCure`` is that model
with covariates. The API reference is :doc:`regression/mixture_cure`.

A warranty example: a defect that depends on the lot
----------------------------------------------------

A controller fails early when a solder joint is defective. Two things
were recorded for each of 1,500 units: whether it came from a production
lot made after a process change (``late_lot``), and the thermal load at
its site (``load``, standardised). Units have been in service for between
300 and 500 days. The data are simulated, so the truth is known: the
defect is more common in the late lots and at high load, and the late
lots' defective units also fail sooner.

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    import surpyval as sp

    rng = np.random.default_rng(5)
    n = 1500
    late_lot = rng.binomial(1, 0.5, n)
    load = rng.normal(size=n)
    p_defect = 1 / (1 + np.exp(-(-0.5 + 1.0 * late_lot + 0.5 * load)))
    defective = rng.random(n) < p_defect
    life = 100 * np.exp(-0.4 * late_lot) * rng.weibull(2.0, n)
    in_service = rng.uniform(300, 500, n)
    failed = defective & (life < in_service)
    df = pd.DataFrame(
        {
            "days": np.where(failed, life, in_service),
            "c": np.where(failed, 0, 1),
            "late_lot": late_lot,
            "load": load,
        }
    )

The Kaplan-Meier curve shows the plateau: failures stop after about 250
days with roughly half the units still running.

.. jupyter-execute::

    km = sp.KaplanMeier.fit(df["days"], c=df["c"])
    ax = km.plot()

Fit the model. ``WeibullMixtureCure`` is ``MixtureCure(Weibull)``: a
Weibull life for the defective units and a logistic regression for the
chance of being defective.

.. jupyter-execute::

    model = sp.WeibullMixtureCure.fit_from_df(
        df, x_col="days", c_col="c", Z_cols=["late_lot", "load"]
    )
    model

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _p = dict(zip(model.parameter_names, model.params))
    assert model.maximum == "verified"
    assert round(_p["lfp_p"], 2) == 0.39, _p
    assert round(_p["lfp_p_late_lot"], 1) == 1.1, _p
    assert round(np.exp(_p["lfp_p_late_lot"]), 1) == 2.9, _p
    assert round(_p["alpha_late_lot"], 2) == 0.42, _p
    assert round(np.exp(-_p["alpha_late_lot"]), 2) == 0.65, _p
    assert abs(_p["alpha_load"]) < 0.05, _p

How to read it:

- ``alpha`` and ``beta`` are the Weibull scale and shape of a defective
  unit's life at the reference row (an early lot at average load), and
  ``lfp_p`` is the probability there of being defective: 0.39. It is the
  same name as the univariate fit's limited-failure proportion.
- ``lfp_p_late_lot`` and ``lfp_p_load`` are the incidence's coefficients,
  on the logit scale, so ``exp`` of one is an odds ratio. A late lot has
  ``exp(1.1) = 2.9`` times the odds of containing the defect.
- ``alpha_late_lot`` and ``alpha_load`` are the latency's coefficients,
  with the sign of SurPyval's AFT models: positive means shorter. A late
  lot's defective units have ``exp(-0.42) = 0.65`` times the life of an
  early lot's. Load shows no effect on *when* a defective unit fails
  (``alpha_load`` is near 0, as simulated): it only makes the defect more
  likely.

Every coefficient has the same sign convention: **positive is worse**, a
higher chance of failing or an earlier failure.

The two parts for a given unit:

.. jupyter-execute::

    rows = pd.DataFrame({"late_lot": [0, 1], "load": [0.0, 0.0]})
    print("defective :", model.susceptible_probability(rows).round(3))
    print("sound     :", model.cure_probability(rows).round(3))
    print(
        "median life of a defective unit:",
        model.qf(0.5, rows, susceptible=True).round(1),
    )

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _pi = model.susceptible_probability(rows)
    assert _pi[1] > _pi[0] + 0.2, _pi

``sf``, ``ff``, ``hf`` and the rest are the *population's*: the survival
of a unit whose status is not known, which levels off at the cure
probability. With ``susceptible=True`` they are the latency's, the
functions of a unit that does fail.

.. jupyter-execute::

    late = pd.DataFrame({"late_lot": [1], "load": [0.0]})
    t = [50, 100, 200, 400]
    print("population :", model.sf(t, late).round(3))
    print("defective  :", model.sf(t, late, susceptible=True).round(3))
    print("95% bounds at 400 days:", model.cb(400, late).round(3))

Is a unit that is still running sound?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The field question is about the units still in service. A new unit from
a late lot is sound with probability ``cure_probability``. One that has
already run 150 days without failing is much more likely to be sound,
because most of the defective ones would have failed by then:

.. jupyter-execute::

    ages = [0, 50, 100, 150, 250]
    model.cure_given_survival(ages, late).round(3)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _cgs = model.cure_given_survival(ages, late)
    assert np.all(np.diff(_cgs) > 0), _cgs
    assert _cgs[3] > 0.9, _cgs

This is the number to plan a recall or a spares buy with: of the late-lot
units still running at a given age, the share that will never fail.

Choosing what acts on what
~~~~~~~~~~~~~~~~~~~~~~~~~~

By default every covariate acts on the incidence and on the latency's
life parameter (the scale of a Weibull or log-logistic, ``mu`` of a
log-normal, the rate of a gamma or exponential). ``covariates`` chooses,
by parameter; with a data frame, ``formulas`` gives each part its own
formula:

.. jupyter-execute::

    chosen = sp.WeibullMixtureCure.fit_from_df(
        df,
        x_col="days",
        c_col="c",
        formulas={"lfp_p": "late_lot + load", "alpha": "late_lot"},
    )
    chosen.parameter_names

.. jupyter-execute::

    print("AIC, load on both parts  :", round(model.aic(), 1))
    print("AIC, load on incidence   :", round(chosen.aic(), 1))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert chosen.aic() < model.aic(), (chosen.aic(), model.aic())

Dropping the load from the latency lowers the AIC: the data agree that
it does not change when a defective unit fails. A covariate on the shape
is possible too (``covariates={"beta": [...]}``), as in a
:doc:`distributional regression <Distributional Regression>`, of which
this model is a member: the fits, checks, bounds (``cb``, ``param_cb``
and ``parameter_cb`` with Wald, likelihood-ratio and bootstrap methods),
weights, ``to_dict`` and ``from_dict`` are that family's.

A medical example: who relapses, and when
-----------------------------------------

In a trial, some patients given a treatment never relapse. Two questions
are asked of the treatment: does it cure more patients, and does it delay
relapse in those it does not cure? The data here are simulated with a
treatment that does the first and not the second, and a log-normal time
to relapse (in months), followed for five to eight years.

.. jupyter-execute::

    rng = np.random.default_rng(11)
    n = 800
    treated = rng.binomial(1, 0.5, n)
    relapses = rng.random(n) < np.where(treated == 1, 0.45, 0.70)
    months = np.exp(rng.normal(2.3, 0.7, n))
    follow = rng.uniform(60, 96, n)
    seen = relapses & (months < follow)
    trial = pd.DataFrame(
        {
            "months": np.where(seen, months, follow),
            "c": np.where(seen, 0, 1),
            "treated": treated,
        }
    )
    cure = sp.LogNormalMixtureCure.fit_from_df(
        trial, x_col="months", c_col="c", Z_cols=["treated"]
    )
    cure.summary().round(3)

.. jupyter-execute::

    arms = pd.DataFrame({"treated": [0, 1]})
    print("cured:", cure.cure_probability(arms).round(3))
    print("odds ratio of relapse:", np.exp(cure.params[-1]).round(2))
    print("95% interval:", np.exp(cure.param_cb("lfp_p_treated")).round(2))
    print("p-value, effect on time to relapse:", round(cure.p_values[3], 2))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _c = cure.cure_probability(arms)
    assert round(_c[0], 1) == 0.3 and _c[1] > 0.5, _c
    assert np.exp(cure.param_cb("lfp_p_treated"))[1] < 1
    assert cure.p_values[3] > 0.05, cure.p_values

About 30% of the untreated and over half of the treated are cured, an
odds ratio of relapse well below 1, while the coefficient on the time to
relapse (``mu_treated``) is not distinguishable from 0. A Cox or AFT
model of the same data would report one diluted "treatment effect" and
break its own assumptions doing it (the hazards of the two arms are not
proportional when one arm has more cured patients).

Was follow-up long enough?
--------------------------

A cure fraction can only be seen if the units were followed past the
point where the susceptible ones have failed. If follow-up stops while
failures are still arriving, the data cannot tell "30% never fail" from
"30% have not failed *yet*", and the fitted cure probability is a guess
that depends on the latency distribution chosen. No model can fix that.

Two checks:

- **Look at the Kaplan-Meier curve.** It should be flat at the end, with
  a good number of units still at risk on the flat part.
- ``model.follow_up()`` is Maller and Zhou's check. It compares the
  length of the plateau (from the last failure to the end of follow-up)
  with how the failures were spaced before it, and returns a p-value:
  small means the failures had died away before follow-up ended.

.. jupyter-execute::

    model.follow_up()

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert model.follow_up()["sufficient"]

When the data have no plateau at all, the likelihood has no maximum
inside the model: the susceptible probability runs to 1. The fit then
warns "No finite maximum", naming ``lfp_p``, and says to fit the latency
alone; ``model.maximum`` is ``"no finite maximum"``. Take the warning at
its word.

The model
---------

A unit with covariates :math:`z` is susceptible with probability
:math:`\pi(z)`, and a susceptible unit's life has survival function
:math:`S_u(t \mid z)`. The population's survival is

.. math::

    S(t \mid z) = 1 - \pi(z) + \pi(z)\, S_u(t \mid z),

which falls to the cure probability :math:`1 - \pi(z)`, not to 0.

**Incidence.** :math:`g(\pi(z)) = g(\pi) + x_\pi(z)' b_\pi`, with
:math:`g` the logit (the default; logistic regression), the probit or the
complementary log-log link, chosen by ``MixtureCure(latency,
incidence=...)``. ``lfp_p`` is :math:`\pi` at the reference row.

**Latency.** A Weibull, log-normal, log-logistic, gamma or exponential
distribution whose parameters depend on the covariates through their own
links, as in a :doc:`distributional regression <Distributional
Regression>`: a scale as :math:`\alpha(z) = \alpha\, e^{-x' b}` (the AFT
model), a location as :math:`\mu(z) = \mu - x' b`, a rate as
:math:`\lambda(z) = \lambda\, e^{x' b}`. For a Weibull latency, a
coefficient :math:`b` on the scale is a log hazard ratio of
:math:`\beta\, b` among the susceptible, :math:`\beta` the shape.

**Likelihood.** Both parts are fitted together by maximum likelihood.
Each kind of observation contributes what the population model says of
it:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Observation
     - Contribution
   * - Failed at :math:`t`
     - :math:`\pi\, f_u(t)`
   * - Still running at :math:`t` (right censored)
     - :math:`1 - \pi + \pi\, S_u(t)`
   * - Failed before :math:`t` (left censored)
     - :math:`\pi\, F_u(t)`
   * - Failed between :math:`l` and :math:`r` (interval censored)
     - :math:`\pi\, (F_u(r) - F_u(l))`
   * - Entered observation at age :math:`a` (left truncated)
     - divided by :math:`1 - \pi + \pi\, S_u(a)`

A unit still running is either cured or a susceptible unit that has not
failed yet; the likelihood adds the two. Counts ``n`` and weights ``w``
multiply each row's term. The covariance is the inverse of the observed
information (the robust one for a weighted fit), and the fit reports
whether it reached a verified maximum (``model.maximum``).

With no covariates on either part the fit is the distribution's own
limited-failure fit, ``latency.fit(x, c, lfp=True)``, to the last digit
of the likelihood. The fits are checked against R's ``flexsurvcure``.

**The chance that a survivor is cured** follows from Bayes' rule:

.. math::

    P(\text{cured} \mid T > t, z)
    = \frac{1 - \pi(z)}{1 - \pi(z) + \pi(z)\, S_u(t \mid z)} .

What is not built
-----------------

- **A semi-parametric (Cox) latency**, as in R's ``smcure``: the latency
  here is parametric. A flexible parametric choice (compare the
  latencies by AIC) covers most uses.
- **Covariates on a dead-on-arrival fraction** (zero-inflation,
  ``zi=True`` in the univariate fits).
- **The incidence estimated by a general classifier** (a gradient-boosted
  or random-forest model in place of the link).
- **Latencies on the whole line** (normal, Gumbel, logistic): a time to
  failure is positive.
- **Covariate paths** (``fit_tvc``), as for every distributional
  regression.

References
----------

- Berkson, J. and Gage, R. P. (1952). Survival curve for cancer patients
  following treatment. *Journal of the American Statistical Association*
  47, 501-515.
- Farewell, V. T. (1982). The use of mixture models for the analysis of
  survival data with long-term survivors. *Biometrics* 38, 1041-1046.
- Maller, R. A. and Zhou, X. (1996). *Survival Analysis with Long-Term
  Survivors*. Wiley.
- Meeker, W. Q. (1987). Limited failure population life tests:
  application to integrated circuit reliability. *Technometrics* 29,
  51-65.
- Peng, Y. and Dear, K. B. G. (2000). A nonparametric mixture model for
  cure rate estimation. *Biometrics* 56, 237-243.
- Amdahl, J. (2022). *flexsurvcure: Flexible Parametric Cure Models*. R
  package.
