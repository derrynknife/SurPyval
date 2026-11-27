Coming from R
=============

This page is for people who do survival analysis in R, with ``survival``
and the packages around it (``flexsurv``, ``cmprsk``, ``mstate``,
``icenReg``, ``coxme``), and want to do the same in SurPyval. It maps the
functions, shows six common tasks side by side, and lists the differences
that catch people out. The R code is shown as written; the SurPyval code
runs when this page is built, and its numbers were checked against
survival 3.8.12, flexsurv 2.3.2 and cmprsk 2.2.12 on the same data (hidden
checks stop the build if they drift). Most of SurPyval's regression models
are tested against R in the test suite as well; the reference scripts are
in ``scripts/reference/``.

``Surv()`` and the data model
-----------------------------

R puts the outcome in a ``Surv`` object; SurPyval passes the same
information as plain arrays, the *xcnt* data model (:doc:`Conventions`):
``x`` the time, ``c`` the censoring flag, ``n`` a count of identical rows
and ``tl``/``tr`` the truncation (late entry, or units that are in the data
only because they had failed by some time).

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - R
     - SurPyval
   * - ``Surv(time, status)``, status 1 = event (or 2 = event, 1 =
       censored, as in ``lung``)
     - ``x=time, c=1 - event``: ``c`` is 0 for an event, 1 for right
       censored
   * - ``Surv(start, stop, status)`` (late entry)
     - ``x=stop, c=..., tl=start``; for time-varying covariates
       ``CoxPH.fit_tvc(i, xl=start, xr=stop, c, Z)``
   * - ``Surv(left, right, type = "interval2")``
     - ``xl=left, xr=right`` (``xl == xr`` exact, ``xl = 0`` left censored,
       ``xr = inf`` right censored), or ``x`` as pairs with ``c = 2``
   * - ``Surv(time, status, type = "left")``
     - ``c = -1``
   * - ``weights =`` (case weights)
     - ``n =`` for whole-number counts, ``w =`` for sampling weights (robust
       variance, as ``survreg(..., robust = TRUE)``)
   * - ``conf.int = 0.95``
     - ``alpha_ci = 0.05``

The functions
-------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - R
     - SurPyval
   * - ``survfit(Surv(t, s) ~ 1)``
     - ``KaplanMeier.fit(t, c)``; also ``NelsonAalen``,
       ``FlemingHarrington``, and ``Turnbull`` for interval-censored or
       truncated data (``icenReg::ic_np``, ``interval::icfit``)
   * - ``survfit(Surv(t, s) ~ group)``
     - one model per group, drawn together with ``plot_nonparametric``
   * - ``summary(fit, times = ...)``
     - ``km.sf(t)``, ``km.cb(t)``; the number at risk with ``at_risk_table``
   * - ``print(fit)`` (median and its interval)
     - ``km.median``, ``km.median_cb()``
   * - ``survdiff``
     - ``logrank``; for ``rho = r`` use
       ``weighting="fleming-harrington"`` with ``rho=r``
       (``"peto-peto"`` uses Peto and Peto's own survival estimate,
       close to ``rho = 1`` but not equal);
       ``pairwise_logrank(..., adjust="bh")`` for
       ``survminer::pairwise_survdiff``, whose default adjustment is
       Benjamini-Hochberg (``pairwise_logrank``'s is Holm)
   * - ``coxph``
     - ``CoxPH`` (Efron ties by default, as R)
   * - ``coxph(... + strata(g))``
     - ``CoxPH.fit(..., strata=g)``
   * - ``coxph(... + tt(z), tt = ...)``
     - ``CoxPH.fit(..., time_interaction={"z": "log"})``
   * - ``coxph(... + frailty(g))``, ``coxme``
     - ``CoxFrailty``, ``CoxMixedEffects``
   * - ``cox.zph``
     - ``cox.diagnose()`` (``tables["proportional_hazards"]``)
   * - ``survreg(..., dist = "weibull")``
     - ``WeibullAFT`` (see the signs and scale below); ``Weibull.fit`` with
       no covariates
   * - ``flexsurvreg``
     - ``Weibull.fit``, ``WeibullAFT``, ``WeibullPH`` and the other
       ``<Dist>AFT`` / ``<Dist>PH`` fitters; ``WeibullDistributional`` for
       covariates on the shape (``anc =``)
   * - ``flexsurvspline``
     - ``RoystonParmar``, ``RoystonParmarRegression(df=, scale=)``,
       ``RoystonParmarPH``, ``RoystonParmarPO``
   * - ``icenReg::ic_sp``
     - ``IntervalCoxPH``
   * - ``aareg``
     - ``AalenAdditive``
   * - ``pyears``
     - ``person_years``
   * - ``cmprsk::cuminc``
     - ``CompetingRisks`` (``surpyval.univariate.competing_risks``)
   * - ``cmprsk::crr``, ``survival::finegray``
     - ``FineGray``
   * - ``mstate`` (``msfit``, ``probtrans``)
     - ``MultiState``, ``MultiStateProportionalHazards``
       (``surpyval.multistate``); see :doc:`Multi-State Models`
   * - ``survival::concordance``, ``timeROC``
     - ``surpyval.metrics.concordance_index``, ``auc_td``
   * - ``survival::lung``, ``mgus2``
     - ``surpyval.datasets.load_lung``, ``load_mgus2``

Kaplan-Meier
------------

In R's ``lung``, ``status`` is 2 for a death and 1 for censored. The copy in
SurPyval has it recoded to 1 for a death and 0 for censored, so ``c = 1 -
status``.

.. code-block:: r

    library(survival)
    fit <- survfit(Surv(time, status) ~ 1, data = lung)
    summary(fit, times = 365)   # surv 0.4092, lower 0.3447, upper 0.4858
    fit                         # median 310, 0.95LCL 285, 0.95UCL 363

.. jupyter-execute::

    import numpy as np
    import surpyval as sp

    lung = sp.datasets.load_lung()
    lung["c"] = 1 - lung["status"]
    km = sp.KaplanMeier.fit(lung["time"], lung["c"])
    print(km.sf(365), km.cb(365, bound_type="log"))
    print(km.median, km.median_cb(bound_type="log"))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(km.cb(365, bound_type="log"), [0.3447216, 0.4858376], atol=1e-7)
    assert round(float(km.sf(365)), 4) == 0.4092, km.sf(365)

``survfit`` puts the interval on the log scale by default (``conf.type =
"log"``); SurPyval's default is log(-log), which keeps the interval inside
[0, 1] and is what ``lifelines`` uses. Pass ``bound_type="log"`` to get R's
numbers. The median's interval is read off the interval curve, as R
does, and on the log scale it is R's 285 to 363:

.. jupyter-execute::

    km.median_cb(bound_type="log")

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert list(km.median_cb(bound_type="log")) == [285.0, 363.0], km.median_cb(bound_type="log")
    assert km.median == 310.0, km.median

Cox regression and ``cox.zph``
------------------------------

.. code-block:: r

    cx <- coxph(Surv(time, status) ~ age + sex + ph.ecog, data = lung)
    coef(cx)      # age 0.011067, sex -0.552612, ph.ecog 0.463728
    cox.zph(cx)   # age 0.188, sex 2.305, ph.ecog 2.054, GLOBAL 4.464 (3 df)

One patient has no ``ph.ecog``. R drops the row (``na.action``); SurPyval
drops it too and says so in a warning.

.. jupyter-execute::

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # "Dropped 1 of 228 rows ..."
        cox = sp.CoxPH.fit_from_df(
            lung, x_col="time", c_col="c", Z_cols=["age", "sex", "ph.ecog"]
        )
    print(cox.beta)
    cox.diagnose().tables["proportional_hazards"]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(cox.beta, [0.01106676, -0.55261240, 0.46372848], atol=1e-7), cox.beta
    _ph = cox.diagnose().tables["proportional_hazards"]
    assert np.allclose(_ph["statistic"], [0.18798767, 2.30543723, 2.05424877, 4.46365759], atol=1e-6), _ph

The coefficients, standard errors and the ``cox.zph`` statistics (with
its default ``transform = "km"``) agree with R to six digits or more.
A stratified model is the ``strata`` argument:

.. code-block:: r

    coef(coxph(Surv(time, status) ~ age + strata(sex), data = lung))  # 0.016215

.. jupyter-execute::

    sp.CoxPH.fit(lung["time"], lung[["age"]], lung["c"], strata=lung["sex"]).beta

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _b = sp.CoxPH.fit(lung["time"], lung[["age"]], lung["c"], strata=lung["sex"]).beta
    assert round(float(_b[0]), 6) == 0.016215, _b

``survreg`` and ``flexsurvreg``
-------------------------------

.. code-block:: r

    sr <- survreg(Surv(time, status) ~ age + sex, data = lung, dist = "weibull")
    coef(sr)    # (Intercept) 6.274853, age -0.012257, sex 0.382085
    sr$scale    # 0.754051
    flexsurvreg(Surv(time, status) ~ age + sex, data = lung, dist = "weibull")
    # shape 1.32617, scale 531.048, age -0.012257, sex 0.382085

.. jupyter-execute::

    aft = sp.WeibullAFT.fit_from_df(lung, x_col="time", c_col="c", Z_cols=["age", "sex"])
    dict(zip(aft.parameter_names, aft.params.round(6)))

The same model, in a third parameterisation:

- ``survreg`` works on log time: its intercept is :math:`\log\alpha`
  (:math:`\log 531.05 = 6.2749`) and its ``scale`` is :math:`1/\beta`
  (:math:`1 / 1.3262 = 0.7541`). ``flexsurvreg``'s ``shape`` and ``scale``
  are SurPyval's :math:`\beta` and :math:`\alpha`.
- **The coefficients have the opposite sign.** In R a positive AFT
  coefficient means a longer life; in SurPyval's AFT models the scale is
  :math:`\alpha e^{-b'Z}`, so a positive coefficient means a shorter one.
  Women (``sex`` 2) live longer here: +0.382 in R, -0.382 in SurPyval, the
  same factor :math:`e^{0.382} = 1.47` on the time scale.
- Proportional hazards coefficients (``coxph``, ``flexsurvreg(dist =
  "weibullPH")``) have the same sign in both.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _p = dict(zip(aft.parameter_names, aft.params))
    assert round(np.log(_p["alpha"]), 4) == 6.2749, _p
    assert round(1 / _p["beta"], 4) == 0.7541, _p
    assert round(_p["sex"], 5) == -0.38208, _p
    assert round(_p["age"], 5) == 0.01226, _p

Flexible parametric models: ``flexsurvspline``
-----------------------------------------------

.. code-block:: r

    library(flexsurv)
    s <- flexsurvspline(Surv(time, status) ~ sex, data = lung, k = 1, scale = "hazard")
    s$res[, "est"]   # gamma0 -6.56996, gamma1 1.14997, gamma2 -0.01516, sex -0.52313
    summary(s, newdata = data.frame(sex = 1), t = 365)   # 0.3659

``k`` in flexsurv counts the interior knots; SurPyval's ``df`` counts the
spline's degrees of freedom, one more (``df = k + 1``). The knots are at
the same places (the extremes and quantiles of the uncensored log
times), so the same ``df`` gives the same model.

.. jupyter-execute::

    rp = sp.RoystonParmarRegression(df=2, scale="hazard").fit(
        lung["time"], lung[["sex"]], lung["c"]
    )
    print(rp.parameter_names, rp.params)
    print(rp.sf(365, [1]))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(rp.params, [-6.569958, 1.149970, -0.015157, -0.523132], atol=1e-4), rp.params
    assert round(float(rp.sf(365, [1])), 4) == 0.3659, rp.sf(365, [1])

The estimates agree with flexsurv to four decimal places (the two
optimisers stop at slightly different points; the log-likelihoods agree to
six digits). On the odds scale (``scale = "odds"``) flexsurv's coefficients
are SurPyval's negated: SurPyval keeps the sign convention of all its
proportional odds models, positive = longer life.

Competing risks: ``cuminc`` and ``crr``
---------------------------------------

The ``mgus2`` data: 1,384 patients with a monoclonal gammopathy, followed
for progression to plasma cell malignancy (PCM) or death without it.

.. code-block:: r

    library(cmprsk)
    etime <- with(mgus2, ifelse(pstat == 0, futime, ptime))
    event <- with(mgus2, ifelse(pstat == 0, 2 * death, 1))
    timepoints(cuminc(etime, event), c(120, 240))$est
    #          120     240
    # 1 1   0.0637  0.0998     (PCM)
    # 1 2   0.5318  0.7240     (death)
    crr(etime, event, cov1 = cbind(age = mgus2$age, male = mgus2$sex == "M"),
        failcode = 1)$coef    # age -0.017338, male -0.260038

SurPyval takes the event type as ``e`` (any labels, blank for a censored
row) and the censoring flag as ``c``, rather than one coded column:

.. jupyter-execute::

    from surpyval.univariate.competing_risks import CompetingRisks

    mgus = sp.datasets.load_mgus2()
    progressed = mgus["pstat"] == 1
    etime = np.where(progressed, mgus["ptime"], mgus["futime"])
    died = (~progressed) & (mgus["death"] == 1)
    c = (~progressed & ~died).astype(int)            # 1 = censored
    e = np.where(progressed, "PCM", np.where(died, "death", None))

    cif = CompetingRisks.fit(etime, e=e, c=c)
    print(cif.cif([120, 240], "PCM"), cif.cif([120, 240], "death"))

    Z = np.column_stack([mgus["age"], mgus["sex"] == "M"]).astype(float)
    fg = sp.FineGray.fit(etime, Z, e, c=c, event="PCM")
    print(fg.beta, fg.standard_errors())

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(cif.cif([120, 240], "PCM"), [0.06372217, 0.09981372], atol=1e-7)
    assert np.allclose(cif.cif([120, 240], "death"), [0.53181770, 0.72402798], atol=1e-7)
    assert np.allclose(fg.beta, [-0.01733815, -0.26003824], atol=1e-6), fg.beta
    assert np.allclose(fg.standard_errors(), [0.00573710, 0.18568103], atol=1e-6)

The standard errors are ``crr``'s robust (sandwich) ones, its ``var``;
``fg.standard_errors(robust=False)`` gives the model-based ones
(``invinf``).

Log-rank: ``survdiff``
----------------------

.. code-block:: r

    survdiff(Surv(time, status) ~ sex, data = lung)$chisq   # 10.3267

.. jupyter-execute::

    sp.logrank(lung["time"], lung["sex"], lung["c"])

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _lr = sp.logrank(lung["time"], lung["sex"], lung["c"])
    assert round(float(_lr.statistic), 4) == 10.3267, _lr.statistic

Differences that catch people out
---------------------------------

- **The censoring flag**: ``c`` marks censoring, not the event. ``c = 1 -
  status`` for 0/1 data and ``c = 2 - status`` for R's 1/2 coding.
- **No formula interface for most fitters.** Arrays (or ``fit_from_df``
  with ``x_col``, ``c_col``, ``Z_cols``) go in; the regression
  ``fit_from_df`` methods also take ``formula="~ age + C(ph_ecog)"`` to
  code categorical columns.
- **AFT signs and the log-time scale** (above). ``survreg``'s
  ``scale`` is :math:`1/\beta` for the Weibull; for the log-normal it is
  SurPyval's :math:`\sigma`.
- **Interval level.** ``alpha_ci = 0.05`` is R's ``conf.int = 0.95``.
  ``bound="lower"`` or ``"upper"`` gives a one-sided bound.
- **Bound scales.** The Kaplan-Meier default is log(-log), R's is log (see
  above). R's ``conf.type`` names are SurPyval's ``bound_type`` names:
  ``"log"``, ``"log-log"``, ``"plain"``, ``"logit"``, ``"arcsine"``.
- **Models are objects with methods.** Instead of ``predict(fit, type =
  "quantile", p = 0.1)``, call ``model.qf(0.1)`` (and ``model.qf(0.1, Z)``
  for a regression); ``sf``, ``hf``, ``Hf``, ``df`` and ``cb`` work the same
  way.

What SurPyval adds
------------------

R's survival ecosystem is the reference for most of these methods, and
much of SurPyval is tested against it. What SurPyval brings is one data
model and one interface across all of them: the same ``x``, ``c``, ``n``,
``tl``, ``tr`` arrays (with left, right and interval censoring and left and
right truncation mixed in one data set) go into a Kaplan-Meier, a Weibull,
a Cox model, a competing-risks model or a joint model, and every fitted
model has the same methods (``sf``, ``qf``, ``cb``, ``to_dict``). It also
brings, in one package, the engineering methods R spreads over several
(``WeibullR`` and others): probability plotting and rank
regression, three-parameter (offset) distributions, accelerated life
models, repairable-system and reliability-growth models, degradation, and
test planning. Every fitted model saves to a plain dictionary or JSON file
and loads back without knowing its class.
