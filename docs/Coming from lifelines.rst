Coming from lifelines
=====================

This page is for people who know `lifelines
<https://lifelines.readthedocs.io>`__ and want to do the same work in
SurPyval. It maps the names, shows six common tasks side by side, and lists
the differences that catch people out. The lifelines code is shown as it is
written there; the SurPyval code below it runs when this page is built, and
the numbers it prints were checked against lifelines 0.30.3 on the same data
(hidden checks stop the build if they drift).

The one thing to remember
-------------------------

**The censoring flag points the other way.** lifelines takes an *event*
column, 1 when the event (death, failure, arrest) was seen. SurPyval takes a
*censoring* flag ``c``, 0 when the event was seen and 1 when the unit was
still event-free at ``x``. So a lifelines ``event_col`` becomes
``c = 1 - event``:

.. code-block:: python

    # lifelines
    kmf.fit(durations=df["T"], event_observed=df["E"])

    # SurPyval
    KaplanMeier.fit(x=df["T"], c=1 - df["E"])

Passing the event column as ``c`` does not raise an error, because both
are columns of 0s and 1s, but every result is then wrong: the deaths are
treated as survivors. SurPyval's ``c`` also has two more values, ``-1``
(left censored: the event happened before ``x``) and ``2`` (interval
censored: between two times), which lifelines spreads over separate
methods (``fit_left_censoring``, ``fit_interval_censoring``). See
:doc:`Conventions` for the whole data model.

Names
-----

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - lifelines
     - SurPyval
   * - ``durations``, ``duration_col``
     - ``x``, ``x_col``
   * - ``event_observed``, ``event_col`` (1 = event)
     - ``c``, ``c_col`` (0 = event, 1 = right censored): pass ``1 - event``
   * - ``entry`` (late entry)
     - ``tl`` (left truncation), ``tl_col``
   * - ``weights``, ``weights_col``
     - ``n`` for whole-number counts of identical rows; ``w`` for sampling
       or propensity weights
   * - ``lower_bound`` / ``upper_bound`` (interval censoring)
     - ``xl`` / ``xr``, or ``x`` as ``[left, right]`` pairs with ``c = 2``
   * - ``alpha=0.05`` (interval level)
     - ``alpha_ci=0.05``
   * - ``KaplanMeierFitter().fit(...)``
     - ``KaplanMeier.fit(...)`` (returns the fitted model)
   * - ``NelsonAalenFitter``
     - ``NelsonAalen``
   * - ``kmf.survival_function_at_times(t)``, ``kmf.predict(t)``
     - ``km.sf(t)``
   * - ``kmf.confidence_interval_``
     - ``km.cb(t)``, ``km.band(...)`` for a simultaneous band
   * - ``kmf.median_survival_time_``
     - ``km.median``, ``km.median_cb()``
   * - ``restricted_mean_survival_time(kmf, t)``
     - ``km.rmst(t)``
   * - ``WeibullFitter`` (``lambda_``, ``rho_``)
     - ``Weibull.fit`` (``alpha``, ``beta``: the same scale and shape)
   * - ``ExponentialFitter``, ``LogNormalFitter``, ``LogLogisticFitter``
     - ``Exponential``, ``LogNormal``, ``LogLogistic`` (and Gamma,
       Gumbel, ExpoWeibull and more; ``sp.fit_best`` tries them all)
   * - ``SplineFitter``, ``CRCSplineFitter``
     - ``RoystonParmar`` (and ``RoystonParmarPH`` / ``RoystonParmarPO`` with
       covariates)
   * - ``CoxPHFitter``
     - ``CoxPH`` (Efron ties by default, as lifelines)
   * - ``CoxPHFitter(strata=[...])``
     - ``CoxPH.fit(..., strata=...)``
   * - ``CoxTimeVaryingFitter``
     - ``CoxPH.fit_tvc`` / ``CoxPH.fit_tvc_from_df``
   * - ``cph.check_assumptions()``, ``proportional_hazard_test``
     - ``cox.diagnose()``
   * - ``WeibullAFTFitter``, ``LogNormalAFTFitter``, ``LogLogisticAFTFitter``
     - ``WeibullAFT``, ``LogNormalAFT``, ``LogLogisticAFT`` (mind the sign of
       the coefficients, below)
   * - ``WeibullAFTFitter().fit(df, ..., ancillary=True)`` (or a DataFrame)
     - ``Distributional(Weibull).fit_from_df(df, x_col=..., c_col=...,
       Z_cols=[...])`` puts every covariate on the shape as well
       (``ancillary=True``); ``formulas={"alpha": ..., "beta": ...}``
       chooses each parameter's own (``beta`` is lifelines' ``rho_``)
   * - ``AalenAdditiveFitter``
     - ``AalenAdditive``
   * - ``AalenJohansenFitter``
     - ``CompetingRisks`` (``surpyval.univariate.competing_risks``)
   * - ``penalizer=``, ``l1_ratio=``
     - ``penalizer=``, ``l1_ratio=``
   * - ``logrank_test``, ``multivariate_logrank_test``,
       ``pairwise_logrank_test``
     - ``logrank`` (any number of groups), ``pairwise_logrank``
   * - ``survival_difference_at_fixed_point_in_time_test``
     - ``sf_diff``
   * - ``add_at_risk_counts``
     - ``plot_nonparametric(..., at_risk_rows="all")``, or the table alone
       with ``at_risk_table``
   * - ``concordance_index``, ``k_fold_cross_validation``
     - ``surpyval.metrics.concordance_index``; no cross-validation helper
   * - ``print_summary()``, ``summary``
     - ``model.summary()`` (a DataFrame for regressions)
   * - ``lifelines.datasets.load_rossi``, ``load_lung``
     - ``surpyval.datasets.load_rossi_static``, ``load_lung`` (same columns)

Kaplan-Meier
------------

The ``lung`` data (228 patients with advanced lung cancer) is in both
packages, with ``status`` 1 for a death.

.. code-block:: python

    # lifelines
    from lifelines import KaplanMeierFitter
    from lifelines.datasets import load_lung

    lung = load_lung()
    kmf = KaplanMeierFitter().fit(lung["time"], lung["status"])
    kmf.predict(365)                 # 0.40924
    kmf.median_survival_time_        # 310.0
    kmf.confidence_interval_survival_function_.loc[:365].iloc[-1]
    # 0.33871, 0.47838

.. jupyter-execute::

    import numpy as np
    import surpyval as sp

    lung = sp.datasets.load_lung()
    km = sp.KaplanMeier.fit(lung["time"], c=1 - lung["status"])
    print(km.sf(365), km.median)
    print(km.cb(365))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert round(float(km.sf(365)), 5) == 0.40924, km.sf(365)
    assert km.median == 310.0, km.median
    assert np.allclose(km.cb(365), [0.33871427, 0.47838077], atol=1e-7), km.cb(365)

Both packages put the Greenwood bound on the log(-log) scale by default, so
the interval matches to every digit printed. (R's ``survfit`` uses the log
scale by default; pass ``bound_type="log"`` to match R.)

A parametric fit
----------------

lifelines' ``WeibullFitter`` writes the survival function as
:math:`S(t) = \exp(-(t/\lambda)^\rho)`; SurPyval's Weibull is the same
function with :math:`\alpha` for the scale and :math:`\beta` for the shape.

.. code-block:: python

    # lifelines
    from lifelines import WeibullFitter
    wf = WeibullFitter().fit(lung["time"], lung["status"])
    wf.lambda_, wf.rho_              # 417.76, 1.3168

.. jupyter-execute::

    wb = sp.Weibull.fit(lung["time"], c=1 - lung["status"])
    print(wb.parameter_names, wb.params)
    print(wb.param_cb("beta"))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(wb.params, [417.7586, 1.31684], rtol=1e-5), wb.params

Cox regression
--------------

The Rossi recidivism data: 432 released prisoners, followed for a year;
``arrest`` is 1 for a re-arrest.

.. code-block:: python

    # lifelines
    from lifelines import CoxPHFitter
    from lifelines.datasets import load_rossi

    rossi = load_rossi()
    cph = CoxPHFitter().fit(rossi, duration_col="week", event_col="arrest")
    cph.params_      # fin -0.3794, age -0.0574, ..., prio 0.0915
    cph.check_assumptions(rossi)

.. jupyter-execute::

    rossi = sp.datasets.load_rossi_static()
    rossi["c"] = 1 - rossi["arrest"]
    covariates = ["fin", "age", "race", "wexp", "mar", "paro", "prio"]

    cox = sp.CoxPH.fit_from_df(rossi, x_col="week", c_col="c", Z_cols=covariates)
    cox.summary()

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _ll = [-0.379422, -0.057438, 0.3139, -0.149796, -0.433704, -0.084871, 0.091497]
    _se = [0.191379, 0.021999, 0.307993, 0.212224, 0.381868, 0.195757, 0.028649]
    assert np.allclose(cox.beta, _ll, atol=1e-6), cox.beta
    assert np.allclose(cox.standard_errors(), _se, atol=1e-6), cox.standard_errors()

The coefficients and standard errors agree with lifelines to the sixth
decimal place. The check of proportional hazards is ``diagnose()``:

.. jupyter-execute::

    d = cox.diagnose()
    d.tables["proportional_hazards"]

These are the Grambsch-Therneau score tests computed as R's ``cox.zph``
does since survival 3.0, and they match it (age: 5.97, p = 0.015; global:
17.69 on 7 degrees of freedom). lifelines' ``proportional_hazard_test``
computes an approximation to that test (the one ``cox.zph`` used before
3.0), so its statistics differ (age: 11.03 there). The two agree on which
covariates to look at here, age and work experience, but not on the
numbers.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _ph = d.tables["proportional_hazards"]
    assert round(float(_ph.loc["age", "statistic"]), 2) == 5.97, _ph
    assert round(float(_ph.loc["GLOBAL", "statistic"]), 2) == 17.69, _ph

An accelerated failure time model
---------------------------------

.. code-block:: python

    # lifelines
    from lifelines import WeibullAFTFitter
    aft = WeibullAFTFitter().fit(rossi, "week", "arrest")
    aft.params_
    # lambda_: fin 0.2722, age 0.0407, ..., Intercept 3.9901
    # rho_:    Intercept 0.3391

.. jupyter-execute::

    aft = sp.WeibullAFT.fit_from_df(rossi, x_col="week", c_col="c", Z_cols=covariates)
    dict(zip(aft.parameter_names, aft.params.round(4)))

The same fit, written differently:

- lifelines reports :math:`\log\lambda` (3.9901) and :math:`\log\rho`
  (0.3391); SurPyval reports :math:`\alpha = e^{3.9901} = 54.06` and
  :math:`\beta = e^{0.3391} = 1.404`.
- **The coefficients have the opposite sign.** In lifelines (and R's
  ``survreg``) a positive AFT coefficient means a *longer* life; in
  SurPyval's AFT models the scale is :math:`\alpha e^{-b'Z}`, so a positive
  coefficient means a *shorter* life. Financial aid (``fin``) is +0.272 in
  lifelines and -0.272 here: both say aid lengthens the time to re-arrest,
  by a factor :math:`e^{0.272} = 1.31`.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _p = dict(zip(aft.parameter_names, aft.params))
    assert round(np.log(_p["alpha"]), 4) == 3.9901, _p
    assert round(np.log(_p["beta"]), 4) == 0.3391, _p
    assert round(_p["fin"], 4) == -0.2722, _p

Comparing groups
----------------

.. code-block:: python

    # lifelines
    from lifelines.statistics import logrank_test
    aid = rossi["fin"] == 1
    logrank_test(rossi.week[aid], rossi.week[~aid],
                 rossi.arrest[aid], rossi.arrest[~aid])
    # test_statistic 3.8376, p 0.0501

SurPyval's ``logrank`` takes one array of times and one of group labels,
for any number of groups:

.. jupyter-execute::

    sp.logrank(rossi["week"], rossi["fin"], rossi["c"])

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _lr = sp.logrank(rossi["week"], rossi["fin"], rossi["c"])
    assert round(float(_lr.statistic), 4) == 3.8376, _lr.statistic
    assert round(float(_lr.p_value), 4) == 0.0501, _lr.p_value

Time-varying covariates
-----------------------

The Stanford heart transplant data in start-stop form, one row per
patient before a transplant and one after:

.. code-block:: python

    # lifelines
    from lifelines import CoxTimeVaryingFitter
    from lifelines.datasets import load_stanford_heart_transplants

    heart = load_stanford_heart_transplants()
    ctv = CoxTimeVaryingFitter().fit(heart, id_col="id", event_col="event",
                                     start_col="start", stop_col="stop")
    ctv.params_    # age 0.0272, year -0.1463, surgery -0.6372, transplant -0.0103

.. jupyter-execute::

    heart = sp.datasets.load_heart_transplants()
    heart["c"] = 1 - heart["event"]
    tv = sp.CoxPH.fit_tvc_from_df(
        heart, i_col="id", xl_col="start", xr_col="stop", c_col="c",
        Z_cols=["age", "year", "surgery", "transplant"],
    )
    tv.beta

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(tv.beta, [0.027167, -0.146346, -0.637210, -0.010251], atol=1e-6), tv.beta

Numbers at risk
---------------

lifelines adds the table under a plot with ``add_at_risk_counts``. In
SurPyval, ``plot_nonparametric`` draws it (``at_risk_rows="all"``), and
``at_risk_table`` gives the numbers without a plot:

.. jupyter-execute::

    table = sp.at_risk_table(km, times=[0, 200, 400, 600, 800])
    table["counts"]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _c = table["counts"]["Kaplan-Meier"]["At risk"]
    assert list(_c) == [228, 144, 57, 24, 8], _c

These are the patients still under observation at each time (time
:math:`\geq t`), as R's ``summary(survfit(...), times = ...)`` reports them.

Differences that catch people out
---------------------------------

- **The censoring flag** (above): ``c = 1 - event``. The bundled data keep
  their original coding, so it is ``c = 1 - df["arrest"]`` and
  ``c = 1 - df["status"]``.
- **AFT signs** (above): SurPyval's AFT coefficients are lifelines' negated.
  Proportional hazards coefficients have the same sign in both (positive =
  higher hazard).
- **Fitted models are returned, not stored.** ``KaplanMeier.fit(...)``
  returns a model; the fitter itself (``sp.KaplanMeier``) keeps nothing, so
  there is no ``fitter.fit(...); fitter.survival_function_``.
- **Functions, not DataFrames.** ``km.sf(t)``, ``model.hf(t)`` and the
  rest take times and return arrays; ``km.x`` and ``km.R`` are the step
  points. A regression model takes the covariates as its second argument:
  ``cox.sf(52, Z)``.
- **Bounds.** ``alpha_ci=0.05`` gives a 95% two-sided interval, as
  lifelines' ``alpha=0.05``. ``bound="lower"`` or ``"upper"`` gives a
  one-sided one; ``method="lr"`` a likelihood-ratio bound and
  ``method="bootstrap"`` a bootstrap one, on almost every parametric model.
- **Weights and counts are different things.** lifelines' ``weights``
  serves both purposes. In SurPyval, ``n`` is a whole-number count of
  identical rows (and gives exactly the same answer as repeating them),
  while ``w`` is a sampling weight, with robust standard errors.
- **Interval and left censoring go in the same call.** Instead of
  ``fit_interval_censoring``, pass ``xl``/``xr`` (or ``c = 2`` with
  ``[left, right]`` pairs) to the same ``fit``. Kaplan-Meier refuses such
  rows and names ``Turnbull``, the estimator for them.

What SurPyval adds
------------------

Most of lifelines' models have a SurPyval counterpart. Beyond them,
SurPyval has, as of lifelines 0.30:

- every censoring and truncation type in one data model, in every fitter
  (left, right and interval censoring with left and right truncation, each
  row its own), and the Turnbull estimator for non-parametric work on such
  data;
- competing risks regression (Fine-Gray, cause-specific Cox, parametric
  per cause) and multi-state models;
- frailty and mixed-effects Cox models, recurrent events (mean cumulative
  function, Crow-AMSAA and other intensity models, Andersen-Gill and
  related regressions), joint models of repeated measurements and survival,
  and degradation models;
- the engineering methods: probability plotting and rank regression,
  offsets (three-parameter Weibull), accelerated life models with
  life-stress relationships, B-lives with likelihood-ratio and bootstrap
  bounds, and test planning;
- Bayesian fits with priors (``fit_bayes``).

lifelines has its own strengths that SurPyval does not try to copy:
clinical-style reporting (``print_summary`` tables and plotting helpers
that are ready for a paper), and a workflow built for penalised and
cross-validated modelling, with a scikit-learn adapter and
``k_fold_cross_validation``. SurPyval has the ``penalizer`` and
``l1_ratio`` arguments too, but no cross-validation tooling.
