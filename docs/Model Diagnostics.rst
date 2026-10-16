Model Diagnostics
=================

A fitted model is only as good as its assumptions. Every regression model
in SurPyval has one method that checks them, ``diagnose()``: it runs the
checks the model's assumptions call for and returns a report saying what
holds, what does not, and what to do about it.

.. code:: python

    d = model.diagnose(alpha=0.05)   # columns=[...] checks a subset
    d.table          # one row per check: test, statistic, p, verdict, covariates
    d.advice         # plain-English remedies, each naming the SurPyval feature
    d.plot()         # the visual checks (matplotlib is imported only here)
    d.ok             # True if nothing failed
    d.raise_if_failed()   # for pipelines
    d.to_dict()      # machine-readable, strict JSON

The name is ``diagnose``, not ``check_assumptions``: besides the
assumptions it looks at the fit, the functional form and the rows that
drive the estimates.

Families with diagnostics so far: **Cox proportional hazards**. The other
models have the method too, and it refuses, naming the issue that tracks
their checks: parametric proportional hazards and accelerated failure time
models (#869), proportional odds and additive hazards (#870), accelerated
life models (#871), frailty (#872), recurrent events (#873), competing
risks (#874) and the univariate distributions (#875); Buckley-James's
has no issue of its own yet (#867).

Verdicts
--------

Each row of ``d.table`` is a test or a screen, with a verdict:

- **fail**: a test whose p-value is below ``alpha``: evidence, at that
  level, against the assumption;
- **warn**: a test whose p-value is borderline, at least ``alpha`` but
  below ``2 * alpha``, or that could not be computed on these data (its
  ``p`` is ``nan`` and its ``detail`` says why); or a screen (influence,
  outliers) that found something worth a look;
- **ok**: otherwise.

A screen is not a test and never fails. ``d.ok`` is True when no row
failed, so warnings do not stop a pipeline. Every row is judged at
``alpha`` on its own: with a dozen rows some fail by chance about
``alpha`` of the time each, even when every assumption holds, so read a
lone failure with a p-value near ``alpha`` with that in mind.

Cox proportional hazards
------------------------

The Rossi recidivism data follow 432 prisoners for a year after release;
``arrest`` is 1 for an arrest, so the censoring flag is ``1 - arrest``.

.. jupyter-execute::

    import numpy as np
    import pandas as pd
    import surpyval as surv
    from surpyval.datasets import load_rossi_static

    rossi = load_rossi_static()
    rossi['censored'] = 1 - rossi['arrest']
    covariates = ['fin', 'age', 'race', 'wexp', 'mar', 'paro', 'prio']
    model = surv.CoxPH.fit_from_df(rossi, x_col='week', c_col='censored',
                                   Z_cols=covariates)
    d = model.diagnose()
    d.ok

.. jupyter-execute::

    d.table

Two covariates fail the proportional-hazards test: ``age`` on the
Kaplan-Meier scale of time and ``wexp`` (work experience) on both. The
advice depends on the kind of covariate: ``wexp`` is binary, so it can be
stratified on; ``age`` is continuous, so its effect can be allowed to
change with time instead.

.. jupyter-execute::

    for advice in d.advice:
        print('-', advice, '\n')

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _failed = d.table[d.table['verdict'] == 'fail']
    _single = {c for c in _failed['covariates'] if len(c) == 1}
    assert _single == {('age',), ('wexp',)}, _failed
    assert len(rossi) == 432
    assert not d.ok

Each piece of advice also says, for a program, which check prompted it and
which SurPyval features it names:

.. jupyter-execute::

    d.advice[0].check, d.advice[0].covariates, d.advice[0].features

The plots show what the tests test. The scaled Schoenfeld residuals of a
covariate estimate its coefficient at each failure time, so under
proportional hazards they scatter around a flat line at the fitted
coefficient; the lowess trend (with its bootstrap band) shows how the
effect drifts:

.. jupyter-execute::

    fig = d.plot('proportional_hazards', random_state=1)

``d.plot()`` draws every visual check: also the martingale residuals
against each continuous covariate (functional form), each row's
influence on each coefficient, the deviance residuals and the Cox-Snell
residuals against the unit exponential.

.. jupyter-execute::

    fig = d.plot(['functional_form', 'outliers', 'overall_fit'])

Following the advice
~~~~~~~~~~~~~~~~~~~~

For ``age``, the first remedy the advice names is a time-varying effect:
split each prisoner's year at a time (here week 20), and fit
:meth:`CoxPH.fit_tvc <surpyval.univariate.regression.proportional_hazards.cox_ph.CoxPH_.fit_tvc>`
with one ``age`` column for each period. Each period then has its own
coefficient:

.. jupyter-execute::

    cut = 20
    rows = []
    for i, r in rossi.iterrows():
        z = [r[c] for c in covariates if c != 'age']
        if r['week'] <= cut:
            rows.append([i, 0, r['week'], r['censored'], *z, r['age'], 0])
        else:
            rows.append([i, 0, cut, 1, *z, r['age'], 0])
            rows.append([i, cut, r['week'], r['censored'], *z, 0, r['age']])
    split = pd.DataFrame(
        rows,
        columns=['i', 'start', 'stop', 'censored',
                 *[c for c in covariates if c != 'age'],
                 'age_to_20', 'age_after_20'],
    )
    tv = surv.CoxPH.fit_tvc_from_df(
        split, i_col='i', xl_col='start', xr_col='stop', c_col='censored',
        Z_cols=[c for c in covariates if c != 'age']
        + ['age_to_20', 'age_after_20'],
    )
    tv.summary().loc[['age_to_20', 'age_after_20'], ['coef', 'se(coef)', 'p']]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _s = tv.summary()
    assert abs(_s.loc['age_to_20', 'coef']) < 0.05 < _s.loc['age_to_20', 'p']
    assert _s.loc['age_after_20', 'coef'] < -0.05
    assert _s.loc['age_after_20', 'p'] < 0.01

Age makes little difference to the chance of arrest in the first twenty
weeks, and older prisoners are markedly less likely to be arrested after
that: the single coefficient of the first fit averaged the two. The split
model's rows belong to prisoners, several rows each, and its own report
says so, offering standard errors that allow for it:

.. jupyter-execute::

    [a for a in tv.diagnose().advice if a.check is None]

For ``wexp``, stratify: ``CoxPH.fit_from_df(..., strata_col='wexp')``
gives each level its own baseline hazard. (``diagnose()`` does not yet
handle a stratified fit; it says so.)

The checks
~~~~~~~~~~

``proportional_hazards``
    For each covariate, whether its coefficient changes with time: the
    score test of adding :math:`Z_j g(t)` to the model, on two scales of
    time :math:`g`, ``km`` (one minus the left-continuous Kaplan-Meier
    estimate) and ``rank`` (the rank of each time); and the joint
    ``GLOBAL`` test on each. A covariate fails if it fails on either
    scale. These are the tests of R's ``survival::cox.zph`` (version 3.0
    and later) and agree with it to the precision of the fit, under
    Efron and Breslow ties, delayed entry and time-varying covariates;
    :meth:`~surpyval.univariate.regression.semi_parametric_regression_model.SemiParametricRegressionModel.check_ph`
    gives the same test on one scale at a time. (lifelines, and R before
    version 3.0, compute Grambsch and Therneau's approximation of it,
    whose statistics differ a little in small samples.)
``functional_form``
    For each covariate with three or more distinct values, the score test
    of adding its square: whether it acts linearly on the log hazard (a
    two-valued covariate's effect is linear by construction). R reports
    the same statistic as ``coxph(..., init = c(coef(fit), 0), iter.max =
    0)$score``.
``influence``
    For each covariate, the largest ``|dfbeta| / se``: how many standard
    errors the coefficient moves when one row is dropped (the one-step
    approximation). It warns when one row moves a coefficient by more
    than one standard error.
``outliers``
    The deviance residuals. A row is an outlier when its ``|deviance
    residual|`` exceeds the two-sided Bonferroni normal cutoff for the
    number of observations :math:`N`, :math:`z_{1 - \alpha / (2N)}`
    (3.66 for 200 observations at :math:`\alpha = 0.05`), so that a model
    that fits flags a row about :math:`\alpha` of the time whatever
    :math:`N`. It warns when any row is flagged.
``overall_fit``
    A plot only: the Nelson-Aalen cumulative hazard of the Cox-Snell
    residuals against the line :math:`y = x` of the unit exponential.
    For a Cox model the residuals sum to the number of events by
    construction, so there is no useful test of them.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    from scipy.stats import norm
    assert round(norm.isf(0.05 / (2 * 200)), 2) == 3.66

A covariate with two distinct values is *binary*, one with three to five
*categorical*, one with more *continuous*. A binary or categorical
covariate that fails proportional hazards is advised stratification
(``strata=``, or ``strata_col=`` from a DataFrame; for a formula's coded
level, stratification on the variable it codes); a continuous one a
time-varying effect through ``CoxPH.fit_tvc``, binning and stratifying, or
a parametric accelerated failure time model (``WeibullAFT``,
``LogNormalAFT``). A covariate that fails its functional form is advised
a transform or bins in a formula. When the rows are grouped by subject (a
``fit_tvc`` model) the report offers the cluster-robust standard errors of
``summary(robust=True)``; a model without clusters says it has none.

What the Cox checks support
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every tie method (Efron's and Breslow's exactly; with ``"exact"`` and
``"kalbfleisch-prentice"`` the residuals use Breslow's risk sets, as R's
``cox.zph`` does, and the report notes it), delayed entry (``tl``), counts
(``n``) and time-varying covariates (``fit_tvc``). ``diagnose()`` refuses,
saying what to do: a stratified fit (not yet supported), a model restored
with ``from_dict`` (it has no data to check), one with an aliased
coefficient and one whose fit has no finite maximum.

Choosing what to check
----------------------

``columns`` checks some covariates only, named as ``summary()`` names
them; the joint tests still cover all of them. ``alpha`` sets the level:

.. jupyter-execute::

    small = model.diagnose(alpha=0.01, columns=['age', 'wexp'])
    small.table[['check', 'transform', 'covariates', 'p', 'verdict']]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert small.ok
    assert set(small.table['verdict']) <= {'ok', 'warn'}

In a pipeline, ``raise_if_failed()`` raises
:class:`~surpyval.diagnostics.report.DiagnosticsFailed` (whose
``diagnosis`` is the report) when a check failed, and returns the report
otherwise; ``to_dict()`` gives the report as strict JSON for an app or an
agent, and :meth:`Diagnosis.from_dict
<surpyval.diagnostics.report.Diagnosis.from_dict>` reads it back.

.. jupyter-execute::

    import json
    try:
        model.diagnose().raise_if_failed()
    except surv.diagnostics.DiagnosticsFailed as error:
        print(str(error).splitlines()[0])
    print(json.dumps(small.to_dict()['table'][0], indent=1))

Adding a family
---------------

Each family is one module in ``surpyval/diagnostics`` that lists its
checks and the advice rules for their failures; the report, the verdicts,
the advice engine, ``columns``, ``alpha`` and the plot layout are shared,
and the models already have the method. The step-by-step guide is the
docstring of :mod:`surpyval.diagnostics`, and
``surpyval/diagnostics/cox.py`` is the family to copy.

API
---

.. automodule:: surpyval.diagnostics.report
    :members: Diagnosis, Advice, DiagnosticsFailed, verdict

.. automodule:: surpyval.diagnostics.registry
    :members: Family, Check, AdviceRule, Panel, PlotOptions, Context, register, diagnose

.. autofunction:: surpyval.univariate.regression.proportional_hazards.diagnostics.check_ph

.. autofunction:: surpyval.univariate.regression.proportional_hazards.diagnostics.added_covariate_score_test

Parametric proportional hazards and accelerated failure time
------------------------------------------------------------

A parametric regression of kind proportional hazards (``WeibullPH``,
``LogNormalPH``, ..., ``PH(dist)``) or accelerated failure time
(``WeibullAFT``, ..., ``AFT(dist)``), fitted with ``fit`` or
``fit_from_df``, has two families of checks: those of its covariates'
effects (proportional hazards, or one shape for every level) and those of
its baseline distribution. Every test is a likelihood-ratio test of the
fitted model against a larger one, with the fit's own likelihood (every
kind of censoring and truncation it took), so its p-value is exact for
the model.

The Rossi data again, with a Weibull proportional hazards model:

.. jupyter-execute::

    ph = surv.WeibullPH.fit_from_df(
        rossi, x_col='week', c_col='censored',
        Z_cols=['fin', 'age', 'wexp', 'prio'],
    )
    d = ph.diagnose()
    d.table[['check', 'covariates', 'statistic', 'df', 'p', 'verdict']]

.. jupyter-execute::

    for advice in d.advice:
        print('-', advice, '\n')

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _failed = d.table[d.table['verdict'] == 'fail']
    assert {c for c in _failed['covariates'] if len(c) == 1} == {('age',)}
    assert d.advice[0].features == ('WeibullPH.fit_tvc', 'LogNormalAFT.fit')

As for the Cox model, ``age`` breaks proportional hazards: its
coefficient, fitted in each quarter of the failure times, is near 0 for
the first half of the arrests and about -0.15 after. The baseline fits:
no other built-in baseline has a markedly lower AIC, and the Cox-Snell
residuals lie on the line of the unit exponential.

.. jupyter-execute::

    fig = d.plot(['proportional_hazards', 'baseline_distribution'])

An accelerated failure time model of R's lung cancer data, with a
log-normal error distribution:

.. jupyter-execute::

    from surpyval.datasets import load_lung

    lung = load_lung().dropna(subset=['ph.ecog'])
    lung['censored'] = 1 - lung['status']
    aft = surv.LogNormalAFT.fit_from_df(
        lung, x_col='time', c_col='censored', Z_cols=['age', 'sex', 'ph.ecog']
    )
    d = aft.diagnose()
    d.table[['check', 'test', 'covariates', 'statistic', 'p', 'verdict']]

.. jupyter-execute::

    for advice in d.advice:
        print('-', advice, '\n')

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _warned = d.table[d.table['check'] == 'error_distribution']
    assert (_warned['verdict'] == 'warn').all()
    assert [a.features for a in d.advice] == [
        ('WeibullAFT.fit',), ('LogNormalPH.fit',)
    ]

The log-normal is the wrong error distribution: the early deaths'
standardised residuals bend away from its line, and the Weibull's AIC is
29 lower.
``ph.ecog``'s one patient with a score of 3 is too few to estimate a
shape for that level, and the row says so. Following the advice:

.. jupyter-execute::

    fig = d.plot('error_distribution')

.. jupyter-execute::

    weibull = surv.WeibullAFT.fit_from_df(
        lung, x_col='time', c_col='censored', Z_cols=['age', 'sex', 'ph.ecog']
    )
    weibull.diagnose().table[['check', 'covariates', 'statistic', 'p', 'verdict']]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _d = weibull.diagnose()
    assert _d.ok and not _d.advice

The checks of a proportional hazards model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``proportional_hazards``
    For each covariate, the likelihood-ratio test of a step in its
    coefficient at the median failure time :math:`\tau`: the model with
    hazard :math:`h_0(t) \exp(\beta'Z + [t > \tau](g_0 + \gamma Z_j))`
    against the one with :math:`\gamma = 0` (1 df); and the joint
    ``GLOBAL`` test of a step in every coefficient (one df per covariate).
    Both models step the level of the baseline at :math:`\tau` too
    (:math:`g_0`), so the test does not depend on where a covariate's
    origin is. Drawn as each covariate's coefficient in each quarter of
    the failure times, with its Wald interval. For a Weibull or
    exponential PH model the statistics agree with R's to 1e-6 (a Poisson
    model of the follow-up split at :math:`\tau`, whose likelihood is the
    same), with delayed entry too.
``functional_form``
    For each covariate with three or more distinct values, the
    likelihood-ratio test of adding its square (R's ``survreg(... +
    I(z^2))`` against ``survreg(...)``, to 1e-6), and the martingale
    residuals against the covariate.
``influence`` and ``outliers``
    The screens of the Cox family: the largest one-step ``|dfbeta| / se``
    of each coefficient (the observed information times each unit's score,
    R's ``residuals(survreg(...), type = 'dfbeta')``) warns above 1, and
    a deviance residual beyond the Bonferroni normal cutoff warns.
``baseline_distribution``
    Two screens. The AIC of the model against that of each other built-in
    baseline (``ExponentialPH``, ``NormalPH``, ``WeibullPH``,
    ``GumbelPH``, ``LogisticPH``, ``LogNormalPH``, ``GammaPH``) refitted
    to the same data, as ``fit_best`` compares distributions; and against
    the accelerated failure time model of the same baseline (not for the
    Weibull and the exponential, whose PH and AFT models are one model).
    Each warns when the other's AIC is more than 2 lower; the
    ``statistic`` is the fitted model's AIC less the other's. Drawn as the
    Cox-Snell residuals on exponential probability paper, and the AIC of
    each candidate.

The checks of an accelerated failure time model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``error_distribution``
    The time on the baseline's scale, :math:`e^{\beta'Z} x`, follows the
    baseline distribution: drawn on its probability paper with the fitted
    line (for the Weibull and the log-normal, the standardised residuals
    of R's ``survreg``). The same two AIC screens: the other built-in error
    distributions (``WeibullAFT``, ``LogNormalAFT``, ...), and the
    proportional hazards model of the same baseline.
``common_shape``
    For each categorical covariate (two to five distinct values, or the
    dummy columns a formula codes one factor with), the likelihood-ratio
    test of one shape for every level against a shape per level (the
    Weibull's ``beta``, the log-normal's ``sigma``, the gamma's
    ``alpha``; ``L - 1`` df for ``L`` levels), which is R's ``survreg(...
    + strata(g))`` against ``survreg(...)`` for the Weibull and the
    log-normal (to 1e-6, with interval censoring too); drawn as each
    level's standardised residuals on the baseline's paper. The
    exponential has no shape to test, and a level with fewer than two
    distinct failure times has too few to estimate one (the test is then
    not computed, and warns).
``functional_form``
    As for the proportional hazards model, of log time.

The advice: a time-varying effect (``fit_tvc``), an AFT model or, for a
binary or categorical covariate, a stratified Cox model for a covariate
that breaks proportional hazards; each level fitted on its own, or a Cox
model stratified on it, for a shape that varies by level; a transform or
bins in a formula for a covariate that is not linear; the better baseline,
or the PH or AFT model, that the AIC screens found; and a look at an
influential or badly predicted row.

What these checks support
~~~~~~~~~~~~~~~~~~~~~~~~~

Every kind of censoring, delayed entry and right truncation, counts
(``n``), ``center=True``, formulas, and any baseline with the
``exp(beta'Z)`` link (``PH(dist)``, ``AFT(dist)``; the AIC comparison
then sets the fitted baseline against the built-in ones). The martingale
and deviance residuals are of the exact and right-censored rows that are
not right truncated (the report says how many it left out); the tests use
every row. ``diagnose()`` refuses, saying what to do: a fit to
time-varying covariates (``fit_tvc``, not yet supported, #869), a
custom covariate link, a model restored with ``from_dict``, one with an
aliased coefficient, and one whose fit is not a verified maximum (the
tests compare it with larger models). The accelerated life models (#871)
and the proportional odds and additive hazards models (#870) have their
own families.

Proportional odds and additive hazards
--------------------------------------

The proportional odds models (the semi-parametric ``ProportionalOdds``
and the parametric ``PO(...)`` family: ``WeibullPO``, ``LogNormalPO``,
...) assume that each covariate multiplies the odds of survival by the
same factor at every time; the additive hazards models (Lin and Ying's
semi-parametric ``AdditiveHazards`` and the parametric ``AH(...)``
family: ``WeibullAH``, ``ExponentialAH``, ...) that each adds the same
amount to the hazard at every time. Both say that an effect does not
change with time, and ``diagnose()`` tests that, covariate by covariate.
On the Rossi data:

.. jupyter-execute::

    po = surv.ProportionalOdds.fit_from_df(
        rossi, x_col='week', c_col='censored',
        Z_cols=['fin', 'age', 'wexp', 'prio'],
    )
    d = po.diagnose()
    d.table

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _failed = d.table[d.table['verdict'] == 'fail']
    assert {c for c in _failed['covariates'] if len(c) == 1} == {('age',)}
    assert d.family == 'Proportional odds'

The effect of ``age`` changes over the year, as it does in the Cox model.
The advice names models that do not assume proportional odds:

.. jupyter-execute::

    for advice in d.advice:
        print('-', advice, '\n')

For a binary or few-levelled covariate the plot is the Kaplan-Meier
log-odds of failure, :math:`\log(F / S)`, of each group against log time,
which proportional odds makes parallel curves; for a continuous one it is
the covariate's coefficient in each period of follow-up, refitted with a
column per period, against the constant the model fitted:

.. jupyter-execute::

    fig = d.plot()

The additive hazards model of the same data tells the same story:

.. jupyter-execute::

    ah = surv.AdditiveHazards.fit_from_df(
        rossi, x_col='week', c_col='censored',
        Z_cols=['fin', 'age', 'wexp', 'prio'],
    )
    d = ah.diagnose()
    d.table[['check', 'test', 'covariates', 'statistic', 'p', 'verdict']]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _failed = d.table[d.table['verdict'] == 'fail']
    assert {c for c in _failed['covariates'] if len(c) == 1} == {('age',)}
    assert d.family == 'Additive hazards'
    assert any('WeibullAH.fit_tvc' in a for a in d.advice)

Its plots are the cumulative regression coefficients of Aalen's
nonparametric additive model, :math:`B_j(t) = \int_0^t b_j(s)\,ds` in
:math:`h(t \mid Z) = b_0(t) + b(t)' Z`, with a pointwise band: under the
fitted model each is the straight line :math:`\beta_j t`, and where one
bends that effect changes.

.. jupyter-execute::

    fig = d.plot('constant_effects')

The advice for ``age`` is a time-varying effect, one ``age`` column per
period (as for the Cox model above), through the parametric additive
models' ``fit_tvc``:

.. jupyter-execute::

    print(d.advice[0])

The checks
~~~~~~~~~~

``proportional_odds`` (proportional odds models), ``constant_effects`` (additive hazards models)
    Follow-up is split into four periods at the quartiles of the event
    times, and :math:`g(t)` is the period a time is in (0 to 3; the notes
    give the cuts). For each covariate the model is refitted with the
    added term :math:`\theta\, u(Z)\, g(t)`, where :math:`u(Z)` is the
    covariate less its mean or, for a covariate with three to five
    values, the indicator of each level after the first, less its mean;
    the test is of :math:`\theta = 0`, the same coefficient in every
    period. For a binary or categorical covariate of a proportional odds
    model that is whether the log-odds of its groups stay parallel. The
    ``GLOBAL`` row adds every covariate's term at once.

    The added covariate changes at the cut times, which every one of
    these estimators fits exactly by splitting each row into one row per
    period, entering at the period's start, so the refit is the model's
    own estimator. For the proportional odds models and the parametric
    additive ones the test is the likelihood-ratio test of the two fits
    (for ``ProportionalOdds``, of its profile likelihood). For Lin and
    Ying's model, which has no likelihood, it is the Wald test of
    :math:`\theta` from the closed-form estimating equations with Lin
    and Ying's sandwich variance; R's ``timereg::aalen``, with ``const()``
    terms on the split rows, gives the same statistic. It is the test
    that ``cox.zph`` makes for a Cox model (there the score test of
    :math:`Z\, g(t)`, which is a trend test of the Schoenfeld residuals),
    with the period as the clock: a step clock makes the extended model an
    exact fit of the family's own estimator, and cuts at the event-time
    quartiles make the test the same whatever the unit of time.

    :math:`u(Z)` is centred because a proportional odds or parametric
    additive model cannot absorb a function of time alone into its
    baseline: centred, the added term leaves an average unit as the model
    has it. For a parametric model the test also reacts to a baseline
    distribution that does not fit the data, which shows as an effect that
    drifts.

A covariate that fails in a proportional odds model is advised a model
that does not assume proportional odds: ``CoxPH.fit``, ``WeibullAFT.fit``
or ``LogNormalAFT.fit`` for ``ProportionalOdds``; for a parametric one the
same baseline as proportional hazards or accelerated failure time
(``WeibullPH.fit`` or ``WeibullAFT.fit`` for ``WeibullPO``; ``PH(dist)``
or ``AFT(dist)`` for a baseline with no pre-built fitter). One that fails
in an additive hazards model is advised a time-varying effect, one column
per period, through ``WeibullAH.fit_tvc`` (``AdditiveHazards`` has no
time-varying fit), or the parametric model's own ``fit_tvc``.

What these checks support
~~~~~~~~~~~~~~~~~~~~~~~~~

Counts (``n``), delayed entry (``tl``), ``fit_from_df`` and formulas,
``center=True``, fixed parameters and, for the parametric models,
time-varying covariates (``fit_tvc``). ``diagnose()`` refuses, saying
what to do: a model restored with ``from_dict`` (it has no data), one
with an aliased coefficient, one whose fit has no finite maximum and, for
now, a parametric model fitted to left- or interval-censored or
right-truncated data, whose rows cannot be split at the cut times
(#870). The functional form, influence and outliers are not checked for
these models yet. A parametric additive hazards fit often ends on the
boundary of its support (as its fit warns); a refit that does not reach a
verified maximum is noted, and its test is then approximate. The tests
refit the model once per covariate and once for the ``GLOBAL`` row, which
for a parametric additive hazards model can take some seconds.

.. autofunction:: surpyval.diagnostics.odds_additive.aalen_cumulative_coefficients

Accelerated life (life-stress) models
-------------------------------------

An accelerated life model, ``AcceleratedLife(dist, life_model).fit(x,
Z)``, assumes two things about how failure times change with stress: the
distribution keeps **one shape at every stress level** (only its life
moves), and its life **follows the life model** :math:`L(Z)` (Arrhenius,
Eyring, the inverse power law, ...) from one level to the next. A test run
at a few stress levels can check both, because the distribution can also
be fitted at each level on its own. Neither can be checked at the use
stress, so the report also looks at what the test can say about the
extrapolation there.

Nelson's insulating fluid (*Accelerated Testing*, 1990) broke down under
seven voltages, from 26 to 38 kV; the times are in minutes:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv
    from surpyval import life_models

    fluid = {
        26: [5.79, 1579.52, 2323.70],
        28: [68.85, 108.29, 110.29, 426.07, 1067.60],
        30: [7.74, 17.05, 20.46, 21.02, 22.66, 43.40, 47.30, 139.07,
             144.12, 175.88, 194.90],
        32: [0.27, 0.40, 0.69, 0.79, 2.75, 3.91, 9.88, 13.95, 15.93, 27.80,
             53.24, 82.85, 89.29, 100.58, 215.10],
        34: [0.19, 0.78, 0.96, 1.31, 2.78, 3.16, 4.15, 4.67, 4.85, 6.50,
             7.35, 8.01, 8.27, 12.06, 31.75, 32.52, 33.91, 36.71, 72.89],
        36: [0.35, 0.59, 0.96, 0.99, 1.69, 1.97, 2.07, 2.58, 2.71, 2.90,
             3.67, 3.99, 5.35, 13.77, 25.50],
        38: [0.09, 0.39, 0.47, 0.73, 0.74, 1.13, 1.40, 2.38],
    }
    x = np.concatenate([np.array(v) for v in fluid.values()])
    kv = np.concatenate([[float(k)] * len(v) for k, v in fluid.items()])

    weibull = surv.AcceleratedLife(surv.Weibull, life_models.Power).fit(x, kv)
    d = weibull.diagnose()
    d

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _t = d.table
    _shape = _t[_t['check'] == 'common_shape'].iloc[0]
    _lof = _t[_t['check'] == 'life_stress'].iloc[0]
    assert d.ok and not d.advice
    assert (round(_shape['statistic'], 2), _shape['df'], round(_shape['p'], 3)) == (8.79, 6, 0.185)
    assert (round(_lof['statistic'], 2), _lof['df'], round(_lof['p'], 2)) == (2.34, 5, 0.8)
    _aic = _t[_t['check'] == 'life_model_comparison'].set_index('life_model')['statistic']
    assert round(_aic['Power'] - _aic['GeneralLogLinear'], 2) == 0.56
    assert round(weibull.params[1], 3) == 0.777 and round(weibull.params[3], 1) == -17.7

The Weibull with the inverse power law (Nelson's model: shape 0.777,
exponent -17.7) passes. The seven shapes do not differ more than chance
allows (p = 0.185), and the lives at the seven voltages follow the power
law (p = 0.80). Of the other life models, the exponential law in the
voltage (``GeneralLogLinear``) has an AIC lower by 0.56 -- no substantial
difference -- and the linear law fits far worse. These two
likelihood-ratio tests agree with R's ``survreg`` to 1e-5.

With the LogNormal instead, the shape (``sigma``) is not common to the
levels:

.. jupyter-execute::

    lognormal = surv.AcceleratedLife(surv.LogNormal, life_models.Power).fit(x, kv)
    d_ln = lognormal.diagnose()
    print(d_ln.advice[0])

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _shape = d_ln.table[d_ln.table['check'] == 'common_shape'].iloc[0]
    assert _shape['verdict'] == 'fail' and round(_shape['p'], 3) == 0.010
    assert d_ln.advice[0].features == ('AcceleratedLife', 'LogNormal.fit')

The plots show what the tests test: a probability plot of every level,
each with its own fit (solid) and the model's (dashed), whose solid lines
are parallel when the shape is common; and each level's life against the
stress, with the life model's curve and the best alternative's:

.. jupyter-execute::

    fig = d.plot()

The use stress
~~~~~~~~~~~~~~

The model has no notion of the stress it will be used at, so the report
screens the extrapolation only when it is told: ``use_stress`` is an
option of this family's ``diagnose()``, a number for a one-stress model
and one per stress column otherwise. Nelson's design voltage is 20 kV:

.. jupyter-execute::

    d20 = weibull.diagnose(use_stress=20)
    rows = d20.table[d20.table['check'].isin(['life_model_comparison', 'extrapolation'])]
    rows[['check', 'life_model', 'test', 'statistic', 'verdict', 'detail']]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _use = d20.table[d20.table['test'] == 'use stress vs tested range'].iloc[0]
    assert round(_use['statistic'], 2) == 0.69 and _use['verdict'] == 'ok'
    _power = float(np.ravel(weibull.phi([[20.0]]))[0])
    _gll = surv.AcceleratedLife(surv.Weibull, life_models.GeneralLogLinear).fit(x, kv)
    _exp_law = float(np.ravel(_gll.phi([[20.0]]))[0])
    assert round(_power, -3) == 125000 and round(_exp_law, -2) == 25500
    assert round(_power / _exp_law) == 5

20 kV is 0.69 of the tested range's width below it, on the log scale the
power law is straight on: not far, so the row is ok. But the comparison
rows show what the data cannot settle: the power law puts the
characteristic life at 20 kV near 125,000 minutes, and the exponential law,
whose AIC is as good, near 25,500 -- five times shorter, from two models
the tests cannot tell apart. A use stress more than one width outside the
tested range warns, and the advice is to quote the life with its bounds
(``cb``), compare the plausible life models there, and test nearer the
use stress if possible:

.. jupyter-execute::

    print(weibull.diagnose(use_stress=10).advice[0])

.. jupyter-execute::

    fig = d20.plot('life_stress')

The checks
~~~~~~~~~~

A *stress level* is a distinct row of ``Z``. A level enters the two tests
when it has at least two failures (and, for the shape test, when the
distribution fitted to it alone reaches a maximum); a note names any left
out.

``common_shape``
    The likelihood-ratio test of each level fitted on its own (its own
    shape and life) against one shape shared by every level, each keeping
    its own life: :math:`2(\sum_i \ell_i - \ell_0)` on :math:`(k - 1)q`
    degrees of freedom, for :math:`k` levels and :math:`q` free shape
    parameters (a parameter the fit held with ``fixed`` stays held). The
    life model plays no part in it. A distribution with no shape (the
    Exponential) has no row. With few failures a level (ten or so), the
    test rejects a common shape somewhat more often than ``alpha``: the
    maximum likelihood estimate of a shape is biased in small samples, so
    read a failure near ``alpha`` with that in mind.
``life_stress``
    The likelihood-ratio test of the life model against a separate life
    at each level (with the common shape): :math:`2(\ell_0 - \ell)` on
    :math:`k - m` degrees of freedom, :math:`m` the life model's free
    parameters -- the lack-of-fit test of the relationship. With no more
    levels than parameters (two levels and a two-parameter life model)
    the model passes through every level's life and there is nothing to
    test: the row warns and says so.
``life_model_comparison``
    A screen: each built-in life model for the model's number of stresses
    (one stress: ``Power``, ``Exponential`` -- the Arrhenius model --,
    ``Eyring``, ``InverseEyring``, ``Linear``, ``GeneralLogLinear``; two:
    ``DualPower``, ``DualExponential``, ``PowerExponential``,
    ``GeneralLogLinear``), refitted with the same distribution to the same
    data, with its AIC. A model with an AIC lower than the fitted one's by
    more than 2 warns. ``InversePower`` and ``InverseExponential`` are
    ``Power`` and ``Exponential`` under another parametrisation, with the
    same likelihood, so only one of each pair is compared. A life model
    is a physical claim as well as a curve: prefer a better AIC only where
    its physics suits the stress.
``extrapolation``
    A screen: for each stress column, the number of levels tested. Two or
    fewer warn: a two-parameter life model then passes through each
    level's life, so nothing in the data checks its form, and the life
    at the use stress rests on the form alone. With ``use_stress``, also
    the use stress's distance outside the tested range, in widths of the
    range, on the scale the life model's log-life is linear in (``log``
    for a power term, ``1 / stress`` for an Arrhenius or Eyring one, the
    stress itself for ``Linear`` and ``GeneralLogLinear``); more than 1
    warns. The comparison rows then give each life model's life at the
    use stress.

With a continuous stress (each unit at its own stress) no level has two
failures: the two tests cannot be computed, their rows warn and say so,
and the comparison of life models, which needs no levels, still runs.
Group the stresses into levels to test them.

The advice: for a shape that differs between levels, to look for a
different failure mode at the extreme levels and refit without them, and
to fit each level on its own (``Weibull.fit``, say) to see the shapes; for
a life model that does not follow the levels, the best alternative by AIC
(``AcceleratedLife(dist, life_models.<name>)``), or, if none is better, a
life model of your own (a ``LifeModel`` subclass) or a second stress; for
a better alternative alone, to refit with it and compare the two models'
lives at the use stress; for a use stress far outside the tested range,
caution (its bounds, the other life models' lives there, testing nearer
it); and for two levels, to test at three or more.
SurPyval has no accelerated-test planning tool, so the advice suggests
simulating a candidate plan with the fitted model's ``random(size, Z=...)``
and refitting, to see how wide the bounds (``cb``) at the use stress
would be.

``diagnose()`` refuses a model restored with ``from_dict`` (it has no
data) and one whose fit has no finite maximum. Every kind of censoring,
truncation, counts and a ``fixed`` parameter are supported.

Recurrent events
----------------

A repairable system fails, is repaired and fails again. Its fitted model
is one of three kinds, and each has its own family of checks:

- a **Poisson process** (``HPP``, ``CrowAMSAA``, ``Duane``, ``CoxLewis``):
  minimal repair, every item with the one intensity;
- an **imperfect-repair** model (``GeneralizedRenewal``,
  ``GeneralizedOneRenewal``, ``ARA``, ``ARI``), whose restoration
  parameter places the repairs between as good as new and as bad as old;
- a **proportional-intensity** model (``ProportionalIntensityHPP``,
  ``ProportionalIntensityNHPP``): a Poisson process whose intensity each
  item's covariates scale.

Eight pumps are each watched for 100 days, and their failures come ever
faster. A constant rate of failures, an HPP, is the simplest model:

.. jupyter-execute::

    rng = np.random.default_rng(1)
    x, i, c = [], [], []
    for pump in range(8):
        # a power law: the expected number of failures by day t is (t / 25)^2
        times = np.sort(100 * rng.random(rng.poisson(16)) ** 0.5)
        x += [*times, 100]
        i += [pump] * (len(times) + 1)
        c += [0] * len(times) + [1]   # the last row ends the observation
    x, i, c = np.array(x), np.array(i), np.array(c)

    hpp = surv.HPP.fit(x, i, c)
    d = hpp.diagnose()
    print(d)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _t = d.table.set_index('test')
    assert (_t.loc[_t['check'] == 'trend', 'verdict'] == 'fail').all()
    assert not d.ok
    assert d.advice[0].startswith('The rate of failures increases')

The three trend tests reject a constant rate, and the advice names the
models whose rate changes. The power law, ``CrowAMSAA``, passes:

.. jupyter-execute::

    crow = surv.CrowAMSAA.fit(x, i, c)
    d = crow.diagnose()
    d.table[['check', 'test', 'statistic', 'p', 'verdict']]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert d.ok and not d.advice
    assert set(d.table['verdict']) == {'ok'}

For a model that lets the rate change, a trend is no fault, so the trend
rows are a model-choice screen (they are ok here: a trend was found). The
plots show the data's mean cumulative function against a constant rate,
the transformed event times the fit test uses (a straight line when the
model fits) and each pump's count against the one it expects:

.. jupyter-execute::

    fig = d.plot()

Mettas and Zhao's six systems, fitted with a Kijima generalized renewal
process, show the repair check: the likelihood-ratio tests of perfect
repair (``q = 0``, as good as new) and of minimal repair (``q = 1``, as
bad as old). Neither is an assumption of the model, so they are a screen
too: a row warns when its repair fits as well, and the advice names the
simpler model.

.. jupyter-execute::

    from surpyval.datasets import load_mettas_and_zhao

    mz = load_mettas_and_zhao()
    grp = surv.GeneralizedRenewal.fit(mz['x'], mz['i'], mz['c'])
    d = grp.diagnose()
    d.table[['check', 'test', 'p', 'verdict']]

.. jupyter-execute::

    for advice in d.advice:
        print('-', advice, '\n')

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _r = d.table[d.table['check'] == 'repair']
    assert list(_r['verdict']) == ['warn', 'warn'] and d.ok
    assert 0.3 < _r['p'].iloc[0] < 0.31 and 0.46 < _r['p'].iloc[1] < 0.47

With 33 failures the data cannot tell perfect from minimal repair (q is
0.106), and nothing fails. ``d.plot('repair')`` draws the profile
log-likelihood of ``q`` with both values marked:

.. jupyter-execute::

    fig = d.plot('repair')

The recurrent checks
~~~~~~~~~~~~~~~~~~~~

``trend``
    The Laplace and MIL-HDBK-189C tests, whose null is a homogeneous
    Poisson process (a constant rate), and the Lewis-Robinson test, whose
    null is any renewal process: the Laplace statistic divided by the
    coefficient of variation of the times between failures. The first two
    are the model's own ``trend_test``; each item is tested on its own
    window. For Lewis-Robinson each item runs from its first renewal (its
    start if it was observed from new, else its first failure) to its last
    failure, so that it is centred under any renewal process. For a model
    with a constant rate (``HPP``, ``ProportionalIntensityHPP``) these are
    tests of it; for the others a screen, whose row warns when its test
    does not reject its simpler null. A trend against a constant rate that
    Lewis-Robinson does not confirm points to a renewal process whose
    times between failures are not exponential (advised
    ``GeneralizedRenewal.fit``); no trend at all, to a constant rate
    (``HPP.fit``, ``ProportionalIntensityHPP.fit``).
``fit``
    The model's ``cramer_von_mises`` test, with 99 bootstrap refits from
    a fixed seed, so a report is the same each time it runs;
    ``model.cramer_von_mises(n_boot=..., random_state=...)`` gives a finer
    p-value. A failure is advised another intensity, another repair model
    or another baseline, as the model calls for.
``between_items``
    (Poisson processes.) The Pearson dispersion test of the items' event
    counts against those the model expects over their windows, with
    :math:`m - 1 - p` degrees of freedom for :math:`m` items and
    :math:`p` covariate coefficients. Items that differ by more than chance
    fail it, and are advised a proportional-intensity model with what
    tells them apart as covariates (or, for one, more covariates). An item
    observed up to its last failure has a fixed count and is left out.
``repair``
    (Imperfect repair.) The model's ``repair_test``: likelihood-ratio
    tests of perfect repair and of minimal repair (ARI: maximal repair,
    ``rho = 1``; G1 has no minimal repair), with the 50:50 mixture of
    chi-squared(0) and chi-squared(1) for a value on the edge of the
    parameter's range. A row warns when its repair is not rejected.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    from surpyval.diagnostics import recurrent as _rec
    assert _rec.N_BOOT == 99

``diagnose()`` refuses, saying what to do, a model built from its
parameters or restored with ``from_dict`` (it has no data), a least-squares
(``how='MSE'``) fit, one with no finite maximum, and data with interval-
or left-censored rows or right truncation. Every recurrent check is about
the model as a whole: ``columns=`` is accepted for a proportional-intensity
model and changes nothing.

Competing risks
---------------

A competing-risks regression has a ``diagnose()`` for each of its two
models (#874), reported cause by cause: the table has a ``cause`` column
(the cause's label, as text) and the plots are titled by cause. The PBC
trial followed 312 patients with primary biliary cirrhosis until a liver
transplant, death or the end of follow-up:

.. jupyter-execute::

    from surpyval.datasets import load_pbc2

    pbc = load_pbc2().groupby('id').first()
    pbc['cause'] = pbc['status'].map({'alive': None, 'transplanted': 'transplant',
                                      'dead': 'death'})
    pbc['drug'] = (pbc['drug'] == 'D-penicil').astype(int)
    pbc['female'] = (pbc['sex'] == 'female').astype(int)
    pbc['log_bili'] = np.log(pbc['serBilir'])
    cr_covariates = ['drug', 'age', 'log_bili', 'female']
    cs = surv.CompetingRisksProportionalHazards.fit_from_df(
        pbc, x_col='years', e_col='cause', Z_cols=cr_covariates)
    d = cs.diagnose()
    print(d)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert d.ok and not d.advice
    assert set(d.table['cause']) == {'death', 'transplant'}
    assert any('Tsiatis' in note for note in d.notes)

The cause-specific model (``model="Cox"``, the default) is one Cox model
per cause with the other causes censored; each is kept, as
``cs.models[cause]``, and its rows are exactly the Cox checks above on it
(proportional hazards on the km and rank scales with a GLOBAL row per
cause, functional form, influence, outliers and the Cox-Snell plot), with
the same rules. Here both causes pass (age is borderline in the
functional form of death, a warning). A cause that fails proportional
hazards is advised to be fitted on its own, ``CoxPH.fit(x, Z,
c=np.where(e == cause, 0, 1), strata=...)`` for a binary or few-levelled
covariate, or with a time-varying effect (``CoxPH.fit_tvc``) for a
continuous one, since ``CompetingRisksProportionalHazards`` takes no
strata; and, where the question is the incidence of a cause rather than
its rate, the Fine-Gray model.

The Fine-Gray model (``model="Fine-Gray"``, or ``FineGray.fit`` for one
cause) assumes proportional *subdistribution* hazards instead, and its
report tests that: the Grambsch-Therneau score test of adding ``Z * g(t)``
to the weighted partial likelihood, the censoring weights held as fitted,
on the ``cif`` scale (``g`` the cause's Aalen-Johansen cumulative incidence
just before ``t``) and the ``rank`` scale, with GLOBAL rows; and the
functional form, by the score test of a covariate's square. It is R's
``cox.zph`` on the weighted counting-process data whose weighted ``coxph``
fit is the Fine-Gray fit, and agrees with it to 1e-6 (the score tests use
the model-based information, as ``FineGray``'s standard errors do).
Influence, outliers and overall fit are not checked for it.

.. jupyter-execute::

    fg = surv.CompetingRisksProportionalHazards.fit_from_df(
        pbc, x_col='years', e_col='cause', Z_cols=cr_covariates,
        model='Fine-Gray')
    d_fg = fg.diagnose()
    d_fg.table.loc[d_fg.table['verdict'] == 'fail',
                   ['check', 'cause', 'transform', 'covariates', 'p']]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _failed = d_fg.table[d_fg.table['verdict'] == 'fail']
    assert ('death', ('log_bili',)) in set(zip(_failed['cause'], _failed['covariates']))
    assert not d_fg.ok

The same data pass the cause-specific checks and fail the Fine-Gray ones
for death: bilirubin's effect on the *incidence* of death is not constant
in time. That is no contradiction. A covariate that raises the hazard of
one cause and not of another cannot have constant ratios of both the
cause-specific and the subdistribution hazards, so the two models rarely
hold exactly together; which to use depends on the question (every
report notes it), the cause-specific model for aetiology (how a covariate
changes a cause's rate) and the Fine-Gray model for prediction (how it
changes a cause's incidence).

.. jupyter-execute::

    for advice in d_fg.advice:
        print('-', advice, '\n')

.. jupyter-execute::

    fig = d_fg.plot('proportional_hazards', random_state=1)

Every competing-risks report also notes what no check can test: the
independence of the causes (of their latent failure times) is not
identifiable from competing-risks data (Tsiatis 1975). The cause-specific
hazards and the cumulative incidences do not need it; a net survival,
one cause's with the others removed, does.

A model restored with ``from_dict`` has no data to check and refuses, as
do one with an aliased coefficient and one whose fit of a cause has no
finite maximum (the message names the cause). ``ParametricCompetingRisks``
refuses for now, naming #874: its checks are each cause's goodness of fit,
which waits for the univariate checks (#875). The Fine-Gray tests and
residuals are also functions, in
``surpyval.univariate.competing_risks.regression.fine_gray_diagnostics``:

.. autofunction:: surpyval.univariate.competing_risks.regression.fine_gray_diagnostics.fine_gray_zph_test

.. autofunction:: surpyval.univariate.competing_risks.regression.fine_gray_diagnostics.fine_gray_added_covariate_test

.. autofunction:: surpyval.univariate.competing_risks.regression.fine_gray_diagnostics.fine_gray_martingale_residuals

Univariate distributions
------------------------

A distribution fitted with ``Weibull.fit(...)`` (or any other continuous
distribution, with an offset, ``lfp``, ``zi``, fixed parameters or
truncation as fitted, by any ``how``) has ``diagnose()`` too. It asks two
questions: does the distribution fit the data, and does another model the
advice can name fit them better?

Eighty units whose failures start only after time 50, fitted with a
2-parameter Weibull:

.. jupyter-execute::

    import surpyval as surv

    x = 50 + surv.Weibull.random(80, 20, 1.5, random_state=3)
    model = surv.Weibull.fit(x)
    d = model.diagnose(random_state=1)
    d.table[['check', 'alternative', 'test', 'statistic', 'p', 'verdict']]

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _gof = d.table[d.table['check'] == 'goodness_of_fit']
    assert (_gof['verdict'] == 'fail').all(), d
    _alt = d.table[d.table['check'] == 'alternatives']
    _best = _alt.loc[_alt['statistic'].idxmax()]
    assert _best['alternative'] == 'Weibull offset', d
    assert _best['statistic'] > 10

The three goodness-of-fit tests fail, and three alternatives have an AIC
more than 2 below the Weibull's; the offset (3-parameter) Weibull most of
all. The advice names each remedy, and says which fits best:

.. jupyter-execute::

    for advice in d.advice:
        print('-', advice, '\n')

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert any('Weibull.fit(offset=)' in a.features and 'fits best' in a
               for a in d.advice)

The P-P plot shows what the tests measure, the Kaplan-Meier estimate of
the fitted CDF at each time against the uniform; beside it are the
refits' Anderson-Darling statistics, with the observed one, and the
probability plot, whose points bend away from the line at the start:

.. jupyter-execute::

    fig = d.plot()

Following the advice, the offset model fits, and nothing the report can
name beats it:

.. jupyter-execute::

    offset = surv.Weibull.fit(x, offset=True)
    print(offset.diagnose(n_boot=100, random_state=1))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert offset.diagnose(n_boot=100, random_state=1).ok
    assert 45 < offset.gamma < 50

The checks
~~~~~~~~~~

``goodness_of_fit``
    The Anderson-Darling, Kolmogorov-Smirnov and Cramer-von Mises
    statistics. Each unit's time is put through the fitted CDF,
    conditioned on its truncation window,
    :math:`u = (F(x) - F(t_l)) / (F(t_r) - F(t_l))`, which is uniform when
    the model is right (and right-censored for a censored unit); the
    statistics compare the Kaplan-Meier estimate :math:`F_n` of the
    :math:`u` with the uniform up to the largest :math:`u`:
    :math:`\sup |F_n(u) - u|`, :math:`N \int (F_n(u) - u)^2 du` and
    :math:`N \int (F_n(u) - u)^2 / (u (1 - u)) du`. With complete,
    untruncated data they are the classical statistics, and agree with R's
    ``fitdistrplus::gofstat`` to 1e-9.

    The parameters are estimated, and the data may be censored, so the
    textbook tables do not apply: the p-values come from a parametric
    bootstrap. Each of ``n_boot`` resamples (200 by default) draws a time
    for every unit from the fitted model within its truncation window,
    censors it as the data were censored -- a censored unit at its own
    censoring time, a failed one at a time drawn from the Kaplan-Meier
    estimate of the censoring distribution beyond its failure, Davison
    and Hinkley's conditional bootstrap, which reproduces fixed-time and
    random censoring -- refits the same model (distribution, ``how``,
    offset, ``lfp``, ``zi``, fixed parameters) and computes its
    statistics. The p-value is the share of refits at least as extreme,
    :math:`(1 + k) / (1 + B)`. For complete data from a location-scale
    family, whose null distribution does not depend on the parameters,
    the bootstrap reproduces the tables: its 5% points are D'Agostino and
    Stephens's, within Monte Carlo error, and the test is exact. With
    censoring it is somewhat conservative in small samples: of 400
    randomly censored exponential samples of 40, 5-9% had a p-value of at
    most 0.1 (two runs). Left- and interval-censored
    data have no such test (a resample would need the inspection times,
    which the data do not record): those rows warn and say so.
``probability_plot``
    A plot only: ``model.plot()``, the failures' plotting positions
    against the fitted line on the distribution's linearising axes.
``alternatives``
    An AIC screen: the best other distribution of ``fit_best``; a mixture
    of two of the model's distribution (for a model without options, with
    20 failures or more); the model with an offset; with a limited
    failure population (when some units are right-censored); and a
    zero-inflated model (when some units failed at time 0). A row warns
    when the alternative's AIC is at least 2 below the model's, and never
    fails; its ``statistic`` is the model's AIC less the alternative's.
    Each warning is advice naming the fit: ``fit_best(...,
    return_table=True)``, ``MixtureModel.fit(..., dist=..., m=2)``,
    ``Weibull.fit(..., offset=True)``, ``lfp=True`` or ``zi=True``. A
    failed test that no alternative explains is advised a look at the
    probability plot, more mixture components or a non-parametric
    estimate (``KaplanMeier.fit``, ``Turnbull.fit``).

Seeds and run time
~~~~~~~~~~~~~~~~~~

``diagnose(n_boot=..., random_state=...)`` sets the bootstrap's size and
seed (``None`` draws from numpy's global generator, as everywhere in
SurPyval); the same seed gives the same report. A refit of a
2-parameter distribution to 100 units takes a few hundredths of a second,
so the default 200 refits take a few seconds; an offset, ``lfp`` or
``zi`` model's refits are slower, and fewer refits (``n_boot=100``) are
still a valid test, with p-values in steps of 1/101.

``diagnose()`` refuses, saying what to do: a model built with
``from_params`` or restored without its data, a fit with no finite
maximum, a discrete distribution (not yet, #875), and the event-probability
models (``Bernoulli``, ``FixedEventProbability``) and ``ExactEventTime``,
which have no distribution of times to test. A ``MixtureModel`` is a
model of its own, without ``diagnose()`` yet.

.. autofunction:: surpyval.diagnostics.univariate.gof_statistics

.. autofunction:: surpyval.diagnostics.univariate.bootstrap
