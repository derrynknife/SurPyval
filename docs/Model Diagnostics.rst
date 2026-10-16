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
