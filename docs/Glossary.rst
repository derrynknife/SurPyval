Glossary
========

Survival analysis grew up separately in several fields, and each named the
same ideas in its own words. An engineer's *failure* is a clinician's
*event*; a *suspension* is a *right censored* observation; *reliability* is
*survival*. SurPyval is for anyone with time-to-event data, so these pages
use the words together where a method is general and keep a field's own word
where the method belongs to that field (reliability growth, test planning,
B-lives).

The API has one name for each thing, whatever the field: the censoring flag
is always ``c``, the survival function is always ``sf``. This page maps the
vocabularies to each other and to those names. It changes nothing in the
API.

The same idea in three vocabularies
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 22 24 32

   * - Engineering
     - Medicine
     - Statistics
     - In SurPyval
   * - failure
     - event (death, relapse, recovery)
     - event
     - a row with ``c = 0``
   * - unit, item, component
     - subject, patient, participant
     - unit, individual, observation
     - one row of data (or ``n`` identical rows)
   * - life, lifetime, time to failure
     - survival time, time to event
     - time to event, duration, failure time
     - ``x``
   * - suspension, survivor, unfailed unit
     - censored subject (alive at last follow-up, lost to follow-up,
       withdrawn)
     - right censored observation
     - a row with ``c = 1``
   * - found failed at the first inspection
     - event before the first visit
     - left censored observation
     - a row with ``c = -1``
   * - failed between inspections; inspection data
     - event between visits
     - interval censored observation
     - a row with ``c = 2`` and ``x = [lower, upper]``
   * - units already in service when records began
     - delayed entry, late entry
     - left truncation
     - ``tl`` (or ``t``)
   * - reliability, :math:`R(t)`
     - survival, survival probability, :math:`S(t)`
     - survival function
     - ``sf``
   * - unreliability, probability of failure, :math:`F(t)`
     - cumulative incidence (with one kind of event)
     - cumulative distribution function (CDF)
     - ``ff``
   * - failure rate, hazard rate
     - hazard, force of mortality
     - hazard function
     - ``hf``
   * - cumulative hazard
     - cumulative hazard
     - cumulative hazard function
     - ``Hf``
   * - B10 life (the time by which 10% have failed)
     - 10th percentile of the survival time; the median is the "B50"
     - quantile
     - ``qf``
   * - mean time to failure (MTTF), mean life
     - mean survival time
     - mean, expectation
     - ``mean``
   * - mean life over a mission or warranty period
     - restricted mean survival time (RMST)
     - restricted mean
     - ``rmst``
   * - mean residual life, remaining useful life
     - life expectancy at a given time
     - mean residual life
     - ``mrl``
   * - life given survival so far, conditional reliability
     - conditional survival
     - conditional survival function
     - ``cs``
   * - stress, operating condition, design factor
     - risk factor, treatment, prognostic factor
     - covariate, explanatory variable
     - ``Z``
   * - failure mode
     - cause of death, type of event
     - competing risk, cause
     - ``e``
   * - repairable system; repeated failures
     - recurrent events (repeated admissions, infections)
     - recurrent events; counting process
     - ``i`` identifies the unit; :doc:`Recurrent Event Analysis`
   * - mean cumulative number of failures
     - mean cumulative count of events
     - mean cumulative function (MCF)
     - ``mcf``
   * - rate of occurrence of failures (ROCOF)
     - event rate
     - intensity function
     - ``iif``
   * - population that never fails; limited failure population
     - cured fraction; cure model
     - defective distribution
     - ``lfp=True``
   * - dead on arrival
     - event at time zero
     - zero-inflation
     - ``zi=True``
   * - failure-free period, minimum life
     - guarantee time (no event can happen before it)
     - threshold, location shift
     - ``offset=True``
   * - degradation measurement, condition reading
     - biomarker, longitudinal measurement
     - longitudinal data, repeated measures
     - ``y``; :doc:`Degradation Analysis`, :doc:`Joint Models`

The censoring flag
------------------

SurPyval's flag ``c`` says where the event lies relative to the recorded
value: ``0`` at it (the event was observed), ``1`` to its right (right
censored), ``-1`` to its left (left censored) and ``2`` between two values
(interval censored). Clinical data sets, and some other software, more often
carry an *event indicator* or *status* column in which ``1`` means the event
happened and ``0`` means censored, which is the reverse for the two commonest
codes. Convert such a column before fitting:

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    weeks = [6, 13, 21, 30, 31, 37, 38, 47, 49, 50]
    event = np.array([1, 1, 1, 1, 0, 1, 0, 1, 0, 0])  # 1 = relapsed

    c = 1 - event           # SurPyval: 0 = event, 1 = right censored
    model = surv.KaplanMeier.fit(weeks, c)
    print(model.sf(26))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert c.tolist() == [0, 0, 0, 0, 1, 0, 1, 0, 1, 1], c
    assert np.isclose(model.sf(26), 0.7), model.sf(26)

For a table, ``SurpyvalData.from_dataframe`` does the conversion itself when
the status column is named with ``event_col`` in place of ``c_col`` (see
:doc:`Data Wrangler Examples`). :doc:`Conventions` defines the flags, and
every other input, exactly.

Words that belong to one field
------------------------------

Some terms have no everyday counterpart elsewhere, because the method was
developed for one field's problem. They are used as they are:

- **Reliability growth**, **Crow-AMSAA**, **Duane**: models of how the rate
  of failures of a repairable system changes as a design is developed. See
  :doc:`Recurrent Event Analysis`.
- **Demonstration test**, **success-run test**, **accelerated life test
  plan**: planning how many units to test, and for how long, before a test
  is run. See :doc:`Test Planning with SurPyval`.
- **Accelerated life**, **life-stress relationship** (Arrhenius, Eyring,
  inverse power law): regression in which a physical stress shortens life by
  a known law. See :doc:`regression analysis`.
- **Probability plot** (Weibull plot): axes scaled so that a chosen
  distribution is a straight line. See :doc:`Parametric Estimation`.
- **Log-rank test**, **hazard ratio**, **cumulative incidence**,
  **subdistribution hazard**: at home in medical statistics, and used with
  the same names for any kind of unit. See
  :doc:`Non-Parametric Estimation`, :doc:`regression analysis` and
  :doc:`Competing Risks Analysis`.

New to all of it? :doc:`Survival Analysis from Scratch` introduces these
ideas one at a time.
