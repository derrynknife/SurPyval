
The data model
==============

Every fitter in SurPyval reads data the same way. Each row says what was
seen of one unit (an item, a patient, a customer: the :doc:`Glossary` maps
the words) over one stretch of its life, and a few letters describe the
row: ``x`` when the row ends, ``c`` what is known at ``x``, ``n`` how many
identical units it stands for, and so on. Only ``x`` is needed; every other
letter has a default. A ``SurpyvalData`` holds the rows, checks them, and
can be passed to any fitter: ``Weibull.fit(data)``, ``CoxPH.fit(data)``,
``CrowAMSAA.fit(data)``, ``MultiState.fit(data)``. A model that cannot use
part of the data refuses it and says which model can, rather than dropping
it.

This page is the reference for the letters and their rules, with one worked
example for each shape of data. :doc:`Types of Data` explains censoring and
truncation, the ideas behind ``c`` and ``t``; :doc:`Conventions` covers the
other conventions (the xrd format, missing values, units, seeds, saving).

.. jupyter-execute::

    import numpy as np
    import surpyval as surv

    data = surv.SurpyvalData(x=[12, 31, 45, 45, 60], c=[0, 0, 1, 1, 0])
    print(data)
    print(surv.Weibull.fit(data).params)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert data.n.sum() == 5 and data.n_items == 5

The letters
-----------

.. list-table::
   :header-rows: 1
   :widths: 10 18 40 18 14

   * - Letter
     - Name
     - In plain words
     - Allowed values
     - Default
   * - ``x``
     - time
     - When the row ends: the time the event (a failure, a death) happened,
       or the last time the unit was seen. Usually a time, but any quantity
       with a distribution works (a distance, a number of cycles, a stress).
     - finite numbers; blank on a row given as a range
     - required (or ``xl`` and ``xr``)
   * - ``xl``, ``xr``
     - range
     - The time is known only to lie between two values: last seen working
       at ``xl``, found failed at ``xr``.
     - numbers, ``-inf`` or ``inf`` at an open end; blank on a row given
       in ``x``
     - none
   * - ``c``
     - censoring flag
     - What is known at ``x``: 0 the event happened at ``x``; 1 the unit
       was still running (still alive, still in its state) at ``x``; -1 the
       event happened before ``x``; 2 it happened within the range.
     - 0, 1, -1, 2
     - 0; read from the range on a range row, and from ``e`` when ``s`` is
       given
   * - ``n``
     - count
     - How many identical units the row stands for.
     - whole numbers, at least 1
     - 1
   * - ``t`` (``tl``, ``tr``)
     - truncation, entry
     - The window in which the unit could have been seen at all:
       ``tl`` the age it entered the study (late entry, units already in
       service), ``tr`` the age past which it could not have been seen.
       Where a unit has several rows, ``tl`` is also where each row starts
       on the unit's clock.
     - numbers, ``-inf`` and ``inf`` for no limit
     - ``-inf`` and ``inf`` (no truncation)
   * - ``i``
     - unit
     - Which unit the row belongs to. ``ids`` is its other name (one
       argument, one list).
     - labels (text or numbers), all of one kind
     - each row its own unit
   * - ``e``
     - event type, state entered
     - What kind of event happened at ``x``: the failure mode (cause of
       death), or with ``s`` the state the unit moved into.
     - labels; blank (None, NaN, empty text) where nothing happened
     - none
   * - ``s``
     - state
     - The state the unit was in during the row, from ``tl`` to ``x``
       (multi-state data).
     - labels, kept as given
     - one starting state for every row
   * - ``a``
     - terminal flag
     - Whether the event at ``x`` ended the unit's history for good (1:
       scrapped, written off, died) or the unit carried on (0: repaired,
       relapsed and recovered).
     - 0, 1, True, False, blank
     - 1 on a unit's last event, 0 on an earlier one
   * - ``m``
     - measurements
     - Readings taken on a unit over its life (a vibration level, a wear
       depth, a biomarker), on their own timeline, linked to the rows by
       ``i``.
     - a ``Measurements`` object
     - none
   * - ``Z``
     - covariates
     - What the row's unit was like, or exposed to, during the row (a
       stress, an operating condition, a treatment, a risk factor).
     - a number per row and covariate
     - none
   * - ``w``
     - weight
     - How many units of the population each unit of the row stands for
       (a sampling, propensity or censoring weight). See
       :doc:`Weighted Data`.
     - positive finite numbers
     - none (every unit counts once)

From a table, ``SurpyvalData.from_dataframe`` reads each letter from a
column named by ``<letter>_col`` (``x_col``, ``xl_col``, ``xr_col``,
``c_col``, ``n_col``, ``tl_col``, ``tr_col``, ``i_col``, ``e_col``,
``s_col``, ``a_col``, ``w_col``, and ``Z_cols`` for several covariates);
``event_col`` reads a status column in which 1 marks the event. ``m`` is a
table of its own (``Measurements.from_dataframe``). See
:doc:`Data Wrangler Examples`.

The rules
---------

Each rule below is checked when the data is built, so a mistake in the data
is found there, with the rows named, rather than inside a fit.

A time on every row: x, or xl and xr
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A row has either an exact time in ``x`` or a range in ``xl`` and ``xr``,
with the other cells blank (None, NaN or pandas' ``NA``). A table often has
one column of exact times and two of ranges; give all three, and each row is
read on its own. A range is read by its bounds:

====================================  ==============================  ===
row                                   read as                         c
====================================  ==============================  ===
``x = v``, ``xl``, ``xr`` blank       exact at v (``c`` as given)     0
``xl = v``, ``xr = inf``              right censored at v             1
``xl = -inf``, ``xr = v``             left censored at v              -1
``xl = v1``, ``xr = v2``              interval censored, (v1, v2]     2
``xl = xr = v``                       exact at v                      0
====================================  ==============================  ===

- A range from 0, ``(0, v)``, stays an interval. The data does not know the
  distribution's support, so it does not guess left censoring from a 0.
- A row that gives both an exact time and a range cannot be read either
  way. It is **left out with a warning** that says how many rows and which,
  and its entries in every other per-row letter go with it. It is never
  reconciled. If every row is left out, the data is refused.
- A row with no time at all is refused, naming the row. So is a range with
  one end blank, except in ``SurpyvalData.from_dataframe`` and the
  competing-risks ``fit_from_df``, which read a blank end as ``-inf`` or
  ``inf``, as they always have.
- A ``c`` given must agree with each range (2 is also taken on any range
  with different ends); a blank ``c`` on a range row is read from the range.

Passing only ``x``, or only ``xl`` and ``xr``, works as it always has. In
``x`` alone, an interval may also be written as a pair ``[lower, upper]``
with ``c = 2``.

.. jupyter-execute::

    import warnings

    nan, inf = float("nan"), float("inf")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = surv.SurpyvalData(
            x=[3, 4, nan],
            xl=[nan, 2, 5],
            xr=[nan, 6, inf],
        )
    print(caught[0].message)
    print(data)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert len(caught) == 1 and "row 1" in str(caught[0].message)
    assert data.n.sum() == 2 and sorted(data.c.tolist()) == [0, 1]

What is known at x: c
~~~~~~~~~~~~~~~~~~~~~

``c`` defaults to 0, an event at ``x``, on every row given in ``x``. Some
data sets carry a status column with the opposite meaning (1 the event
happened); the :doc:`Glossary` shows how to convert one. With states ``s``,
``c`` may be left out: it is then read from ``e`` (a state entered is 0, a
blank is 1), and a ``c`` given must agree with ``e`` row by row. A row with
``c = 1`` and a state in ``e``, or a move with no state entered, is refused,
naming the rows.

Units with several rows: i, tl and the order of the rows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Without ``i`` every row is its own unit. With ``i``, a unit may have several
rows: stays in states, stretches with different covariates, or repeated
events. Each row is a stretch of the unit's own clock, from ``tl`` to
``x``; where ``tl`` is not given, a row starts where the unit's previous row
ended.

- A unit's rows come **in time order**: each starts at or after the
  previous row's ``x`` and ends after it. Rows are never reordered for you,
  since that would hide a mistake in a unit's history. A row that is out of
  order or overlaps the previous one is refused, naming the rows.
- Only events may share a time: simultaneous events, or the end of the
  record at an event's time.
- A unit counts once however many rows it has: ``data.n_items`` counts
  units, by the ``n`` of each unit's first row.

.. jupyter-execute::

    try:
        surv.SurpyvalData(x=[5, 2], c=[0, 0], i=[1, 1])
    except ValueError as error:
        print(error)

The end of a unit's history: a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``a`` says whether the event that ends a row ended the unit's history for
good. It is the same idea as entering an absorbing state in multi-state
data. What a unit's rows mean depends on how they end:

======================================================  ===============================================================
a unit's rows                                           what they are
======================================================  ===============================================================
rows with ``c = 1``, then one event with ``a = 1``      single-event life data with changing covariates
events with ``a = 0``, then more rows                   recurrent events (repairs, relapses; the unit carries on)
events with ``a = 0``, then an event with ``a = 1``     recurrent events ending in a terminal event
a last row with ``c = 1``                               still running at the end of the record
======================================================  ===============================================================

- **Default.** An event row with no ``a`` (or a blank cell) is read as 1 on
  its unit's last row and 0 on an earlier one, since the unit evidently
  carried on. ``data.a_inferred`` marks the rows read so; the recurrent fits
  say in their notes when they relied on it.
- **An** ``a`` **given always wins.**
- A ``c = 1`` row has no event, so ``a = 1`` there is refused.
- A row after a row with ``a = 1`` of the same unit is refused: nothing
  happens to a unit after its history has ended.
- Without ``a`` and with every unit on one row, ``data.a`` is None and the
  data is exactly what it was before ``a`` existed. With ``s``, ``a`` is
  checked but not inferred: the transitions say which states end a history.

.. jupyter-execute::

    try:
        surv.SurpyvalData(x=[2, 5, 9], c=[0, 0, 0], a=[0, 1, 0], i=[1, 1, 1])
    except ValueError as error:
        print(error)

Readings over time: m
~~~~~~~~~~~~~~~~~~~~~

Readings are not life events. They are taken when someone looks, not when
something happens, so they are kept apart from the rows in a
:class:`~surpyval.utils.measurements.Measurements` object and linked to the
rows by the unit ``i``. A reading is a time on the same clock as ``x`` and a
value; it has no censoring and no count (a reading not taken is left out).

- ``Measurements`` takes ``{i: {x: value}}`` for one measurement (unnamed,
  or named with ``name=``) or ``{name: {i: {x: value}}}`` for several. The
  shape is read from what the innermost dicts hold. Several measurements
  need not share times or units.
- ``Measurements.from_dataframe(df, i_col=, x_col=, m_col=, m_name_col=)``
  reads a long table with one row per reading; ``to_dataframe()`` writes one.
- Times and values are finite numbers, one reading per unit, measurement and
  time.
- Unit labels are **kept as typed**: ``"mm/s"`` prints and saves as
  ``"mm/s"``. Labels are only compared (``"h"`` and ``"hours"`` agree), never
  converted.
- ``m`` needs the rows' units, ``i``. A unit with readings but no rows is
  warned of, by count and id, and kept.
- Readings belong to a unit, not to a row: where a unit has several rows,
  its readings are kept while any of its rows is, and a fit is never handed
  them once per row.
- Indexing and every reshape keep the readings of the units kept;
  ``data.without("m")`` leaves them out; ``data.with_measurement(name)``
  keeps one.

``data.markers`` is the same store under its older name: the one
measurement's ``{i: {x: value}}``, as the joint models have always taken
it. Data with one unnamed measurement prints and saves exactly as before;
reading ``data.markers`` of data with several measurements is refused until
one is chosen with ``data.with_measurement(name)``.

One example of each shape
-------------------------

Single-event life data
~~~~~~~~~~~~~~~~~~~~~~

Each unit has one row: its failure (event) time, or the time it was last
seen still running. This is the shape every distribution, non-parametric
estimator and regression takes.

.. jupyter-execute::

    life = surv.SurpyvalData(
        x=[96, 145, 145, 210, 250, 250],
        c=[0, 0, 0, 1, 0, 1],
    )
    print(life)
    print(surv.KaplanMeier.fit(life).sf(200))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert life.n_items == 6 and life.n.tolist() == [1, 2, 1, 1, 1]

Exact times and ranges together
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Some units were seen to fail; others were found failed at an inspection, or
were still working at the last one. One column of exact times and two of
ranges hold them all:

.. jupyter-execute::

    mixed = surv.SurpyvalData(
        x=[310, nan, 520, nan, nan],
        xl=[nan, 200, nan, 400, 600],
        xr=[nan, 300, nan, 500, inf],
    )
    print(mixed.c)
    print(surv.Weibull.fit(mixed).params)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert mixed.c.tolist() == [2, 0, 2, 0, 1]

The flags were read from the ranges: (200, 300] and (400, 500] are interval
censored (2), and (600, inf) is right censored (1).

Recurrent events, and a terminal event: a
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Unit 1 was repaired at 2 and 5 and scrapped at 9; unit 2 was repaired at 4
and still running at 7. ``x`` is the unit's age at each event, not the time
since the previous one.

.. jupyter-execute::

    repairs = surv.SurpyvalData(
        x=[2, 5, 9, 4, 7], c=[0, 0, 0, 0, 1], i=[1, 1, 1, 2, 2]
    )
    print(repairs)
    print(repairs.summary())

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert repairs.a.tolist() == [0, 0, 1, 0, 0] and repairs.n_items == 2
    assert repairs.a_inferred.tolist() == [True, True, True, True, False]

``a`` was not given, so it was read: the events before a unit's last row are
0, and unit 1's last event is 1. The recurrent models in
``surpyval.recurrent`` take this data and say in their notes how they read
it. Apart from :class:`~surpyval.JointFrailty`, which models the terminal
event, they fit a terminal event as an event followed by the end of the
unit's observation:

.. jupyter-execute::

    print(surv.CrowAMSAA.fit(repairs))

A fit of one row per unit refuses this data, saying what it is and which
models take it:

.. jupyter-execute::

    try:
        surv.Weibull.fit(repairs)
    except ValueError as error:
        print(error)

Multi-state data: s and e
~~~~~~~~~~~~~~~~~~~~~~~~~

An illness-death study: patient A was well until 3, then ill until dying at
5; B was still well when last seen at 7; C died at 4 without falling ill.
Each stay in a state is a row; ``s`` is the state during the row and ``e``
the state entered at its end.

=======  ====  ===  ======  ======  ===
unit     tl    x    s       e       c
=======  ====  ===  ======  ======  ===
A        0     3    well    ill     0
A        3     5    ill     dead    0
B        0     7    well            1
C        0     4    well    dead    0
=======  ====  ===  ======  ======  ===

.. jupyter-execute::

    from surpyval.multistate import MultiState

    stays = surv.SurpyvalData(
        x=[3, 5, 7, 4],
        tl=[0, 3, 0, 0],
        s=["well", "ill", "well", "well"],
        e=["ill", "dead", None, "dead"],
        i=["A", "A", "B", "C"],
    )
    print(stays)
    print(MultiState.fit(stays))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert stays.c.tolist() == [0, 0, 1, 0]
    assert stays.n_items == 3

``c`` was left out, so it was read from ``e``. States are kept as given
(text or whole numbers, never turned into floats). Without ``s`` there is
one starting state: ordinary survival data is that case with one way out,
and competing-risks data is that case with several, the failure mode in
``e`` being the state entered. A fit that is not multi-state refuses data
in more than one state; ``stays[stays.s == "well"]`` keeps the stays in one
state, which fit as competing risks. See :doc:`Multi-State Models`.

Measurements: m
~~~~~~~~~~~~~~~

Wear readings on three units, attached to their lives. The degradation
models fit the readings against a failure threshold; the joint models fit
them together with the lives.

.. jupyter-execute::

    wear = surv.Measurements(
        {
            "wear": {
                "A": {0: 0.0, 10: 1.2, 20: 2.5, 30: 3.9},
                "B": {0: 0.0, 10: 0.8, 20: 1.9, 30: 2.6},
                "C": {0: 0.0, 10: 1.5, 20: 2.8},
            }
        },
        units={"wear": "mm"},
    )
    worn = surv.SurpyvalData(x=[35, 40, 25], c=[0, 1, 0], i=["A", "B", "C"], m=wear)
    print(worn.summary())
    print(surv.WienerProcess.fit(worn, threshold=5.0))

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert worn.m.names == ["wear"] and worn.m.units == {"wear": "mm"}
    assert worn.m.n_readings == 11

See :doc:`Degradation Analysis` and :doc:`Joint Models`.

Covariates: Z
~~~~~~~~~~~~~

One row per unit, with what each unit was like (here, whether it was
treated). A regression takes the covariates; a plain distribution refuses
them and names ``data.without("covariates")``.

.. jupyter-execute::

    treated = surv.SurpyvalData(
        x=[5, 8, 12, 3, 9, 15, 7, 11],
        c=[0, 0, 1, 0, 0, 1, 0, 1],
        Z=[[0], [0], [0], [1], [1], [1], [0], [1]],
    )
    print(surv.CoxPH.fit(treated).params)

Covariates that change: counting-process rows
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a covariate changes during a unit's life (a load raised, a treatment
started), split the unit's life at each change. Each stretch is a row with
the covariates fixed in it: ``c = 1`` ("still running when the covariate
changed"), and the next row starts there, ``tl`` equal to the previous
row's ``x``. The last row ends in the event or the end of the record.

Unit A ran under load 0 until 4, then under load 1 until it failed at 9;
B failed at 6 under load 0; C changed load at 3 and was still running at
10; and so on.

.. jupyter-execute::

    stretches = surv.SurpyvalData(
        x=[4, 9, 6, 3, 10, 5, 12, 8, 7, 11, 14],
        tl=[0, 4, 0, 0, 3, 0, 5, 0, 0, 7, 0],
        c=[1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 1],
        i=["A", "A", "B", "C", "C", "D", "D", "E", "F", "F", "G"],
        Z=[[0], [1], [0], [0], [1], [0], [1], [1], [0], [1], [0]],
    )
    print(stretches.a)
    model = surv.CoxPH.fit_tvc(
        i=stretches.i, xl=stretches.tl, xr=stretches.x,
        c=stretches.c, Z=stretches.Z,
    )
    print(model.params)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert stretches.n_items == 7
    assert np.isfinite(model.params).all()

``CoxPH.fit(stretches)`` refuses this data, since it would count each unit
once per row, and names the ``fit_tvc`` call above. The regressions'
``fit_tvc`` also takes a covariate's changes as ``{age: value}`` or a
``StepSchedule`` per unit (see :doc:`Data Wrangler Examples`).

Landmarks: tl = L
~~~~~~~~~~~~~~~~~

A landmark analysis asks about units that have lasted to a time ``L``,
using only what was known at ``L``. :func:`~surpyval.landmark` builds that
data from the life data: the units still at risk at ``L``, entered at ``L``
(``tl = L``, so the time before ``L`` is not counted), followed to
``L + horizon``, with their covariates (and their last reading of each
measurement) as known at ``L``. The result is ordinary data, so any model
fits it. See :doc:`Landmark Analysis`.

.. jupyter-execute::

    people = surv.SurpyvalData(
        x=[5, 8, 12, 3, 9, 15, 7, 11],
        c=[0, 0, 1, 0, 0, 1, 0, 1],
        i=list("ABCDEFGH"),
        Z=[[0], [0], [0], [1], [1], [1], [0], [1]],
    )
    at_six = surv.landmark(people, at=6, horizon=5)
    print(at_six)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.all(at_six.tl == 6) and at_six.x.max() == 11

Which models take what
----------------------

Every fitter's ``fit`` takes a ``SurpyvalData`` in place of the arrays.
What it does with each letter:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Models
     - Take
     - Refuse, and what to use instead
   * - Distributions and non-parametric estimators
       (:doc:`Parametric Estimation`, :doc:`Non-Parametric Estimation`)
     - ``x``, ``xl``/``xr``, ``c``, ``n``, ``t``; ``w`` for maximum
       likelihood and Kaplan-Meier, Nelson-Aalen and Fleming-Harrington
     - several rows per unit (the recurrent models, or a regression's
       ``fit_tvc``); ``Z`` (a regression); ``e`` (the competing-risks
       models); ``s`` (the multi-state models); ``m`` (the joint models)
   * - Regressions: Cox, PH, AFT, PO, additive hazards and others
       (:doc:`regression analysis`)
     - as above, plus ``Z``; ``w`` for the parametric regressions, Cox and
       Aalen's additive model
     - stretches of a unit (``fit_tvc``, as in the example above); repeated
       events (the recurrent models)
   * - Competing risks (:doc:`Competing Risks Analysis`)
     - as above, plus ``e``, the failure mode
     - several rows per unit; ``s``
   * - Recurrent events: MCF, Crow-AMSAA and the other intensity models,
       renewal, recurrent regressions (:doc:`Recurrent Event Analysis`)
     - ``x``, ``c``, ``n``, ``tl``, ``i`` with several rows per unit,
       gaps in observation as ``windows`` where the model has them, ``Z``
       and ``e`` where the model has them; ``a`` read as the end of the
       unit's observation, with a note
     - data without ``i``; ``s``; ``m``; ``w``; the panel-count models
       (``PanelCountMCF``, ``PanelCountRegression``) refuse a
       ``SurpyvalData`` altogether, since their rows are inspections with
       the count found, 0 included
   * - Joint frailty model (``JointFrailty``)
     - recurrent events with ``a``: ``a = 0`` a recurrent event, ``a = 1``
       the terminal event; ``Z``, ``tl``
     - ranges, left or interval censoring, counts, gaps, covariates that
       change within a unit, ``e``, ``s``, ``m``
   * - Multi-state models (:doc:`Multi-State Models`)
     - ``x``, ``s``, ``e``, ``tl``, ``i``, ``n``; ``Z`` for the two
       regressions; ``a`` is ignored (the transitions say which states
       absorb)
     - ``c`` of -1 or 2, right truncation, covariates that change, ``m``;
       several rows per unit without ``s``. The panel multi-state models
       take arrays of visits only
   * - Degradation models: ``WienerProcess``, ``GammaProcess``,
       ``DegradationAnalysis`` (:doc:`Degradation Analysis`)
     - ``m`` (a ``Measurements``, or data holding one) and a ``threshold``;
       ``Z`` as each unit's stress
     - counts, weights, truncation, ``e``, ``s``, several rows per unit;
       several measurements (``with_measurement(name)``)
   * - Joint models of readings and life (:doc:`Joint Models`)
     - one row per unit, with ``m`` (one measurement) linked by ``i``;
       ``Z``
     - units with several rows; several measurements
       (``with_measurement(name)``)
   * - Landmark data, :func:`~surpyval.landmark`
       (:doc:`Landmark Analysis`)
     - ``x``, ``c``, ``n``, ``tl``, ``i``, ``Z`` (fixed, or as stretches),
       ``m``
     - repeated events; gaps between a unit's rows

Every refusal is a ``ValueError`` (a ``TypeError`` where an argument is
given beside the data that holds it, or a model takes no ``SurpyvalData``)
that names what the data holds and what to do, so the table above is
also what the error messages say. A fit is never handed part of the data
silently.
