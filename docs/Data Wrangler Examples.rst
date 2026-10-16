
Data Wrangling Examples
=======================

Every SurPyval fitter takes data in the same form: values ``x``, censoring flags ``c``, counts ``n`` and truncation ``t`` (or ``tl`` and ``tr``), described in :doc:`Conventions`. Data rarely arrives that way. It comes as a list of failures and a list of survivors, as text with "+" marks, as a spreadsheet of install and removal dates, or as counts found at each inspection. This page shows how to get from each of these to something ``fit()`` accepts. The tool for it is :class:`~surpyval.SurpyvalData`: its constructors read each common layout, it validates the data exactly as the fitters do, and the parametric, non-parametric and regression ``fit`` methods take one in place of the arrays. The general recipe is always the same: decide, for each item, *what is known about its value* (exact, above a value, below a value, or between two values) and *whether it could have been missed altogether* (truncation), then write that down as ``x``, ``c``, ``n`` and ``t``.

Let's say we have a list of failures and a list of right censored (suspended) values. ``SurpyvalData.from_observations`` takes a list of each kind of observation, and the fitters take what it returns:

.. jupyter-execute::

    import surpyval as surv
    from surpyval import SurpyvalData

    # Failure data
    f = [2, 3, 4, 5, 6, 7, 8, 8, 9]
    # 'suspended' or right censored data
    s = [1, 2, 10]

    data = SurpyvalData.from_observations(observed=f, right_censored=s)
    print(data.x, data.c, data.n)

    model = surv.Weibull.fit(data)
    print(model)

A ``fit`` given a ``SurpyvalData`` takes everything from it -- values, flags, counts, truncation and, for a regression, the covariates ``Z`` -- so it refuses ``c``, ``n``, ``t`` (and the rest) given alongside it. ``data.to_xcnt()`` gives the arrays themselves, ``x, c, n, t``, where a function of your own needs them.


You can even bring in your left censored data as well:

.. jupyter-execute::

    # Failure data
    f = [2, 3, 4, 5, 6, 7, 8, 8, 9]
    # 'suspended' or right censored data
    s = [1, 2, 10]
    # left censored data
    l = [7, 8, 9]

    data = SurpyvalData.from_observations(
        observed=f, right_censored=s, left_censored=l
    )
    print(data.x, data.c, data.n)

    model = surv.Weibull.fit(data)
    print(model)


Another common type of data that is provided is in a simple text list with "+" indicating that the observation was censored at that point. Using some simple python list comprehensions can help.

.. jupyter-execute::

    # Example provided data
    data = "1, 2, 3+, 5, 6, 8, 10, 3+, 5+"

    f = [float(x) for x in data.split(',') if "+" not in x]
    s = [float(x[0:-1]) for x in data.split(',') if "+" in x]

    data = SurpyvalData.from_observations(observed=f, right_censored=s)

    model = surv.Weibull.fit(data)
    model

Again, this can be extended to left censored data as well:

.. jupyter-execute::

    data = "1, 2, 3+, 5, 6, 8, 10, 3+, 5+, 15-, 16-, 17-"
    split_data = data.split(',')

    f = [float(x) for x in split_data if ("+" not in x) & ("-" not in x)]
    s = [float(x[0:-1]) for x in split_data if "+" in x]
    l = [float(x[0:-1]) for x in split_data if "-" in x]

    data = SurpyvalData.from_observations(
        observed=f, right_censored=s, left_censored=l
    )

    model = surv.Weibull.fit(data)

Surpyval also offers the ability to use a pandas DataFrame as an input to every fitter. All you need to do is tell ``fit_from_df`` which columns to look at for x, c, n, and the truncation, tl and tr. Columns for c, n, tl and tr are optional, and tl and tr can also be given as a single number that applies to every row. Further, if you have interval censored data you can use the 'xl' and 'xr' column names instead. Any other ``fit`` option is passed straight to ``fit``, and the model is the one ``fit`` gives on the same arrays. The parametric distributions (``Weibull.fit_from_df``), the non-parametric estimators (``KaplanMeier.fit_from_df``), ``RoystonParmar`` and ``MixtureModel`` take the arguments below; the regression, competing-risks and recurrent-event fitters name their columns ``x_col``, ``c_col``, ``Z_cols``, ``e_col``, ``i_col``, ... (see :doc:`Conventions`).

.. jupyter-execute::

    import pandas as pd

    xr = [2, 4, 6, 8, 10]
    xl = [1, 2, 3, 4, 5]
    df = pd.DataFrame({'xl' : xl, 'xr' : xr})

    model = surv.Weibull.fit_from_df(df, xl_col='xl', xr_col='xr')
    print(model)

If you have mixed interval and observed or censored data, the two columns can still describe every row, provided no ``c`` column is given: a row with the same value in both columns is an exact observation, a row whose ``xr`` is infinite is right censored at ``xl``, and a row whose ``xl`` is minus infinity is left censored at ``xr``. This is exactly the shape of inspection data recorded as "last seen working" and "first seen failed":

.. jupyter-execute::

    import numpy as np

    df = pd.DataFrame({
        'last_ok':      [-np.inf, 150, 300, 420,    500, 610],
        'first_failed': [200,     150, 380, np.inf, 500, np.inf],
    })
    # row 0: failed before the first inspection at 200 -> left censored
    # rows 1 and 4: failure time known exactly         -> observed
    # row 2: failed between inspections at 300 and 380 -> interval censored
    # rows 3 and 5: never seen failed                  -> right censored
    data = SurpyvalData(xl=df['last_ok'], xr=df['first_failed'])
    print(data.c)

    model = surv.Weibull.fit_from_df(
        df, xl_col='last_ok', xr_col='first_failed'
    )
    print(model.params)

(``SurpyvalData`` validates its input exactly as every fitter does; building one yourself is a quick way to check how SurPyval has read your data. Its rows are sorted, so the flags come back in order of time.)


One record per item
-------------------

When each item's record carries more than a time and a flag (its own
observation window, what failed, its covariates), describe it with an
``ItemLifetime`` and build the data from the list. How many identical items
each one stands for goes in ``counts``:

.. jupyter-execute::

    from surpyval import ItemLifetime

    items = [
        ItemLifetime("B7", observed=1820, covariates=[1.0]),
        ItemLifetime("B8", right_censored=2400, covariates=[0.0]),
        ItemLifetime("B9", interval_censored=[1000, 1500], covariates=[1.0]),
        ItemLifetime("B10", observed=950, left_truncation=200, covariates=[0.0]),
    ]
    data = SurpyvalData.from_items(items, counts=[3, 40, 2, 5])
    print(data.n)
    surv.Weibull.fit(data.without("covariates")).params

Records are updated as items age, each update giving a new, checked record
(the old one is left as it was), and copied for a new item with ``clone``.
A record is fixed once made, so it is always one that passed its checks:

.. jupyter-execute::

    b8 = ItemLifetime("B8", right_censored=2400)
    print(b8.extend(2800))            # still running at 2800
    print(b8.fail(2650, "spall"))     # failed at 2650
    print(b8.found_failed(3000))      # found failed at the 3000 inspection
    print(b8.clone("B11"))            # the same record for another item

    from surpyval import ItemHistory

    pump = ItemHistory("P1", observed=[420], right_censored=600)
    print(pump.add_failure(700))      # failed again at 700

Every check names the item, and ``data.to_items()`` gives the items back
(``SurpyvalData.from_items(data.to_items(), counts=data.n)`` is the same
data). A plain fit of data that holds more than a plain fit takes (covariates,
event types) is of ``data.without("covariates", "event_types")``, which keeps
everything else -- B10's late entry here -- rather than of records rebuilt by
hand. An item's ``event_type`` goes to the competing-risks fitters
(``CompetingRisks.fit(data)``), and its step ``time_varying_covariates``
(``{age: values}`` or a ``StepSchedule``) to a regression's ``fit_tvc``:
``WeibullPH.fit(data)`` splits each item's life where its covariates step.
A continuously varying ``CovariatePath`` is kept with the data but not
fitted yet (#908).

The fitted model predicts for an item from its own covariates,
``model.sf_tvc(times, item)``, with no schedule to build by hand: the
item's fixed covariates and its time-varying ones are lined up as the fit
took them. Datasets from several sources join through their records,
``SurpyvalData.from_items(site_a.to_items() + site_b.to_items(),
counts=[*site_a.n, *site_b.n])``, which refuses an item id found in both.

.. jupyter-execute::

    from surpyval import StepSchedule

    stress = StepSchedule.from_changepoints([0, 500], [[0.0], [1.0]])
    rng = np.random.default_rng(1)
    life = rng.weibull(2.0, 60) * 900
    items = [
        ItemLifetime(k, observed=t, time_varying_covariates=stress)
        for k, t in enumerate(life)
    ]
    surv.WeibullPH.fit(SurpyvalData.from_items(items)).params

Covariates given by name, ``covariates={"age": 54, "arm": 1}``, are
named in the data (``covariate_names``) and in the fitted model, whose
coefficients take the names; an item's names may come in any order:

.. jupyter-execute::

    rng = np.random.default_rng(3)
    trial = []
    for k in range(80):
        arm, age = k % 2, float(rng.uniform(40, 70))
        t = rng.weibull(1.5) * 1000 * np.exp(-0.6 * arm)
        trial.append(ItemLifetime(k, observed=t, covariates={"arm": arm, "age": age}))
    data = SurpyvalData.from_items(trial)
    print(data.covariate_names)
    surv.WeibullPH.fit(data).params

From a DataFrame
----------------

Most data arrives as a table with one row per item: its id, its age, a
column saying whether it failed, what failed, and its covariates.
``SurpyvalData.from_dataframe`` reads it, naming the columns as every
DataFrame entry point does (``x_col``, ``c_col``, ... the ``fit`` argument
with ``_col``); a spreadsheet's status column, 1 or True where the item
failed, is ``event_col`` (the opposite coding of ``c``):

.. jupyter-execute::

    fleet_df = pd.DataFrame({
        "unit":   ["P1", "P2", "P3", "P4", "P5"],
        "hours":  [820, 1450, 2000, 1100, 1700],
        "failed": [True, True, False, True, False],
        "mode":   ["seal", "bearing", None, "seal", None],
        "site":   [1, 0, 1, 0, 0],
        "load":   [0.8, 0.5, 0.4, 0.9, 0.6],
    })
    fleet = SurpyvalData.from_dataframe(
        fleet_df, i_col="unit", x_col="hours", event_col="failed",
        e_col="mode", Z_cols=["site", "load"],
    )
    print(fleet.summary())
    print(fleet.item("P3"))

A table of the age last seen working and the age first seen failed, with a
blank where one is not known, is read with ``xl_col`` and ``xr_col`` (a
blank ``xr_col`` is an item still running, a blank ``xl_col`` one already
failed at the first look), and ``to_dataframe()`` gives any dataset back as
a table:

.. jupyter-execute::

    inspections = pd.DataFrame({
        "last_ok":      [None, 150, 300, 420,  500, 610],
        "first_failed": [200,  150, 380, None, 500, None],
    })
    data = SurpyvalData.from_dataframe(
        inspections, xl_col="last_ok", xr_col="first_failed"
    )
    print(data.to_dataframe())

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert fleet.ids == ["P1", "P2", "P3", "P4", "P5"]
    assert data.c.tolist() == [0, -1, 2, 1, 0, 1], data.c

One dataset, several analyses
-----------------------------

A dataset built with covariates and failure modes (the ``fleet`` above) is
refused by a fit that takes neither, rather than fitted as if they were not
there. Reshape it for the analysis instead; each operation returns a new
dataset:

.. jupyter-execute::

    everything = fleet.without("covariates", "event_types")   # a plain fit
    seals = fleet.for_event_type("seal").without("covariates")  # one mode
    by_load = fleet.with_covariates(["load"])   # a regression on load alone
    site1 = fleet.select(where=lambda item: item.covariate("site") == 1)
    first_1500 = fleet.censor_at(1500)   # as seen at the end of a study
    print(surv.Weibull.fit(everything).params)
    print(seals.summary())

Updating a dataset as items age
-------------------------------

A fleet's data is refreshed as the months pass: some items fail, the rest
run on, new ones are installed. The records' update methods (``fail``,
``found_failed``, ``extend``, ``add_failure``) give the new records, and
``update`` takes them back into the dataset -- a record whose id is in the
data replaces that item's, a new id is added -- while ``extend_running``
moves every item still running on to the age it was last seen, and
``drop`` removes items. Counts, covariates, event types, attributes and
the unit carry through:

.. jupyter-execute::

    month_end = {"P1": 900, "P2": 1500, "P3": 2100, "P4": 1200, "P5": 1800}
    refreshed = fleet.update([
        fleet.item("P3").fail(2050, "bearing"),           # failed this month
        ItemLifetime("P6", right_censored=90, covariates={"site": 1, "load": 0.7}),
    ]).extend_running(lambda item: month_end.get(item.id, 90))
    print(refreshed.summary())
    print(refreshed.item("P3"), refreshed.item("P5"), sep="\n")

A fitted model predicts for the items from their records: a record (or a
dataset, or a list of records) stands as ``Z`` in ``sf``, ``ff``, ``hf``,
``Hf``, ``df``, ``qf`` and ``mean``, its covariates read by name, and
``predict`` gives a table, one row per item and time, with ``given="age"``
for the survival of each running item given the age it has reached (an
item that has failed gets ``nan``):

.. jupyter-execute::

    model = surv.WeibullPH.fit(refreshed.without("event_types"))
    print(model.sf(2000, refreshed.item("P5")))
    print(model.predict(refreshed, times=[2000, 2500], given="age").round(3))

Comparing sites, arms or product lines
--------------------------------------

What describes an item without being fitted -- its site, its batch, a
trial's arm -- is an ``attribute`` (any value, text included), given on the
record (``attributes={"site": "North"}``) or read from the table
(``attribute_cols=``); a fit never refuses it. ``groupby`` splits the data
by an attribute (or a covariate) and ``select(where={name: value})`` keeps
one group, so a fit per group is one line:

.. jupyter-execute::

    trial = SurpyvalData.from_dataframe(
        pd.DataFrame({
            "weeks":  [5, 7, 9, 4, 6, 8, 10, 3],
            "event":  [1, 1, 0, 1, 1, 0, 1, 1],
            "age":    [50, 60, 70, 55, 65, 45, 75, 40],
            "arm":    ["drug", "placebo"] * 4,
            "site":   ["A", "A", "B", "B", "A", "B", "A", "B"],
        }),
        x_col="weeks", event_col="event", Z_cols=["age"],
        attribute_cols=["arm", "site"],
    )
    plain = trial.without("covariates")   # the age is for a regression
    km_by_arm = {arm: surv.KaplanMeier.fit(part) for arm, part in plain.groupby("arm").items()}
    weibull_by_site = {site: surv.Weibull.fit(part) for site, part in plain.groupby("site").items()}
    print({arm: km.sf(6).round(3) for arm, km in km_by_arm.items()})
    print({site: model.params.round(2) for site, model in weibull_by_site.items()})
    print(trial.select(where={"arm": "drug", "site": "A"}).summary())

Combining every input type in one fit
-------------------------------------

The real strength of the surpyval format is that these ingredients — exact,
left-, right- and interval-censoring, repeat counts, and truncation — compose
freely in a **single** ``fit`` call. Each row of ``x`` carries its own censoring
flag ``c`` and count ``n``; a row is interval-censored simply by giving it two
values ``[lower, upper]``:

.. jupyter-execute::

    x = [10, 12, [15, 20], 22, 25, 30]   # a [lo, hi] row is interval-censored
    c = [0,  1,   2,       -1,  0,  1]    # observed, right, interval, left, ...
    n = [5,  3,   2,        1,  4,  2]    # each row repeated n times
    model = surv.Weibull.fit(x=x, c=c, n=n, tl=5)   # all left-truncated at 5
    model

Nothing above is special-cased: any mix of the flags is accepted, and the fitter
condenses the data to its densest form internally.


Same data, whichever format you have it in
-------------------------------------------

Because the formats all describe the same thing, the same dataset gives the same
fit however you assemble it. Here failure / suspension / left / interval lists
are read with ``SurpyvalData.from_observations`` and compared to the
hand-built ``xcnt`` form — the fitted parameters agree exactly:

.. jupyter-execute::

    f = [12, 18, 18, 25]        # exact failures
    s = [30, 30]                # right-censored (suspended)
    l = [8]                     # left-censored
    i = [[15, 20], [22, 26]]    # interval-censored

    data = SurpyvalData.from_observations(
        observed=f, right_censored=s, left_censored=l, interval_censored=i
    )
    from_lists = surv.Weibull.fit(data)

    hand = surv.Weibull.fit(
        x=[12, 18, 25, 30, 8, [15, 20], [22, 26]],
        c=[0, 0, 0, 1, -1, 2, 2],
        n=[1, 2, 1, 2, 1, 1, 1],
    )
    print("from lists :", from_lists.params)
    print("hand xcnt  :", hand.params)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(from_lists.params, hand.params, rtol=1e-12)


Truncation, four ways
---------------------

Truncation can be a single shared bound, a per-observation array, an upper
(right) bound, or a two-sided observation window — pass ``tl`` and/or ``tr``:

.. jupyter-execute::

    x = [674, 792, 1153, 1450, 1555]

    print("shared tl   :", surv.Weibull.fit(x=x, tl=500).params)
    print("per-unit tl :", surv.Weibull.fit(
        x=x, tl=[100, 200, 300, 400, 500]).params)
    print("right tr    :", surv.Weibull.fit(x=x, tr=2000).params)
    print("window tl,tr:", surv.Weibull.fit(x=x, tl=100, tr=2000).params)

The same columns can live in a DataFrame — ``c_col``, ``n_col``, ``tl_col``
and ``tr_col`` are all optional columns you point ``fit_from_df`` at:

.. jupyter-execute::

    df = pd.DataFrame({
        'x':  [674, 792, 1153, 1450, 1555, 2000],
        'c':  [0,   0,   0,    1,    0,    1],
        'n':  [1,   2,   1,    3,    1,    2],
        'tl': [500, 500, 500,  500,  500,  500],
    })
    surv.Weibull.fit_from_df(
        df, x_col='x', c_col='c', n_col='n', tl_col='tl'
    )

Data kept as a list of each kind of observation takes its truncation with
each list: ``observed_left_truncation``, ``right_censored_left_truncation``
and so on (and ``..._right_truncation``), one number for every item of
that kind or one bound per item, in the same order. ``left_truncation`` and
``right_truncation`` are one window shared by everything:

.. jupyter-execute::

    data = SurpyvalData.from_observations(
        observed=[674, 792, 1153, 1555],
        observed_left_truncation=[100, 200, 300, 500],
        right_censored=[1450, 2000],
        right_censored_left_truncation=500,
    )
    print(data.t)
    surv.Weibull.fit(data).params


From dates to durations
-----------------------

Field data often arrives as a table of dates: when each unit was installed, when (if ever) it was removed, and why. Two decisions turn this into survival data:

- **The value** is the time in service: removal date minus install date, or, for units still in service, the end of the study minus the install date.
- **The flag** is ``0`` only when the unit was removed *because it failed*. A unit still in service is right censored, and so is a unit removed for any other reason (a preventive replacement, an upgrade, a sale): all we know is that it had not failed by then.

.. jupyter-execute::

    log = pd.DataFrame({
        "unit":      ["A", "B", "C", "D", "E", "F"],
        "installed": ["2020-01-10", "2020-03-01", "2020-06-15",
                      "2021-01-05", "2021-02-20", "2021-07-01"],
        "removed":   ["2021-05-01", None, "2022-01-10",
                      None, "2022-03-01", "2022-02-14"],
        "reason":    ["failure", None, "failure",
                      None, "preventive", "failure"],
    })
    log["installed"] = pd.to_datetime(log["installed"])
    log["removed"] = pd.to_datetime(log["removed"])

    study_end = pd.Timestamp("2022-06-30")
    log["x"] = (log["removed"].fillna(study_end) - log["installed"]).dt.days
    log["c"] = np.where(log["reason"] == "failure", 0, 1)
    print(log[["unit", "x", "c"]])

    surv.Weibull.fit_from_df(log, x_col="x", c_col="c")

A common mistake is to count the preventive replacement of unit E as a failure. It was not one, and counting it as one makes the item look less reliable than it is. Another is to leave units B and D out because they "have no failure date": they are the survivors, and they carry most of the information about how long the units last.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    _right = surv.Weibull.fit_from_df(log, x_col="x", c_col="c")
    _e_failed = np.where(log["removed"].isna(), 1, 0)     # E as a failure
    _wrong = surv.Weibull.fit(log["x"], _e_failed)
    assert _wrong.params[0] < _right.params[0], _wrong.params
    assert _wrong.mean() < _right.mean()

If units only came under observation some time after they were installed (for example, records only start at the date a database was set up), each unit's age on that date is its left truncation value, ``tl``.

Inspection data needs one more step. A failure found at the first inspection after the records begin is interval censored from the unit's last good inspection, ``xl``, which came *before* the records begin. Its age when the records begin is then not its ``tl``: the unit is in the data because its failure was detected after that date, and all that is known is that it was working at its last good inspection, so that inspection is its left truncation time (``tl = xl``). Given the age at the start of the records instead, the fit refuses the row (an interval cannot start below its own left truncation time) and the message says this. Do not repair it by moving ``xl`` up to ``tl``: that claims the unit was seen working when it was not, and biases the fit.

.. jupyter-execute::

    inspections = pd.DataFrame({
        "last_ok":   [80.0, 150.0, 210.0, 260.0],   # age at the last good inspection
        "found":     [120.0, 190.0, 250.0, 300.0],  # age when found failed
        "age_at_start": [100.0, 100.0, 100.0, 100.0],
    })
    # Found failed at the first inspection after the records began
    straddles = inspections["last_ok"] < inspections["age_at_start"]
    inspections["tl"] = np.where(straddles, inspections["last_ok"],
                                 inspections["age_at_start"])
    print(inspections)
    surv.Weibull.fit_from_df(inspections, xl_col="last_ok", xr_col="found",
                             tl_col="tl").params

Counts found at inspections
---------------------------

When units are only checked periodically, the data is a count of failures found at each inspection: a life table. Each count is interval censored between the previous inspection and the one at which it was found, and the units still working at the last inspection are right censored there. The counts go in ``n``:

.. jupyter-execute::

    inspections = [0, 100, 200, 300, 400]
    found_failed = [3, 7, 9, 4]        # failures found at 100, 200, 300, 400
    still_working = 12                 # at the last inspection

    x = [[lo, hi] for lo, hi in zip(inspections[:-1], inspections[1:])]
    x = x + [inspections[-1]]
    c = [2] * len(found_failed) + [1]
    n = found_failed + [still_working]
    print(x)

    life_table = surv.Weibull.fit(x=x, c=c, n=n)
    print(life_table.params)
    print(surv.Turnbull.fit(x=x, c=c, n=n).sf(inspections[1:]))

``from_observations`` reads the table as it is written, with a count for
each item (an interval in which nothing was found can stay in, counted 0):

.. jupyter-execute::

    data = SurpyvalData.from_observations(
        interval_censored=[[0, 100], [100, 200], [200, 300], [300, 400]],
        interval_censored_counts=[3, 7, 9, 4],
        right_censored=[400],
        right_censored_counts=12,
    )
    print(surv.Weibull.fit(data).params)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(surv.Weibull.fit(data).params, life_table.params)

Resist the temptation to put each failure at the inspection time at which it was found (or at the midpoint of the interval). Doing so invents precision the data does not have, and pushes every failure later (or to an arbitrary point) within its interval. Interval censoring uses exactly what was observed. Turnbull, the non-parametric estimator that handles interval censoring, is a good check on the parametric fit.

Recurrent events from a maintenance log
---------------------------------------

For repairable items, each item can fail many times. The recurrent event models take one row per event, with the item identifier ``i``, the **cumulative** age of the item at the event in ``x``, and one extra row per item, flagged ``c = 1``, giving how far that item has been observed. A maintenance log with hour-meter readings at each failure, plus each machine's current reading, is exactly this:

.. jupyter-execute::

    from surpyval.recurrent import NonParametricCounting

    failures = pd.DataFrame({"machine":    ["M1", "M1", "M1", "M2", "M2", "M3"],
                             "hour_meter": [420, 1310, 2050, 980, 2400, 1720]})
    current = pd.DataFrame({"machine":    ["M1", "M2", "M3"],
                            "hour_meter": [2600, 2900, 3100]})

    events = pd.concat([failures.assign(c=0), current.assign(c=1)],
                       ignore_index=True)

    mcf = NonParametricCounting.fit(x=events["hour_meter"],
                                    i=events["machine"],
                                    c=events["c"])
    print(mcf.mcf([1000, 2000, 2600]))

The same log, written as each machine's history keyed by its id, is read by
``EventHistoryData.from_histories``: ``observed`` holds each machine's age
at its failures and ``right_censored`` its current reading. Every recurrent
``fit`` takes the result in place of the arrays:

.. jupyter-execute::

    from surpyval import EventHistoryData

    data = EventHistoryData.from_histories(
        observed={"M1": [420, 1310, 2050], "M2": [980, 2400], "M3": [1720]},
        right_censored={"M1": 2600, "M2": 2900, "M3": 3100},
    )
    print(NonParametricCounting.fit(data).mcf([1000, 2000, 2600]))

A machine left out of ``right_censored`` is taken to have been observed
until its last failure. ``left_truncation`` (the age at which a machine came
under observation) and ``covariates`` (one row per machine, for the
proportional intensity models) are keyed the same way, and ``gaps=True``
reads the times between failures, adding them up within each machine.
``data.to_histories()`` gives the dictionaries back.

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.allclose(
        NonParametricCounting.fit(data).mcf([1000, 2000, 2600]),
        mcf.mcf([1000, 2000, 2600]),
    )

The same histories give lifetime data for the lifetime models:
``to_first_events()`` is each machine's time to its first failure (right
censored where it has none), and ``to_gaps()`` the times between its
failures, each repair taken to make it as good as new, its last gap right
censored at the end of its observation. Both keep each machine's delayed
entry, covariates and ids (``("M1", 2)`` for M1's second gap):

.. jupyter-execute::

    print(data.to_first_events().x)
    surv.Weibull.fit(data.to_gaps()).params

When an item's history holds more than exact failures (failures found
between inspections, a count of failures before the first look, what each
failure was, a window it was watched in, covariates), describe each item
with an ``ItemHistory`` and build the dataset from the list:

.. jupyter-execute::

    from surpyval import ItemHistory

    pump = ItemHistory(
        "P1",
        observed=[420, 1310],
        observed_event_types=["seal", "bearing"],
        interval_censored=[[2100, 2300]],   # found failed at an inspection
        interval_censored_counts=2,
        interval_censored_event_types="seal",
        right_censored=2600,
        left_truncation=150,                # records start at 150 hours
    )
    fan = ItemHistory("F1", observed=[800], observed_event_types="bearing",
                      right_censored=3000)
    data = EventHistoryData.from_items([pump, fan])
    data.to_items()[1]

Every check names the item and says what is wrong with its history, and
``data.to_items()`` gives the items back. Two kinds of history are kept with
the data for the models that will use them, and refused by every fit until
then:

- ``time_varying_covariates`` (#905): ``{age: values}``, a ``StepSchedule``
  or a continuously varying ``CovariatePath``.
- The item's path through states (#887), recorded like its failures: an
  ``initial_state``, changes seen as they happened (``observed_states={age:
  state}``), changes between two inspections (``interval_censored_states=
  [[lower, upper, state]]``) and a state already reached at the first look
  (``left_censored_states``).

``markers`` (repeated measurements, #891) are reserved for joint models of
histories; a lifetime's record, ``ItemLifetime``, takes them now (see
:doc:`Joint Models`).

A log held in a DataFrame, one row per failure, is read by
``EventHistoryData.from_dataframe``: the item and age columns, the type of
each failure, and each item's end of observation as a ``c = 1`` row, a
column (``end_col``) or ``ends={item: age}``, which also names the items
that never failed and so have no row:

.. jupyter-execute::

    log = pd.DataFrame({
        "machine": ["M1", "M1", "M1", "M2", "M2"],
        "hours":   [420, 1310, 2050, 980, 2400],
        "part":    ["seal", "bearing", "seal", "seal", "bearing"],
    })
    data = EventHistoryData.from_dataframe(
        log, i_col="machine", x_col="hours", e_col="part",
        ends={"M1": 2600, "M2": 2900, "M3": 3100},
    )
    print(data.summary())

If your log records the time *between* failures instead, take a cumulative sum within each item first, for example ``df.groupby("machine")["hours_between"].cumsum()``. Leaving out the end-of-observation rows is a mistake: without them the models cannot know that M3, for example, has run a further 1,380 hours without failing since its failure at 1,720 hours, and would treat it as if it had been scrapped straight after that failure. See :doc:`Recurrent Event Modelling with SurPyval` for the models that use this data.


Format converters
-----------------

``SurpyvalData`` reads the common layouts and writes them back out, and the
non-parametric estimators' "at risk / deaths" (``xrd``) form too:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Converts
   * - ``SurpyvalData(x, c, n, t, xl, xr, tl, tr, Z)``
     - ``xcnt`` arrays (or ``xl`` / ``xr`` columns) → data, validated,
       grouped and sorted as the fitters see it
   * - ``SurpyvalData.from_observations(observed, right_censored,
       left_censored, interval_censored, left_truncation=,
       right_truncation=)``
     - a list of each kind of observation (the ``fsli`` layout), each
       with counts (``observed_counts=``, ...), covariates
       (``observed_covariates=``, ...) and truncation bounds
       (``observed_left_truncation=``, ...) of its own, one per item, or
       one truncation window shared by all → data
   * - ``SurpyvalData.from_xrd(x, r, d)``
     - at-risk / deaths (``xrd``) → data
   * - ``data.to_xcnt()``
     - data → the arrays ``x, c, n, t``
   * - ``data.to_observations()``
     - data → the lists ``from_observations`` reads: each value once,
       with each kind's counts, truncation bounds and covariates where the
       data has them
   * - ``data.to_xrd()``
     - data → ``x, r, d``
   * - ``SurpyvalData.from_dataframe(df, x_col=..., c_col=... | event_col=...,
       i_col=..., e_col=..., Z_cols=[...])``
     - one row per item from a DataFrame (``xl_col`` / ``xr_col`` for last
       seen working / first seen failed, blanks allowed);
       ``data.to_dataframe()`` gives it back
   * - ``data.to_json()``, ``SurpyvalData.from_json(...)``
     - save and restore the data, as JSON text or a file (``to_dict`` /
       ``from_dict`` give the dictionary; ``surpyval.from_json`` reads it
       too; every data class has them)
   * - ``handle_xicn(x, i, c, n, ...)``
     - validates recurrent event data (``xicnt``) and returns the
       ``EventHistoryData`` object the recurrent models use

The conversions are not all lossless, and it pays to know where they drop
information:

- ``to_xrd`` is exact for observed and right censored data without right
  truncation; left truncation is allowed, and is used to build the risk
  sets. Left or interval censored values have no single time at which to
  count a death, so for those (or for right truncated data) it fits the
  Turnbull estimator and returns its numbers at risk and deaths, which can
  be fractional.
- ``from_xrd`` cannot recover left truncation, and raises an error if the
  risk set ever grows from one time to the next. xrd data must list each
  time once; rows given out of order are sorted by time (keeping their
  ``r`` and ``d``), and a time listed twice is an error. More deaths than
  items at risk is an error too.
- ``to_observations`` and ``from_observations`` lose nothing:
  ``SurpyvalData.from_observations(**data.to_observations())`` is the same
  data, counts, truncation and covariates included.

Observed and right censored data folds into the ``xrd`` form, the count at
risk and the number of deaths at each distinct time:

.. jupyter-execute::

    x, c, n = [1, 2, 3, 4, 5], [0, 0, 1, 0, 1], [2, 1, 1, 3, 1]
    times, at_risk, deaths = SurpyvalData(x, c, n).to_xrd()
    print("times  :", times)
    print("at risk:", at_risk)
    print("deaths :", deaths)

Going back with ``from_xrd`` recovers the same rows here. The number
censored at each time is what is left of the risk set after the deaths,
``r[j] - d[j] - r[j + 1]``, and it is placed at that time:

.. jupyter-execute::

    x_back, c_back, n_back, _ = SurpyvalData.from_xrd(
        times, at_risk, deaths
    ).to_xcnt()
    print("x:", x_back)
    print("c:", c_back)
    print("n:", n_back)

.. jupyter-execute::
    :hide-code:
    :hide-output:

    assert np.array_equal(x_back, x) and np.array_equal(c_back, c)
    assert np.array_equal(n_back, n)

Before version 0.25 these conversions were functions at the top level of
``surpyval``. They still work, with a warning, until 2027-04-12, and the
first release on or after that date removes them:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Deprecated
     - Use
   * - ``fs_to_xcnt(f, s)``
     - ``SurpyvalData.from_observations(observed=f, right_censored=s)``
       (``.to_xcnt()`` for the arrays)
   * - ``fsl_to_xcnt(f, s, l)``, ``fsli_to_xcnt(f, s, l, i)``,
       ``fsli_handler(f, s, l, i)``
     - ``SurpyvalData.from_observations(...)`` with ``left_censored=l``
       and ``interval_censored=i``
   * - ``fs_to_xrd(f, s)``
     - ``SurpyvalData.from_observations(observed=f,
       right_censored=s).to_xrd()``
   * - ``xcnt_handler(x, c, n, t, ...)``
     - ``SurpyvalData(x, c, n, t, ...)`` (``.to_xcnt()`` for the arrays)
   * - ``xcnt_to_xrd(x, c, n, t)``
     - ``SurpyvalData(x, c, n, t).to_xrd()``
   * - ``xrd_to_xcnt(x, r, d)``, ``xrd_handler(x, r, d)``
     - ``SurpyvalData.from_xrd(x, r, d)`` (``.to_xcnt()`` for the arrays)
   * - ``xcn_to_fs(x, c, n)``
     - ``SurpyvalData(x, c, n).to_observations()``

If the risk set ever grows from one time to the next, some items entered
late (left truncation). ``xrd`` has no record of when each item entered,
so the conversion cannot be undone, and SurPyval refuses rather than
guessing:

.. jupyter-execute::

    try:
        SurpyvalData.from_xrd([1, 2, 3], [4, 6, 3], [1, 2, 3])
    except ValueError as err:
        print(err)
