
Data Wrangling Examples
=======================

Every SurPyval fitter takes data in the same form: values ``x``, censoring flags ``c``, counts ``n`` and truncation ``t`` (or ``tl`` and ``tr``), described in :doc:`Conventions`. Data rarely arrives that way. It comes as a list of failures and a list of survivors, as text with "+" marks, as a spreadsheet of install and removal dates, or as counts found at each inspection. This page shows how to get from each of these to something ``fit()`` accepts. The general recipe is always the same: decide, for each item, *what is known about its value* (exact, above a value, below a value, or between two values) and *whether it could have been missed altogether* (truncation), then write that down as ``x``, ``c``, ``n`` and ``t``.

Lets just say we have a list of right censored data and a list of failures. How can we wrangle these into data for the :code:`fit()` method to accept?

.. jupyter-execute::

    import surpyval as surv

    # Failure data
    f = [2, 3, 4, 5, 6, 7, 8, 8, 9]
    # 'suspended' or right censored data
    s = [1, 2, 10]

    # convert to xcnt format!
    x, c, n, t = surv.fs_to_xcnt(f, s)
    print(x, c, n)

    model = surv.Weibull.fit(x, c, n)
    print(model)


You can even bring in your left censored data as well:

.. jupyter-execute::

    # Failure data
    f = [2, 3, 4, 5, 6, 7, 8, 8, 9]
    # 'suspended' or right censored data
    s = [1, 2, 10]
    # left censored data
    l = [7, 8, 9]

    # convert to xcnt format!
    x, c, n, t = surv.fsl_to_xcnt(f, s, l)
    print(x, c, n)

    model = surv.Weibull.fit(x, c, n)
    print(model)


Another common type of data that is provided is in a simple text list with "+" indicating that the observation was censored at that point. Using some simple python list comprehensions can help.

.. jupyter-execute::

    # Example provided data
    data = "1, 2, 3+, 5, 6, 8, 10, 3+, 5+"

    f = [float(x) for x in data.split(',') if "+" not in x]
    s = [float(x[0:-1]) for x in data.split(',') if "+" in x]

    data = surv.fs_to_xcnt(f, s)

    model = surv.Weibull.fit(*data)
    model

Again, this can be extended to left censored data as well:

.. jupyter-execute::

    data = "1, 2, 3+, 5, 6, 8, 10, 3+, 5+, 15-, 16-, 17-"
    split_data = data.split(',')

    f = [float(x) for x in split_data if ("+" not in x) & ("-" not in x)]
    s = [float(x[0:-1]) for x in split_data if "+" in x]
    l = [float(x[0:-1]) for x in split_data if "-" in x]

    # Create the x, c, n data
    data = surv.fsl_to_xcnt(f, s, l)

    model = surv.Weibull.fit(*data)

Surpyval also offers the ability to use a pandas DataFrame as an input to the parametric fitters. All you need to do is tell ``fit_from_df`` which columns to look at for x, c, n, and the truncation, tl and tr. Columns for c, n, tl and tr are optional, and tl and tr can also be given as a single number that applies to every row. Further, if you have interval censored data you can use the 'xl' and 'xr' column names instead.

.. jupyter-execute::

    import pandas as pd

    xr = [2, 4, 6, 8, 10]
    xl = [1, 2, 3, 4, 5]
    df = pd.DataFrame({'xl' : xl, 'xr' : xr})

    model = surv.Weibull.fit_from_df(df, xl='xl', xr='xr')
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
    x, c, n, t = surv.xcnt_handler(xl=df['last_ok'], xr=df['first_failed'])
    print(c)

    model = surv.Weibull.fit_from_df(df, xl='last_ok', xr='first_failed')
    print(model.params)

(``xcnt_handler`` is the function every fitter uses to validate its input; calling it yourself is a quick way to check how SurPyval has read your data. Its output is sorted, so the flags come back in order of time.)


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
are converted with ``fsli_to_xcnt`` and compared to the hand-built ``xcnt``
form — the fitted parameters agree exactly:

.. jupyter-execute::

    f = [12, 18, 18, 25]        # exact failures
    s = [30, 30]                # right-censored (suspended)
    l = [8]                     # left-censored
    i = [[15, 20], [22, 26]]    # interval-censored

    x, c, n, t = surv.fsli_to_xcnt(f, s, l, i)
    from_lists = surv.Weibull.fit(x=x, c=c, n=n)

    hand = surv.Weibull.fit(
        x=[12, 18, 25, 30, 8, [15, 20], [22, 26]],
        c=[0, 0, 0, 1, -1, 2, 2],
        n=[1, 2, 1, 2, 1, 1, 1],
    )
    print("from lists :", from_lists.params)
    print("hand xcnt  :", hand.params)


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

The same columns can live in a DataFrame — ``c``, ``n``, ``tl`` and ``tr`` are
all optional columns you point ``fit_from_df`` at:

.. jupyter-execute::

    df = pd.DataFrame({
        'x':  [674, 792, 1153, 1450, 1555, 2000],
        'c':  [0,   0,   0,    1,    0,    1],
        'n':  [1,   2,   1,    3,    1,    2],
        'tl': [500, 500, 500,  500,  500,  500],
    })
    surv.Weibull.fit_from_df(df, x='x', c='c', n='n', tl='tl')


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

    surv.Weibull.fit_from_df(log, x="x", c="c")

A common mistake is to count the preventive replacement of unit E as a failure. It was not one, and counting it as one makes the item look less reliable than it is. Another is to leave units B and D out because they "have no failure date": they are the survivors, and they carry most of the information about how long the units last.

If units only came under observation some time after they were installed (for example, records only start at the date a database was set up), each unit's age on that date is its left truncation value, ``tl``.

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

If your log records the time *between* failures instead, take a cumulative sum within each item first, for example ``df.groupby("machine")["hours_between"].cumsum()``. Leaving out the end-of-observation rows is a mistake: without them the models cannot know that M3, for example, has run a further 1,380 hours without failing since its failure at 1,720 hours, and would treat it as if it had been scrapped straight after that failure. See :doc:`Recurrent Event Modelling with SurPyval` for the models that use this data.


Format converters
-----------------

surpyval ships helpers to reshape common external layouts into the ``xcnt``
format its fitters use, and into the "at risk / deaths" (``xrd``) format the
non-parametric estimators think in:

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Helper
     - Converts
   * - ``fs_to_xcnt(f, s)``
     - failures + suspensions (right-censored) → ``xcnt``
   * - ``fsl_to_xcnt(f, s, l)``
     - as above, plus left-censored
   * - ``fsli_to_xcnt(f, s, l, i)``
     - as above, plus interval-censored
   * - ``fs_to_xrd(f, s)``
     - failures + suspensions → ``xrd``
   * - ``xcnt_to_xrd(x, c, n, t)``
     - ``xcnt`` → at-risk / deaths (``xrd``)
   * - ``xrd_to_xcnt(x, r, d)``
     - ``xrd`` → ``xcnt``
   * - ``xcn_to_fs(x, c, n)``
     - ``xcnt`` → failure / suspension lists
   * - ``xcnt_handler(x, c, n, t, xl, xr, tl, tr)``
     - validates any ``xcnt`` input and returns it grouped and sorted, as the
       fitters see it

The conversions are not all lossless, and it pays to know where they drop
information:

- ``xcnt_to_xrd`` (and ``fs_to_xrd``) only accept observed and right censored
  data without right truncation: a left or interval censored value has no
  single time at which to count its death. Left truncation is allowed, and is
  used to build the risk sets.
- ``xrd_to_xcnt`` cannot recover left truncation, and raises an error if the
  risk set ever grows from one time to the next.
- ``xcn_to_fs`` returns only the observed (``c = 0``) and right censored
  (``c = 1``) values; left and interval censored rows are left out.

Observed and right censored ``xcnt`` data folds into the ``xrd`` form, the count at risk
and the number of deaths at each distinct time:

.. jupyter-execute::

    x, c, n = [1, 2, 3, 4, 5], [0, 0, 1, 0, 1], [2, 1, 1, 3, 1]
    times, at_risk, deaths = surv.xcnt_to_xrd(x, c, n)
    print("times  :", times)
    print("at risk:", at_risk)
    print("deaths :", deaths)
