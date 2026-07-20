
Data Wrangling Examples
=======================


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

Surpyval also offers the ability to use a pandas DataFrame as an input. All you need to do is tell it which columns to look at for x, c, n, and t. Columns for c, n, and t are optional. Further, if you have interval censored data you can use the 'xl' and 'xr' column names instead. If you have mixed interval and observed or censored data, just make sure the value in the 'xl' column is the value of the observation or left or right censoring.


.. jupyter-execute::

    import pandas as pd

    xr = [2, 4, 6, 8, 10]
    xl = [1, 2, 3, 4, 5]
    df = pd.DataFrame({'xl' : xl, 'xr' : xr})

    model = surv.Weibull.fit_from_df(df, xl='xl', xr='xr')
    print(model)


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
   * - ``xcnt_to_xrd(x, c, n, t)``
     - ``xcnt`` → at-risk / deaths (``xrd``)
   * - ``xrd_to_xcnt(x, r, d)``
     - ``xrd`` → ``xcnt``
   * - ``xcn_to_fs(x, c, n)``
     - ``xcnt`` → failure / suspension lists

For example, any ``xcnt`` data folds into the ``xrd`` form — the count at risk
and the number of deaths at each distinct time:

.. jupyter-execute::

    x, c, n = [1, 2, 3, 4, 5], [0, 0, 1, 0, 1], [2, 1, 1, 3, 1]
    times, at_risk, deaths = surv.xcnt_to_xrd(x, c, n)
    print("times  :", times)
    print("at risk:", at_risk)
    print("deaths :", deaths)
