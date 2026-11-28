Speed
=====

How long does a fit take? This page gives measured times for the common
models at 1,000 to 1,000,000 rows, says what the time goes on, and how to
make a slow fit faster. The short answer: most fits of up to a million
rows take seconds; a few things scale badly, and the table says which.

How these were measured
-----------------------

The numbers come from ``scripts/bench_speed.py`` in the repository, which
simulates each data set with a fixed seed and times the fit alone (the best
of three runs for a fit under a second, one run otherwise). It is not run
in CI; run it yourself to see your own machine::

    python scripts/bench_speed.py                  # everything, about an hour
    python scripts/bench_speed.py weibull cox km   # some of it
    python scripts/bench_speed.py --max-n 10000    # small sizes only

The machine was an Apple M5 laptop (macOS 27.2, arm64) with Python 3.12.2,
SurPyval 0.29 (the ``dev`` branch of 25 November 2026), NumPy 2.5.5, SciPy
1.18.1, pandas 2.3.3 and autograd 1.9.1, with other work running on it at
the same time. Read the times as orders of magnitude and for how they grow
with ``n``; a quiet machine is somewhat faster. The ``Weibull.fit``,
``fit_from_df``, ``WeibullAFT.fit`` and bootstrap rows were measured again
later that day (pandas 3.1.1), after the bootstrap's set-up was made one
sort and every fit stopped indexing its rows one by one; the other rows
are from before that, and the fits among them of ungrouped data without
unit ids may now be up to a second per million rows quicker.

The data: Weibull lives (scale 100, shape 2) with about 30% right censored;
for the regressions, three covariates (one normal, one binary, one uniform)
acting through proportional hazards. A dash means the size was not run:
the script stops a model once one fit takes over a minute.

Fit times
---------

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Model
     - n = 1,000
     - 10,000
     - 100,000
     - 1,000,000
   * - ``Weibull.fit`` (MLE, censored)
     - 10 ms
     - 19 ms
     - 0.11 s
     - 1.3 s
   * - the same data, grouped with counts ``n``
     - 12 ms
     - 12 ms
     - 11 ms
     - 13 ms
   * - ``Weibull.fit_from_df``
     - 10 ms
     - 20 ms
     - 0.11 s
     - 1.4 s
   * - ``KaplanMeier.fit``
     - 0.5 ms
     - 2.8 ms
     - 31 ms
     - 0.38 s
   * - ``CoxPH.fit``, 3 covariates, Efron ties
     - 3.5 ms
     - 15 ms
     - 0.11 s
     - 1.3 s
   * - ``CoxPH.fit``, no ties
     - 2.6 ms
     - 17 ms
     - 0.16 s
     - 1.9 s
   * - ``WeibullAFT.fit``, 3 covariates
     - 42 ms
     - 73 ms
     - 0.40 s
     - 5.2 s
   * - ``Turnbull.fit``, everyone inspected on one schedule
     - 1.2 ms
     - 1.9 ms
     - 12 ms
     - 0.12 s
   * - ``Turnbull.fit``, each unit on its own schedule
     - 16 ms
     - 0.12 s
     - 2.3 s
     - 42 s
   * - ``CrowAMSAA.fit``, ``n`` failures over ``n/10`` systems
     - 1.4 ms
     - 7.9 ms
     - 83 ms
     - 1.1 s
   * - ``RandomSurvivalForest``, 100 trees, Weibull leaves
     - 5.9 s
     - 33 s
     - 213 s
     - --
   * - ``RandomSurvivalForest``, 100 trees, non-parametric leaves
     - 2.3 s
     - 15 s
     - 126 s
     - --

The joint model is timed by the number of items, each with about ten
readings (so 10,000 items is about 100,000 readings), with a linear path
whose level and rate vary between items:

.. list-table::
   :header-rows: 1
   :widths: 40 12 12 12 12 12

   * - Model
     - 100 items
     - 300
     - 1,000
     - 3,000
     - 10,000
   * - ``WeibullJointPH.fit``
     - 0.24 s
     - 0.47 s
     - 1.2 s
     - 3.5 s
     - 14 s

Bounds
------

Each bound is on a fitted Weibull's reliability at one time, ``model.cb(50,
method=...)``, timed on its first call (a model keeps what a bound has
worked out, so a second call is quicker):

.. list-table::
   :header-rows: 1
   :widths: 40 15 15 15 15

   * - Bound
     - n = 1,000
     - 10,000
     - 100,000
     - 1,000,000
   * - Wald (Fisher matrix), the default
     - 3 ms
     - 3 ms
     - 1 ms
     - 1 ms
   * - likelihood ratio, ``method="lr"``
     - 0.20 s
     - 0.64 s
     - 4.3 s
     - 64 s
   * - bootstrap, ``method="bootstrap"``, 200 samples (the default)
     - 4.1 s
     - 6.9 s
     - 39 s
     - 550 s
   * - bootstrap, 50 samples
     - 1.5 s
     - 2.7 s
     - 15 s
     - 126 s
   * - likelihood ratio, grouped data
     - 0.52 s
     - 0.52 s
     - 0.47 s
     - 0.52 s
   * - bootstrap, 200 samples, grouped data
     - 6.4 s
     - 8.8 s
     - 43 s
     - 470 s

The first bound of a session takes about 0.3 s more than these, once,
while the derivatives it needs are set up.

What the time goes on
---------------------

- **A parametric fit is an optimisation.** Each step evaluates the
  log-likelihood and its gradient over every row, so the time grows in
  proportion to the number of *distinct* rows. The fixed cost (finding a
  start, setting up the gradient) is about 10 ms, which is why 1,000 and
  10,000 rows take about the same time.
- **A regression has more parameters**, so more steps and a bigger
  Hessian: the Weibull AFT with three covariates is two to three times a
  plain Weibull at every size.
- **Cox is cheap.** The partial likelihood needs one sort of the times and
  then running sums, so a million rows take a second or two; ties (Efron)
  do not slow it down.
- **Turnbull depends on the number of distinct interval endpoints**, not
  the number of rows. When every unit is inspected on one schedule there
  are a few dozen intervals and a million rows take a tenth of a second;
  when each unit has its own inspection times, the number of intervals
  grows with ``n`` and so does the self-consistency iteration: 42 s at a
  million rows.
- **The likelihood-ratio bound** solves for the edges of the region where
  the likelihood is within a chi-squared quantile of its peak, which takes
  several constrained re-fits. Its cost grows with ``n`` like a fit's,
  times a few dozen.
- **The bootstrap** re-fits the model ``n_boot`` times, so it costs
  ``n_boot`` fits of a simulated data set the size of the data, plus a
  set-up step that estimates the censoring distribution (so that each
  resample is censored as the data were). The set-up is one sort, 0.2 s
  at a million rows; the re-fits are the rest. Each simulated unit gets its
  own failure time, so grouping the data with ``n`` does not shorten the
  re-fits of the (unweighted) parametric bootstrap. A fit with sampling
  weights is resampled by cases instead, and grouped weighted data are
  resampled by row (each row drawn a multinomial number of times), so
  there each re-fit is only as big as the grouped data.
- **The random survival forest** grows each tree by searching every
  candidate split of every chosen covariate at every node, and with
  ``kind="weibull"`` fits a Weibull in each leaf. It runs in pure Python
  and NumPy, one tree at a time by default. Profiled on the data above,
  one tree takes about 0.4 s at 10,000 rows and 4 s at 100,000, and the
  split search is almost all of it (86% to 89%); the leaf fits are 3% to
  4%, and slicing the rows into each node's children 5% to 7%. With
  Weibull leaves the search is scored by the Weibull likelihood of the
  two children of every candidate cut (at most 64 cuts per covariate per
  node), and solving for each child's shape is about 60% of the whole
  fit: a few Newton steps per cut, each a pass over the node's rows. It
  is already worked out for every cut at once from the node's rows, not
  a new data set per cut. With ``kind="non-parametric"`` the log-rank
  statistic of every cut comes from running sums over the covariate's
  order, one sort per covariate per node (80% to 90% of a tree that takes
  0.17 s at 10,000 rows and 1.5 s at 100,000). So there is no cut being
  scored twice, and nothing left to remove that keeps the trees exactly
  as they are; what is left is the number of passes.
- **The limited-failure and zero-inflated bootstrap** (``lfp=True`` or
  ``zi=True`` with ``method="bootstrap"``) gives the bias-corrected and
  accelerated (BCa) interval, whose acceleration comes from the jackknife:
  the model is refitted once without each distinct row (a grouped row of
  ``n`` units counts once, weighted by ``n``). That is one extra fit per
  distinct row, on top of the ``n_boot`` refits, and a limited-failure
  fit is a slow one (it tries several starts). For a Weibull with
  ``lfp=True`` and 50 bootstrap samples, the jackknife took 7.5 s of
  14 s at 91 distinct rows, 44 s of 51 s at 270 and 91 s of 98 s at 892:
  past about a hundred distinct rows it is most of the time, and it grows
  in proportion to them. Grouping identical rows with ``n`` shortens it
  in proportion. It is the
  exact jackknife; a shortcut from the scores (the infinitesimal
  jackknife) would change the bounds, so it is not used.
- **A fit's own set-up.** Before it searches, a fit validates, groups
  and sorts the rows (about a third of a second at a million rows) and,
  unless ``init`` is given, finds a start from a probability plot (about
  as long again). A start you give with ``init`` does not skip that: the
  default start is still tried after it, and the better of the two kept,
  so that a poor ``init`` cannot give a worse model than none would.
- **The joint model** integrates over each item's random effects (a
  quadrature grid per item), so its cost is per item and per reading.

Making a fit faster
-------------------

- **Group identical rows with** ``n``. A fit's cost is per distinct row, so
  data recorded to the hour, the day or the cycle, with many units at each
  value, fits almost instantly once grouped: a million censored rows
  recorded to whole time units become about 500 distinct ``(x, c)`` rows,
  and the fit drops from 1.3 s to 13 ms. The answer is exactly the same:

  .. jupyter-execute::

      import numpy as np
      import surpyval as sp

      rng = np.random.default_rng(1)
      life = np.ceil(100 * rng.weibull(2, 100_000))   # recorded to whole days
      cens = np.ceil(160 * rng.weibull(2, 100_000))
      x = np.minimum(life, cens)
      c = (cens < life).astype(int)

      rows, counts = np.unique(np.column_stack([x, c]), axis=0, return_counts=True)
      grouped = sp.Weibull.fit(rows[:, 0], rows[:, 1].astype(int), counts)
      print(len(rows), "distinct rows:", grouped.params)
      print("ungrouped:", sp.Weibull.fit(x, c).params)

  .. jupyter-execute::
      :hide-code:
      :hide-output:

      assert np.allclose(grouped.params, sp.Weibull.fit(x, c).params, rtol=1e-10)

  Grouping changes nothing in the estimate or the bounds (``n = 3`` is the
  same as three identical rows), but it does not help a model whose rows
  differ in their covariates.
- **Use fewer bootstrap samples, or none.** ``n_boot=50`` takes a quarter
  of the default 200, and is enough to see whether the Wald bound is
  reasonable; for a reported interval keep 200 or more. With more than a
  few hundred failures the Wald bound is usually close to the
  likelihood-ratio and bootstrap bounds and costs almost nothing.
- **Prefer the likelihood-ratio bound to the bootstrap** in small and
  medium samples: it is the better bound when the data are few, and it is
  about ten times quicker than 200 bootstrap re-fits. On grouped data
  it takes about half a second whatever the number of units.
- **``fit_from_df`` costs the same as ``fit``.** It is a convenience, not a
  speed-up: the columns are read into the same arrays. Group the
  DataFrame first (``df.groupby(["t", "c"]).size()``) and pass the counts
  as ``n_col`` to get the grouping speed-up.
- **Forests: fewer trees, shallower trees, more cores.** ``n_trees``,
  ``max_depth`` and ``min_leaf_samples`` cut the work in proportion;
  ``kind="non-parametric"`` leaves are about twice as quick as Weibull
  leaves; ``n_jobs`` grows trees in parallel (with ``n_jobs=4`` 100 trees on 10,000 rows took 16 s, from 33 s; on 1,000 rows it saved nothing, as starting the workers costs what it saves).
- **Turnbull with many distinct inspection times**: round the inspection
  times to the precision that matters (a day, a week), which shrinks the
  number of intervals.
- **Joint models**: ``integration="laplace"`` replaces the quadrature grid
  by Laplace's approximation, much faster when three or more path
  parameters vary between items.

Where it scales badly
---------------------

Said plainly:

- **Random survival forests are slow beyond tens of thousands of rows.**
  100 trees take half a minute at 10,000 rows and three and a half minutes
  at 100,000; a million rows would take over half an hour on one core and
  was not run. Non-parametric leaves are about twice as
  quick, as the log-rank split needs no shape to be solved for at each
  cut. For a forest on that much data, subsample, or use a
  dedicated gradient-boosting or forest library and SurPyval to check the
  predictions (``surpyval.metrics``).
- **The limited-failure or zero-inflated bootstrap on many distinct
  rows** adds one refit per distinct row for its jackknife (see above);
  group the data, or use the likelihood-ratio bound.
- **Turnbull on data where every unit has its own inspection times** grows
  faster than ``n``: 2.3 s at 100,000 rows, 42 s at a million.
- **The bootstrap on large data sets** costs ``n_boot`` re-fits of the
  whole data set: 200 samples take 40 s at 100,000 rows and about nine
  minutes at a million (50 samples, two minutes). Grouping does not
  shorten the parametric bootstrap. At that size use the Wald bound, or
  the likelihood-ratio bound on grouped data (half a second at any size).
  (Before 25 November a set-up step made it far slower: 50 samples took
  22 minutes at a million rows.)
- **Likelihood-ratio bounds on a million ungrouped rows** take about a
  minute; grouped, half a second.

Everything else on this page fits a million rows in a few seconds or less.
