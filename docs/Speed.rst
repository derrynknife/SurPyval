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
with ``n``; a quiet machine is somewhat faster.

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
     - 12 ms
     - 28 ms
     - 0.19 s
     - 2.2 s
   * - the same data, grouped with counts ``n``
     - 12 ms
     - 12 ms
     - 11 ms
     - 13 ms
   * - ``Weibull.fit_from_df``
     - 13 ms
     - 28 ms
     - 0.20 s
     - 2.5 s
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
     - 53 ms
     - 93 ms
     - 0.57 s
     - 5.7 s
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
     - 4.8 s
     - 10 s
     - 62 s
     - --
   * - bootstrap, 50 samples
     - 0.85 s
     - 2.2 s
     - 18 s
     - 1,300 s
   * - likelihood ratio, grouped data
     - 0.52 s
     - 0.52 s
     - 0.47 s
     - 0.52 s
   * - bootstrap, 200 samples, grouped data
     - 12 s
     - 19 s
     - 85 s
     - --

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
  ``n_boot`` fits, plus a set-up step that estimates the censoring
  distribution (so that each resample is censored as the data were). That
  set-up grows with the number of rows times the number of distinct
  censoring times, so on large data sets it is most of the cost: at
  200,000 rows it took 44 of the 52 seconds of a five-sample bootstrap. It
  works unit by unit, so grouping the data with ``n`` does not shorten
  it.
- **The random survival forest** grows each tree by searching every
  candidate split of every chosen covariate at every node, and with
  ``kind="weibull"`` fits a Weibull in each leaf. It runs in pure Python
  and NumPy, one tree at a time by default.
- **The joint model** integrates over each item's random effects (a
  quadrature grid per item), so its cost is per item and per reading.

Making a fit faster
-------------------

- **Group identical rows with** ``n``. A fit's cost is per distinct row, so
  data recorded to the hour, the day or the cycle, with many units at each
  value, fits almost instantly once grouped: a million censored rows
  recorded to whole time units become about 500 distinct ``(x, c)`` rows,
  and the fit drops from 2.2 s to 13 ms. The answer is exactly the same:

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
  ten to twenty times quicker than 200 bootstrap re-fits. On grouped data
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
  was not run. For a forest on that much data, subsample, or use a
  dedicated gradient-boosting or forest library and SurPyval to check the
  predictions (``surpyval.metrics``).
- **Turnbull on data where every unit has its own inspection times** grows
  faster than ``n``: 2.3 s at 100,000 rows, 42 s at a million.
- **The bootstrap on large data sets.** 200 samples take a minute at
  100,000 rows, and even 50 samples took 22 minutes (1,300 s) at a
  million, because of the set-up step above rather than the re-fits.
  Grouping does not help it. At that size use the Wald bound, or the
  likelihood-ratio bound on grouped data (half a second at any size).
- **Likelihood-ratio bounds on a million ungrouped rows** take about a
  minute; grouped, half a second.

Everything else on this page fits a million rows in a few seconds or less.
