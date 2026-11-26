Machine Learning
================

Tree-based survival models: a survival tree, and a random survival
forest built from them. Both accept the full surpyval data model --
arbitrary censoring and truncation -- and return a fitted model at each
leaf (parametric, or a Nelson-Aalen or Turnbull estimate for
``kind="non-parametric"``), so a prediction is a distribution rather
than a point. Import them from ``surpyval.ml``.

.. py:module:: surpyval.ml

The tree ``kind`` couples the split with the leaf model, and every kind
takes every kind of censoring and truncation:

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * - ``kind``
     - Observed and right-censored data, with or without left truncation
     - Left- or interval-censored data, or right truncation (any truncation)
   * - ``"non-parametric"``
     - Risk-set log-rank split; Nelson-Aalen leaves
     - Turnbull-score split (log-rank scores of the pooled Turnbull estimate,
       each less the score of its truncation window); Turnbull leaves
   * - ``"exponential"``
     - Exponential deviance split; Exponential leaves
     - The same, on the full likelihood
   * - ``"weibull"`` (default)
     - Weibull deviance split; Weibull leaves
     - The same, on the full likelihood

For ``"non-parametric"`` the column is chosen at each node (and leaf) from
the rows that reach it. Either
``selection`` (``"greedy"`` or ``"ctree"``) works with every kind and data
type.

.. note::

   Before v0.27 these lived in the beta tier as ``surpyval.beta.ml``. They
   are now part of the stable surface, with the same stability promise and
   deprecation cycle as the rest of the package. The old name still imports
   them, with a ``FutureWarning``, until the first release on or after
   12 April 2027 (see :doc:`surpyval.beta`).

For a narrative introduction see the survival-forest section of
:doc:`Regression Modelling with SurPyval`.

Extremely randomised trees
--------------------------

By default a node tries every value of each feature it considers as a cut
and keeps the best. With ``splitrule="random"`` it draws one cut for each
feature instead, uniformly between the smallest and largest value the
feature takes at that node, and keeps the best of those few by the same
split statistic (the extremely randomised trees of Geurts, Ernst and
Wehenkel, 2006). The cut is chosen without looking at the outcome, so a
node costs one score for each feature rather than one for each distinct
value, and the trees of a forest differ more from each other. The kind,
``selection``, leaf models and minimum leaf sizes are unchanged, and the
draws come from ``random_state`` like the feature draws.

A forest keeps its bootstrap resampling under this rule (so its
out-of-bag scores exist); pass ``bootstrap=False`` for the original
Extra-Trees, in which every tree sees the whole sample. On the Rossi
recidivism data (432 rows) and the lung data (168 complete rows), forests
of 100 Weibull trees over 10 seeds had a mean out-of-bag log-likelihood
per observation of -1.604 searched and -1.604 random (the seed-to-seed
standard deviations were 0.005 and 0.003), and -2.169 searched and -2.155
random (0.009 and 0.006), with the random forests fitting in 0.55 of the
time (``scripts/studies/extra_trees_oob.py``).

.. code-block:: python

    forest = RandomSurvivalForest.fit(x, Z, c=c, splitrule="random",
                                      random_state=0)

A few things worth knowing:

- Fitted from a DataFrame (``fit_from_df`` with ``Z_cols`` or a
  ``formula``, or ``fit`` with a DataFrame ``Z``), a tree or forest keeps
  the covariate names as ``feature_names``; ``print(tree)`` shows its splits
  by name (``temp <= 42``), a forest's ``feature_importances()`` method
  returns a ``pandas.Series`` keyed by name, and predictions read a
  DataFrame by those names. Fitted from arrays the covariates are shown as
  ``Z0``, ``Z1``, ...
- A ``"weibull"`` or ``"exponential"`` tree grows until ``min_leaf_samples``
  or ``min_leaf_failures`` stops it, which suits a forest. For a tree used on
  its own, set ``min_split_gain="aic"`` (or ``"bic"``, or a log-likelihood
  gain), or use ``selection="ctree"``.
- On observed and right-censored data the likelihood splits are found
  directly (no optimiser per candidate), at a cost of the same order as the
  log-rank split's; with left or interval censoring or truncation each
  candidate needs an optimiser and growing a forest takes much longer. A
  parametric leaf is fitted when the tree first predicts, so the first
  prediction of a ``"weibull"`` forest can take longer than growing it.
- ``sf(x, Z)`` (and ``ff``, ``df``, ``hf``, ``Hf``) pair row ``i`` of a
  covariate matrix with the time ``x[i]``, as the regression models do: one
  row is used at every time and one time for every row, and other counts
  are refused. ``grid=True`` gives every time for every row, shape
  ``(len(Z),) + x.shape``, which was the default before 0.24.
- A forest grows its trees one after another by default. ``n_jobs`` (as in
  joblib and scikit-learn; ``-1`` for every core) grows them in worker
  processes; given a ``random_state`` the forest is the same whatever
  ``n_jobs`` is.

Competing-risks trees
---------------------

With several causes of failure, a tree grown on "any failure" splits on
whatever changes the overall risk, and its leaves cannot say which cause a
unit is at risk of. Give ``SurvivalTree.fit`` the cause of each failure as
``e`` (a missing value for a right-censored row; ``fit_from_df`` takes it
as ``e_col``) and the tree becomes a competing-risks tree:

- **Splits that tell the causes apart** (``split``). The default,
  ``"cause-specific"``, scores a cut by the sum over the causes of the
  squared cause-specific log-rank statistics (each cause's failures as the
  events, everything else censored): the chi-squared statistic of "this cut
  changes no cause's hazard", so a cut that raises one cause and lowers
  another counts, rather than cancelling. ``"gray"`` scores a cut by Gray's
  two-sample statistic for one nominated cause's cumulative incidence
  (``event=``); it targets the incidence itself, but costs a full Gray test
  per candidate cut, so it is much slower on a large node, and it refuses
  delayed entry, as ``gray_test`` does. With one cause the cause-specific
  rule is the plain log-rank split.
- **A competing-risks model in each leaf.** For ``kind="non-parametric"``
  the Aalen-Johansen estimate (``CompetingRisks``, with the Kaplan-Meier
  all-cause survival, so ``sf`` is one less the sum of the incidences); for
  ``"weibull"`` or ``"exponential"`` one distribution per cause, each
  fitted to the cause-specific view of the leaf as a plain leaf of that
  kind is, assembled as ``ParametricCompetingRisks``. A cause with no
  failure in a leaf has zero incidence there, and ``print(tree)`` says so.
  ``min_leaf_failures`` counts every failure; for a parametric kind each
  child must also keep that many failures of each cause the node has at
  least twice that many of.
- **Predictions per cause.** ``cif(x, Z, event)`` is the cumulative
  incidence of a cause where the covariates put a unit, ``total_cif(x, Z)``
  the sum over the causes, and ``sf``, ``ff``, ``Hf``, ``hf`` and ``df``
  the all-cause functions; the shapes are those of ``sf``.

A tree with causes takes exact and right-censored times, with delayed
entry (``tl``) for the cause-specific rule. Left- and interval-censored
rows and right truncation are refused for now (fit ``CompetingRisks`` or
``ParametricCompetingRisks`` to such data), as are ``selection="ctree"``
and a non-zero ``min_split_gain``, whose tests and likelihoods are of a
single event. Without ``e`` a tree is exactly as before.

Competing-risks forests
-----------------------

``RandomSurvivalForest.fit`` takes ``e`` (and ``split``, ``event``) the
same way, and grows every tree as a competing-risks tree for every cause in
the data, even where a tree's bootstrap sample misses a rare cause (that
tree then gives it zero incidence).

- ``cif(x, Z, event)`` is the average over the trees of their leaf
  incidences of the cause, ``total_cif`` the sum over the causes, and
  ``sf`` the average all-cause survival.
- ``oob_log_likelihood()`` scores each failure by the ensemble's
  sub-density of *its own cause*, the slope of that cause's cumulative
  incidence at its time, and each censored row by the all-cause survival:
  the competing-risks likelihood, so a forest that gets the time right but
  the cause wrong scores badly. ``oob_log_likelihood(event=k)`` restricts
  it to cause ``k``: the cause-specific likelihood (a failure from ``k``
  scores the cause's own density, every other row its own survival), which
  judges the cause's hazard alone.
- ``feature_importances(event=k)`` shuffles each feature and measures the
  drop in that restricted score, so a feature that changes only one
  cause's risk is important for that cause and near zero for the others;
  without ``event`` it uses the full score.
- ``mortality(x, Z, event=k)`` sums the cause's incidence over the times
  (the expected time lost to the cause), and ``score(x, Z, c, e=e,
  event=k)`` is its concordance with the failures from ``k``, a failure
  from another cause counting as later than all of them (Wolbers et al.).

Reading a fitted forest
-----------------------

A forest predicts well but has no coefficients to read. Three read-outs
say what it has learnt, all from the fitted model, without refitting:

- **How the prediction moves with one covariate**:
  ``forest.partial_dependence(covariate, values, x)``. Each unit keeps its
  other covariates, the chosen one is set to each of ``values``, and its
  survival at the times ``x`` is predicted: one curve per unit, its
  *individual conditional expectation* (ICE). Their average is the
  *partial dependence*: the forest's average prediction for each value.
  Parallel ICE curves mean the covariate acts the same way for everyone;
  curves that cross or fan out point to an interaction the average hides.
  ``result.plot(at=t)`` draws both at one time (``centred=True`` starts
  every curve at 0 to compare shapes), and ``result.to_frame()`` gives the
  average as a table. Setting a covariate to a value rarely seen with a
  unit's other covariates asks the forest about a region it has little
  data in, so read the ends of the range with care.
- **Which covariates the trees split on first**: ``forest.minimal_depth()``.
  A covariate that matters is usually split on near the root, where a split
  divides the most units. Its *minimal depth* in a tree is the depth of its
  first split (the root is 0; a tree that never uses it gives it the tree's
  depth). The table gives the mean over the trees (smaller is more
  important) and the share of trees that use it at all. It needs no
  out-of-bag data, so it also works on a forest restored with
  ``from_dict``.

  *Which covariates to keep.* A covariate with no effect is still split on
  by chance, and the more nodes a tree has near the root, the sooner chance
  alone reaches it. ``forest.minimal_depth_threshold()`` (also
  ``result.attrs["threshold"]``) is the mean minimal depth you would expect
  from a covariate with no effect in these trees, and the table's
  ``selected`` column is true for a covariate whose mean minimal depth is
  below it: it is split on earlier than chance explains. With ``p``
  covariates, ``l_d`` the number of split nodes at depth ``d`` of a tree,
  ``D`` its depth and ``L_d = l_0 + ... + l_(d-1)``, such a covariate's
  minimal depth ``D_v`` has
  ``P(D_v = d) = (1 - 1/p)^L_d (1 - (1 - 1/p)^l_d)`` for
  ``d = 0, ..., D - 1``, and the rest of the probability at ``D`` (a tree
  that never splits on it, counted at ``D`` as in the table); the
  threshold is the mean of this, worked out tree by tree and averaged over
  the trees. ``1/p`` is the chance a given covariate is the one split on
  at a node, whatever ``n_features_split`` is. It is a guide rather than a
  test: it assumes the covariates are interchangeable, which correlated
  covariates are not. Ishwaran, Kogalur, Gorodeski, Minn and Lauer (2010),
  "High-dimensional variable selection for survival data", *Journal of the
  American Statistical Association* 105(489), 205-217.
- **How much each covariate matters, at each horizon**:
  ``forest.feature_importances(times=[1, 5, 10])``. Each covariate is
  shuffled among each tree's out-of-bag units, which breaks its link with
  the outcome, and the out-of-bag Brier score at each time is computed
  again; the importance is how much it gets worse. A covariate can matter
  for early failures and not late ones. ``metric="auc"`` uses the
  time-dependent AUC instead. Without ``times`` the importance is the
  out-of-bag log-likelihood one, a single number per covariate.

``SurvivalTree`` has ``partial_dependence`` and ``minimal_depth`` too.

Random Survival Forest
----------------------

.. autoclass:: surpyval.ml.forest.forest.RandomSurvivalForest
   :members:

Survival Tree
-------------

.. autoclass:: surpyval.ml.forest.tree.SurvivalTree
   :members:

Tree Nodes
----------

The nodes a fitted tree is built from. Users do not normally construct
these directly; they are documented because a serialised tree is a
nested structure of them, and because ``TerminalNode.model`` is how a
leaf's fitted distribution is reached.

.. autoclass:: surpyval.ml.forest.node.Node
   :members:

.. autoclass:: surpyval.ml.forest.node.IntermediateNode
   :members:

.. autoclass:: surpyval.ml.forest.node.TerminalNode
   :members:

.. autoclass:: surpyval.ml.forest.node.CompetingRisksLeaf
   :members:

Read-outs
---------

.. automodule:: surpyval.ml.forest.readouts
   :members: PartialDependence

Cause-aware splits
------------------

.. automodule:: surpyval.ml.forest.cause_specific_split
   :members: cause_specific_split, gray_split
