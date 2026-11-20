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
single event. Without ``e`` a tree is exactly as before. The forest does
not take causes yet.

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

Cause-aware splits
------------------

.. automodule:: surpyval.ml.forest.cause_specific_split
   :members: cause_specific_split, gray_split
