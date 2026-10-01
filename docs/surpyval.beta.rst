Machine Learning (beta)
=======================

Tree-based survival models: a survival tree, and a random survival
forest built from them. Both accept the full surpyval data model --
arbitrary censoring and truncation -- and return a fitted model at each
leaf (parametric, or a Nelson-Aalen or Turnbull estimate for
``kind="non-parametric"``), so a prediction is a distribution rather
than a point.

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

.. warning::

   These live under ``surpyval.beta`` because their API is not yet
   settled and may change between minor versions without the deprecation
   cycle the rest of the package follows. They are tested and usable;
   they are not covered by the same stability promise.

For a narrative introduction see the survival-forest section of
:doc:`Regression Modelling with SurPyval`.

Random Survival Forest
----------------------

.. autoclass:: surpyval.beta.ml.forest.forest.RandomSurvivalForest
   :members:

Survival Tree
-------------

.. autoclass:: surpyval.beta.ml.forest.tree.SurvivalTree
   :members:

Tree Nodes
----------

The nodes a fitted tree is built from. Users do not normally construct
these directly; they are documented because a serialised tree is a
nested structure of them, and because ``TerminalNode.model`` is how a
leaf's fitted distribution is reached.

.. autoclass:: surpyval.beta.ml.forest.node.Node
   :members:

.. autoclass:: surpyval.beta.ml.forest.node.IntermediateNode
   :members:

.. autoclass:: surpyval.beta.ml.forest.node.TerminalNode
   :members:
