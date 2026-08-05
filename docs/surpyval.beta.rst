Machine Learning (beta)
=======================

Tree-based survival models: a survival tree, and a random survival
forest built from them. Both accept the full surpyval data model --
arbitrary censoring and truncation -- and return a fitted parametric
model at each leaf, so a prediction is a distribution rather than a
point.

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
