:orphan:

Machine Learning (beta)
=======================

``surpyval.beta.ml`` is the old name of :doc:`surpyval.ml`. The survival
tree and the random survival forest were promoted from the beta tier to
the stable surface in v0.27, and now follow the same stability promise and
deprecation cycle as the rest of the package. The old name still imports
them, with a ``FutureWarning`` at the import, until the first release on
or after 12 April 2027: change

.. code-block:: python

    from surpyval.beta.ml import SurvivalTree, RandomSurvivalForest

to

.. code-block:: python

    from surpyval.ml import SurvivalTree, RandomSurvivalForest

A tree or forest saved by an earlier version (``to_json``, ``to_dict`` or
a pickle) still loads. The ``surpyval.beta`` tier is empty.
