"""The old name of :mod:`surpyval.ml`.

The survival tree and random survival forest were promoted from the beta
tier to the stable surface in v0.27, as ``surpyval.ml``. This module
imports them from there, with one ``FutureWarning`` at the import that
asked for it, until the first release on or after 12 April 2027
(``surpyval.utils.deprecation.BETA_ML_REMOVED_AFTER``). It also keeps the
old submodule names (``surpyval.beta.ml.forest.tree`` and the rest)
importable, so that a pickle of a tree or forest saved by 0.26 still
loads.
"""

import sys

from surpyval.ml import RandomSurvivalForest, SurvivalTree, forest
from surpyval.utils.deprecation import beta_ml_alias as _beta_ml_alias

__all__ = ["RandomSurvivalForest", "SurvivalTree"]

# The same module objects under their old names: ``surpyval.ml`` has
# imported every one of its submodules by now.
_OLD, _NEW = "surpyval.beta.ml.", "surpyval.ml."
for _name, _module in list(sys.modules.items()):
    if _name.startswith(_NEW):
        sys.modules.setdefault(_OLD + _name[len(_NEW) :], _module)
del _name, _module

_beta_ml_alias(__file__)
