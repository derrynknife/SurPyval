"""Machine-learning survival models.

Home of SurPyval's tree-based models: the survival tree and the random
survival forest, which split the data on its covariates and fit a
survival model at each leaf. See ``surpyval.ml.forest`` for the
implementation.

Promoted from the beta tier (``surpyval.beta.ml``) in v0.27: the old name
still imports them, with a ``FutureWarning``, until the first release on
or after 12 April 2027 (``surpyval.utils.deprecation.BETA_ML_REMOVED_AFTER``).
"""

from .forest import RandomSurvivalForest, SurvivalTree

__all__ = ["RandomSurvivalForest", "SurvivalTree"]
