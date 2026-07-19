"""Deprecated alias package.

``surpyval.experimental`` was renamed: the alpha-stage models live in
``surpyval.alpha`` and the survival tree / random survival forest in
``surpyval.beta.ml``. This module re-exports both for backwards
compatibility and will be removed in a future release.
"""

import warnings

from surpyval.alpha import ParallelModel, SeriesModel
from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree

warnings.warn(
    "surpyval.experimental is deprecated: use surpyval.alpha "
    "(ParallelModel, SeriesModel) or surpyval.beta.ml "
    "(SurvivalTree, RandomSurvivalForest) instead.",
    DeprecationWarning,
    stacklevel=2,
)
