"""Deprecated alias package.

``surpyval.experimental`` was renamed: the survival tree / random
survival forest now live in ``surpyval.beta.ml``. This module
re-exports them for backwards compatibility and will be removed in
v0.22.0. The former ``SeriesModel``/``ParallelModel``
reliability-block composition was removed entirely in v0.17.0 (#284);
reliability block diagrams are covered by the Repyability package.
"""

import warnings

from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree
from surpyval.utils.deprecation import REMOVED_IN

warnings.warn(
    "surpyval.experimental is deprecated and will be removed in v{}: use "
    "surpyval.beta.ml (SurvivalTree, RandomSurvivalForest) "
    "instead.".format(REMOVED_IN),
    DeprecationWarning,
    stacklevel=2,
)
