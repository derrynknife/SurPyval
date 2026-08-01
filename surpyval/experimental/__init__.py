"""Deprecated alias package.

``surpyval.experimental`` was renamed: the survival tree / random
survival forest now live in ``surpyval.beta.ml``. This module
re-exports them for backwards compatibility and will be removed in a
future release. The former ``SeriesModel``/``ParallelModel``
reliability-block composition was removed entirely in v0.17.0 (#284);
reliability block diagrams are covered by the Repyability package.
"""

import warnings

from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree

warnings.warn(
    "surpyval.experimental is deprecated: use surpyval.beta.ml "
    "(SurvivalTree, RandomSurvivalForest) instead.",
    DeprecationWarning,
    stacklevel=2,
)
