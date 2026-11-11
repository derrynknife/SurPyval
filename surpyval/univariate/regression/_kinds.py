"""The former private home of the regression family names.

They are public in :mod:`surpyval.univariate.regression.kinds`; this
module keeps the old import path working.
"""

from .kinds import (  # noqa: F401
    ACCELERATED_FAILURE_TIME,
    ACCELERATED_LIFE,
    ADDITIVE_HAZARD,
    DISTRIBUTIONAL,
    HAZARD_BASED,
    PROPORTIONAL_HAZARD,
    PROPORTIONAL_ODDS,
    TIME_SCALING,
)
