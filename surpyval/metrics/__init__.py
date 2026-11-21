from typing import Any

from .concordance import (
    ConcordanceComparison,
    ConcordanceResult,
    compare_concordance,
    concordance_index,
)
from .dynamic import (
    DynamicAccuracy,
    DynamicCalibration,
    dynamic_auc,
    dynamic_brier,
    dynamic_calibration,
)
from .validation import (
    PredictionScores,
    Score,
    TimeScores,
    auc_td,
    brier_score,
    integrated_brier_score,
    score_predictions,
    survival_probability,
)

__all__ = [
    "ConcordanceComparison",
    "ConcordanceResult",
    "DynamicAccuracy",
    "DynamicCalibration",
    "PredictionScores",
    "Score",
    "TimeScores",
    "auc_td",
    "brier_score",
    "compare_concordance",
    "concordance_index",
    "dynamic_auc",
    "dynamic_brier",
    "dynamic_calibration",
    "integrated_brier_score",
    "score_predictions",
    "survival_probability",
]


def __getattr__(name: str) -> Any:
    if name == "concordance_estimate":
        from surpyval.utils.deprecation import REMOVED_IN_NEXT, renamed_name

        from .concordance import _concordance_estimate

        return renamed_name(
            "surpyval.metrics",
            name,
            "concordance_index(x, c, risk, weighting=..., tau=..., "
            "alpha_ci=...)",
            _concordance_estimate,
            REMOVED_IN_NEXT,
            FutureWarning,
        )
    raise AttributeError(
        f"module 'surpyval.metrics' has no attribute {name!r}"
    )
