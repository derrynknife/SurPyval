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
from .pit import DCalibration, d_calibration, pit_values
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
    "DCalibration",
    "DynamicAccuracy",
    "DynamicCalibration",
    "PredictionScores",
    "Score",
    "TimeScores",
    "auc_td",
    "brier_score",
    "compare_concordance",
    "concordance_index",
    "d_calibration",
    "dynamic_auc",
    "dynamic_brier",
    "dynamic_calibration",
    "integrated_brier_score",
    "pit_values",
    "score_predictions",
    "survival_probability",
]


def __getattr__(name: str) -> Any:
    if name == "concordance_estimate":
        from surpyval.utils.deprecation import REMOVED_IN, renamed_name

        from .concordance import concordance_with_se

        return renamed_name(
            "surpyval.metrics",
            name,
            "concordance_index(x, c, risk, weighting=..., tau=..., "
            "alpha_ci=...)",
            concordance_with_se,
            REMOVED_IN,
            FutureWarning,
        )
    raise AttributeError(
        f"module 'surpyval.metrics' has no attribute {name!r}"
    )
