from typing import Any

from .concordance import (
    ConcordanceComparison,
    ConcordanceResult,
    compare_concordance,
    concordance_index,
)
from .conformal import ConformalLowerBound, conformal_lower_bound
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
    "ConformalLowerBound",
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
    "conformal_lower_bound",
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
        from surpyval.utils.removed_names import (
            CONCORDANCE_ESTIMATE_REMOVED,
            from_import,
            removed_message,
        )

        message = removed_message(
            f"module 'surpyval.metrics' has no attribute {name!r}",
            CONCORDANCE_ESTIMATE_REMOVED,
        )
        if from_import():
            raise ImportError(
                message.replace(
                    "module 'surpyval.metrics' has no attribute",
                    "cannot import name",
                    1,
                ),
                name="surpyval.metrics",
            )
        raise AttributeError(message)
    raise AttributeError(
        f"module 'surpyval.metrics' has no attribute {name!r}"
    )
