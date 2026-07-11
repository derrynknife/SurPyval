"""Degradation analysis.

Classic (pseudo-failure-time) degradation analysis: fit a degradation
path model to each unit's repeated measurements, extrapolate each
fitted path to the failure threshold to get per-unit pseudo failure
times, and fit a lifetime distribution to those times.

Examples
--------

>>> from surpyval.degradation import DegradationAnalysis
>>> model = DegradationAnalysis.fit(x, y, i, threshold=150)  # doctest: +SKIP
"""

from .path_models import (
    PATH_MODELS,
    ExponentialPath,
    GompertzPath,
    LinearPath,
    LloydLipowPath,
    LogarithmicPath,
    MichaelisMentenPath,
    OffsetExponentialPath,
    PathModel,
    PowerPath,
    QuadraticPath,
    get_path_model,
)

from .degradation_analysis import (  # isort: skip
    DegradationAnalysis,
    DegradationModel,
    RULPrediction,
)
