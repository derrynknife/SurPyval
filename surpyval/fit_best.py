import warnings
from collections.abc import Iterable

import numpy as np
import numpy.typing as npt

from surpyval.univariate.parametric import (
    Beta,
    Beta4,
    Exponential,
    ExpoWeibull,
    Gamma,
    Gumbel,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Parametric,
    ParametricFitter,
    Rayleigh,
    Uniform,
    Weibull,
)

distributions: list[ParametricFitter] = [
    Beta,
    Beta4,
    Exponential,
    ExpoWeibull,
    Gamma,
    Gumbel,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Rayleigh,
    Uniform,
    Weibull,
]

METRICS = ["aic", "aic_c", "bic", "neg_ll"]


def fit_best(
    x: npt.ArrayLike,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    metric: str = "aic",
    include: Iterable[str] | None = None,
    exclude: Iterable[str] | None = None,
) -> Parametric | None:
    include_set = set(include) if include is not None else set()
    exclude_set = set(exclude) if exclude is not None else set()

    if metric not in METRICS:
        raise ValueError(
            '`metric` must be on of "{}"'.format('", "'.join(METRICS))
        )

    if (len(include_set) > 0) and (len(exclude_set) > 0):
        raise ValueError("Provide either an include or an exclude, not both.")

    if len(exclude_set) > 0:
        candidates = [
            dist for dist in distributions if dist.name not in exclude_set
        ]
    elif len(include_set) > 0:
        candidates = [
            dist for dist in distributions if dist.name in include_set
        ]
    else:
        candidates = distributions

    measure = np.inf
    model: Parametric | None = None
    for dist in candidates:
        try:
            temp_model = dist.fit(x, c, n, t)
            tmp_measure = getattr(temp_model, metric)()
        except Exception as e:
            warnings.warn(str(e))
            warnings.warn(f"{dist.name} distribution failed to fit")
            continue
        if tmp_measure < measure:
            measure = tmp_measure
            model = temp_model
    return model
