import warnings
from typing import List

import numpy as np

from surpyval.univariate.parametric import (
    Beta,
    Exponential,
    ExpoWeibull,
    Gamma,
    Gumbel,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    ParametricFitter,
    Rayleigh,
    Uniform,
    Weibull,
)

distributions: List[ParametricFitter] = [
    Beta,
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


def fit_best(x, c=None, n=None, t=None, metric="aic", include=[], exclude=[]):
    include = set(include)
    exclude = set(exclude)

    if metric not in METRICS:
        raise ValueError(
            '`metric` must be on of "{}"'.format('", "'.join(METRICS))
        )

    if (len(include) > 0) & (len(exclude) > 0):
        raise ValueError("Provide either an include or an exclude, not both.")

    if len(exclude) > 0:
        include = [dist for dist in distributions if dist.name not in exclude]
    elif len(include) > 0:
        include = [dist for dist in distributions if dist.name in include]
    else:
        include = distributions

    measure = np.inf
    model = None
    for dist in include:
        try:
            temp_model = dist.fit(x, c, n, t)
            tmp_measure = getattr(temp_model, metric)()
        except Exception as e:
            warnings.warn(str(e))
            warnings.warn("{} distribution failed to fit".format(dist.name))
            continue
        if tmp_measure < measure:
            measure = tmp_measure
            model = temp_model
    return model
