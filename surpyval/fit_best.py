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
    OptimisedFitMixin,
    Parametric,
    Rayleigh,
    Uniform,
    Weibull,
)

# Typed as OptimisedFitMixin, not ParametricFitter: every entry has
# `.fit(x, c, n, t)` called on it below, and Bernoulli, Binomial and
# ExactEventTime do not have that signature. Under the old annotation
# adding one of them here type checked and failed at runtime.
distributions: list[OptimisedFitMixin] = [
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
    """
    Fit every candidate continuous distribution to the data and return
    the fitted model with the best value of ``metric``.

    The candidates are the fittable continuous univariate distributions
    (Beta, Beta4, Exponential, ExpoWeibull, Gamma, Gumbel, Logistic,
    LogLogistic, LogNormal, Normal, Rayleigh, Uniform and Weibull).
    Distributions whose fit fails or does not converge are skipped with
    a warning; if every candidate fails, ``None`` is returned.

    Parameters
    ----------
    x : array_like
        The observed event times (or intervals), in any of the formats
        ``fit`` accepts.
    c : array_like, optional
        The censoring indicators.
    n : array_like, optional
        The counts for each observation.
    t : array_like, optional
        The truncation intervals.
    metric : str, optional
        The model-selection criterion to minimise: ``"aic"`` (default),
        ``"aic_c"``, ``"bic"`` or ``"neg_ll"``.
    include : iterable of str, optional
        Only try distributions with these names. Mutually exclusive
        with ``exclude``.
    exclude : iterable of str, optional
        Try every candidate except distributions with these names.
        Mutually exclusive with ``include``.

    Returns
    -------
    Parametric or None
        The fitted model that minimises ``metric``, or ``None`` when no
        candidate converged.

    Examples
    --------
    >>> from surpyval import fit_best
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> from surpyval import Weibull
    >>> x = Weibull.random(50, 10, 2)
    >>> model = fit_best(x, metric="bic")
    """
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
