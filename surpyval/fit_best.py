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


def _candidate_names(names: Iterable[str] | None, argument: str) -> set[str]:
    """The lower-cased distribution names in ``include`` / ``exclude``.

    Matched without regard to case, and checked: a name that is not a
    candidate (a typo, or a distribution ``fit_best`` does not try) used
    to leave ``include`` with nothing to fit and ``fit_best`` returning
    ``None`` in silence.
    """
    if names is None:
        return set()
    if isinstance(names, str):
        # A bare name, not an iterable of its characters
        names = [names]
    wanted = {str(name).lower() for name in names}
    known = {dist.name.lower() for dist in distributions}
    unknown = sorted(wanted - known)
    if unknown:
        raise ValueError(
            f"Unknown distribution name(s) in `{argument}`: {unknown}. "
            "fit_best tries "
            f"{sorted(dist.name for dist in distributions)}."
        )
    return wanted


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
        Only try distributions with these names (matched without regard
        to case; a name that is not a candidate raises a ``ValueError``).
        Mutually exclusive with ``exclude``.
    exclude : iterable of str, optional
        Try every candidate except distributions with these names, checked
        in the same way. Mutually exclusive with ``include``.

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
    include_set = _candidate_names(include, "include")
    exclude_set = _candidate_names(exclude, "exclude")

    if metric not in METRICS:
        raise ValueError(
            '`metric` must be on of "{}"'.format('", "'.join(METRICS))
        )

    if (len(include_set) > 0) and (len(exclude_set) > 0):
        raise ValueError("Provide either an include or an exclude, not both.")

    if len(exclude_set) > 0:
        candidates = [
            dist
            for dist in distributions
            if dist.name.lower() not in exclude_set
        ]
    elif len(include_set) > 0:
        candidates = [
            dist for dist in distributions if dist.name.lower() in include_set
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
