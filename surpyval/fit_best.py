from __future__ import annotations

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
from surpyval.univariate.parametric.parametric_fitter import (
    OutsideSupportError,
)
from surpyval.utils.no_maximum import quiet_maximum_warnings
from surpyval.utils.validation import check_option

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


def _non_regular(dist: OptimisedFitMixin) -> bool:
    """Whether the family's support ends are among its parameters.

    Such a family (the Uniform, the Beta4) breaks the regularity
    conditions behind AIC and BIC: its likelihood is highest with a
    support end on an extreme observation, where the log-likelihood has
    no zero gradient and is not approximately quadratic, so the ``2k``
    penalty does not measure its optimism. Its support is resolved from
    its fitted parameters, which the fitter marks with a ``nan`` support.
    """
    return bool(np.isnan(np.asarray(dist.support, dtype=float)).any())


# Tried only when named in ``include``
NON_REGULAR = [dist.name for dist in distributions if _non_regular(dist)]


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
    with a regular likelihood (Beta, Exponential, ExpoWeibull, Gamma,
    Gumbel, Logistic, LogLogistic, LogNormal, Normal, Rayleigh and
    Weibull). A candidate the data lie outside the support of (a Beta
    for data outside (0, 1)) is passed over quietly; candidates whose fit
    fails are skipped and named in one warning. If every candidate fails,
    ``None`` is returned.

    AIC, AIC_c and BIC compare maximised log-likelihoods, and their
    penalties assume a *regular* maximum: an interior point of the
    parameter space with a zero gradient near which the log-likelihood is
    quadratic, and a support that does not depend on the parameters. Two
    kinds of candidate break that, and are set aside -- ranked only when
    no regular candidate fitted:

    - a family whose support ends are parameters (Uniform, Beta4). Its
      likelihood is highest with an end on the extreme observations,
      where the ``2k`` penalty undercounts: on 50 Weibull(100, 2) draws
      the Uniform's AIC beat the Weibull's by 14. These are left out of
      the default candidates and tried only when named in ``include``;
    - a fit that is not a verified maximum, as its ``maximum`` attribute
      records (see :class:`~surpyval.univariate.parametric.parametric.\
Parametric`): ``"no finite maximum"`` (a Beta4 whose shape falls below
      1, say), whose likelihood has no maximum, so its value, and every
      criterion made from it, means nothing -- on ``[1, ..., 7]`` the
      Beta4 "won" with a log-likelihood of +24.8 against the Weibull's
      -14.6 -- or ``"unverified"``, a search that did not reach a
      verified maximum (an ExpoWeibull running towards a limit of its
      shapes, say), whose value is only where the search stopped.

    When a candidate is set aside, one warning names it and says why; the
    warning its own fit would give that it is not a verified maximum is
    held back, replaced by that one.

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
        The only way to try the Uniform or the Beta4, which are then set
        aside (see above). Mutually exclusive with ``exclude``.
    exclude : iterable of str, optional
        Try every candidate except distributions with these names, checked
        in the same way. Mutually exclusive with ``include``.

    Returns
    -------
    Parametric or None
        The fitted model that minimises ``metric``, or ``None`` when no
        candidate converged.

    Raises
    ------
    ValueError
        If candidates fitted but none has a finite ``metric`` -- for
        ``"aic_c"``, when every candidate has at least as many parameters
        as observed failures minus one.

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

    check_option("metric", metric, METRICS)

    if (len(include_set) > 0) and (len(exclude_set) > 0):
        raise ValueError("Provide either an include or an exclude, not both.")

    if len(exclude_set) > 0:
        candidates = [
            dist
            for dist in distributions
            if dist.name.lower() not in exclude_set
            and dist.name not in NON_REGULAR
        ]
    elif len(include_set) > 0:
        candidates = [
            dist for dist in distributions if dist.name.lower() in include_set
        ]
    else:
        candidates = [
            dist for dist in distributions if dist.name not in NON_REGULAR
        ]

    # The best (measure, model) among the regular fits (True) and among
    # those set aside (False), which are ranked only when no regular
    # candidate fitted.
    best: dict[bool, tuple[float, Parametric | None]] = {
        True: (np.inf, None),
        False: (np.inf, None),
    }
    set_aside: list[str] = []
    n_fitted = 0
    failed: list[str] = []
    for dist in candidates:
        failure = None
        # A candidate's own warning that its fit is not a verified maximum
        # is held back: the fitted model records it (``maximum``), and the
        # one warning below says it for every candidate set aside.
        with (
            warnings.catch_warnings(record=True) as caught,
            quiet_maximum_warnings(),
        ):
            warnings.simplefilter("always")
            try:
                temp_model = dist.fit(x, c, n, t)
                tmp_measure = getattr(temp_model, metric)()
            except OutsideSupportError:
                # A candidate that cannot describe the data (a Beta for
                # data outside (0, 1)) is not a failure to report (#485).
                continue
            except Exception as e:
                failure = str(e)
        if failure is not None:
            # A failed candidate's other warnings go with it
            failed.append(f"{dist.name} ({failure})")
            continue
        for w in caught:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        n_fitted += 1
        maximum = temp_model.maximum
        if maximum == "no finite maximum":
            set_aside.append(
                f"{dist.name} (its likelihood has no finite maximum)"
            )
        elif maximum != "verified":
            set_aside.append(
                f"{dist.name} (its fit is not a verified maximum)"
            )
        elif _non_regular(dist):
            set_aside.append(
                f"{dist.name} (its support ends are parameters, fitted "
                "on the extreme observations)"
            )
        regular = maximum == "verified" and not _non_regular(dist)
        if tmp_measure < best[regular][0]:
            best[regular] = (tmp_measure, temp_model)
    model = best[True][1]
    if model is None:
        model = best[False][1]
    if failed:
        # One warning for all of them, with the count (principle 22); it
        # was two warnings per candidate.
        warnings.warn(
            f"fit_best skipped {len(failed)} candidate(s) that failed to "
            "fit: " + "; ".join(failed),
            stacklevel=2,
        )
    if set_aside:
        chosen = "none" if model is None else model.dist.name
        warnings.warn(
            f"fit_best set aside {', '.join(set_aside)}: {metric} assumes "
            "a regular maximum of the likelihood, which these fits do not "
            "have, so they are ranked only when no regular candidate "
            f"fitted. Chosen: {chosen}.",
            stacklevel=2,
        )
    if model is None and n_fitted > 0:
        # Every candidate fitted but none has a finite value of the
        # metric: AIC_c is undefined (nan) once the sample size d is at
        # most k + 1, which with heavy censoring can hold for every
        # candidate. Returning None here read as "nothing converged".
        raise ValueError(
            f"{n_fitted} candidate(s) fitted, but none has a finite "
            f"{metric!r}. AIC_c needs more observed failures than "
            "parameters plus one; compare with metric='aic' instead."
            if metric == "aic_c"
            else f"{n_fitted} candidate(s) fitted, but none has a finite "
            f"{metric!r}."
        )
    return model
