from __future__ import annotations

import warnings
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any, Callable, Literal, overload

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
    MixtureModel,
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
from surpyval.utils.surpyval_data import SurpyvalData
from surpyval.utils.validation import check_option

if TYPE_CHECKING:
    # Imported where it is used: ``import surpyval`` does not load pandas
    import pandas as pd

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


def _split_include(
    include: Iterable[Any] | None,
) -> tuple[list[Any] | None, list[MixtureModel]]:
    """``include`` as its distribution names and its mixtures.

    A mixture is named by a model of its components,
    ``MixtureModel(Weibull, 2)`` (#613): it is a candidate only when named
    so, and is fitted afresh, leaving the model given as it was. A bare
    name or model is taken as a list of one.
    """
    if include is None:
        return None, []
    if isinstance(include, (str, MixtureModel)):
        include = [include]
    names: list[Any] = []
    mixtures: list[MixtureModel] = []
    for item in include:
        if isinstance(item, MixtureModel):
            mixtures.append(item)
        elif isinstance(item, type) and issubclass(item, MixtureModel):
            raise ValueError(
                "`include` takes a mixture as a model of its components, "
                "e.g. MixtureModel(surpyval.Weibull, 2), not the "
                "MixtureModel class."
            )
        else:
            names.append(item)
    return names, mixtures


def _mixture_label(model: MixtureModel) -> str:
    """How a mixture candidate is named in ``fit_best``'s warnings."""
    return f"MixtureModel({model.dist.name}, {model.m})"


def _fit_mixture(model: MixtureModel, data: dict[str, Any]) -> MixtureModel:
    """A fresh mixture of ``model``'s components fitted to ``data``; the
    model given stays as it was."""
    return MixtureModel(dist=model.dist, m=model.m).fit(**data)


def _candidate_names(names: Iterable[Any] | None, argument: str) -> set[str]:
    """The lower-cased distribution names in ``include`` / ``exclude``.

    Matched without regard to case, and checked: a name that is not a
    candidate (a typo, or a distribution ``fit_best`` does not try) used
    to leave ``include`` with nothing to fit and ``fit_best`` returning
    ``None`` in silence.
    """
    if names is None:
        return set()
    if isinstance(names, (str, MixtureModel)):
        # A bare name, not an iterable of its characters
        names = [names]
    names = list(names)
    if any(isinstance(name, MixtureModel) for name in names):
        raise ValueError(
            f"`{argument}` takes distribution names only; a mixture is a "
            "candidate only when named in `include`."
        )
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


def _reason(error: Exception) -> str:
    """A failed candidate's reason, short: the exception's type and the
    first line of its message, cut at 100 characters (a message can quote
    the offending data, #570)."""
    text = str(error).strip().splitlines()
    first = text[0] if text else ""
    if len(first) > 100:
        first = first[:97] + "..."
    return (
        f"{type(error).__name__}: {first}" if first else type(error).__name__
    )


@overload
def fit_best(
    x: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    metric: str = "aic",
    include: Iterable[str | MixtureModel] | None = None,
    exclude: Iterable[str] | None = None,
    tl: npt.ArrayLike | float | None = None,
    tr: npt.ArrayLike | float | None = None,
    xl: npt.ArrayLike | None = None,
    xr: npt.ArrayLike | None = None,
    return_table: Literal[False] = ...,
) -> Parametric | MixtureModel | None: ...


@overload
def fit_best(
    x: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    metric: str = "aic",
    include: Iterable[str | MixtureModel] | None = None,
    exclude: Iterable[str] | None = None,
    tl: npt.ArrayLike | float | None = None,
    tr: npt.ArrayLike | float | None = None,
    xl: npt.ArrayLike | None = None,
    xr: npt.ArrayLike | None = None,
    *,
    return_table: Literal[True],
) -> tuple[Parametric | MixtureModel | None, pd.DataFrame]: ...


def fit_best(
    x: npt.ArrayLike | None = None,
    c: npt.ArrayLike | None = None,
    n: npt.ArrayLike | None = None,
    t: npt.ArrayLike | None = None,
    metric: str = "aic",
    include: Iterable[str | MixtureModel] | None = None,
    exclude: Iterable[str] | None = None,
    tl: npt.ArrayLike | float | None = None,
    tr: npt.ArrayLike | float | None = None,
    xl: npt.ArrayLike | None = None,
    xr: npt.ArrayLike | None = None,
    return_table: bool = False,
) -> (
    Parametric
    | MixtureModel
    | None
    | tuple[Parametric | MixtureModel | None, pd.DataFrame]
):
    """
    Fit every candidate continuous distribution to the data and return
    the fitted model with the best value of ``metric``.

    The candidates are the fittable continuous univariate distributions
    with a regular likelihood (Beta, Exponential, ExpoWeibull, Gamma,
    Gumbel, Logistic, LogLogistic, LogNormal, Normal, Rayleigh and
    Weibull). The data are given as to ``fit`` (``x``, ``c``, ``n``,
    ``t``, ``tl``, ``tr``, ``xl``, ``xr``) and are checked once, as
    ``fit`` checks them: an input error -- a malformed censoring flag,
    say -- raises the ``ValueError`` that ``Weibull.fit`` would, rather
    than failing every candidate. A candidate the data lie outside the
    support of (a Beta for data outside (0, 1)) is passed over quietly,
    except that a warning names the lifetime families (support from 0)
    passed over because some times are at or below 0, and says what was
    chosen instead -- often a family that puts failures before time 0 --
    and how to fit units that failed at time 0 (``zi=True``; #646); a
    candidate that cannot be fitted to the (valid) data is skipped, and
    the skipped candidates are named, each with its reason, in one
    warning. If no candidate fitted, ``None`` is returned -- unless every
    candidate failed for the same reason, which is then about the data
    and is raised, or the data lie outside the support of every
    candidate tried, which raises a ``ValueError`` saying so.

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

    By default ``fit_best`` compares single families. A mixture is a
    candidate only when named in ``include`` as a model of its
    components, ``MixtureModel(Weibull, 2)`` (#613): it is fitted to the
    same data and ranked on the same criterion, which a mixture has as
    every model does (#572), its parameters counting each component's
    and the free weights. Whether the data are one population or two is
    then decided on that criterion, as between any two candidates.

    Parameters
    ----------
    x : array_like, optional
        The observed event times (or intervals), in any of the formats
        ``fit`` accepts. If not given, ``xl`` and ``xr`` must be.
    c : array_like, optional
        The censoring indicators.
    n : array_like, optional
        The counts for each observation.
    t : array_like, optional
        The truncation intervals, ``[tl, tr]`` per row.
    metric : str, optional
        The model-selection criterion to minimise: ``"aic"`` (default),
        ``"aic_c"``, ``"bic"`` or ``"neg_ll"``.
    include : iterable of str or MixtureModel, optional
        Only try distributions with these names (matched without regard
        to case; a name that is not a candidate raises a ``ValueError``),
        and the mixtures given as models of their components,
        ``MixtureModel(Weibull, 2)``, each fitted afresh (the model given
        is left as it was). The only way to try the Uniform or the Beta4
        (which are then set aside, see above) or a mixture. Mutually
        exclusive with ``exclude``. An empty ``include`` is refused.
    exclude : iterable of str, optional
        Try every candidate except distributions with these names, checked
        in the same way. Mutually exclusive with ``include``.
    tl, tr : array_like or scalar, optional
        The left and right truncation of each row (or of every row), as
        for ``fit``.
    xl, xr : array_like, optional
        The left and right ends of each observation, in place of ``x``,
        as for ``fit``.
    return_table : bool, optional
        If True, return ``(model, table)``: the model as below and a
        :class:`pandas.DataFrame` ranking every candidate tried (#666).
        Default False, which returns the model alone.

    Returns
    -------
    Parametric, MixtureModel or None
        The fitted model that minimises ``metric`` (a ``MixtureModel``
        only when one was named in ``include``), or ``None`` when no
        candidate converged.
    pandas.DataFrame
        Only with ``return_table=True``: one row per candidate, best
        first, with columns ``model`` (the candidate's name), the
        criterion (named by ``metric``), ``delta`` (its value less the
        chosen model's), ``weight`` (for ``"aic"``, ``"aic_c"`` and
        ``"bic"``, the Akaike / Schwarz weight
        ``exp(-delta / 2)``, normalised over the ranked candidates; nan for
        ``"neg_ll"`` and for a candidate not ranked), ``status``
        (``"chosen"``, ``"ranked"``, ``"set aside"``, ``"failed"`` or
        ``"outside support"``) and ``reason`` (why a candidate was set
        aside, failed or passed over; empty otherwise). The fitted
        candidates come first, in order of the criterion, the ranked
        before the set aside; then the failed and those passed over.

    Raises
    ------
    ValueError
        If the data or an argument is invalid (as ``fit`` would say),
        every candidate failed for the same reason, the data lie outside
        the support of every candidate tried, or candidates fitted but
        none has a finite ``metric`` -- for ``"aic_c"``, when every
        candidate has at least as many parameters as observed failures
        minus one.

    Examples
    --------
    >>> from surpyval import fit_best
    >>> import numpy as np
    >>> np.random.seed(1)
    >>> from surpyval import Weibull
    >>> x = Weibull.random(50, 10, 2)
    >>> model = fit_best(x, metric="bic")

    Left truncated data, as ``fit`` takes it:

    >>> model = fit_best(x[x > 5], tl=5, include=["Weibull", "Gamma"])
    >>> model.dist.name
    'Weibull'

    A two-component Weibull mixture as a candidate beside the Weibull, on
    data from two populations:

    >>> from surpyval import MixtureModel
    >>> x = np.concatenate(
    ...     [Weibull.random(60, 5, 6), Weibull.random(60, 30, 6)]
    ... )
    >>> best = fit_best(x, include=["Weibull", MixtureModel(Weibull, 2)])
    >>> type(best).__name__, best.m
    ('MixtureModel', 2)

    The ranking of every candidate, with the criterion's gaps and weights:

    >>> best, table = fit_best(
    ...     x, include=["Weibull", "Exponential", "Gamma"], return_table=True
    ... )
    >>> list(table.columns)
    ['model', 'aic', 'delta', 'weight', 'status', 'reason']
    >>> table["model"].iloc[0] == best.dist.name
    True
    >>> table.loc[0, "status"], float(table.loc[0, "delta"])
    ('chosen', 0.0)
    """
    names, mixtures = _split_include(include)
    if include is not None and not names and not mixtures:
        raise ValueError(
            "`include` is empty: name the distributions to try, e.g. "
            "include=['Weibull', 'Gamma'], or leave it out to try the "
            "default candidates."
        )
    include_set = _candidate_names(names, "include")
    exclude_set = _candidate_names(exclude, "exclude")

    check_option("metric", metric, METRICS)

    if (include_set or mixtures) and exclude_set:
        raise ValueError("Provide either an include or an exclude, not both.")

    if len(exclude_set) > 0:
        candidates = [
            dist
            for dist in distributions
            if dist.name.lower() not in exclude_set
            and dist.name not in NON_REGULAR
        ]
    elif include_set or mixtures:
        candidates = [
            dist for dist in distributions if dist.name.lower() in include_set
        ]
    else:
        candidates = [
            dist for dist in distributions if dist.name not in NON_REGULAR
        ]

    # The data are checked once, as ``fit`` checks them, so an input error
    # raises as it would there. It used to fail every candidate the same
    # way, and come back as None with a warning quoting the data once per
    # candidate (#570). The same SurpyvalData is then given to every
    # candidate: each ``fit`` built and checked it again (and estimated
    # its non-parametric start again), most of fit_best's time on large
    # data that the fits themselves no longer take.
    surv_data = SurpyvalData(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)

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
    errors: list[Exception] = []
    outside: list[str] = []
    # Each candidate as (name, fit, whether its likelihood is regular).
    data: dict[str, Any] = dict(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)
    fits: list[tuple[str, Callable[[], Any], bool]] = [
        (
            dist.name,
            partial(dist.fit_from_surpyval_data, surv_data),
            not _non_regular(dist),
        )
        for dist in candidates
    ]
    fits += [
        (_mixture_label(mix), partial(_fit_mixture, mix, data), True)
        for mix in mixtures
    ]
    labels: dict[int, str] = {}
    # One row per candidate for ``return_table``: (name, measure, model,
    # status, reason); a fitted row's status is settled once the chosen
    # model is known.
    rows: list[tuple[str, float, Any, str, str]] = []
    for name, fit, regular_family in fits:
        failure: Exception | None = None
        # A candidate's own warning that its fit is not a verified maximum
        # is held back: the fitted model records it (``maximum``), and the
        # one warning below says it for every candidate set aside.
        with (
            warnings.catch_warnings(record=True) as caught,
            quiet_maximum_warnings(),
        ):
            warnings.simplefilter("always")
            try:
                temp_model = fit()
                tmp_measure = getattr(temp_model, metric)()
            except OutsideSupportError:
                # A candidate that cannot describe the data (a Beta for
                # data outside (0, 1)) is not a failure to report (#485).
                outside.append(name)
                rows.append(
                    (
                        name,
                        np.nan,
                        None,
                        "outside support",
                        "the data lie outside its support",
                    )
                )
                continue
            except Exception as e:
                failure = e
        if failure is not None:
            # A failed candidate's other warnings go with it
            failed.append(f"{name} ({_reason(failure)})")
            errors.append(failure)
            rows.append((name, np.nan, None, "failed", _reason(failure)))
            continue
        for w in caught:
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        n_fitted += 1
        labels[id(temp_model)] = name
        maximum = temp_model.maximum
        why = ""
        if maximum == "no finite maximum":
            why = "its likelihood has no finite maximum"
        elif maximum != "verified":
            why = "its fit is not a verified maximum"
        elif not regular_family:
            why = (
                "its support ends are parameters, fitted on the extreme "
                "observations"
            )
        if why:
            set_aside.append(f"{name} ({why})")
        rows.append((name, float(tmp_measure), temp_model, "", why))
        regular = maximum == "verified" and regular_family
        if tmp_measure < best[regular][0]:
            best[regular] = (tmp_measure, temp_model)
    model = best[True][1]
    if model is None:
        model = best[False][1]
    if n_fitted == 0:
        _raise_if_about_the_data(errors, outside)
    _warn_if_lifetimes_passed_over(
        outside, candidates, model, labels, surv_data
    )
    if failed:
        # One warning for all of them, with the count (principle 22); it
        # was two warnings per candidate.
        warnings.warn(
            f"fit_best skipped {len(failed)} candidate(s) that could not "
            "be fitted to these data: " + "; ".join(failed),
            stacklevel=2,
        )
    if set_aside:
        chosen = "none" if model is None else labels[id(model)]
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
    if return_table:
        return model, _ranking(rows, model, metric)
    return model


def _ranking(
    rows: list[tuple[str, float, Any, str, str]], model: Any, metric: str
) -> pd.DataFrame:
    """``fit_best``'s table of every candidate, best first (#666).

    The ranked candidates are those the chosen model was picked from: the
    regular fits, or the set-aside ones when no regular candidate fitted.
    """
    import pandas as pd

    chosen_measure = np.inf
    ranked_set_aside = False
    for _, measure, fitted, _, why in rows:
        if fitted is model and model is not None:
            chosen_measure = measure
            ranked_set_aside = bool(why)
    records: list[dict[str, Any]] = []
    for name, measure, fitted, status, why in rows:
        if not status:
            if fitted is model:
                status = "chosen"
            elif bool(why) == ranked_set_aside:
                status = "ranked"
            else:
                status = "set aside"
        records.append(
            {
                "model": name,
                metric: measure,
                "delta": measure - chosen_measure,
                "status": status,
                "reason": why,
            }
        )
    order = {
        "chosen": 0,
        "ranked": 0,
        "set aside": 1,
        "failed": 2,
        "outside support": 3,
    }
    records.sort(
        key=lambda r: (
            order[r["status"]],
            np.inf if np.isnan(r[metric]) else r[metric],
        )
    )
    table = pd.DataFrame(
        records, columns=["model", metric, "delta", "status", "reason"]
    )
    table[metric] = table[metric].astype(float)
    table["delta"] = table["delta"].astype(float)
    weight = np.full(len(table), np.nan)
    if metric != "neg_ll":
        ranked = table["status"].isin(["chosen", "ranked"]).to_numpy()
        delta = table["delta"].to_numpy()
        ok = ranked & np.isfinite(delta)
        if ok.any():
            relative = np.exp(-0.5 * delta[ok])
            weight[ok] = relative / relative.sum()
    table.insert(3, "weight", weight)
    return table


def _warn_if_lifetimes_passed_over(
    outside: list[str],
    candidates: list,
    model: Any,
    labels: dict[int, str],
    data: SurpyvalData,
) -> None:
    """Warn when the lifetime families -- those whose support starts at
    0 -- were passed over because some times are at or below 0 (#646).

    Maintenance records hold zero ages (failed on the day of
    installation); every positive family is then outside its support, and
    the best of the rest was returned in silence: a Normal that put 3% of
    the units failing before day 0. A Beta passed over for data outside
    (0, 1) stays quiet (#485)."""
    lifetimes = [
        dist.name
        for dist in candidates
        if dist.name in outside
        and float(dist.support[0]) == 0.0
        and np.isinf(float(dist.support[1]))
    ]
    if not lifetimes:
        return
    x = np.asarray(data.x, dtype=float)
    lowest = x[:, -1] if x.ndim == 2 else x
    at_or_below = int(np.sum(np.asarray(data.n)[lowest <= 0]))
    chosen = "nothing" if model is None else labels[id(model)]
    warnings.warn(
        f"fit_best passed over {', '.join(lifetimes)}: {at_or_below} "
        "observation(s) at or below 0 are outside their support. Chosen: "
        f"{chosen}, which may put failures before time 0. For units that "
        "failed at time 0 (dead on arrival), fit a lifetime family with "
        "zi=True (e.g. Weibull.fit(x, zi=True)); otherwise correct those "
        "times.",
        stacklevel=2,
    )


def _raise_if_about_the_data(
    errors: list[Exception], outside: list[str]
) -> None:
    """With no candidate fitted: raise what is about the data rather than
    about a family. Every candidate failing with the same error is the
    data's error (raised as it is); no candidate whose support holds the
    data is the data's too."""
    if errors and len({(type(e), str(e)) for e in errors}) == 1:
        raise errors[0]
    if outside and not errors:
        raise OutsideSupportError(
            "The data lie outside the support of every candidate tried "
            f"({', '.join(outside)}); fit a distribution whose support "
            "holds them (Normal, Gumbel or Logistic for negative values)."
        )
