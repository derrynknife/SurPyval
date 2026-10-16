"""The keyword check of the data classes' constructors (#931): an unknown
argument is refused with the argument that was meant -- the ``xcnt``
spelling (``tl``, ``Z``, ``e``, ...) or another common one, from an alias
table, else the closest accepted name.
"""

from __future__ import annotations

import difflib
from typing import Any, Iterable

__all__ = ["refuse_unknown_keywords"]

# A record's outcome, as the lifetime alias table names it.
_OUTCOME = (
    "the outcome is one of observed=, right_censored=, left_censored= or "
    "interval_censored=[lower, upper]"
)

LIFETIME_ALIASES: dict[str, str] = {
    "x": _OUTCOME,
    "c": _OUTCOME,
    "xl": "an interval is interval_censored=[lower, upper]",
    "xr": "an interval is interval_censored=[lower, upper]",
    "interval": "an interval is interval_censored=[lower, upper]",
    "time": "an age at failure is observed=",
    "age": "an age at failure is observed=",
    "failed": "an age at failure is observed=",
    "failure": "an age at failure is observed=",
    "failure_time": "an age at failure is observed=",
    "censored": "an age last seen working is right_censored=",
    "suspended": "an age last seen working is right_censored=",
    "suspension": "an age last seen working is right_censored=",
    "last_seen": "an age last seen working is right_censored=",
    "n": (
        "how many identical items it stands for is the dataset's counts= "
        "(SurpyvalData.from_items(items, counts=...)); an ItemLifetime is "
        "one item"
    ),
    "count": (
        "how many identical items it stands for is the dataset's counts= "
        "(SurpyvalData.from_items(items, counts=...)); an ItemLifetime is "
        "one item"
    ),
    "counts": (
        "how many identical items it stands for is the dataset's counts= "
        "(SurpyvalData.from_items(items, counts=...)); an ItemLifetime is "
        "one item"
    ),
    "t": "truncation is left_truncation= and right_truncation=",
    "tl": "a record's left truncation is left_truncation=",
    "tr": "a record's right truncation is right_truncation=",
    "entry": "a record's left truncation is left_truncation=",
    "start": "a record's left truncation is left_truncation=",
    "left_truncated": "a record's left truncation is left_truncation=",
    "right_truncated": "a record's right truncation is right_truncation=",
    "Z": "a record's covariates are covariates=",
    "covariate": "a record's covariates are covariates=",
    "covariate_names": (
        "covariates are named by giving them as covariates={name: value}"
    ),
    "e": "what failed is event_type=",
    "event_types": "what failed is event_type=",
    "cause": "what failed is event_type=",
    "mode": "what failed is event_type=",
    "failure_mode": "what failed is event_type=",
    "type": "what failed is event_type=",
    "tvc": "a covariate schedule is time_varying_covariates=",
    "time_varying": "a covariate schedule is time_varying_covariates=",
    "schedule": "a covariate schedule is time_varying_covariates=",
    "path": "a covariate schedule is time_varying_covariates=",
}

HISTORY_ALIASES: dict[str, str] = {
    "x": "the ages at failure are observed=[...]",
    "failures": "the ages at failure are observed=[...]",
    "failed": "the ages at failure are observed=[...]",
    "events": "the ages at failure are observed=[...]",
    "times": "the ages at failure are observed=[...]",
    "ages": "the ages at failure are observed=[...]",
    "c": (
        "the kind of each failure is its list: observed=, "
        "interval_censored= or left_censored=; the end of observation is "
        "right_censored="
    ),
    "n": (
        "a count of failures is interval_censored_counts= or "
        "left_censored_counts=; an exact failure is one age in observed= "
        "(repeat it for simultaneous failures)"
    ),
    "counts": (
        "a count of failures is interval_censored_counts= or "
        "left_censored_counts="
    ),
    "end": "the end of observation is right_censored=",
    "ends": "the end of observation is right_censored=",
    "censored": "the end of observation is right_censored=",
    "last_seen": "the end of observation is right_censored=",
    "intervals": "failures between inspections are interval_censored=",
    "xl": "failures between inspections are interval_censored=[[l, u]]",
    "xr": "failures between inspections are interval_censored=[[l, u]]",
    "t": "truncation is left_truncation= and right_truncation=",
    "tl": "a record's left truncation is left_truncation=",
    "tr": "a record's right truncation is right_truncation=",
    "entry": "a record's left truncation is left_truncation=",
    "start": "a record's left truncation is left_truncation=",
    "Z": "a record's covariates are covariates=",
    "covariate": "a record's covariates are covariates=",
    "covariate_names": (
        "covariates are named by giving them as covariates={name: value}"
    ),
    "e": (
        "the type of each failure is observed_event_types= (and "
        "interval_censored_event_types=, left_censored_event_types=)"
    ),
    "event_type": (
        "the type of each failure is observed_event_types= (and "
        "interval_censored_event_types=, left_censored_event_types=)"
    ),
    "event_types": (
        "the type of each failure is observed_event_types= (and "
        "interval_censored_event_types=, left_censored_event_types=)"
    ),
    "types": (
        "the type of each failure is observed_event_types= (and "
        "interval_censored_event_types=, left_censored_event_types=)"
    ),
    "cause": "the type of each failure is observed_event_types=",
    "causes": "the type of each failure is observed_event_types=",
    "mode": "the type of each failure is observed_event_types=",
    "modes": "the type of each failure is observed_event_types=",
    "tvc": "a covariate schedule is time_varying_covariates=",
    "time_varying": "a covariate schedule is time_varying_covariates=",
    "schedule": "a covariate schedule is time_varying_covariates=",
    "path": "a covariate schedule is time_varying_covariates=",
    "state": (
        "a path through states is initial_state= with observed_states=, "
        "interval_censored_states= or left_censored_states="
    ),
    "states": (
        "a path through states is initial_state= with observed_states=, "
        "interval_censored_states= or left_censored_states="
    ),
}

FROM_ITEMS_ALIASES: dict[str, str] = {
    "n": "how many identical items each stands for is counts=",
    "count": "how many identical items each stands for is counts=",
    "weights": "how many identical items each stands for is counts=",
    "ids": "each item's id is its record's",
    "x": "each item's ages are on its record",
    "c": "each item's outcome is on its record",
    "tl": "each item's truncation is on its record (left_truncation=)",
    "tr": "each item's truncation is on its record (right_truncation=)",
    "Z": "each item's covariates are on its record (covariates=)",
    "covariates": "each item's covariates are on its record (covariates=)",
    "e": "each failure's type is on its record",
    "event_types": "each failure's type is on its record",
}

FROM_OBSERVATIONS_ALIASES: dict[str, str] = {
    "x": "the times go in the list of their kind: observed=, "
    "right_censored=, left_censored= or interval_censored=",
    "c": "the kind of each time is its list: observed=, right_censored=, "
    "left_censored= or interval_censored=",
    "times": "the times go in the list of their kind: observed=, "
    "right_censored=, left_censored= or interval_censored=",
    "failures": "the failure times are observed=",
    "failed": "the failure times are observed=",
    "suspensions": "the suspensions are right_censored=",
    "suspended": "the suspensions are right_censored=",
    "censored": "the suspensions are right_censored=",
    "xl": "the intervals are interval_censored=[[lower, upper], ...]",
    "xr": "the intervals are interval_censored=[[lower, upper], ...]",
    "n": "counts are per kind: observed_counts=, right_censored_counts=, "
    "...",
    "counts": "counts are per kind: observed_counts=, "
    "right_censored_counts=, ...",
    "t": "truncation is left_truncation= and right_truncation= (shared), "
    "or per kind, observed_left_truncation= and so on",
    "tl": "left truncation is left_truncation= (shared), or per kind, "
    "observed_left_truncation= and so on",
    "tr": "right truncation is right_truncation= (shared), or per kind, "
    "observed_right_truncation= and so on",
    "Z": "covariates are per kind: observed_covariates=, "
    "right_censored_covariates=, ...",
    "covariates": "covariates are per kind: observed_covariates=, "
    "right_censored_covariates=, ...",
    "e": "event types are per kind: observed_event_types=, "
    "interval_censored_event_types=, left_censored_event_types=",
    "event_types": "event types are per kind: observed_event_types=, "
    "interval_censored_event_types=, left_censored_event_types=",
    "ids": "ids are per kind: observed_ids=, right_censored_ids=, ...",
    "i": "ids are per kind: observed_ids=, right_censored_ids=, ...",
}

FROM_HISTORIES_ALIASES: dict[str, str] = {
    "x": "the failures are observed={item: [ages]}",
    "failures": "the failures are observed={item: [ages]}",
    "events": "the failures are observed={item: [ages]}",
    "i": "the dicts are keyed by each item's id",
    "ids": "the dicts are keyed by each item's id",
    "items": "the dicts are keyed by each item's id",
    "c": "the end of observation is right_censored={item: age}",
    "end": "the end of observation is right_censored={item: age}",
    "ends": "the end of observation is right_censored={item: age}",
    "censored": "the end of observation is right_censored={item: age}",
    "tl": "left truncation is left_truncation={item: age}",
    "tr": "right truncation is right_truncation={item: age}",
    "t": "truncation is left_truncation= and right_truncation=",
    "Z": "covariates are covariates={item: row}",
    "e": (
        "event types have no place in a history: build the data from "
        "ItemHistory records (observed_event_types=) with from_items"
    ),
    "event_types": (
        "event types have no place in a history: build the data from "
        "ItemHistory records (observed_event_types=) with from_items"
    ),
}


def refuse_unknown_keywords(
    where: str,
    unknown: dict,
    accepted: Iterable[str],
    aliases: dict[str, str],
) -> None:
    """
    Refuse the ``unknown`` keyword arguments ``where`` was given, as
    Python does (a ``TypeError`` for an unexpected keyword argument),
    naming what was meant: the alias table's entry for the name, else the
    closest of the ``accepted`` names.

    Examples
    --------
    >>> from surpyval.utils.record_keywords import refuse_unknown_keywords
    >>> refuse_unknown_keywords(
    ...     "ItemLifetime", {"tl": 2}, ["left_truncation"], {}
    ... )  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
    ...
    TypeError: ItemLifetime() got an unexpected keyword argument 'tl';
    did you mean 'left_truncation'?
    """
    if not unknown:
        return
    name = next(iter(unknown))
    accepted = list(accepted)
    head = f"{where}() got an unexpected keyword argument {name!r}"
    if name in aliases:
        raise TypeError(f"{head}: {aliases[name]}.")
    close = difflib.get_close_matches(name, accepted, n=1, cutoff=0.6)
    if close:
        raise TypeError(f"{head}; did you mean {close[0]!r}?")
    raise TypeError(
        f"{head}; it takes " + ", ".join(repr(a) for a in accepted) + "."
    )


def accepted_names(init: Any, *skip: str) -> list[str]:
    """The keyword names a function accepts, read from its signature,
    without ``skip`` (``self``, ``cls``, the ``**unknown``)."""
    import inspect

    out = []
    for name, p in inspect.signature(init).parameters.items():
        if name in skip or p.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        out.append(name)
    return out
