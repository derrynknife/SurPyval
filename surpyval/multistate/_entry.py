"""The entry points of the multi-state fits: a :class:`SurpyvalData` as
the data, and the other names of their arguments.

``fit(data)`` takes the data's own arrays (``x``, ``e``, ``s`` as
``state``, ``tl``, ``i``, ``n`` and, for the regressions, ``Z`` named by
``covariate_names``) through :func:`accepts_surpyval_data`, as the other
families do; :func:`multistate_data` checks first what only a multi-state
fit can say: data without ``s`` starts every row from the one initial
state (competing risks), and what these models cannot take is refused
with the reason and the fix.
"""

from __future__ import annotations

import copy
import functools
import inspect
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np

from surpyval.utils.surpyval_data import SurpyvalData, units_with_several_rows
from surpyval.utils.warnings import named

F = TypeVar("F", bound=Callable[..., Any])

#: The initial state of data without ``s`` when no ``transitions`` are
#: given: the name ``TransitionMatrix.competing_risks`` gives it.
DEFAULT_INITIAL_STATE = "event-free"


def other_name(alias: str, name: str) -> Callable[[F], F]:
    """Let a method take its argument ``name`` as ``alias`` too: one
    argument under two names, neither deprecated (``s`` for ``state``,
    ``s_col`` for ``state_col``). Giving both is a ``TypeError``. The
    signature shows ``alias`` as a keyword."""

    def decorate(method: F) -> F:
        signature = inspect.signature(method)

        @functools.wraps(method)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if alias in kwargs:
                value = kwargs.pop(alias)
                if value is not None:
                    bound = signature.bind_partial(*args, **kwargs)
                    if bound.arguments.get(name) is not None:
                        raise TypeError(
                            f"Both {name}= and {alias}= were given: they "
                            "are the same argument (the state of each "
                            "row), so give one."
                        )
                    kwargs[name] = value
            return method(*args, **kwargs)

        params = list(signature.parameters.values())
        extra = inspect.Parameter(
            alias, inspect.Parameter.KEYWORD_ONLY, default=None
        )
        last = [p for p in params if p.kind is p.VAR_KEYWORD]
        rest = [p for p in params if p.kind is not p.VAR_KEYWORD]
        setattr(
            wrapper,
            "__signature__",
            signature.replace(parameters=rest + [extra] + last),
        )
        return wrapper  # type: ignore[return-value]

    return decorate


def _rows(mask: np.ndarray) -> str:
    """The first few row numbers of ``mask``, as text."""
    rows = np.flatnonzero(mask)
    shown = ", ".join(str(int(r)) for r in rows[:5])
    return shown + (", ..." if rows.size > 5 else "")


def initial_state(transitions: Any) -> Any:
    """The state every row of data without ``s`` starts from: the one
    state of ``transitions`` with no transition into it, or
    :data:`DEFAULT_INITIAL_STATE` where no transitions are given."""
    if transitions is None:
        return DEFAULT_INITIAL_STATE
    from surpyval.multistate.transitions import TransitionMatrix

    tm = (
        transitions
        if isinstance(transitions, TransitionMatrix)
        else TransitionMatrix(transitions)
    )
    entered = {b for _, b in tm.transitions}
    starts = [s for s in tm.states if s not in entered]
    if len(starts) != 1:
        raise ValueError(
            "The SurpyvalData has no states (s), so every row would start "
            "from the one initial state, but the transitions have "
            + (
                f"{len(starts)} states with no transition into them "
                f"({starts!r})"
                if starts
                else "no state without a transition into it"
            )
            + ": give each row's state in the data, "
            "SurpyvalData(x, ..., s=s, e=e, i=i)."
        )
    return starts[0]


def _checked(data: SurpyvalData, regression: bool, transitions: Any) -> Any:
    """``data`` as the multi-state fits take it, or a refusal saying why
    not and what to do: a shallow copy with ``s`` filled where it has
    none, and its covariates as a named table."""
    n_rows = len(data.c)
    if data.e is None:
        raise ValueError(
            "The SurpyvalData has no e: a multi-state model needs the "
            "state each row moved into (e, blank where the unit was still "
            "in its state). Build it with SurpyvalData(x, ..., s=s, e=e); "
            "for survival data with no states, KaplanMeier or a "
            "distribution's fit takes it as it is."
        )
    odd = ~np.isin(data.c, (0, 1))
    if odd.any():
        raise ValueError(
            f"Rows {_rows(odd)} are left or interval censored (c = -1 or "
            "2): a change of state seen only between two times is not "
            "supported by this model. Give each move at the time it was "
            "seen, or leave those rows out "
            "(data[(data.c == 0) | (data.c == 1)])."
        )
    out = data
    if np.ndim(data.x) == 2:
        # Ranges read row by row (x, xl and xr together) keep every row's
        # time as [lower, upper]: with only exact and right-censored rows
        # left, each row's time is its lower end.
        out = copy.copy(data)
        out.x = np.asarray(data.x)[:, 0]
    if bool(np.isfinite(data.tr).any()):
        raise ValueError(
            "The SurpyvalData has right truncation (tr), which this model "
            "does not take (it takes each row's entry, tl). Leave it out "
            "of the data."
        )
    if data.time_varying_covariates is not None:
        raise ValueError(
            "The SurpyvalData has time-varying covariates, which the "
            "multi-state models do not take: split each stay at the "
            "changes into rows with the covariates fixed, or use "
            "data.without('time_varying_covariates')."
        )
    if data.Z is not None and not regression:
        raise ValueError(
            "This model takes no covariates, which the SurpyvalData "
            "holds: use data.without('covariates'), or "
            "MultiStateProportionalHazards or "
            "ParametricMultiStateRegression, which take them."
        )
    if data.s is None:
        _refuse_several_rows(data)
        masked = (data.c == 0) & np.array([v is None for v in data.e])
        if masked.any():
            raise ValueError(
                f"Rows {_rows(masked)} failed (c = 0) with no state "
                "entered in e: a multi-state model needs the state each "
                "move went into. Give it in e, or use CompetingRisks, "
                "which takes masked causes."
            )
        out = copy.copy(out) if out is data else out
        s = np.empty(n_rows, dtype=object)
        s[:] = [initial_state(transitions)] * n_rows
        out.s = s
    if regression and data.Z is not None and data.covariate_names:
        import pandas as pd

        if out is data:
            out = copy.copy(data)
        out.Z = pd.DataFrame(
            np.asarray(data.Z), columns=list(data.covariate_names)
        )
    return out


def _refuse_several_rows(data: SurpyvalData) -> None:
    """Refuse data without ``s`` whose units have several rows: they are
    not stays from the one initial state, so fitting them as such would
    be another model."""
    units = units_with_several_rows(data)
    if not units:
        return
    which = (
        f"Unit{'s' if len(units) > 1 else ''} {named(units)} "
        f"{'have' if len(units) > 1 else 'has'} several rows"
    )
    repeated = data.a is not None and bool(
        ((np.asarray(data.c) != 1) & (np.asarray(data.a) == 0)).any()
    )
    kind = (
        "repeated events (a = 0 where the unit carried on), which the "
        "recurrent models in surpyval.recurrent take"
        if repeated
        else "stretches of the unit's life with the covariates fixed in "
        "each, which a regression's fit_tvc takes (CoxPH.fit_tvc, for "
        "example)"
    )
    raise ValueError(
        f"{which} and the SurpyvalData has no states (s): without s every "
        "row is a unit starting from the one initial state, and these "
        f"rows read as {kind}. For a multi-state model, give each row's "
        "state in the data, SurpyvalData(x, ..., s=s, e=e, i=i)."
    )


def multistate_data(method: F) -> F:
    """Check a SurpyvalData given to a multi-state ``fit`` (see the
    module docstring), before :func:`accepts_surpyval_data` fills the
    fit's arguments from it. Apply it above that decorator."""
    signature = inspect.signature(method)
    regression = "Z" in signature.parameters

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        data = bound.arguments.get("x")
        if not isinstance(data, SurpyvalData):
            return method(*args, **kwargs)
        if bound.arguments.get("state") is not None:
            raise TypeError(
                "x is a SurpyvalData, which holds the data: leave out "
                "state= (or s=; give the states in the data, "
                "SurpyvalData(x, ..., s=s, e=e, i=i))."
            )
        bound.arguments["x"] = _checked(
            data, regression, bound.arguments.get("transitions")
        )
        return method(*bound.args, **bound.kwargs)

    return wrapper  # type: ignore[return-value]
