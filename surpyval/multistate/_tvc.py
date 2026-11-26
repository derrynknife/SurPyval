r"""Covariates that change during a stay, for the parametric multi-state
regressions.

Each unit's covariate path (a ``StepSchedule``, or a ``CovariatePath``
cut into steps) is given once; every spell of the unit is split at the
path's changes inside it into rows on which the covariates are fixed.
Each row but the last of a spell is censored at its end (the unit was
still in the state), and each row but the first enters at the change
that starts it.

For a regression whose hazard at time :math:`t` depends only on
:math:`t` and the covariates at :math:`t` (proportional hazards,
additive hazards, proportional odds), the cumulative hazard is additive
over the rows, so a spell's factor of the likelihood,

.. math::
    \exp\Bigl\{-\sum_k \bigl[H(b_k \mid z_k) - H(a_k \mid z_k)\bigr]\Bigr\}
    \, h(x \mid z_K)^{d},

is exactly the product of its rows' left-truncated factors. On the
forward clock :math:`a_k, b_k` are times since the start; on the reset
clock, times since the unit entered the state, so a row that carries on
its unit's stay continues the stay's clock rather than restarting it
(see ``parametric._stays``). An accelerated-failure-time regression
rescales the time already spent, so its cumulative hazard is not
additive over the rows; it is refused.

Predictions along a path, on the forward clock, take each transition's
hazard at :math:`t` with the covariates at :math:`t`, and its cumulative
hazard as the model's ``Hf_tvc`` along the path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.multistate._data import SpellData

#: The kinds of regression whose hazard at a time depends only on the
#: covariates then (their cumulative hazard is additive over intervals).
ADDITIVE_KINDS = (
    "Proportional Hazard",
    "Additive Hazard",
    "Proportional Odds",
)
#: The share of a step, at its end, given the path's value at the end of
#: a spell that ends in a move (as the univariate fits on paths do).
_PATH_END = 1e-6


def is_path(Z: Any) -> bool:
    """Whether ``Z`` is a covariate path (a ``StepSchedule`` or a
    ``CovariatePath``) rather than rows of covariates."""
    from surpyval.univariate.regression.tvc_path import CovariatePath
    from surpyval.univariate.regression.tvc_schedule import StepSchedule

    return isinstance(Z, (StepSchedule, CovariatePath))


def check_families(fitters: list, pairs: list, what: str) -> None:
    """Refuse a regression whose cumulative hazard is not additive over
    intervals of fixed covariates (accelerated failure time and the
    others), saying what to use."""
    from surpyval.univariate.regression.tvc_fit import TVCFitMixin

    for fitter, pair in zip(fitters, pairs):
        if not isinstance(fitter, TVCFitMixin):
            name = getattr(fitter, "name", type(fitter).__name__)
            raise ValueError(
                f"{what} takes a proportional hazards, additive hazards or "
                f"proportional odds regression for each transition, whose "
                "hazard at a time depends only on the covariates then; "
                f"{name} for {pair!r} is not one. In an accelerated "
                "failure time model a change of covariates rescales the "
                "time already spent, which splitting a stay at the "
                "changes cannot give. Use WeibullPH (or another PH, AH or "
                "PO regression) for that transition, or fit the "
                "transition alone with the univariate fit_tvc."
            )


def unit_paths(
    paths: Any, i: Any, data: SpellData, width: int | None = None
) -> list:
    """The path of each unit (by its code in ``data.cluster``), checked:
    one for every unit and none for another, every one a path with the
    same number of covariates."""
    import pandas as pd

    from surpyval.utils.data_formats import missing_events

    if not isinstance(paths, dict):
        raise ValueError(
            "time_varying_covariates must be a dict {unit: StepSchedule or "
            f"CovariatePath}}, one for each unit in 'i'; got {type(paths)}."
        )
    if not data.ids_given or i is None:
        raise ValueError(
            "time_varying_covariates are given by unit, so 'i' (the unit "
            "of each spell) is needed."
        )
    ids = np.empty(len(i), dtype=object)
    ids[:] = [v.item() if isinstance(v, np.generic) else v for v in i]
    if np.any(missing_events(ids)):
        raise ValueError("'i' has missing values: label every row.")
    _, labels = pd.factorize(pd.Series(ids, dtype=object))
    labels = list(labels)
    missing = [u for u in labels if u not in paths]
    if missing:
        raise ValueError(
            f"time_varying_covariates has no path for {len(missing)} "
            f"unit(s) (the first is {missing[0]!r}): give one for every "
            "unit, StepSchedule.constant(z) for a unit whose covariates "
            "do not change."
        )
    extra = [u for u in paths if u not in set(labels)]
    if extra:
        raise ValueError(
            f"time_varying_covariates has paths for {len(extra)} unit(s) "
            f"that 'i' does not have (the first is {extra[0]!r})."
        )
    out = []
    for u in labels:
        path = paths[u]
        if not is_path(path):
            raise ValueError(
                f"The time-varying covariates of unit {u!r} must be a "
                f"StepSchedule or a CovariatePath; got {type(path)}."
            )
        if width is None:
            width = path.p
        if path.p != width:
            raise ValueError(
                f"Unit {u!r}'s time-varying covariates have {path.p} "
                f"column(s) where another unit's have {width}: every "
                "unit's path gives the same covariates."
            )
        out.append((u, path))
    return out


def split(
    data: SpellData,
    Z: npt.NDArray,
    paths: list,
    low: float,
    steps: int | None = None,
) -> tuple[SpellData, npt.NDArray]:
    """The spells split at their units' covariate changes, and each
    row's covariates: the fixed ones ``Z`` (one row per spell), then the
    path's. ``paths[u]`` is ``(label, path)`` of the unit coded ``u``; a
    ``CovariatePath`` is cut into ``steps`` steps a stretch. ``low`` is
    where a spell with no entry time (``tl = -inf``) starts: the start of
    the clock."""
    from dataclasses import replace

    from surpyval.univariate.regression.tvc_path import CovariatePath

    start = np.where(np.isfinite(data.tl), data.tl, low)
    rows: dict = {k: [] for k in ("x", "tl", "g", "h", "n", "c", "Z")}
    schedules: dict = {}
    for u, (label, path) in enumerate(paths):
        mine = data.cluster == u
        first = float(np.min(start[mine]))
        if path.start > first:
            raise ValueError(
                f"Unit {label!r}'s time-varying covariates start at "
                f"{path.start:g}, after its first spell starts "
                f"({first:g}): nothing says what they were before. Give "
                "the path from the start of the unit's first spell (from "
                "-inf on a baseline defined below 0)."
            )
        if isinstance(path, CovariatePath):
            last = float(np.max(data.x[mine]))
            path_steps = path.to_steps(
                last if last > first else first + 1.0,
                int(steps or 8),
                start=first,
            )
            schedules[u] = (path_steps, path)
        else:
            schedules[u] = (path, None)
    for r in range(data.x.size):
        schedule, path = schedules[int(data.cluster[r])]
        a, b = float(start[r]), float(data.x[r])
        horizon = b if b > schedule.start else schedule.start + 1.0
        s0, s1, values = schedule.segments(horizon)
        keep = (s1 > a) & (s0 < b)
        s0 = np.maximum(s0[keep], a)
        s1 = np.minimum(s1[keep], b)
        values = values[keep]
        if path is not None and data.h[r] >= 0 and s0.size:
            # The hazard of the move is the path's at the move, not at the
            # middle of the step it fell in.
            at_end = path(np.array([b]))[0]
            if not np.array_equal(values[-1], at_end):
                cut = s1[-1] - _PATH_END * (s1[-1] - s0[-1])
                if s0[-1] < cut < s1[-1]:
                    s0 = np.append(s0, cut)
                    s1 = np.append(s1[:-1], [cut, s1[-1]])
                    values = np.vstack([values, at_end])
        last = s0.size - 1
        for k in range(s0.size):
            rows["x"].append(s1[k])
            rows["tl"].append(data.tl[r] if k == 0 else s0[k])
            rows["g"].append(data.g[r])
            rows["h"].append(data.h[r] if k == last else -1)
            rows["n"].append(data.n[r])
            rows["c"].append(data.cluster[r])
            rows["Z"].append(np.r_[Z[r], values[k]])
    out = replace(
        data,
        x=np.asarray(rows["x"], dtype=float),
        tl=np.asarray(rows["tl"], dtype=float),
        g=np.asarray(rows["g"], dtype=np.intp),
        h=np.asarray(rows["h"], dtype=np.intp),
        n=np.asarray(rows["n"], dtype=float),
        cluster=np.asarray(rows["c"], dtype=np.intp),
        xl=None,
    )
    return out, np.asarray(rows["Z"], dtype=float).reshape(out.x.size, -1)


def path_cumhaz(models: list, path: Any, origin: float) -> Any:
    """``H(t, rate)`` along ``path`` for :func:`_forward.plan`: each
    transition's cumulative hazard from ``origin`` (its hazard with
    ``rate``), the covariates at each time those of the path."""
    from surpyval.univariate.regression.tvc_path import CovariatePath

    if isinstance(path, CovariatePath):

        def values(t: npt.NDArray) -> npt.NDArray:
            return np.asarray(path(t), dtype=float)

    else:
        edges = np.asarray(path.edges, dtype=float)

        def values(t: npt.NDArray) -> npt.NDArray:
            horizon = float(np.max(t)) if t.size else origin + 1.0
            horizon = max(horizon, origin, float(edges[0])) + 1.0
            s0, _, Zs = path.segments(horizon)
            k = np.clip(np.searchsorted(s0, t, side="right") - 1, 0, None)
            return Zs[k]

    def H(t: npt.NDArray, rate: bool) -> npt.NDArray:
        t = np.asarray(t, dtype=float).ravel()
        out = np.empty((t.size, len(models)))
        with np.errstate(all="ignore"):
            for h, m in enumerate(models):
                if rate:
                    out[:, h] = np.ravel(m.hf(t, values(t)))
                else:
                    out[:, h] = np.ravel(m.Hf_tvc(t, path, start=origin))
        return out

    return H


def path_breaks(path: Any, origin: float, horizon: float) -> list:
    """The times in ``(origin, horizon)`` where the path jumps or has a
    kink, for the grid of the forward equations."""
    from surpyval.univariate.regression.tvc_path import CovariatePath

    if isinstance(path, CovariatePath):
        knots = np.asarray(getattr(path, "_knots", []), dtype=float)
        if path.period is not None:
            reps = np.arange(
                np.floor(origin / path.period),
                np.ceil(horizon / path.period) + 1,
            )
            knots = (reps[:, None] * path.period + knots[None, :]).ravel()
    else:
        s0, _, _ = path.segments(max(horizon, path.start + 1.0))
        knots = s0
    knots = np.unique(knots[np.isfinite(knots)])
    return [float(v) for v in knots if origin < v < horizon]
