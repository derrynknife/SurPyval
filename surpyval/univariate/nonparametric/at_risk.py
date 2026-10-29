"""The numbers at risk, censored and with events of one or more fitted
non-parametric models at chosen times: :func:`at_risk_table`, the table
:func:`~surpyval.plot_nonparametric` draws under its plot, as data.

It imports no plotting library.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.validation import check_option

from .nonparametric import NonParametric

__all__ = ["at_risk_table"]

_COUNTS = ("cumulative", "interval")
_ROWS = ("At risk", "Censored", "Events", "In interval")


def at_risk_table(
    models: Any,
    times: npt.ArrayLike | None = None,
    rows: str | Sequence[str] = ("At risk",),
    counts: str = "cumulative",
    *,
    labels: Sequence[str] | None = None,
) -> dict[str, Any]:
    """
    The number-at-risk table of one or more fitted non-parametric models,
    as data: the counts :func:`~surpyval.plot_nonparametric` draws under
    its plot, without drawing anything (no plotting library is loaded).

    Parameters
    ----------
    models : NonParametric, list or dict
        One fitted model (Kaplan-Meier, Nelson-Aalen, Fleming-Harrington
        or Turnbull), a list of them, or ``{label: model}``.
    times : array like, optional
        The times to count at, finite. Defaults to the start of the time
        axis (0, or the smallest value if it is negative) and every
        distinct time in the models' estimates.
    rows : str or list of str, optional
        The rows for each model, from ``'At risk'``, ``'Censored'``,
        ``'Events'`` and ``'In interval'``, or ``'all'`` for the first
        three (and ``'In interval'`` when any model has left or interval
        censored data). Defaults to ``('At risk',)``.
    counts : ('cumulative', 'interval'), optional
        Whether the 'Censored' and 'Events' rows count everything up to
        each time (the default) or only what happened since the previous
        time.
    labels : list of str, optional
        One label per model, as :func:`~surpyval.plot_nonparametric`
        labels them: the keys of a dict, otherwise the estimator's name
        (or "Group 1", "Group 2", ... when names repeat).

    Returns
    -------
    dict
        ``{"times": numpy array, "rows": [row names], "counts": {label:
        {row: numpy array}}}``, one count per time. The counts are the
        same as those of ``plot_nonparametric`` (see its Notes for the
        definitions, including those for left and interval censored
        data). For an estimate with weights ``w`` they count the units
        themselves, not their weights.

    Examples
    --------
    >>> import surpyval
    >>> from surpyval import KaplanMeier
    >>> a = KaplanMeier.fit([2, 3, 5, 7, 8, 11], c=[0, 0, 1, 0, 0, 1])
    >>> b = KaplanMeier.fit([4, 6, 9, 10, 12, 14], c=[0, 1, 0, 0, 1, 0])
    >>> table = surpyval.at_risk_table(
    ...     {"A": a, "B": b}, times=[0, 5, 10], rows="all"
    ... )
    >>> table["times"], table["rows"]
    (array([ 0.,  5., 10.]), ['At risk', 'Censored', 'Events'])
    >>> table["counts"]["A"]["At risk"], table["counts"]["A"]["Events"]
    (array([6., 4., 1.]), array([0., 2., 4.]))
    """
    check_option("counts", counts, _COUNTS)
    pairs = _as_models(models, labels)
    chosen = _check_rows(rows, pairs, True, name="rows")
    if times is None:
        grid = _default_times(pairs)
    else:
        grid = np.atleast_1d(np.asarray(times, dtype=float))
        if grid.ndim != 1 or not np.isfinite(grid).all():
            raise ValueError(
                "'times' must be a one-dimensional array of finite times."
            )
        grid = np.sort(grid)
    return {
        "times": grid,
        "rows": chosen,
        "counts": {
            label: _counts(model, label, grid, chosen, counts)
            for label, model in pairs
        },
    }


def _default_times(pairs: list[tuple[str, NonParametric]]) -> npt.NDArray:
    """The start of the time axis and every distinct finite time in the
    models' estimates."""
    xs = [np.asarray(m.x, dtype=float).ravel() for _, m in pairs]
    every = np.concatenate(xs) if xs else np.array([])
    every = every[np.isfinite(every)]
    start = min(0.0, float(every.min())) if every.size else 0.0
    return np.unique(np.append(start, every))


def _as_models(models: Any, labels: Any) -> list[tuple[str, NonParametric]]:
    """The models as ``(label, model)`` pairs, from one model, a sequence
    of models or a mapping ``{label: model}``."""
    if isinstance(models, Mapping):
        pairs = [(str(k), v) for k, v in models.items()]
        given = True
    elif isinstance(models, NonParametric):
        pairs = [(str(models.model), models)]
        given = False
    elif isinstance(models, (list, tuple)):
        pairs = [("", m) for m in models]
        given = False
    else:
        pairs = [("", models)]
        given = False
    if not pairs:
        raise ValueError(
            "'models' is empty; pass a fitted non-parametric model (e.g. "
            "surpyval.KaplanMeier.fit(x, c)), a list of them or a dict "
            "{label: model}."
        )
    for _, m in pairs:
        if not isinstance(m, NonParametric):
            hint = (
                " It is a fitter: fit it first, e.g. KaplanMeier.fit(x, c)."
                if hasattr(m, "fit") and not hasattr(m, "sf")
                else (
                    " A parametric or regression model draws itself with "
                    "its own plot() method."
                    if hasattr(m, "sf")
                    else ""
                )
            )
            raise TypeError(
                "plot_nonparametric draws fitted non-parametric models "
                "(KaplanMeier, NelsonAalen, FlemingHarrington or Turnbull "
                "fits); got {}.{}".format(type(m).__name__, hint)
            )
    if labels is not None:
        labels = [str(v) for v in labels]
        if len(labels) != len(pairs):
            raise ValueError(
                "'labels' has {} entries for {} models; give one label "
                "per model.".format(len(labels), len(pairs))
            )
        if len(set(labels)) != len(labels):
            raise ValueError(
                "'labels' repeats a label ({}); give each model its own, "
                "as the legend, the table and the result are keyed by "
                "them.".format(", ".join(labels))
            )
        return list(zip(labels, [m for _, m in pairs]))
    if not given and len(pairs) > 1:
        names = [m.model for _, m in pairs]
        if len(set(names)) == len(names):
            return list(zip(names, [m for _, m in pairs]))
        return [(f"Group {i + 1}", m) for i, (_, m) in enumerate(pairs)]
    return pairs


def _data_bounds(model: NonParametric) -> dict[str, npt.NDArray] | None:
    """Each observation's entry time, the bounds ``(lower, upper]`` of its
    event time, its censoring flag and its count, from the data the model
    was fitted to; None for a model that does not hold its data."""
    data = getattr(model, "data", None)
    if not isinstance(data, dict) or "x" not in data or "c" not in data:
        return None
    x = np.asarray(data["x"], dtype=float)
    c = np.asarray(data["c"]).astype(int)
    n = np.asarray(data.get("n", np.ones(len(c))), dtype=float)
    t = data.get("t")
    if t is None:
        tl = np.full(len(c), -np.inf)
    else:
        tl = np.asarray(t, dtype=float)[:, 0]
    if x.ndim == 2:
        left, right = x[:, 0], x[:, 1]
    else:
        left, right = x, x
    lower = np.where(c == -1, -np.inf, left)
    upper = np.where(c == 1, np.inf, np.where(c == 0, left, right))
    return {"tl": tl, "lower": lower, "upper": upper, "c": c, "n": n}


def _counts_from_data(
    b: dict[str, npt.NDArray], times: npt.NDArray, rows: Sequence[str]
) -> dict[str, npt.NDArray]:
    """The table's rows at ``times``, cumulative to each time, counted
    from the data; see ``plot_nonparametric`` for the definitions."""
    T = times[:, None]
    entered = b["tl"][None, :] < T
    n = b["n"][None, :]
    c = b["c"][None, :]
    out = {}
    for row in rows:
        if row == "At risk":
            mask = entered & (T <= b["lower"][None, :])
        elif row == "Censored":
            mask = (c == 1) & (b["lower"][None, :] <= T)
        elif row == "Events":
            mask = (c != 1) & (b["upper"][None, :] <= T)
        else:  # "In interval"
            mask = (
                entered
                & (b["lower"][None, :] < T)
                & (T < b["upper"][None, :])
                & ((c == -1) | (c == 2))
            )
        out[row] = (mask * n).sum(axis=1)
    return out


def _counts_from_ladder(
    model: NonParametric, times: npt.NDArray, rows: Sequence[str]
) -> dict[str, npt.NDArray]:
    """The at-risk and event rows of a model that holds only its ladder
    (``from_xrd``, or restored without its data): the number at risk at
    ``t`` is the risk set of the first time at or after ``t`` (none past
    the last), and the events are the deaths up to ``t``."""
    x = np.asarray(model.x, dtype=float)
    r = np.asarray(model.r, dtype=float)
    d = np.asarray(model.d, dtype=float)
    out = {}
    for row in rows:
        if row == "At risk":
            idx = np.searchsorted(x, times, side="left")
            r_pad = np.append(r, 0.0)
            out[row] = r_pad[idx]
        elif row == "Events":
            idx = np.searchsorted(x, times, side="right")
            out[row] = np.append(0.0, np.cumsum(d))[idx]
        else:
            raise ValueError(
                "The {!r} row is counted from the data the model was "
                "fitted with, which this model does not hold (a model "
                "from 'from_xrd', or one restored from a dictionary "
                "written without to_dict(with_data=True)). Use "
                "at_risk_rows=('At risk', 'Events'), or save the model "
                "with its data.".format(row)
            )
    return out


def _counts(
    model: NonParametric,
    label: str,
    times: npt.NDArray,
    rows: Sequence[str],
    counts: str,
) -> dict[str, npt.NDArray]:
    b = _data_bounds(model)
    if b is not None:
        out = _counts_from_data(b, times, rows)
    elif model.model == "Turnbull" or getattr(model, "r", None) is None:
        raise ValueError(
            "The numbers at risk of {!r} are counted from the data it was "
            "fitted with, which this model does not hold (a model from "
            "'fit_from_ecdf', or a Turnbull model restored without "
            "to_dict(with_data=True)). Pass at_risk=False, or refit the "
            "model to its data.".format(label)
        )
    else:
        out = _counts_from_ladder(model, times, rows)
    if counts == "interval":
        for row in ("Censored", "Events"):
            if row in out:
                out[row] = np.diff(np.append(0.0, out[row]))
    return out


def _has_interval_data(model: NonParametric) -> bool:
    b = _data_bounds(model)
    return b is not None and bool(np.isin(b["c"], (-1, 2)).any())


def _check_rows(
    at_risk_rows: Any,
    pairs: list[tuple[str, NonParametric]],
    at_risk: bool,
    name: str = "at_risk_rows",
) -> list[str]:
    if isinstance(at_risk_rows, str):
        if at_risk_rows == "all":
            rows = ["At risk", "Censored", "Events"]
            if any(_has_interval_data(m) for _, m in pairs):
                rows.append("In interval")
        else:
            rows = [at_risk_rows]
    else:
        rows = list(at_risk_rows)
    for row in rows:
        check_option(name, row, _ROWS + ("all",))
    if not rows and at_risk:
        hint = " or pass at_risk=False" if name == "at_risk_rows" else ""
        raise ValueError(f"'{name}' is empty; name at least one row{hint}.")
    return rows
