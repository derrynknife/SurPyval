"""One plot of one or more non-parametric estimates, with the numbers at
risk under it: :func:`plot_nonparametric`.

matplotlib is imported only when a plot is drawn, so importing this
module does not load it (#470).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval import plotting
from surpyval.utils.units import axis_label
from surpyval.utils.validation import BOUNDS, check_alpha_ci, check_option

from .at_risk import (
    _COUNTS,
    _as_models,
    _check_rows,
    _counts,
    _data_bounds,
)
from .nonparametric import NonParametric

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = ["plot_nonparametric", "NonParametricPlot"]

_ON = ("sf", "ff", "Hf", "hf")
_CI = ("pointwise", "band")
_CI_STYLES = ("fill", "lines")

_YLABELS = {
    "sf": "Survival probability",
    "ff": "Cumulative incidence",
    "Hf": "Cumulative hazard",
    "hf": "Hazard rate (smoothed)",
}

# The table's text: its size and the gap between rows, in points, and
# where its first line sits below the axes (clear of the tick labels and
# the x axis label).
_TABLE_FONT = 9.0
_ROW_HEIGHT = 12.5
_TABLE_TOP = 44.0
# The gap between the row names and the axes' left edge, in points
_LABEL_GAP = 24.0


@dataclass
class NonParametricPlot:
    """What :func:`plot_nonparametric` drew.

    Attributes
    ----------
    fig : matplotlib Figure
        The figure the axes are on.
    ax : matplotlib Axes
        The axes the estimates are drawn on (the table is text on the same
        axes, below the x axis).
    lines : dict
        ``{label: Line2D}``: the curve of each estimate.
    times : numpy array or None
        The times the table counts at; None without a table.
    counts : dict
        ``{label: {row: numpy array}}``: the numbers in the table, one
        value per time in ``times`` (empty without a table).
    quantiles : dict
        ``{label: {p: time}}``: the quantile of each estimate at each
        probability asked for (NaN where the estimate does not reach it).

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import surpyval
    >>> km = surpyval.KaplanMeier.fit([1, 2, 3, 4], c=[0, 1, 0, 0])
    >>> out = surpyval.plot_nonparametric(km, at_risk_times=[0, 2, 4])
    >>> out.times, out.counts["Kaplan-Meier"]["At risk"]
    (array([0., 2., 4.]), array([4., 3., 1.]))
    >>> matplotlib.pyplot.close(out.fig)
    """

    fig: Figure
    ax: Axes
    lines: dict = field(default_factory=dict)
    times: npt.NDArray | None = None
    counts: dict = field(default_factory=dict)
    quantiles: dict = field(default_factory=dict)


def _fmt(v: float) -> str:
    if not np.isfinite(v):
        return "-"
    if float(v).is_integer():
        return str(int(v))
    return f"{v:.1f}"


def _curve(model: NonParametric, on: str, q: npt.NDArray) -> npt.NDArray:
    if on == "sf":
        return model.sf(q)
    if on == "ff":
        return model.ff(q)
    return model.Hf(q)


def _bounds(
    model: NonParametric,
    on: str,
    q: npt.NDArray,
    ci: str,
    alpha_ci: float,
    bound: str,
    bound_type: str | None,
    band_method: str,
) -> npt.NDArray:
    """The bounds on ``on`` at ``q``: ``[lower, upper]`` rows, or one
    column for a one-sided pointwise bound."""
    if ci == "pointwise":
        return model.cb(
            q,
            on,
            alpha_ci,
            bound,
            bound_type=bound_type or "exp",
        )
    sf_band = model.band(
        q,
        method=band_method,
        bound_type=bound_type or "arcsine",
        alpha_ci=alpha_ci,
    )
    if on == "sf":
        return sf_band
    flipped = sf_band[:, ::-1]
    if on == "ff":
        return 1.0 - flipped
    with np.errstate(divide="ignore"):
        return 0.0 - np.log(flipped)


def _quantile_level(on: str, p: float) -> float:
    if on == "sf":
        return 1.0 - p
    if on == "ff":
        return p
    with np.errstate(divide="ignore"):
        return float(-np.log1p(-p))


def plot_nonparametric(
    models: Any,
    on: str = "sf",
    ax: Axes | None = None,
    *,
    labels: Sequence[str] | None = None,
    colors: Any = None,
    ci: str | bool | None = None,
    alpha_ci: float = 0.05,
    bound: str = "two-sided",
    bound_type: str | None = None,
    band_method: str = "hall-wellner",
    ci_style: str = "fill",
    show_censors: bool | None = None,
    quantiles: float | Sequence[float] | None = None,
    xlim: tuple[float, float] | None = None,
    log_x: bool = False,
    hf_bandwidth: float | None = None,
    at_risk: bool = True,
    at_risk_times: npt.ArrayLike | None = None,
    at_risk_rows: str | Sequence[str] = ("At risk",),
    counts: str = "cumulative",
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    legend: bool | None = None,
) -> NonParametricPlot:
    r"""
    Plot one or more non-parametric estimates on one axes, with their
    confidence bounds, censoring ticks, chosen quantiles and a
    number-at-risk table under the x axis.

    Any fitted non-parametric model can be drawn: a Kaplan-Meier,
    Nelson-Aalen, Fleming-Harrington or Turnbull fit, with or without
    truncation. Groups are drawn by passing a list of models or a dict
    ``{label: model}``, for example one fit per level of a
    :meth:`SurpyvalData.groupby <surpyval.SurpyvalData.groupby>` split.

    Parameters
    ----------
    models : NonParametric, list or dict
        One fitted model, a list of them, or ``{label: model}``.
    on : ('sf', 'ff', 'Hf', 'hf'), optional
        What to draw: the survival function (default), the cumulative
        incidence ``ff`` (one minus the survival), the cumulative hazard
        ``Hf``, or the hazard rate ``hf``, which is drawn as the
        kernel-smoothed rate of ``smoothed_hf`` (the step estimate has no
        rate of its own, so it has no bounds, ticks or quantiles).
    ax : matplotlib Axes, optional
        The axes to draw on. Without one a new figure is made, with room
        for the table. On axes of your own, save with
        ``bbox_inches="tight"`` (or use a constrained layout) so that the
        table below the axes is not cut off.
    labels : list of str, optional
        One label per model, in order; the keys of a dict otherwise, and
        the estimator's name (or "Group 1", "Group 2", ... when names
        repeat) for a list.
    colors : list or dict, optional
        One matplotlib colour per model, or ``{label: colour}``. Without
        them the models take SurPyval's palette in order.
    ci : ('pointwise', 'band') or bool, optional
        The confidence bounds to draw: ``'pointwise'`` the model's ``cb``,
        ``'band'`` its simultaneous ``band`` (on ``sf``, ``ff`` and
        ``Hf``: the band on the survival function, carried to the others
        as ``cb`` carries its bounds), and False none. True means
        ``'pointwise'``. Defaults to pointwise bounds where the model has
        them (not on ``hf``, nor for a model from ``fit_from_ecdf``).
    alpha_ci : float, optional
        The significance level of the bounds. Defaults to 0.05.
    bound : ('two-sided', 'lower', 'upper'), optional
        Pointwise bounds only: a one-sided bound is drawn as a dashed
        step. Defaults to 'two-sided'.
    bound_type : str, optional
        The scale of the bounds, as ``cb``'s (``'exp'``, the default, or
        ``'normal'``) or ``band``'s (``'arcsine'``, the default, ``'exp'``
        or ``'normal'``) ``bound_type``.
    band_method : ('hall-wellner', 'nair'), optional
        ``band``'s method. Defaults to 'hall-wellner'.
    ci_style : ('fill', 'lines'), optional
        Two-sided bounds as a shaded wash of the curve's colour (default)
        or as two dashed steps.
    show_censors : bool, optional
        Whether to tick the right censored times on the curve. Defaults to
        ticking them for every model that holds its data (not on ``hf``).
    quantiles : float or list of float, optional
        Probabilities (e.g. 0.5 for the median) at which to draw reference
        lines: a dotted line across at the curve's level and one down to
        the x axis at each model's quantile (``qf(p)``), labelled with its
        value. A model whose estimate never reaches ``p`` gets no line,
        and NaN in the result's ``quantiles``.
    xlim : (float, float), optional
        The time range shown. Defaults to the start (0, or the smallest
        value if negative; the smallest positive value on a log axis) to
        the largest observed value.
    log_x : bool, optional
        Draw time on a log scale. Defaults to False.
    hf_bandwidth : float, optional
        ``smoothed_hf``'s bandwidth, for ``on='hf'``.
    at_risk : bool, optional
        Whether to draw the number-at-risk table. Defaults to True.
    at_risk_times : array like, optional
        The times the table counts at. Defaults to the x axis's major
        ticks within the time range; the axis's ticks are then fixed at
        these times, so the columns stay under their ticks.
    at_risk_rows : str or list of str, optional
        The rows for each model, from ``'At risk'``, ``'Censored'``,
        ``'Events'`` and ``'In interval'``, or ``'all'`` for the first
        three (and ``'In interval'`` when any model has left or interval
        censored data). Defaults to ``('At risk',)``.
    counts : ('cumulative', 'interval'), optional
        Whether the 'Censored' and 'Events' rows count everything up to
        each time (the default, as lifelines' ``add_at_risk_counts``
        does) or only what happened since the previous time.
    xlabel, ylabel, title : str, optional
        The axis labels and title. Defaults: "Time" (or what the models'
        unit measures), the quantity drawn, and "<estimator> estimate"
        when every model has the same estimator.
    legend : bool, optional
        Whether to draw a legend. Defaults to one when there are two or
        more models, or when labels were given.

    Returns
    -------
    NonParametricPlot
        ``fig`` and ``ax`` to customise further, the curves ``lines``,
        the table's ``times`` and ``counts`` and the ``quantiles``.

    Notes
    -----
    The table counts with the same risk sets as the estimators (the
    ``(entry, exit]`` convention of R's ``survfit``): a unit is **at
    risk** at ``t`` if it entered before ``t`` (its left truncation time
    is below ``t``) and its event or censoring time is ``t`` or later, so
    with left truncation the number at risk can rise as units enter.
    **Censored** and **Events** count the right censored units and the
    events at or before ``t``.

    With left or interval censoring (a Turnbull fit) an event time is
    only known to lie in an interval ``(lower, upper]``. The rows then
    count what the data *show*, not the NPMLE's fractional expected
    counts: **At risk** is the units known to be event-free at ``t``
    (``t <= lower``), **Events** the units whose event is known to have
    happened by ``t`` (``upper <= t``), and **In interval** the units
    whose interval spans ``t`` (``lower < t < upper``): their event may
    or may not have happened yet. For right censored and exact data these
    are the usual counts.

    A model that does not hold its data (``from_xrd``, or restored
    without ``with_data=True``) counts 'At risk' and 'Events' from its
    risk sets and deaths; one from ``fit_from_ecdf`` has no counts, so
    pass ``at_risk=False`` for it.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import surpyval
    >>> from surpyval import KaplanMeier
    >>> a = KaplanMeier.fit([2, 3, 5, 7, 8, 11], c=[0, 0, 1, 0, 0, 1])
    >>> b = KaplanMeier.fit([4, 6, 9, 10, 12, 14], c=[0, 1, 0, 0, 1, 0])
    >>> out = surpyval.plot_nonparametric(
    ...     {"A": a, "B": b}, quantiles=0.5, at_risk_times=[0, 5, 10]
    ... )
    >>> out.counts["A"]["At risk"], out.counts["B"]["At risk"]
    (array([6., 4., 1.]), array([6., 5., 3.]))
    >>> out.quantiles["A"][0.5], out.quantiles["B"][0.5]
    (np.float64(7.0), np.float64(10.0))
    >>> plt.close(out.fig)
    """
    check_option("on", on, _ON)
    check_alpha_ci(alpha_ci)
    check_option("bound", bound, BOUNDS)
    check_option("ci_style", ci_style, _CI_STYLES)
    check_option("counts", counts, _COUNTS)
    check_option("band_method", band_method, ("hall-wellner", "nair"))
    pairs = _as_models(models, labels)
    ci = _check_ci(ci, on, bound, quantiles)
    rows = _check_rows(at_risk_rows, pairs, at_risk)
    color_of = _colors(colors, pairs)

    import matplotlib.pyplot as plt

    fig: Any
    if ax is None:
        n_lines = (len(rows) + (1 if len(rows) > 1 else 0)) * len(pairs)
        height = 4.5 + (0.3 + n_lines * _ROW_HEIGHT / 72 if at_risk else 0)
        fig, ax = plt.subplots(figsize=(8, height), layout="constrained")
    else:
        fig = ax.get_figure()

    span = _time_range(pairs, xlim, log_x)
    result = NonParametricPlot(fig=fig, ax=ax)
    style = {
        "on": on,
        "ci": ci,
        "alpha_ci": alpha_ci,
        "bound": bound,
        "bound_type": bound_type,
        "band_method": band_method,
        "ci_style": ci_style,
        "show_censors": show_censors,
        "quantiles": quantiles,
        "log_x": log_x,
        "hf_bandwidth": hf_bandwidth,
    }
    for (label, model), color, last in zip(pairs, color_of, span.ends):
        _draw_model(ax, result, label, model, color, last, span, style)

    _finish_axes(ax, pairs, on, span, log_x, xlabel, ylabel, title)
    if legend is None:
        legend = len(pairs) > 1 or labels is not None
    if legend:
        plotting.style_legend(ax, loc="best")

    if at_risk:
        _draw_table(ax, result, pairs, span, at_risk_times, rows, counts)
    return result


@dataclass
class _Span:
    """The time axis: the curves start at ``start``; ``lo`` and ``hi`` are
    the limits shown; ``ends`` the last observed value of each model."""

    start: float
    lo: float
    hi: float
    ends: list


def _check_ci(ci: Any, on: str, bound: str, quantiles: Any) -> Any:
    if ci is True:
        ci = "pointwise"
    if ci not in (None, False):
        check_option("ci", ci, _CI + (False, None))
        if on == "hf":
            raise ValueError(
                "There are no confidence bounds on the smoothed hazard "
                "rate (on='hf'); pass ci=False, or draw on='Hf' with its "
                "bounds."
            )
        if ci == "band" and bound != "two-sided":
            raise ValueError(
                "A simultaneous band is two-sided; pass bound='two-sided' "
                "with ci='band', or ci='pointwise' for a one-sided bound."
            )
    if quantiles is not None and on == "hf":
        raise ValueError(
            "Quantile lines are drawn on the sf, ff or Hf curves; pass "
            "on='sf' (or 'ff', 'Hf') with quantiles, or drop quantiles."
        )
    return ci


def _colors(colors: Any, pairs: list[tuple[str, NonParametric]]) -> list:
    if isinstance(colors, Mapping):
        return [colors.get(label) for label, _ in pairs]
    if colors is None:
        return [None] * len(pairs)
    if isinstance(colors, str):
        return [colors] * len(pairs)
    color_of = list(colors)
    if len(color_of) != len(pairs):
        raise ValueError(
            "'colors' has {} entries for {} models; give one colour "
            "per model, or a dict {{label: colour}}.".format(
                len(color_of), len(pairs)
            )
        )
    return color_of


def _time_range(
    pairs: list[tuple[str, NonParametric]],
    xlim: Any,
    log_x: bool,
) -> _Span:
    finite = []
    for _, m in pairs:
        xs = np.asarray(m.x, dtype=float)
        b = _data_bounds(m)
        if b is not None:
            xs = np.concatenate([xs, b["lower"], b["upper"]])
        finite.append(xs[np.isfinite(xs)])
    all_x = np.concatenate(finite)
    if log_x:
        positive = all_x[all_x > 0]
        if positive.size == 0:
            raise ValueError(
                "log_x needs positive times; these models have none. Pass "
                "log_x=False."
            )
        start = float(positive.min())
    else:
        start = float(min(0.0, all_x.min()))
    ends = [float(v.max()) for v in finite]
    if xlim is None:
        hi = float(all_x.max())
        return _Span(start, start, hi if hi > start else start + 1.0, ends)
    lo, hi = (float(v) for v in xlim)
    if not lo < hi:
        raise ValueError(
            "'xlim' must be (lower, upper) with lower < upper; got "
            "{}.".format(tuple(xlim))
        )
    if log_x and lo <= 0:
        raise ValueError(
            "A log time axis starts above 0; give xlim a positive lower "
            "limit."
        )
    return _Span(min(start, lo), lo, hi, ends)


def _draw_model(
    ax: Any,
    result: NonParametricPlot,
    label: str,
    model: NonParametric,
    color: Any,
    last: float,
    span: _Span,
    style: dict[str, Any],
) -> None:
    """One model's curve, bounds, censoring ticks and quantile lines."""
    on = style["on"]
    color = plotting.next_color(ax, color)
    line_kwargs: dict[str, Any] = {"label": label}
    if color is not None:
        line_kwargs["color"] = color
    if plotting.theme_enabled():
        line_kwargs["linewidth"] = plotting.LINE_WIDTH

    mx = np.asarray(model.x, dtype=float)
    if on == "hf":
        if style["log_x"]:
            grid = np.geomspace(max(mx.min(), span.start), mx.max(), 300)
        else:
            grid = np.linspace(mx.min(), mx.max(), 300)
        hf = model.smoothed_hf(grid, style["hf_bandwidth"])
        (result.lines[label],) = ax.plot(grid, hf, **line_kwargs)
        return

    # The step: from the start of the axis, through every value of the
    # ladder, to the end of the data
    steps = np.unique(np.concatenate([[span.start], mx, [max(last, mx[-1])]]))
    steps = steps[steps >= span.start]
    y = _curve(model, on, steps)
    (line,) = ax.step(steps, y, where="post", **line_kwargs)
    color = line.get_color()
    result.lines[label] = line

    ci = style["ci"]
    has_variance = getattr(model, "greenwood", None) is not None
    if ci not in (None, False) or (ci is None and has_variance):
        _draw_bounds(ax, model, steps[steps <= mx[-1]], color, style)

    b = _data_bounds(model)
    ticks = style["show_censors"]
    if ticks is None:
        ticks = b is not None
    if ticks and b is None:
        raise ValueError(
            "Ticking the censored times of {!r} needs the data it was "
            "fitted with, which this model does not hold. Pass "
            "show_censors=False, or save the model with "
            "to_dict(with_data=True).".format(label)
        )
    if ticks and b is not None:
        xc = np.unique(b["lower"][b["c"] == 1])
        xc = xc[np.isfinite(xc) & (xc >= span.start)]
        if xc.size:
            ax.plot(
                xc,
                _curve(model, on, xc),
                linestyle="",
                marker="|",
                markersize=10,
                markeredgewidth=1.5,
                color=color,
            )

    if style["quantiles"] is not None:
        result.quantiles[label] = _draw_quantiles(
            ax, model, on, style["quantiles"], color, span.lo
        )


def _draw_bounds(
    ax: Any,
    model: NonParametric,
    q: npt.NDArray,
    color: Any,
    style: dict[str, Any],
) -> None:
    ci = style["ci"]
    kind = "pointwise" if ci is None else str(ci)
    cbs = _bounds(
        model,
        style["on"],
        q,
        kind,
        style["alpha_ci"],
        style["bound"],
        style["bound_type"],
        style["band_method"],
    )
    if np.ndim(cbs) == 2 and style["ci_style"] == "fill":
        ax.fill_between(
            q,
            cbs[:, 0],
            cbs[:, 1],
            step="post",
            color=color,
            alpha=plotting.BAND_ALPHA if plotting.theme_enabled() else 0.3,
            linewidth=0,
        )
        return
    cols = [cbs] if np.ndim(cbs) == 1 else [cbs[:, 0], cbs[:, 1]]
    for col in cols:
        ax.step(
            q,
            col,
            where="post",
            color=color,
            linestyle="--",
            linewidth=1.0,
        )


def _draw_quantiles(
    ax: Any,
    model: NonParametric,
    on: str,
    quantiles: Any,
    color: Any,
    lo: float,
) -> dict[float, float]:
    """Dotted lines across at each quantile's level and down at its time,
    labelled with the time; returns ``{p: time}``."""
    out = {}
    muted = plotting.INK["muted"] if plotting.theme_enabled() else "grey"
    for p in np.atleast_1d(np.asarray(quantiles, dtype=float)):
        qt = model.qf(float(p))
        out[float(p)] = qt
        if not np.isfinite(qt):
            continue
        level = _quantile_level(on, float(p))
        ax.plot([lo, qt], [level, level], ls=":", lw=1.0, color=muted)
        ax.plot([qt, qt], [0.0, level], ls=":", lw=1.0, color=color)
        ax.annotate(
            f"{qt:.4g}",
            xy=(qt, 0.0),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color=color,
        )
    return out


def _finish_axes(
    ax: Any,
    pairs: list[tuple[str, NonParametric]],
    on: str,
    span: _Span,
    log_x: bool,
    xlabel: str | None,
    ylabel: str | None,
    title: str | None,
) -> None:
    if log_x:
        ax.set_xscale("log")
    ax.set_xlim(span.lo, span.hi)
    if on in ("sf", "ff"):
        ax.set_ylim(0, 1.02)
    else:
        ax.set_ylim(bottom=0)
    units = {getattr(m, "unit", None) for _, m in pairs}
    unit = units.pop() if len(units) == 1 else None
    ax.set_xlabel(xlabel if xlabel is not None else axis_label(unit))
    ax.set_ylabel(ylabel if ylabel is not None else _YLABELS[on])
    if title is None:
        names = {m.model for _, m in pairs}
        if len(names) == 1:
            name = names.pop()
            title = (
                "Survival estimate"
                if name == "from_ecdf"
                else f"{name} estimate"
            )
    if title:
        ax.set_title(title)
    plotting.style_axes(ax)


def _table_times(ax: Any, span: _Span, at_risk_times: Any) -> npt.NDArray:
    """The table's times: the major ticks in range, unless given."""
    if at_risk_times is None:
        times = np.asarray(ax.get_xticks(), dtype=float)
        times = times[(times >= span.lo) & (times <= span.hi)]
        if times.size == 0:
            times = np.array([span.lo, span.hi])
        return times
    times = np.atleast_1d(np.asarray(at_risk_times, dtype=float))
    if times.ndim != 1 or not np.isfinite(times).all():
        raise ValueError(
            "'at_risk_times' must be a one-dimensional array of finite "
            "times."
        )
    return np.sort(times)


def _draw_table(
    ax: Any,
    result: NonParametricPlot,
    pairs: list[tuple[str, NonParametric]],
    span: _Span,
    at_risk_times: Any,
    rows: list[str],
    counts: str,
) -> None:
    from matplotlib import transforms

    times = _table_times(ax, span, at_risk_times)
    # The axis's ticks fixed at the table's times, so the columns stay
    # under them
    ax.set_xticks(times)
    ax.set_xlim(span.lo, span.hi)
    result.times = times
    for label, model in pairs:
        result.counts[label] = _counts(model, label, times, rows, counts)

    # Text at each time's x, in rows a fixed number of points below the
    # axes; the rows' names to the left of the axes
    fig = ax.get_figure()
    blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    text_color = plotting.INK["secondary"] if plotting.theme_enabled() else "k"
    in_range = (times >= span.lo) & (times <= span.hi)
    line_no = 0

    def put(x: float, text: str, cell: bool, **kwargs: Any) -> None:
        kwargs.setdefault("color", text_color)
        dy = -(_TABLE_TOP + line_no * _ROW_HEIGHT)
        tf = transforms.offset_copy(
            blend if cell else ax.transAxes,
            fig=fig,
            x=0.0 if cell else -_LABEL_GAP,
            y=dy,
            units="points",
        )
        ax.text(
            x,
            0,
            text,
            transform=tf,
            fontsize=_TABLE_FONT,
            va="top",
            ha="center" if cell else "right",
            clip_on=False,
            # "at-risk-cell" for a count, "at-risk-label" for a row's name
            gid="at-risk-cell" if cell else "at-risk-label",
            **kwargs,
        )

    if len(rows) == 1:
        put(0.0, rows[0], False, fontweight="bold")
        line_no += 1
    for label, _ in pairs:
        color = result.lines[label].get_color()
        if len(rows) > 1:
            put(0.0, label, False, color=color)
            line_no += 1
        for row in rows:
            if len(rows) == 1:
                put(0.0, label, False, color=color)
            else:
                put(0.0, "  " + row, False)
            values = result.counts[label][row]
            for t, v in zip(times[in_range], values[in_range]):
                put(t, _fmt(v), True)
            line_no += 1
