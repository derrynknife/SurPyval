from typing import Any, Callable

import numpy.typing as npt

"""
Shared probability plot construction for parametric models.

Used by both ``Parametric`` and ``MixtureModel`` which previously each
carried their own near-identical copy of this logic.
"""

import re
import warnings

import autograd.numpy as np

from surpyval.univariate.nonparametric import (
    canonical_heuristic,
    plotting_positions,
)
from surpyval.utils.numeric import _round_vals


def adjust_heuristic(
    c: npt.NDArray, t: npt.NDArray | None, heuristic: str
) -> str:
    """
    Force the Turnbull heuristic when the data is interval censored or
    truncated, warning that the requested heuristic was changed. The name
    is read whatever its case.
    """
    heuristic = canonical_heuristic(heuristic)
    if 2 in c:
        if heuristic != "Turnbull":
            warnings.warn(
                "Interval censored data, heuristic changed to Turnbull",
                stacklevel=2,
            )
            heuristic = "Turnbull"

    if np.isfinite(t).any():
        if heuristic != "Turnbull":
            warnings.warn(
                "Truncated censored data, heuristic changed to Turnbull",
                stacklevel=2,
            )
            heuristic = "Turnbull"

    return heuristic


def probability_plot_data(
    dist: Any,
    ff: Callable[..., Any],
    x: npt.NDArray,
    c: npt.NDArray | None,
    n: npt.NDArray,
    t: npt.NDArray,
    heuristic: str = "Nelson-Aalen",
    gamma: float = 0.0,
    params: npt.NDArray | None = None,
    cb_func: Callable[..., Any] | None = None,
) -> Any:
    """
    Compute everything needed to draw a probability plot of the data
    against the fitted CDF ``ff``.

    ``dist`` provides the plotting configuration (``plot_x_scale``,
    ``y_ticks`` and the special cased ``name``). ``gamma`` shifts the
    plotting positions for offset distributions. ``cb_func``, if given,
    is called with the model x values to compute confidence bounds on
    the CDF.

    ``x_`` and ``F`` are every row of the plotting-position estimator
    (with a finite ``x``), and ``failed`` is a boolean mask over them,
    True where the estimator records a failure (``d > 0``). Only those
    rows are points on the plot: a suspension (a right-censored unit)
    changes the plotting positions of the failures after it, but has no
    position of its own -- its row carries the ``F`` of the failure
    before it, where drawn it looked like one more failure (#478). This
    is the convention of Abernethy's *New Weibull Handbook* and of
    Weibull++. The Turnbull rows where the estimate puts no mass are not
    failures either. ``x_censored`` holds the suspension times, for a
    plot that wants to mark them.
    """
    x_, r, d, F = plotting_positions(
        x=x,
        c=c,
        n=n,
        t=t,
        heuristic=heuristic,
    )

    mask = np.isfinite(x_)
    x_ = x_[mask] - gamma
    failed = d[mask] > 0
    F = F[mask]

    x_arr = np.asarray(x, dtype=float)
    c_arr = np.zeros(x_arr.shape[0]) if c is None else np.asarray(c)
    if x_arr.ndim == 2:
        x_arr = x_arr[:, 0]
    x_censored = np.unique(x_arr[(c_arr == 1) & np.isfinite(x_arr)]) - gamma

    # Adjust the plotting points in event data is truncated.
    tl_min = t[0][0]
    if np.isfinite(tl_min):
        Ftl = ff(tl_min)
    else:
        Ftl = 0

    tr_max = t[-1][-1]
    if np.isfinite(tr_max):
        Ftr = ff(tr_max)
    else:
        Ftr = 1

    # Adjust the plotting points due to truncation
    F = Ftl + F * (Ftr - Ftl)

    out = _axes_data(dist, ff, x_, F, gamma, params, cb_func, x_censored)
    out["failed"] = failed
    return out


def curve_plot_data(
    dist: Any,
    ff: Callable[..., Any],
    qf: Callable[..., Any],
    gamma: float = 0.0,
    params: npt.NDArray | None = None,
) -> Any:
    """
    The probability plot data of a model without data (#485): the axes
    span the model's 1% to 99% quantiles, and there are no plotting
    points or confidence bounds -- only the model's CDF.
    """
    probs = np.array([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    with np.errstate(all="ignore"):
        x_ = np.asarray(qf(probs), dtype=float) - gamma
    keep = np.isfinite(x_)
    if dist.plot_x_scale == "log":
        keep &= x_ > 0
    if keep.sum() < 2:
        raise ValueError(
            "Can't plot this model: its quantiles are not finite, so there "
            "is no range of x to plot it over"
        )
    out = _axes_data(dist, ff, x_[keep], probs[keep], gamma, params, None)
    # The quantiles only placed the axes; nothing was observed.
    out["x_"] = np.array([])
    out["F"] = np.array([])
    out["failed"] = np.array([], dtype=bool)
    return out


def _axes_data(
    dist: Any,
    ff: Callable[..., Any],
    x_: npt.NDArray,
    F: npt.NDArray,
    gamma: float,
    params: npt.NDArray | None,
    cb_func: Callable[..., Any] | None,
    x_censored: npt.NDArray | None = None,
) -> Any:
    """The axes, ticks and model curve of a probability plot of the rows
    ``(x_, F)``, with the suspension times ``x_censored``."""
    if x_censored is None:
        x_censored = np.array([])
    # The time axis spans every time, suspensions included, although only
    # the failures are drawn as points.
    x_axis = np.concatenate([x_, x_censored])

    # x-axis
    if dist.plot_x_scale == "log":
        log_x = np.log10(x_axis[x_axis > 0])
        x_min = np.min(log_x)
        x_max = np.max(log_x)
        vals_non_sig = 10 ** np.linspace(x_min, x_max, 7)
        x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
        x_minor_ticks = (
            10**x_minor_ticks * np.array(np.arange(1, 11)).reshape((10, 1))
        ).flatten()
        diff = (x_max - x_min) / 10
        x_scale_min = 10 ** (x_min - diff)
        x_scale_max = 10 ** (x_max + diff)
        x_model = 10 ** np.linspace(x_min - diff, x_max + diff, 100)
    elif dist._plot_x_bounds(x_axis, params) is not None:
        x_min = np.min(x_axis)
        x_max = np.max(x_axis)
        x_scale_min, x_scale_max = dist._plot_x_bounds(x_axis, params)
        vals_non_sig = np.linspace(x_scale_min, x_scale_max, 11)[1:-1]
        x_minor_ticks = np.linspace(x_scale_min, x_scale_max, 22)[1:-1]
        x_model = np.linspace(x_scale_min, x_scale_max, 102)[1:-1]
    else:
        x_min = np.min(x_axis)
        x_max = np.max(x_axis)
        vals_non_sig = np.linspace(x_min, x_max, 7)
        x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
        diff = (x_max - x_min) / 10
        x_scale_min = x_min - diff
        x_scale_max = x_max + diff
        x_model = np.linspace(x_scale_min, x_scale_max, 100)

    cdf = ff(x_model + gamma)

    # The probability axis spans the points (a suspension's row repeats
    # the F of a failure, or is 0 before the first), or with no failure
    # to plot (a zero-failure fit) the fitted curve.
    inside = F[(F > 0) & (F < 1)]
    if inside.size == 0:
        inside = cdf[(cdf > 0) & (cdf < 1)]
    y_scale_min = np.min(inside) / 2
    y_scale_max = 1 - (1 - np.max(inside)) / 10

    x_ticks = _round_vals(vals_non_sig)
    x_ticks_labels = [
        (
            str(int(x))
            if (re.match(r"([0-9]+\.0+)", str(x)) is not None) and (x > 1)
            else str(x)
        )
        for x in _round_vals(vals_non_sig + gamma)
    ]

    y_ticks = np.array(dist.y_ticks)
    y_ticks = y_ticks[
        np.where((y_ticks > y_scale_min) & (y_ticks < y_scale_max))[0]
    ]

    y_ticks_labels = [
        (
            str(int(y)) + "%"
            if (re.match(r"([0-9]+\.0+)", str(y)) is not None) and (y > 1)
            else str(y)
        )
        for y in y_ticks * 100
    ]

    if cb_func is not None:
        cbs = cb_func(x_model + gamma)
    else:
        cbs = []

    return {
        "x_scale_min": x_scale_min,
        "x_scale_max": x_scale_max,
        "y_scale_min": y_scale_min,
        "y_scale_max": y_scale_max,
        "y_ticks": y_ticks,
        "y_ticks_labels": y_ticks_labels,
        "x_ticks": x_ticks,
        "x_ticks_labels": x_ticks_labels,
        "cdf": cdf,
        "x_model": x_model,
        "x_minor_ticks": x_minor_ticks,
        "cbs": cbs,
        "x_scale": dist.plot_x_scale,
        "x_": x_,
        "F": F,
        "x_censored": x_censored,
        "gamma": gamma,
    }


#: Probabilities that may label a probability plot's axis, in the order
#: they are wanted: the median, then the decades and their mirrors, then
#: the halves and fifths between. Each is kept where it is far enough,
#: on the distribution's own scale, from those kept before it.
_PROBABILITY_TICKS = (
    0.5, 0.1, 0.9, 0.01, 0.99, 0.001, 0.999, 0.0001, 0.9999,
    0.05, 0.95, 0.2, 0.8, 0.3, 0.7, 0.02, 0.98, 0.005, 0.995,
    0.4, 0.6, 0.002, 0.998, 0.0005, 0.9995, 0.03, 0.97, 0.003, 0.997,
    0.0002, 0.9998,
)  # fmt: skip

#: The faint lines between them: 1 to 9 of each decade, and 91% to 99%
#: (finer near 1 they crowded into a band on a Weibull's scale).
_PROBABILITY_MINOR = tuple(
    sorted(
        {m * 10.0**e for e in range(-4, 0) for m in range(1, 10)}
        | {1 - m / 100 for m in range(1, 10)}
    )
)


def _probability_ticks(
    y_transform: Callable[..., Any], lo: float, hi: float, n: int = 14
) -> tuple[list, list]:
    """The labelled and the faint probabilities of an axis spanning
    ``(lo, hi)``: from ``_PROBABILITY_TICKS`` in order, each kept where it
    is at least ``1 / n`` of the axis, on its own scale, from every one
    kept before (a Weibull's 99% and 99.9% sit close; a Normal's do not),
    so that every distribution's axis is labelled as densely as it has
    room for, the tails included."""

    def at(p: float) -> float:
        return float(np.ravel(y_transform(np.array([p])))[0])

    with np.errstate(all="ignore"):
        span = at(hi) - at(lo)
        kept: list = []
        if np.isfinite(span) and span > 0:
            gap = span / n
            for p in _PROBABILITY_TICKS:
                if lo < p < hi and all(
                    abs(at(p) - at(k)) >= gap for k in kept
                ):
                    kept.append(p)
    minor = [p for p in _PROBABILITY_MINOR if lo < p < hi]
    return sorted(kept), minor


def _probability_label(p: float) -> str:
    """``0.001`` as ``"0.1%"``, ``0.5`` as ``"50%"``."""
    return f"{100 * p:.10g}%"


def _nice_value(v: float) -> str:
    """A time for a tick label: thousands separated, no trailing zeros."""
    if abs(v) >= 1000 and float(v).is_integer():
        return f"{v:,.0f}"
    return f"{v:.6g}"


def _roundness(v: float) -> int:
    """How round a time is, for picking ticks (lower is rounder): its
    significant figures, less one for a leading 1, 2 or 5 (``1000`` and
    ``500`` before ``800``, which is before ``1300``)."""
    digits = f"{abs(v):.6g}".replace(".", "").strip("0") or "0"
    return 2 * len(digits) - (digits[0] in "125" and len(digits) == 1)


def _log_ticks(
    lo: float, hi: float, gamma: float = 0.0, n: int = 8
) -> tuple[list, list, list]:
    """The labelled and faint positions, and the labels, of a log time
    axis spanning ``(lo, hi)`` (times less ``gamma``, an offset): times
    of 1, 2 and 5 a decade, the decades first, kept at least ``1 / n`` of
    the axis apart; 1 to 9 a decade where those give fewer than three (a
    narrow range). With an offset the labelled times are round times
    (``gamma`` plus the position), not round positions."""
    if not (lo > 0 and hi > lo):
        return [], [], []
    shown_lo, shown_hi = lo + gamma, hi + gamma
    if gamma == 0:
        d0, d1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
        pick: list[tuple[float, int]] = [
            (m * 10.0**e, rank)
            for e in range(d0, d1 + 1)
            for m, rank in ((1, 0), (5, 1), (2, 2))
        ]
        fine = [m * 10.0**e for e in range(d0, d1 + 1) for m in range(1, 10)]
    else:
        from matplotlib.ticker import MaxNLocator

        values = MaxNLocator(nbins=24, steps=[1, 2, 5, 10]).tick_values(
            shown_lo, shown_hi
        )
        # The roundest first: 1, 2 or 5 and zeros, then fewer figures
        pick = [(float(v), _roundness(float(v))) for v in values if v > 0]
        fine = []

    def thin(candidates: list) -> list:
        gap = (np.log10(hi) - np.log10(lo)) / n
        kept: list = []
        for v, _ in sorted(candidates, key=lambda c: c[1]):
            position = v - gamma
            if not (lo <= position <= hi) or position <= 0:
                continue
            if all(
                abs(np.log10(position) - np.log10(k - gamma)) >= gap
                for k in kept
            ):
                kept.append(v)
        return sorted(kept)

    kept = thin(pick)
    if len(kept) < 3 and gamma == 0:
        kept = thin([(v, 0) for v in fine])
    minor = [v for v in fine if lo <= v <= hi]
    return [v - gamma for v in kept], minor, [_nice_value(v) for v in kept]


def _linear_ticks(
    lo: float, hi: float, gamma: float = 0.0
) -> tuple[list, list]:
    """The positions and labels of a linear time axis spanning ``(lo,
    hi)`` (times less ``gamma``): round times, about seven."""
    from matplotlib.ticker import MaxNLocator

    values = MaxNLocator(nbins=7, steps=[1, 2, 2.5, 5, 10]).tick_values(
        lo + gamma, hi + gamma
    )
    kept = [float(v) for v in values if lo <= v - gamma <= hi]
    return [v - gamma for v in kept], [_nice_value(v) for v in kept]


def draw_probability_plot(
    ax: Any,
    d: Any,
    y_transform: Callable[..., Any],
    inv_y_transform: Callable[..., Any],
    title: str,
    plot_bounds: Any = False,
    show_censored: bool = False,
    color: Any = None,
    label: "str | None" = None,
    xlabel: str = "Time",
    **line_kwargs: Any,
) -> Any:
    """
    Draw the probability plot described by the ``probability_plot_data``
    dictionary ``d`` onto the matplotlib axes ``ax``, in SurPyval's theme
    (:mod:`surpyval.plotting`). The points are the rows ``d["failed"]``
    selects, the failures; with ``show_censored``, each suspension time
    is marked by a tick on the time axis (they have no plotting position
    of their own).

    The points, the fitted line and its bounds are drawn in one colour,
    ``color``, or by default the theme's next colour for these axes, so
    that several models plotted on one axes can be told apart (#510).
    ``label`` and any other keyword arguments go to the fitted line (a
    ``matplotlib.lines.Line2D``); the two-sided bounds are a pale band of
    the line's colour. The probability axis is labelled as densely as the
    distribution's scale has room for (:func:`_probability_ticks`), a log
    time axis at 1, 2 and 5 a decade, a linear one at round times; on
    axes that already hold a plot the limits are widened to cover both
    and the ticks chosen again. The x label is ``xlabel`` unless the axes
    already have one.
    """
    from matplotlib.ticker import FixedLocator, NullFormatter

    from surpyval import plotting

    y_lim = [max(d["y_scale_min"], 1e-4), min(d["y_scale_max"], 0.9999)]
    x_lim = [d["x_scale_min"], d["x_scale_max"]]
    if ax.has_data():
        # Drawing over an earlier plot: keep its range in view.
        (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
        x_lim = [min(x_lim[0], x0), max(x_lim[1], x1)]
        y_lim = [min(y_lim[0], y0), max(y_lim[1], y1)]
    log = d["x_scale"] == "log"
    gamma = float(d.get("gamma", 0.0))

    ax.set_xscale(d["x_scale"])
    ax.set_yscale("function", functions=(y_transform, inv_y_transform))

    color = plotting.next_color(ax, color)
    line_kwargs.setdefault("linewidth", plotting.LINE_WIDTH)
    line_kwargs.setdefault("solid_capstyle", "round")
    # The fitted line first, so that it takes the next colour of the
    # axes' cycle where the theme is off; the points and bounds share it.
    (line,) = ax.plot(
        d["x_model"], d["cdf"], color=color, label=label, **line_kwargs
    )
    color = line.get_color()

    if plot_bounds and (len(d["cbs"]) != 0):
        cbs = np.asarray(d["cbs"], dtype=float)
        if cbs.ndim == 2 and cbs.shape[1] == 2:
            ax.fill_between(
                d["x_model"],
                cbs[:, 0],
                cbs[:, 1],
                color=color,
                alpha=plotting.BAND_ALPHA,
                linewidth=0,
                zorder=line.get_zorder() - 1,
            )
        else:
            ax.plot(
                d["x_model"],
                cbs,
                color=color,
                linestyle="--",
                linewidth=0.8 * line.get_linewidth(),
            )

    failed = d["failed"]
    themed = plotting.theme_enabled()
    ax.scatter(
        d["x_"][failed],
        d["F"][failed],
        color=color,
        s=plotting.MARKER_SIZE**2 if themed else None,
        edgecolors=plotting.INK["surface"] if themed else None,
        linewidths=1.0 if themed else None,
        zorder=line.get_zorder() + 1,
    )

    if show_censored and len(d.get("x_censored", [])) != 0:
        # A rug of suspension times along the bottom of the axes
        ax.plot(
            d["x_censored"],
            np.zeros(len(d["x_censored"])),
            linestyle="none",
            marker="|",
            markersize=12,
            color=color,
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            label="_suspensions",
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    # The probability axis
    y_ticks, y_minor = _probability_ticks(y_transform, *y_lim)
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.set_yticklabels([_probability_label(p) for p in y_ticks])
    ax.yaxis.set_minor_locator(FixedLocator(y_minor))
    ax.yaxis.set_minor_formatter(NullFormatter())

    # The time axis
    if log:
        x_ticks, x_minor, x_labels = _log_ticks(x_lim[0], x_lim[1], gamma)
        ax.xaxis.set_minor_locator(FixedLocator(x_minor))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        x_ticks, x_labels = _linear_ticks(x_lim[0], x_lim[1], gamma)
    if x_ticks:
        ax.xaxis.set_major_locator(FixedLocator(x_ticks))
        ax.set_xticklabels(x_labels)

    ax.set_title(title)
    ax.set_ylabel("Fraction failed")
    if not ax.get_xlabel():
        # "Time", or what the model's unit measures: "Time (h)" (#845)
        ax.set_xlabel(xlabel)
    if plotting.theme_enabled():
        plotting.style_axes(ax, minor_grid=True)
    else:
        ax.grid(visible=True, which="major", alpha=0.4)
        ax.grid(visible=True, which="minor", alpha=0.1)
    return ax
