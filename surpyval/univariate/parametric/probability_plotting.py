from typing import Any, Callable

import numpy.typing as npt

"""
Shared probability plot construction for parametric models.

Used by both ``Parametric`` and ``MixtureModel`` which previously each
carried their own near-identical copy of this logic.
"""

import re
import warnings

from surpyval import np
from surpyval.univariate.nonparametric import plotting_positions
from surpyval.utils import _round_vals

CB_COLOUR = "#e94c54"


def adjust_heuristic(
    c: npt.NDArray, t: npt.NDArray | None, heuristic: str
) -> str:
    """
    Force the Turnbull heuristic when the data is interval censored or
    truncated, warning that the requested heuristic was changed.
    """
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

    The plotted points (``x_``, ``F``) are the failures only: the times
    at which the plotting-position estimator records a failure (``d >
    0``). A suspension (a right-censored unit) changes the plotting
    positions of the failures after it, but is not itself a point on the
    plot: drawn at the ``F`` of the failure before it, as it used to be,
    it looked like one more failure (#478). This is the convention of
    Abernethy's *New Weibull Handbook* and of Weibull++. The suspension
    times are returned separately, as ``x_censored``, for a plot that
    wants to mark them.
    """
    x_, r, d, F = plotting_positions(
        x=x,
        c=c,
        n=n,
        t=t,
        heuristic=heuristic,
    )

    mask = np.isfinite(x_) & (d > 0)
    x_ = x_[mask] - gamma
    r = r[mask]
    d = d[mask]
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

    # The time axis spans every time, suspensions included, as it did
    # when they were plotted.
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

    # The probability axis spans the points, or with no failure to plot
    # (a zero-failure fit) the fitted curve.
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
    }


def draw_probability_plot(
    ax: Any,
    d: Any,
    y_transform: Callable[..., Any],
    inv_y_transform: Callable[..., Any],
    title: str,
    plot_bounds: Any = False,
    show_censored: bool = False,
) -> Any:
    """
    Draw the probability plot described by the ``probability_plot_data``
    dictionary ``d`` onto the matplotlib axes ``ax``. With
    ``show_censored``, each suspension time is marked by a tick on the
    time axis (they have no plotting position of their own).
    """
    from matplotlib.ticker import FixedLocator

    # Set limits and scale
    ax.set_ylim([max(d["y_scale_min"], 1e-4), min(d["y_scale_max"], 0.9999)])
    ax.set_xscale(d["x_scale"])
    ax.set_yscale("function", functions=(y_transform, inv_y_transform))
    ax.set_yticks(d["y_ticks"])
    ax.set_yticklabels(d["y_ticks_labels"])
    ax.yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 51)))
    ax.set_xticks(d["x_ticks"])
    ax.set_xticklabels(d["x_ticks_labels"])

    if d["x_scale"] == "log":
        ax.set_xticks(d["x_minor_ticks"], minor=True)
        ax.set_xticklabels([], minor=True)

    ax.grid(visible=True, which="major", color="g", alpha=0.4, linestyle="-")
    ax.grid(visible=True, which="minor", color="g", alpha=0.1, linestyle="-")

    ax.set_title(title)
    ax.set_ylabel("CDF")
    ax.scatter(d["x_"], d["F"])

    if show_censored and len(d.get("x_censored", [])) != 0:
        # A rug of suspension times along the bottom of the axes
        ax.plot(
            d["x_censored"],
            np.zeros(len(d["x_censored"])),
            linestyle="none",
            marker="|",
            markersize=12,
            color="grey",
            transform=ax.get_xaxis_transform(),
            clip_on=False,
            label="suspensions",
        )

    ax.set_xlim([d["x_scale_min"], d["x_scale_max"]])
    if plot_bounds and (len(d["cbs"]) != 0):
        ax.plot(d["x_model"], d["cbs"], color=CB_COLOUR)

    ax.plot(d["x_model"], d["cdf"], color="k", linestyle="--")
    return ax
