"""
Shared probability plot construction for parametric models.

Used by both ``Parametric`` and ``MixtureModel`` which previously each
carried their own near-identical copy of this logic.
"""

import re
import warnings

from matplotlib.ticker import FixedLocator

from surpyval import np
from surpyval.univariate.nonparametric import plotting_positions
from surpyval.utils import _round_vals

CB_COLOUR = "#e94c54"


def adjust_heuristic(c, t, heuristic):
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
    dist,
    ff,
    x,
    c,
    n,
    t,
    heuristic="Nelson-Aalen",
    gamma=0.0,
    params=None,
    cb_func=None,
):
    """
    Compute everything needed to draw a probability plot of the data
    against the fitted CDF ``ff``.

    ``dist`` provides the plotting configuration (``plot_x_scale``,
    ``y_ticks`` and the special cased ``name``). ``gamma`` shifts the
    plotting positions for offset distributions. ``cb_func``, if given,
    is called with the model x values to compute confidence bounds on
    the CDF.
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
    r = r[mask]
    d = d[mask]
    F = F[mask]

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

    y_scale_min = np.min(F[F > 0]) / 2
    y_scale_max = 1 - (1 - np.max(F[F < 1])) / 10

    # x-axis
    if dist.plot_x_scale == "log":
        log_x = np.log10(x_[x_ > 0])
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
    elif dist._plot_x_bounds(x_, params) is not None:
        x_min = np.min(x_)
        x_max = np.max(x_)
        x_scale_min, x_scale_max = dist._plot_x_bounds(x_, params)
        vals_non_sig = np.linspace(x_scale_min, x_scale_max, 11)[1:-1]
        x_minor_ticks = np.linspace(x_scale_min, x_scale_max, 22)[1:-1]
        x_model = np.linspace(x_scale_min, x_scale_max, 102)[1:-1]
    else:
        x_min = np.min(x_)
        x_max = np.max(x_)
        vals_non_sig = np.linspace(x_min, x_max, 7)
        x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
        diff = (x_max - x_min) / 10
        x_scale_min = x_min - diff
        x_scale_max = x_max + diff
        x_model = np.linspace(x_scale_min, x_scale_max, 100)

    cdf = ff(x_model + gamma)

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
    }


def draw_probability_plot(
    ax,
    d,
    y_transform,
    inv_y_transform,
    title,
    plot_bounds=False,
):
    """
    Draw the probability plot described by the ``probability_plot_data``
    dictionary ``d`` onto the matplotlib axes ``ax``.
    """
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

    ax.set_xlim([d["x_scale_min"], d["x_scale_max"]])
    if plot_bounds and (len(d["cbs"]) != 0):
        ax.plot(d["x_model"], d["cbs"], color=CB_COLOUR)

    ax.plot(d["x_model"], d["cdf"], color="k", linestyle="--")
    return ax
