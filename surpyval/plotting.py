"""SurPyval's plot theme: one look for every model's ``plot``.

Every ``plot`` method draws in this theme: a fixed, colour-blind-checked
palette taken in order (the first model on an axes is blue, the next
orange, and so on, whatever else the axes hold), 2-point lines, filled
markers with a white ring, confidence bounds as a pale wash of the
series' colour, hairline grids one shade off the background, no top or
right spine, and text in greys rather than in series colours. A colour
or style passed to ``plot`` wins; the theme only fills in what is not
given.

>>> import surpyval
>>> surpyval.plotting.PALETTE[:3]
('#2a78d6', '#eb6834', '#1baf7a')

``set_theme(False)`` turns it off: the plots then draw with matplotlib's
own colours and leave the axes' styling alone. ``style_axes`` gives any
axes of your own the theme's chrome, and ``rc_params`` the whole theme as
``matplotlib.rcParams`` (``plt.rcParams.update(rc_params())``) for plots
made without SurPyval.

The palette is the reference categorical palette of eight hues, in an
order whose neighbours stay apart under the common colour-vision
deficiencies (OKLab difference of 9 or more between adjacent slots, and
19.6 or more for normal vision); the third to fifth slots are below 3:1
contrast on white, so a plot of two or more series carries a legend.

matplotlib is imported only by the functions that draw, so ``import
surpyval`` does not load it (#470).
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "PALETTE",
    "INK",
    "set_theme",
    "theme_enabled",
    "next_color",
    "style_axes",
    "style_legend",
    "observed_and_fitted",
    "rc_params",
]

#: The categorical palette, taken in this order (never cycled past
#: eight: a ninth series takes the first colour again).
PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#6250d6",  # violet
    "#e34948",  # red
)

#: The theme's neutrals: text, the observed data drawn behind a model,
#: grids and axes.
INK: dict[str, str] = {
    "primary": "#0b0b0b",  # titles
    "secondary": "#52514e",  # axis labels, legend text, observed steps
    "muted": "#898781",  # tick labels, rugs
    "grid": "#e1e0d9",  # major gridlines
    "grid_minor": "#f0efea",  # minor gridlines (probability paper)
    "axis": "#c3c2b7",  # spines and ticks
    "surface": "#ffffff",  # the axes' background and the markers' ring
}

#: The opacity of a confidence band's wash.
BAND_ALPHA = 0.15
#: Line width of a fitted curve or estimate, in points.
LINE_WIDTH = 1.8
#: Marker size of a data point, in points.
MARKER_SIZE = 6.0

_ENABLED = [True]
# The attribute on an axes that counts the series SurPyval drew on it.
_SERIES = "_surpyval_series"


def set_theme(enabled: bool = True) -> None:
    """Turn SurPyval's plot theme on (the default) or off for the plots
    that follow. Off, ``plot`` draws with matplotlib's colours and does
    not restyle the axes.

    Examples
    --------
    >>> import surpyval
    >>> surpyval.plotting.set_theme(False)
    >>> surpyval.plotting.theme_enabled()
    False
    >>> surpyval.plotting.set_theme(True)
    """
    _ENABLED[0] = bool(enabled)


def theme_enabled() -> bool:
    """Whether SurPyval's plot theme is on (see :func:`set_theme`)."""
    return _ENABLED[0]


def next_color(ax: Any, color: Any = None) -> Any:
    """The colour of the next series SurPyval draws on ``ax``: ``color``
    where one is given, else the palette's next slot (counted per axes,
    so the second model on an axes is orange); None with the theme off,
    which leaves the choice to matplotlib's cycle."""
    count = int(getattr(ax, _SERIES, 0))
    setattr(ax, _SERIES, count + 1)
    if color is not None:
        return color
    if not theme_enabled():
        return None
    return PALETTE[count % len(PALETTE)]


def style_axes(ax: Any, *, grid: bool = True, minor_grid: bool = False) -> Any:
    """Give ``ax`` the theme's chrome: a white background, hairline
    gridlines one shade off it (and fainter minor ones with
    ``minor_grid``, as probability paper has), no top or right spine,
    grey ticks and labels, and a dark title. Does nothing with the theme
    off. Returns ``ax``."""
    if not theme_enabled():
        return ax
    ax.set_facecolor(INK["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK["axis"])
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(
        which="major",
        colors=INK["axis"],
        labelcolor=INK["secondary"],
        labelsize=9,
        length=3,
        width=0.8,
    )
    ax.tick_params(which="minor", length=0)
    ax.xaxis.label.set_color(INK["secondary"])
    ax.yaxis.label.set_color(INK["secondary"])
    ax.title.set_color(INK["primary"])
    ax.title.set_fontsize(11)
    ax.title.set_fontweight("bold")
    if grid:
        ax.grid(True, which="major", color=INK["grid"], linewidth=0.8, ls="-")
        if minor_grid:
            ax.grid(
                True,
                which="minor",
                color=INK["grid_minor"],
                linewidth=0.6,
                ls="-",
            )
    ax.set_axisbelow(True)
    return ax


def style_legend(ax: Any, **kwargs: Any) -> Any:
    """Draw ``ax``'s legend in the theme (no frame, grey text), or as
    matplotlib draws it with the theme off; returns the legend."""
    if not theme_enabled():
        return ax.legend(**kwargs)
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("labelcolor", INK["secondary"])
    legend = ax.legend(**kwargs)
    if legend.get_title() is not None:
        legend.get_title().set_color(INK["secondary"])
    return legend


def observed_and_fitted(
    ax: Any,
    observed: tuple[Any, Any],
    fitted: tuple[Any, Any],
    band: Any = None,
    *,
    observed_label: str = "Observed",
    fitted_label: str = "Model",
    band_label: "str | None" = None,
) -> Any:
    """Draw an estimate from the data, ``observed`` (times and values, a
    step), behind a model's curve, ``fitted`` (times and values), with the
    model's two-sided ``band`` (an array of ``[lower, upper]`` rows at the
    fitted times) as a pale wash of its colour: the observed step in grey
    and the model in the theme's next colour, so that only the model is
    loud. With the theme off, the step is red and the model blue, as they
    were. Returns the model's line."""
    themed = theme_enabled()
    ax.step(
        *observed,
        where="post",
        color=INK["secondary"] if themed else "r",
        linewidth=1.2 if themed else None,
        label=observed_label,
    )
    color = next_color(ax) if themed else "b"
    (line,) = ax.plot(
        *fitted,
        color=color,
        linewidth=LINE_WIDTH if themed else None,
        label=fitted_label,
    )
    if band is not None:
        ax.fill_between(
            fitted[0],
            band[:, 0],
            band[:, 1],
            color=line.get_color(),
            alpha=BAND_ALPHA if themed else 0.2,
            linewidth=0,
            label=band_label,
        )
    return line


def rc_params() -> dict[str, Any]:
    """The theme as ``matplotlib.rcParams``, for figures made without
    SurPyval: ``plt.rcParams.update(surpyval.plotting.rc_params())``.

    Examples
    --------
    >>> import surpyval
    >>> surpyval.plotting.rc_params()["axes.spines.top"]
    False
    """
    from cycler import cycler

    return {
        "axes.prop_cycle": cycler(color=list(PALETTE)),
        "axes.facecolor": INK["surface"],
        "axes.edgecolor": INK["axis"],
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.labelcolor": INK["secondary"],
        "axes.titlecolor": INK["primary"],
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "grid.color": INK["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK["axis"],
        "ytick.color": INK["axis"],
        "xtick.labelcolor": INK["secondary"],
        "ytick.labelcolor": INK["secondary"],
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": MARKER_SIZE,
        "lines.solid_capstyle": "round",
        "legend.frameon": False,
        "legend.labelcolor": INK["secondary"],
        "figure.facecolor": INK["surface"],
    }
