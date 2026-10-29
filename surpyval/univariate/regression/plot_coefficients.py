"""A forest (coefficient) plot of one or more fitted regression models:
:func:`plot_coefficients`.

It draws what each model's ``summary()`` holds: every coefficient with
its two-sided Wald interval, on the coefficient's scale or, where the
link is log-linear, as a ratio (``exp(coef)``). matplotlib is imported
only when a plot is drawn (#470).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from surpyval import plotting
from surpyval.utils.validation import check_alpha_ci, check_option

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure, SubFigure

__all__ = ["plot_coefficients", "CoefficientPlot"]

_ON = ("coef", "exp")
_SORTS = ("estimate", "name")

# What exp(coef) is, by the model's kind (its ``kind`` attribute, or its
# class name where it has none).
_RATIOS = {
    "Cox": "Hazard ratio",
    "Proportional Hazard": "Hazard ratio",
    "CoxFrailty": "Hazard ratio",
    "Frailty": "Hazard ratio",
    "FineGrayModel": "Subdistribution hazard ratio",
    "Accelerated Failure Time": "Acceleration factor",
    "Proportional Odds": "Odds ratio",
}

# The vertical spread of the models drawn at one covariate (dodging), as
# a fraction of the gap between covariates.
_DODGE = 0.6


@dataclass
class CoefficientPlot:
    """What :func:`plot_coefficients` drew.

    Attributes
    ----------
    fig : matplotlib Figure
        The figure the axes are on.
    ax : matplotlib Axes
        The axes the coefficients are drawn on.
    on : str
        The scale drawn: ``'coef'`` or ``'exp'``.
    covariates : list
        The covariates, top to bottom.
    values : dict
        ``{label: DataFrame}``: for each model, the ``estimate``,
        ``lower`` and ``upper`` drawn for each covariate (indexed by the
        covariate, in the order of ``covariates``; ``nan`` for a
        covariate the model does not have, or an aliased one).
    points : dict
        ``{label: Line2D}``: each model's point estimates.
    intervals : dict
        ``{label: LineCollection}``: each model's intervals.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import surpyval
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static().assign(censored=lambda d: 1 - d.arrest)
    >>> model = surpyval.CoxPH.fit_from_df(
    ...     df, x_col="week", c_col="censored", Z_cols=["fin", "prio"]
    ... )
    >>> out = surpyval.plot_coefficients(model)
    >>> out.on, out.covariates
    ('exp', ['fin', 'prio'])
    >>> out.values["CoxPH"].round(3).values.tolist()
    [[0.671, 0.462, 0.974], [1.11, 1.053, 1.169]]
    >>> matplotlib.pyplot.close(out.fig)
    """

    fig: Figure | SubFigure
    ax: Axes
    on: str
    covariates: list = field(default_factory=list)
    values: dict = field(default_factory=dict)
    points: dict = field(default_factory=dict)
    intervals: dict = field(default_factory=dict)


def _name(model: Any) -> str:
    """A model's default label: its fitter's family, else its class."""
    kind = getattr(model, "kind", None)
    if kind == "Cox":
        return "CoxPH"
    dist = getattr(model, "dist", None)
    if kind and dist is not None and hasattr(dist, "name"):
        short = {
            "Proportional Hazard": "PH",
            "Accelerated Failure Time": "AFT",
            "Proportional Odds": "PO",
            "Additive Hazard": "AH",
        }.get(str(kind))
        if short:
            return f"{dist.name}{short}"
    return type(model).__name__


def _as_models(models: Any, labels: Any) -> list[tuple[str, Any]]:
    if isinstance(models, Mapping):
        pairs = [(str(k), v) for k, v in models.items()]
    elif isinstance(models, (list, tuple)):
        pairs = [(_name(m), m) for m in models]
    else:
        pairs = [(_name(models), models)]
    if not pairs:
        raise ValueError(
            "'models' is empty; pass a fitted regression model (e.g. "
            "surpyval.CoxPH.fit(x, Z, c=c)), a list of them or a dict "
            "{label: model}."
        )
    if labels is not None:
        labels = [str(v) for v in labels]
        if len(labels) != len(pairs):
            raise ValueError(
                f"'labels' has {len(labels)} labels for {len(pairs)} "
                "models; give one label per model."
            )
        pairs = [(lab, m) for lab, (_, m) in zip(labels, pairs)]
    seen: dict[str, int] = {}
    for lab, _ in pairs:
        seen[lab] = seen.get(lab, 0) + 1
    if any(v > 1 for v in seen.values()):
        pairs = [(f"Model {k + 1}", m) for k, (_, m) in enumerate(pairs)]
    for lab, m in pairs:
        if not callable(getattr(m, "summary", None)):
            raise TypeError(
                f"'models': {lab!r} ({type(m).__name__}) has no "
                "coefficient table; pass fitted regression models (Cox, "
                "the parametric PH/AFT/PO/AH families, AdditiveHazards, "
                "BuckleyJames, frailty and Fine-Gray models), whose "
                "summary() lists their coefficients."
            )
    return pairs


def _table(
    label: str,
    model: Any,
    alpha_ci: float,
    covariates: Sequence[str] | None,
) -> pd.DataFrame:
    """The rows of ``model.summary(alpha_ci)`` to draw, indexed by the
    covariate's name: the ``"coefficients"`` part of a table with parts,
    or ``covariates`` (names from any part) where given."""
    table = model.summary(alpha_ci=alpha_ci)
    if not isinstance(table, pd.DataFrame) or "coef" not in table:
        raise TypeError(
            f"'models': {label!r} ({type(model).__name__}) has no "
            "coefficient table (its summary() has no 'coef' column); "
            "pass a fitted regression model."
        )
    if isinstance(table.index, pd.MultiIndex):
        parts = table.index.get_level_values(0)
        names = [str(v) for v in table.index.get_level_values(-1)]
        if covariates is None:
            keep = np.asarray(parts == "coefficients")
        else:
            keep = np.isin(names, list(covariates))
        table = table[keep]
        table.index = pd.Index(
            [n for n, k in zip(names, keep) if k], name="covariate"
        )
    else:
        table = table.copy()
        table.index = pd.Index([str(v) for v in table.index], name="covariate")
        if covariates is not None:
            table = table[table.index.isin(list(covariates))]
    if table.empty:
        if covariates is None:
            raise ValueError(
                f"'models': {label!r} has no regression coefficients in its "
                "summary() (an accelerated life model's are its life "
                "model's parameters); name the rows to draw with "
                "covariates=[...]."
            )
        raise ValueError(
            f"'covariates': none of {list(covariates)} is in the "
            f"summary() of {label!r}, whose rows are "
            f"{[str(v) for v in model.summary().index]}."
        )
    return table


def _columns(table: pd.DataFrame, on: str) -> list[str]:
    level = [c for c in table.columns if c.startswith("coef lower ")]
    pct = level[0][len("coef lower ") :]
    if on == "coef":
        return ["coef", "coef lower " + pct, "coef upper " + pct]
    return ["exp(coef)", "exp(coef) lower " + pct, "exp(coef) upper " + pct]


def _has_ratio(table: pd.DataFrame) -> bool:
    finite_coef = table["coef"].notna()
    return bool(finite_coef.any()) and bool(
        table.loc[finite_coef, "exp(coef)"].notna().all()
    )


def _kind(model: Any) -> str:
    kind = getattr(model, "kind", None)
    return str(kind) if kind else type(model).__name__


def _xlabel(models: list[tuple[str, Any]], on: str) -> str:
    if on == "coef":
        return "Coefficient"
    ratios = {_RATIOS.get(_kind(m)) for _, m in models}
    if len(ratios) == 1 and None not in ratios:
        return f"{ratios.pop()}, exp(coef)"
    return "exp(coef)"


def _order(
    covariates: list[str], first: pd.DataFrame, sort: str | None
) -> list[str]:
    if sort is None:
        return covariates
    if sort == "name":
        return sorted(covariates)
    est = first["estimate"].reindex(covariates)
    # NaN (a covariate the first model lacks) last
    keys = np.where(est.isna(), np.inf, est.to_numpy(dtype=float))
    return [covariates[k] for k in np.argsort(keys, kind="stable")]


def plot_coefficients(
    models: Any,
    on: str | None = None,
    ax: Axes | None = None,
    *,
    alpha_ci: float = 0.05,
    covariates: Sequence[str] | None = None,
    labels: Sequence[str] | None = None,
    colors: Any = None,
    sort: str | None = None,
    xlabel: str | None = None,
    title: str | None = None,
    legend: bool | None = None,
) -> CoefficientPlot:
    """
    A forest plot: each coefficient of one or more fitted regression
    models, as a point with its two-sided Wald confidence interval, one
    row per covariate, with a reference line at no effect.

    The values are those of each model's ``summary(alpha_ci)``, so any
    model with a coefficient table can be drawn: ``CoxPH``, the
    parametric PH, AFT, PO and AH families, ``AdditiveHazards``,
    ``ProportionalOdds``, ``BuckleyJames``, the frailty models and
    ``FineGray``. Several models are drawn side by side at each
    covariate, for example the same covariates under a Cox and a Weibull
    AFT fit, or one fit per subgroup.

    Parameters
    ----------
    models : model, list or dict
        One fitted regression model, a list of them, or ``{label:
        model}``.
    on : ('coef', 'exp'), optional
        The scale: ``'coef'`` the coefficients, with the reference line
        at 0; ``'exp'`` their exponents ``exp(coef)`` (a hazard ratio for
        Cox and proportional hazards, the acceleration factor for AFT, an
        odds ratio for proportional odds), on a log axis with the
        reference line at 1. Defaults to ``'exp'`` when every model has a
        ratio scale, else ``'coef'``. An additive model (``AH``,
        ``AdditiveHazards``) has none: its coefficients are differences
        in the hazard, not ratios, so ``'exp'`` is refused for it.
    ax : matplotlib Axes, optional
        The axes to draw on. Without one a new figure is made.
    alpha_ci : float, optional
        The significance level of the intervals. Defaults to 0.05.
    covariates : list of str, optional
        The rows to draw, by name. Defaults to every regression
        coefficient (the ``"coefficients"`` part of a parametric
        model's table, which leaves out the baseline's and a frailty's
        parameters). Names can come from any part of the table, e.g. an
        accelerated life model's life-model parameters.
    labels : list of str, optional
        One label per model, in order; the keys of a dict otherwise, and
        the model's family name (or "Model 1", "Model 2", ... when names
        repeat) for a list.
    colors : list or dict, optional
        One matplotlib colour per model, or ``{label: colour}``. Without
        them the models take SurPyval's palette in order.
    sort : ('estimate', 'name'), optional
        The order of the rows, top to bottom: as in the (first model's)
        table (default), by the first model's estimate, smallest at the
        top, or by name.
    xlabel, title : str, optional
        The x axis label (defaults to the scale drawn) and the title.
    legend : bool, optional
        Whether to draw a legend. Defaults to one when there are two or
        more models.

    Returns
    -------
    CoefficientPlot
        ``fig`` and ``ax`` to customise further, the ``covariates`` drawn
        top to bottom, the ``values`` drawn and the artists.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import surpyval
    >>> from surpyval.datasets import load_rossi_static
    >>> df = load_rossi_static().assign(censored=lambda d: 1 - d.arrest)
    >>> kw = dict(x_col="week", c_col="censored", Z_cols=["fin", "age"])
    >>> cox = surpyval.CoxPH.fit_from_df(df, **kw)
    >>> aft = surpyval.WeibullAFT.fit_from_df(df, **kw)
    >>> out = surpyval.plot_coefficients(
    ...     {"Cox": cox, "Weibull AFT": aft}, on="coef"
    ... )
    >>> out.values["Cox"]["estimate"].round(3).tolist()
    [-0.328, -0.071]
    >>> out.values["Weibull AFT"]["estimate"].round(3).tolist()
    [-0.239, -0.052]
    >>> plt.close(out.fig)

    The Cox coefficients are log hazard ratios and the AFT's log
    acceleration factors (an AFT model ages a unit at ``exp(coef)`` times
    the baseline's rate), so a negative value means the same in both:
    financial aid and age both lower the risk of arrest.
    """
    if on is not None:
        check_option("on", on, _ON)
    if sort is not None:
        check_option("sort", sort, _SORTS)
    check_alpha_ci(alpha_ci)
    pairs = _as_models(models, labels)
    tables = [
        (label, model, _table(label, model, alpha_ci, covariates))
        for label, model in pairs
    ]
    if on is None:
        on = "exp" if all(_has_ratio(t) for _, _, t in tables) else "coef"
    elif on == "exp":
        for label, model, t in tables:
            if not _has_ratio(t):
                raise ValueError(
                    f"on='exp': {label!r} ({_kind(model)}) has no ratio "
                    "scale (exp(coef) is nan in its summary(): an "
                    "additive effect, or a parameter that is not a "
                    "log-linear coefficient); use on='coef'."
                )
    values: dict[str, pd.DataFrame] = {}
    for label, _, t in tables:
        cols = _columns(t, on)
        values[label] = pd.DataFrame(
            {
                "estimate": t[cols[0]].to_numpy(dtype=float),
                "lower": t[cols[1]].to_numpy(dtype=float),
                "upper": t[cols[2]].to_numpy(dtype=float),
            },
            index=t.index,
        )
    names: list[str] = []
    for frame in values.values():
        names += [v for v in frame.index if v not in names]
    order = _order(names, next(iter(values.values())), sort)
    values = {k: v.reindex(order) for k, v in values.items()}
    return _draw(pairs, values, order, on, ax, colors, xlabel, title, legend)


def _color(colors: Any, k: int, label: str) -> Any:
    if colors is None:
        return None
    if isinstance(colors, Mapping):
        return colors.get(label)
    return list(colors)[k]


def _draw(
    pairs: list[tuple[str, Any]],
    values: dict[str, pd.DataFrame],
    order: list[str],
    on: str,
    ax: Axes | None,
    colors: Any,
    xlabel: str | None,
    title: str | None,
    legend: bool | None,
) -> CoefficientPlot:
    import matplotlib.pyplot as plt

    fig: Figure | SubFigure
    if ax is None:
        height = max(2.5, 0.45 * len(order) * max(1, len(pairs) ** 0.5) + 1)
        fig, ax = plt.subplots(figsize=(7, height), layout="constrained")
    else:
        fig = ax.figure
    plotting.style_axes(ax)
    ax.grid(False, axis="y")
    k = len(pairs)
    rows = np.arange(len(order), dtype=float)
    offsets = (np.arange(k) - (k - 1) / 2) * (_DODGE / max(k, 1))
    out = CoefficientPlot(fig=fig, ax=ax, on=on, covariates=list(order))
    for j, (label, _) in enumerate(pairs):
        frame = values[label]
        color = plotting.next_color(ax, _color(colors, j, label))
        if color is None:  # the theme is off: matplotlib's colours
            color = f"C{j % 10}"
        y = rows + offsets[j]
        out.intervals[label] = ax.hlines(
            y,
            frame["lower"].to_numpy(),
            frame["upper"].to_numpy(),
            color=color,
            linewidth=plotting.LINE_WIDTH,
        )
        (out.points[label],) = ax.plot(
            frame["estimate"].to_numpy(),
            y,
            "o",
            color=color,
            markersize=plotting.MARKER_SIZE,
            markeredgecolor=plotting.INK["surface"],
            label=label,
            linestyle="none",
        )
    ref = 1.0 if on == "exp" else 0.0
    ax.axvline(
        ref,
        color=plotting.INK["secondary"],
        linewidth=0.9,
        linestyle="--",
        zorder=0,
        gid="reference",
    )
    if on == "exp":
        ax.set_xscale("log")
    ax.set_yticks(rows, [str(v) for v in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.set_xlabel(xlabel if xlabel is not None else _xlabel(pairs, on))
    if title is not None:
        ax.set_title(title)
    if legend if legend is not None else k > 1:
        plotting.style_legend(ax, fontsize="small")
    out.values = values
    return out
