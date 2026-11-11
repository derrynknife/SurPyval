"""
The helpers the diagnostics families share.

The Cox, parametric PH/AFT, frailty, competing-risks, proportional-odds
and additive-hazards checks (and the calibration, accelerated-life,
recurrent and univariate ones) each used to keep a copy of these. They
live here once; each family imports what it uses.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import numpy as np

from .registry import Context, Panel

#: The most distinct values a covariate has and still counts as
#: categorical (one with two is binary); more is continuous.
CATEGORICAL_LEVELS = 5


def _fmt(value: float) -> str:
    """``value`` to three significant figures, as the reports print it."""
    return f"{value:.3g}"


def _note(ctx: Context, text: str) -> None:
    """Add ``text`` to the report's notes, once."""
    if text not in ctx.notes:
        ctx.notes.append(text)


def _source(name: str) -> str:
    """The variable a formula-coded column comes from (``sex`` for
    ``C(sex)[T.male]`` or ``sex[T.male]``), else the name itself."""
    match = re.match(r"^C\(\s*([A-Za-z_][\w.]*)[^)]*\)\[", name)
    if match is None:
        match = re.match(r"^([A-Za-z_][\w.]*)\[", name)
    return match.group(1) if match else name


def _covariate_kind(levels: int) -> str:
    """``"binary"``, ``"categorical"`` or ``"continuous"``, for a
    covariate with ``levels`` distinct values."""
    if levels <= 2:
        return "binary"
    if levels <= CATEGORICAL_LEVELS:
        return "categorical"
    return "continuous"


def _trend_sign(g: np.ndarray, scaled: np.ndarray, weights: Any) -> float:
    """The sign of the weighted slope of a covariate's scaled Schoenfeld
    residuals ``scaled`` on the time scale ``g``: the direction its
    coefficient moves in."""
    return float(np.sign(np.cov(g, scaled, aweights=weights)[0, 1]))


def _ph_name(row: dict) -> str:
    """The covariate a proportional-hazards row is about."""
    return row["covariates"][0]


def _gap(delta: float) -> str:
    """Another model's AIC against the fitted model's, ``delta`` the
    fitted model's less the other's, in words."""
    if delta == 0:
        return "level with it"
    return "{} {}".format(_fmt(abs(delta)), "lower" if delta > 0 else "higher")


def _is_cox(model: Any) -> bool:
    """Whether ``model`` is a semi-parametric (Cox-type) regression."""
    from surpyval.univariate.regression.semi_parametric_regression_model import (  # noqa: E501
        SemiParametricRegressionModel,
    )

    return isinstance(model, SemiParametricRegressionModel)


class _Fit(SimpleNamespace):
    """A maximum found by a family's ``_maximise``: ``theta``, ``value``
    (the negative log-likelihood there) and ``ok`` (a verified
    maximum)."""

    theta: np.ndarray
    value: float
    ok: bool


def _form_panel(
    name: str, z: np.ndarray, residual: np.ndarray, ylabel: str
) -> Panel:
    """The functional-form plot of covariate ``name``: each row's
    ``residual`` against its value ``z``, with a lowess trend and a line
    at 0 (Cox, frailty and parametric PH/AFT)."""
    from .smooth import lowess

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        grid = np.linspace(z.min(), z.max(), 60)
        ax.scatter(z, residual, s=10, color=INK["muted"], alpha=0.6)
        ax.plot(grid, lowess(z, residual, grid), color=PALETTE[0])
        ax.axhline(0.0, color=INK["axis"], lw=0.8)
        ax.set_xlabel(name)
        ax.set_ylabel(ylabel)

    return Panel(f"{name}: functional form", draw)


def _cox_snell_panel(
    r: np.ndarray, c: np.ndarray, n: np.ndarray, xlabel: str, title: str
) -> Panel:
    """The overall-fit plot: the Nelson-Aalen cumulative hazard of the
    Cox-Snell residuals ``r`` (censoring ``c``, counts ``n``) against the
    line of slope 1 they follow when the model fits."""
    order = np.argsort(r, kind="stable")
    r, c, n = r[order], c[order], n[order]
    uniq, first = np.unique(r, return_index=True)
    at_risk = n[::-1].cumsum()[::-1][first]
    deaths = np.bincount(
        np.searchsorted(uniq, r), weights=n * (c == 0), minlength=uniq.size
    )
    H = np.cumsum(deaths / at_risk)
    keep = deaths > 0

    def draw(ax: Any, options: Any) -> None:
        from surpyval.plotting import INK, PALETTE

        ax.step(uniq[keep], H[keep], where="post", color=PALETTE[0])
        top = float(max(uniq[keep].max(), H[keep].max())) if keep.any() else 1
        ax.plot([0, top], [0, top], color=INK["secondary"], ls="--", lw=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("cumulative hazard")

    return Panel(title, draw)
