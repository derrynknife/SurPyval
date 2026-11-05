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

from .registry import Context

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
