"""The coefficient table of a regression model: ``summary()`` and the
table its ``repr`` prints (#484), with the columns of lifelines'
``summary`` and R's ``summary(coxph)``."""

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import norm

from surpyval.utils.covariates import (
    coefficient_names as names_of_coefficients,
)

#: The columns of the coefficient table, and the ones ``repr`` prints
#: (with shorter names for the interval).
_REPR_COLUMNS = {
    "coef": "coef",
    "exp(coef)": "exp(coef)",
    "se(coef)": "se(coef)",
    "coef lower {}": "lower {}",
    "coef upper {}": "upper {}",
    "z": "z",
    "p": "p",
}


def _level(alpha_ci: float) -> str:
    return "{:g}%".format(100 * (1 - alpha_ci))


def coefficient_table(
    names: "list[str]",
    coef: npt.ArrayLike,
    se: npt.ArrayLike,
    alpha_ci: float = 0.05,
    exp: bool = True,
    p: "npt.ArrayLike | None" = None,
) -> pd.DataFrame:
    """The coefficients ``coef`` (named ``names``) with their standard
    errors ``se``: ``exp(coef)`` (``nan`` where ``exp`` is False, a link
    whose exponent is not a ratio), a two-sided ``1 - alpha_ci`` Wald
    interval for the coefficient and for its exponent, and the Wald
    statistic ``z`` and its two-sided p-value (``p`` where the fit
    computed its own). An aliased coefficient (``nan``, #476) is ``nan``
    throughout."""
    coef = np.asarray(coef, dtype=float)
    se = np.asarray(se, dtype=float)
    q = norm.ppf(1 - alpha_ci / 2)
    with np.errstate(all="ignore"):
        z = coef / se
        if p is None:
            p = 2 * norm.sf(np.abs(z))
        lower, upper = coef - q * se, coef + q * se
        nan = np.full(coef.shape, np.nan)
        level = _level(alpha_ci)
        table = {
            "coef": coef,
            "exp(coef)": np.exp(coef) if exp else nan,
            "se(coef)": se,
            "coef lower " + level: lower,
            "coef upper " + level: upper,
            "exp(coef) lower " + level: np.exp(lower) if exp else nan,
            "exp(coef) upper " + level: np.exp(upper) if exp else nan,
            "z": z,
            "p": np.asarray(p, dtype=float),
        }
    out = pd.DataFrame(table, index=pd.Index(list(names), name="covariate"))
    return out


def format_table(table: pd.DataFrame, columns: "list[str]") -> str:
    """``table``'s ``columns`` as text, four significant figures, each line
    indented; a column that is all ``nan`` is left out."""
    shown = table[[c for c in columns if not table[c].isna().all()]]
    text = shown.to_string(
        float_format=lambda v: "{:.4g}".format(v), na_rep="nan"
    )
    return "\n".join("    " + line for line in text.splitlines())


def coefficient_repr(table: pd.DataFrame, alpha_ci: float = 0.05) -> str:
    """The coefficient table as ``repr`` prints it."""
    level = _level(alpha_ci)
    renamed = table.rename(
        columns={
            k.format(level): v.format(level) for k, v in _REPR_COLUMNS.items()
        }
    )
    renamed.index.name = None
    return format_table(
        renamed, [v.format(level) for v in _REPR_COLUMNS.values()]
    )


def coefficient_names(model: Any, n: int) -> "list[str]":
    """The names of a model's ``n`` coefficients (#614): its covariates'
    columns, ``feature_names`` (a formula, ``fit_from_df`` or a DataFrame
    ``Z``), where it has them, else ``coef_0``, ``coef_1``, ... (see
    :func:`surpyval.utils.covariates.coefficient_names`)."""
    return names_of_coefficients(n, getattr(model, "feature_names", None))
