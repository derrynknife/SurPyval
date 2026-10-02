"""Covariate handling: DataFrame columns, formulas and covariate rows."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

from surpyval.utils.warnings import caller_stacklevel

if TYPE_CHECKING:
    from pandas import DataFrame

# pandas and formulaic are imported where they are used, not here: every
# model imports this module, and they took half of ``import surpyval``
# (#470).


def optional_column(df: DataFrame, name: "str | None") -> "npt.NDArray | None":
    """The named DataFrame column as an array, or ``None`` when no
    column is named -- for ``fit_from_df`` methods whose optional
    arguments (censoring, counts, truncation, ...) may not be given."""
    return None if name is None else df[name].to_numpy()


def check_covariate_rows(Z: npt.ArrayLike, n_rows: int) -> None:
    """Refuse a covariate array whose row count does not match the data.

    Every regression fitter pairs covariate row ``i`` with observation
    ``i``; a mismatch used to surface as a bare ``IndexError`` from deep
    inside a boolean mask, which says nothing about the cause.
    """
    Z_rows = np.shape(Z)[0] if np.ndim(Z) > 0 else 1
    if Z_rows != n_rows:
        raise ValueError(
            "Z has {} row(s) but there are {} observations; give one "
            "covariate row per observation.".format(Z_rows, n_rows)
        )


def finite_covariate_mask(Z: npt.ArrayLike) -> npt.NDArray:
    """Mask of the rows of ``Z`` whose covariates are all finite.

    A NaN or infinite covariate has no place in any regression likelihood:
    it makes the objective nan, and an optimiser that sees nan at every
    point returns its starting values as though they were a fit. Every
    regression fitter therefore drops such rows -- and says how many, so
    the loss of data is never silent. Raises if no row is left.
    """
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.ndim == 0:
        Z_arr = Z_arr.reshape(1, 1)
    finite = np.isfinite(Z_arr.reshape(Z_arr.shape[0], -1)).all(axis=1)
    dropped = int((~finite).sum())
    if dropped:
        if dropped == finite.shape[0]:
            raise ValueError(
                "Every row has a missing (NaN) or infinite covariate value; "
                "there is nothing to fit."
            )
        warnings.warn(
            "Dropped {} of {} rows with a missing (NaN) or infinite "
            "covariate value.".format(dropped, finite.shape[0]),
            UserWarning,
            stacklevel=caller_stacklevel(),
        )
    return finite


def formula_model_matrix(source: Any, df: Any, **kwargs: Any) -> Any:
    """Materialise a formula (or a fitted ``ModelSpec``) against ``df``
    with one row per row of ``df``.

    ``formulaic`` drops rows with a missing value by default, which
    misaligns the matrix with every other column taken from ``df`` (the
    times, censoring flags, counts). Its ``na_action="ignore"`` keeps the
    rows but silently codes a missing *categorical* as the reference
    level. So the rows are dropped as usual and then put back as all-nan
    rows, which callers either drop (with a warning) when fitting or turn
    into nan predictions in place.

    A categorical value outside a term's levels (a level the fitted spec
    never saw, or one missing from ``C(g, levels=[...])``) raises a
    ``ValueError`` naming the column and the levels: ``formulaic`` codes
    it as the reference level, with only a ``DataMismatchWarning`` (#371).
    So does, with a fitted spec, a level that had no rows in the fitted
    data; fitting a formula with such a level warns and records it (#377).
    """
    from formulaic import Formula
    from formulaic.errors import (  # type: ignore[import-untyped]
        DataMismatchWarning,
    )

    from surpyval.univariate.regression.regression_data import (
        record_empty_levels,
        refuse_empty_levels,
        unseen_levels_error,
    )

    positional = df.reset_index(drop=True)

    def materialise() -> Any:
        if isinstance(source, str):
            return Formula(source).get_model_matrix(positional, **kwargs)
        return source.get_model_matrix(positional, **kwargs)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DataMismatchWarning)
            model_matrix = materialise()
    except DataMismatchWarning as mismatch:
        spec = None if isinstance(source, str) else source
        if spec is None:
            # A formula's levels (``C(g, levels=[...])``) are known only
            # once it is materialised: do so again, allowing the mismatch.
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    spec = materialise().model_spec
            except Exception:
                pass
        raise unseen_levels_error(spec, positional, str(mismatch)) from None
    spec = model_matrix.model_spec
    if isinstance(source, str):
        # The rows kept (no missing value) are those the model is fitted to.
        record_empty_levels(spec, positional.loc[model_matrix.index])
    else:
        refuse_empty_levels(source, positional)
    if len(model_matrix) != len(positional):
        model_matrix = model_matrix.reindex(range(len(positional)))
    return model_matrix, spec


def numeric_columns(df: Any, cols: "list[str]") -> npt.NDArray:
    """``df[cols]`` as a float array. A column that is not numeric (a
    ``"yes"`` / ``"no"`` column, say) raises a ``ValueError`` that names
    it and points to ``formula=``, which codes categorical columns; it
    was numpy's bare "could not convert string to float" (#485)."""
    try:
        return np.asarray(df[cols].values, dtype=float)
    except (ValueError, TypeError):
        bad = []
        for col in cols:
            try:
                np.asarray(df[col].values, dtype=float)
            except (ValueError, TypeError):
                bad.append(col)
        raise ValueError(
            "Covariate column(s) {} are not numeric. Encode them as "
            "numbers, or pass `formula=` instead of `Z_cols` (e.g. "
            "formula={!r}), which codes a categorical column for "
            "you.".format(bad, " + ".join(str(c) for c in cols))
        ) from None


def wrangle_and_check_form_and_Z_cols(
    Z_cols: "str | list[str] | None",
    formula: "str | None",
    df: Any,
) -> tuple:
    if (Z_cols is None) and (formula is None):
        raise ValueError("'Z_cols' or 'formula' cannot both be None")

    if (Z_cols is not None) and (formula is not None):
        raise ValueError(
            "Either 'Z_cols' or 'formula' must be provided; not both"
        )

    if Z_cols is not None:
        if isinstance(Z_cols, str):
            Z_cols = [Z_cols]
        unknown = [x for x in Z_cols if x not in df.columns]
        if len(unknown) > 0:
            raise ValueError("{} not in dataframe columns".format(unknown))
        Z = numeric_columns(df, list(Z_cols))
        form = None
        feature_names = list(Z_cols)
        model_spec = None
    else:
        # Materialise with the implicit intercept so categoricals get
        # reference-level coding, then drop the intercept column — the
        # baseline hazard plays that role, and a full one-hot is collinear
        # with it (#252). An explicit "0 + ..." formula opts out. The
        # model keeps the formula as given, a str, as every fitter does.
        form = formula
        model_matrix, model_spec = formula_model_matrix(formula, df)
        if "Intercept" in model_matrix.columns:
            model_matrix = model_matrix.drop(columns=["Intercept"])
        feature_names = list(model_matrix.columns)
        Z = model_matrix.values.astype(float)

    # The same row mask for both branches, applied here to Z and returned so
    # the caller applies it to the times, flags and counts too. The formula
    # branch used to build the mask but never apply it to Z, so any missing
    # value made Z one row short of x.
    mask = finite_covariate_mask(Z)
    return Z[mask], mask, form, feature_names, model_spec
