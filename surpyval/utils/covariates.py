"""Covariate handling: DataFrame columns, formulas and covariate rows."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Sequence
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


#: The covariate range above which a coefficient is searched and judged in
#: units of ``1 / range`` (:func:`coefficient_floor`, #612). Below it (and
#: from 1 up) the unit is 1, as it was, so ordinary data fits as before.
LARGE_RANGE = 100.0


def coefficient_floor(
    n_search: int,
    coefs: "list[tuple[int, int]]",
    Z: "npt.ArrayLike | None",
) -> npt.NDArray:
    """Per-component ``floor`` of a regression search vector of
    ``n_search`` components, for ``preconditioned_bfgs`` and
    ``is_local_minimum``: each covariate coefficient's natural unit, the
    change that moves the linear predictor by 1 across its covariate's
    observed range, ``1 / range(Z_j)``, where that range is below 1 or
    above :data:`LARGE_RANGE` (100); 1 for a covariate whose range is in
    between, and for every other component. ``coefs`` are
    ``(position, column)`` pairs: a coefficient's position in the search
    vector and its covariate's column of ``Z`` (as ``free_coefficients``
    in ``univariate/regression/_fit_skeleton.py`` gives them).

    Both the search and the verification measure a component in units of
    ``max(|x|, floor)``. A coefficient starts at 0, where the floor alone
    sets its unit, and with a floor of 1 the unit depended on the
    covariate's: the gradient in a coefficient is proportional to its
    covariate's spread, so for a covariate spanning 3e-4 (an Arrhenius
    ``1/T`` in kelvin) it was below both BFGS's tolerance and the
    verification's at the start. A WeibullPH fit stopped there after no
    iterations, its coefficient exactly 0, and reported a verified
    maximum 0.41 below the maximum it reached with ``1000/T`` (#577). In
    the coefficient's natural unit the gradient is the same whatever the
    covariate's units.

    The same holds from above: a covariate in units of 1e4 (a date in
    days, an income) has a coefficient of order 1e-4, which a search in
    units of 1 takes in steps ten thousand times its size, and 14 of the
    conformance fits stopped up to 0.023 short of their maximum, saying
    so (#612). Its floor is ``1 / range`` too. The floor stays 1 for a
    covariate whose range is from 1 to 100, so that nothing changes for
    a binary covariate or ordinary data, as ``search_floor`` keeps the
    univariate fits' floor of 1 for data of order 1 and up.

    Examples
    --------
    Two distribution parameters, then the coefficients of a binary
    covariate, of a temperature's reciprocal in kelvin and of a date in
    days:

    >>> import numpy as np
    >>> from surpyval.utils.covariates import coefficient_floor
    >>> kelvin = np.array([358.15, 378.15, 398.15])
    >>> days = np.array([18000.0, 19500.0, 21000.0])
    >>> Z = np.column_stack([[0, 1, 1], 1 / kelvin, days])
    >>> floor = coefficient_floor(5, [(2, 0), (3, 1), (4, 2)], Z)
    >>> [float(f"{v:.4g}") for v in floor]
    [1.0, 1.0, 1.0, 3565.0, 0.0003333]
    """
    floor = np.ones(n_search)
    if Z is None:
        return floor
    Z_arr = np.asarray(Z, dtype=float)
    if Z_arr.size == 0:
        return floor
    Z_arr = Z_arr.reshape(Z_arr.shape[0], -1)
    spread = np.max(Z_arr, axis=0) - np.min(Z_arr, axis=0)
    for pos, j in coefs:
        if j < spread.size and (
            0.0 < spread[j] < 1.0 or spread[j] > LARGE_RANGE
        ):
            floor[pos] = 1.0 / spread[j]
    return floor


def coefficient_names(
    n: int,
    columns: "Sequence[Any] | None" = None,
    taken: "Iterable[str]" = (),
) -> "list[str]":
    """The names of a regression model's ``n`` covariate coefficients
    (#614): each covariate's column name where the fit has them (a
    formula, ``fit_from_df`` or a DataFrame ``Z``), else ``coef_0``,
    ``coef_1``, ...

    A name already in use -- by a parameter of the model in ``taken``
    (a baseline's ``alpha``, a frailty's ``theta``) or by an earlier
    coefficient (two columns of one name) -- gets the first free suffix
    ``.1``, ``.2``, ..., as R's ``make.unique`` and pandas name repeated
    columns: a column ``alpha`` of a Weibull model is ``alpha.1``.

    Until v0.23 the coefficients were ``beta_0``, ``beta_1``, ..., which
    sat next to the Weibull's shape ``beta``; a dictionary saved with
    those names loads with these (:func:`loaded_coefficient_names`).

    Examples
    --------
    >>> from surpyval.utils.covariates import coefficient_names
    >>> coefficient_names(2)
    ['coef_0', 'coef_1']
    >>> coefficient_names(3, ["age", "alpha", "age"], taken=["alpha", "beta"])
    ['age', 'alpha.1', 'age.1']
    """
    if columns is None or len(columns) != n:
        base = ["coef_{}".format(j) for j in range(n)]
    else:
        base = [str(column) for column in columns]
    used = set(taken)
    out = []
    for name in base:
        unique, k = name, 0
        while unique in used:
            k += 1
            unique = "{}.{}".format(name, k)
        used.add(unique)
        out.append(unique)
    return out


def loaded_coefficient_names(
    names: "Sequence[str]",
    first: int,
    count: int,
    columns: "Sequence[Any] | None",
) -> "list[str]":
    """``names``, a saved model's parameter names, whose ``count``
    coefficients start at ``first``, with the coefficients renamed as
    :func:`coefficient_names` names them (``columns``, the model's
    ``feature_names``, or ``coef_j``; the other names taken) where they
    are a dictionary's names before v0.23 (``beta_0``, ``beta_1``, ... in
    order); unchanged otherwise. A dictionary saved before #614 loads
    with the names the model has now.

    Examples
    --------
    >>> from surpyval.utils.covariates import loaded_coefficient_names
    >>> loaded_coefficient_names(["alpha", "beta", "beta_0", "theta"],
    ...                          2, 1, None)
    ['alpha', 'beta', 'coef_0', 'theta']
    """
    names = list(names)
    coefs = names[first : first + count]
    if coefs != ["beta_{}".format(j) for j in range(count)] or not count:
        return names
    others = names[:first] + names[first + count :]
    new = coefficient_names(count, columns, others)
    return names[:first] + new + names[first + count :]
