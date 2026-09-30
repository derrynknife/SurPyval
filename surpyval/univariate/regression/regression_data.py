"""
Helpers for fitting and predicting parametric regression models directly
from pandas DataFrames.

These utilities let a user fit a regression model by naming the columns of a
DataFrame (or by providing a ``formula``) so that the names of the covariates
are retained on the fitted model. The same metadata is then used at prediction
time so that a DataFrame can be passed to ``sf``, ``ff``, ``df``, ``hf``,
``Hf`` and ``random`` and the correct columns will be selected automatically.
"""

from __future__ import annotations

import inspect
import re
import warnings
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from formulaic import Formula, ModelSpec
from formulaic.parser.types import Factor  # type: ignore[import-untyped]

from surpyval.utils import (
    _caller_stacklevel,
    formula_model_matrix,
    numeric_columns,
    refuse_time_values,
)

from ._aliasing import covariate_columns

if TYPE_CHECKING:
    from .parametric_regression_model import ParametricRegressionModel


def drop_intercept(model_matrix: Any) -> Any:
    """Drop the intercept column from a materialised model matrix.

    Formulas are materialised *with* their implicit intercept so that
    ``formulaic`` gives categorical terms reference-level (reduced-rank)
    coding — with no intercept it emits a full one-hot whose columns sum to
    a constant, which is exactly collinear with the baseline distribution's
    scale (or the Cox baseline), leaving the coefficients non-identified
    (#252). The intercept column itself is then removed because the baseline
    plays that role.
    """
    if "Intercept" in model_matrix.columns:
        return model_matrix.drop(columns=["Intercept"])
    return model_matrix


def check_finite_event_times(x: npt.ArrayLike, c: npt.ArrayLike) -> None:
    """Refuse an exactly observed (``c == 0``) time that is not finite.

    A failure at infinity is not an observation: the semi-parametric
    fitters used to take it as an event time (``CoxPH`` then returned a
    coefficient of 19.4 on two rows, #394). A right-censored infinite
    time -- a unit that never failed -- is accepted, as elsewhere.
    """
    x_arr = np.asarray(x, dtype=float)
    event = np.asarray(c) == 0
    exact = x_arr[event] if x_arr.ndim == 1 else x_arr[event].ravel()
    if not np.isfinite(exact).all():
        raise ValueError(
            "Exactly observed values (c=0) must be finite; an item that "
            "had not failed by the end of observation is right censored "
            "(c=1)."
        )


def design_matrix_from_df(
    df: pd.DataFrame,
    Z_cols: str | list[str] | None = None,
    formula: str | None = None,
) -> tuple[npt.NDArray, list[str], Any]:
    """
    Build a covariate design matrix ``Z`` from a pandas DataFrame.

    Exactly one of ``Z_cols`` or ``formula`` must be provided.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the covariate columns.
    Z_cols : str or list of str, optional
        The column name(s) of the covariates to use.
    formula : str, optional
        A ``formulaic`` formula describing the design matrix, e.g.
        ``"age + sex + age:sex"``. The formula is materialised with its
        implicit intercept so categoricals get reference-level
        (reduced-rank) coding, and the intercept column is then dropped —
        the baseline distribution provides the intercept, and a full
        one-hot encoding would be exactly collinear with it (#252). An
        explicit ``"0 + ..."`` opts out and keeps every level's column;
        with the baseline as the intercept, the fit then aliases the last
        level (#476).

    Returns
    -------
    Z : numpy.ndarray
        The two dimensional design matrix.
    feature_names : list of str
        The names of the columns of ``Z``.
    model_spec : formulaic.ModelSpec or None
        The fitted ``formulaic`` model specification when a ``formula`` was
        used. This is retained so that the exact same encoding (including
        categorical factor levels) can be reproduced at prediction time.
        ``None`` when ``Z_cols`` was used.
    """
    if (Z_cols is None) and (formula is None):
        raise ValueError("One of 'Z_cols' or 'formula' must be provided")

    if (Z_cols is not None) and (formula is not None):
        raise ValueError(
            "Either 'Z_cols' or 'formula' must be provided; not both"
        )

    if formula is not None:
        # One row per row of ``df`` (all-nan where a formula column is
        # missing): formulaic's default drops such rows, which left ``Z``
        # shorter than the times taken from the same frame. The fitter then
        # drops the nan rows from every array together, with a warning.
        model_matrix, model_spec = formula_model_matrix(formula, df)
        model_matrix = drop_intercept(model_matrix)
        feature_names = list(model_matrix.columns)
        Z = np.asarray(model_matrix, dtype=float)
        return Z, feature_names, model_spec

    # Exactly one of Z_cols / formula is provided (validated above), so at
    # this point (formula is None) Z_cols must be set.
    assert Z_cols is not None
    if isinstance(Z_cols, str):
        Z_cols = [Z_cols]
    else:
        Z_cols = list(Z_cols)

    unknown = [c for c in Z_cols if c not in df.columns]
    if len(unknown) > 0:
        raise ValueError("{} not in dataframe columns".format(unknown))

    Z = numeric_columns(df, Z_cols)
    return Z, Z_cols, None


def prepare_Z(
    Z: "npt.ArrayLike | pd.DataFrame",
    feature_names: list[str] | None = None,
    model_spec: Any = None,
) -> npt.NDArray:
    """
    Convert a covariate input ``Z`` into a numeric design matrix.

    If ``Z`` is a pandas DataFrame, the columns are selected using the
    ``feature_names`` and/or ``model_spec`` that were stored when the model was
    fit from a DataFrame, ensuring the same covariates (and encoding) are used
    for prediction. Any other input is read as an array of floats, in the
    fitted column order.

    Parameters
    ----------
    Z : array_like or pandas.DataFrame
        The covariates to prepare.
    feature_names : list of str, optional
        The covariate column names recorded at fit time.
    model_spec : formulaic.ModelSpec, optional
        The formula model specification recorded at fit time.

    Returns
    -------
    Z : numpy.ndarray
        The numeric design matrix: from a DataFrame, the recorded columns
        (or the formula's expansion); otherwise ``Z`` as a float array, with
        ``None`` read as a missing value (``nan``).
    """
    if not isinstance(Z, pd.DataFrame):
        # As floats: a list (or object array) holding ``None`` is then a
        # missing value, predicted as nan, not a TypeError from ``Z @ beta``.
        return np.asarray(Z, dtype=float)

    if model_spec is not None:
        # A row with a missing value comes back as an all-nan row -- so its
        # prediction is nan, in place -- rather than being dropped, which
        # shifted every later prediction onto the wrong ``x``.
        Z = categorical_columns_as_objects(Z, model_spec)
        model_matrix, _ = formula_model_matrix(model_spec, Z)
        return np.asarray(drop_intercept(model_matrix), dtype=float)

    if feature_names is not None:
        unknown = [c for c in feature_names if c not in Z.columns]
        if len(unknown) > 0:
            raise ValueError("{} not in dataframe columns".format(unknown))
        return Z[feature_names].values.astype(float)

    raise ValueError(
        "A pandas DataFrame was passed as Z but the model was not fit with "
        "named covariates. Fit the model with 'fit_from_df' (or pass a numpy "
        "array) to predict from a DataFrame."
    )


def categorical_columns_as_objects(Z: pd.DataFrame, model_spec: Any) -> Any:
    """Recast the numeric columns of ``Z`` that the formula codes as
    categorical, so they are coded against the fitted levels.

    A column entered bare as a categorical factor -- a ``pd.Categorical``
    of integers at fit time, say -- but passed for prediction with a plain
    numeric (or boolean) dtype is read by ``formulaic`` as a *numerical*
    factor, and the fitted structure then broadcasts its raw value into
    every level's column (``k[T.2]`` and ``k[T.3]`` both equal to ``k``):
    a silently wrong design matrix. As ``object`` it is categorical again.
    """
    recast = {}
    for expr, (kind, _state) in model_spec.encoder_state.items():
        if kind is not Factor.Kind.CATEGORICAL or expr not in Z.columns:
            continue
        column = Z[expr]
        if isinstance(column.dtype, pd.CategoricalDtype):
            continue
        if pd.api.types.is_numeric_dtype(column):
            recast[expr] = column.astype(object)
    return Z.assign(**recast) if recast else Z


def unseen_levels_error(
    model_spec: Any, df: pd.DataFrame, detail: str
) -> ValueError:
    """The error for a categorical value outside a formula term's levels.

    ``formulaic`` codes a value that is not one of a categorical term's
    levels -- a level not seen when the model was fitted, or not in a
    ``C(g, levels=[...])`` list -- as the reference level (all-zero
    columns), with only a ``DataMismatchWarning``: a silently wrong design
    matrix (#371). :func:`surpyval.utils.formula_model_matrix` turns that
    warning into this error, which names each column and its unknown
    levels. A term that codes a single column as is (``g``, ``C(g)``,
    ``C(g, levels=...)``, ``C(g, contr.sum)``, ...) is checked here
    directly; for any other categorical term (``C(k + 1)``, say), or
    without a ``model_spec``, ``formulaic``'s own ``detail`` is quoted
    instead. Missing values are not levels: their rows come back as
    ``nan``. A declared level with no rows in the fitted data (see
    :func:`record_empty_levels`) is not a fitted level either.
    """
    found, unresolved = _unknown_levels(model_spec, df)
    if found:
        return _unknown_levels_error(found)
    terms = f" in the formula term(s) {unresolved}" if unresolved else ""
    return ValueError(f"Unknown categorical level(s){terms}: {detail}")


# The key under which the fitted state of a categorical formula term lists
# its levels that had no rows in the fitted data (#377). ``formulaic``
# reads only the state's "categories", so the key travels with the spec
# (and, through ``model_spec_to_meta``, with a saved model) unused by it.
EMPTY_LEVELS_KEY = "empty_levels"


def record_empty_levels(model_spec: Any, df: pd.DataFrame) -> None:
    """Warn about, and record on ``model_spec``, each categorical level
    with no rows in the fitted data ``df``.

    A level declared with ``C(g, levels=[...])`` (or an unused category of
    a ``pd.Categorical`` column) gets a column of the design matrix, and
    so a coefficient, even when no row of the data has it: nothing
    estimates that coefficient, and a prediction for the level returned a
    made-up number (#377). The fit goes ahead -- declaring the full list
    of levels keeps the coding the same across data splits -- with one
    ``UserWarning`` naming each column and its empty levels, which are
    stored under ``EMPTY_LEVELS_KEY`` in the term's encoder state, so that
    predicting for them raises as for a level the model was never fitted
    with (:func:`refuse_empty_levels`). ``df`` holds the rows the design
    matrix kept (those with no missing value). Terms that do not code a
    column as is (``C(k + 1)``, say) are not checked.
    """
    found = []
    for expr, (kind, state) in model_spec.encoder_state.items():
        if kind is not Factor.Kind.CATEGORICAL:
            continue
        column = _factor_column(model_spec, str(expr), df.columns)
        if column is None:
            continue
        values = df[column]
        present = set(pd.unique(values[~pd.isna(values)]))
        empty = [lv for lv in state.get("categories", []) if lv not in present]
        if empty:
            state[EMPTY_LEVELS_KEY] = empty
            found.append(
                f"column {column!r} has no rows at the level(s) "
                f"{[_native(v) for v in empty]} of the formula term "
                f"{str(expr)!r}"
            )
    if found:
        warnings.warn(
            "Categorical level(s) with no rows in the fitted data: "
            + "; ".join(found)
            + ". Their coefficients have nothing to be estimated from, so "
            "predicting for such a level raises a ValueError, as for a "
            "level the model was not fitted with.",
            UserWarning,
            stacklevel=_caller_stacklevel(),
        )


def refuse_empty_levels(model_spec: Any, df: pd.DataFrame) -> None:
    """Raise the unknown-level ``ValueError`` if a row of ``df`` has a
    level that had no rows in the fitted data (see
    :func:`record_empty_levels`); ``formulaic`` codes it without
    complaint, since the level was declared."""
    if not any(
        EMPTY_LEVELS_KEY in state
        for _, state in model_spec.encoder_state.values()
    ):
        return
    found, _ = _unknown_levels(model_spec, df)
    if found:
        raise _unknown_levels_error(found)


def _unknown_levels(
    model_spec: Any, df: pd.DataFrame
) -> tuple[list[str], list[str]]:
    """The description of each column of ``df`` with a value outside the
    fitted levels of its categorical term, and the categorical terms that
    cannot be checked by column."""
    found = []
    unresolved = []
    encoder_state = {} if model_spec is None else model_spec.encoder_state
    for expr, (kind, state) in encoder_state.items():
        if kind is not Factor.Kind.CATEGORICAL:
            continue
        column = _factor_column(model_spec, str(expr), df.columns)
        if column is None:
            unresolved.append(str(expr))
            continue
        values = df[column]
        empty = list(state.get(EMPTY_LEVELS_KEY, []))
        levels = [lv for lv in state.get("categories", []) if lv not in empty]
        unseen = set(pd.unique(values[~pd.isna(values)])).difference(levels)
        if unseen:
            unseen_list = sorted((_native(v) for v in unseen), key=str)
            text = (
                f"column {column!r} has the level(s) {unseen_list}, which "
                f"are not among the levels {[_native(v) for v in levels]} "
                f"of the formula term {str(expr)!r}"
            )
            declared = [v for v in unseen_list if v in empty]
            if declared:
                text += (
                    f" ({declared} declared, but with no rows in the fitted "
                    "data)"
                )
            found.append(text)
    return found, unresolved


def _unknown_levels_error(found: list[str]) -> ValueError:
    return ValueError(
        "Unknown categorical level(s): "
        + "; ".join(found)
        + ". The model has no coefficient for a level it was not fitted "
        "with; only the levels with rows in the fitted data can be used."
    )


def _factor_column(model_spec: Any, expr: str, columns: Any) -> str | None:
    """The DataFrame column a categorical term codes as is -- a bare
    column ``g`` or a ``C(g, ...)`` wrapper -- or ``None`` for any other
    term (whose values are computed from the data)."""
    for factor, variables in model_spec.factor_variables.items():
        if str(factor) != expr:
            continue
        data = [str(v) for v in variables if v.source == "data"]
        if len(data) != 1 or data[0] not in columns:
            return None
        column = data[0]
        wrapped = re.match(r"C\(\s*" + re.escape(column) + r"\s*[,)]", expr)
        return column if expr == column or wrapped else None
    return None


def _native(value: Any) -> Any:
    # ``np.str_('d')`` prints as "np.str_('d')" in a list since numpy 2.
    return value.item() if isinstance(value, np.generic) else value


def formula_to_string(formula: Any) -> str:
    """The text of a formula, in a form that parses back to the same terms.

    A formula given as text is returned unchanged. ``str`` of a parsed
    ``formulaic.Formula`` (which the Cox and competing-risks fitters keep)
    lists the intercept when there is one but says nothing when there is
    not, so ``"0 + z + g"`` came back as ``"z + g"`` -- which parses *with*
    an intercept, giving a different design matrix. The ``"0 + "`` is put
    back here.
    """
    if isinstance(formula, str):
        return formula
    text = str(formula)
    if all(str(term) != "1" for term in formula):
        text = "0 + " + text
    return text


def serialise_covariate_meta(model: Any, out: dict) -> None:
    """Store a fitted model's covariate metadata into its ``to_dict``.

    ``feature_names``, ``formula`` and -- when the model was fit from a
    formula -- the ``formula_meta`` needed to rebuild the design-matrix
    transformer on load, so a restored model expands raw covariates
    (e.g. categoricals) exactly as the original did (#244). Every
    DataFrame-fittable model class used to carry this block verbatim.
    """
    if model.feature_names is not None:
        out["feature_names"] = list(model.feature_names)
    if model.formula is not None:
        out["formula"] = formula_to_string(model.formula)
        if getattr(model, "_model_spec", None) is not None:
            out["formula_meta"] = model_spec_to_meta(model._model_spec)


def restore_covariate_meta(model: Any, model_dict: dict) -> None:
    """The ``from_dict`` counterpart of :func:`serialise_covariate_meta`:
    read back ``feature_names``/``formula`` and rebuild the formula's
    design-matrix transformer from the stored ``formula_meta`` (#244)."""
    model.feature_names = model_dict.get("feature_names")
    model.formula = model_dict.get("formula")
    formula_meta = model_dict.get("formula_meta")
    if model.formula is not None and formula_meta is not None:
        model._model_spec, model.formula = _rebuild_model_spec(
            model.formula, formula_meta, model.feature_names
        )


# Keys that mark an encoded value in a stored formula state (see
# ``_state_to_json``); a dictionary using one as a key is stored as items.
_STATE_TAGS = ("__ndarray__", "__tuple__", "__items__")


def _state_to_json(value: Any, expr: str) -> Any:
    """``value`` (a ``formulaic`` encoder or transform state) as strict
    JSON that :func:`_state_from_json` reads back to an equal value.

    Numbers, strings, booleans, ``None`` and lists are kept as they are;
    a numeric array, a tuple and a dictionary with a non-string key (e.g.
    ``poly``'s ``{0: ..., 1: ...}``) are tagged so they come back as the
    same type. Anything else cannot be stored faithfully and raises.
    """
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise _unserialisable_state(expr, value)
        return {"__ndarray__": value.tolist(), "dtype": value.dtype.name}
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, list):
        return [_state_to_json(v, expr) for v in value]
    if isinstance(value, tuple):
        return {"__tuple__": [_state_to_json(v, expr) for v in value]}
    if isinstance(value, dict):
        if all(isinstance(k, str) and k not in _STATE_TAGS for k in value):
            return {k: _state_to_json(v, expr) for k, v in value.items()}
        return {
            "__items__": [
                [_state_to_json(k, expr), _state_to_json(v, expr)]
                for k, v in value.items()
            ]
        }
    raise _unserialisable_state(expr, value)


def _state_from_json(value: Any) -> Any:
    """The inverse of :func:`_state_to_json`."""
    if isinstance(value, list):
        return [_state_from_json(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "__ndarray__" in value:
        return np.asarray(value["__ndarray__"], dtype=value["dtype"])
    if "__tuple__" in value:
        return tuple(_state_from_json(v) for v in value["__tuple__"])
    if "__items__" in value:
        return {
            _hashable(_state_from_json(k)): _state_from_json(v)
            for k, v in value["__items__"]
        }
    return {k: _state_from_json(v) for k, v in value.items()}


def _hashable(key: Any) -> Any:
    # A tuple key was stored as a tagged list; lists cannot be dict keys.
    return tuple(key) if isinstance(key, list) else key


def _unserialisable_state(expr: str, value: Any) -> NotImplementedError:
    return NotImplementedError(
        f"Serialising the formula term '{expr}' is not supported: its "
        f"fitted state holds a {type(value).__name__}, which cannot be "
        "stored as JSON. Refit with the covariate entered directly or "
        "with a covariate list instead of a formula."
    )


def model_spec_to_meta(model_spec: Any) -> dict:
    """
    Capture the JSON-safe state needed to rebuild a ``formulaic`` model spec.

    A model fit with a ``formula`` carries a ``formulaic`` ``ModelSpec`` that
    knows how to expand raw covariates into the fitted design matrix. That
    spec is not itself JSON-serialisable, so this extracts what the encoding
    depends on besides the formula string (stored separately by the
    caller):

    - ``encoder_state``: each factor's kind and, for a categorical factor
      -- a bare column or a wrapped term such as ``C(g, levels=...)`` or
      ``C(g, contr.sum)`` -- its levels, in order, with their JSON types
      (so integer levels stay integers), and those with no rows in the
      fitted data (``"empty_levels"``, see :func:`record_empty_levels`);
    - ``transform_state``: the fitted statistics of the data-dependent
      transforms (``scale``, ``center``, ``poly``, ``bs``, ``cs``, ...);
    - ``data_variables``: the DataFrame columns the formula reads.

    The formula re-applies the contrasts and any literal ``levels=``
    itself. When the formula uses only bare categorical columns with
    string levels, none of them empty, and no transform state, the
    ``factor_levels`` / ``numeric_features`` pair the v0.17 - v0.20
    readers expect is written too, so those releases restore it
    identically; any other formula
    leaves the pair out, so an older release fails to rebuild it rather
    than rebuilding a different encoding.

    The metadata is rebuilt (:func:`rebuild_model_spec`) before it is
    returned, so a formula that could not be restored -- a term whose state
    is not JSON-representable, or one that reads a name from outside the
    DataFrame and ``formulaic``'s transforms -- raises
    ``NotImplementedError`` here, at save time, rather than on load.
    """
    data_variables = sorted(
        str(v) for v in model_spec.variables if v.source == "data"
    )

    encoder_state = {}
    factor_levels: dict[str, list] | None = {}
    for factor, (kind, state) in model_spec.encoder_state.items():
        factor = str(factor)
        # ``contrasts`` is kept for introspection only; the formula
        # re-evaluates the contrasts from its own text.
        state = {k: v for k, v in state.items() if k != "contrasts"}
        encoder_state[factor] = {
            "kind": kind.value,
            "state": _state_to_json(state, factor),
        }
        if "categories" in state and factor_levels is not None:
            levels = encoder_state[factor]["state"]["categories"]
            # A v0.20 reader would not know a level with no fitted rows
            # (#377) and would predict for it, so it gets no pair.
            if (
                factor in data_variables
                and EMPTY_LEVELS_KEY not in state
                and all(isinstance(lv, str) for lv in levels)
            ):
                factor_levels[factor] = levels
            else:
                factor_levels = None

    transform_state = {
        str(expr): _state_to_json(state, str(expr))
        for expr, state in model_spec.transform_state.items()
    }

    meta: dict[str, Any] = {
        "data_variables": data_variables,
        "encoder_state": encoder_state,
        "transform_state": transform_state,
    }
    # Rebuild it now, so a formula that cannot be restored fails here, at
    # save time, rather than when the file is loaded.
    formula = formula_to_string(model_spec.formula)
    try:
        rebuilt = rebuild_model_spec(formula, meta)
    except Exception as err:
        raise NotImplementedError(
            f"Serialising the formula '{formula}' is not supported: its "
            f"design-matrix transformer cannot be rebuilt ({err})."
        ) from err
    if list(rebuilt.column_names) != list(model_spec.column_names):
        raise NotImplementedError(
            f"Serialising the formula '{formula}' is not supported: it "
            "rebuilds the columns {} instead of {}.".format(
                list(rebuilt.column_names), list(model_spec.column_names)
            )
        )
    if factor_levels is not None and not transform_state:
        meta["factor_levels"] = factor_levels
        meta["numeric_features"] = sorted(
            set(data_variables) - set(factor_levels)
        )
    return meta


def rebuild_model_spec(
    formula: str, meta: dict, feature_names: list[str] | None = None
) -> Any:
    """
    Reconstruct a ``formulaic`` model spec from a formula and stored metadata.

    The stored encoder and transform states (see :func:`model_spec_to_meta`)
    are attached to a fresh spec of the same formula, which is materialised
    once against a one-row template of missing values -- each categorical
    column typed to its stored levels -- so ``formulaic`` also derives the
    column structure (with the implicit intercept, matching fit time --
    #252). The states fix the levels, their order and every fitted
    transform statistic, so the returned spec expands raw covariates
    exactly as the original did. Metadata written by v0.17 - v0.20
    (``factor_levels`` / ``numeric_features`` only) is read too.

    With ``feature_names`` given, the rebuilt columns are checked against
    it, and a ``ValueError`` raised if they differ.
    """
    return _rebuild_model_spec(formula, meta, feature_names)[0]


def _rebuild_model_spec(
    formula: str, meta: dict, feature_names: list[str] | None
) -> tuple[Any, str]:
    """:func:`rebuild_model_spec`, also returning the formula text used
    (which differs only for a repaired v0.17 - v0.20 Cox formula)."""
    formula = str(formula)
    if "encoder_state" in meta:
        encoder_state = {
            expr: (
                Factor.Kind(entry["kind"]),
                _state_from_json(entry.get("state", {})),
            )
            for expr, entry in meta["encoder_state"].items()
        }
        transform_state = {
            expr: _state_from_json(state)
            for expr, state in meta.get("transform_state", {}).items()
        }
        data_variables = list(meta.get("data_variables", []))
    else:
        # The v0.17 - v0.20 layout: bare categorical columns (levels stored
        # as strings) and numeric columns only.
        factor_levels = meta.get("factor_levels", {})
        encoder_state = {
            col: (Factor.Kind.CATEGORICAL, {"categories": list(levels)})
            for col, levels in factor_levels.items()
        }
        transform_state = {}
        data_variables = sorted(
            set(factor_levels) | set(meta.get("numeric_features", []))
        )

    def rebuild(text: str) -> Any:
        return _materialise_spec(
            text, encoder_state, transform_state, data_variables
        )

    spec = rebuild(formula)
    if feature_names is None or _spec_columns(spec) == list(feature_names):
        return spec, formula
    if "encoder_state" not in meta and not formula.startswith("0 +"):
        # v0.17 - v0.20 Cox and competing-risks models stored ``str`` of
        # their parsed formula, which drops a "0 +" (see
        # ``formula_to_string``); the stored names tell the two apart.
        repaired = "0 + " + formula
        spec = rebuild(repaired)
        if _spec_columns(spec) == list(feature_names):
            return spec, repaired
    raise ValueError(
        "The stored formula {!r} rebuilds the design-matrix columns {} but "
        "the model was fit with {}; the serialised model is inconsistent."
        "".format(formula, _spec_columns(spec), list(feature_names))
    )


def _spec_columns(spec: Any) -> list[str]:
    return [c for c in spec.column_names if c != "Intercept"]


def _materialise_spec(
    formula: str,
    encoder_state: dict,
    transform_state: dict,
    data_variables: list[str],
) -> Any:
    """A spec of ``formula`` with the given states, materialised once so it
    also carries the column structure.

    The template has one row, all missing: ``formulaic`` evaluates every
    factor on it (so each gets its kind, and the structure follows) and
    then drops the row. Missing values keep bounded transforms (``bs``,
    ``log``, ...) from rejecting a placeholder, and no transform refits
    its state from them, as the stored state is used. A bare categorical
    column is typed with its levels so it is read as categorical.
    """
    template: dict[str, Any] = {v: [np.nan] for v in data_variables}
    for expr, (kind, state) in encoder_state.items():
        if expr in template and kind is Factor.Kind.CATEGORICAL:
            template[expr] = pd.Categorical(
                [np.nan], categories=list(state.get("categories", []))
            )
    spec = ModelSpec(
        formula=Formula(formula),
        encoder_state=encoder_state,
        transform_state=transform_state,
    )
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        model_matrix = spec.get_model_matrix(pd.DataFrame(template))
    return model_matrix.model_spec


class DataFrameRegressionMixin:
    """
    Mixin adding a ``fit_from_df`` method to a parametric regression fitter.

    The fitter must expose a ``fit(x, Z, c=None, n=None, t=None, init=None,
    fixed=None)`` method returning a ``ParametricRegressionModel``.
    """

    # Provided by the host fitter class this mixin is combined with.
    fit: Callable[..., "ParametricRegressionModel"]

    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        Z_cols: str | list[str] | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        formula: str | None = None,
        init: npt.ArrayLike | None = None,
        fixed: dict[str, float] | None = None,
        center: bool = False,
    ) -> "ParametricRegressionModel":
        """
        Fit the regression model using a pandas DataFrame as the input.

        The names of the covariates are retained on the fitted model so that a
        DataFrame can later be passed to the prediction methods (``sf``,
        ``ff``, ``df``, ``hf``, ``Hf``, ``random``) and the correct columns
        will be selected automatically.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe containing the data.
        x_col : str
            The column name of the observed times.
        Z_cols : str or list of str, optional
            The column name(s) of the covariates. Mutually exclusive with
            ``formula``.
        c_col : str, optional
            The column name of the censoring indicator.
        n_col : str, optional
            The column name of the number of observations at each time.
        tl_col : str, optional
            The column name of the left truncation values.
        tr_col : str, optional
            The column name of the right truncation values.
        formula : str, optional
            A ``formulaic`` formula describing the covariates, e.g.
            ``"age + sex"``. Mutually exclusive with ``Z_cols``.
        init : array_like, optional
            The initial values for the parameters.
        fixed : dict, optional
            A dictionary of parameters to fix to a specific value.
        center : bool, optional
            Report the baseline at the covariate means (stored as
            ``model.center``) instead of at ``Z = 0``; see ``fit``. Not
            available for an accelerated life model.

        Returns
        -------
        ParametricRegressionModel
            The fitted model, with ``feature_names`` (and ``formula``) set.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval import Weibull, WeibullPH
        >>> np.random.seed(1)
        >>> age = np.random.uniform(20, 60, 100)
        >>> weight = np.random.uniform(50, 100, 100)
        >>> time = Weibull.random(100, 10, 2) * np.exp(-0.02 * (age - 40))
        >>> df = pd.DataFrame({
        ...     "time": time,
        ...     "age": age,
        ...     "weight": weight,
        ...     "censored": np.zeros(100, dtype=int),
        ... })
        >>> model = WeibullPH.fit_from_df(
        ...     df, x_col="time", Z_cols=["age", "weight"], c_col="censored"
        ... )
        >>> model.feature_names
        ['age', 'weight']
        >>> model.sf([10, 20], df[["age", "weight"]].head(2)).round(4)
        array([0.4757, 0.0024])
        """
        Z, feature_names, model_spec = design_matrix_from_df(
            df, Z_cols, formula
        )

        x = df[x_col].values

        c = None if c_col is None else df[c_col].values
        n = None if n_col is None else df[n_col].values

        if (tl_col is None) and (tr_col is None):
            t = None
        else:
            n_rows = len(df)
            for name, col in (("tl", tl_col), ("tr", tr_col)):
                if col is not None:
                    refuse_time_values(df[col], name)  # (#480)
            tl = (
                np.full(n_rows, -np.inf)
                if tl_col is None
                else df[tl_col].values.astype(float)
            )
            tr = (
                np.full(n_rows, np.inf)
                if tr_col is None
                else df[tr_col].values.astype(float)
            )
            t = np.column_stack([tl, tr])

        # Passed only when asked for: the accelerated life fitter, which
        # shares this method, has no ``center`` (#463).
        extra: dict = {}
        if center:
            if "center" not in inspect.signature(self.fit).parameters:
                raise ValueError(
                    "center=True is not available for this model: its "
                    "covariates enter through a life model of the stress, "
                    "not a linear predictor with an origin to move."
                )
            extra["center"] = True
        with covariate_columns(feature_names, Z, model_spec):
            model = self.fit(
                x, Z, c=c, n=n, t=t, init=init, fixed=fixed, **extra
            )

        model.feature_names = feature_names
        model.formula = formula
        model._model_spec = model_spec

        return model
