"""
Helpers for fitting and predicting parametric regression models directly
from pandas DataFrames.

These utilities let a user fit a regression model by naming the columns of a
DataFrame (or by providing a ``formula``) so that the names of the covariates
are retained on the fitted model. The same metadata is then used at prediction
time so that a DataFrame can be passed to ``sf``, ``ff``, ``df``, ``hf``,
``Hf`` and ``random`` and the correct columns will be selected automatically.
"""

from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import numpy.typing as npt
import pandas as pd
from formulaic import Formula

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
        one-hot encoding would be exactly collinear with it (#252). Pass an
        explicit ``"0 + ..."`` to opt out and keep full-rank coding.

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
        model_matrix = Formula(formula).get_model_matrix(df)
        model_spec = model_matrix.model_spec
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

    Z = df[Z_cols].values.astype(float)
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
    for prediction. Any other input is returned unchanged so that the existing
    array based interface keeps working.

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
    Z : numpy.ndarray or array_like
        A numeric design matrix when ``Z`` was a DataFrame, otherwise ``Z``
        unchanged.
    """
    if not isinstance(Z, pd.DataFrame):
        return np.asarray(Z)

    if model_spec is not None:
        model_matrix = drop_intercept(model_spec.get_model_matrix(Z))
        return np.asarray(model_matrix, dtype=float)

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


def model_spec_to_meta(model_spec: Any) -> dict:
    """
    Capture the JSON-safe state needed to rebuild a ``formulaic`` model spec.

    A model fit with a ``formula`` carries a ``formulaic`` ``ModelSpec`` that
    knows how to expand raw covariates into the fitted design matrix -- in
    particular the levels of any categorical factor (``sex`` -> ``sex[F]``,
    ``sex[M]``). That spec is not itself JSON-serialisable, so this extracts
    the minimum needed to regenerate an equivalent one on load: the categorical
    factor levels and the names of the numeric covariate columns. The formula
    string is stored separately by the caller.

    Data-dependent (stateful) transforms such as ``scale(x)`` or ``center(x)``
    keep fitted statistics in the spec's ``transform_state`` that cannot be
    recovered from levels alone; a formula using one raises
    ``NotImplementedError`` rather than round-tripping to a silently wrong
    encoding.
    """
    if getattr(model_spec, "transform_state", None):
        raise NotImplementedError(
            "Serialising a formula that uses a data-dependent transform "
            "(e.g. scale() or center()) is not supported: its fitted "
            "statistics cannot be restored. Refit with the covariate entered "
            "directly (the baseline distribution absorbs location and scale), "
            "or with a covariate list instead of a formula."
        )

    value_vars = {
        str(v)
        for v in model_spec.variables
        if any(str(r).endswith("VALUE") for r in v.roles)
    }

    factor_levels: dict[str, list] = {}
    for factor, (_kind, state) in model_spec.encoder_state.items():
        if "categories" not in state:
            continue
        factor = str(factor)
        if factor not in value_vars:
            # A wrapped categorical such as ``C(sex)`` does not name a bare
            # column, so the template below cannot type it. These are rare;
            # fall back to Wald-free refitting rather than guess.
            raise NotImplementedError(
                "Serialising a wrapped categorical term "
                f"('{factor}') is not yet supported; enter the column "
                "directly (surpyval treats string / object columns as "
                "categorical automatically)."
            )
        factor_levels[factor] = [str(c) for c in state["categories"]]

    numeric_features = sorted(value_vars - set(factor_levels))
    return {
        "factor_levels": factor_levels,
        "numeric_features": numeric_features,
    }


def rebuild_model_spec(formula: str, meta: dict) -> Any:
    """
    Reconstruct a ``formulaic`` model spec from a formula and stored metadata.

    A small template DataFrame is built with each categorical column typed to
    the stored levels and each numeric column a placeholder, then the same
    formula (with its implicit intercept, matching fit time — #252) is
    re-materialised against it. ``formulaic`` derives an encoder state
    identical to fit time (the encoding depends on the formula and the factor
    levels, not the row values), so the returned spec expands raw covariates
    exactly as the original did.
    """
    formula = str(formula)
    factor_levels = meta.get("factor_levels", {})
    numeric_features = meta.get("numeric_features", [])

    height = max((len(lv) for lv in factor_levels.values()), default=1)
    template: dict[str, Any] = {}
    for col, levels in factor_levels.items():
        reps = (height + len(levels) - 1) // len(levels)
        template[col] = pd.Categorical(
            (list(levels) * reps)[:height], categories=list(levels)
        )
    for col in numeric_features:
        # 1.0 (not 0.0) keeps log / reciprocal terms finite while the spec is
        # derived; only the encoder structure is kept, not these values.
        template[col] = np.ones(height)

    model_matrix = Formula(formula).get_model_matrix(pd.DataFrame(template))
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

        model = self.fit(x, Z, c=c, n=n, t=t, init=init, fixed=fixed)

        model.feature_names = feature_names
        model.formula = formula
        model._model_spec = model_spec

        return model
