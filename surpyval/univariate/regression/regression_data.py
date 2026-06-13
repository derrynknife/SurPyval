"""
Helpers for fitting and predicting parametric regression models directly
from pandas DataFrames.

These utilities let a user fit a regression model by naming the columns of a
DataFrame (or by providing a ``formula``) so that the names of the covariates
are retained on the fitted model. The same metadata is then used at prediction
time so that a DataFrame can be passed to ``sf``, ``ff``, ``df``, ``hf``,
``Hf`` and ``random`` and the correct columns will be selected automatically.
"""

import numpy as np
import pandas as pd
from formulaic import Formula


def design_matrix_from_df(df, Z_cols=None, formula=None):
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
        ``"age + sex + age:sex"``. An intercept is never added (the baseline
        distribution provides the intercept), so the formula is parsed as
        ``"0 + " + formula``.

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
        model_matrix = Formula("0 + " + formula).get_model_matrix(df)
        feature_names = list(model_matrix.columns)
        model_spec = model_matrix.model_spec
        Z = np.asarray(model_matrix, dtype=float)
        return Z, feature_names, model_spec

    if isinstance(Z_cols, str):
        Z_cols = [Z_cols]
    else:
        Z_cols = list(Z_cols)

    unknown = [c for c in Z_cols if c not in df.columns]
    if len(unknown) > 0:
        raise ValueError("{} not in dataframe columns".format(unknown))

    Z = df[Z_cols].values.astype(float)
    return Z, Z_cols, None


def prepare_Z(Z, feature_names=None, model_spec=None):
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
        return Z

    if model_spec is not None:
        return np.asarray(model_spec.get_model_matrix(Z), dtype=float)

    if feature_names is not None:
        unknown = [c for c in feature_names if c not in Z.columns]
        if len(unknown) > 0:
            raise ValueError(
                "{} not in dataframe columns".format(unknown)
            )
        return Z[feature_names].values.astype(float)

    raise ValueError(
        "A pandas DataFrame was passed as Z but the model was not fit with "
        "named covariates. Fit the model with 'fit_from_df' (or pass a numpy "
        "array) to predict from a DataFrame."
    )


class DataFrameRegressionMixin:
    """
    Mixin adding a ``fit_from_df`` method to a parametric regression fitter.

    The fitter must expose a ``fit(x, Z, c=None, n=None, t=None, init=None,
    fixed=None)`` method returning a ``ParametricRegressionModel``.
    """

    def fit_from_df(
        self,
        df,
        x_col,
        Z_cols=None,
        c_col=None,
        n_col=None,
        tl_col=None,
        tr_col=None,
        formula=None,
        init=None,
        fixed=None,
    ):
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
        >>> from surpyval import WeibullPH
        >>> model = WeibullPH.fit_from_df(
        ...     df, x_col="time", Z_cols=["age", "weight"], c_col="censored"
        ... )
        >>> model.sf([10, 20], df[["age", "weight"]])
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
