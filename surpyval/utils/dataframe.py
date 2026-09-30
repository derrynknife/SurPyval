"""``fit_from_df`` for every fitter (#511).

Each family's ``fit_from_df`` names the columns of a
:class:`pandas.DataFrame` in place of the arrays its ``fit`` takes, and
passes every other keyword on to ``fit``, so the two entry points give the
same model for the same data (principle 14). The argument names follow the
family's existing ``fit_from_df``:

- one lifetime per row (the parametric distributions, the non-parametric
  estimators, ``RoystonParmar``, ``MixtureModel``) and the copulas: the
  column is named by the ``fit`` argument it fills (``x``, ``c``, ``n``,
  ``xl``, ``xr``, ``tl``, ``tr``), as ``Weibull.fit_from_df`` always has;
- the recurrent-event fitters, the regressions and the competing-risks
  fitters: ``x_col``, ``i_col``, ``c_col``, ``n_col``, ``tl_col``,
  ``tr_col``, ``e_col`` and ``Z_cols``, as ``CoxPH.fit_from_df``,
  ``CompetingRisks.fit_from_df`` and ``CauseSpecificNHPP.fit_from_df`` do.

The columns are read as they are and handed to ``fit``, which does all
the checking: a missing value is treated exactly as the same value in an
array would be (Conventions, "Missing values"). A duration or date column
is refused, since its storage ticks would be read as numbers (#480).
"""

import functools
import inspect
import types
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from surpyval.utils import is_missing_event, refuse_time_values


class fitter_method:
    """A method that binds to the fitter it is called on, instance or class.

    Most fitters are instances (``Weibull``, ``KaplanMeier``, ``CrowAMSAA``)
    whose ``fit`` is an instance method; a few are classes whose ``fit`` is
    a class method (``SurvivalTree``, ``RandomSurvivalForest``) or works
    both ways (``MixtureModel``). The shared ``fit_from_df`` calls
    ``self.fit`` either way.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        self.func = func
        functools.update_wrapper(self, func)  # type: ignore[arg-type]

    def __get__(self, obj: Any, objtype: Any = None) -> Any:
        return types.MethodType(self.func, objtype if obj is None else obj)


def require_frame(df: Any) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        raise ValueError(
            f"df must be a pandas DataFrame, got {type(df).__name__}"
        )
    return df


def frame_column(
    df: pd.DataFrame, name: Any, arg: str, time: bool = False
) -> npt.NDArray:
    """The column ``name`` of ``df`` as an array, for argument ``arg``.

    A name that is not a column raises a ``ValueError`` naming the
    argument and the columns there are; ``time=True`` also refuses
    durations and dates (#480).
    """
    if name not in df.columns:
        raise ValueError(
            f"{arg}={name!r} is not a column of the DataFrame; its columns "
            f"are {list(df.columns)}"
        )
    column = df[name]
    if time:
        refuse_time_values(column, arg)
    return column.to_numpy()


def frame_columns(
    df: pd.DataFrame, names: Any, arg: str, time: bool = False
) -> npt.NDArray:
    """Several columns (a name or a list of names) as a 2-D array."""
    names = [names] if isinstance(names, str) else list(names)
    return np.column_stack([frame_column(df, k, arg, time) for k in names])


def _fitter_name(fitter: Any) -> str:
    name = getattr(fitter, "name", None)
    if not isinstance(name, str):
        cls = fitter if inspect.isclass(fitter) else type(fitter)
        name = cls.__name__.rstrip("_")
    return name


def call_fit(
    fitter: Any,
    arrays: dict[str, Any],
    arg_names: dict[str, str],
    fit_options: dict[str, Any],
) -> Any:
    """``fitter.fit(**arrays, **fit_options)``, refusing a column that its
    ``fit`` has no argument for (``arg_names`` maps each ``fit`` argument to
    the ``fit_from_df`` argument that named its column)."""
    params: Mapping[str, inspect.Parameter]
    try:
        params = inspect.signature(fitter.fit).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins only
        params = {}
    takes_any = any(p.kind is p.VAR_KEYWORD for p in params.values())
    for key in arrays:
        if params and not takes_any and key not in params:
            raise ValueError(
                f"{_fitter_name(fitter)}.fit takes no `{key}`, so "
                f"fit_from_df cannot use the column given as "
                f"`{arg_names.get(key, key)}`"
            )
    for key in fit_options:
        if key in arrays:
            raise ValueError(
                f"`{key}` is read from the DataFrame; name its column "
                f"with `{arg_names.get(key, key)}` rather than passing it "
                "as an option"
            )
    return fitter.fit(**arrays, **fit_options)


def _truncation(df: pd.DataFrame, value: Any, arg: str) -> npt.NDArray:
    """A truncation column (or a scalar applying to every row)."""
    if isinstance(value, str):
        return frame_column(df, value, arg, time=True).astype(float)
    if np.isscalar(value):
        return np.full(len(df), value, dtype=float)
    raise ValueError(f"`{arg}` must be a scalar or a column label string")


class UnivariateDataFrameMixin:
    """``fit_from_df`` for a fitter of one lifetime per row.

    The host's ``fit`` takes ``x`` (and, where it has them, ``c``, ``n``
    and ``t``); every other ``fit`` option is passed through.
    """

    @fitter_method
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x: str | None = None,
        c: str | None = None,
        n: str | None = None,
        xl: str | None = None,
        xr: str | None = None,
        tl: str | float | None = None,
        tr: str | float | None = None,
        **fit_options: Any,
    ) -> Any:
        r"""
        Fit to data held in the columns of a :class:`pandas.DataFrame`.

        The column names are passed in place of the arrays :meth:`fit`
        takes; every other :meth:`fit` option can be passed as a keyword.
        The columns are handed to :meth:`fit` as they are, so the result,
        and the treatment of a missing value, is that of :meth:`fit` on
        the same arrays.

        Parameters
        ----------

        df : DataFrame
            DataFrame of data to be used to create surpyval model

        x : string, optional
            column name for the column in df containing the variable data.
            If not provided must provide both xl and xr.

        c : string, optional
            column name for the column in df containing the censor flag of x.
            If not provided assumes all values of x are observed.

        n : string, optional
            column name in for the column in df containing the counts of x.
            If not provided assumes each x is one observation.

        xl : string, optional
            column name for the column in df containing the left interval for
            interval censored data. If left interval is -Inf, assumes left
            censored. If xl[i] == xr[i] assumes observed. Cannot be provided
            with x, must be provided with xr.

        xr : string, optional
            column name for the column in df containing the right interval
            for interval censored data. If right interval is Inf, assumes
            right censored. If xl[i] == xr[i] assumes observed. Cannot be
            provided with x, must be provided with xl.

        tl : string or scalar, optional
            If string, column name in for the column in df containing the left
            truncation data. If scalar assumes each x is left truncated by
            that value. If not provided assumes x is not left truncated.

        tr : string or scalar, optional
            If string, column name in for the column in df containing the
            right truncation data. If scalar assumes each x is right truncated
            by that value. If not provided assumes x is not right truncated.

        fit_options : dict, optional
            Every other option of :meth:`fit` (``how``, ``offset``,
            ``fixed``, ``dist``, ...), passed to it unchanged.

        Returns
        -------

        model
            The model :meth:`fit` returns.

        Raises
        ------
        ValueError
            If ``df`` is not a DataFrame, a name is not one of its columns,
            ``x`` is given with ``xl`` / ``xr``, or a column is given that
            :meth:`fit` has no argument for.

        Examples
        --------
        >>> import surpyval as surv
        >>> from surpyval.datasets import load_bofors_steel
        >>> df = load_bofors_steel()
        >>> model = surv.Weibull.fit_from_df(df, x='x', n='n', offset=True)
        >>> print(model)
        Parametric SurPyval Model
        =========================
        Distribution        : Weibull
        Fitted by           : MLE
        Data                : 389 units: 389 events at 10 unique times
        Offset (gamma)      : 39.76557772434183
        Parameters          :
             alpha: 7.141983615103902
              beta: 2.62047590823775
        >>> km = surv.KaplanMeier.fit_from_df(df, x='x', n='n')
        >>> km.sf([45, 48]).round(4)
        array([0.5861, 0.2571])
        """
        df = require_frame(df)
        if (x is not None) and ((xl is not None) or (xr is not None)):
            raise ValueError("Cannot use `x` and (`xl` and `xr`) together")
        arrays: dict[str, Any] = {}
        names = {"x": "x", "c": "c", "n": "n", "t": "tl` / `tr"}
        if x is not None:
            arrays["x"] = frame_column(df, x, "x", time=True).astype(float)
        elif xl is not None and xr is not None:
            arrays["x"] = np.column_stack(
                [
                    frame_column(df, xl, "xl", time=True).astype(float),
                    frame_column(df, xr, "xr", time=True).astype(float),
                ]
            )
            names["x"] = "xl` / `xr"
        else:
            raise ValueError(
                "Name the column of times with `x`, or the interval ends "
                "with both `xl` and `xr`"
            )
        if c is not None:
            arrays["c"] = frame_column(df, c, "c")
        if n is not None:
            arrays["n"] = frame_column(df, n, "n")
        if tl is not None or tr is not None:
            rows = len(df)
            arrays["t"] = np.column_stack(
                [
                    (
                        np.full(rows, -np.inf)
                        if tl is None
                        else _truncation(df, tl, "tl")
                    ),
                    (
                        np.full(rows, np.inf)
                        if tr is None
                        else _truncation(df, tr, "tr")
                    ),
                ]
            )
        return call_fit(self, arrays, names, fit_options)


class RecurrentDataFrameMixin:
    """``fit_from_df`` for a recurrent-event fitter: an event log with one
    row per event (or end of observation) and a column of unit ids."""

    @fitter_method
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        i_col: str | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        **fit_options: Any,
    ) -> Any:
        """
        Fit to an event log held in the columns of a
        :class:`pandas.DataFrame`.

        The column names are passed in place of the arrays :meth:`fit`
        takes, with the names every recurrent ``fit_from_df`` uses
        (``CauseSpecificNHPP.fit_from_df`` too); every other :meth:`fit`
        option (``how``, ``dist``, ``init``, ``windows``, ...) is passed to
        it unchanged. The columns are handed to :meth:`fit` as they are,
        so the result, and the refusal of a missing value in a unit's
        history, is that of :meth:`fit` on the same arrays.

        Parameters
        ----------
        df : pandas.DataFrame
            The event log.
        x_col : str
            Column of event (and end-of-observation) times.
        i_col : str, optional
            Column of item / unit ids. Defaults to a single item.
        c_col : str, optional
            Column of censoring flags (0 an event, 1 the end of a unit's
            observation).
        n_col : str, optional
            Column of event counts per row.
        tl_col, tr_col : str, optional
            Columns of per-row left / right truncation (the start and end
            of each unit's observation window), for a fitter whose
            :meth:`fit` takes ``tl`` / ``tr``.
        **fit_options
            Every other option of :meth:`fit`.

        Returns
        -------
        model
            The model :meth:`fit` returns.

        Raises
        ------
        ValueError
            If ``df`` is not a DataFrame, a name is not one of its columns,
            or a column is given that :meth:`fit` has no argument for.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import CrowAMSAA
        >>> log = pd.DataFrame({
        ...     "hours": [120, 380, 610, 700, 90, 400, 520, 650],
        ...     "truck": [1, 1, 1, 1, 2, 2, 2, 2],
        ...     "c": [0, 0, 0, 1, 0, 0, 0, 1],
        ... })
        >>> model = CrowAMSAA.fit_from_df(
        ...     log, x_col="hours", i_col="truck", c_col="c"
        ... )
        >>> model.params.round(4)
        array([260.1738,   1.1522])
        """
        df = require_frame(df)
        columns = {
            "x": x_col,
            "i": i_col,
            "c": c_col,
            "n": n_col,
            "tl": tl_col,
            "tr": tr_col,
        }
        arrays = _read_columns(df, columns)
        names = {k: f"{k}_col" for k in columns}
        return call_fit(self, arrays, names, fit_options)


_TIME_ARGS = frozenset({"x", "xl", "xr", "tl", "tr"})


def _read_columns(
    df: pd.DataFrame, columns: dict[str, str | None]
) -> dict[str, Any]:
    """Each named column (``fit`` argument -> column name) as an array."""
    return {
        key: frame_column(df, name, f"{key}_col", time=key in _TIME_ARGS)
        for key, name in columns.items()
        if name is not None
    }


def _regression_fit_from_df(
    fitter: Any,
    df: pd.DataFrame,
    x_col: str,
    Z_cols: str | list[str],
    columns: dict[str, str | None],
    fit_options: dict[str, Any],
) -> Any:
    df = require_frame(df)
    arrays = _read_columns(df, {"x": x_col, **columns})
    arrays["Z"] = frame_columns(df, Z_cols, "Z_cols").astype(float)
    names = {k: f"{k}_col" for k in arrays} | {"Z": "Z_cols"}
    return call_fit(fitter, arrays, names, fit_options)


class RecurrentRegressionDataFrameMixin:
    """``fit_from_df`` for a recurrent-event regression: an event log with
    a column of unit ids and covariate columns."""

    @fitter_method
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        Z_cols: str | list[str],
        i_col: str | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        **fit_options: Any,
    ) -> Any:
        """
        Fit to an event log held in the columns of a
        :class:`pandas.DataFrame`.

        The column names are passed in place of the arrays :meth:`fit`
        takes, with the names every recurrent and regression
        ``fit_from_df`` uses; every other :meth:`fit` option is passed to
        it unchanged. The columns are handed to :meth:`fit` as they are,
        so the result, and the refusal of a missing value in a unit's
        history, is that of :meth:`fit` on the same arrays.

        Parameters
        ----------
        df : pandas.DataFrame
            The event log.
        x_col : str
            Column of event (and end-of-observation) times.
        Z_cols : str or list of str
            Column(s) of the covariates, constant within a unit, in the
            order of the fitted coefficients.
        i_col : str, optional
            Column of item / unit ids. Defaults to a single item.
        c_col : str, optional
            Column of censoring flags (0 an event, 1 the end of a unit's
            observation).
        n_col : str, optional
            Column of event counts per row.
        tl_col, tr_col : str, optional
            Columns of per-row left / right truncation.
        **fit_options
            Every other option of :meth:`fit` (``dist``, ``init``, ...).

        Returns
        -------
        ProportionalIntensityModel
            The model :meth:`fit` returns.

        Raises
        ------
        ValueError
            If ``df`` is not a DataFrame or a name is not one of its
            columns.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> log = pd.DataFrame({
        ...     "hours": [120, 380, 610, 700, 90, 200, 330, 400, 520],
        ...     "truck": [1, 1, 1, 1, 2, 2, 2, 2, 2],
        ...     "c": [0, 0, 0, 1, 0, 0, 0, 0, 1],
        ...     "load": [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ... })
        >>> model = ProportionalIntensityHPP.fit_from_df(
        ...     log, x_col="hours", Z_cols="load", i_col="truck", c_col="c"
        ... )
        >>> model.coeffs.round(4)
        array([0.5849])
        """
        columns = {
            "i": i_col,
            "c": c_col,
            "n": n_col,
            "tl": tl_col,
            "tr": tr_col,
        }
        return _regression_fit_from_df(
            self, df, x_col, Z_cols, columns, fit_options
        )


class RegressionDataFrameMixin:
    """``fit_from_df`` for a regression fitter without a formula interface
    (the survival trees and forest)."""

    @fitter_method
    def fit_from_df(
        self,
        df: pd.DataFrame,
        x_col: str,
        Z_cols: str | list[str],
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        **fit_options: Any,
    ) -> Any:
        """
        Fit to data held in the columns of a :class:`pandas.DataFrame`.

        The column names are passed in place of the arrays :meth:`fit`
        takes, with the names of every regression ``fit_from_df``; every
        other :meth:`fit` option is passed to it unchanged. The columns
        are handed to :meth:`fit` as they are, so the result, and the
        treatment of a missing covariate, is that of :meth:`fit` on the
        same arrays.

        Parameters
        ----------
        df : pandas.DataFrame
            The data.
        x_col : str
            Column of observed times.
        Z_cols : str or list of str
            Column(s) of the covariates, in the order :meth:`fit` reads
            them (the columns of ``Z``).
        c_col : str, optional
            Column of censoring flags.
        n_col : str, optional
            Column of counts.
        tl_col, tr_col : str, optional
            Columns of left / right truncation.
        **fit_options
            Every other option of :meth:`fit`.

        Returns
        -------
        model
            The model :meth:`fit` returns.

        Raises
        ------
        ValueError
            If ``df`` is not a DataFrame or a name is not one of its
            columns.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from surpyval.beta.ml import SurvivalTree
        >>> rng = np.random.default_rng(0)
        >>> df = pd.DataFrame({"z": rng.uniform(0, 1, 60)})
        >>> df["x"] = rng.weibull(2, 60) * np.where(df["z"] > 0.5, 5, 20)
        >>> tree = SurvivalTree.fit_from_df(
        ...     df, x_col="x", Z_cols="z", random_state=0
        ... )
        >>> tree.sf(10.0, [[0.2], [0.8]]).round(3)
        array([0.486, 0.029])
        """
        columns = {"c": c_col, "n": n_col, "tl": tl_col, "tr": tr_col}
        return _regression_fit_from_df(
            self, df, x_col, Z_cols, columns, fit_options
        )


def cause_column(df: pd.DataFrame, e_col: str) -> npt.NDArray:
    """The column of causes, a blank / NaN cell read as ``None`` (a
    censored row), as ``CompetingRisksProportionalHazards.fit_from_df``
    reads it."""
    e = frame_column(df, e_col, "e_col").astype(object).copy()
    e[[is_missing_event(v) for v in e]] = None
    return e
