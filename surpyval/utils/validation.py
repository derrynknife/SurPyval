"""The input checks the fitters share.

Internal: their behaviour is covered by the fitters' own documentation,
and they may change without notice.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import numpy.typing as npt

from surpyval.utils.covariates import finite_covariate_mask
from surpyval.utils.data_formats import resolve_cr_censoring, xcnt_handler

FG_BASELINE_OPTIONS = ["Nelson-Aalen", "Kaplan-Meier"]

# The sides a confidence bound can take, everywhere a ``bound`` is asked.
BOUNDS = ("two-sided", "lower", "upper")

# The functions a parametric model's ``cb`` bounds ('R' and 'F' are the
# aliases of 'sf' and 'ff').
CB_ON = ("sf", "R", "ff", "F", "Hf", "hf", "df")


def format_options(accepted: Any) -> str:
    """The values in ``accepted`` as a phrase, ``'a', 'b' or 'c'``, for
    the messages that list what an option or name takes.

    Examples
    --------
    >>> from surpyval.utils.validation import format_options
    >>> format_options(("two-sided", "lower", "upper"))
    "'two-sided', 'lower' or 'upper'"
    >>> format_options(("z",))
    "'z'"
    """
    shown = [repr(a) for a in accepted]
    if len(shown) < 2:
        return "".join(shown)
    return "{} or {}".format(", ".join(shown[:-1]), shown[-1])


def option_error(
    name: str, value: Any, accepted: Any, note: str | None = None
) -> ValueError:
    """The ``ValueError`` :func:`check_option` raises, for the code that
    finds an unknown value at the end of its own ``if``/``elif`` chain.

    Examples
    --------
    >>> from surpyval.utils.validation import option_error
    >>> option_error("scale", "logit", ("hazard", "odds", "normal"))
    ValueError("'scale' must be one of 'hazard', 'odds' or 'normal'; got 'logit'")
    """  # noqa: E501
    one_of = "" if len(accepted) == 1 else "one of "
    message = "'{}' must be {}{}; got {!r}".format(
        name, one_of, format_options(accepted), value
    )
    if note:
        message += ". " + note
    return ValueError(message)


def check_option(
    name: str, value: Any, accepted: Any, note: str | None = None
) -> None:
    """Refuse an option ``value`` that is not one of ``accepted``, with
    the one message every enumerated option uses (principles 2 and 21):
    ``'<name>' must be one of <accepted>; got <value>`` (``must be
    <accepted>`` when only one value is accepted), followed by ``note``
    when given (why a value is refused, or what to use instead).

    The options are strings, so anything that is not a string (an
    array, ``None``, a number) is refused too, rather than compared.

    Examples
    --------
    >>> from surpyval.utils.validation import BOUNDS, check_option
    >>> check_option("bound", "lower", BOUNDS)
    >>> check_option("bound", "both", BOUNDS)
    Traceback (most recent call last):
    ...
    ValueError: 'bound' must be one of 'two-sided', 'lower' or 'upper'; got 'both'
    >>> check_option("dist", "t", ("z",))
    Traceback (most recent call last):
    ...
    ValueError: 'dist' must be 'z'; got 't'
    """  # noqa: E501
    if not (isinstance(value, str) and value in accepted):
        raise option_error(name, value, accepted, note)


def _check_x_not_empty(func: Callable) -> Callable:
    # Decorator to check that x is not empty
    def wrap(obj: Any, x: Any, *args: Any, **kwargs: Any) -> Any:
        x = np.array(x)
        if x.size == 0:
            return 0
        # Make sure we are using a numpy array (of 1D)
        x = np.atleast_1d(x)
        result = func(obj, x, *args, **kwargs)
        return result

    return wrap


def check_no_censoring(c: npt.NDArray) -> bool:
    return any(c != 0)


def no_left_or_int(c: npt.NDArray) -> bool:
    return any((c == -1) | (c == 2))


def validate_1d(arr: npt.ArrayLike, name: str) -> npt.NDArray:
    """
    Coerce ``arr`` to a one-dimensional float array, naming it in the
    error when it is not. A scalar becomes a length-one array; anything
    two-dimensional or higher is rejected.
    """
    out = np.atleast_1d(np.asarray(arr, dtype=float))
    if out.ndim != 1:
        raise ValueError("'{}' must be one-dimensional".format(name))
    return out


def check_left_or_int_cens(c: npt.NDArray) -> None:
    if (-1 in c) or (2 in c):
        raise ValueError(
            "Left or interval censoring not implemented with Competing Risks"
        )


def check_Z_and_x(Z: npt.NDArray, x: npt.NDArray) -> None:
    if x.shape[0] != Z.shape[0]:
        raise ValueError("Z must have len(x) number of rows")


def check_e_and_x(e: npt.NDArray, x: npt.NDArray) -> None:
    if e.shape != x.shape:
        raise ValueError(
            "Event vector, e, and duration vector, x, must have same shape"
        )


def check_c_and_e(c: npt.NDArray, e: npt.NDArray) -> None:
    if any(e_i is not None for e_i in e[c == 1]) or any(
        e_i is None for e_i in e[c != 1]
    ):
        raise ValueError(
            "A missing event type (None / NaN) is allowed only for a "
            "censored observation (c = 1), and every censored observation "
            "must have one."
        )


def validate_cr_df_inputs(
    df: Any,
    x_col: str,
    e_col: str,
    c_col: "str | None" = None,
    n_col: "str | None" = None,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    x = df[x_col].values
    e = df[e_col].values

    if c_col:
        c = df[c_col].values
    else:
        c = None

    if n_col:
        n = df[n_col].values
    else:
        n = None
    return x, c, n, e


def validate_cr_inputs(
    x: npt.ArrayLike,
    c: "npt.ArrayLike | None",
    n: "npt.ArrayLike | None",
    e: npt.ArrayLike,
    method: str,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    # Validates the inputs prior to be used by the CoxPH model.
    # Use existing surpyval validator. But don't group and sort
    # so as to put it out of order of the event array, e.
    # A missing event (None / NaN) marks a censored observation; if c is not
    # given it is derived from the events.
    e, c = resolve_cr_censoring(e, c)
    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)

    e = np.array(e)
    x, c, n = (np.array(a).astype(float) for a in [x, c, n])

    # Check same shape
    check_e_and_x(e, x)

    # Not implemented, yet.
    check_left_or_int_cens(c)

    # Ensure all cases where c is 0, e is not None and
    # where c is 1 e is None
    check_c_and_e(c, e)

    # Two baselines
    # TODO: Add fleming-harrington
    check_option("how", method, FG_BASELINE_OPTIONS)

    return x, c, n, e


def validate_event(mapping: dict, event: Any) -> None:
    if event is not None and event not in mapping:
        raise ValueError("Event type not in model")


def validate_cif_event(event: Any) -> None:
    if event is None:
        raise ValueError("CIF needs event type, not None")


def validate_coxph_df_inputs(
    df: Any,
    x_col: str,
    c_col: "str | None",
    n_col: "str | None",
    Z_cols: "str | list[str] | None",
    formula: "str | None",
    tl_col: "str | None" = None,
    strata_col: "str | None" = None,
) -> tuple:
    from surpyval.univariate.regression.regression_data import (
        design_matrix_from_df,
    )

    # Rows with a missing covariate drop (with one warning), and the times,
    # flags, counts, entry times and strata with them.
    Z, feature_names, model_spec = design_matrix_from_df(df, Z_cols, formula)
    mask = finite_covariate_mask(Z)
    Z = Z[mask]

    x = df.loc[mask, x_col].values

    if c_col is None:
        c = None
    else:
        c = df.loc[mask, c_col].values

    if n_col is None:
        n = None
    else:
        n = df.loc[mask, n_col].values

    # Delayed-entry times and stratum labels go through the same row mask as
    # the covariates so they stay aligned with ``x`` when rows with missing
    # covariates drop.
    tl = None if tl_col is None else df.loc[mask, tl_col].values
    strata = None if strata_col is None else df.loc[mask, strata_col].values

    x, c, n, _ = xcnt_handler(x, c, n, group_and_sort=False)

    return x, c, n, tl, strata, Z, formula, feature_names, model_spec
