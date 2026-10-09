"""The input checks the fitters share.

Internal: their behaviour is covered by the fitters' own documentation,
and they may change without notice.
"""

from __future__ import annotations

import warnings
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


def alpha_ci_error(alpha_ci: Any, note: str | None = None) -> ValueError:
    """The one refusal of an ``alpha_ci`` outside (0, 1), for every
    method that takes it.

    Examples
    --------
    >>> from surpyval.utils.validation import alpha_ci_error
    >>> alpha_ci_error(1.5)
    ValueError("'alpha_ci' must be strictly between 0 and 1; got 1.5")
    """
    message = "'alpha_ci' must be strictly between 0 and 1; got {}".format(
        _plain_value(alpha_ci)
    )
    if note:
        message += ". " + note
    return ValueError(message)


def _plain_value(value: Any) -> str:
    """A number as the user wrote it (``1.5``, not ``np.float64(1.5)``);
    anything else as its repr."""
    if isinstance(value, (bool, np.bool_)):
        return repr(bool(value))
    if isinstance(value, (int, np.integer)):
        return repr(int(value))
    if isinstance(value, (float, np.floating)):
        return repr(float(value))
    return repr(value)


# The modules of the wrappers between a caller and a method that checks
# its ``alpha_ci`` (``keeps_query_shape``, the renamed-argument shims):
# not callers in their own right.
_WRAPPER_MODULES = ("shapes.py", "deprecation.py")


def _called_by_user(depth: int) -> bool:
    """Whether the function ``depth`` frames above the caller of this one
    was called from outside surpyval (or from its tests), looking past
    the wrappers of :data:`_WRAPPER_MODULES`."""
    import os
    import sys

    utils_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(utils_dir) + os.sep
    tests_dir = os.path.join(package_dir, "tests") + os.sep
    wrappers = tuple(os.path.join(utils_dir, m) for m in _WRAPPER_MODULES)
    frame = sys._getframe(depth + 2).f_back
    while frame is not None:
        name = os.path.abspath(frame.f_code.co_filename)
        if name not in wrappers:
            break
        frame = frame.f_back
    if frame is None:
        return True
    name = os.path.abspath(frame.f_code.co_filename)
    return not name.startswith(package_dir) or name.startswith(tests_dir)


def check_alpha_ci(alpha_ci: Any, note: str | None = None) -> None:
    """Refuse an ``alpha_ci`` that is not strictly between 0 and 1 (with
    :func:`alpha_ci_error`), and warn of one above 0.5 (#647). Every
    method that takes ``alpha_ci`` calls it first.

    ``alpha_ci`` is the significance level, the bound's total tail
    probability: 0.05 gives a 95% interval. A level above 0.5 is almost
    always the confidence given for it (``alpha_ci=0.95``, as
    ``reliability``'s ``CI=0.95``), which gives a 5% interval, so it is
    warned of, once per call: only where the method checking it was
    called by the user, not where one of surpyval's methods called it in
    turn (``plot`` calls ``cb``, which checks it again).

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.validation import check_alpha_ci
    >>> check_alpha_ci(0.05)
    >>> check_alpha_ci(1.5)
    Traceback (most recent call last):
    ...
    ValueError: 'alpha_ci' must be strictly between 0 and 1; got 1.5
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     check_alpha_ci(0.95)
    >>> print(caught[0].message)  # doctest: +NORMALIZE_WHITESPACE
    alpha_ci is the significance level: alpha_ci=0.95 gives a 5%
    interval; for a 95% interval pass alpha_ci=0.05.
    """
    try:
        inside = bool(0 < alpha_ci < 1)
    except (TypeError, ValueError):
        # Not a number, or an array (whose comparison has no truth value)
        inside = False
    if not inside:
        raise alpha_ci_error(alpha_ci, note)
    if alpha_ci > 0.5 and _called_by_user(0):
        from surpyval.utils.warnings import caller_stacklevel

        level = float(alpha_ci)
        warnings.warn(
            "alpha_ci is the significance level: alpha_ci={:g} gives a {:g}% "
            "interval; for a {:g}% interval pass alpha_ci={:g}.".format(
                level,
                round(100 * (1 - level), 10),
                round(100 * level, 10),
                round(1 - level, 12),
            ),
            UserWarning,
            stacklevel=caller_stacklevel(),
        )


def unknown_cause_error(cause: Any, causes: Any) -> ValueError:
    """The one refusal of a cause (event type) a competing-risks model or
    its data does not have.

    Examples
    --------
    >>> from surpyval.utils.validation import unknown_cause_error
    >>> unknown_cause_error("c", ["a", "b"])
    ValueError("Unknown cause 'c'; the causes are ['a', 'b']")
    """
    return ValueError(
        "Unknown cause {!r}; the causes are {}".format(cause, list(causes))
    )


def no_covariance_error(
    why: str = "the Hessian was singular at the optimum",
) -> ValueError:
    """The one refusal of a standard error or Wald bound from a model that
    carries no parameter covariance.

    Examples
    --------
    >>> from surpyval.utils.validation import no_covariance_error
    >>> print(no_covariance_error())
    The model carries no parameter covariance (the Hessian was singular at the optimum); its standard errors and confidence bounds are unavailable.
    """  # noqa: E501
    return ValueError(
        "The model carries no parameter covariance ({}); its standard "
        "errors and confidence bounds are unavailable.".format(why)
    )


def missing_cause_error(what: str) -> ValueError:
    """The one refusal of a cause-specific call made without its cause.

    Examples
    --------
    >>> from surpyval.utils.validation import missing_cause_error
    >>> missing_cause_error("The CIF")
    ValueError('The CIF is of one cause at a time; pass `event`.')
    """
    return ValueError(
        "{} is of one cause at a time; pass `event`.".format(what)
    )


def warn_outside_unit_interval(
    p: npt.ArrayLike, what: str = "qf", closed: bool = True
) -> npt.NDArray:
    """Where the probabilities ``p`` given to a quantile function are
    outside [0, 1], with one warning saying so if any are (#576).

    Their quantile is NaN, as scipy's ``ppf`` gives (#485); but a value
    such as 1.5 is not missing, it is a mistake -- most often a
    percentage given for a probability, ``qf(10)`` for the B10 life --
    and it used to give NaN in silence. NaN itself is a missing value
    (principle 3), and is not warned of.

    ``what`` names the method in the message (``"quantile_cb"`` for the
    bounds on the quantiles, #626); ``closed=False`` takes the open
    interval (0, 1) instead, for a parametric ``quantile_cb``, which does
    not bound ``qf(0)`` and ``qf(1)``, the ends of the support.

    Examples
    --------
    >>> import warnings
    >>> from surpyval.utils.validation import warn_outside_unit_interval
    >>> with warnings.catch_warnings(record=True) as caught:
    ...     warnings.simplefilter("always")
    ...     outside = warn_outside_unit_interval([0.1, 10.0, float("nan")])
    >>> outside
    array([False,  True, False])
    >>> print(caught[0].message)  # doctest: +ELLIPSIS
    qf: 1 of the 3 probabilities given is outside [0, 1] (10.0), ...
    """
    u = np.asarray(p, dtype=float)
    if closed:
        outside = (u < 0) | (u > 1)
        interval = "[0, 1]"
    else:
        outside = (u <= 0) | (u >= 1)
        interval = "(0, 1)"
    noun = "quantile" if what == "qf" else "bound"
    if outside.any():
        from surpyval.utils.warnings import caller_stacklevel

        k = int(outside.sum())
        warnings.warn(
            "{}: {} of the {} probabilities given {} outside {} ({}), "
            "so {} {}{} NaN. `p` is a probability: for the B10 life "
            "pass 0.1, not 10.".format(
                what,
                k,
                u.size,
                "is" if k == 1 else "are",
                interval,
                ", ".join(str(float(v)) for v in u[outside][:3])
                + (", ..." if k > 3 else ""),
                "its" if k == 1 else "their",
                noun,
                " is" if k == 1 else "s are",
            ),
            stacklevel=caller_stacklevel(),
        )
    return outside


def all_in_unit_interval(u: npt.NDArray) -> bool:
    """Whether every value of the float array ``u`` is in [0, 1], none
    NaN: the probabilities a quantile function is nearly always given,
    for which :func:`warn_outside_unit_interval` and the NaN check have
    nothing to do (#769).

    It takes two reductions and makes no array, where those checks make
    four (NaN propagates through ``min`` and ``max``, and fails both
    comparisons). An empty ``u`` gives False, for the full checks.

    Examples
    --------
    >>> import numpy as np
    >>> from surpyval.utils.validation import all_in_unit_interval
    >>> all_in_unit_interval(np.array([0.0, 0.5, 1.0]))
    True
    >>> all_in_unit_interval(np.array([0.5, float("nan")]))
    False
    >>> all_in_unit_interval(np.array(1.5))
    False
    """
    if not u.size:
        return False
    return 0.0 <= float(u.min()) and float(u.max()) <= 1.0


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
    return bool(np.any(np.asarray(c) != 0))


def no_left_or_int(c: npt.NDArray) -> bool:
    c = np.asarray(c)
    return bool(np.any((c == -1) | (c == 2)))


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
        raise unknown_cause_error(event, mapping)


def validate_cif_event(event: Any) -> None:
    if event is None:
        raise missing_cause_error("The CIF")


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
    from surpyval.utils.dataframe import check_columns

    # A missing column is named, with the columns there are, as the
    # parametric families' fit_from_df names it (#571, #663); it was a
    # bare KeyError.
    check_columns(
        df,
        x_col=x_col,
        c_col=c_col,
        n_col=n_col,
        tl_col=tl_col,
        strata_col=strata_col,
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
