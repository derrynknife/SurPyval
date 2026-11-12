"""Data handling shared by every model: SurPyval's data formats and the
conversions between them (``data_formats``), the checks the fitters make
of their inputs (``validation``), covariate handling (``covariates``),
rounding helpers (``numeric``) and where warnings point (``warnings``).

``__all__`` lists what this namespace promises: the documented handlers,
converters and helpers (see the Utilities page). Every other name
importable here is internal and may change without notice.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from surpyval.utils.covariates import (
    check_covariate_rows,
    finite_covariate_mask,
    formula_model_matrix,
    numeric_columns,
    optional_column,
)
from surpyval.utils.data_formats import (
    _check_truncation_bounds,
    _entered_before,
    _get_idx,
    _handled_xcnt_to_xrd,
    _time_kind,
    _truncation_bound,
    _whole_number_array,
    _zero_coded,
    coerce_xcnt_x,
    format_truncation,
    fs_to_xcnt,
    fs_to_xrd,
    fsl_to_xcnt,
    fsli_handler,
    fsli_to_xcnt,
    group_xcnt,
    is_missing_event,
    missing_events,
    refuse_time_values,
    resolve_cr_censoring,
    validate_float_array,
    xcn_to_fs,
    xcnt_handler,
    xcnt_sort,
    xcnt_to_xrd,
    xrd_handler,
    xrd_to_xcnt,
)
from surpyval.utils.numeric import (
    _round_vals,
    ffill_or_zero,
    round_sig,
)
from surpyval.utils.validation import (
    FG_BASELINE_OPTIONS,
    _check_x_not_empty,
    check_c_and_e,
    check_e_and_x,
    check_left_or_int_cens,
    check_no_censoring,
    check_option,
    check_Z_and_x,
    no_left_or_int,
    validate_1d,
    validate_cif_event,
    validate_coxph_df_inputs,
    validate_cr_df_inputs,
    validate_cr_inputs,
    validate_event,
)
from surpyval.utils.warnings import caller_stacklevel

# The old private name, until its callers import ``caller_stacklevel``.
_caller_stacklevel = caller_stacklevel

__all__ = [
    "xcnt_handler",
    "fsli_handler",
    "xrd_handler",
    "fs_to_xcnt",
    "fsl_to_xcnt",
    "fsli_to_xcnt",
    "xcn_to_fs",
    "fs_to_xrd",
    "xcnt_to_xrd",
    "xrd_to_xcnt",
    "round_sig",
    "xcnt_sort",
    "group_xcnt",
    "coerce_xcnt_x",
    "format_truncation",
    "is_missing_event",
    "missing_events",
    "resolve_cr_censoring",
]

# The promised names are documented here, as ``surpyval.utils.<name>``, so
# they name this namespace as their module (as numpy's public functions
# name ``numpy``): ``help()`` and the doctests report them as before.
for _name in __all__:
    globals()[_name].__module__ = __name__
del _name

COX_PH_METHODS = ["breslow", "efron", "exact", "kalbfleisch-prentice", "kp"]


def validate_coxph(
    x: npt.ArrayLike | None,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    Z: npt.ArrayLike | None,
    tl: npt.ArrayLike | None,
    method: str,
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    check_option("tie_method", method, COX_PH_METHODS)

    # The Cox partial likelihood accommodates left-truncation (delayed entry)
    # by adjusting the risk sets, but has no way to incorporate right or
    # interval truncation (that needs a reverse-time / retro-hazard model). A
    # 2-D ``tl`` is the natural way a user would try to pass a [tl, tr] pair,
    # so it is refused with a clear, Cox-specific message rather than the
    # generic truncation error (which points at a ``t`` argument CoxPH does
    # not have).
    from surpyval.univariate.regression.regression_data import (
        semi_parametric_inputs,
    )

    # The partial likelihood is built from risk sets at exact event times,
    # so it can only use observed (0) and right-censored (1) rows. A left-
    # or interval-censored row has no event time to place in a risk set;
    # the generators would otherwise read ``c != 0`` as "right-censored"
    # and silently fit the wrong likelihood (or, for interval rows, index
    # a 2-D ``x`` as if it were 1-D). Refuse them and point at a model that
    # has a full likelihood for them. Rows with a NaN / infinite covariate
    # are dropped with a warning, as every regression fitter does.
    x_a, c_a, n_a, tl_a, Z_arr = semi_parametric_inputs(
        x,
        Z,
        c,
        n,
        tl,
        censoring=(
            "CoxPH supports only observed (c=0) and right-censored (c=1) "
            "observations (with optional left-truncation `tl`); the Cox "
            "partial likelihood has no term for left-censored (c=-1) or "
            "interval-censored (c=2) data. Use IntervalCoxPH.fit(x, Z, "
            "c=c), the Cox model with a non-parametric baseline for such "
            "data, or a parametric regression model, e.g. WeibullPH.fit(x, "
            "Z, c=c) or WeibullAFT.fit(x, Z, c=c), which handle every "
            "censoring type."
        ),
        truncation=(
            "CoxPH supports left-truncation (delayed entry) only, supplied "
            "as a one-dimensional `tl`. Right or interval truncation is not "
            "available for the Cox partial likelihood; use a parametric "
            "proportional-hazards fitter (e.g. WeibullPH) with t=[tl, tr] "
            "for right/interval-truncated data."
        ),
    )
    return x_a, c_a, n_a, tl_a, Z_arr


def validate_fine_gray_inputs(
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    e: npt.ArrayLike,
    c: npt.ArrayLike | None,
    n: npt.ArrayLike | None,
    w: npt.ArrayLike | float | None = None,
    tl: npt.ArrayLike | float | None = None,
) -> tuple:
    """``(x, Z, e, c, n)`` of a competing-risks regression, checked, with
    the rows that have a missing covariate dropped; and the weights ``w``
    of the rows kept as a sixth entry where ``w`` is given (checked by
    ``check_weights``, one per row or one for every row), and the entry
    times ``tl`` of the rows kept as the last entry where ``tl`` is given
    (checked as ``CoxPH``'s are: one-dimensional, each before its row's
    time)."""
    # A missing event (None / NaN) marks a censored observation; if c is not
    # given it is derived from the events.
    e, c = resolve_cr_censoring(e, c)
    from surpyval.univariate.regression.regression_data import (
        semi_parametric_inputs,
    )

    rows: tuple = (np.array(e),)
    if w is not None:
        from surpyval.utils.weights import check_weights

        # The weights travel with the rows, so a row dropped for a
        # missing covariate takes its weight with it.
        rows += (check_weights(w, rows[0].shape[0]),)
    # Rows with a NaN / infinite covariate are dropped with a warning, as
    # every regression fitter does (they used to be dropped silently here);
    # left- and interval-censored rows are refused below, after the drop.
    x_a, c_a, n_a, tl_a, Z_arr, e_arr, *w_kept = semi_parametric_inputs(
        x,
        Z,
        c,
        n,
        tl,
        censoring=None,
        truncation=(
            "The cause-specific Cox model takes left truncation (delayed "
            "entry) only, as a one-dimensional `tl` of entry times, one "
            "per row: the Cox partial likelihood has no term for right or "
            "interval truncation. Pass the entry times alone, tl=entry."
        ),
        rows=rows,
    )

    check_e_and_x(e_arr, x_a)
    check_Z_and_x(Z_arr, x_a)
    e_arr = check_c_and_e(c_a, e_arr)
    check_left_or_int_cens(c_a)

    out: tuple = (x_a, Z_arr, e_arr, c_a, n_a)
    if w is not None:
        out += (np.asarray(w_kept[0], float),)
    if tl is not None:
        out += (tl_a,)
    return out
