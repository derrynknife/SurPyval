"""Time-varying covariates for the proportional-intensity models.

An item's intensity at time ``t`` is the baseline's times ``exp(Z(t)'
beta)``, with ``Z(t)`` the covariates in force then, piecewise constant in
time. The data are *start-stop* rows, the layout of the life regressions'
``fit_tvc`` (:mod:`surpyval.univariate.regression.tvc_fit`): each row is
an interval ``(xl, xr]`` of one item's history over which its covariates
hold one value. An item may fail many times, so any of its rows may end
in a failure (``c = 0``), not only its last.

The log-likelihood is the Poisson process's with the intensity that
changes at the rows' ends:

.. math::
    \\ell = \\sum_{\\text{failures}} \\left[\\ln \\lambda_0(x_r)
    + Z'\\beta\\right]
    - \\sum_{\\text{rows}} e^{Z'\\beta}
    \\left[\\Lambda_0(x_r) - \\Lambda_0(x_l)\\right],

and a row holding a count of failures (``c = 2``, ``n`` of them somewhere
in the interval) adds the Poisson probability of that count. With one
covariate value per item it is the likelihood of the fixed-covariate fit.

This module holds the data handling (:func:`handle_recurrent_tvc`,
:func:`handle_recurrent_tvc_timeline`), the likelihoods and
:class:`ProportionalIntensityTVCFitMixin`, which gives the two fitters
``fit_tvc_timeline`` and the DataFrame entry points on top of their own
``fit_tvc``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import autograd.numpy as anp
import numpy as np
import numpy.typing as npt
from scipy.special import gammaln

from surpyval.univariate.regression._aliasing import covariate_columns
from surpyval.utils.data_formats import refuse_boolean_censoring
from surpyval.utils.dataframe import frame_column, require_frame

if TYPE_CHECKING:
    import pandas as pd

#: The censoring flags of a start-stop row of a repairable item.
_FLAGS = (0, 1, 2)


def _named(item: Any) -> str:
    """An item id for a message: ``1``, not ``np.int64(1)``."""
    return repr(item.item() if isinstance(item, np.generic) else item)


def _as_covariates(Z: npt.ArrayLike, nrow: int, rows: str) -> npt.NDArray:
    """``Z`` as a float matrix with one row per data row."""
    try:
        Z = np.asarray(Z, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "Z must be numeric: one row of covariate values per "
            f"{rows}. Code a categorical covariate as 0/1 columns."
        ) from None
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)
    if Z.ndim != 2 or Z.shape[0] != nrow:
        raise ValueError(
            f"Z must have one row per {rows} (same length as i); got "
            f"shape {Z.shape}, expected {nrow} rows"
        )
    return Z


def _as_flags(c: npt.ArrayLike, nrow: int) -> npt.NDArray:
    """``c`` as whole-number flags, each one of 0, 1 and 2."""
    refuse_boolean_censoring(c)
    try:
        flags = np.asarray(c, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "c must be numeric: 0 (a failure at the end of the row), 1 "
            "(no failure) or 2 (a count of failures, with n)"
        ) from None
    if flags.ndim != 1 or flags.shape[0] != nrow:
        raise ValueError(
            "i and c must have the same length; c has shape "
            f"{flags.shape}, expected {nrow} rows"
        )
    if np.any(flags == -1):
        raise ValueError(
            "c = -1 (a left-censored count) has no start-stop form: give "
            "the count as a c = 2 row over the interval (xl, xr] it was "
            "counted in, with the number of failures in n"
        )
    if not np.isin(flags, _FLAGS).all():
        raise ValueError(
            "c must be 0 (a failure at the end of the row), 1 (no "
            "failure: a covariate change or the end of observation) or 2 "
            "(a count of failures within the row, with n)"
        )
    return flags.astype(int)


def _as_counts(
    n: npt.ArrayLike | None, c: npt.NDArray, nrow: int
) -> npt.NDArray:
    """``n`` as the number of failures each row holds: a count on a
    ``c = 2`` row, and 1 on the others."""
    if n is None:
        counts = np.ones(nrow)
    else:
        try:
            counts = np.asarray(n, dtype=float)
        except (TypeError, ValueError):
            raise ValueError(
                "n must be numeric: the number of failures on a c = 2 row"
            ) from None
    if counts.ndim != 1 or counts.shape[0] != nrow:
        raise ValueError(
            "i and n must have the same length; n has shape "
            f"{counts.shape}, expected {nrow} rows"
        )
    counted = c == 2
    bad = ~np.isfinite(counts) | (counts < 0) | (counts != np.round(counts))
    if np.any(bad & counted):
        raise ValueError(
            "n must be a whole number of failures, 0 or more, on every "
            f"c = 2 row; {int(np.sum(bad & counted))} row(s) are not"
        )
    if np.any((counts != 1) & ~counted):
        raise ValueError(
            "n is the number of failures counted on a c = 2 row; a c = 0 "
            "row is one failure at its end and a c = 1 row none, so n "
            f"must be 1 there ({int(np.sum((counts != 1) & ~counted))} "
            "row(s) are not). For failures recorded only as a number "
            "within an interval, give the row c = 2."
        )
    return counts


def handle_recurrent_tvc(
    i: npt.ArrayLike,
    xl: npt.ArrayLike,
    xr: npt.ArrayLike,
    c: npt.ArrayLike,
    Z: npt.ArrayLike,
    n: npt.ArrayLike | None = None,
) -> dict[str, npt.NDArray]:
    """
    Validate start-stop rows of repairable items and return them sorted by
    item and start time, as the dict of arrays (``i``, ``xl``, ``xr``,
    ``c``, ``n``, ``Z``) the time-varying likelihoods read.

    The arguments are those of the fitters' ``fit_tvc``. Unlike the life
    regressions' ``handle_tvc``
    (:mod:`surpyval.univariate.regression.proportional_hazards.tvc`), any
    row of an item may end in a failure.

    Examples
    --------
    >>> from surpyval.recurrent.regression.tvc import handle_recurrent_tvc
    >>> rows = handle_recurrent_tvc(
    ...     i=[2, 1, 1], xl=[0, 4, 0], xr=[5, 9, 4], c=[1, 1, 0],
    ...     Z=[0.0, 1.0, 0.0],
    ... )
    >>> rows["xl"]
    array([0., 4., 0.])
    >>> rows["c"]
    array([0, 1, 1])
    """
    ident = np.asarray(i)
    if ident.ndim != 1:
        raise ValueError("i must be one-dimensional: the item of each row")
    nrow = ident.shape[0]
    if nrow == 0:
        raise ValueError("no interval rows were supplied")
    try:
        xl = np.asarray(xl, dtype=float)
        xr = np.asarray(xr, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "xl and xr must be numeric: the start and end of each row's "
            "interval (xl, xr]"
        ) from None
    for name, arr in (("xl", xl), ("xr", xr)):
        if arr.ndim != 1 or arr.shape[0] != nrow:
            raise ValueError(
                f"i and {name} must have the same length; {name} has "
                f"shape {arr.shape}, expected {nrow} rows"
            )
    flags = _as_flags(c, nrow)
    counts = _as_counts(n, flags, nrow)
    Z = _as_covariates(Z, nrow, "interval")
    if not (np.isfinite(xl).all() and np.isfinite(xr).all()):
        raise ValueError(
            "xl and xr must be finite: every row is an interval (xl, xr] "
            "of an item's history, and a missing or infinite end leaves "
            "the item's time at risk unknown. Remove the item, or give "
            "the interval's ends."
        )
    if np.any(xl >= xr):
        raise ValueError(
            "every interval must have xl < xr (a row covers the "
            f"open-closed window (xl, xr]); {int(np.sum(xl >= xr))} "
            "row(s) do not"
        )
    if not np.isfinite(Z).all():
        raise ValueError(
            "Z has a missing or infinite value: each row is part of an "
            "item's history, so it cannot be dropped without removing "
            "the item from risk over that interval. Give the covariates "
            "in force on every interval, or remove the item."
        )
    order = np.lexsort((xl, ident))
    ident, xl, xr = ident[order], xl[order], xr[order]
    flags, counts, Z = flags[order], counts[order], Z[order]
    same_item = ident[1:] == ident[:-1]
    overlap = same_item & (xl[1:] < xr[:-1] - 1e-12)
    if overlap.any():
        item = _named(ident[1:][overlap][0])
        raise ValueError(
            f"item {item} has overlapping intervals; the (xl, xr] rows "
            "of an item must not overlap (they may touch, or leave a gap "
            "in which the item is not at risk)"
        )
    return {
        "i": ident,
        "xl": xl,
        "xr": xr,
        "c": flags,
        "n": counts,
        "Z": Z,
    }


def handle_recurrent_tvc_timeline(
    i: npt.ArrayLike,
    x: npt.ArrayLike,
    Z: npt.ArrayLike,
    c: npt.ArrayLike,
    n: npt.ArrayLike | None = None,
) -> tuple[
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
    npt.NDArray,
]:
    """
    Convert each item's covariate *timeline* into the start-stop rows
    ``fit_tvc`` takes: ``(i, xl, xr, c, Z, n)``.

    Each row of a timeline is something that happened to an item at time
    ``x``: its entry (its first row), a failure (``c = 0``), a change of
    covariates or the end of its record (``c = 1``), or a count of the
    failures since its previous row (``c = 2``, with ``n``). ``Z`` on a
    row holds from that time until the item's next row. So an item with
    ``m`` rows gives ``m - 1`` intervals: the one ending at row ``k`` runs
    from row ``k - 1``'s time, with row ``k - 1``'s covariates and row
    ``k``'s ``c`` and ``n``.

    Examples
    --------
    >>> from surpyval.recurrent.regression.tvc import (
    ...     handle_recurrent_tvc_timeline,
    ... )
    >>> i, xl, xr, c, Z, n = handle_recurrent_tvc_timeline(
    ...     i=[1, 1, 1, 1], x=[0, 3, 5, 9], Z=[0.0, 0.0, 1.0, 1.0],
    ...     c=[1, 0, 1, 1],
    ... )
    >>> xl, xr
    (array([0., 3., 5.]), array([3., 5., 9.]))
    >>> c, Z.ravel()
    (array([0, 1, 1]), array([0., 0., 1.]))
    """
    ident = np.asarray(i)
    if ident.ndim != 1:
        raise ValueError("i must be one-dimensional: the item of each row")
    nrow = ident.shape[0]
    if nrow == 0:
        raise ValueError("no timeline rows were supplied")
    try:
        x = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        raise ValueError(
            "x must be numeric: the time of each timeline row"
        ) from None
    if x.ndim != 1 or x.shape[0] != nrow:
        raise ValueError(
            "i and x must have the same length; x has shape "
            f"{x.shape}, expected {nrow} rows"
        )
    if not np.isfinite(x).all():
        raise ValueError(
            "x (the timeline times) must be finite: a missing time "
            "leaves the item's history unknown. Remove the item, or give "
            "the time."
        )
    flags = _as_flags(c, nrow)
    counts = _as_counts(n, flags, nrow)
    Z = _as_covariates(Z, nrow, "timeline row")

    order = np.lexsort((x, ident))
    ident, x, flags = ident[order], x[order], flags[order]
    counts, Z = counts[order], Z[order]
    first = np.r_[True, ident[1:] != ident[:-1]]
    last = np.r_[first[1:], True]
    if np.any(first & last):
        item = _named(ident[first & last][0])
        raise ValueError(
            f"item {item} has a single timeline row; an item needs at "
            "least its entry row and one later row (a failure, a change "
            "or the end of its record) to define an interval"
        )
    if np.any(first & (flags != 1)):
        item = _named(ident[first & (flags != 1)][0])
        raise ValueError(
            f"item {item}: its first timeline row has c = "
            f"{int(flags[first & (flags != 1)][0])}, but an item's first "
            "row is its entry, the time its record starts, and holds no "
            "failure: give it c = 1 (add an entry row before the first "
            "failure if the record starts earlier)"
        )
    later = ~first
    if np.any(x[later] <= x[:-1][later[1:]]):
        at = np.flatnonzero(later)[x[later] <= x[:-1][later[1:]]][0]
        raise ValueError(
            f"item {_named(ident[at])} has two timeline rows at the same time "
            f"({x[at]:g}); each item's rows must have strictly increasing "
            "x. Put a covariate change that coincides with a failure on "
            "the failure's row."
        )
    Z_held = Z[:-1][later[1:]]
    if not np.isfinite(Z_held).all():
        raise ValueError(
            "Z has a missing or infinite value on a timeline row that is "
            "not an item's last: each row's covariates hold until the "
            "item's next row, so they cannot be dropped. Give the "
            "covariates in force from every row, or remove the item."
        )
    return (
        ident[later],
        x[:-1][later[1:]],
        x[later],
        flags[later],
        Z_held,
        counts[later],
    )


def tvc_neg_ll(
    rows: dict[str, npt.NDArray],
    cif: Callable[..., Any],
    log_iif: Callable[..., Any],
    k_dist: int,
) -> Callable[[Any], Any]:
    """The negative log-likelihood of start-stop ``rows`` (from
    :func:`handle_recurrent_tvc`) as a function of ``[*baseline
    parameters, *coefficients]``, for a baseline with cumulative intensity
    ``cif(x, *parameters)`` and log intensity ``log_iif(x, *parameters)``
    and ``k_dist`` parameters. Written with autograd's numpy, so it is
    differentiable where the baseline's functions are."""
    xl, xr, Z = rows["xl"], rows["xr"], rows["Z"]
    event = rows["c"] == 0
    counted = rows["c"] == 2
    x_event = xr[event]
    n_counted = rows["n"][counted]
    # A count of 0 adds nothing but its exposure, whatever the interval.
    some = n_counted > 0
    log_factorials = gammaln(n_counted + 1).sum()

    def neg_ll(params: Any) -> Any:
        theta, beta = params[:k_dist], params[k_dist:]
        eta = anp.dot(Z, beta)
        delta = cif(xr, *theta) - cif(xl, *theta)
        ll = -anp.sum(anp.exp(eta) * delta)
        ll = ll + anp.sum(log_iif(x_event, *theta) + eta[event])
        ll = ll + anp.sum(
            n_counted[some]
            * (anp.log(delta[counted][some]) + eta[counted][some])
        )
        return -(ll - log_factorials)

    return neg_ll


def _hpp_cif(x: Any, log_rate: Any) -> Any:
    return anp.exp(log_rate) * x


def _hpp_log_iif(x: Any, log_rate: Any) -> Any:
    return log_rate + 0.0 * x


def hpp_tvc_neg_ll(rows: dict[str, npt.NDArray]) -> Callable[[Any], Any]:
    """:func:`tvc_neg_ll` for the constant baseline, in ``[log(rate),
    *coefficients]``, the space the HPP fit searches."""
    return tvc_neg_ll(rows, _hpp_cif, _hpp_log_iif, 1)


def nhpp_tvc_neg_ll(
    rows: dict[str, npt.NDArray], dist: Any
) -> Callable[[Any], Any]:
    """:func:`tvc_neg_ll` for the counting-process baseline ``dist``, in
    its natural parameters followed by the coefficients."""
    return tvc_neg_ll(rows, dist.cif, dist.log_iif, len(dist.parameter_names))


def count_events(rows: dict[str, npt.NDArray]) -> float:
    """The number of failures the rows hold: the ``c = 0`` rows and the
    counts of the ``c = 2`` rows (the sample size of BIC, as
    :func:`surpyval.recurrent.inference.bic_sample_size`)."""
    c, n = rows["c"], rows["n"]
    return float((c == 0).sum() + n[c == 2].sum())


def check_tvc_rows(rows: dict[str, npt.NDArray], dist: Any) -> None:
    """Refuse rows a proportional-intensity model cannot be fitted to: no
    failure at all (the likelihood only rewards a lower intensity), or a
    time outside the baseline's support."""
    name = getattr(dist, "name", type(dist).__name__)
    if count_events(rows) == 0:
        raise ValueError(
            "The data has no failures (no c = 0 row and no c = 2 row with "
            f"a count above 0), so the {name} intensity cannot be "
            "estimated."
        )
    lower, upper = getattr(dist, "support", (-np.inf, np.inf))
    if np.any(rows["xl"] < lower) or np.any(rows["xr"] > upper):
        raise ValueError(
            f"The {name} intensity is defined on [{lower}, {upper}], but "
            "the data has intervals (xl, xr] reaching outside it."
        )


def rows_as_event_times(
    rows: dict[str, npt.NDArray],
) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """The rows as the ``x``, ``i``, ``c`` and per-row ``tl`` of an
    ordinary recurrent-event fit, covariates left out: each item's
    failures, a count's failures spread evenly over its interval, and the
    end of its record where that is not a failure. Gaps in an item's
    record are ignored, so this is only a place to *start* a search from
    (the covariate-free fit of the baseline)."""
    ident, xl, xr, c, n = (rows[k] for k in ("i", "xl", "xr", "c", "n"))
    x_out: list = []
    i_out: list = []
    c_out: list = []
    tl_out: list = []
    starts = np.flatnonzero(np.r_[True, ident[1:] != ident[:-1]])
    ends = np.r_[starts[1:], ident.shape[0]]
    for item, (a, b) in enumerate(zip(starts, ends)):
        times = [xr[a:b][c[a:b] == 0]]
        for k in np.flatnonzero(c[a:b] == 2):
            m = int(n[a + k])
            times.append(
                xl[a + k]
                + (np.arange(m) + 0.5) / max(m, 1) * (xr[a + k] - xl[a + k])
            )
        failures = np.sort(np.concatenate(times))
        closes = c[b - 1] != 0
        x_out += [*failures] + ([xr[b - 1]] if closes else [])
        c_out += [0] * failures.size + ([1] if closes else [])
        count = failures.size + int(closes)
        i_out += [item] * count
        tl_out += [xl[a]] * count
    return (
        np.asarray(x_out, dtype=float),
        np.asarray(i_out),
        np.asarray(c_out, dtype=int),
        np.asarray(tl_out, dtype=float),
    )


def mark_tvc(model: Any, rows: dict[str, npt.NDArray]) -> Any:
    """Record on a fitted ``model`` that it was fitted to time-varying
    covariates: ``is_tvc``, the rows as fitted, and no ``data`` (it has no
    one covariate row per item for the methods that read ``data``)."""
    model.is_tvc = True
    model._tvc_fit_rows = rows
    model.n_items = int(np.unique(rows["i"]).shape[0])
    return model


class ProportionalIntensityTVCFitMixin:
    """``fit_tvc_timeline`` and the DataFrame forms of the time-varying
    fit, for a proportional-intensity fitter that has ``fit_tvc(i, xl,
    xr, c, Z, n, ...)``."""

    if TYPE_CHECKING:

        def fit_tvc(self, *args: Any, **kwargs: Any) -> Any: ...

    def fit_tvc_timeline(
        self,
        i: npt.ArrayLike,
        x: npt.ArrayLike,
        Z: npt.ArrayLike,
        c: npt.ArrayLike,
        n: npt.ArrayLike | None = None,
        **fit_options: Any,
    ) -> Any:
        """
        Fit the model from each item's covariate *timeline*: a log with
        one row each time something happens to an item.

        The timeline is expanded to start-stop intervals and fitted as
        :meth:`fit_tvc`, so the fit is identical to that of the
        equivalent start-stop rows. The arguments are those of the life
        regressions' ``fit_tvc_timeline``; an item here may fail many
        times, so ``c`` is read on every row, not only an item's last.

        Parameters
        ----------
        i : array_like
            The item of each timeline row.
        x : array_like
            The time of each row, strictly increasing within an item: the
            first is the item's entry (the start of its record), the last
            the end of its record.
        Z : array_like
            The covariate row in force from ``x`` until the item's next
            row (a 1-D array is a single covariate); the value on an
            item's last row is ignored. A missing value on any other row
            raises a ``ValueError``.
        c : array_like
            What happened at ``x``: ``0`` a failure, ``1`` no failure (the
            entry, a change of covariates or the end of the record), ``2``
            a count ``n`` of failures since the item's previous row. An
            item's first row is its entry and must be ``1``.
        n : array_like, optional
            The number of failures on a ``c = 2`` row; 1 (the default) on
            the others.
        **fit_options
            Passed to :meth:`fit_tvc` (``init``, ``unit``, and
            ``baseline`` for the NHPP fitter).

        Returns
        -------
        ProportionalIntensityModel
            The fitted model, with ``is_tvc`` set.

        Examples
        --------
        Two pumps. The first runs at low load until month 10 and at high
        load after it; the second the other way round. Each row is an
        entry, a failure, a change of load or the end of the record:

        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        >>> x = [0, 6, 10, 13, 17, 20, 0, 3, 7, 10, 16, 20]
        >>> c = [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
        >>> load = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        >>> model = ProportionalIntensityHPP.fit_tvc_timeline(i, x, load, c)
        >>> model.params.round(4), model.coeffs.round(4)
        (array([0.1]), array([0.6931]))

        Two failures in 20 pump-months at low load and four in 20 at high
        load: a rate of 0.1 a month, doubled (``exp(0.6931)``) by high
        load.
        """
        i_ss, xl, xr, c_ss, Z_ss, n_ss = handle_recurrent_tvc_timeline(
            i, x, Z, c, n
        )
        return self.fit_tvc(i_ss, xl, xr, c_ss, Z_ss, n_ss, **fit_options)

    def _fit_tvc_frame(
        self,
        fit: Callable[..., Any],
        df: pd.DataFrame,
        columns: dict[str, str | None],
        Z_cols: str | list[str],
        fit_options: dict[str, Any],
    ) -> Any:
        """Call the array fit ``fit`` with the columns of ``df`` that
        ``columns`` names (``{argument: column}``) and the covariates
        ``Z_cols``, which name the model's coefficients."""
        df = require_frame(df)
        for name in (*columns, "Z"):
            if name in fit_options:
                suffix = "_cols" if name == "Z" else "_col"
                raise TypeError(
                    f"{fit.__name__}_from_df() got an unexpected keyword "
                    f"argument '{name}'; name the column with "
                    f"'{name}{suffix}'"
                )
        arrays: dict[str, Any] = {}
        for arg, column in columns.items():
            if column is None:
                continue
            arrays[arg] = frame_column(
                df, column, f"{arg}_col", time=arg in ("x", "xl", "xr")
            )
        names = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        from surpyval.utils.covariates import numeric_columns
        from surpyval.utils.dataframe import check_columns

        check_columns(df, Z_cols=names)
        arrays["Z"] = numeric_columns(df, names)
        with covariate_columns(names):
            model = fit(**arrays, **fit_options)
        model.feature_names = [str(name) for name in names]
        return model

    def fit_tvc_from_df(
        self,
        df: pd.DataFrame,
        i_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: str | list[str],
        n_col: str | None = None,
        **fit_options: Any,
    ) -> Any:
        """Fit start-stop time-varying-covariate rows from a DataFrame.

        ``i_col``, ``xl_col``, ``xr_col``, ``c_col`` and ``n_col`` name
        the columns passed to :meth:`fit_tvc` as ``i``, ``xl``, ``xr``,
        ``c`` and ``n``, and ``Z_cols`` (a column name or a list of them)
        the covariates, whose names then name the coefficients and are
        kept as the model's ``feature_names``. Other keyword arguments go
        to :meth:`fit_tvc` (``init``, ``unit``, and ``baseline`` for the
        NHPP fitter). Returns the fitted ``ProportionalIntensityModel``.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> df = pd.DataFrame({
        ...     "pump": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        ...     "start": [0, 6, 10, 13, 17, 0, 3, 7, 10, 16],
        ...     "stop": [6, 10, 13, 17, 20, 3, 7, 10, 16, 20],
        ...     "c": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        ...     "high_load": [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        ... })
        >>> model = ProportionalIntensityHPP.fit_tvc_from_df(
        ...     df, "pump", "start", "stop", "c", Z_cols="high_load"
        ... )
        >>> model.parameter_names
        ['lambda', 'high_load']
        >>> model.coeffs.round(4)
        array([0.6931])
        """
        return self._fit_tvc_frame(
            self.fit_tvc,
            df,
            {"i": i_col, "xl": xl_col, "xr": xr_col, "c": c_col, "n": n_col},
            Z_cols,
            fit_options,
        )

    def fit_tvc_timeline_from_df(
        self,
        df: pd.DataFrame,
        i_col: str,
        x_col: str,
        Z_cols: str | list[str],
        c_col: str,
        n_col: str | None = None,
        **fit_options: Any,
    ) -> Any:
        """Fit a covariate timeline from a DataFrame.

        ``i_col``, ``x_col``, ``c_col`` and ``n_col`` name the columns
        passed to :meth:`fit_tvc_timeline` as ``i``, ``x``, ``c`` and
        ``n``, and ``Z_cols`` (a column name or a list of them) the
        covariates in force from each row, whose names then name the
        coefficients and are kept as the model's ``feature_names``. Other
        keyword arguments go to :meth:`fit_tvc` (``init``, ``unit``, and
        ``baseline`` for the NHPP fitter). Returns the fitted
        ``ProportionalIntensityModel``.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import ProportionalIntensityHPP
        >>> log = pd.DataFrame({
        ...     "pump": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        ...     "month": [0, 6, 10, 13, 17, 20, 0, 3, 7, 10, 16, 20],
        ...     "c": [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        ...     "high_load": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ... })
        >>> model = ProportionalIntensityHPP.fit_tvc_timeline_from_df(
        ...     log, "pump", "month", "high_load", "c"
        ... )
        >>> model.coeffs.round(4)
        array([0.6931])
        """
        return self._fit_tvc_frame(
            self.fit_tvc_timeline,
            df,
            {"i": i_col, "x": x_col, "c": c_col, "n": n_col},
            Z_cols,
            fit_options,
        )
