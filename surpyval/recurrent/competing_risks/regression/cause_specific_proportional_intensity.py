"""
Cause-specific proportional-intensity regression: recurrent events of
several mutually exclusive types (marks ``e``) on items described by
covariates.

This is the regression counterpart of :class:`CauseSpecificNHPP`. Each
event type has its own baseline intensity and its own coefficients,

.. math::
    \\lambda_k(t \\mid Z) = \\lambda_{0k}(t)\\, e^{\\beta_k' Z},

so a covariate can raise the rate of one type of event and leave another
alone. For a marked Poisson process the types are **independent thinned
Poisson processes**, so the likelihood factorises over them: each type's
parameters are the maximum-likelihood proportional-intensity fit to that
type's events over every item's *full* observation window, the other
types' events ignored, exactly as a censored period would be. So each
type is fitted by the ordinary proportional-intensity fitters
(``ProportionalIntensityHPP`` / ``ProportionalIntensityNHPP``) and is an
ordinary :class:`ProportionalIntensityModel`; nothing here has a likelihood
of its own.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from surpyval.recurrent.competing_risks.parametric.cause_specific_nhpp import (
    CauseSpecificNHPP,
    cause_rows,
    event_labels,
    parameter_interval,
)
from surpyval.recurrent.inference import check_alpha_ci
from surpyval.recurrent.parametric import HPP, Duane
from surpyval.recurrent.parametric.counting_process import CountingProcess
from surpyval.recurrent.regression.hpp_proportional_intensity import (
    ProportionalIntensityHPP,
)
from surpyval.recurrent.regression.nhpp_proportional_intensity import (
    ProportionalIntensityNHPP,
)
from surpyval.recurrent.regression.proportional_intensity import (
    ProportionalIntensityModel,
)
from surpyval.recurrent.regression.tvc import (
    ProportionalIntensityTVCFitMixin,
    handle_recurrent_tvc,
    handle_recurrent_tvc_timeline,
)
from surpyval.recurrent.serialisation import intensity_dist_by_name
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
    to_native,
)
from surpyval.univariate.competing_risks.labels import (
    label_from_native,
    label_mask,
    ordered_labels,
)
from surpyval.univariate.information_criteria import corrected_aic
from surpyval.univariate.regression._aliasing import (
    collect_aliased,
    covariate_columns,
    dataframe_covariates,
    warn_collected,
)
from surpyval.univariate.regression.regression_data import named_covariates
from surpyval.utils.data_formats import is_missing_event
from surpyval.utils.dataframe import (
    call_fit,
    frame_column,
    frame_columns,
    refuse_recurrent_column_names,
    require_frame,
)
from surpyval.utils.event_history_data import (
    accepts_event_history_data,
    refuses_unmodelled_data,
)
from surpyval.utils.linalg import delta_method_se, log_transformed_cb
from surpyval.utils.no_maximum import combined_maximum
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    refuse_right_truncation,
)
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.units import takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_option,
    unknown_cause_error,
)
from surpyval.utils.warnings import warn_no_covariance

_NAME = "CauseSpecificProportionalIntensity"

# What each method this model does not have yet says to use instead
# (principles 2 and 22). ``{model}`` is "model.models[event]", one event
# type's own fitted proportional-intensity model.
_PER_TYPE = (
    "Each event type is an ordinary proportional-intensity model, a "
    "Poisson process of its own: call {name} on model.models[event] for "
    "one type"
)
_NOT_AVAILABLE = {
    "diagnose": _PER_TYPE + ", and compare candidate baselines and "
    "covariates over every type with aic() and bic().",
    "residuals": _PER_TYPE + ".",
    "cramer_von_mises": _PER_TYPE + ".",
    "trend_test": _PER_TYPE
    + ", or read each type's trend parameter and its interval from "
    "summary().",
    "plot": _PER_TYPE
    + ", or plot cif(x, Z, event) and total_cif(x, Z) with their bounds "
    "(cif_cb, total_cif_cb) against x at a covariate row of interest.",
    "count_terminated_simulation": _PER_TYPE
    + " (at one covariate vector Z); the types are independent, so their "
    "simulated events can be pooled.",
    "count_terminated_simulation_data": _PER_TYPE
    + " (at one covariate vector Z); the types are independent, so their "
    "simulated events can be pooled.",
    "time_terminated_simulation": _PER_TYPE
    + " (at one covariate vector Z); the types are independent, so their "
    "simulated events can be pooled.",
    "time_terminated_simulation_data": _PER_TYPE
    + " (at one covariate vector Z); the types are independent, so their "
    "simulated events can be pooled.",
    "mcf": "The expected number of events is exact for this model: use "
    "cif(x, Z, event) for one type and total_cif(x, Z) for every type "
    "(cif_tvc and total_cif_tvc along a covariate schedule).",
    "forecast": "For the expected number of events of an item of age a "
    "over the next h, use total_cif_tvc(a + h, Z, xl=[a], given=a) (every "
    "type) or cif_tvc(a + h, Z, event, xl=[a], given=a) (one type), with "
    "bounds from total_cb_tvc and cb_tvc; for one type's forecast with "
    "its prediction interval, surpyval.forecast(model.models[event], "
    "age, horizon, Z=Z).",
    "growth": "The baseline's trend does not depend on the covariates: "
    "read each type's trend parameter (beta of CrowAMSAA and CoxLewis, "
    "alpha of Duane) and its interval from summary(), and each type's "
    "intensity at a covariate row from iif(x, Z, event) and iif_cb.",
    "mtbf": "It is the reciprocal of the intensity: 1 / iif(x, Z, event).",
    "mtbf_cb": "It is the reciprocal of the bounds on the intensity: "
    "1 / iif_cb(x, Z, event), with the lower and upper bounds swapped.",
    "total_mtbf": "It is the reciprocal of the total intensity: "
    "1 / total_iif(x, Z).",
    "total_mtbf_cb": "It is the reciprocal of the bounds on the total "
    "intensity: 1 / total_iif_cb(x, Z), with the lower and upper bounds "
    "swapped.",
}


def _not_available(name: str) -> Callable[..., Any]:
    """A method that refuses, saying what to use instead."""
    instead = _NOT_AVAILABLE[name].format(name=name + "()")

    def refuse(self: Any, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{name}() is not available on a {_NAME} model yet. {instead}"
        )

    refuse.__name__ = name
    refuse.__qualname__ = f"{_NAME}.{name}"
    refuse.__doc__ = (
        f"Not available on this model yet: raises a "
        f"``NotImplementedError`` that says what to use instead. "
        f"{instead}\n\n"
        "        Examples\n"
        "        --------\n"
        "        >>> from surpyval.recurrent import (\n"
        "        ...     HPP, CauseSpecificProportionalIntensity,\n"
        "        ... )\n"
        "        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, "
        "38, 60]\n"
        "        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]\n"
        "        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]\n"
        '        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", '
        "None,\n"
        '        ...      "a", "a", "b", None]\n'
        "        >>> harsh = [0] * 5 + [1] * 9\n"
        "        >>> model = CauseSpecificProportionalIntensity.fit(\n"
        "        ...     x, harsh, i=i, c=c, e=e, baseline=HPP\n"
        "        ... )\n"
        "        >>> try:\n"
        f"        ...     model.{name}()\n"
        "        ... except NotImplementedError as error:\n"
        f"        ...     print(str(error)[:{len(name) + 19}])\n"
        f"        {name}() is not available\n"
    )
    return refuse


def _fitter(baseline: Any) -> tuple[Any, dict]:
    """The single-type fitter for ``baseline`` and the options that name
    the baseline to it."""
    if isinstance(baseline, type(HPP)):
        return ProportionalIntensityHPP, {}
    if not isinstance(baseline, CountingProcess):
        raise ValueError(
            "baseline must be an intensity model from surpyval.recurrent "
            "(HPP, Duane, CrowAMSAA or CoxLewis), the baseline of every "
            f"event type; got {baseline!r}"
        )
    return ProportionalIntensityNHPP, {"baseline": baseline}


def _marks(e: Any, nrow: int | None = None) -> np.ndarray:
    """``e`` as an object array, one mark per row, a missing value (``NaN``,
    pandas ``NA``) as ``None``."""
    if e is None:
        raise ValueError(
            "e (the event type of each failure) is required for a "
            "cause-specific model: give the type of each failure row, and "
            "None on the other rows."
        )
    try:
        values = list(e)
    except TypeError:
        raise ValueError(
            "e must hold one event type per row (None on a row that is "
            f"not a failure); got {e!r}"
        ) from None
    out = np.empty(len(values), dtype=object)
    for k, value in enumerate(values):
        out[k] = None if is_missing_event(value) else value
    if nrow is not None and out.shape[0] != nrow:
        raise ValueError(
            "i and e must have the same length; e has "
            f"{out.shape[0]} value(s), expected {nrow} rows"
        )
    return out


class _TimeVaryingFrame(ProportionalIntensityTVCFitMixin):
    """The single-type models' reader of a DataFrame of time-varying
    rows, used for this model's ``_from_df`` forms."""


_FRAME = _TimeVaryingFrame()


class CauseSpecificProportionalIntensity(SerialisableMixin):
    """
    Proportional-intensity regression for a recurrent process with
    several event types (failure modes): each type has its own baseline
    intensity and its own covariate coefficients,

    .. math::
        \\lambda_k(t \\mid Z) = \\lambda_{0k}(t)\\, e^{\\beta_k' Z}.

    It is the regression counterpart of :class:`CauseSpecificNHPP`. The
    types of a marked Poisson process are independent, so each is fitted
    by the single-type regression (``ProportionalIntensityHPP`` for
    ``baseline=HPP``, else ``ProportionalIntensityNHPP``) to its own
    events over every item's whole observation window. The fitted
    per-type models, each an ordinary
    :class:`~surpyval.recurrent.regression.proportional_intensity.ProportionalIntensityModel`,
    are in ``models[event]``; the methods here take the type as ``event``
    after the covariates ``Z``, and ``total_`` in front sums over the
    types.

    Attributes
    ----------
    event_types : list
        The event types, in a fixed order.
    models : dict
        Each type's fitted ``ProportionalIntensityModel``.
    baseline : CountingProcess
        The baseline intensity model of every type.
    feature_names : list of str or None
        The covariates' column names (a DataFrame ``Z``, or the
        ``_from_df`` forms), which name the coefficients; ``None`` for an
        array ``Z``, whose coefficients are ``coef_0``, ``coef_1``, ...
    is_tvc : bool
        Whether the model was fitted to covariates that change over time
        (``fit_tvc``, ``fit_tvc_timeline``).

    Examples
    --------
    Four pumps watched for 60 months, two of them (3 and 4) in harsh
    service. Each failure is of the seal (``"a"``) or the bearing
    (``"b"``); the ``c=1`` rows end each pump's record:

    >>> from surpyval.recurrent import HPP, CauseSpecificProportionalIntensity
    >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
    >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
    >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
    >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
    ...      "a", "a", "b", None]
    >>> harsh = [0] * 5 + [1] * 9
    >>> model = CauseSpecificProportionalIntensity.fit(
    ...     x, harsh, i=i, c=c, e=e, baseline=HPP
    ... )
    >>> model
    Cause-specific proportional intensity (Constant baseline) with causes: \
['a', 'b']

    Seal failures: 2 in the 120 pump-months of normal service and 5 in
    the 120 of harsh service, a rate ratio of 2.5. Bearing failures: 1
    and 2, a rate ratio of 2:

    >>> model.coefficients()["exp(coef)"].round(2)
    event  covariate
    a      coef_0       2.5
    b      coef_0       2.0
    Name: exp(coef), dtype: float64

    The expected number of failures of a pump in harsh service by month
    60, of each type and in total:

    >>> model.cif(60, [1], "a").round(2), model.cif(60, [1], "b").round(2)
    (np.float64(2.5), np.float64(1.0))
    >>> model.total_cif(60, [1]).round(2)
    np.float64(3.5)
    """

    # Populated by the fit classmethods; declared for the type checker.
    df: Any
    data: Any
    event_types: list
    models: dict
    baseline: Any
    feature_names: list[str] | None = None
    # (documented under Attributes above; such a fit keeps no ``data``)
    is_tvc: bool = False
    # A restored model's saved inference: each type's parameter
    # covariance (None where its fit had none).
    _covariances: dict

    # -- fitting -----------------------------------------------------------

    @classmethod
    def _assemble(
        cls, event_types: list, models: dict, baseline: Any
    ) -> CauseSpecificProportionalIntensity:
        out = cls()
        out.event_types = list(event_types)
        out.models = models
        out.baseline = baseline
        out.feature_names = models[event_types[0]].feature_names
        return out

    @classmethod
    @refuses_unmodelled_data
    def fit_from_recurrent_data(
        cls,
        data: Any,
        baseline: Any = Duane,
        init: ArrayLike | None = None,
    ) -> CauseSpecificProportionalIntensity:
        """
        Fit from prepared :class:`EventHistoryData` carrying event-type
        marks ``e`` and covariates ``Z``, as built by
        ``surpyval.handle_xicn``. :meth:`fit` builds one from its arrays
        and calls this.

        Parameters
        ----------
        data : EventHistoryData
            Recurrent data with event types (``data.e``) and covariates
            (``data.Z``), exact events (``c=0``) and end-of-observation
            rows (``c=1``).
        baseline : CountingProcess, optional
            The baseline intensity model of every type, as for
            :meth:`fit`.
        init : array_like, optional
            Initial parameters for each type's search, as for :meth:`fit`.

        Returns
        -------
        CauseSpecificProportionalIntensity

        Examples
        --------
        >>> from surpyval import handle_xicn
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> data = handle_xicn(x, i, c, e=e, Z=harsh)
        >>> model = CauseSpecificProportionalIntensity.fit_from_recurrent_data(
        ...     data, baseline=HPP
        ... )
        >>> model.event_types
        ['a', 'b']
        """
        fitter, options = _fitter(baseline)
        if data.e is None:
            raise ValueError(
                "The data has no event-type marks; pass `e` (the event "
                "type of each failure row, None on the other rows) to fit "
                "a cause-specific model."
            )
        if data.Z is None:
            raise ValueError(
                "EventHistoryData has no covariates; pass `Z` (one row "
                "per row of x, the same on every row of an item), or use "
                "CauseSpecificNHPP for event types without covariates."
            )
        if data.x.ndim != 1:
            raise ValueError(
                f"{_NAME} needs exact (1D) event times: counts of events "
                "in an interval (c=2, c=-1) carry no event type per "
                "event. Fit each type's own counts with "
                "ProportionalIntensityNHPP.fit instead."
            )
        # Selection on the first failure of any type, which a model of
        # each type apart cannot account for (#924)
        refuse_right_truncation(data, _NAME)
        unsupported = sorted(set(data.c.tolist()) - {0, 1})
        if unsupported:
            raise ValueError(
                f"{_NAME} takes exact events (c=0) and end-of-observation "
                f"rows (c=1) only; c has the code(s) {unsupported}."
            )
        windows = {
            item: CauseSpecificNHPP._item_window(data, item)
            for item in data.ids
        }
        Z = np.asarray(data.Z, dtype=float)
        Z_of = {item: Z[data.i == item][0] for item in data.ids}
        models = {}
        # One aliasing warning for the fits of every type (principle 22)
        with collect_aliased() as aliased:
            for cause in data.event_types:
                cx, ci, cc, ctl = cause_rows(data, cause, windows)
                cause_data = handle_xicn(
                    cx,
                    ci,
                    cc,
                    tl=ctl,
                    Z=np.array([Z_of[item] for item in ci]),
                )
                models[cause] = fitter.fit_from_recurrent_data(
                    cause_data, init=init, **options
                )
        warn_collected(aliased, "in the fit of each event type")
        out = cls._assemble(data.event_types, models, baseline)
        out.data = data
        return out

    @classmethod
    @dataframe_covariates
    @takes_unit
    @accepts_event_history_data
    def fit(
        cls,
        x: ArrayLike,
        Z: ArrayLike,
        i: ArrayLike | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        e: ArrayLike | None = None,
        tl: ArrayLike | None = None,
        tr: ArrayLike | None = None,
        baseline: Any = Duane,
        init: ArrayLike | None = None,
    ) -> CauseSpecificProportionalIntensity:
        """
        Fit the model to event times with their types and each item's
        (fixed) covariates.

        The arguments are those of the single-type regressions' ``fit``
        (``ProportionalIntensityNHPP.fit``), with the event type ``e`` of
        each row as :meth:`CauseSpecificNHPP.fit` takes it. Each type is
        fitted to its own events over every item's whole observation
        window, the other types' events left out.

        Parameters
        ----------
        x : array_like
            The event (and end-of-observation) times, pooled over items.
        Z : array_like or dict
            Covariates: a matrix with one row per row of ``x`` (a 1-D
            array is a single covariate), a DataFrame (its column names
            then name the coefficients) or a ``{item: covariates}`` dict.
            They describe the item, so they must be the same on every row
            of an item; for covariates that change during an item's life
            use :meth:`fit_tvc`.
        i : array_like, optional
            The item each row belongs to. Defaults to a single item.
        c : array_like, optional
            Censoring flag of each row: 0 an observed event, 1 the
            right-censored end of an item's observation. Counts of events
            (``c=2``, ``c=-1``) are refused: they carry no type per event.
        n : array_like, optional
            The number of events each row stands for; every row stands
            for one, so ``n > 1`` is refused (repeat the row for
            simultaneous events). Defaults to 1.
        e : array_like
            The event type (mark) of each row; ``None`` / ``NaN`` on the
            end-of-observation rows. A mark may be any hashable label.
            Required.
        tl : array_like or scalar, optional
            Left-truncation (delayed-entry) time of each item, as for
            :meth:`CauseSpecificNHPP.fit`.
        tr : array_like or scalar, optional
            Accepted for a uniform interface; a finite value (right
            truncation) is refused, as for :meth:`CauseSpecificNHPP.fit`:
            give the end of an item's observation as a ``c=1`` row.
        baseline : CountingProcess, optional
            The baseline intensity model of every type: ``Duane`` (the
            default, as for ``ProportionalIntensityNHPP``), ``CrowAMSAA``
            or ``CoxLewis`` from ``surpyval.recurrent``, or ``HPP`` for
            constant rates (each type is then fitted by
            ``ProportionalIntensityHPP``).
        init : array_like, optional
            Initial parameters for each type's search: the baseline
            parameters followed by the covariate coefficients.

        Returns
        -------
        CauseSpecificProportionalIntensity
            The fitted model, with each type's
            ``ProportionalIntensityModel`` in ``models[event]``.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ...     ProportionalIntensityHPP,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> seal = model.models["a"]
        >>> seal.params.round(4), seal.coeffs.round(4)
        (array([0.0167]), array([0.9163]))

        Type ``"a"``'s fit is the single-type regression of its events
        alone, each pump still watched to month 60:

        >>> alone = ProportionalIntensityHPP.fit(
        ...     [20, 60, 30, 60, 8, 15, 40, 60, 12, 25, 60],
        ...     [0] * 4 + [1] * 7,
        ...     i=[1, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4],
        ...     c=[0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        ... )
        >>> alone.params.round(4), alone.coeffs.round(4)
        (array([0.0167]), array([0.9163]))
        """
        data = handle_xicn(x, i, c, n, tl=tl, tr=tr, Z=Z, e=e)
        return cls.fit_from_recurrent_data(data, baseline=baseline, init=init)

    @classmethod
    @column_arguments("x", "i", "c", "n", "e", "tl", "tr")
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        Z_cols: str | list[str],
        e_col: str,
        i_col: str | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        **fit_options: Any,
    ) -> CauseSpecificProportionalIntensity:
        """
        Fit to an event log held in the columns of a
        :class:`pandas.DataFrame`, naming the columns to read. See
        :meth:`fit` for the meaning of each; ``Z_cols`` (a column name or
        a list of them) names the covariates, whose names then name the
        coefficients, and every other option of :meth:`fit`
        (``baseline``, ``init``, ``unit``) is passed to it.

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> log = pd.DataFrame({
        ...     "month": [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25,
        ...               38, 60],
        ...     "pump": [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4],
        ...     "c": [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        ...     "mode": ["a", "b", None, "a", None, "a", "a", "b", "a",
        ...              None, "a", "a", "b", None],
        ...     "harsh": [0] * 5 + [1] * 9,
        ... })
        >>> model = CauseSpecificProportionalIntensity.fit_from_df(
        ...     log, x_col="month", Z_cols="harsh", e_col="mode",
        ...     i_col="pump", c_col="c", baseline=HPP,
        ... )
        >>> model.feature_names
        ['harsh']
        >>> model.coefficients()["coef"].round(4)
        event  covariate
        a      harsh        0.9163
        b      harsh        0.6931
        Name: coef, dtype: float64
        """
        refuse_recurrent_column_names(fit_options, "Z")
        columns = {
            "e": e_col,
            "i": i_col,
            "c": c_col,
            "n": n_col,
            "tl": tl_col,
            "tr": tr_col,
        }
        df = require_frame(df)
        arrays = {
            key: frame_column(
                df, name, f"{key}_col", time=key in ("x", "tl", "tr")
            )
            for key, name in {"x": x_col, **columns}.items()
            if name is not None
        }
        arrays["Z"] = frame_columns(df, Z_cols, "Z_cols").astype(float)
        arg_names = {k: f"{k}_col" for k in arrays} | {"Z": "Z_cols"}
        names = [Z_cols] if isinstance(Z_cols, str) else list(Z_cols)
        # An aliasing warning names the columns, as do the coefficients.
        with covariate_columns(names):
            model = call_fit(cls, arrays, arg_names, fit_options)
        model.feature_names = [str(name) for name in names]
        model.df = df
        return model

    @classmethod
    @dataframe_covariates
    @takes_unit
    def fit_tvc(
        cls,
        i: ArrayLike,
        xl: ArrayLike,
        xr: ArrayLike,
        c: ArrayLike,
        Z: ArrayLike,
        n: ArrayLike | None = None,
        e: ArrayLike | None = None,
        baseline: Any = Duane,
        init: ArrayLike | None = None,
    ) -> CauseSpecificProportionalIntensity:
        """
        Fit the model to covariates that change over time, in start-stop
        format.

        Each row is one interval ``(xl, xr]`` of item ``i``'s history over
        which its covariate row ``Z`` is constant, exactly as the
        single-type regressions' ``fit_tvc``
        (``ProportionalIntensityNHPP.fit_tvc``) takes them, with the type
        ``e`` of the failure that ends a ``c = 0`` row. The covariates in
        force multiply each type's intensity by that type's own factor.
        Each type is fitted to the same rows with the rows that end in
        another type's failure read as ending in none (``c = 1``): for
        that type they are a change of row and nothing more.

        Parameters
        ----------
        i : array_like
            The item of each interval row.
        xl, xr : array_like
            The open-closed interval ``(xl, xr]`` of each row, in the
            item's own time (its age).
        c : array_like
            What ends the row at ``xr``: ``0`` a failure (of type ``e``),
            ``1`` no failure (a covariate change or the end of the item's
            record). A count of failures within a row (``c = 2``) is
            refused: it carries no type per failure.
        Z : array_like
            The covariate row in force on each interval (a 1-D array is a
            single covariate). A missing value raises a ``ValueError``.
        n : array_like, optional
            Accepted for a uniform interface; every row stands for one,
            so anything but 1 is refused.
        e : array_like
            The type of the failure ending each ``c = 0`` row; ``None`` /
            ``NaN`` on the other rows (a mark on a ``c = 1`` row is
            ignored). A failure row without a mark warns and counts
            towards no type, as in :meth:`fit`. Required.
        baseline : CountingProcess, optional
            The baseline intensity model of every type, as for
            :meth:`fit`.
        init : array_like, optional
            Initial parameters for each type's search: the baseline
            parameters followed by the covariate coefficients.

        Returns
        -------
        CauseSpecificProportionalIntensity
            The fitted model, with ``is_tvc`` set.

        Examples
        --------
        Two pumps watched for 20 months. The first runs at low load to
        month 10 and at high load after it; the second the other way
        round. Each failure is of the seal or the motor:

        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> i = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> xl = [0, 6, 10, 13, 17, 0, 3, 7, 10, 16]
        >>> xr = [6, 10, 13, 17, 20, 3, 7, 10, 16, 20]
        >>> c = [0, 1, 0, 0, 1, 0, 0, 1, 0, 1]
        >>> e = ["motor", None, "seal", "seal", None, "seal", "motor", None,
        ...      "seal", None]
        >>> high_load = [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
        >>> model = CauseSpecificProportionalIntensity.fit_tvc(
        ...     i, xl, xr, c, high_load, e=e, baseline=HPP
        ... )
        >>> model.is_tvc
        True

        Seal failures: one in the 20 pump-months at low load and three in
        the 20 at high load. Motor failures: one in each:

        >>> model.coefficients()["exp(coef)"].round(3)
        event  covariate
        motor  coef_0       1.0
        seal   coef_0       3.0
        Name: exp(coef), dtype: float64

        The expected number of seal failures by months 12 and 24 of a
        pump that switches to high load at month 12:

        >>> model.cif_tvc([12, 24], [[0], [1]], "seal", xl=[0, 12]).round(3)
        array([0.6, 2.4])
        """
        fitter, options = _fitter(baseline)
        # The single-type fits' own checks of the rows (lengths, flags,
        # counts, overlaps), before the types are told apart.
        handle_recurrent_tvc(i, xl, xr, c, Z, n)
        ident = np.asarray(i)
        nrow = ident.shape[0]
        flags = np.asarray(c, dtype=float).astype(int)
        if np.any(flags == 2):
            raise ValueError(
                f"c = 2 (a count of failures within a row) is not taken by "
                f"{_NAME}: a count carries no event type per failure, and "
                "the rows of an item cannot overlap to hold one count per "
                "type. Fit each type's own counts with "
                "ProportionalIntensityNHPP.fit_tvc (or "
                "ProportionalIntensityHPP.fit_tvc) instead; the types are "
                "independent, so those fits are this model's."
            )
        marks = _marks(e, nrow)
        failed = flags == 0
        unmarked = failed & np.array([m is None for m in marks], dtype=bool)
        if unmarked.any():
            # As the fixed-covariate fit (and CauseSpecificNHPP) reads a
            # failure without a mark: it counts towards no type.
            warnings.warn(
                f"{int(unmarked.sum())} row(s) that end in a failure "
                "(c = 0) have no event type (e is missing). They count "
                "towards no type, so the types' intensities are "
                "underestimated by those failures; give their types if "
                "they are known.",
                stacklevel=2,
            )
        event_types = ordered_labels(marks[failed])
        if not event_types:
            raise ValueError(
                "The data has no failures (no c = 0 row), so no event "
                "type's intensity can be estimated."
            )
        models = {}
        with collect_aliased() as aliased:
            for cause in event_types:
                ends_in_cause = failed & label_mask(marks, cause)
                models[cause] = fitter.fit_tvc(
                    ident,
                    xl,
                    xr,
                    np.where(ends_in_cause, 0, 1),
                    Z,
                    init=init,
                    **options,
                )
        warn_collected(aliased, "in the fit of each event type")
        out = cls._assemble(event_types, models, baseline)
        out.is_tvc = True
        return out

    @classmethod
    def fit_tvc_timeline(
        cls,
        i: ArrayLike,
        x: ArrayLike,
        Z: ArrayLike,
        c: ArrayLike,
        n: ArrayLike | None = None,
        e: ArrayLike | None = None,
        **fit_options: Any,
    ) -> CauseSpecificProportionalIntensity:
        """
        Fit the model from each item's covariate *timeline*: a log with
        one row each time something happens to an item.

        The rows are those of the single-type regressions'
        ``fit_tvc_timeline``, with the type ``e`` of each failure row.
        The timeline is expanded to start-stop intervals and fitted as
        :meth:`fit_tvc`.

        Parameters
        ----------
        i : array_like
            The item of each timeline row.
        x : array_like
            The time of each row, strictly increasing within an item: the
            first is the item's entry, the last the end of its record.
        Z : array_like
            The covariate row in force from ``x`` until the item's next
            row; the value on an item's last row is ignored.
        c : array_like
            What happened at ``x``: ``0`` a failure (of type ``e``), ``1``
            no failure (the entry, a change of covariates or the end of
            the record). An item's first row is its entry and must be
            ``1``.
        n : array_like, optional
            Accepted for a uniform interface; anything but 1 is refused.
        e : array_like
            The type of the failure on each ``c = 0`` row; ``None`` /
            ``NaN`` on the others. Required.
        **fit_options
            Passed to :meth:`fit_tvc` (``baseline``, ``init``, ``unit``).

        Returns
        -------
        CauseSpecificProportionalIntensity
            The fitted model, with ``is_tvc`` set.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        >>> x = [0, 6, 10, 13, 17, 20, 0, 3, 7, 10, 16, 20]
        >>> c = [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
        >>> e = [None, "motor", None, "seal", "seal", None,
        ...      None, "seal", "motor", None, "seal", None]
        >>> load = [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
        >>> model = CauseSpecificProportionalIntensity.fit_tvc_timeline(
        ...     i, x, load, c, e=e, baseline=HPP
        ... )
        >>> model.coefficients()["exp(coef)"].round(3)
        event  covariate
        motor  coef_0       1.0
        seal   coef_0       3.0
        Name: exp(coef), dtype: float64
        """
        ident = np.asarray(i)
        marks = _marks(e, ident.shape[0] if ident.ndim == 1 else None)
        i_ss, xl, xr, c_ss, Z_ss, n_ss = handle_recurrent_tvc_timeline(
            i, x, Z, c, n
        )
        # The rows as the timeline reader sorted them (by item, then
        # time), each interval ending at a row that is not an item's first
        order = np.lexsort((np.asarray(x, dtype=float), ident))
        sorted_ident = ident[order]
        later = np.r_[False, sorted_ident[1:] == sorted_ident[:-1]]
        return cls.fit_tvc(
            i_ss, xl, xr, c_ss, Z_ss, n_ss, marks[order][later], **fit_options
        )

    @classmethod
    def fit_tvc_from_df(
        cls,
        df: Any,
        i_col: str,
        xl_col: str,
        xr_col: str,
        c_col: str,
        Z_cols: str | list[str],
        e_col: str,
        n_col: str | None = None,
        **fit_options: Any,
    ) -> CauseSpecificProportionalIntensity:
        """Fit start-stop time-varying-covariate rows from a DataFrame.

        ``i_col``, ``xl_col``, ``xr_col``, ``c_col``, ``e_col`` and
        ``n_col`` name the columns passed to :meth:`fit_tvc` as ``i``,
        ``xl``, ``xr``, ``c``, ``e`` and ``n``, and ``Z_cols`` (a column
        name or a list of them) the covariates, whose names then name the
        coefficients. Other keyword arguments go to :meth:`fit_tvc`
        (``baseline``, ``init``, ``unit``).

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> df = pd.DataFrame({
        ...     "pump": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        ...     "start": [0, 6, 10, 13, 17, 0, 3, 7, 10, 16],
        ...     "stop": [6, 10, 13, 17, 20, 3, 7, 10, 16, 20],
        ...     "c": [0, 1, 0, 0, 1, 0, 0, 1, 0, 1],
        ...     "mode": ["motor", None, "seal", "seal", None, "seal",
        ...              "motor", None, "seal", None],
        ...     "high_load": [0, 0, 1, 1, 1, 1, 1, 1, 0, 0],
        ... })
        >>> model = CauseSpecificProportionalIntensity.fit_tvc_from_df(
        ...     df, "pump", "start", "stop", "c", "high_load", "mode",
        ...     baseline=HPP,
        ... )
        >>> model.models["seal"].parameter_names
        ['lambda', 'high_load']
        >>> model.models["seal"].coeffs.round(4)
        array([1.0986])
        """
        return _FRAME._fit_tvc_frame(
            cls.fit_tvc,
            df,
            {
                "i": i_col,
                "xl": xl_col,
                "xr": xr_col,
                "c": c_col,
                "e": e_col,
                "n": n_col,
            },
            Z_cols,
            fit_options,
        )

    @classmethod
    def fit_tvc_timeline_from_df(
        cls,
        df: Any,
        i_col: str,
        x_col: str,
        Z_cols: str | list[str],
        c_col: str,
        e_col: str,
        n_col: str | None = None,
        **fit_options: Any,
    ) -> CauseSpecificProportionalIntensity:
        """Fit a covariate timeline from a DataFrame.

        ``i_col``, ``x_col``, ``c_col``, ``e_col`` and ``n_col`` name the
        columns passed to :meth:`fit_tvc_timeline` as ``i``, ``x``,
        ``c``, ``e`` and ``n``, and ``Z_cols`` (a column name or a list
        of them) the covariates in force from each row, whose names then
        name the coefficients. Other keyword arguments go to
        :meth:`fit_tvc` (``baseline``, ``init``, ``unit``).

        Examples
        --------
        >>> import pandas as pd
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> log = pd.DataFrame({
        ...     "pump": [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
        ...     "month": [0, 6, 10, 13, 17, 20, 0, 3, 7, 10, 16, 20],
        ...     "c": [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1],
        ...     "mode": [None, "motor", None, "seal", "seal", None,
        ...              None, "seal", "motor", None, "seal", None],
        ...     "high_load": [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
        ... })
        >>> fit = CauseSpecificProportionalIntensity.fit_tvc_timeline_from_df
        >>> model = fit(
        ...     log, "pump", "month", "high_load", "c", "mode", baseline=HPP
        ... )
        >>> model.models["seal"].coeffs.round(4)
        array([1.0986])
        """
        return _FRAME._fit_tvc_frame(
            cls.fit_tvc_timeline,
            df,
            {
                "i": i_col,
                "x": x_col,
                "c": c_col,
                "e": e_col,
                "n": n_col,
            },
            Z_cols,
            fit_options,
        )

    # -- the fit as a whole ------------------------------------------------

    @property
    def maximum(self) -> str:
        """What the types' fits reached, one of ``MAXIMUM_STATES``
        (``surpyval.utils.no_maximum``): the worst of the per-type
        models' ``maximum`` (principles 12 and 13).

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... ).maximum
        'verified'
        """
        return combined_maximum(
            getattr(self.models[k], "maximum", "unknown")
            for k in self.event_types
        )

    @property
    def aliased(self) -> np.ndarray:
        """The columns of ``Z`` whose coefficients the data cannot
        determine in some event type's fit: a constant column (where the
        baseline has a scale, which is the intercept) or a linear
        combination of the others. Their coefficients are ``nan`` in that
        type's model's ``coeffs`` and rows of :meth:`coefficients`, one
        warning names them, and predictions take them as 0.

        Examples
        --------
        A second covariate that repeats the first cannot be told from it:

        >>> import warnings
        >>> import numpy as np
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = np.array([0.0] * 5 + [1.0] * 9)
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     model = CauseSpecificProportionalIntensity.fit(
        ...         x, np.c_[harsh, harsh], i=i, c=c, e=e, baseline=HPP
        ...     )
        >>> len(caught)
        1
        >>> model.aliased
        array([1])
        >>> model.models["a"].coeffs.round(4)
        array([0.9163,    nan])
        """
        coeffs = np.array(
            [
                np.asarray(self.models[k].coeffs, dtype=float)
                for k in self.event_types
            ]
        )
        return np.flatnonzero(np.isnan(coeffs).any(axis=0))

    def neg_ll(self) -> float:
        """
        The negative log-likelihood of the joint model: the sum of the
        types' fits' (each type's events are a Poisson process of their
        own, so the likelihood factorises). Raises the ``ValueError`` of
        a type's model where there is no likelihood (a model restored
        with ``from_dict`` / ``from_json``).

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> each = [model.models[k].neg_ll() for k in model.event_types]
        >>> bool(abs(model.neg_ll() - sum(each)) < 1e-12)
        True

        Two parameters per type (a rate and a coefficient), and ten
        events:

        >>> bool(abs(model.aic() - (2 * 4 + 2 * model.neg_ll())) < 1e-12)
        True
        >>> import numpy as np
        >>> bool(abs(model.bic()
        ...          - (4 * np.log(10) + 2 * model.neg_ll())) < 1e-12)
        True
        """
        return float(sum(self.models[k].neg_ll() for k in self.event_types))

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood of the joint model, ``-neg_ll()``:
        the sum of the types'."""
        return -self.neg_ll()

    def _ic_terms(self) -> tuple[float, float]:
        # The parameters estimated over every type (an aliased
        # coefficient is not one), and the number of events of any type.
        k_total, n_total = 0.0, 0.0
        for cause in self.event_types:
            model = self.models[cause]
            model._check_fitted()
            k_total += float(model._ic_k())
            n_total += float(model._n_obs)
        return k_total, n_total

    def aic(self) -> float:
        """
        Akaike's information criterion of the joint model, ``2 K + 2
        neg_ll()`` with ``K`` the number of parameters estimated over all
        types: the sum of the types' AICs. Lower is better. See
        :meth:`neg_ll` for an example.
        """
        k_total, _ = self._ic_terms()
        return float(2 * k_total + 2 * self.neg_ll())

    def bic(self) -> float:
        """
        The Bayesian information criterion of the joint model, ``K ln n +
        2 neg_ll()``, with the ``K`` of :meth:`aic` and ``n`` the number
        of events of any type, as :meth:`CauseSpecificNHPP.bic`. It is
        not the sum of the types' BICs. Lower is better. See
        :meth:`neg_ll` for an example.
        """
        k_total, n_total = self._ic_terms()
        return float(k_total * np.log(n_total) + 2 * self.neg_ll())

    def aic_c(self) -> float:
        """
        The small-sample corrected AIC of the joint model, ``aic() +
        (2K^2 + 2K) / (n - K - 1)``, with the ``K`` and ``n`` of
        :meth:`bic`; ``nan`` where ``n <= K + 1``.
        """
        k_total, n_total = self._ic_terms()
        return corrected_aic(self.aic(), k_total, n_total)

    def __repr__(self) -> str:
        return (
            "Cause-specific proportional intensity ({} baseline) with "
            "causes: {}".format(
                self.models[self.event_types[0]].dist.name, self.event_types
            )
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted model to a plain, JSON-serialisable dict:
        the baseline's name, the event types, each type's fitted
        proportional-intensity model and each type's parameter covariance.

        The covariances are what the restored model's bounds,
        :meth:`coefficients` and :meth:`summary` are computed from, so
        they are the fitted model's. The data are not kept: the
        likelihood and the information criteria are refused on the
        restored model.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> restored = CauseSpecificProportionalIntensity.from_dict(
        ...     model.to_dict()
        ... )
        >>> restored.total_cif_cb(60.0, [1]).round(2)
        array([1.67, 7.34])
        >>> model.total_cif_cb(60.0, [1]).round(2)
        array([1.67, 7.34])

        See Also
        --------
        from_dict, to_json, from_json
        """
        out: dict = {
            "model": _NAME,
            "baseline": self.models[self.event_types[0]].dist.name,
            "event_types": to_native(list(self.event_types)),
            "models": [
                self.models[cause].to_dict() for cause in self.event_types
            ],
            "covariances": [
                self._saved_covariance(cause) for cause in self.event_types
            ],
        }
        if self.feature_names is not None:
            out["feature_names"] = list(self.feature_names)
        return stamp_schema(out)

    def _saved_covariance(self, cause: Any) -> list | None:
        """A type's parameter covariance as ``to_dict`` writes it: the
        fit's, a restored model's saved one, or ``None`` without either."""
        model = self.models[cause]
        if hasattr(model, "_neg_ll"):
            return model._covariance_matrix()[0].tolist()
        saved = getattr(self, "_covariances", {}).get(cause)
        return None if saved is None else np.asarray(saved).tolist()

    @classmethod
    def from_dict(cls, model_dict: dict) -> CauseSpecificProportionalIntensity:
        """
        Rebuild a model from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, _NAME, "a cause-specific proportional-intensity model"
        )
        # JSON writes a tuple label as a list; turn it back into a tuple.
        event_types = [label_from_native(v) for v in model_dict["event_types"]]
        models = {
            cause: ProportionalIntensityModel.from_dict(sub)
            for cause, sub in zip(event_types, model_dict["models"])
        }
        first = models[event_types[0]]
        baseline = (
            HPP
            if first.kind == "HPP"
            else intensity_dist_by_name(model_dict["baseline"])
        )
        out = cls._assemble(event_types, models, baseline)
        names = model_dict.get("feature_names")
        out.feature_names = None if names is None else list(names)
        saved = model_dict.get("covariances") or [None] * len(event_types)
        out._covariances = {
            cause: (
                None
                if cov is None
                else np.array(
                    [[np.nan if v is None else v for v in row] for row in cov],
                    dtype=float,
                )
            )
            for cause, cov in zip(event_types, saved)
        }
        return out

    # -- evaluation --------------------------------------------------------

    def _check_cause(self, cause: Any) -> None:
        if cause not in self.models:
            raise unknown_cause_error(cause, self.event_types)

    def cif(self, x: ArrayLike, Z: ArrayLike, event: Any) -> np.ndarray:
        """
        The expected number of events of type ``event`` by ``x`` of an
        item with covariates ``Z``: that type's model's ``cif(x, Z)``.

        Parameters
        ----------
        x : array_like
            The times.
        Z : array_like
            The item's covariates: one vector used at every time, or one
            row per time in ``x``.
        event : hashable
            The event type; one of ``event_types``.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.cif([30, 60], [0], "a").round(3)
        array([0.5, 1. ])
        >>> model.cif([30, 60], [1], "a").round(3)
        array([1.25, 2.5 ])
        """
        self._check_cause(event)
        return self.models[event].cif(x, Z)

    def iif(self, x: ArrayLike, Z: ArrayLike, event: Any) -> np.ndarray:
        """
        The intensity (rate) of events of type ``event`` at ``x`` for an
        item with covariates ``Z``: that type's model's ``iif(x, Z)``.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.iif(30, [1], "a").round(4), model.iif(30, [1], "b").round(4)
        (np.float64(0.0417), np.float64(0.0167))
        """
        self._check_cause(event)
        return self.models[event].iif(x, Z)

    def cif_tvc(
        self,
        x: ArrayLike,
        Z: Any,
        event: Any,
        xl: ArrayLike | None = None,
        given: float | None = None,
    ) -> np.ndarray:
        """
        The expected number of events of type ``event`` by ``x`` of an
        item that follows a covariate schedule: that type's model's
        ``cif_tvc(x, Z, xl, given)``.

        Parameters
        ----------
        x : array_like
            The times.
        Z : StepSchedule or array_like
            The covariate schedule: a
            :class:`~surpyval.univariate.regression.tvc_schedule.StepSchedule`,
            or an array of per-segment covariate rows with ``xl`` the
            time each takes effect.
        event : hashable
            The event type; one of ``event_types``.
        xl : array_like, optional
            The start of each row of an array ``Z``.
        given : float, optional
            An age to count from: the expected number of events in
            ``(given, x]``. The schedule then need only start by
            ``given``.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )

        A pump moved to harsh service at month 30:

        >>> model.cif_tvc([30, 60], [[0], [1]], "a", xl=[0, 30]).round(3)
        array([0.5 , 1.75])
        >>> model.cif_tvc(60, [[1]], "a", xl=[30], given=30).round(3)
        np.float64(1.25)
        """
        self._check_cause(event)
        return self.models[event].cif_tvc(x, Z, xl=xl, given=given)

    def iif_tvc(
        self,
        x: ArrayLike,
        Z: Any,
        event: Any,
        xl: ArrayLike | None = None,
    ) -> np.ndarray:
        """
        The intensity of events of type ``event`` at ``x`` with the
        covariates in force then along a schedule: that type's model's
        ``iif_tvc(x, Z, xl)``. ``Z`` and ``xl`` are as for
        :meth:`cif_tvc`.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.iif_tvc([10, 40], [[0], [1]], "a", xl=[0, 30]).round(4)
        array([0.0167, 0.0417])
        """
        self._check_cause(event)
        return self.models[event].iif_tvc(x, Z, xl=xl)

    @keeps_query_shape
    def total_cif(self, x: ArrayLike, Z: ArrayLike) -> np.ndarray:
        """
        The expected number of events of any type by ``x`` of an item
        with covariates ``Z``: the sum of the types' :meth:`cif`.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.total_cif([30, 60], [1]).round(3)
        array([1.75, 3.5 ])
        """
        return self._sum(lambda model: model.cif(x, Z))

    @keeps_query_shape
    def total_iif(self, x: ArrayLike, Z: ArrayLike) -> np.ndarray:
        """
        The intensity of events of any type at ``x`` for an item with
        covariates ``Z``: the sum of the types' :meth:`iif`.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.total_iif([30, 30], [[0], [1]]).round(4)
        array([0.025 , 0.0583])
        """
        return self._sum(lambda model: model.iif(x, Z))

    @keeps_query_shape
    def total_cif_tvc(
        self,
        x: ArrayLike,
        Z: Any,
        xl: ArrayLike | None = None,
        given: float | None = None,
    ) -> np.ndarray:
        """
        The expected number of events of any type by ``x`` of an item
        that follows a covariate schedule: the sum of the types'
        :meth:`cif_tvc`. ``Z``, ``xl`` and ``given`` are as there.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.total_cif_tvc([30, 60], [[0], [1]], xl=[0, 30]).round(3)
        array([0.75, 2.5 ])
        """
        return self._sum(lambda model: model.cif_tvc(x, Z, xl=xl, given=given))

    @keeps_query_shape
    def total_iif_tvc(
        self,
        x: ArrayLike,
        Z: Any,
        xl: ArrayLike | None = None,
    ) -> np.ndarray:
        """
        The intensity of events of any type at ``x`` with the covariates
        in force then along a schedule: the sum of the types'
        :meth:`iif_tvc`.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.total_iif_tvc([10, 40], [[0], [1]], xl=[0, 30]).round(4)
        array([0.025 , 0.0583])
        """
        return self._sum(lambda model: model.iif_tvc(x, Z, xl=xl))

    def _sum(self, each: Callable[[Any], Any]) -> np.ndarray:
        """The sum over the types of ``each(model)``."""
        total: Any = None
        for cause in self.event_types:
            value = np.asarray(each(self.models[cause]), dtype=float)
            total = value if total is None else total + value
        return total

    # -- inference ---------------------------------------------------------

    def _has_likelihood(self, cause: Any) -> bool:
        return hasattr(self.models[cause], "_neg_ll")

    def _inference(
        self, cause: Any, warn: bool = True
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(estimates, values, covariance)`` of a type's parameters,
        baseline first: the estimates as fitted (``nan`` for an aliased
        coefficient), the values the model predicts with (0 there), and
        the covariance (``nan`` rows and columns there): the fit's, or
        the one a restored model saved. Without either it raises the
        type's model's own refusal. ``warn=False`` leaves a covariance
        that could not be found as ``nan`` without the warning."""
        model = self.models[cause]
        if hasattr(model, "_neg_ll"):
            model._refuse_if_penalised()
            cov, found = model._covariance_matrix()
            estimates = np.asarray(model._mle, dtype=float)
        else:
            saved = getattr(self, "_covariances", {}).get(cause)
            if saved is None:
                model._check_fitted()
            cov = np.asarray(saved, dtype=float)
            estimates = np.r_[
                np.asarray(model.params, dtype=float),
                np.asarray(model.coeffs, dtype=float),
            ]
            found = bool(np.all(np.isfinite(cov[~np.isnan(estimates)])))
        if warn and not found:
            warn_no_covariance()
        values = np.where(np.isnan(estimates), 0.0, estimates)
        return estimates, values, cov

    def _wald_cb(
        self,
        curve: Callable[[Any], Callable[[np.ndarray], np.ndarray]],
        causes: Any,
        alpha_ci: float,
        bound: str,
    ) -> np.ndarray:
        """Log-scale Wald bounds on the sum over ``causes`` of the curve
        ``curve(model)(theta)`` of each type's parameters: the types'
        delta-method variances added (their estimates are independent),
        then the single-type construction, so one type's are its own
        model's."""
        estimate: Any = 0.0
        variance: Any = 0.0
        for cause in causes:
            estimates, values, cov = self._inference(cause)
            # An aliased coefficient is held at 0 and has no variance.
            held = np.isnan(estimates)
            cov = cov.copy()
            cov[held, :] = 0.0
            cov[:, held] = 0.0
            function_at = curve(self.models[cause])
            estimate = estimate + function_at(values)
            variance = (
                variance + delta_method_se(function_at, values, cov) ** 2
            )
        return log_transformed_cb(estimate, np.sqrt(variance), alpha_ci, bound)

    def _fixed_cb(
        self,
        function: str,
        x: ArrayLike,
        Z: ArrayLike,
        causes: Any,
        alpha_ci: float,
        bound: str,
    ) -> np.ndarray:
        """Bounds on the sum over ``causes`` of the ``cif`` or ``iif``
        (``function``) at ``x`` for covariates ``Z``."""
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        Zq = np.asarray(named_covariates(Z, self.feature_names), dtype=float)

        def curve(model: Any) -> Callable[[np.ndarray], np.ndarray]:
            k = len(model.params)
            baseline = getattr(model.dist, function)

            def function_at(theta: np.ndarray) -> np.ndarray:
                return baseline(xq, *theta[:k]) * np.exp(Zq @ theta[k:])

            return function_at

        return self._wald_cb(curve, causes, alpha_ci, bound)

    def _schedule_cb(
        self,
        x: ArrayLike,
        Z: Any,
        xl: ArrayLike | None,
        given: float | None,
        on: str,
        causes: Any,
        alpha_ci: float,
        bound: str,
    ) -> np.ndarray:
        """Bounds on the sum over ``causes`` of the expected count
        (``on="cif"``) or the intensity (``"iif"``) along a schedule, as
        the single-type ``cb_tvc`` forms one type's."""
        check_alpha_ci(alpha_ci)
        check_option("on", on, ("cif", "iif"))
        check_option("bound", bound, BOUNDS)
        if on == "iif" and given is not None:
            raise ValueError(
                "given applies to on='cif' (the events counted from an "
                "age): the intensity at a time depends only on the "
                "covariates then. Leave given out, with a schedule from "
                "the time origin."
            )
        for cause in causes:
            self._inference(cause, warn=False)
        first = self.models[self.event_types[0]]
        xq = np.atleast_1d(np.asarray(x, dtype=float))
        shape = xq.shape + ((2,) if bound == "two-sided" else ())
        lo, from_given = first._given(given)
        schedule = first._tvc_schedule(Z, xl, lo, from_given)
        if np.isnan(lo) or xq.size == 0:
            return np.full(shape, np.nan)
        segments = first._tvc_segments(schedule, lo, xq)

        def curve(model: Any) -> Callable[[np.ndarray], np.ndarray]:
            k = len(model.params)

            def function_at(theta: np.ndarray) -> np.ndarray:
                if on == "iif":
                    return model._iif_tvc(xq, segments, theta[:k], theta[k:])
                return model._cif_tvc(
                    xq, segments, lo, from_given, theta[:k], theta[k:]
                )

            return function_at

        return self._wald_cb(curve, causes, alpha_ci, bound)

    @keeps_query_shape
    def _saved_fixed_cb(
        self,
        x: ArrayLike,
        function: str,
        Z: ArrayLike,
        event: Any,
        alpha_ci: float,
        bound: str,
    ) -> np.ndarray:
        return self._fixed_cb(function, x, Z, (event,), alpha_ci, bound)

    @keeps_query_shape
    def _saved_schedule_cb(
        self,
        x: ArrayLike,
        Z: Any,
        event: Any,
        xl: ArrayLike | None,
        given: float | None,
        on: str,
        alpha_ci: float,
        bound: str,
    ) -> np.ndarray:
        return self._schedule_cb(
            x, Z, xl, given, on, (event,), alpha_ci, bound
        )

    def cif_cb(
        self,
        x: ArrayLike,
        Z: ArrayLike,
        event: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the expected number of events of type
        ``event`` (:meth:`cif`): that type's model's ``cif_cb(x, Z)``,
        the delta method on the log scale.

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        Z : array_like
            The item's covariates.
        event : hashable
            The event type; one of ``event_types``.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.cif_cb(60.0, [1], "a").round(2)
        array([1.04, 6.01])
        >>> model.models["a"].cif_cb(60.0, [1]).round(2)
        array([1.04, 6.01])
        """
        self._check_cause(event)
        if self._has_likelihood(event):
            return self.models[event].cif_cb(x, Z, alpha_ci, bound)
        return self._saved_fixed_cb(x, "cif", Z, event, alpha_ci, bound)

    def iif_cb(
        self,
        x: ArrayLike,
        Z: ArrayLike,
        event: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the intensity of events of type ``event``
        (:meth:`iif`): that type's model's ``iif_cb(x, Z)``. The
        arguments are those of :meth:`cif_cb`.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.iif_cb(30.0, [1], "a").round(4)
        array([0.0173, 0.1001])
        """
        self._check_cause(event)
        if self._has_likelihood(event):
            return self.models[event].iif_cb(x, Z, alpha_ci, bound)
        return self._saved_fixed_cb(x, "iif", Z, event, alpha_ci, bound)

    def cb_tvc(
        self,
        x: ArrayLike,
        Z: Any,
        event: Any,
        xl: ArrayLike | None = None,
        given: float | None = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds along a covariate schedule on the expected
        number of events of type ``event`` (``on="cif"``,
        :meth:`cif_tvc`) or on its intensity (``on="iif"``,
        :meth:`iif_tvc`): that type's model's ``cb_tvc``.

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        Z : StepSchedule or array_like
            The covariate schedule, as for :meth:`cif_tvc`.
        event : hashable
            The event type; one of ``event_types``.
        xl : array_like, optional
            The start of each row of an array ``Z``.
        given : float, optional
            An age to count from (``on="cif"`` only).
        on : {'cif', 'iif'}, optional
            What to bound. Default ``"cif"``.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.cb_tvc(60.0, [[0], [1]], "a", xl=[0, 30]).round(2)
        array([0.83, 3.67])
        """
        self._check_cause(event)
        if self._has_likelihood(event):
            return self.models[event].cb_tvc(
                x,
                Z,
                xl=xl,
                given=given,
                on=on,
                alpha_ci=alpha_ci,
                bound=bound,
            )
        return self._saved_schedule_cb(
            x, Z, event, xl, given, on, alpha_ci, bound
        )

    @keeps_query_shape
    def total_cif_cb(
        self,
        x: ArrayLike,
        Z: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the expected number of events of any type
        (:meth:`total_cif`) for an item with covariates ``Z``.

        The types are independent Poisson processes, each fitted to its
        own events, so their estimates are independent and the variance
        of the total is the sum of the types' delta-method variances. The
        bounds are then formed on the log scale, as each type's are
        (:meth:`cif_cb`) and as :meth:`CauseSpecificNHPP.total_cif_cb`
        forms them, so a model with one type has the total's bounds equal
        to that type's.

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        Z : array_like
            The item's covariates.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.total_cif(60.0, [1]).round(2)
        np.float64(3.5)
        >>> model.total_cif_cb(60.0, [1]).round(2)
        array([1.67, 7.34])
        """
        return self._fixed_cb("cif", x, Z, self.event_types, alpha_ci, bound)

    @keeps_query_shape
    def total_iif_cb(
        self,
        x: ArrayLike,
        Z: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the intensity of events of any type
        (:meth:`total_iif`) for an item with covariates ``Z``: the types'
        delta-method variances added, and the bounds formed on the log
        scale, as :meth:`total_cif_cb`.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.total_iif_cb(30.0, [1]).round(4)
        array([0.0278, 0.1224])
        """
        return self._fixed_cb("iif", x, Z, self.event_types, alpha_ci, bound)

    @keeps_query_shape
    def total_cb_tvc(
        self,
        x: ArrayLike,
        Z: Any,
        xl: ArrayLike | None = None,
        given: float | None = None,
        on: str = "cif",
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds along a covariate schedule on the expected
        number of events of any type (``on="cif"``,
        :meth:`total_cif_tvc`) or on the total intensity (``on="iif"``,
        :meth:`total_iif_tvc`): the types' delta-method variances added,
        and the bounds formed on the log scale, as :meth:`total_cif_cb`.
        The arguments are those of :meth:`cb_tvc`, without ``event``. A
        schedule that never changes gives :meth:`total_cif_cb` and
        :meth:`total_iif_cb`.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.total_cb_tvc(60.0, [[0], [1]], xl=[0, 30]).round(2)
        array([1.35, 4.65])
        """
        return self._schedule_cb(
            x, Z, xl, given, on, self.event_types, alpha_ci, bound
        )

    def coefficients(self, alpha_ci: float = 0.05) -> Any:
        """
        The coefficient table: one row per event type and covariate, to
        see which covariate matters for which type.

        The columns are those of the regressions' coefficient tables
        (``CoxPH.summary``): the coefficient ``coef``; ``exp(coef)``, the
        **rate ratio**, the factor by which one more unit of the
        covariate multiplies the rate of that type of event (2 doubles
        it, 1 leaves it alone, 0.5 halves it); the standard error; a
        two-sided ``1 - alpha_ci`` Wald interval for both; and the Wald
        statistic ``z`` with its two-sided p-value ``p`` for "this
        covariate has no effect on this type" (a rate ratio of 1). An
        aliased coefficient is ``nan`` throughout.

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(event, covariate)``, with columns ``coef``,
            ``exp(coef)``, ``se(coef)``, ``coef lower <level>``, ``coef
            upper <level>``, ``exp(coef) lower <level>``, ``exp(coef)
            upper <level>``, ``z`` and ``p``.

        Notes
        -----
        A wide interval that contains 1 says the data do not show an
        effect on that type, not that there is none: a type with few
        events has wide intervals.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> table = model.coefficients()
        >>> table[["coef", "exp(coef)", "se(coef)", "p"]].round(3)
        ... # doctest: +NORMALIZE_WHITESPACE
                          coef  exp(coef)  se(coef)      p
        event covariate
        a     coef_0     0.916        2.5     0.837  0.273
        b     coef_0     0.693        2.0     1.225  0.571
        """
        import pandas as pd

        from surpyval.univariate.regression._summary import coefficient_table

        check_alpha_ci(alpha_ci)
        tables, events = [], []
        for cause in self.event_types:
            model = self.models[cause]
            estimates, _, cov = self._inference(cause, warn=False)
            k = len(model.params)
            var = np.diag(cov)[k:]
            with np.errstate(invalid="ignore"):
                se = np.where(var > 0, np.sqrt(var), np.nan)
            table = coefficient_table(
                model._coefficient_names(), estimates[k:], se, alpha_ci
            )
            tables.append(table)
            events += [cause] * len(table)
        stacked = pd.concat(tables)
        stacked.index = pd.MultiIndex.from_arrays(
            [event_labels(events), list(stacked.index)],
            names=["event", "covariate"],
        )
        return stacked

    def summary(self, alpha_ci: float = 0.05) -> Any:
        """
        The fitted parameters of every event type with their standard
        errors and Wald intervals: the single-type ``summary`` of each
        type's model, stacked, one row per type and parameter (the
        baseline's parameters, then the coefficients), as
        :meth:`CauseSpecificNHPP.summary`.

        The intervals are those of ``param_cb`` (on the log scale for a
        positive parameter, on its own scale for a coefficient). For the
        coefficients as rate ratios with tests, see :meth:`coefficients`.

        Parameters
        ----------
        alpha_ci : float, optional
            The intervals' total tail probability. Default 0.05.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``(event, parameter)``, with columns ``estimate``,
            ``se``, ``lower <level>`` and ``upper <level>``.

        Examples
        --------
        >>> from surpyval.recurrent import (
        ...     HPP, CauseSpecificProportionalIntensity,
        ... )
        >>> x = [20, 45, 60, 30, 60, 8, 15, 31, 40, 60, 12, 25, 38, 60]
        >>> i = [1, 1, 1, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4]
        >>> c = [0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["a", "b", None, "a", None, "a", "a", "b", "a", None,
        ...      "a", "a", "b", None]
        >>> harsh = [0] * 5 + [1] * 9
        >>> model = CauseSpecificProportionalIntensity.fit(
        ...     x, harsh, i=i, c=c, e=e, baseline=HPP
        ... )
        >>> model.summary().round(2)  # doctest: +NORMALIZE_WHITESPACE
                         estimate    se  lower 95%  upper 95%
        event parameter
        a     lambda         0.02  0.01       0.00       0.07
              coef_0         0.92  0.84      -0.72       2.56
        b     lambda         0.01  0.01       0.00       0.06
              coef_0         0.69  1.22      -1.71       3.09
        """
        import pandas as pd

        check_alpha_ci(alpha_ci)
        level = f"{100 * (1 - alpha_ci):g}%"
        rows, events, names = [], [], []
        for cause in self.event_types:
            model = self.models[cause]
            estimates, _, cov = self._inference(cause, warn=False)
            for k, (value, (lo, hi)) in enumerate(
                zip(estimates, model._parameter_bounds())
            ):
                events.append(cause)
                names.append(model.parameter_names[k])
                rows.append(
                    [
                        value,
                        *parameter_interval(
                            value, cov[k, k], lo, hi, alpha_ci
                        ),
                    ]
                )
        return pd.DataFrame(
            rows,
            index=pd.MultiIndex.from_arrays(
                [event_labels(events), names], names=["event", "parameter"]
            ),
            columns=["estimate", "se", f"lower {level}", f"upper {level}"],
            dtype=float,
        )

    # -- not carried over yet: each refuses, saying what to use ------------

    diagnose = _not_available("diagnose")
    residuals = _not_available("residuals")
    cramer_von_mises = _not_available("cramer_von_mises")
    trend_test = _not_available("trend_test")
    plot = _not_available("plot")
    count_terminated_simulation = _not_available("count_terminated_simulation")
    count_terminated_simulation_data = _not_available(
        "count_terminated_simulation_data"
    )
    time_terminated_simulation = _not_available("time_terminated_simulation")
    time_terminated_simulation_data = _not_available(
        "time_terminated_simulation_data"
    )
    mcf = _not_available("mcf")
    forecast = _not_available("forecast")
    growth = _not_available("growth")
    mtbf = _not_available("mtbf")
    mtbf_cb = _not_available("mtbf_cb")
    total_mtbf = _not_available("total_mtbf")
    total_mtbf_cb = _not_available("total_mtbf_cb")
