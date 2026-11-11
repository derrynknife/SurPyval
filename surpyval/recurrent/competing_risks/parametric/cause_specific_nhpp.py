"""
Parametric cause-specific intensity model for recurrent events with competing
failure modes.

This is the parametric counterpart of :class:`CauseSpecificMCF`. A single
repairable item experiences events of several mutually exclusive types (marks
``e``) over time, and we fit a separate intensity model per type.

For a marked Poisson (NHPP) process the cause-specific processes are
**independent thinned Poisson processes**: an event of one cause neither
advances nor interrupts another cause's intensity. The joint likelihood
therefore factorises over causes, and each cause's intensity is the maximum-
likelihood NHPP fit to that cause's events over the *full* observation window
of every item -- other-cause events are simply ignored, exactly as a censored
period would be. Concretely, for each cause we keep that cause's events
(``c=0``) and add one right-censored window-close (``c=1``) per item at the
item's observation end, then hand the result to the ordinary NHPP fitter. This
reuses the whole intensity-fitting, inference and diagnostic machinery
unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from surpyval import plotting
from surpyval.recurrent.inference import check_alpha_ci, require_data
from surpyval.recurrent.parametric.crow_amsaa import CrowAMSAA
from surpyval.recurrent.parametric.parametric_recurrence import (
    OPPOSITE_BOUND,
    ParametricRecurrenceModel,
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
)
from surpyval.univariate.information_criteria import corrected_aic
from surpyval.utils import optional_column
from surpyval.utils.deprecation import ambiguous_item_ids
from surpyval.utils.event_history_data import (
    accepts_event_history_data,
    refuses_unmodelled_data,
)
from surpyval.utils.linalg import (
    delta_method_se,
    log_transformed_cb,
    wald_bound_on_support,
)
from surpyval.utils.no_maximum import combined_maximum
from surpyval.utils.recurrent_utils import handle_xicn
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.shapes import keeps_query_shape
from surpyval.utils.units import label_axes, takes_unit
from surpyval.utils.validation import (
    BOUNDS,
    check_option,
    unknown_cause_error,
)
from surpyval.utils.warnings import warn_no_covariance

# The parameter of each intensity model that carries its trend, and the
# value at which the intensity is constant. Above that value the intensity
# rises with time in every one of them.
_TREND_PARAMETER = {
    "Crow-AMSAA": ("beta", 1.0),
    "Duane": ("alpha", 1.0),
    "Cox-Lewis": ("beta", 0.0),
}


def event_labels(causes: Any) -> np.ndarray:
    """The cause labels as an object array, one entry per label (a tuple
    label stays one entry)."""
    out = np.empty(len(causes), dtype=object)
    for k, cause in enumerate(causes):
        out[k] = cause
    return out


def parameter_interval(
    value: float,
    var: float,
    lower: float | None,
    upper: float | None,
    alpha_ci: float,
) -> tuple[float, float, float]:
    """``(se, lower, upper)`` of a parameter as the single-process
    ``summary`` reports it: the ``param_cb`` interval, or ``nan`` where
    the variance is not positive (an estimate on a boundary)."""
    if not (np.isfinite(value) and var > 0):
        return np.nan, np.nan, np.nan
    if (lower is not None and value <= lower) or (
        upper is not None and value >= upper
    ):
        return float(np.sqrt(var)), np.nan, np.nan
    cb = wald_bound_on_support(
        float(value), float(var), lower, upper, alpha_ci, "two-sided"
    )
    return float(np.sqrt(var)), float(cb[0]), float(cb[1])


def cause_rows(data: Any, cause: Any, windows: dict) -> tuple:
    """One cause's rows of marked recurrent ``data``, as the ``(x, i, c,
    tl)`` of an ordinary recurrent fit: that cause's events (``c=0``),
    then one right-censored window-close (``c=1``) per item at the end of
    its window (for every item, also one with no events of this cause, so
    the compensator is integrated over the whole window). ``windows`` maps
    each item to its ``(entry, end)``. The other causes' events are left
    out, exactly as a censored period would be."""
    cx, ci, cc, ctl = [], [], [], []
    is_cause = label_mask(data.e, cause)
    for k, item in enumerate(data.i):
        if data.c[k] == 0 and is_cause[k]:
            entry, _ = windows[item]
            cx.append(float(data.x[k]))
            ci.append(item)
            cc.append(0)
            ctl.append(entry)
    for item in data.ids:
        entry, end = windows[item]
        cx.append(end)
        ci.append(item)
        cc.append(1)
        ctl.append(entry)
    return cx, ci, cc, ctl


class _AllCauses:
    """The superposition of a cause-specific model's processes, as
    :func:`surpyval.forecast` reads a Poisson process: its cumulative
    intensity, and the fitted items (their ids and observation windows)."""

    def __init__(self, model: CauseSpecificNHPP) -> None:
        self.cif = model.total_cif
        first = model.models[model.event_types[0]]
        self.data = getattr(first, "data", None)


class CauseSpecificNHPP(SerialisableMixin):
    """
    Parametric cause-specific intensity model for a recurrent process with
    competing event types.

    One NHPP intensity model (``CrowAMSAA`` by default, or any counting-process
    fitter) is fitted per event type, sharing each item's observation window
    across causes. Access the per-cause fitted models through
    ``self.models[cause]`` -- each is an ordinary
    :class:`ParametricRecurrenceModel` with its full ``cif``/``iif``/inference/
    diagnostic behaviour -- or use the convenience methods below.

    Examples
    --------
    Two pumps, each repaired for seal or motor failures and observed to
    times 10 and 12 (the ``c=1`` rows, which have no event type):

    >>> from surpyval.recurrent import CauseSpecificNHPP
    >>> x = [2, 5, 7, 10, 3, 4, 8, 12]
    >>> i = [1, 1, 1, 1, 2, 2, 2, 2]
    >>> c = [0, 0, 0, 1, 0, 0, 0, 1]
    >>> e = ["seal", "motor", "seal", None, "seal", "seal", "motor", None]
    >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
    >>> model
    Cause-specific Crow-AMSAA with causes: ['motor', 'seal']

    The expected number of seal repairs per pump by time 10, and of
    repairs of either kind:

    >>> model.cif([10], "seal").round(4)
    array([1.8376])
    >>> model.total_cif([10]).round(4)
    array([2.6773])
    """

    # Populated by the fit classmethods; declared for the type checker.
    df: Any
    data: Any
    event_types: list
    models: dict
    dist: Any
    how: str
    # A restored model's saved inference: each cause's parameter
    # covariance (None where its fit had none) and the end of observation.
    _covariances: dict
    _end: float | None

    @property
    def maximum(self) -> str:
        """What the causes' fits reached, one of ``MAXIMUM_STATES``
        (``surpyval.utils.no_maximum``): the worst of the per-cause
        models' ``maximum`` (principles 12 and 13).

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> CauseSpecificNHPP.fit(x, i=i, c=c, e=e).maximum
        'verified'
        """
        return combined_maximum(
            getattr(self.models[k], "maximum", "unknown")
            for k in self.event_types
        )

    # -- information criteria (#711): the likelihood factorises over causes

    def neg_ll(self) -> float:
        """
        The negative log-likelihood of the joint model: the sum of the
        causes' fits' (each cause's events are a Poisson process of their
        own, so the likelihood factorises). Raises the ``ValueError`` of a
        cause's model where there is no likelihood: a ``how="MSE"`` fit,
        or a model restored with ``from_dict`` / ``from_json``.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> round(model.neg_ll(), 4)
        37.9917

        Two Crow-AMSAA parameters per cause, and nine events:

        >>> round(model.aic(), 4), round(2 * 4 + 2 * model.neg_ll(), 4)
        (83.9833, 83.9833)
        >>> round(model.bic(), 4)
        84.7722
        """
        return float(sum(self.models[k].neg_ll() for k in self.event_types))

    @property
    def log_likelihood(self) -> float:
        """The maximised log-likelihood of the joint model, ``-neg_ll()``:
        the sum of the causes'."""
        return -self.neg_ll()

    def _ic_terms(self) -> tuple[int, float]:
        # The parameters estimated over every cause, and the number of
        # events of any cause (the sum of the causes' own counts).
        k_total, n_total = 0, 0.0
        for cause in self.event_types:
            model = self.models[cause]
            model._check_fitted()
            k_total += int(model._estimated().sum())
            n_total += float(model._n_obs)
        return k_total, n_total

    def aic(self) -> float:
        """
        Akaike's information criterion of the joint model, ``2 K + 2
        neg_ll()`` with ``K`` the number of parameters estimated over all
        causes: the sum of the causes' AICs. Lower is better.
        """
        k_total, _ = self._ic_terms()
        return float(2 * k_total + 2 * self.neg_ll())

    def bic(self) -> float:
        """
        The Bayesian information criterion of the joint model, ``K ln n +
        2 neg_ll()``, with the ``K`` of :meth:`aic` and ``n`` the number
        of events of any cause (end-of-observation rows add nothing), the
        rule of every SurPyval BIC. It is not the sum of the causes' BICs,
        which would charge each cause's parameters ``ln`` of its own
        events. Lower is better.
        """
        k_total, n_total = self._ic_terms()
        return float(k_total * np.log(n_total) + 2 * self.neg_ll())

    def aic_c(self) -> float:
        """
        The small-sample corrected AIC of the joint model, ``aic() +
        (2K^2 + 2K) / (n - K - 1)``, with the ``K`` and ``n`` of
        :meth:`bic`; ``nan`` where ``n <= K + 1``, as on every other
        model.
        """
        k_total, n_total = self._ic_terms()
        return corrected_aic(self.aic(), k_total, n_total)

    def __repr__(self) -> str:
        return "Cause-specific {} with causes: {}".format(
            self.dist.name, self.event_types
        )

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted cause-specific NHPP to a plain,
        JSON-serialisable dict: the shared intensity model's name, the list of
        event types, each cause's fitted intensity model, each cause's
        parameter covariance (``null`` for a fit without a likelihood) and
        the end of observation.

        The covariances and the end of observation are what the restored
        model's Wald bounds, :meth:`growth` and :meth:`summary` are
        computed from, so they are the fitted model's. The data are not
        kept: what needs them (a forecast of the fitted items,
        ``method="crow"``, ``plot``, the likelihood) is refused on the
        restored model.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> restored = CauseSpecificNHPP.from_dict(model.to_dict())
        >>> restored.total_cif_cb(60.0).round(3)
        array([2.341, 8.649])
        >>> model.total_cif_cb(60.0).round(3)
        array([2.341, 8.649])

        See Also
        --------
        from_dict, to_json, from_json
        """
        return stamp_schema(
            {
                "model": "CauseSpecificNHPP",
                "dist": self.dist.name,
                "event_types": to_native(list(self.event_types)),
                "models": [
                    self.models[cause].to_dict() for cause in self.event_types
                ],
                "covariances": [
                    self._saved_covariance(cause) for cause in self.event_types
                ],
                "end": self._end_of_observation(),
            }
        )

    def _saved_covariance(self, cause: Any) -> list | None:
        """A cause's parameter covariance as ``to_dict`` writes it: the
        fit's, a restored model's saved one, or ``None`` without either."""
        model = self.models[cause]
        if hasattr(model, "_neg_ll"):
            return model._covariance_matrix()[0].tolist()
        saved = getattr(self, "_covariances", {}).get(cause)
        return None if saved is None else np.asarray(saved).tolist()

    def _end_of_observation(self) -> float | None:
        """The latest time any fitted item was observed to; a restored
        model's saved one; ``None`` without either."""
        data = getattr(self, "data", None)
        if data is not None:
            return float(np.max(data.x))
        return getattr(self, "_end", None)

    @classmethod
    def from_dict(cls, model_dict: dict) -> CauseSpecificNHPP:
        """
        Rebuild a cause-specific NHPP from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "CauseSpecificNHPP", "a cause-specific NHPP"
        )
        out = cls()
        out.dist = intensity_dist_by_name(model_dict["dist"])
        # JSON writes a tuple label as a list; turn it back into a tuple.
        out.event_types = [
            label_from_native(v) for v in model_dict["event_types"]
        ]
        out.models = {
            cause: ParametricRecurrenceModel.from_dict(sub)
            for cause, sub in zip(out.event_types, model_dict["models"])
        }
        # Written since the bounds through this model were added; a dict
        # saved before then has neither, and its bounds refuse.
        saved = model_dict.get("covariances") or [None] * len(out.event_types)
        out._covariances = {
            cause: (
                None
                if cov is None
                else np.array(
                    [[np.nan if v is None else v for v in row] for row in cov],
                    dtype=float,
                )
            )
            for cause, cov in zip(out.event_types, saved)
        }
        end = model_dict.get("end")
        out._end = None if end is None else float(end)
        return out

    # --- per-item observation windows ------------------------------------

    @staticmethod
    def _item_window(data: Any, item: Any) -> tuple:
        """The ``(entry, end)`` observation window of a single item.

        Entry is the item's left-truncation bound (delayed entry). The end is
        where the item leaves observation: its right-censoring (``c=1``) row
        when it has one, otherwise its finite right-truncation ``tr``,
        otherwise its last recorded event (failure-terminated).
        """
        mask = data.i == item
        entry = float(data.tl[mask][0])
        x_upper = data.x[mask] if data.x.ndim == 1 else data.x[mask][:, 1]
        c_item = data.c[mask]
        if (c_item == 1).any():
            end = float(x_upper[c_item == 1][0])
        else:
            end = float(x_upper.max())
        return entry, end

    @classmethod
    @refuses_unmodelled_data
    def fit_from_recurrent_data(
        cls,
        data: Any,
        dist: Any = CrowAMSAA,
        how: str = "MLE",
        init: ArrayLike | None = None,
    ) -> CauseSpecificNHPP:
        """
        Fit the cause-specific intensity model from prepared
        :class:`EventHistoryData` carrying event-type marks ``e``.

        Parameters
        ----------
        data : EventHistoryData
            Recurrent data with event-type marks (``data.e`` not ``None``).
        dist : counting-process fitter, optional
            The intensity model fitted per cause (``CrowAMSAA`` by default;
            ``HPP``, ``Duane``, ``CoxLewis`` are also valid).
        how : str, optional
            ``"MLE"`` or ``"MSE"``; passed through to the per-cause fit.
        init : array_like, optional
            Initial parameters for each per-cause optimisation.

        Returns
        -------
        CauseSpecificNHPP
        """
        if data.e is None:
            raise ValueError(
                "EventHistoryData has no event-type marks; pass `e` to "
                "fit a cause-specific intensity model."
            )
        if data.x.ndim != 1:
            raise ValueError(
                "Cause-specific intensity models require exact (1D) event "
                "times."
            )
        from surpyval.utils.recurrent_utils import refuse_right_truncation

        # Selection on the first failure of any type, which a model of each
        # type apart cannot account for (#924)
        refuse_right_truncation(data, "CauseSpecificNHPP")
        unsupported = sorted(set(data.c.tolist()) - {0, 1})
        if unsupported:
            raise ValueError(
                "Cause-specific intensity models support only exact (c=0) "
                "and right-censored (c=1) rows; got censoring code(s) "
                "{}.".format(unsupported)
            )

        out = cls()
        out.data = data
        out.event_types = data.event_types
        out.dist = dist

        # Each item's shared observation window [entry, end].
        windows = {item: cls._item_window(data, item) for item in data.ids}

        out.models = {}
        for cause in out.event_types:
            cx, ci, cc, ctl = cause_rows(data, cause, windows)
            cause_data = handle_xicn(cx, ci, cc, tl=ctl)
            out.models[cause] = dist.fit_from_recurrent_data(
                cause_data, how=how, init=init
            )
        return out

    @classmethod
    @ambiguous_item_ids("fit(x, c, n, t)")
    @takes_unit
    @accepts_event_history_data
    def fit(
        cls,
        x: ArrayLike,
        i: ArrayLike | None = None,
        c: ArrayLike | None = None,
        n: ArrayLike | None = None,
        e: ArrayLike | None = None,
        tl: ArrayLike | None = None,
        tr: ArrayLike | None = None,
        dist: Any = CrowAMSAA,
        how: str = "MLE",
        init: ArrayLike | None = None,
    ) -> CauseSpecificNHPP:
        """
        Fit a cause-specific intensity model.

        Parameters
        ----------
        x : array like
            Event (and censoring) times.
        i : array like, optional
            Item / subject id for each row. Defaults to a single item.
        c : array like, optional
            Censoring flag for each row (0 observed, 1 right censored).
        n : array like, optional
            The number of events each row stands for. This model takes exact
            events (``c=0``) and end-of-observation rows (``c=1``), each of
            which stands for one, so every ``n`` is 1 (``n > 1`` is refused:
            repeat the row for simultaneous events). Defaults to 1.
        e : array like
            Event type (mark) for each row. ``None``/``NaN`` for censored rows.
            A mark may be any hashable label: an integer, a string, a
            tuple, or a mix of these.
        tl : array like or scalar, optional
            Left-truncation (delayed-entry) time of each item: a scalar for
            every item, one value per row (the same on every row of an
            item), or one value per item, in the sorted order of the item
            ids (``np.unique(i)``; read per row when there are as many rows
            as items).
        tr : array like or scalar, optional
            Right-truncation time of each item, given like ``tl``. It closes
            the item's window, as a ``c=1`` row does; an item with both
            must have them at the same time (a ``c=1`` row before ``tr``
            raises a ``ValueError``).
        dist : counting-process fitter, optional
            The intensity model fitted per cause (``CrowAMSAA`` by default).
        how : str, optional
            ``"MLE"`` or ``"MSE"``.
        init : array_like, optional
            Initial parameters for each per-cause optimisation.

        Returns
        -------
        CauseSpecificNHPP
        """
        if e is None:
            raise ValueError(
                "`e` (event types) is required for a cause-specific "
                "intensity model."
            )
        data = handle_xicn(x, i, c, n, tl=tl, tr=tr, e=e)
        return cls.fit_from_recurrent_data(data, dist=dist, how=how, init=init)

    @classmethod
    @ambiguous_item_ids(
        "CompetingRisks.fit_from_df(df, x_col, e_col, c_col)", column=True
    )
    @column_arguments("x", "i", "c", "n", "tl", "tr")
    @takes_unit
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        i_col: str | None = None,
        c_col: str | None = None,
        n_col: str | None = None,
        tl_col: str | None = None,
        tr_col: str | None = None,
        dist: Any = CrowAMSAA,
        how: str = "MLE",
        init: ArrayLike | None = None,
    ) -> CauseSpecificNHPP:
        """
        Fit a cause-specific intensity model from a :class:`pandas.DataFrame`,
        naming the columns to read. See :meth:`fit` for the meaning of each.
        """

        model = cls.fit(
            x=df[x_col].to_numpy(),
            i=optional_column(df, i_col),
            c=optional_column(df, c_col),
            n=optional_column(df, n_col),
            e=optional_column(df, e_col),
            tl=optional_column(df, tl_col),
            tr=optional_column(df, tr_col),
            dist=dist,
            how=how,
            init=init,
        )
        model.df = df
        return model

    # --- evaluation ------------------------------------------------------

    def _check_cause(self, cause: Any) -> None:
        if cause not in self.models:
            raise unknown_cause_error(cause, self.event_types)

    def cif(self, x: ArrayLike, event: Any) -> np.ndarray:
        """Cause-specific cumulative intensity: the expected count of
        events of type ``event``."""
        self._check_cause(event)
        return self.models[event].cif(x)

    def iif(self, x: ArrayLike, event: Any) -> np.ndarray:
        """Cause-specific instantaneous intensity of events of type
        ``event``."""
        self._check_cause(event)
        return self.models[event].iif(x)

    def mcf(self, x: ArrayLike, event: Any) -> np.ndarray:
        """Cause-specific mean cumulative function (alias of :meth:`cif`)."""
        return self.cif(x, event)

    def total_cif(self, x: ArrayLike) -> np.ndarray:
        """
        Total cumulative intensity across all causes -- the expected number of
        events of any type, which (the causes being independent thinnings of
        the overall process) is the sum of the cause-specific intensities.
        """
        total: np.ndarray | None = None
        for cause in self.event_types:
            contribution = self.models[cause].cif(x)
            total = contribution if total is None else total + contribution
        assert total is not None  # event_types is never empty on a fit model
        return total

    @keeps_query_shape
    def total_iif(self, x: ArrayLike) -> np.ndarray:
        """
        Total instantaneous intensity across all causes: the rate of events
        of any type, the sum of the causes' intensities.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> model.total_iif([30.0, 60.0]).round(4)
        array([0.0665, 0.0559])
        >>> each = [model.iif([30.0, 60.0], k) for k in model.event_types]
        >>> (each[0] + each[1]).round(4)
        array([0.0665, 0.0559])
        """
        return self._total("iif", np.asarray(x, dtype=float))

    @keeps_query_shape
    def total_mtbf(self, x: ArrayLike) -> np.ndarray:
        """
        The instantaneous mean time between events of any type at ``x``,
        ``1 / total_iif(x)``: the MTBF of one item, whatever the cause, if
        it stayed as it is at ``x``.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> round(float(model.total_mtbf(60.0)), 2)
        17.88
        """
        with np.errstate(divide="ignore"):
            return 1.0 / self._total("iif", np.asarray(x, dtype=float))

    def _total(self, fname: str, x: np.ndarray) -> np.ndarray:
        """The sum over causes of the intensity function ``fname``."""
        total = np.zeros(np.shape(x))
        for cause in self.event_types:
            model = self.models[cause]
            total = total + getattr(model.dist, fname)(x, *model.params)
        return total

    # --- inference -------------------------------------------------------

    def _inference(
        self, cause: Any, warn: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(estimates, covariance)`` of a cause's parameters: its fit's,
        or the covariance a restored model saved. Without either (a
        ``how="MSE"`` fit, a model saved before the covariances were) it
        raises the cause's model's own refusal. ``warn=False`` leaves a
        covariance that could not be found as ``nan`` without the
        warning."""
        model = self.models[cause]
        if hasattr(model, "_neg_ll"):
            cov, found = model._covariance_matrix()
            estimates = np.asarray(model._mle, dtype=float)
        else:
            saved = getattr(self, "_covariances", {}).get(cause)
            if saved is None:
                model._check_fitted()
            cov = np.asarray(saved, dtype=float)
            found = bool(np.all(np.isfinite(cov)))
            estimates = np.asarray(model.params, dtype=float)
        if warn and not found:
            warn_no_covariance()
        return estimates, cov

    def _wald_cb(
        self,
        fname: str,
        x: np.ndarray,
        causes: Any,
        alpha_ci: float,
        bound: str,
        inference: dict | None = None,
    ) -> np.ndarray:
        """Log-scale Wald bounds on the sum over ``causes`` of the
        intensity function ``fname`` at the flat ``x``: the causes'
        delta-method variances added (their estimates are independent),
        then the single-process construction, so one cause's are its own
        model's."""
        x = np.atleast_1d(np.asarray(x, dtype=float))
        estimate = np.zeros(x.shape)
        variance = np.zeros(x.shape)
        for cause in causes:
            dist = self.models[cause].dist
            if inference is None:
                estimates, cov = self._inference(cause)
            else:
                estimates, cov = inference[cause]

            def curve(params: np.ndarray, dist: Any = dist) -> np.ndarray:
                return getattr(dist, fname)(x, *params)

            estimate = estimate + curve(estimates)
            variance = (
                variance
                + delta_method_se(curve, estimates, cov, scale_free=True) ** 2
            )
        return log_transformed_cb(estimate, np.sqrt(variance), alpha_ci, bound)

    def _has_likelihood(self, cause: Any) -> bool:
        return hasattr(self.models[cause], "_neg_ll")

    def cif_cb(
        self,
        x: ArrayLike,
        event: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the cumulative intensity of events of type
        ``event``: that cause's model's
        :meth:`~surpyval.recurrent.parametric.parametric_recurrence.ParametricRecurrenceModel.cif_cb`
        (the delta method on the log scale).

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        event : hashable
            The event type; one of ``event_types``.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.

        Returns
        -------
        numpy.ndarray
            The confidence bounds on the cause's cumulative intensity.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> model.cif_cb(60.0, "a").round(3)
        array([1.041, 6.006])
        >>> model.models["a"].cif_cb(60.0).round(3)
        array([1.041, 6.006])
        """
        self._check_cause(event)
        if self._has_likelihood(event):
            return self.models[event].cif_cb(x, alpha_ci, bound)
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        return self._saved_cb(x, "cif", event, alpha_ci, bound)

    @keeps_query_shape
    def _saved_cb(
        self, x: ArrayLike, fname: str, event: Any, alpha_ci: float, bound: str
    ) -> np.ndarray:
        """A cause's Wald bounds on a restored model, from its saved
        covariance (or its model's refusal without one)."""
        return self._wald_cb(
            fname, np.asarray(x, dtype=float), (event,), alpha_ci, bound
        )

    def iif_cb(
        self,
        x: ArrayLike,
        event: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> np.ndarray:
        """
        Confidence bounds on the instantaneous intensity of events of type
        ``event``: that cause's model's
        :meth:`~surpyval.recurrent.parametric.parametric_recurrence.ParametricRecurrenceModel.iif_cb`.

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        event : hashable
            The event type; one of ``event_types``.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.
        method : {'wald', 'crow'}, optional
            ``"wald"`` (default): the delta method on the log of the
            intensity. ``"crow"``: Crow's exact bounds, for a ``CrowAMSAA``
            cause at the end of a test of the designs the single-process
            method takes; it needs the data, so a restored model refuses
            it.

        Returns
        -------
        numpy.ndarray
            The confidence bounds on the cause's intensity.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> model.iif_cb(60.0, "a").round(4)
        array([0.0098, 0.1166])
        """
        self._check_cause(event)
        if method == "crow" or self._has_likelihood(event):
            return self.models[event].iif_cb(x, alpha_ci, bound, method)
        check_option("method", method, ("wald", "crow"))
        check_option("bound", bound, BOUNDS)
        check_alpha_ci(alpha_ci)
        return self._saved_cb(x, "iif", event, alpha_ci, bound)

    def mtbf(self, x: ArrayLike, event: Any) -> np.ndarray:
        """
        The instantaneous mean time between events of type ``event`` at
        ``x``, ``1 / iif(x, event)``: that cause's model's
        :meth:`~surpyval.recurrent.parametric.parametric_recurrence.ParametricRecurrenceModel.mtbf`.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> round(float(model.mtbf(60.0, "a")), 2)
        29.62
        """
        self._check_cause(event)
        return self.models[event].mtbf(x)

    def mtbf_cb(
        self,
        x: ArrayLike,
        event: Any,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
        method: str = "wald",
    ) -> np.ndarray:
        """
        Confidence bounds on the instantaneous MTBF of events of type
        ``event`` (:meth:`mtbf`): that cause's model's
        :meth:`~surpyval.recurrent.parametric.parametric_recurrence.ParametricRecurrenceModel.mtbf_cb`,
        the reciprocals of the bounds on its intensity, so
        ``bound="lower"`` is a lower bound on the MTBF.

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        event : hashable
            The event type; one of ``event_types``.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.
        method : {'wald', 'crow'}, optional
            As :meth:`iif_cb`.

        Returns
        -------
        numpy.ndarray
            The confidence bounds on the cause's MTBF.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> model.mtbf_cb(60.0, "a").round(2)
        array([  8.57, 102.3 ])
        """
        self._check_cause(event)
        if method == "crow" or self._has_likelihood(event):
            return self.models[event].mtbf_cb(x, alpha_ci, bound, method)
        check_option("method", method, ("wald", "crow"))
        check_option("bound", bound, BOUNDS)
        check_alpha_ci(alpha_ci)
        return ParametricRecurrenceModel._reciprocal_cb(
            self._saved_cb(x, "iif", event, alpha_ci, OPPOSITE_BOUND[bound]),
            bound,
        )

    @keeps_query_shape
    def total_cif_cb(
        self,
        x: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the total cumulative intensity
        (:meth:`total_cif`): the expected number of events of any type.

        The causes are independent Poisson processes, each fitted to its
        own events, so their estimates are independent and the variance of
        the total is the sum of the causes' delta-method variances. The
        bounds are then formed on the log scale, as each cause's are
        (:meth:`cif_cb`), so they cannot go negative, and a model with one
        cause has the total's bounds equal to that cause's.

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.

        Returns
        -------
        numpy.ndarray
            The confidence bounds on the total cumulative intensity.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> round(float(model.total_cif(60.0)), 3)
        4.5
        >>> model.total_cif_cb(60.0).round(3)
        array([2.341, 8.649])
        >>> lower = model.total_cif_cb(60.0, bound="lower", alpha_ci=0.1)
        >>> round(float(lower), 3)
        2.936
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        return self._wald_cb(
            "cif",
            np.asarray(x, dtype=float),
            self.event_types,
            alpha_ci,
            bound,
        )

    @keeps_query_shape
    def total_iif_cb(
        self,
        x: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the total instantaneous intensity
        (:meth:`total_iif`): the causes' delta-method variances added, and
        the bounds formed on the log scale, as :meth:`total_cif_cb`.

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.

        Returns
        -------
        numpy.ndarray
            The confidence bounds on the total intensity.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> model.total_iif_cb(60.0).round(4)
        array([0.0221, 0.1415])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        return self._wald_cb(
            "iif",
            np.asarray(x, dtype=float),
            self.event_types,
            alpha_ci,
            bound,
        )

    @keeps_query_shape
    def total_mtbf_cb(
        self,
        x: ArrayLike,
        alpha_ci: float = 0.05,
        bound: str = "two-sided",
    ) -> np.ndarray:
        """
        Confidence bounds on the instantaneous MTBF of events of any type
        (:meth:`total_mtbf`): the reciprocals of the bounds on the total
        intensity (:meth:`total_iif_cb`), so ``bound="lower"`` is a lower
        bound on the MTBF.

        Parameters
        ----------
        x : array_like
            Values at which to compute the confidence bounds.
        alpha_ci : float, optional
            The total tail probability of the bound(s). Default is 0.05.
        bound : {'two-sided', 'lower', 'upper'}, optional
            Two-sided bounds are returned as an ``(len(x), 2)`` array with
            columns ``[lower, upper]``; one-sided bounds have the shape of
            ``x``.

        Returns
        -------
        numpy.ndarray
            The confidence bounds on the MTBF of events of any type.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> model.total_mtbf_cb(60.0).round(2)
        array([ 7.07, 45.25])
        """
        check_alpha_ci(alpha_ci)
        check_option("bound", bound, BOUNDS)
        cb = self._wald_cb(
            "iif",
            np.asarray(x, dtype=float),
            self.event_types,
            alpha_ci,
            OPPOSITE_BOUND[bound],
        )
        return ParametricRecurrenceModel._reciprocal_cb(cb, bound)

    def summary(self, alpha_ci: float = 0.05) -> Any:
        """
        The fitted parameters of every cause with their standard errors
        and Wald intervals: the single-process ``summary`` of each cause's
        model, stacked, one row per cause and parameter.

        The intervals are those of ``param_cb`` (on the log scale for a
        positive parameter). A parameter whose variance is not positive
        (an estimate on a boundary) has ``nan`` for its standard error and
        interval.

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
        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> model.summary().round(3)  # doctest: +NORMALIZE_WHITESPACE
                         estimate      se  lower 95%  upper 95%
        event parameter
        a     alpha        19.369  14.497      4.467     83.987
              beta          0.810   0.362      0.337      1.947
        b     alpha        21.141  19.356      3.514    127.186
              beta          0.664   0.332      0.249      1.770
        """
        import pandas as pd

        check_alpha_ci(alpha_ci)
        level = f"{100 * (1 - alpha_ci):g}%"
        rows, events, names = [], [], []
        for cause in self.event_types:
            model = self.models[cause]
            estimates, cov = self._inference(cause, warn=False)
            for k, (value, (lo, hi)) in enumerate(
                zip(estimates, model.dist.bounds)
            ):
                events.append(cause)
                names.append(model.dist.parameter_names[k])
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

    def growth(self, x: float | None = None, alpha_ci: float = 0.05) -> Any:
        """
        Whether each cause is getting better, staying level or getting
        worse: one row per cause with its trend parameter, the test of no
        trend, and where the cause stands at ``x``.

        The trend of an intensity model is carried by one parameter:
        ``beta`` of ``CrowAMSAA`` and ``alpha`` of ``Duane`` (the intensity
        falls with time below 1, is constant at 1 and rises above it) and
        ``beta`` of ``CoxLewis`` (falls below 0, constant at 0, rises
        above). Each cause's row has that parameter's estimate and
        interval (``param_cb``), the Wald test that it equals its no-trend
        value, on the scale of the interval (the log of a positive
        parameter), and the conclusion at ``alpha_ci``; then the cause's
        intensity and MTBF at ``x`` with their bounds, and its share of
        the total intensity there.

        Parameters
        ----------
        x : float, optional
            The time at which the intensities, MTBFs and shares are read.
            By default the end of observation: the latest time any fitted
            item was observed to (kept by ``to_dict``).
        alpha_ci : float, optional
            The total tail probability of the intervals, and the level at
            which ``trend`` is judged. Default 0.05.

        Returns
        -------
        pandas.DataFrame
            Indexed by ``event``, with the columns

            - ``parameter``: the name of the trend parameter;
            - ``estimate``, ``lower <level>``, ``upper <level>``: its
              estimate and confidence interval;
            - ``no trend``: its value under a constant intensity;
            - ``p_value``: the two-sided p-value of the test that the
              parameter is at ``no trend``;
            - ``trend``: ``"increasing"`` (the intensity is rising: events
              of this type are becoming more frequent) or ``"decreasing"``
              when ``p_value < alpha_ci``, ``"none"`` otherwise, and
              ``"undetermined"`` where the parameter has no variance (an
              estimate on a boundary);
            - ``iif``, ``iif lower <level>``, ``iif upper <level>``: the
              cause's intensity at ``x`` and its bounds;
            - ``mtbf``, ``mtbf lower <level>``, ``mtbf upper <level>``:
              its MTBF at ``x`` and its bounds;
            - ``share``: the cause's share of the total intensity at ``x``.

        Raises
        ------
        ValueError
            If the model was fitted with ``dist=HPP``, whose intensity is
            constant by construction; if ``x`` is left out on a model with
            neither data nor a saved end of observation; and the refusal
            of a cause's model without a likelihood (``how="MSE"``).

        Notes
        -----
        ``trend="none"`` says the data do not show a trend, not that there
        is none: with few events of a cause the interval is wide.
        ``trend`` is ``"none"`` exactly when the interval contains the
        ``no trend`` value.

        Examples
        --------
        Two systems with events of types ``"a"`` and ``"b"``; with so few
        events neither shows a trend:

        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> table = model.growth()
        >>> columns = ["parameter", "estimate", "no trend", "p_value", "trend"]
        >>> table[columns].round(3)  # doctest: +NORMALIZE_WHITESPACE
              parameter  estimate  no trend  p_value trend
        event
        a          beta     0.810       1.0    0.638  none
        b          beta     0.664       1.0    0.414  none
        >>> table[["iif", "mtbf", "share"]].round(3)
        ... # doctest: +NORMALIZE_WHITESPACE
                 iif    mtbf  share
        event
        a      0.034  29.616  0.604
        b      0.022  45.147  0.396
        """
        import pandas as pd
        from scipy.stats import norm

        check_alpha_ci(alpha_ci)
        if self.dist.name not in _TREND_PARAMETER:
            raise ValueError(
                "growth() needs an intensity model with a trend parameter, "
                "and this model was fitted with dist={}, whose intensity is "
                "constant by construction. Refit with dist=CrowAMSAA (the "
                "default), Duane or CoxLewis to estimate each cause's "
                "trend; the causes' constant rates are in summary().".format(
                    type(self.dist).__name__
                )
            )
        if x is None:
            x = self._end_of_observation()
            if x is None:
                raise ValueError(
                    "x is required: this model carries neither data nor a "
                    "saved end of observation, so give x, the time at which "
                    "to read each cause's intensity."
                )
        at = np.asarray(x, dtype=float)
        if at.ndim != 0 or not np.isfinite(at):
            raise ValueError(
                f"x must be one finite time; got {x!r}. Call growth() once "
                "for each time."
            )
        when = np.atleast_1d(at)
        name, null = _TREND_PARAMETER[self.dist.name]
        level = f"{100 * (1 - alpha_ci):g}%"
        total = float(self._total("iif", when)[0])
        inference = {
            cause: self._inference(cause, warn=False)
            for cause in self.event_types
        }
        rows = []
        for cause in self.event_types:
            dist = self.models[cause].dist
            k = list(dist.parameter_names).index(name)
            estimates, cov = inference[cause]
            value, var = float(estimates[k]), float(cov[k, k])
            lo, hi = dist.bounds[k]
            _, lower, upper = parameter_interval(value, var, lo, hi, alpha_ci)
            p_value, trend = np.nan, "undetermined"
            if np.isfinite(lower):
                if lo is None:
                    z = (value - null) / np.sqrt(var)
                else:
                    z = (
                        np.log((value - lo) / (null - lo))
                        * (value - lo)
                        / np.sqrt(var)
                    )
                p_value = float(2.0 * norm.sf(abs(z)))
                trend = "none"
                if p_value < alpha_ci:
                    trend = "increasing" if value > null else "decreasing"
            rate = float(dist.iif(when, *self.models[cause].params)[0])
            cb = self._wald_cb(
                "iif", when, (cause,), alpha_ci, "two-sided", inference
            )[0]
            with np.errstate(divide="ignore", invalid="ignore"):
                rows.append(
                    [
                        name,
                        value,
                        lower,
                        upper,
                        null,
                        p_value,
                        trend,
                        rate,
                        cb[0],
                        cb[1],
                        1.0 / rate,
                        1.0 / cb[1],
                        1.0 / cb[0],
                        rate / total,
                    ]
                )
        return pd.DataFrame(
            rows,
            index=pd.Index(
                event_labels(self.event_types),
                name="event",
                tupleize_cols=False,
            ),
            columns=[
                "parameter",
                "estimate",
                f"lower {level}",
                f"upper {level}",
                "no trend",
                "p_value",
                "trend",
                "iif",
                f"iif lower {level}",
                f"iif upper {level}",
                "mtbf",
                f"mtbf lower {level}",
                f"mtbf upper {level}",
                "share",
            ],
        )

    # --- forecasting -----------------------------------------------------

    def _superposed(self) -> _AllCauses:
        return _AllCauses(self)

    def forecast(
        self,
        age: ArrayLike | None = None,
        horizon: ArrayLike | None = None,
        n: ArrayLike | None = None,
        limit: ArrayLike | None = None,
        alpha_ci: float = 0.05,
        event: Any = None,
    ) -> Any:
        """
        Forecast the events of units in service over a time ahead:
        :func:`surpyval.forecast` for this model, of events of type
        ``event`` or, without one, of events of any type.

        A unit at age :math:`a` has a Poisson number of events of a cause
        over the next :math:`h`, with mean :math:`\\Lambda_k(a + h) -
        \\Lambda_k(a)`. The causes are independent, so its number of
        events of any type is Poisson too, with the sum of the causes'
        means, and so is the count over all the units. The forecast gives
        the expected count and the prediction interval of the count, for
        each horizon and each period between horizons.

        Parameters
        ----------
        age : array_like, optional
            The current age of each unit (its time since new), at least 0.
            For a model fitted to data it may be left out: the units are
            then the fitted items, each at the end of its observation.
        horizon : scalar or array_like
            The time ahead to forecast over, or several increasing times
            ahead. Required.
        n : array_like, optional
            The number of units at each age; 1 each by default.
        limit : scalar or array_like, optional
            An age past which a unit's events are not counted.
        alpha_ci : float, optional
            The significance level of the prediction intervals (default
            0.05, a 95% interval).
        event : hashable, optional
            The event type to forecast; one of ``event_types``. By default
            events of every type are counted.

        Returns
        -------
        surpyval.Forecast
            As :func:`surpyval.forecast` returns for a Poisson process:
            ``expected``, ``lower`` and ``upper`` for each horizon, the
            same for each period, and each unit's expected count
            (``per_unit``).

        Raises
        ------
        ValueError
            As :func:`surpyval.forecast`; if ``age`` is left out on a model
            that carries no data (one restored with ``from_dict``); and
            for an ``event`` the model does not have.

        Notes
        -----
        The interval is that of the count given the fitted intensities: it
        does not carry the uncertainty of the fit, which
        :meth:`total_cif_cb` and :meth:`cif_cb` show, so with little data
        it is somewhat too narrow. The causes' expected counts add up to
        the forecast of every type; their interval ends do not.

        Examples
        --------
        The next 20 time units of the two fitted systems, each from the
        end of its observation at 60:

        >>> from surpyval.recurrent import CauseSpecificNHPP
        >>> x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
        >>> i = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        >>> e = ["a", "b", "a", "b", "a", None, "b", "a", "b", "a", None]
        >>> model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e)
        >>> every = model.forecast(horizon=20)
        >>> every.expected.round(3), every.lower, every.upper
        (array([2.155]), array([0.]), array([5.]))
        >>> a = model.forecast(horizon=20, event="a")
        >>> b = model.forecast(horizon=20, event="b")
        >>> (a.expected + b.expected).round(3)
        array([2.155])

        A new unit's first 20 time units:

        >>> model.forecast(age=0.0, horizon=20).expected.round(3)
        array([1.99])
        """
        from surpyval.forecasting import forecast

        if event is None:
            return forecast(
                self, age, horizon, n=n, limit=limit, alpha_ci=alpha_ci
            )
        self._check_cause(event)
        return forecast(
            self.models[event],
            age,
            horizon,
            n=n,
            limit=limit,
            alpha_ci=alpha_ci,
        )

    def plot(self, ax: Any = None) -> Any:
        """Overlay the fitted cause-specific CIFs on a single axis, over
        the observed time range of the data they were fitted to."""
        require_data(self, "plot")
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        x_plot = np.linspace(0, float(self.data.x.max()), 200)
        for cause in self.event_types:
            ax.plot(
                x_plot,
                self.cif(x_plot, cause),
                label=str(cause),
                color=plotting.next_color(ax),
                linewidth=(
                    plotting.LINE_WIDTH if plotting.theme_enabled() else None
                ),
            )
        plotting.style_axes(ax)
        plotting.style_legend(ax, title="Cause")
        label_axes(ax, getattr(self, "unit", None), "Mean cumulative function")
        return ax
