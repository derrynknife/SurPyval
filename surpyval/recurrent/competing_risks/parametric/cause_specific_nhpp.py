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

from surpyval.recurrent.inference import require_data
from surpyval.recurrent.parametric.crow_amsaa import CrowAMSAA
from surpyval.recurrent.parametric.parametric_recurrence import (
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
from surpyval.utils.no_maximum import combined_maximum
from surpyval.utils.recurrent_utils import handle_xicn
from surpyval.utils.removed_names import column_arguments
from surpyval.utils.validation import unknown_cause_error


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

    def _ic_terms(self) -> "tuple[int, float]":
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
        event types, and each cause's fitted intensity model.

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
            }
        )

    @classmethod
    def from_dict(cls, model_dict: dict) -> "CauseSpecificNHPP":
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
        tr_item = float(data.tr[mask][0])
        if (c_item == 1).any():
            end = float(x_upper[c_item == 1][0])
        elif tr_item == tr_item and tr_item != float("inf"):
            end = tr_item
        else:
            end = float(x_upper.max())
        return entry, end

    @classmethod
    def fit_from_recurrent_data(
        cls,
        data: Any,
        dist: Any = CrowAMSAA,
        how: str = "MLE",
        init: "ArrayLike | None" = None,
    ) -> "CauseSpecificNHPP":
        """
        Fit the cause-specific intensity model from prepared
        :class:`RecurrentEventData` carrying event-type marks ``e``.

        Parameters
        ----------
        data : RecurrentEventData
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
                "RecurrentEventData has no event-type marks; pass `e` to "
                "fit a cause-specific intensity model."
            )
        if data.x.ndim != 1:
            raise ValueError(
                "Cause-specific intensity models require exact (1D) event "
                "times."
            )
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
        windows = {item: cls._item_window(data, item) for item in data.items}

        out.models = {}
        for cause in out.event_types:
            cx, ci, cc, ctl = [], [], [], []
            is_cause = label_mask(data.e, cause)
            for k, item in enumerate(data.i):
                if data.c[k] == 0 and is_cause[k]:
                    entry, _ = windows[item]
                    cx.append(float(data.x[k]))
                    ci.append(item)
                    cc.append(0)
                    ctl.append(entry)
            # One right-censored window-close per item (present for every item,
            # even those with no events of this cause, so the compensator is
            # integrated over the whole window).
            for item in data.items:
                entry, end = windows[item]
                cx.append(end)
                ci.append(item)
                cc.append(1)
                ctl.append(entry)

            cause_data = handle_xicn(cx, ci, cc, tl=ctl)
            out.models[cause] = dist.fit_from_recurrent_data(
                cause_data, how=how, init=init
            )
        return out

    @classmethod
    def fit(
        cls,
        x: ArrayLike,
        i: "ArrayLike | None" = None,
        c: "ArrayLike | None" = None,
        n: "ArrayLike | None" = None,
        e: "ArrayLike | None" = None,
        tl: "ArrayLike | None" = None,
        tr: "ArrayLike | None" = None,
        dist: Any = CrowAMSAA,
        how: str = "MLE",
        init: "ArrayLike | None" = None,
    ) -> "CauseSpecificNHPP":
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
            every item, or one value per row (the same on every row of an
            item).
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
    @column_arguments("x", "i", "c", "n", "tl", "tr")
    def fit_from_df(
        cls,
        df: Any,
        x_col: str,
        e_col: str,
        i_col: "str | None" = None,
        c_col: "str | None" = None,
        n_col: "str | None" = None,
        tl_col: "str | None" = None,
        tr_col: "str | None" = None,
        dist: Any = CrowAMSAA,
        how: str = "MLE",
        init: "ArrayLike | None" = None,
    ) -> "CauseSpecificNHPP":
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
        total: "np.ndarray | None" = None
        for cause in self.event_types:
            contribution = self.models[cause].cif(x)
            total = contribution if total is None else total + contribution
        assert total is not None  # event_types is never empty on a fit model
        return total

    def plot(self, ax: Any = None) -> Any:
        """Overlay the fitted cause-specific CIFs on a single axis, over
        the observed time range of the data they were fitted to."""
        require_data(self, "plot")
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        x_plot = np.linspace(0, float(self.data.x.max()), 200)
        for cause in self.event_types:
            ax.plot(x_plot, self.cif(x_plot, cause), label=str(cause))
        ax.legend()
        return ax
