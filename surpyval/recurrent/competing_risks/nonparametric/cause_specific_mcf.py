"""
Cause-specific Mean Cumulative Function (MCF) for recurrent events with
competing failure modes.

This is the recurrent-process analogue of the univariate competing-risks
CIF: a single repairable item can experience events of several mutually
exclusive types over time, and we want a separate MCF per type. The
at-risk set is shared across causes (an item is at risk for every cause
until it leaves observation); only the event counts are split by cause.

See ``surpyval.univariate.competing_risks`` for the univariate
(time-to-first-event) competing-risks models.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from surpyval.recurrent.nonparametric.mcf import (
    _MCF_RANGE,
    NonParametricCounting,
    _lawless_nadeau_var,
    _observation_origin,
)
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
from surpyval.univariate.nonparametric.nonparametric import (
    _check_support,
    _support_from_dict,
)
from surpyval.utils import optional_column
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_unsupported_nonparametric,
)


def _cause_model(data: Any, cause: Any) -> Any:
    """Single-cause ``NonParametricCounting`` for ``cause``: the shared
    Nelson-Aalen estimator on the cause's counts over the shared risk set,
    with the Lawless-Nadeau robust variance of those counts (the per-step
    variance ``from_xrd`` computes ignores each item's covariance across
    steps, so it understates the variance when items differ in their
    rates)."""
    x, r, d = data.to_cause_specific_xrd(cause)
    # ``from_xrd`` is a classmethod, so calling it through the
    # singleton instance binds the class exactly as ``type(...)`` did.
    model = NonParametricCounting.from_xrd(x, r, d)
    # Only this cause's events count; the other causes' events are
    # non-events for it, while each item stays in the (shared) risk set.
    model.var = _lawless_nadeau_var(
        data, x, r, d, counted=label_mask(data.e, cause)
    )
    model.origin = _observation_origin(data)
    return model


class CauseSpecificMCF(SerialisableMixin):
    """
    Cause-specific Mean Cumulative Function for a recurrent process with
    competing event types.

    The model fits one ``NonParametricCounting`` MCF per event type, sharing
    the at-risk set across causes. Each cause's MCF carries the
    Lawless-Nadeau robust variance of that cause's events (the other
    causes' events count as non-events for it), as the overall MCF does.
    Access the per-cause models through ``self.models[cause]`` or use the
    convenience methods below.

    Examples
    --------
    Two pumps, each repaired for seal or motor failures and observed to
    times 10 and 12 (the ``c=1`` rows, which have no event type):

    >>> from surpyval.recurrent import CauseSpecificMCF
    >>> x = [2, 5, 7, 10, 3, 4, 8, 12]
    >>> i = [1, 1, 1, 1, 2, 2, 2, 2]
    >>> c = [0, 0, 0, 1, 0, 0, 0, 1]
    >>> e = ["seal", "motor", "seal", None, "seal", "seal", "motor", None]
    >>> model = CauseSpecificMCF.fit(x, i=i, c=c, e=e)
    >>> model
    Cause-specific MCF with causes: ['motor', 'seal']

    The mean number of seal repairs per pump by times 4 and 10:

    >>> model.mcf([4, 10], "seal")
    array([1.5, 2. ])
    """

    # Populated by the fit classmethods; declared for the type checker.
    df: Any
    data: Any
    event_types: list
    models: dict
    x: "np.ndarray"
    r: "np.ndarray"
    #: The ``(lower, upper)`` interval the MCFs are defined on, set by
    #: :meth:`set_support`; ``None`` (the default) when it has not been set.
    support: "tuple[float, float] | None" = None

    def __repr__(self) -> str:
        return "Cause-specific MCF with causes: {}".format(self.event_types)

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted cause-specific MCF to a plain, JSON-serialisable
        dict: the list of event types and each cause's per-cause MCF estimate.

        See Also
        --------
        from_dict, to_json, from_json
        """
        out = {
            "model": "CauseSpecificMCF",
            "event_types": to_native(list(self.event_types)),
            "models": [
                self.models[cause].to_dict() for cause in self.event_types
            ],
        }
        # Only when set: without it the dictionary is readable by v0.20.
        if self.support is not None:
            out["support"] = [float(v) for v in self.support]
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "CauseSpecificMCF":
        """
        Rebuild a cause-specific MCF from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        require_model_tag(
            model_dict, "CauseSpecificMCF", "a cause-specific MCF"
        )
        out = cls()
        # JSON writes a tuple label as a list; turn it back into a tuple.
        out.event_types = [
            label_from_native(v) for v in model_dict["event_types"]
        ]
        out.models = {
            cause: NonParametricCounting.from_dict(sub)
            for cause, sub in zip(out.event_types, model_dict["models"])
        }
        support = _support_from_dict(model_dict)
        if support is not None:
            out.set_support(*support)
        return out

    def set_support(self, lower: float, upper: float) -> "CauseSpecificMCF":
        """
        Give every cause's MCF the explicit support ``[lower, upper]``.

        Each cause's :meth:`mcf` and :meth:`mcf_cb` are then 0 from
        ``lower`` to the origin (where observation begins), the value at
        the last observed time from there to ``upper``, and NaN outside
        them, instead of NaN before the origin and after the last observed
        time; see ``NonParametricCounting.set_support``. The bounds are
        kept by ``to_dict``.

        Parameters
        ----------
        lower : float
            The lower end of the support; at most the origin.
        upper : float
            The upper end; at least the last observed time, and above
            ``lower``.

        Returns
        -------
        CauseSpecificMCF
            The model itself, so the call can be chained.

        Raises
        ------
        ValueError
            If a bound is NaN or not a number, ``lower`` is not below
            ``upper``, or the bounds do not contain ``[origin, last]``.

        Examples
        --------
        >>> from surpyval.recurrent import CauseSpecificMCF
        >>> x = [2, 5, 7, 10, 3, 4, 8, 12]
        >>> i = [1, 1, 1, 1, 2, 2, 2, 2]
        >>> c = [0, 0, 0, 1, 0, 0, 0, 1]
        >>> e = ["seal", "motor", "seal", None, "seal", "seal", "motor", None]
        >>> model = CauseSpecificMCF.fit(x, i=i, c=c, e=e)
        >>> model.mcf([-1, 4, 15], "seal")
        array([nan, 1.5, nan])
        >>> model.set_support(-5, 20).mcf([-10, -1, 4, 15, 25], "seal")
        array([nan, 0. , 1.5, 2. , nan])
        """
        # The causes share the risk set, so their grids and origins agree;
        # checked against their union all the same.
        models = [self.models[cause] for cause in self.event_types]
        support = _check_support(
            lower,
            upper,
            min(m._origin() for m in models),
            max(float(m.x.max()) for m in models),
            _MCF_RANGE,
        )
        for model in models:
            model.support = support
        self.support = support
        return self

    def mcf(
        self, x: ArrayLike, event: Any, interp: str = "step"
    ) -> np.ndarray:
        """Cause-specific MCF evaluated at ``x`` for the event type
        ``event`` (see ``NonParametricCounting.mcf``, and
        :meth:`set_support` for its values outside the data)."""
        return self.models[event].mcf(x, interp=interp)

    def mcf_cb(self, x: ArrayLike, event: Any, **kwargs: Any) -> Any:
        """Confidence bounds on the cause-specific MCF for the event type
        ``event``; ``kwargs`` are those of
        ``NonParametricCounting.mcf_cb``."""
        return self.models[event].mcf_cb(x, **kwargs)

    def plot(
        self,
        *,
        alpha_ci: float = 0.05,
        plot_bounds: bool = True,
        ax: Any = None,
    ) -> Any:
        """Overlay the MCF of every cause on a single axis.

        With ``plot_bounds`` each cause's pointwise two-sided
        ``1 - alpha_ci`` bounds are drawn as dashed steps in the colour of
        its MCF. The arguments are keyword only.
        """
        if ax is None:
            import matplotlib.pyplot as plt

            ax = plt.gcf().gca()
        for cause in self.event_types:
            model = self.models[cause]
            (line,) = ax.step(
                model.x, model.mcf_hat, where="post", label=str(cause)
            )
            if plot_bounds and model.var is not None:
                cb = model.mcf_cb(model.x, alpha_ci=alpha_ci)
                ax.step(
                    model.x,
                    cb,
                    where="post",
                    color=line.get_color(),
                    linestyle="--",
                    linewidth=0.8,
                )
        ax.legend()
        return ax

    @classmethod
    def fit_from_recurrent_data(cls, data: Any) -> "CauseSpecificMCF":
        if data.e is None:
            raise ValueError(
                "RecurrentEventData has no event-type marks; pass `e` to "
                "fit a cause-specific MCF."
            )
        reject_unsupported_nonparametric(data, "CauseSpecificMCF")
        out = cls()
        out.data = data
        out.event_types = data.event_types
        out.x, out.r, _ = data.to_xrd()
        out.models = {}
        for cause in out.event_types:
            out.models[cause] = _cause_model(data, cause)
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
    ) -> "CauseSpecificMCF":
        """
        Fit a cause-specific MCF.

        Parameters
        ----------
        x : array like
            Event (and censoring) times.
        i : array like, optional
            Item / subject id for each row. Defaults to a single item.
        c : array like, optional
            Censoring flag for each row (0 observed, 1 right censored).
        n : array like, optional
            Count of events at each row. Defaults to 1.
        e : array like
            Event type (mark) for each row. ``None`` for censored rows.
            A mark may be any hashable label: an integer, a string, a
            tuple, or a mix of these.
        tl : array like or scalar, optional
            Left-truncation (delayed-entry) time of each item: a scalar for
            every item, or one value per row (the same on every row of an
            item). The at-risk set
            is shared across causes, so a delayed entry shrinks the risk set
            for every cause until the item enters at ``tl``.
        tr : array like or scalar, optional
            Right-truncation time of each item, given like ``tl``: the end of
            its observation
            window. The item stays in the (shared) at-risk set up to ``tr``,
            exactly as if it had an end-of-observation (``c=1``) row there.

        Returns
        -------
        CauseSpecificMCF
        """
        if e is None:
            raise ValueError(
                "`e` (event types) is required for a cause-specific MCF."
            )
        # Route through the shared recurrent handler so the marked data gets
        # the same validation, sorting and (scalar or per-row) truncation
        # handling as every other recurrent fit.
        data = handle_xicn(x, i, c, n, tl=tl, tr=tr, e=e)
        return cls.fit_from_recurrent_data(data)

    @classmethod
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
    ) -> "CauseSpecificMCF":
        """
        Fit a cause-specific MCF from a :class:`pandas.DataFrame`, naming the
        columns to read.

        Parameters
        ----------
        df : pandas.DataFrame
            The data.
        x_col : str
            Column of event (and censoring) times.
        e_col : str
            Column of event-type marks. Use ``None`` (or ``NaN``) marks for
            censored rows.
        i_col : str, optional
            Column of item / subject ids. Defaults to a single item.
        c_col : str, optional
            Column of censoring flags (0 observed, 1 right censored).
        n_col : str, optional
            Column of event counts per row.
        tl_col, tr_col : str, optional
            Columns of per-row left / right truncation bounds.

        Returns
        -------
        CauseSpecificMCF
        """

        model = cls.fit(
            x=df[x_col].to_numpy(),
            i=optional_column(df, i_col),
            c=optional_column(df, c_col),
            n=optional_column(df, n_col),
            e=optional_column(df, e_col),
            tl=optional_column(df, tl_col),
            tr=optional_column(df, tr_col),
        )
        model.df = df
        return model
