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

import json

from autograd import numpy as np
from matplotlib import pyplot as plt

from surpyval.recurrent.nonparametric.mcf import NonParametricCounting
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_unsupported_nonparametric,
)


def _counting_model_from_xrd(x, r, d):
    """
    Build a ``NonParametricCounting`` model from an explicit ``(x, r, d)``
    triple so its ``mcf``/``mcf_cb``/``plot`` machinery can be reused for a
    single cause. Mirrors the estimator in
    ``NonParametricCounting.fit_from_recurrent_data``.
    """
    out = type(NonParametricCounting)()
    out.x, out.r, out.d = x, r, d
    out.mcf_hat = np.cumsum(d / r)
    var = 1.0 / r**2 * (d * (1 - 1.0 / r) ** 2 + (r - d) * (0 - 1.0 / r) ** 2)
    var = (d > 0).astype(int) * var
    out.var = np.cumsum(var)
    return out


class CauseSpecificMCF:
    """
    Cause-specific Mean Cumulative Function for a recurrent process with
    competing event types.

    The model fits one ``NonParametricCounting`` MCF per event type, sharing
    the at-risk set across causes. Access the per-cause models through
    ``self.models[cause]`` or use the convenience methods below.
    """

    def __repr__(self):
        return "Cause-specific MCF with causes: {}".format(self.event_types)

    # -- serialisation -----------------------------------------------------

    def to_dict(self):
        """
        Serialise this fitted cause-specific MCF to a plain, JSON-serialisable
        dict: the list of event types and each cause's per-cause MCF estimate.

        See Also
        --------
        from_dict, to_json, from_json
        """
        return {
            "model": "CauseSpecificMCF",
            "event_types": list(self.event_types),
            "models": [
                self.models[cause].to_dict() for cause in self.event_types
            ],
        }

    def to_json(self, fp):
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, model_dict):
        """
        Rebuild a cause-specific MCF from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        if model_dict.get("model") != "CauseSpecificMCF":
            raise ValueError(
                "Must create a cause-specific MCF from a CauseSpecificMCF dict"
            )
        out = cls()
        out.event_types = list(model_dict["event_types"])
        out.models = {
            cause: NonParametricCounting.from_dict(sub)
            for cause, sub in zip(out.event_types, model_dict["models"])
        }
        return out

    @classmethod
    def from_json(cls, fp):
        """Load a cause-specific MCF from a JSON file written by
        :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    def mcf(self, x, cause, interp="step"):
        """Cause-specific MCF evaluated at ``x`` for the given ``cause``."""
        return self.models[cause].mcf(x, interp=interp)

    def mcf_cb(self, x, cause, **kwargs):
        """Confidence bounds on the cause-specific MCF for ``cause``."""
        return self.models[cause].mcf_cb(x, **kwargs)

    def plot(self, confidence=0.95, plot_bounds=True, ax=None):
        """Overlay the MCF of every cause on a single axis."""
        if ax is None:
            ax = plt.gcf().gca()
        for cause in self.event_types:
            model = self.models[cause]
            ax.step(model.x, model.mcf_hat, where="post", label=str(cause))
        ax.legend()
        return ax

    @classmethod
    def fit_from_recurrent_data(cls, data):
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
            x, r, d = data.to_cause_specific_xrd(cause)
            out.models[cause] = _counting_model_from_xrd(x, r, d)
        return out

    @classmethod
    def fit(cls, x, i=None, c=None, n=None, e=None, tl=None, tr=None):
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
        tl : array like or scalar, optional
            Left-truncation (delayed-entry) time per item. The at-risk set
            is shared across causes, so a delayed entry shrinks the risk set
            for every cause until the item enters at ``tl``.
        tr : array like or scalar, optional
            Right-truncation time per item.

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
        data = handle_xicn(
            x, i, c, n, tl=tl, tr=tr, e=e, as_recurrent_data=True
        )
        return cls.fit_from_recurrent_data(data)

    @classmethod
    def fit_from_df(
        cls,
        df,
        x_col,
        e_col,
        i_col=None,
        c_col=None,
        n_col=None,
        tl_col=None,
        tr_col=None,
    ):
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

        def col(name):
            return None if name is None else df[name].to_numpy()

        model = cls.fit(
            x=df[x_col].to_numpy(),
            i=col(i_col),
            c=col(c_col),
            n=col(n_col),
            e=col(e_col),
            tl=col(tl_col),
            tr=col(tr_col),
        )
        model.df = df
        return model
