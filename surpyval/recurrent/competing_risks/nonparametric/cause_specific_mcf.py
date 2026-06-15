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

from autograd import numpy as np
from matplotlib import pyplot as plt

from surpyval.recurrent.nonparametric.mcf import NonParametricCounting
from surpyval.utils.recurrent_event_data import RecurrentEventData
from surpyval.utils.recurrent_utils import reject_unsupported_nonparametric


def _counting_model_from_xrd(x, r, d):
    """
    Build a ``NonParametricCounting`` model from an explicit ``(x, r, d)``
    triple so its ``mcf``/``mcf_cb``/``plot`` machinery can be reused for a
    single cause. Mirrors the estimator in
    ``NonParametricCounting.fit_from_recurrent_data``.
    """
    out = NonParametricCounting()
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
                "`e` (event types) is required for a " "cause-specific MCF."
            )
        x = np.asarray(x)
        if i is None:
            i = np.ones_like(x, dtype=int)
        if c is None:
            c = np.zeros_like(x, dtype=int)
        if n is None:
            n = np.ones_like(x, dtype=int)
        # RecurrentEventData expects per-row truncation bounds, so broadcast
        # a scalar tl/tr to the full length here (handle_xicn does this via
        # format_truncation, but the mark column means we build the data
        # object directly).
        if tl is not None:
            tl = np.broadcast_to(np.asarray(tl, dtype=float), x.shape).copy()
        if tr is not None:
            tr = np.broadcast_to(np.asarray(tr, dtype=float), x.shape).copy()
        data = RecurrentEventData(x, i, c, n, e, tl=tl, tr=tr)
        return cls.fit_from_recurrent_data(data)
