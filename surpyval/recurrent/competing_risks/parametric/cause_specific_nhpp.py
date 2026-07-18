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

from matplotlib import pyplot as plt

from surpyval.recurrent.parametric.crow_amsaa import CrowAMSAA
from surpyval.utils.recurrent_utils import handle_xicn


class CauseSpecificNHPP:
    """
    Parametric cause-specific intensity model for a recurrent process with
    competing event types.

    One NHPP intensity model (``CrowAMSAA`` by default, or any counting-process
    fitter) is fitted per event type, sharing each item's observation window
    across causes. Access the per-cause fitted models through
    ``self.models[cause]`` -- each is an ordinary
    :class:`ParametricRecurrenceModel` with its full ``cif``/``iif``/inference/
    diagnostic behaviour -- or use the convenience methods below.
    """

    def __repr__(self):
        return "Cause-specific {} with causes: {}".format(
            self.dist.name, self.event_types
        )

    # --- per-item observation windows ------------------------------------

    @staticmethod
    def _item_window(data, item):
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
        cls, data, dist=CrowAMSAA, how="MLE", init=None
    ):
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
            is_cause = [ev == cause for ev in data.e]
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

            cause_data = handle_xicn(
                cx, ci, cc, tl=ctl, as_recurrent_data=True
            )
            out.models[cause] = dist.fit_from_recurrent_data(
                cause_data, how=how, init=init
            )
        return out

    @classmethod
    def fit(
        cls,
        x,
        i=None,
        c=None,
        n=None,
        e=None,
        tl=None,
        tr=None,
        dist=CrowAMSAA,
        how="MLE",
        init=None,
    ):
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
            Count of events at each row. Defaults to 1.
        e : array like
            Event type (mark) for each row. ``None``/``NaN`` for censored rows.
        tl : array like or scalar, optional
            Left-truncation (delayed-entry) time per item.
        tr : array like or scalar, optional
            Right-truncation time per item.
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
        data = handle_xicn(
            x, i, c, n, tl=tl, tr=tr, e=e, as_recurrent_data=True
        )
        return cls.fit_from_recurrent_data(data, dist=dist, how=how, init=init)

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
        dist=CrowAMSAA,
        how="MLE",
        init=None,
    ):
        """
        Fit a cause-specific intensity model from a :class:`pandas.DataFrame`,
        naming the columns to read. See :meth:`fit` for the meaning of each.
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
            dist=dist,
            how=how,
            init=init,
        )
        model.df = df
        return model

    # --- evaluation ------------------------------------------------------

    def _check_cause(self, cause):
        if cause not in self.models:
            raise ValueError(
                "Unrecognised cause {!r}; known causes are {}".format(
                    cause, self.event_types
                )
            )

    def cif(self, x, cause):
        """Cause-specific cumulative intensity (expected ``cause`` count)."""
        self._check_cause(cause)
        return self.models[cause].cif(x)

    def iif(self, x, cause):
        """Cause-specific instantaneous intensity for ``cause``."""
        self._check_cause(cause)
        return self.models[cause].iif(x)

    def mcf(self, x, cause):
        """Cause-specific mean cumulative function (alias of :meth:`cif`)."""
        return self.cif(x, cause)

    def total_cif(self, x):
        """
        Total cumulative intensity across all causes -- the expected number of
        events of any type, which (the causes being independent thinnings of
        the overall process) is the sum of the cause-specific intensities.
        """
        total = None
        for cause in self.event_types:
            contribution = self.models[cause].cif(x)
            total = contribution if total is None else total + contribution
        return total

    def plot(self, ax=None):
        """Overlay the fitted cause-specific CIFs on a single axis."""
        import numpy as np

        if ax is None:
            ax = plt.gcf().gca()
        x_plot = np.linspace(0, float(self.data.x.max()), 200)
        for cause in self.event_types:
            ax.plot(x_plot, self.cif(x_plot, cause), label=str(cause))
        ax.legend()
        return ax
