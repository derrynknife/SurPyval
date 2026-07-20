import json

from autograd import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm

from surpyval.utils.fitter import singleton_fitter
from surpyval.utils.recurrent_utils import (
    handle_xicn,
    reject_unsupported_nonparametric,
)
from surpyval.serialisation import stamp_schema


@singleton_fitter
class NonParametricCounting:
    # -- serialisation -----------------------------------------------------

    def to_dict(self):
        """
        Serialise this fitted MCF (mean cumulative function) estimate to a
        plain, JSON-serialisable dict.

        Stores the step arrays that ``mcf``/``mcf_cb`` read: the event times
        ``x``, the estimate ``mcf_hat`` and its Greenwood variance ``var``.
        The raw ``data`` is not stored (it is only needed to re-fit or to plot
        raw counts).

        See Also
        --------
        from_dict, to_json, from_json
        """
        return stamp_schema(
            {
                "model": "NonParametricCounting",
                "x": np.asarray(self.x, dtype=float).tolist(),
                "mcf_hat": np.asarray(self.mcf_hat, dtype=float).tolist(),
                "var": np.asarray(self.var, dtype=float).tolist(),
            }
        )

    def to_json(self, fp):
        """Write :meth:`to_dict` to ``fp`` as JSON."""
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, model_dict):
        """
        Rebuild an MCF estimate from a :meth:`to_dict` dictionary.

        See Also
        --------
        to_dict, to_json, from_json
        """
        if model_dict.get("model") != "NonParametricCounting":
            raise ValueError(
                "Must create an MCF estimate from a NonParametricCounting dict"
            )
        out = cls()
        out.x = np.array(model_dict["x"], dtype=float)
        out.mcf_hat = np.array(model_dict["mcf_hat"], dtype=float)
        out.var = np.array(model_dict["var"], dtype=float)
        return out

    @classmethod
    def from_json(cls, fp):
        """Load an MCF estimate from a JSON file written by :meth:`to_json`."""
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))

    def mcf(self, x, interp="step"):
        x = np.atleast_1d(x)
        # Let's not assume we can predict above the highest measurement
        if interp == "step":
            idx = np.searchsorted(self.x, x, side="right") - 1
            mcf = self.mcf_hat[idx]
            mcf[np.where(x < self.x.min())] = 0
            mcf[np.where(x > self.x.max())] = np.nan
            mcf[np.where(x < 0)] = np.nan
            return mcf
        elif interp == "linear":
            mcf = np.hstack([[0], self.mcf_hat])
            x_data = np.hstack([[0], self.x])
            mcf = np.interp(x, x_data, mcf)
            mcf[np.where(x > self.x.max())] = np.nan
            return mcf
        else:
            raise ValueError("`interp` must be either 'step' or 'linear'")

    def mcf_cb(
        self,
        x,
        bound="two-sided",
        interp="step",
        confidence=0.95,
        bound_type="exp",
        dist="z",
    ):
        # Greenwood's variance with a normal (z) critical value. Ref found:
        # http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis
        if bound_type not in ["exp", "normal"]:
            raise ValueError("'bound_type' must be in ['exp', 'normal']")
        if dist != "z":
            raise ValueError(
                "'dist' must be 'z'. The 't' option (Student-t with the "
                "at-risk count as degrees of freedom) has been removed: it "
                "had no asymptotic justification, was undefined once the "
                "risk set fell to one item, and widened the bounds "
                "arbitrarily as the risk set shrank. The normal ('z') "
                "critical value is what the asymptotic theory of the MCF "
                "estimator justifies."
            )
        x = np.atleast_1d(x)
        if bound in ["upper", "lower"]:
            stat = norm.ppf(1 - confidence, 0, 1)
            if bound == "upper":
                stat = -stat
        elif bound == "two-sided":
            stat = norm.ppf((1 - confidence) / 2, 0, 1)
            stat = np.array([-1, 1]).reshape(2, 1) * stat

        if bound_type == "exp":
            # Exponential Greenwood confidence
            mcf_cb = self.mcf_hat * np.exp(
                stat * np.sqrt(self.var) / self.mcf_hat
            )
        else:
            # Normal Greenwood confidence
            mcf_cb = self.mcf_hat + np.sqrt(self.var * self.mcf_hat**2) * stat
        # Let's not assume we can predict above the highest measurement
        if interp == "step":
            mcf_cb[np.where(x < self.x.min())] = 0
            mcf_cb[np.where(x > self.x.max())] = np.nan
            mcf_cb[np.where(x < 0)] = np.nan
            idx = np.searchsorted(self.x, x, side="right") - 1
            if bound == "two-sided":
                mcf_cb = mcf_cb[:, idx].T
            else:
                mcf_cb = mcf_cb[idx]
        elif interp == "linear":
            if bound == "two-sided":
                R1 = np.interp(x, self.x, mcf_cb[0, :])
                R2 = np.interp(x, self.x, mcf_cb[1, :])
                mcf_cb = np.vstack([R1, R2]).T
            else:
                mcf_cb = np.interp(x, self.x, mcf_cb)
            mcf_cb[np.where(x > self.x.max())] = np.nan
        return mcf_cb

    def plot(self, confidence=0.95, plot_bounds=True, ax=None, start=0.0):
        if ax is None:
            ax = plt.gcf().gca()

        # Prepend the start point so the step plot always begins from it
        # (the MCF is 0 before the first observed event).
        if start is not None and start < self.x.min():
            x = np.hstack([[start], self.x])
            mcf_hat = np.hstack([[0.0], self.mcf_hat])
        else:
            x = self.x
            mcf_hat = self.mcf_hat

        ax.step(x, mcf_hat, where="post", label="MCF")
        if plot_bounds:
            if self.var is not None:
                cb = self.mcf_cb(
                    self.x, bound="two-sided", confidence=confidence
                )
                if start is not None and start < self.x.min():
                    cb = np.vstack([[0.0, 0.0], cb])
                ax.step(
                    x,
                    cb,
                    where="post",
                    label=f"{confidence * 100}% Confidence Bounds",
                    color="red",
                )
        return ax

    def fit_from_recurrent_data(self, data):
        reject_unsupported_nonparametric(data, "NonParametricCounting")
        out = type(self)()
        out.data = data
        out.x, out.r, out.d = data.to_xrd()
        d = out.d
        r = out.r

        out.mcf_hat = np.cumsum(d / r)
        var = (
            1.0
            / r**2
            * (d * (1 - 1.0 / r) ** 2 + (r - d) * (0 - 1.0 / r) ** 2)
        )
        var = (d > 0).astype(int) * var
        out.var = np.cumsum(var)

        return out

    def fit(self, x, i=None, c=None, n=None, tl=None, tr=None, windows=None):
        """
        Fit a nonparametric (Nelson-Aalen) MCF.

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
        tl : array like or scalar, optional
            Left-truncation (delayed-entry) time per item. An item only
            enters the at-risk set once observation begins at ``tl``, so
            earlier event times are estimated over a smaller risk set.
        tr : array like or scalar, optional
            Right-truncation time per item.
        windows : dict, optional
            Gapped (multi-window) observation: a mapping ``{item: [(start,
            end), ...]}`` giving each item's disjoint observation windows.
            When given, every row in ``x`` must be an observed event (``c=0``).
            Each window becomes its own at-risk period, so an item is correctly
            absent from the risk set during a gap. Mutually exclusive with
            ``tl``/``tr``.

        Returns
        -------
        NonParametricCounting
        """
        data = handle_xicn(
            x, i, c, n, tl=tl, tr=tr, as_recurrent_data=True, windows=windows
        )
        return self.fit_from_recurrent_data(data)
