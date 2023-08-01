from autograd import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm, t

from surpyval.utils.recurrent_utils import handle_xicn


class NonParametricCounting:
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
        # Greenwoods variance using t-stat. Ref found:
        # http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis
        assert bound_type in ["exp", "normal"]
        assert dist in ["t", "z"]
        x = np.atleast_1d(x)
        if bound in ["upper", "lower"]:
            if dist == "t":
                stat = t.ppf(1 - confidence, self.r - 1)
            else:
                stat = norm.ppf(1 - confidence, 0, 1)
            if bound == "upper":
                stat = -stat
        elif bound == "two-sided":
            if dist == "t":
                stat = t.ppf((1 - confidence) / 2, self.r - 1)
            else:
                stat = norm.ppf((1 - confidence) / 2, 0, 1)
            stat = np.array([-1, 1]).reshape(2, 1) * stat

        if bound_type == "exp":
            # Exponential Greenwood confidence
            mcf_cb = self.mcf_hat * np.exp(
                stat * np.sqrt(self.var) / self.mcf_hat
            )
        else:
            # Normal Greenwood confidence
            mcf_cb = (
                self.mcf_hat + np.sqrt(self.var * self.mcf_hat**2) * stat
            )
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

    def plot(self, confidence=0.95, plot_bounds=True, ax=None):
        if ax is None:
            ax = plt.gcf().gca()

        ax.step(self.x, self.mcf_hat, where="post", label="MCF")
        if plot_bounds:
            if self.var is not None:
                ax.step(
                    self.x,
                    self.mcf_cb(
                        self.x, bound="two-sided", confidence=confidence
                    ),
                    where="post",
                    label=f"{confidence * 100}% Confidence Bounds",
                    color="red",
                )
        return ax

    @classmethod
    def fit_from_recurrent_data(cls, data):
        out = cls()
        out.data = data
        x, r, d = data.to_xrd()
        out.x, out.r, out.d = data.to_xrd()

        out.mcf_hat = np.cumsum(d / r)
        var = (
            1.0
            / r**2
            * (d * (1 - 1.0 / r) ** 2 + (r - d) * (0 - 1.0 / r) ** 2)
        )
        var = (d > 0).astype(int) * var
        out.var = np.cumsum(var)

        return out

    @classmethod
    def fit(cls, x, i=None, c=None, n=None):
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data)
