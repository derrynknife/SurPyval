import re
import warnings

from autograd import jacobian
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedLocator
from scipy.optimize import minimize
from scipy.special import ndtri as z

from surpyval import np
from surpyval.univariate.nonparametric import plotting_positions
from surpyval.utils import _round_vals
from surpyval.utils.surpyval_data import SurpyvalData


class MixtureModel:
    """
    A class for creating a Mixture Model fitter.

    This class implements a Mixture Model, which is a probabilistic model that
    combines multiple probability distributions to model complex data. Models
    can be fit with either the Expectation Maximisation (EM) algorithm or
    Maximum Likelihood Estimation (MLE). The EM algorithm is the default and
    is based on the paper found `here \
    <https://www.sciencedirect.com/science/article/pii/S0307904X12002545>`__.

    Parameters
    ----------

    dist : surpyval distribution
        The distribution to be used in the mixture model. Must be a
        surpyval distribution.

    m : int, optional
        The number of sub-distributions to be used in the mixture model.
        Defaults to 2.
    """

    def __init__(self, dist, m=2):
        self.m = m
        self.dist = dist

    def __repr__(self):
        if hasattr(self, "params"):
            param_string = "\n".join(
                [
                    "{:>10}".format(name) + ": " + str(p)
                    for p, name in zip(self.params.T, self.dist.param_names)
                ]
            )
            out = (
                "Parametric Mixture SurPyval Model"
                + "\n================================="
                + "\nDistribution        : {dist}"
                + "\nSub-Distributions   : {m}"
                + "\nFitted by           : EM"
            ).format(dist=self.dist.name, m=self.m)

            out = (
                out
                + "\nWeights             : "
                + "\n\t{params}".format(
                    params=",\n\t".join([str(w) for w in self.w])
                )
            )

            out = (
                out
                + "\nParameters          :\n"
                + "{params}".format(params=param_string)
            )

            return out
        else:
            return "Unable to fit values"

    def likelihood(self, params):
        data = self.data
        like_o = self.dist.df(data.x_o, *params)
        like_r = self.dist.sf(data.x_r, *params)
        like_l = self.dist.ff(data.x_l, *params)
        like_i = self.dist.ff(data.x_il, *params) - self.dist.ff(
            data.x_ir, *params
        )
        like = np.zeros(len(self.data.x))
        like[data.c == 0] = like_o
        like[data.c == 1] = like_r
        like[data.c == -1] = like_l
        like[data.c == 2] = like_i
        like = np.power(like, data.n)
        return like

    def Q(self, params):
        params = params.reshape(self.m, self.dist.k)
        f = np.zeros_like(self.p)
        for i in range(self.m):
            like = self.likelihood(params[i])
            f[i] = np.multiply(self.p[i], like)
        f = -np.sum(np.log(f.sum(axis=0)))
        self.loglike = f
        return f

    def expectation(self):
        for i in range(self.m):
            like = self.likelihood(self.params[i])
            like = np.multiply(self.w[i], like)
            self.p[i] = like
        self.p = np.divide(self.p, np.sum(self.p, axis=0))
        self.w = np.sum(self.p, axis=1) / len(self.data.x)

    def maximisation(self):
        bounds = self.dist.bounds * self.m
        res = minimize(self.Q, self.params.ravel(), bounds=bounds)
        self.params = res.x.reshape(self.m, self.dist.k)

    def EM(self):
        self.expectation()
        self.maximisation()

    def _em(self, tol=1e-10, max_iter=1000):
        i = 0
        self.EM()
        f0 = self.loglike
        self.EM()
        f1 = self.loglike
        while (np.abs(f0 - f1) > tol) and (i < max_iter):
            f0 = f1
            self.EM()
            f1 = self.loglike
            i += 1
        if i >= 1000:
            print("Max iterations reached")

    def initialise_params(self):
        splits_x = np.array_split(self.data.x, self.m)
        splits_c = np.array_split(self.data.c, self.m)
        splits_n = np.array_split(self.data.n, self.m)
        params = np.zeros(shape=(self.m, self.dist.k))

        for i in range(self.m):
            params[i, :] = self.dist.fit(
                x=splits_x[i], c=splits_c[i], n=splits_n[i]
            ).params
        self.params = params
        self.w = np.ones(shape=(self.m)) / self.m

    def fit(
        self,
        x=None,
        c=None,
        n=None,
        t=None,
        tl=None,
        tr=None,
        xl=None,
        xr=None,
    ):
        """
        Parameters
        ----------

        x : array like, optional
            Array of observations of the random variables. If x is
            :code:`None`, xl and xr must be provided.
        c : array like, optional
            Array of censoring flag. -1 is left censored, 0 is observed, 1 is
            right censored, and 2 is intervally censored. If not provided
            will assume all values are observed.
        n : array like, optional
            Array of counts for each x. If data is proivded as counts, then
            this can be provided. If :code:`None` will assume each
            observation is 1.
        t : 2D-array like, optional
            2D array like of the left and right values at which the
            respective observation was truncated. If not provided it assumes
            that no truncation occurs.

        tl : array like or scalar, optional
            Values of left truncation for observations. If it is a scalar
            value assumes each observation is left truncated at the value.
            If an array, it is the respective 'late entry' of the observation

        tr : array like or scalar, optional
            Values of right truncation for observations. If it is a scalar
            value assumes each observation is right truncated at the value.
            If an array, it is the respective right truncation value for each
            observation

        xl : array like, optional
            Array like of the left array for 2-dimensional input of x. This
            is useful for data that is all intervally censored. Must be used
            with the :code:`xr` input.

        xr : array like, optional
            Array like of the right array for 2-dimensional input of x. This
            is useful for data that is all intervally censored. Must be used
            with the :code:`xl` input.

        Examples
        --------

        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17 ,17, 18, 19]
        >>> # Create a Weibull Mixture Model fitter with 2 sub-distributions
        >>> wmm = surv.MixtureModel(dist=surv.Weibull, m=2)
        >>> wmm.fit(x)
        >>> wmm
        Parametric Mixture SurPyval Model
        =================================
        Distribution        : Weibull
        Sub-Distributions   : 2
        Fitted by           : EM
        Weights             :
            0.6094710980384728,
            0.39052890196152723
        Parameters          :
            alpha: [ 5.8855232  17.23187124]
             beta: [ 2.04051304 11.01565277]
        """

        data = SurpyvalData(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)

        if len(x) < self.m * (self.dist.k + 1):
            raise ValueError("More parameters than data points")

        self.data = data
        self.p = np.ones(shape=(self.m, len(self.data.x))) / self.m

        self.initialise_params()

        self._em()

    def R_cb(self, t, cb=0.05):
        def ssf(params):
            params = np.reshape(params, (self.m, self.dist.k + 1))
            F = np.zeros_like(t)
            for i in range(self.m):
                F = F + params[i, 0] * self.dist.ff(t, *params[i, 1::])
            return 1 - F

        pvars = self.hess_inv[np.triu_indices(self.hess_inv.shape[0])]
        with np.errstate(all="ignore"):
            jac = jacobian(ssf)(self.res.x)

        var_u = []
        for i, j in enumerate(jac):
            j = np.atleast_2d(j).T * j
            j = j[np.triu_indices(j.shape[0])]
            var_u.append(np.sum(j * pvars))
        diff = (
            z(cb / 2)
            * np.sqrt(np.array(var_u))
            * np.array([1.0, -1.0]).reshape(2, 1)
        )
        R_hat = self.sf(t)
        exponent = diff / (R_hat * (1 - R_hat))
        R_cb = R_hat / (R_hat + (1 - R_hat) * np.exp(exponent))
        return R_cb.T

    def mean(self):
        mean = 0
        for i in range(self.m):
            mean += self.w[i] * self.dist.mean(*self.params[i])
        return mean

    def random(self, size):
        sizes = np.random.multinomial(size, self.w)
        rvs = np.zeros(size)
        s_last = 0
        for i, s in enumerate(sizes):
            rvs[s_last : s + s_last] = self.dist.random(s, *self.params[i, :])
            s_last = s
        # Shuffles the data (inplace) so that the data is random
        np.random.shuffle(rvs)
        return rvs

    def df(self, x):
        """
        The probability density function of the fitted model.

        Parameters
        ----------

        x : array like
            The values at which the probability density function will be
            evaluated.

        Returns
        -------

        array like
            The probability density function evaluated at x.
        """
        df = np.zeros_like(x)
        for i in range(self.m):
            df += self.w[i] * self.dist.df(x, *self.params[i])
        return df

    def ff(self, x):
        """
        The cumulative density function of the fitted model.

        Parameters
        ----------

        x : array like
            The values at which the cumulative density function will be
            evaluated.

        Returns
        -------

        array like
            The cumulative density function evaluated at x.
        """
        F = np.zeros_like(x)
        for i in range(self.m):
            F = F + self.w[i] * self.dist.ff(x, *self.params[i])
        return F

    def sf(self, x):
        """
        The survival function of the fitted model.

        Parameters
        ----------

        x : array like
            The values at which the survival function will be evaluated.

        Returns
        -------

        array like
            The survival function evaluated at x.
        """
        return 1 - self.ff(x)

    def cs(self, x, X):
        """
        The conditional survival function of the fitted model.

        Parameters
        ----------

        x : array like
            The values at which the conditional survival function will be
            evaluated.

        X : array like
            The values at which the item is known to have survived to.

        Returns
        -------

        array like
            The conditional survival function evaluated at x given X.
        """
        return self.sf(x + X) / self.sf(X)

    def get_plot_data(self, heuristic="Nelson-Aalen"):
        x_, r, d, F = plotting_positions(
            x=self.data.x,
            c=self.data.c,
            n=self.data.n,
            t=self.data.t,
            heuristic=heuristic,
        )

        mask = np.isfinite(x_)
        x_ = x_[mask]
        r = r[mask]
        d = d[mask]
        F = F[mask]

        # Adjust the plotting points in event data is truncated.
        tl_min = self.data.t[0][0]
        if np.isfinite(tl_min):
            Ftl = self.ff(tl_min)
        else:
            Ftl = 0

        tr_max = self.data.t[-1][-1]
        if np.isfinite(tr_max):
            Ftr = self.ff(tr_max)
        else:
            Ftr = 1

        # Adjust the plotting points due to truncation
        F = Ftl + F * (Ftr - Ftl)

        y_scale_min = np.min(F[F > 0]) / 2
        y_scale_max = 1 - (1 - np.max(F[F < 1])) / 10

        # x-axis
        if self.dist.plot_x_scale == "log":
            log_x = np.log10(x_[x_ > 0])
            x_min = np.min(log_x)
            x_max = np.max(log_x)
            vals_non_sig = 10 ** np.linspace(x_min, x_max, 7)
            x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
            x_minor_ticks = (
                10**x_minor_ticks
                * np.array(np.arange(1, 11)).reshape((10, 1))
            ).flatten()
            diff = (x_max - x_min) / 10
            x_scale_min = 10 ** (x_min - diff)
            x_scale_max = 10 ** (x_max + diff)
            x_model = 10 ** np.linspace(x_min - diff, x_max + diff, 100)
        elif self.dist.name in ("Beta"):
            x_min = np.min(x_)
            x_max = np.max(x_)
            x_scale_min = 0
            x_scale_max = 1
            vals_non_sig = np.linspace(x_scale_min, x_scale_max, 11)[1:-1]
            x_minor_ticks = np.linspace(x_scale_min, x_scale_max, 22)[1:-1]
            x_model = np.linspace(x_scale_min, x_scale_max, 102)[1:-1]
        elif self.dist.name in ("Uniform"):
            x_min = np.min(self.params)
            x_max = np.max(self.params)
            x_scale_min = x_min
            x_scale_max = x_max
            vals_non_sig = np.linspace(x_scale_min, x_scale_max, 11)[1:-1]
            x_minor_ticks = np.linspace(x_scale_min, x_scale_max, 22)[1:-1]
            x_model = np.linspace(x_scale_min, x_scale_max, 102)[1:-1]
        else:
            x_min = np.min(x_)
            x_max = np.max(x_)
            vals_non_sig = np.linspace(x_min, x_max, 7)
            x_minor_ticks = np.arange(np.floor(x_min), np.ceil(x_max))
            diff = (x_max - x_min) / 10
            x_scale_min = x_min - diff
            x_scale_max = x_max + diff
            x_model = np.linspace(x_scale_min, x_scale_max, 100)

        cdf = self.ff(x_model)

        x_ticks = _round_vals(vals_non_sig)
        x_ticks_labels = [
            str(int(x))
            if (re.match(r"([0-9]+\.0+)", str(x)) is not None) & (x > 1)
            else str(x)
            for x in _round_vals(vals_non_sig)
        ]

        y_ticks = np.array(self.dist.y_ticks)
        y_ticks = y_ticks[
            np.where((y_ticks > y_scale_min) & (y_ticks < y_scale_max))[0]
        ]

        y_ticks_labels = [
            str(int(y)) + "%"
            if (re.match(r"([0-9]+\.0+)", str(y)) is not None) & (y > 1)
            else str(y)
            for y in y_ticks * 100
        ]

        return {
            "x_scale_min": x_scale_min,
            "x_scale_max": x_scale_max,
            "y_scale_min": y_scale_min,
            "y_scale_max": y_scale_max,
            "y_ticks": y_ticks,
            "y_ticks_labels": y_ticks_labels,
            "x_ticks": x_ticks,
            "x_ticks_labels": x_ticks_labels,
            "cdf": cdf,
            "x_model": x_model,
            "x_minor_ticks": x_minor_ticks,
            "x_scale": self.dist.plot_x_scale,
            "x_": x_,
            "F": F,
        }

    def plot(
        self,
        heuristic="Nelson-Aalen",
        ax=None,
    ):
        """
        A method to do a probability plot

        Parameters
        ----------
        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean', \
            'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None',\
            'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier',\
            'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probablility plot will
            be calculated.

        ax: matplotlib.axes.Axes, optional
            The axis onto which the plot will be created. Optional, if not
            provided a new axes will be created.

        Returns
        -------
        matplotlib.axes.Axes
            a matplotlib axes containing the plot

        """
        if ax is None:
            ax = plt.gcf().gca()

        if not hasattr(self, "params"):
            raise Exception("Can't plot model that failed to fit")

        if 2 in self.data.c:
            if heuristic != "Turnbull":
                warnings.warn(
                    "Interval censored data, heuristic changed to Turnbull'",
                    stacklevel=1,
                )
                heuristic = "Turnbull"

        if np.isfinite(self.data.t).any():
            if heuristic != "Turnbull":
                warnings.warn(
                    "Truncated censored data, heuristic changed to Turnbull'",
                    stacklevel=1,
                )
                heuristic = "Turnbull"

        d = self.get_plot_data(heuristic=heuristic)

        # Set limits and scale
        ax.set_ylim(
            [max(d["y_scale_min"], 1e-4), min(d["y_scale_max"], 0.9999)]
        )
        ax.set_xscale(d["x_scale"])
        functions = (
            lambda x: self.dist.mpp_y_transform(x, *self.params),
            lambda x: self.dist.mpp_inv_y_transform(x, *self.params),
        )
        ax.set_yscale("function", functions=functions)
        ax.set_yticks(d["y_ticks"])
        ax.set_yticklabels(d["y_ticks_labels"])
        ax.yaxis.set_minor_locator(FixedLocator(np.linspace(0, 1, 51)))
        ax.set_xticks(d["x_ticks"])
        ax.set_xticklabels(d["x_ticks_labels"])

        if d["x_scale"] == "log":
            ax.set_xticks(d["x_minor_ticks"], minor=True)
            ax.set_xticklabels([], minor=True)

        ax.grid(
            visible=True, which="major", color="g", alpha=0.4, linestyle="-"
        )
        ax.grid(
            visible=True, which="minor", color="g", alpha=0.1, linestyle="-"
        )

        ax.set_title("{} Mixture Probability Plot".format(self.dist.name))
        ax.set_ylabel("CDF")
        ax.scatter(d["x_"], d["F"])

        ax.set_xlim([d["x_scale_min"], d["x_scale_max"]])

        ax.plot(d["x_model"], d["cdf"], color="k", linestyle="--")
        return ax
