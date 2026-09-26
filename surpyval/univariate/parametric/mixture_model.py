import warnings
from typing import Any

import numpy.typing as npt
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import logsumexp

from surpyval import Distribution, np
from surpyval.serialisation import (
    SerialisableMixin,
    require_model_tag,
    stamp_schema,
)
from surpyval.utils.surpyval_data import SurpyvalData

from .probability_plotting import (
    adjust_heuristic,
    draw_probability_plot,
    probability_plot_data,
)

# The log-likelihood floor of one observation under one component: far
# below any log-likelihood an observation the component can explain has,
# and finite, so the EM objective stays finite (see
# ``MixtureModel.log_likelihood``).
LOG_FLOOR = -1e4


class MixtureModel(SerialisableMixin, Distribution):
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

    def __init__(self, dist: Any, m: int = 2) -> None:
        self.m = m
        self.dist = dist
        # These are None until ``fit`` runs and arrays afterwards, so the
        # honest annotation is the union -- and every use is downstream of
        # a fit. Narrowing them to the fitted type would be a lie before
        # the fit; narrowing to Optional would need an assert at each of
        # the thirty-odd uses without making anything safer, because
        # calling a predict method on an unfitted model is a contract
        # error the AttributeError already reports.
        self.data: Any = None
        self.params: Any = None
        self.w: Any = None
        self.p: Any = None
        #: The observed-data *negative* log-likelihood at the current
        #: parameters (despite the name), which the EM iteration tracks.
        self.loglike: Any = None

    # -- serialisation -----------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialise this fitted mixture model to a plain, JSON-serialisable dict.

        Stores the base distribution's name, the number of components ``m``,
        the per-component parameters and the mixing weights, so the reloaded
        model reproduces ``sf``/``ff``/``df``/``mean``/``random`` exactly. The
        fitted data and EM responsibilities are not stored.
        """
        from .parametric import is_custom_distribution

        out = {
            "model": "MixtureModel",
            "dist": self.dist.name,
            "m": int(self.m),
            "params": np.asarray(self.params, dtype=float).tolist(),
            "w": np.asarray(self.w, dtype=float).tolist(),
        }
        if is_custom_distribution(self.dist):
            # Resolved through the CustomDistribution registry on reading
            out["custom"] = True
        return stamp_schema(out)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "MixtureModel":
        """Rebuild a mixture model from a :meth:`to_dict` dictionary.

        The restored model evaluates the mixture (``sf``, ``ff``, ``df``,
        ``cs``, ``mean``, ``random``) exactly, but holds no data, so
        :meth:`plot` and :meth:`get_plot_data` raise. A mixture of a
        ``CustomDistribution`` or a ``Discretize`` distribution is read
        back as described in ``Parametric.from_dict``.
        """
        from .parametric import resolve_distribution

        require_model_tag(model_dict, "MixtureModel", "a mixture model")
        dist = resolve_distribution(
            model_dict["dist"], bool(model_dict.get("custom", False))
        )
        out = cls(dist=dist, m=int(model_dict["m"]))
        out.params = np.array(model_dict["params"], dtype=float)
        out.w = np.array(model_dict["w"], dtype=float)
        return out

    def __repr__(self) -> str:
        if self.params is not None:
            param_string = "\n".join(
                [
                    f"{name:>10}: {p}"
                    for p, name in zip(self.params.T, self.dist.param_names)
                ]
            )
            weight_string = ",\n\t".join([str(w) for w in self.w])
            out = (
                "Parametric Mixture SurPyval Model"
                "\n================================="
                f"\nDistribution        : {self.dist.name}"
                f"\nSub-Distributions   : {self.m}"
                "\nFitted by           : EM"
                f"\nWeights             : \n\t{weight_string}"
                f"\nParameters          :\n{param_string}"
            )

            return out
        else:
            return "Unable to fit values"

    def likelihood(self, params: Any) -> Any:
        """Per-observation likelihood of one component (no count powers:
        counts ``n`` enter the log-likelihood as multipliers -- raising the
        per-component likelihood to ``n`` *before* mixing is wrong, since
        ``sum_i w_i f_i^n != (sum_i w_i f_i)^n`` (#254)."""
        data = self.data
        like_o = self.dist.df(data.x_o, *params)
        like_r = self.dist.sf(data.x_r, *params)
        like_l = self.dist.ff(data.x_l, *params)
        like_i = self.dist.ff(data.x_ir, *params) - self.dist.ff(
            data.x_il, *params
        )
        like = np.zeros(len(self.data.x))
        like[data.c == 0] = like_o
        like[data.c == 1] = like_r
        like[data.c == -1] = like_l
        like[data.c == 2] = like_i
        return like

    def log_likelihood(self, params: Any) -> Any:
        """Per-observation log-likelihood of one component, floored at
        ``LOG_FLOOR``.

        Formed from the distribution's log functions rather than as the
        log of :meth:`likelihood`: a density or interval probability that
        underflows to 0 made ``log`` return -inf, a responsibility times
        -inf made the M-step objective infinite, and the optimiser
        stopped after one step (a two-Weibull mixture on interval data
        stalled 250 log-likelihood units short, with no warning). The
        floor keeps an observation a component cannot explain at a finite,
        heavily penalised value instead.
        """
        data = self.data
        dist = self.dist
        out = np.zeros(len(data.x))
        with np.errstate(all="ignore"):
            if (data.c == 0).any():
                out[data.c == 0] = dist.log_df(data.x_o, *params)
            if (data.c == 1).any():
                out[data.c == 1] = dist.log_sf(data.x_r, *params)
            if (data.c == -1).any():
                out[data.c == -1] = dist.log_ff(data.x_l, *params)
            if (data.c == 2).any():
                window = dist.ff(data.x_ir, *params) - dist.ff(
                    data.x_il, *params
                )
                out[data.c == 2] = np.log(np.maximum(window, 0.0))
        out = np.nan_to_num(out, nan=LOG_FLOOR, neginf=LOG_FLOOR)
        return np.maximum(out, LOG_FLOOR)

    def _log_resp(self, w: npt.NDArray, params: Any) -> Any:
        """``log w_i + log L_i`` for every component (rows) and
        observation (columns)."""
        with np.errstate(divide="ignore"):
            log_w = np.log(np.asarray(w, dtype=float))
        return np.array(
            [log_w[i] + self.log_likelihood(params[i]) for i in range(self.m)]
        )

    def _window_prob(self, params_i: npt.NDArray) -> Any:
        """One component's probability of landing in each observation's
        truncation window ``(tl, tr]`` -- the per-component piece of the
        truncation correction."""
        tl, tr = self.data.tl, self.data.tr
        lo = np.zeros(len(self.data.x))
        fin = np.isfinite(tl)
        if fin.any():
            lo[fin] = self.dist.ff(tl[fin], *params_i)
        hi = np.ones(len(self.data.x))
        fin = np.isfinite(tr)
        if fin.any():
            hi[fin] = self.dist.ff(tr[fin], *params_i)
        return hi - lo

    def neg_ll_of(self, w: npt.NDArray, params: Any) -> Any:
        """Observed negative log-likelihood of the mixture: counts multiply
        in the log domain, and truncated observations are conditioned on
        their window through the mixture probability of the window."""
        # log-sum-exp over the components, so the mixture density of an
        # observation is not lost to underflow in any one of them.
        with np.errstate(all="ignore"):
            ll = np.sum(self.data.n * logsumexp(self._log_resp(w, params), 0))
            if self._truncated:
                win = np.zeros(len(self.data.x))
                for i in range(self.m):
                    win += w[i] * self._window_prob(params[i])
                ll -= np.sum(self.data.n * np.log(win))
        return -ll

    def Q(self, params: Any) -> Any:
        """EM M-step objective: the (negative) expected complete-data
        log-likelihood over the component labels -- counts times
        responsibilities times each component's log-likelihood."""
        params = params.reshape(self.m, self.dist.k)
        total = 0.0
        for i in range(self.m):
            # Finite by construction (see log_likelihood), so a zero
            # responsibility contributes exactly 0 and none gives inf.
            loglike = self.log_likelihood(params[i])
            total -= np.sum(self.data.n * self.p[i] * loglike)
        return total

    def expectation(self) -> Any:
        """EM E-step: set each observation's responsibilities ``p`` (the
        probability it belongs to each component, given the current fit)
        and the count-weighted mixing weights ``w``."""
        # Normalised in the log domain: dividing likelihoods that had all
        # underflowed to 0 gave 0/0 responsibilities (and the overflow
        # and invalid-value warnings of a discrete mixture).
        log_r = self._log_resp(self.w, self.params)
        with np.errstate(all="ignore"):
            self.p = np.exp(log_r - logsumexp(log_r, axis=0))
        # Mixing weights are count-weighted responsibility totals.
        self.w = (self.p * self.data.n).sum(axis=1) / self.data.n.sum()

    def maximisation(self) -> Any:
        """EM M-step: refit every component's parameters by minimising
        :meth:`Q` with the current responsibilities held fixed."""
        bounds = self.dist.bounds * self.m
        res = minimize(self.Q, self.params.ravel(), bounds=bounds)
        self.params = res.x.reshape(self.m, self.dist.k)

    def EM(self) -> Any:
        """One EM iteration (:meth:`expectation` then
        :meth:`maximisation`), after which ``loglike`` holds the observed
        negative log-likelihood."""
        self.expectation()
        self.maximisation()
        # Convergence is tracked on the observed likelihood, not the
        # M-step objective.
        self.loglike = self.neg_ll_of(self.w, self.params)

    def _em(self, tol: float = 1e-10, max_iter: int = 1000) -> Any:
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
        if i >= max_iter:
            warnings.warn(
                "EM algorithm reached max iterations before converging"
            )

    def initialise_params(self) -> Any:
        """The EM starting point: cut the (sorted) data into ``m``
        consecutive blocks, fit one component to each block, and weight
        the components equally."""
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
        x: npt.ArrayLike | None = None,
        c: npt.ArrayLike | None = None,
        n: npt.ArrayLike | None = None,
        t: npt.ArrayLike | None = None,
        tl: npt.ArrayLike | None = None,
        tr: npt.ArrayLike | None = None,
        xl: npt.ArrayLike | None = None,
        xr: npt.ArrayLike | None = None,
    ) -> Any:
        """
        Fit the mixture to data, in place.

        Unlike the single-distribution fitters, this does not return a new
        model: it sets the fitted ``params`` (one row per sub-distribution)
        and mixing weights ``w`` on this object, which is then used as the
        model. Untruncated data is fitted by the EM algorithm; truncated
        data by direct maximisation of the truncated likelihood.

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
            Array of counts for each x. If data is provided as counts, then
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

        Returns
        -------

        None
            The fit is stored on this object.

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
                0.6184891886499861,
                0.381510811350014
        Parameters          :
             alpha: [ 6.32508961 17.37701969]
              beta: [ 1.83105154 12.01392721]
        """

        data = SurpyvalData(x=x, c=c, n=n, t=t, tl=tl, tr=tr, xl=xl, xr=xr)

        # Count observations from the validated data so ``xl``/``xr``-only
        # input works (#254), and weigh by counts.
        if data.n.sum() < self.m * (self.dist.k + 1):
            raise ValueError("More parameters than data points")

        self.data = data
        self._truncated = bool(np.isfinite(data.t).any())
        self.p = np.ones(shape=(self.m, len(self.data.x))) / self.m

        self.initialise_params()

        if self._truncated:
            # The truncation correction couples the components through the
            # mixture window probability, so the label-based EM does not
            # apply; maximise the observed truncated likelihood directly,
            # warm-started from the split-fit initialisation (#254).
            self._direct_mle()
        else:
            self._em()

    def _direct_mle(self) -> Any:
        """Directly maximise the observed (truncation-corrected) negative
        log-likelihood over the mixing weights (via softmax logits) and the
        component parameters."""
        k = self.dist.k

        def unpack(theta: npt.NDArray) -> Any:
            logits = np.append(theta[: self.m - 1], 0.0)
            logits = logits - logits.max()
            w = np.exp(logits)
            w = w / w.sum()
            params = theta[self.m - 1 :].reshape(self.m, k)
            return w, params

        def obj(theta: npt.NDArray) -> Any:
            w, params = unpack(theta)
            return self.neg_ll_of(w, params)

        bounds: list[tuple[float | None, float | None]] = [(None, None)] * (
            self.m - 1
        )
        for _ in range(self.m):
            for lo, hi in self.dist.bounds:
                lo_s = None if lo is None else (1e-10 if lo == 0 else lo)
                bounds.append((lo_s, hi))

        x0 = np.concatenate([np.zeros(self.m - 1), self.params.ravel()])
        with np.errstate(all="ignore"):
            res = minimize(obj, x0, bounds=bounds)
        self.w, self.params = unpack(res.x)
        self.loglike = float(res.fun)

    def mean(self, *args: Any, **kwargs: Any) -> Any:
        r"""
        The mean of the fitted mixture, :math:`\sum_{j} w_{j} E[X_{j}]`.

        Returns
        -------
        float
            The weighted sum of the component means.

        Examples
        --------
        >>> import surpyval as surv
        >>> x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
        >>> wmm = surv.MixtureModel(dist=surv.Weibull, m=2)
        >>> wmm.fit(x)
        >>> round(float(wmm.mean()), 4)
        9.8294
        """
        mean = 0
        for i in range(self.m):
            mean += self.w[i] * self.dist.mean(*self.params[i])
        return mean

    def random(self, size: int, *args: Any, **kwargs: Any) -> Any:
        """
        Draw random samples from the fitted mixture.

        The number drawn from each component is multinomial with the
        mixing weights ``w``, and the draws are shuffled together.

        Parameters
        ----------
        size : int
            The number of samples to draw.

        Returns
        -------
        numpy array
            ``size`` values from the mixture, in random order.
        """
        sizes = np.random.multinomial(size, self.w)
        rvs = np.zeros(size)
        s_last = 0
        for i, s in enumerate(sizes):
            rvs[s_last : s + s_last] = self.dist.random(s, *self.params[i, :])
            s_last += s
        # Shuffles the data (inplace) so that the data is random
        np.random.shuffle(rvs)
        return rvs

    def df(self, x: Any, *args: Any, **kwargs: Any) -> Any:
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
        x = np.asarray(x, dtype=float)
        df = np.zeros_like(x)
        for i in range(self.m):
            df += self.w[i] * self.dist.df(x, *self.params[i])
        return df

    def ff(self, x: Any, *args: Any, **kwargs: Any) -> Any:
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

    def sf(self, x: Any, *args: Any, **kwargs: Any) -> Any:
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

    def cs(self, x: Any, X: Any, *args: Any, **kwargs: Any) -> Any:
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
        # As arrays: ``x + X`` on a list concatenated (or raised) rather
        # than adding.
        x = np.asarray(x, dtype=float)
        X = np.asarray(X, dtype=float)
        return self.sf(x + X) / self.sf(X)

    def _require_data(self, what: str) -> None:
        if self.params is None:
            raise ValueError(f"{what} needs a fitted mixture")
        if self.data is None:
            raise ValueError(
                f"{what} needs the data the mixture was fitted to, which a "
                "mixture restored with from_dict does not carry (to_dict "
                "stores only the parameters and weights)."
            )

    def get_plot_data(self, heuristic: str = "Nelson-Aalen") -> Any:
        """The plotting positions and fitted curve that :meth:`plot`
        draws, computed from the fitted data with ``heuristic``."""
        self._require_data("get_plot_data()")
        return probability_plot_data(
            dist=self.dist,
            ff=self.ff,
            x=self.data.x,
            c=self.data.c,
            n=self.data.n,
            t=self.data.t,
            heuristic=heuristic,
            params=self.params,
        )

    def plot(self, heuristic: str = "Nelson-Aalen", ax: Any = None) -> Any:
        """
        A method to do a probability plot

        Parameters
        ----------
        heuristic : {'Blom', 'Median', 'ECDF', 'Modal', 'Midpoint', 'Mean', \
            'Weibull', 'Benard', 'Beard', 'Hazen', 'Gringorten', 'None',\
            'Tukey', 'DPW', 'Fleming-Harrington', 'Kaplan-Meier',\
            'Nelson-Aalen', 'Filliben', 'Larsen', 'Turnbull'}, optional
            The method that the plotting point on the probability plot will
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

        if self.params is None:
            raise ValueError("Can't plot model that failed to fit")
        self._require_data("plot()")

        heuristic = adjust_heuristic(self.data.c, self.data.t, heuristic)

        d = self.get_plot_data(heuristic=heuristic)

        return draw_probability_plot(
            ax,
            d,
            lambda x: self.dist.mpp_y_transform(x, *self.params),
            lambda x: self.dist.mpp_inv_y_transform(x, *self.params),
            title=f"{self.dist.name} Mixture Probability Plot",
        )
