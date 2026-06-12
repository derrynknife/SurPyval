import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import norm, t


def interp_function(x, y, kind):
    return interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)


class NonParametric:
    """
    Result of ``.fit()`` method for every non-parametric
    surpyval distribution. This means that each of the
    methods in this class can be called with a model created
    from the ``NelsonAalen``, ``KaplanMeier``,
    ``FlemingHarrington``, or ``Turnbull`` estimators.
    """

    def __repr__(self):
        out = (
            "Non-Parametric SurPyval Model"
            + "\n============================="
            + "\nModel            : {dist}"
        ).format(dist=self.model)

        if hasattr(self, "data"):
            if "estimator" in self.data:
                out += "\nEstimator        : {turnbull}".format(
                    turnbull=self.data["estimator"]
                )

        return out

    def sf(self, x, interp="step"):
        r"""

        Surival (or Reliability) function with the
        non-parametric estimates from the data.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which t
            he survival function will be calculated.

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the survival function at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.sf(2)
        array([0.63762815])
        >>> model.sf([1., 1.5, 2., 2.5])
        array([0.81873075, 0.81873075, 0.63762815, 0.63762815])
        """
        x = np.atleast_1d(x)
        idx = np.argsort(x)
        rev = np.argsort(idx)
        x = x[idx]
        if interp == "step":
            idx = np.searchsorted(self.x, x, side="right") - 1
            R = self.R[idx]
            R = np.where(idx < 0, 1, R)
            R = np.where(np.isposinf(x), 0, R)
        else:
            R = interp_function(self.x, self.R, kind=interp)(x)

        R = R[rev]
        # Maybe set a parameter where 'extrapolate' is False
        # x = x[rev]
        # R = np.where(x < self.x.min(), np.nan, R)
        # R = np.where(x > self.x.max(), np.nan, R)
        return R

    def ff(self, x, interp="step"):
        r"""

        CDF (failure or unreliability) function with the
        non-parametric estimates from the data

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which
            the survival function will be calculated.

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.ff(2)
        array([0.36237185])
        >>> model.ff([1., 1.5, 2., 2.5])
        array([0.18126925, 0.18126925, 0.36237185, 0.36237185])
        """
        return 1 - self.sf(x, interp=interp)

    def hf(self, x, interp="step"):
        r"""

        Instantaneous hazard function with the non-parametric
        estimates from the data. This is calculated using simply
        the difference between consecutive H(x).

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which
            the survival function will be calculated

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the failure function at each x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.hf([1.5, 2.5, 3.5])
        array([0.25      , 0.25      , 0.33333333])
        """
        x = np.atleast_1d(x)
        idx = np.argsort(x)
        rev = np.argsort(idx)
        x = x[idx]
        hf = np.diff(
            np.hstack(
                [self.Hf(x[0], interp=interp), self.Hf(x, interp=interp)]
            )
        )
        if hf.size > 1:
            hf[0] = hf[1]
        hf = pd.Series(hf)
        hf[hf == 0] = np.nan
        hf = hf.ffill().values
        return hf[rev]

    def df(self, x, interp="step"):
        r"""

        Density function with the non-parametric estimates
        from the data. This is calculated using the relationship
        between the hazard function and the density:

        .. math::
            f(x) = h(x)e^{-H(x)}

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the
            survival function will be calculated

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x


        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.df([1.5, 2.5, 3.5])
        array([0.20468269, 0.15940704, 0.15229351])
        """
        return self.hf(x, interp=interp) * np.exp(-self.Hf(x, interp=interp))

    def Hf(self, x, interp="step"):
        r"""

        Cumulative hazard rate with the non-parametric estimates
        from the data. This is calculated using the relationship
        between the hazard function and the density:

        .. math::
            H(x) = -\ln (R(x))

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the
            function will be calculated.

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the density function at x

        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.Hf(2)
        array([0.45])
        >>> model.Hf([1., 1.5, 2., 2.5])
        array([0.2 , 0.2 , 0.45, 0.45])
        """
        return -np.log(self.sf(x, interp=interp))

    def cb(
        self,
        x,
        on="sf",
        bound="two-sided",
        interp="step",
        alpha_ci=0.05,
        bound_type="exp",
        dist="z",
    ):
        r"""

        Confidence bounds of the ``on`` function at the
        ``alpha_ci`` level of significance. Can be the upper,
        lower, or two-sided confidence by changing value of ``bound``.
        Can change the bound type to be regular (normal) or exponential
        using either the 't' or 'z' statistic.

        The variance used is the one appropriate to the estimator with
        which the model was fitted: Greenwood's formula for Kaplan-Meier,
        Aalen's (Poisson) variance for Nelson-Aalen, and the tie-corrected
        variance for Fleming-Harrington.

        Parameters
        ----------

        x : array like or scalar
            The values of the random variables at which the confidence bounds
            will be calculated
        on : ('sf', 'ff', 'Hf'), optional
            The function on which the confidence bound will be calculated.
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Compute either the two-sided, upper or lower confidence bound(s).
            Defaults to two-sided.
        interp : ('step', 'linear', 'cubic'), optional
            How to interpolate the values between observations. Survival
            statistics traditionally uses step functions, but can use
            interpolated values if desired. Defaults to step.
        alpha_ci : scalar, optional
            The level of significance at which the bound will be computed.
        bound_type : ('exp', 'normal'), str, optional
            The method with which the bounds will be calculated. Using
            'normal' (i.e. the plain Greenwood-style interval) will allow
            for the bounds to exceed 1 or be less than 0 and tends to
            undercover in small samples. Defaults to 'exp' (the
            log(-log) transformed interval) as this ensures the bounds
            are within 0 and 1 and has better coverage.
        dist : ('t', 'z'), str, optional
            The statistic to use in calculating the bounds (student-t or
            normal). Defaults to the normal (z). The 't' option is a
            conservative heuristic which uses the at risk count minus one
            at each point as the degrees of freedom; it is not supported
            by asymptotic theory, widens (overcovers) as the at risk
            count falls, and is undefined when only one item remains at
            risk.

        Returns
        -------

        cb : scalar or numpy array
            The value(s) of the upper, lower, or both confidence bound(s) of
            the selected function at x. For two-sided bounds the result has
            one row per value of x with the columns being the
            ``[lower, upper]`` bounds of the ``on`` function.

        Notes
        -----

        If the last observation is an event (i.e. there is no right
        censoring) the Kaplan-Meier variance is undefined at, and after,
        that point. The bounds there are filled with the last finite
        upper bound and 0 for the lower bound, rather than NaN, so that
        bounds can be drawn up to the last observation.

        Examples
        --------
        >>> from surpyval import NelsonAalen
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = NelsonAalen.fit(x)
        >>> model.cb([1., 1.5, 2., 2.5], bound='lower')
        array([0.35485348, 0.35485348, 0.2345113 , 0.2345113 ])
        >>> model.cb([1., 1.5, 2., 2.5])
        array([[0.24175891, 0.97222045],
               [0.24175891, 0.97222045],
               [0.16288538, 0.89441253],
               [0.16288538, 0.89441253]])

        References
        ----------

        http://reliawiki.org/index.php/Non-Parametric_Life_Data_Analysis

        """
        if on in ["df", "hf"]:
            raise ValueError(
                "NonParametric cannot do confidence bounds on "
                + "density or hazard rate functions. Try Hf, "
                + "ff, or sf"
            )

        old_err_state = np.seterr(all="ignore")

        # Reverse for ff and F
        if on in ["ff", "F", "Hf", "hf", "df"] and bound == "lower":
            bound = "upper"
        elif on in ["ff", "F", "Hf", "hf", "df"] and bound == "upper":
            bound = "lower"

        cb = self.R_cb(
            x,
            bound=bound,
            interp=interp,
            alpha_ci=alpha_ci,
            bound_type=bound_type,
            dist=dist,
        )

        if (on == "ff") or (on == "F"):
            cb = 1.0 - cb

        elif on == "Hf":
            cb = -np.log(cb)

        elif (on == "sf") or (on == "R"):
            if bound == "two-sided":
                cb = np.fliplr(cb)

        np.seterr(**old_err_state)

        return cb

    def R_cb(
        self,
        x,
        bound="two-sided",
        interp="step",
        alpha_ci=0.05,
        bound_type="exp",
        dist="z",
    ):
        if bound_type not in ["exp", "normal"]:
            raise ValueError("'bound_type' must be in ['exp', 'normal']")
        if dist not in ["t", "z"]:
            raise ValueError("'dist' must be in ['t', 'z']")
        if getattr(self, "greenwood", None) is None:
            raise ValueError(
                "Model has no variance estimate so confidence bounds "
                + "cannot be computed. This occurs for models created "
                + "with 'fit_from_ecdf' since the at risk and death "
                + "counts are unknown."
            )

        confidence = 1.0 - alpha_ci

        old_err_state = np.seterr(all="ignore")

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
            # print(self.greenwood)
            R_out = self.greenwood * 1.0 / (np.log(self.R) ** 2)
            R_out = np.log(-np.log(self.R)) - stat * np.sqrt(R_out)
            R_out = np.exp(-np.exp(R_out))
            R_out = np.where(self.greenwood == 0, 1, R_out)
        else:
            # Normal Greenwood confidence
            R_out = self.R + np.sqrt(self.greenwood * self.R**2) * stat

        # Allows for confidence bound to be estimated up to the last value.
        # only used in event that there is no right censoring.
        if bound == "upper":
            R_out = np.where(np.isfinite(R_out), R_out, np.nanmin(R_out))
        elif bound == "lower":
            R_out = np.where(np.isfinite(R_out), R_out, 0)
        else:
            R_out[0, :] = np.where(
                np.isfinite(R_out[0, :]), R_out[0, :], np.nanmin(R_out[0, :])
            )
            R_out[1, :] = np.where(np.isfinite(R_out[1, :]), R_out[1, :], 0)

        if interp == "step":
            idx = np.searchsorted(self.x, x, side="right") - 1
            if bound == "two-sided":
                R_out = R_out[:, idx]
                R_out = np.where(idx < 0, 1, R_out)
            else:
                R_out = R_out[idx]
                R_out = np.where(idx < 0, 1, R_out)

        else:
            if bound == "two-sided":
                R1 = interp_function(self.x, R_out[0, :], kind=interp)(x)
                R2 = interp_function(self.x, R_out[1, :], kind=interp)(x)
                R_out = np.vstack([R1, R2])
            else:
                R_out = interp_function(self.x, R_out, kind=interp)(x)

        # The question remains. Should bounds above and below observed values
        # be calculable?...
        mask = (x < self.x.min()) | (x > self.x.max())
        R_out = np.where(mask, np.nan, R_out)

        if bound == "two-sided":
            R_out = R_out.T

        np.seterr(**old_err_state)

        return R_out

    def random(self, size):
        r"""
        Draws random samples from the fitted distribution. Each observed
        value x is drawn with the probability mass the estimated survival
        function assigns to it. If the estimate does not reach zero (e.g.
        due to right censoring) the remaining mass is distributed over the
        observed values, i.e. sampling is conditional on an event occurring
        at one of the observed values.
        """
        with np.errstate(all="ignore"):
            p = -np.diff(np.hstack([[1.0], self.R]))
        p = np.where(np.isfinite(p), p, 0)
        p = p / p.sum()
        return np.random.choice(self.x, size=size, p=p)

    def qf(self, p):
        r"""
        Quantile function of the non-parametric estimate. Returns the
        smallest observed value at which the estimated CDF reaches, or
        exceeds, the probability p.

        Parameters
        ----------

        p : array like or scalar
            The probabilities at which the quantile will be computed.
            Values must be in (0, 1].

        Returns
        -------

        q : numpy array
            The value(s) of the quantile at each p. NaN where the
            estimated CDF never reaches p (e.g. due to right censoring).

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = KaplanMeier.fit(x)
        >>> model.qf(0.5)
        array([3.])
        >>> model.qf([0.1, 0.5, 0.9])
        array([1., 3., 5.])
        """
        p = np.atleast_1d(p).astype(float)
        if ((p <= 0) | (p > 1)).any():
            raise ValueError("'p' must be in the range (0, 1]")
        idx = np.searchsorted(self.F, p, side="left")
        x_padded = np.hstack([self.x.astype(float), [np.nan]])
        return x_padded[np.minimum(idx, len(self.x))]

    @property
    def median(self):
        r"""
        The median survival time; the smallest observed value at which
        the estimated CDF reaches, or exceeds, 0.5. NaN if the estimate
        never reaches 0.5 (e.g. due to right censoring).
        """
        return self.qf(0.5)[0]

    def quantile_cb(self, p, alpha_ci=0.05, bound_type="exp", dist="z"):
        r"""
        Two-sided confidence interval of the quantile at each
        probability p using the Brookmeyer-Crowley method: the interval
        is the set of times at which the pointwise confidence interval
        of the survival function contains 1 - p.

        Parameters
        ----------

        p : array like or scalar
            The probabilities at which the quantile interval will be
            computed. Values must be in (0, 1].
        alpha_ci : scalar, optional
            The level of significance at which the interval will be
            computed. Defaults to 0.05.
        bound_type : ('exp', 'normal'), str, optional
            The method for the underlying survival function bounds.
        dist : ('t', 'z'), str, optional
            The statistic used in the underlying survival function
            bounds.

        Returns
        -------

        cb : numpy array
            Array of shape (len(p), 2) with the ``[lower, upper]``
            interval of the quantile for each p. The upper limit is NaN
            where the relevant bound of the survival function never
            crosses 1 - p (i.e. the interval is open to the right).
        """
        p = np.atleast_1d(p).astype(float)
        if ((p <= 0) | (p > 1)).any():
            raise ValueError("'p' must be in the range (0, 1]")

        bounds = self.cb(
            self.x,
            on="sf",
            bound="two-sided",
            alpha_ci=alpha_ci,
            bound_type=bound_type,
            dist=dist,
        )
        lower_sf, upper_sf = bounds[:, 0], bounds[:, 1]

        out = np.empty((p.size, 2))
        for i, p_i in enumerate(p):
            level = 1.0 - p_i
            # Times enter the interval when the lower survival bound
            # falls to the level, and leave it once the upper survival
            # bound falls below the level.
            in_lower = lower_sf <= level
            in_upper = upper_sf < level
            out[i, 0] = (
                self.x[np.argmax(in_lower)] if in_lower.any() else np.nan
            )
            out[i, 1] = (
                self.x[np.argmax(in_upper)] if in_upper.any() else np.nan
            )
        return out

    def mean(self, tau=None):
        r"""
        The (restricted) mean survival time: the area under the
        estimated survival function from 0 to tau.

        If the survival function reaches zero this is the mean of the
        estimated distribution. With right censoring the survival
        function does not reach zero and the unrestricted mean is
        undefined; the restricted mean up to tau (defaulting to the
        largest observed value) is reported instead, which is the
        standard restricted mean survival time (RMST).

        Parameters
        ----------

        tau : scalar, optional
            The horizon up to which the survival function is
            integrated. Defaults to the largest observed value. If tau
            is beyond the last observation the survival function is
            extended at its final value.

        Returns
        -------

        mean : float
            The restricted mean survival time.

        Examples
        --------
        >>> from surpyval import KaplanMeier
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> model = KaplanMeier.fit(x)
        >>> model.mean()
        3.0
        """
        if np.min(self.x) < 0:
            raise ValueError(
                "Mean survival time requires non-negative observations"
            )
        if tau is None:
            tau = np.max(self.x)

        xs = self.x[self.x < tau].astype(float)
        times = np.hstack([[0.0], xs, [tau]])
        surv = np.hstack([[1.0], self.R[: xs.size]])
        return float(np.sum(np.diff(times) * surv))

    def mean_cb(self, tau=None, alpha_ci=0.05):
        r"""
        Two-sided confidence interval of the (restricted) mean survival
        time, using the normal approximation with the standard variance
        estimate:

        .. math::
            \widehat{Var}(\hat{\mu}) = \sum_{i: x_i \leq \tau}
                A_i^2 v_i

        where :math:`A_i` is the area under the survival function from
        :math:`x_i` to :math:`\tau` and :math:`v_i` is the variance
        increment of the cumulative hazard at :math:`x_i` (e.g. the
        Greenwood increment for the Kaplan-Meier estimator).

        Parameters
        ----------

        tau : scalar, optional
            The horizon up to which the survival function is
            integrated. Defaults to the largest observed value.
        alpha_ci : scalar, optional
            The level of significance at which the interval will be
            computed. Defaults to 0.05.

        Returns
        -------

        cb : numpy array
            The ``[lower, upper]`` interval of the (restricted) mean.
        """
        if getattr(self, "greenwood", None) is None:
            raise ValueError(
                "Model has no variance estimate so confidence bounds "
                + "cannot be computed. This occurs for models created "
                + "with 'fit_from_ecdf' since the at risk and death "
                + "counts are unknown."
            )
        if tau is None:
            tau = np.max(self.x)

        mu = self.mean(tau=tau)

        # Area under the survival curve from each observation to tau
        xs = self.x.astype(float)
        upper_t = np.minimum(np.hstack([xs[1:], [np.inf]]), tau)
        widths = np.clip(upper_t - np.minimum(xs, tau), 0, None)
        seg_area = widths * self.R
        # A[i] is the area from x[i] to tau
        A = np.cumsum(seg_area[::-1])[::-1]

        v = np.diff(np.hstack([[0.0], self.greenwood]))
        with np.errstate(all="ignore"):
            terms = np.where(A > 0, A**2 * v, 0.0)
        var = np.sum(terms)

        z = norm.ppf(1 - alpha_ci / 2)
        se = np.sqrt(var)
        return np.array([mu - z * se, mu + z * se])

    def bootstrap_cb(
        self,
        x,
        bound="two-sided",
        alpha_ci=0.05,
        B=200,
        random_state=None,
    ):
        r"""
        Confidence bounds of the survival function computed with a
        non-parametric bootstrap: the data are resampled with
        replacement, the model is refitted with the same estimator, and
        the percentile interval across the refits is taken at each x.

        This is the recommended way to compute bounds for the Turnbull
        estimator. The Greenwood-style bounds from ``cb()`` treat the
        expected (fractional) at risk and death counts from the
        Turnbull EM as if they were observed counts, which ignores the
        uncertainty in the EM allocation itself; the bootstrap does
        not.

        Note that the Turnbull NPMLE only identifies the probability
        mass within each Turnbull interval, not how it is distributed
        inside one. Point estimates and bounds evaluated strictly
        inside an interval therefore reflect the step convention rather
        than an estimate of the underlying continuous survival
        function, and are best evaluated at the interval bounds.

        Parameters
        ----------

        x : array like or scalar
            The values at which the confidence bounds will be
            calculated.
        bound : ('two-sided', 'upper', 'lower'), str, optional
            Compute either the two-sided, upper or lower confidence
            bound(s). Defaults to two-sided.
        alpha_ci : scalar, optional
            The level of significance at which the bound will be
            computed. Defaults to 0.05.
        B : int, optional
            The number of bootstrap resamples. Defaults to 200. Larger
            values give smoother bounds at a linear cost in runtime;
            note that refitting the Turnbull estimator is relatively
            expensive.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for reproducible resampling.

        Returns
        -------

        cb : numpy array
            For two-sided bounds an array of shape (len(x), 2) with
            ``[lower, upper]`` columns; otherwise an array of the
            requested bound at each x.
        """
        if getattr(self, "data", None) is None or "x" not in self.data:
            raise ValueError(
                "Bootstrap requires the data the model was fitted "
                + "with. Models created with 'from_xrd' or "
                + "'fit_from_ecdf' cannot be bootstrapped."
            )
        # Imported here as the package imports this module on init.
        from surpyval.univariate import nonparametric as nonp
        from surpyval.utils import xcnt_to_xrd

        x_eval = np.atleast_1d(x).astype(float)
        x_data = self.data["x"]
        c_data = self.data["c"]
        n_data = self.data["n"]
        t_data = self.data["t"]

        rng = np.random.default_rng(random_state)
        N = int(n_data.sum())
        probs = n_data / n_data.sum()

        old_err_state = np.seterr(all="ignore")
        R_boot = np.empty((B, x_eval.size))
        for b in range(B):
            n_b = rng.multinomial(N, probs)
            keep = n_b > 0
            if self.model == "Turnbull":
                fitted = nonp.turnbull(
                    x_data[keep],
                    c_data[keep],
                    n_b[keep],
                    t_data[keep],
                    estimator=self.data["estimator"],
                )
                x_b, R_b = fitted["x"], fitted["R"]
            else:
                x_b, r_b, d_b = xcnt_to_xrd(
                    x_data[keep], c_data[keep], n_b[keep], t_data[keep]
                )
                R_b = nonp.FIT_FUNCS[self.model](r_b, d_b)
            idx = np.searchsorted(x_b, x_eval, side="right") - 1
            R_boot[b, :] = np.where(
                idx < 0, 1.0, R_b[np.clip(idx, 0, len(x_b) - 1)]
            )
        np.seterr(**old_err_state)

        if bound == "two-sided":
            qs = np.quantile(R_boot, [alpha_ci / 2, 1 - alpha_ci / 2], axis=0)
            return qs.T
        elif bound == "lower":
            return np.quantile(R_boot, alpha_ci, axis=0)
        elif bound == "upper":
            return np.quantile(R_boot, 1 - alpha_ci, axis=0)
        else:
            raise ValueError(
                "'bound' must be in ['two-sided', 'upper', 'lower']"
            )

    def get_plot_data(self, **kwargs):
        y_scale_min = 0
        y_scale_max = 1

        # x-axis
        x_min = min(0, np.min(self.x))
        x_max = np.max(self.x)

        diff = (x_max - x_min) / 10
        x_scale_min = x_min
        x_scale_max = x_max + diff

        cbs = self.R_cb(self.x, **kwargs)

        return {
            "x_scale_min": x_scale_min,
            "x_scale_max": x_scale_max,
            "y_scale_min": y_scale_min,
            "y_scale_max": y_scale_max,
            "cbs": cbs,
            "x_": self.x,
            "R": self.R,
            "F": self.F,
        }

    def plot(self, ax=None, **kwargs):
        r"""
        Creates a plot of the survival function.

        Two-sided confidence bounds are drawn as a shaded band in the
        same colour as the survival curve, and right censored
        observations are marked with ticks on the curve. Any keyword
        arguments not listed below (e.g. ``color`` or ``label``) are
        passed to the matplotlib plotting call for the survival curve.

        Parameters
        ----------

        ax : matplotlib axis, optional
            The axis on which the plot will be drawn. Defaults to the
            current axis.
        plot_bounds : bool, optional
            Whether to draw the confidence bounds. Defaults to True.
        show_censors : bool, optional
            Whether to mark right censored observations on the curve.
            Defaults to True.
        interp : ('step', 'linear', 'cubic'), optional
            How to draw the curve between observations.
        bound, alpha_ci, bound_type, dist : optional
            Passed to the confidence bound calculation; see ``cb()``.

        Returns
        -------

        ax : matplotlib axis
        """
        if ax is None:
            ax = plt.gcf().gca()

        plot_bounds = kwargs.pop("plot_bounds", True)
        show_censors = kwargs.pop("show_censors", True)
        interp = kwargs.pop("interp", "step")
        bound = kwargs.pop("bound", "two-sided")
        alpha_ci = kwargs.pop("alpha_ci", 0.05)
        bound_type = kwargs.pop("bound_type", "exp")
        dist = kwargs.pop("dist", "z")

        d = self.get_plot_data(
            interp=interp,
            bound=bound,
            alpha_ci=alpha_ci,
            bound_type=bound_type,
            dist=dist,
        )
        # MAKE THE PLOT
        # Set the y limits
        ax.set_ylim([d["y_scale_min"], d["y_scale_max"]])

        # Label it
        ax.set_title("Model Survival Plot")
        ax.set_ylabel("R")
        if interp != "step":
            (line,) = ax.plot(d["x_"], d["R"], **kwargs)
        else:
            (line,) = ax.step(d["x_"], d["R"], where="post", **kwargs)
        color = line.get_color()

        if plot_bounds:
            cbs = d["cbs"]
            band_kwargs = {"alpha": 0.3, "color": color, "linewidth": 0}
            if interp == "step":
                band_kwargs["step"] = "post"
            if np.ndim(cbs) == 2:
                ax.fill_between(d["x_"], cbs[:, 0], cbs[:, 1], **band_kwargs)
            elif interp == "step":
                ax.step(
                    d["x_"], cbs, where="post", color=color, linestyle="--"
                )
            else:
                ax.plot(d["x_"], cbs, color=color, linestyle="--")

        if show_censors and getattr(self, "data", None) is not None:
            x_data = self.data["x"]
            c_data = self.data["c"]
            if np.ndim(x_data) == 1 and (c_data == 1).any():
                x_cens = x_data[c_data == 1]
                ax.plot(
                    x_cens,
                    self.sf(x_cens, interp=interp),
                    linestyle="",
                    marker="|",
                    markersize=10,
                    markeredgewidth=1.5,
                    color=color,
                )

        return ax

    @classmethod
    def fit_from_ecdf(cls, x, R):
        out = cls()
        out.model = "from_ecdf"
        out.R = R
        out.x = x
        out.F = 1 - out.R
        with np.errstate(all="ignore"):
            out.H = -np.log(out.R)
        # Without r and d there is no variance estimate, and therefore
        # no confidence bounds, for the model.
        out.greenwood = None

        return out
