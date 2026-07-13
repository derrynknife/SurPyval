import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d
from scipy.stats import norm, t

from surpyval.distribution import NonParametricDistribution

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def interp_function(
    x: npt.ArrayLike, y: npt.ArrayLike, kind: str
) -> Callable[[npt.ArrayLike], npt.NDArray]:
    if kind == "cubic":
        # A plain cubic spline can overshoot and produce a non-monotone
        # (even out-of-[0, 1]) survival curve, which then propagates into
        # ``Hf``, ``hf`` and the interpolated confidence bounds. PCHIP is a
        # shape-preserving piecewise-cubic Hermite interpolant, so it stays
        # monotone wherever the data are monotone. It requires strictly
        # increasing abscissae, so collapse any duplicated ``x`` (e.g. the
        # zero-width Turnbull bounds at exactly observed times) to the last
        # value there, which is where the step function has settled.
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        keep = np.append(np.diff(x) > 0, True)
        pchip = PchipInterpolator(x[keep], y[keep], extrapolate=False)
        return lambda q: pchip(np.asarray(q, dtype=float))
    return interp1d(x, y, kind=kind, bounds_error=False, fill_value=np.nan)


class NonParametric(NonParametricDistribution):
    """
    Result of ``.fit()`` method for every non-parametric
    surpyval distribution. This means that each of the
    methods in this class can be called with a model created
    from the ``NelsonAalen``, ``KaplanMeier``,
    ``FlemingHarrington``, or ``Turnbull`` estimators.
    """

    # Attributes populated by the fitter (``NonParametricFitter.fit`` /
    # ``from_xrd`` / ``fit_from_ecdf``). Declared here so static type
    # checkers know their types.
    x: npt.NDArray
    r: npt.NDArray
    d: npt.NDArray
    R: npt.NDArray
    F: npt.NDArray
    H: npt.NDArray
    model: str
    greenwood: npt.NDArray
    data: dict[str, Any]

    def __repr__(self) -> str:
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

    def sf(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
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

    def ff(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
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

    def hf(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
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

    def df(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
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

    def Hf(self, x: npt.ArrayLike, interp: str = "step") -> npt.NDArray:
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
        x: npt.ArrayLike,
        on: str = "sf",
        bound: str = "two-sided",
        interp: str = "step",
        alpha_ci: float = 0.05,
        bound_type: str = "exp",
        dist: str = "z",
    ) -> npt.NDArray:
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
        x: npt.ArrayLike,
        bound: str = "two-sided",
        interp: str = "step",
        alpha_ci: float = 0.05,
        bound_type: str = "exp",
        dist: str = "z",
    ) -> npt.NDArray:
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

    def random(
        self, size: int, random_state: int | None = None
    ) -> npt.NDArray:
        r"""
        Draws random samples from the fitted distribution. Each observed
        value x is drawn with the probability mass the estimated survival
        function assigns to it. If the estimate does not reach zero (e.g.
        due to right censoring) the remaining mass is distributed over the
        observed values, i.e. sampling is conditional on an event occurring
        at one of the observed values.

        Parameters
        ----------

        size : int
            The number of random samples to draw.
        random_state : int or numpy.random.Generator, optional
            Seed or generator for reproducible sampling. Matches the
            ``random_state`` argument of ``bootstrap_cb`` and ``band``.

        Returns
        -------

        random : numpy array
            The random samples drawn from the observed values.
        """
        with np.errstate(all="ignore"):
            p = -np.diff(np.hstack([[1.0], self.R]))
        p = np.where(np.isfinite(p), p, 0)
        p = p / p.sum()
        rng = np.random.default_rng(random_state)
        return rng.choice(self.x, size=size, p=p)

    def qf(self, p: npt.ArrayLike) -> npt.NDArray:
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
    def median(self) -> float:
        r"""
        The median survival time; the smallest observed value at which
        the estimated CDF reaches, or exceeds, 0.5. NaN if the estimate
        never reaches 0.5 (e.g. due to right censoring).
        """
        return self.qf(0.5)[0]

    def quantile_cb(
        self,
        p: npt.ArrayLike,
        alpha_ci: float = 0.05,
        bound_type: str = "exp",
        dist: str = "z",
    ) -> npt.NDArray:
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

    def mean(self, tau: float | None = None) -> float:
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

    def mean_cb(
        self, tau: float | None = None, alpha_ci: float = 0.05
    ) -> npt.NDArray:
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
        x: npt.ArrayLike,
        bound: str = "two-sided",
        alpha_ci: float = 0.05,
        B: int = 200,
        random_state: int | None = None,
    ) -> npt.NDArray:
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

    @staticmethod
    def _band_critical_value(
        a_l: float,
        a_u: float,
        alpha_ci: float,
        standardized: bool,
        n_sims: int,
        random_state: int | None,
    ) -> float:
        # Critical value of the supremum of a Brownian bridge (for the
        # Hall-Wellner band), or of a standardized Brownian bridge
        # B(u)/sqrt(u(1 - u)) (for the equal precision band), over
        # [a_l, a_u]. Computed by Monte Carlo simulation of bridge
        # paths on a grid.
        rng = np.random.default_rng(random_state)
        m = 1000
        u = np.arange(1, m + 1) / m
        in_band = (u >= a_l) & (u <= a_u)
        sups = np.empty(n_sims)
        chunk = 1000
        for start in range(0, n_sims, chunk):
            size = min(chunk, n_sims - start)
            W = np.cumsum(rng.standard_normal((size, m)) / np.sqrt(m), axis=1)
            bridge = W - u * W[:, -1:]
            paths = np.abs(bridge[:, in_band])
            if standardized:
                paths = paths / np.sqrt(u[in_band] * (1 - u[in_band]))
            sups[start : start + size] = paths.max(axis=1)
        return np.quantile(sups, 1 - alpha_ci)

    def band(
        self,
        x: npt.ArrayLike | None = None,
        method: str = "hall-wellner",
        bound_type: str = "exp",
        alpha_ci: float = 0.05,
        n_sims: int = 10_000,
        random_state: int | None = 1,
    ) -> npt.NDArray:
        r"""
        Simultaneous confidence band of the survival function.

        The pointwise bounds from ``cb()`` cover the true value of the
        survival function at each single time with probability
        1 - alpha_ci, but the probability that the *whole* true curve
        lies between them is lower, since the curve has many
        opportunities to escape. A confidence band is widened so that,
        with probability 1 - alpha_ci, the entire survival function
        lies within the band over the observed range. Use the band, not
        the pointwise bounds, to assess whether a hypothesised curve
        (e.g. a fitted parametric distribution) is consistent with the
        data as a whole.

        Two classical bands are available:

        - "hall-wellner": width proportional to
          :math:`(1 + n\sigma^2(t))/\sqrt{n}`; tends to be relatively
          wider in the middle of the curve.
        - "nair" (equal precision): width proportional to the pointwise
          standard error, i.e. the band is the pointwise interval
          scaled by a larger critical value, so its width follows the
          pointwise bounds everywhere.

        Critical values are computed by Monte Carlo simulation of the
        limiting Brownian bridge process, with a fixed default seed so
        results are reproducible.

        The band is only defined between the first and last observed
        events (where the variance estimate is positive and finite);
        NaN is returned outside that range. The asymptotic theory for
        these bands is for right censored data; for Turnbull models
        with interval censoring prefer ``bootstrap_cb()``.

        Parameters
        ----------

        x : array like or scalar, optional
            The values at which the band will be evaluated. Defaults to
            the observed values.
        method : ('hall-wellner', 'nair'), str, optional
            The type of band. Defaults to 'hall-wellner'.
        bound_type : ('exp', 'normal'), str, optional
            As for ``cb()``: 'exp' applies the band on the log(-log)
            scale, keeping it within [0, 1]. Defaults to 'exp'.
        alpha_ci : scalar, optional
            The level of significance of the band. Defaults to 0.05.
        n_sims : int, optional
            Number of simulated paths for the critical value.
        random_state : int or numpy.random.Generator, optional
            Seed for the critical value simulation. Defaults to a fixed
            seed for reproducibility.

        Returns
        -------

        band : numpy array
            Array of shape (len(x), 2) with the ``[lower, upper]`` band
            values for the survival function at each x.

        References
        ----------

        Hall, W. J. and Wellner, J. A. (1980), "Confidence bands for a
        survival curve from censored data", Biometrika 67, 133-143.

        Nair, V. N. (1984), "Confidence bands for survival functions
        with censored data: a comparative study", Technometrics 26,
        265-275.

        Klein, J. P. and Moeschberger, M. L. (2003), "Survival
        Analysis", 2nd ed., Section 4.4.
        """
        if method not in ["hall-wellner", "nair"]:
            raise ValueError("'method' must be in ['hall-wellner', 'nair']")
        if bound_type not in ["exp", "normal"]:
            raise ValueError("'bound_type' must be in ['exp', 'normal']")
        if getattr(self, "greenwood", None) is None:
            raise ValueError(
                "Model has no variance estimate so confidence bands "
                + "cannot be computed. This occurs for models created "
                + "with 'fit_from_ecdf' since the at risk and death "
                + "counts are unknown."
            )

        if getattr(self, "data", None) is not None and "n" in self.data:
            N = float(self.data["n"].sum())
        else:
            N = float(np.max(self.r))

        old_err_state = np.seterr(all="ignore")

        sigma2 = self.greenwood
        valid = (
            np.isfinite(sigma2) & (sigma2 > 0) & (self.R > 0) & (self.R < 1)
        )

        if not valid.any():
            np.seterr(**old_err_state)
            raise ValueError(
                "Band is undefined: no observations with a positive, "
                + "finite variance estimate"
            )

        a = N * sigma2 / (1 + N * sigma2)
        a_l = a[valid].min()
        a_u = a[valid].max()

        crit = self._band_critical_value(
            a_l,
            a_u,
            alpha_ci,
            standardized=(method == "nair"),
            n_sims=n_sims,
            random_state=random_state,
        )

        if method == "nair":
            half_width = crit * np.sqrt(sigma2)
        else:
            half_width = crit * (1 + N * sigma2) / np.sqrt(N)

        if bound_type == "exp":
            # Band applied on the log(-log) scale, mirroring the
            # pointwise exponential Greenwood bounds.
            theta = np.log(-np.log(self.R))
            se = half_width / np.abs(np.log(self.R))
            lower = np.exp(-np.exp(theta + se))
            upper = np.exp(-np.exp(theta - se))
        else:
            lower = self.R - half_width * self.R
            upper = self.R + half_width * self.R

        lower = np.where(valid, lower, np.nan)
        upper = np.where(valid, upper, np.nan)

        if x is None:
            x = self.x
        x = np.atleast_1d(x).astype(float)
        idx = np.searchsorted(self.x, x, side="right") - 1
        idx_c = np.clip(idx, 0, len(self.x) - 1)
        out = np.empty((x.size, 2))
        out[:, 0] = np.where(idx < 0, np.nan, lower[idx_c])
        out[:, 1] = np.where(idx < 0, np.nan, upper[idx_c])
        outside = (x < self.x.min()) | (x > self.x.max())
        out[outside] = np.nan

        np.seterr(**old_err_state)

        return out

    def smoothed_hf(
        self, x: npt.ArrayLike, bandwidth: float | None = None
    ) -> npt.NDArray:
        r"""
        Kernel smoothed estimate of the hazard rate, using an
        Epanechnikov kernel over the increments of the cumulative
        hazard estimate:

        .. math::
            \hat{h}(t) = \frac{1}{b} \sum_{i} K\left (
                \frac{t - x_i}{b} \right ) \Delta \hat{H}(x_i)

        Contributions are renormalised near the boundaries of the
        observed range so the estimate is not biased downward where the
        kernel window extends past the data.

        This is a better estimate of the hazard rate than ``hf()``,
        which simply differences the cumulative hazard between the
        requested points.

        Parameters
        ----------

        x : array like or scalar
            The values at which the hazard rate will be estimated.
        bandwidth : scalar, optional
            The kernel bandwidth in the units of x. Defaults to a rough
            rule of thumb (one eighth of the observed range); for
            serious use choose by inspection or cross-validation.

        Returns
        -------

        hf : numpy array
            The estimated hazard rate at each x. NaN outside the
            observed range.

        References
        ----------

        Klein, J. P. and Moeschberger, M. L. (2003), "Survival
        Analysis", 2nd ed., Section 6.2.
        """
        x = np.atleast_1d(x).astype(float)

        with np.errstate(all="ignore"):
            dH = np.diff(np.hstack([[0.0], self.H]))
        dH = np.where(np.isfinite(dH), dH, 0.0)

        x_min = self.x.min()
        x_max = self.x.max()
        if bandwidth is None:
            bandwidth = (x_max - x_min) / 8
        if bandwidth <= 0:
            raise ValueError("'bandwidth' must be positive")

        u = (x[:, None] - self.x[None, :]) / bandwidth
        kern = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
        h = (kern * dH).sum(axis=1) / bandwidth

        # Renormalise by the kernel mass that lies within the observed
        # range, correcting the downward bias near the boundaries. The
        # Epanechnikov CDF is (2 + 3v - v^3) / 4 on [-1, 1].
        def epa_cdf(v):
            v = np.clip(v, -1, 1)
            return (2 + 3 * v - v**3) / 4

        lo = (x - x_max) / bandwidth
        hi = (x - x_min) / bandwidth
        mass = epa_cdf(hi) - epa_cdf(lo)
        with np.errstate(all="ignore"):
            h = np.where(mass > 0, h / mass, np.nan)

        h = np.where((x < x_min) | (x > x_max), np.nan, h)
        return h

    def get_plot_data(self, **kwargs) -> dict:
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

    def plot(self, ax: "Axes | None" = None, **kwargs) -> Any:
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
        ax.set_ylim((d["y_scale_min"], d["y_scale_max"]))

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
            band_kwargs: dict[str, Any] = {
                "alpha": 0.3,
                "color": color,
                "linewidth": 0,
            }
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
    def fit_from_ecdf(
        cls, x: npt.ArrayLike, R: npt.ArrayLike
    ) -> "NonParametric":
        out = cls()
        out.model = "from_ecdf"
        out.R = np.asarray(R)
        out.x = np.asarray(x)
        out.F = 1 - out.R
        with np.errstate(all="ignore"):
            out.H = -np.log(out.R)
        # Without r and d there is no variance estimate, and therefore
        # no confidence bounds, for the model.
        out.greenwood = None  # type: ignore[assignment]

        return out

    # The estimator ladder and derived curves that fully describe a fitted
    # model; everything the public methods need is a function of these.
    _SERIALIZED_ARRAYS = ("x", "r", "d", "R", "F", "H", "greenwood")

    def to_dict(self, with_data: bool = False) -> dict:
        r"""
        Serialize the fitted non-parametric model to a plain dictionary,
        mirroring the parametric ``to_dict``. The estimator ladder
        (``x``, ``r``, ``d``), the derived curves (``R``, ``F``, ``H``),
        the variance estimate (``greenwood``) and the estimator name are
        stored, which is everything the model's methods need to be
        reconstructed with :meth:`from_dict`.

        Parameters
        ----------

        with_data : bool, optional
            Also store the raw ``x``/``c``/``n``/``t`` data the model was
            fitted with (needed to reconstruct a model that can call
            :meth:`bootstrap_cb`). Defaults to False.

        Returns
        -------

        model_dict : dict
            The serialized model.
        """
        out: dict[str, Any] = {"parameterization": "non-parametric"}
        out["model"] = self.model
        for attr in self._SERIALIZED_ARRAYS:
            value = getattr(self, attr, None)
            out[attr] = None if value is None else np.asarray(value).tolist()

        if "estimator" in getattr(self, "data", {}):
            out["estimator"] = self.data["estimator"]

        if with_data and getattr(self, "data", None) is not None:
            data_dict: dict[str, Any] = {}
            for ch in ["x", "c", "n", "t"]:
                value = self.data.get(ch, None)
                data_dict[ch] = (
                    None if value is None else np.asarray(value).tolist()
                )
            out["data"] = data_dict

        return out

    def to_json(self, fp: str | Path) -> None:
        with open(fp, "w+") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def from_dict(cls, model_dict: dict) -> "NonParametric":
        r"""
        Reconstruct a fitted non-parametric model from a dictionary
        produced by :meth:`to_dict`.
        """
        if model_dict.get("parameterization") != "non-parametric":
            raise ValueError(
                "Must create a non-parametric model from a non-parametric "
                "model dict"
            )
        out = cls()
        out.model = model_dict["model"]
        for attr in cls._SERIALIZED_ARRAYS:
            value = model_dict.get(attr, None)
            if value is None:
                # ``greenwood`` is legitimately absent (no variance
                # estimate, e.g. ``fit_from_ecdf``); leave it as None so
                # the confidence-bound guards fire as they would on the
                # original model.
                if attr == "greenwood":
                    out.greenwood = None  # type: ignore[assignment]
            else:
                setattr(out, attr, np.asarray(value))

        if "data" in model_dict or "estimator" in model_dict:
            data: dict[str, Any] = {}
            raw = model_dict.get("data", {})
            for ch in ["x", "c", "n", "t"]:
                value = raw.get(ch, None)
                if value is not None:
                    data[ch] = np.asarray(value)
            if "estimator" in model_dict:
                data["estimator"] = model_dict["estimator"]
            out.data = data

        return out

    @classmethod
    def from_json(cls, fp: str | Path) -> "NonParametric":
        with open(fp, "r") as f:
            return cls.from_dict(json.load(f))
