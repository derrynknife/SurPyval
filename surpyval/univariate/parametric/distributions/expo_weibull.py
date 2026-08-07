from scipy import integrate
from scipy.special import xlogy

from surpyval import np
from surpyval.univariate import parametric as para
from surpyval.univariate.parametric.parametric_fitter import (
    OptimisedFitMixin,
    ParametricFitter,
)


class ExpoWeibull_(OptimisedFitMixin, ParametricFitter):
    def __init__(self, name):
        super().__init__(
            name=name,
            k=3,
            bounds=(
                (0, None),
                (0, None),
                (0, None),
            ),
            support=(0, np.inf),
            param_names=["alpha", "beta", "mu"],
            param_map={"alpha": 0, "beta": 1, "mu": 2},
            plot_x_scale="log",
        )
        self.supports_mpp = False

    def _gumbel_seed(self, x, c, n, refine):
        """
        Seed alpha and beta from a Gumbel fit to log(x).

        The ExpoWeibull with mu = 1 is a Weibull, and a Weibull's logs
        are Gumbel distributed with mu = log(alpha) and sigma = 1 / beta,
        so a fit of log(x) gives both shape parameters at once.

        ``refine`` runs the Gumbel MLE rather than reading the
        probability plot alone. Without an offset the plot is already
        good enough: refining cost 15-30% of the fit and changed nothing
        over 54 parameter combinations plus right, left and heavily tied
        data, every fit reaching the same optimum to the optimiser's own
        tolerance. With an offset the plot alone is measurably worse --
        five of 48 offset fits landed on a worse optimum, one of them at
        685.85 against 622.26 -- so the offset path refines.
        """
        log_x = np.log(x)
        log_x[np.isnan(log_x)] = 0
        gumb = para.Gumbel.fit(log_x, c, n, how="MLE" if refine else "MPP")
        if refine and not gumb.res.success:
            gumb = para.Gumbel.fit(log_x, c, n, how="MPP")
        mu, sigma = gumb.params
        alpha, beta = np.exp(mu), 1.0 / sigma
        if np.isinf(alpha) | np.isnan(alpha):
            alpha = np.median(x)
        if np.isinf(beta) | np.isnan(beta):
            beta = 1.0
        return alpha, beta

    def _parameter_initialiser(self, data, offset=False):
        x, c, n = data.x, data.c, data.n
        if offset:
            # Estimate the offset first and seed alpha and beta from the
            # shifted data. Taking logs before removing the shift reads
            # log(x) instead of log(x - gamma), and a large shift
            # compresses those logs into a narrow band: at gamma = 100
            # with a true beta of 2 the seed came back at beta = 23 and
            # the MLE then failed outright, falling back to MPP.
            #
            # min(x) - 1 rather than a fraction of the range because the
            # fitter overwrites the returned offset with exactly that
            # (see ParametricFitter.fit_from_surpyval_data); seeding
            # alpha and beta against a different shift than the one
            # actually installed defeats the point of shifting at all.
            gamma = np.min(x) - 1.0
            alpha, beta = self._gumbel_seed(x - gamma, c, n, refine=True)
            return np.array([gamma, alpha, beta, 1.0], dtype=float)
        return np.array(
            [*self._gumbel_seed(x, c, n, refine=False), 1.0],
            dtype=float,
        )

    def sf(self, x, alpha, beta, mu):
        r"""

        Survival (or reliability) function for the ExpoWeibull Distribution:

        .. math::
            R(x) = 1 - \left [ 1 - e^{-\left ( \frac{x}{\alpha} \right )^\beta}
             \right ]^{\mu}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.sf(x, 3, 4, 1.2)
        array([9.94911330e-01, 8.72902497e-01, 4.23286791e-01, 5.06674866e-02,
               5.34717283e-04])
        """
        # -expm1(mu * log1p(-exp(-t))) is the cancellation-free form of
        # 1 - (1 - e^-t)^mu: the naive form underflows to exactly 0 once
        # e^-t < 1e-16 (x ~ 2.5 alpha for beta ~ 4), sending Hf/log_sf to
        # inf/-inf for representable tail probabilities (#257).
        return -np.expm1(mu * np.log1p(-np.exp(-((x / alpha) ** beta))))

    def ff(self, x, alpha, beta, mu):
        r"""

        Failure (CDF or unreliability) function for the ExpoWeibull
        Distribution:

        .. math::
            F(x) = \left [ 1 - e^{-\left ( \frac{x}{\alpha} \right )^\beta}
            \right ]^{\mu}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        ff : scalar or numpy array
            The value(s) of the failure function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.ff(x, 3, 4, 1.2)
        array([0.00508867, 0.1270975 , 0.57671321, 0.94933251, 0.99946528])
        """
        return np.power(1 - np.exp(-((x / alpha) ** beta)), mu)

    def cs(self, x, X, alpha, beta, mu):
        r"""

        Conditional survival (or reliability) function for the ExpoWeibull
        Distribution:

        .. math::
            R(x, X) = \frac{R(x + X)}{R(X)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        cs : scalar or numpy array
            The value(s) of the conditional survival function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.cs(x, 1, 3, 4, 1.2)
        array([8.77367129e-01, 4.25451775e-01, 5.09266354e-02, 5.37452200e-04,
               1.35732908e-07])
        """
        return self.sf(x + X, alpha, beta, mu) / self.sf(X, alpha, beta, mu)

    def df(self, x, alpha, beta, mu):
        r"""

        Density function for the ExpoWeibull Distribution:

        .. math::
            f(x) = \mu \left ( \frac{\beta}{\alpha} \right ) \left ( \frac{x}
            {\alpha} \right )^{\beta - 1} \left [ 1 - e^{-\left ( \frac{x}
            {\alpha} \right )^\beta} \right ]^{\mu - 1} e^{- \left ( \frac{x}
            {\alpha} \right )^\beta}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        df : scalar or numpy array
            The value(s) of the density function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.df(x, 3, 4, 1.2)
        array([0.02427515, 0.27589838, 0.53701385, 0.15943643, 0.00330058])
        """
        return (
            (beta * mu * x ** (beta - 1))
            / (alpha**beta)
            * (1 - np.exp(-((x / alpha) ** beta))) ** (mu - 1)
            * np.exp(-((x / alpha) ** beta))
        )

    def hf(self, x, alpha, beta, mu):
        r"""

        Instantaneous hazard rate for the ExpoWeibull Distribution:

        .. math::
            h(x) = \frac{f(x)}{R(x)}

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        hf : scalar or numpy array
            The value(s) of the instantaneous hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.hf(x, 3, 4, 1.2)
        array([0.02439931, 0.3160701 , 1.26867613, 3.14672068, 6.17256436])
        """
        return self.df(x, alpha, beta, mu) / self.sf(x, alpha, beta, mu)

    def Hf(self, x, alpha, beta, mu):
        r"""

        Instantaneous hazard rate for the ExpoWeibull Distribution:

        .. math::
            H(x) = -\ln \left ( R(x) \right )

        Parameters
        ----------

        x : numpy array or scalar
            The values at which the function will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        Hf : scalar or numpy array
            The value(s) of the cumulative hazard rate at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.Hf(x, 3, 4, 1.2)
        array([5.10166141e-03, 1.35931416e-01, 8.59705336e-01, 2.98247086e+00,
               7.53377239e+00])
        """
        return -np.log(self.sf(x, alpha, beta, mu))

    def qf(self, p, alpha, beta, mu):
        r"""

        Instantaneous hazard rate for the ExpoWeibull Distribution:

        .. math::
            q(p) =

        Parameters
        ----------

        p : numpy array or scalar
            The percentiles at which the quantile will be calculated
        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        Q : scalar or numpy array
            The quantiles for the Weibull distribution at each value p

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> p = np.array([.1, .2, .3, .4, .5])
        >>> ExpoWeibull.qf(p, 3, 4, 1.2)
        array([1.89361341, 2.2261045 , 2.46627621, 2.66992747, 2.85807988])
        """
        return alpha * (-np.log1p(-(p ** (1.0 / mu)))) ** (1 / beta)

    def log_df(self, x, alpha, beta, mu):
        return (
            np.log(beta)
            + np.log(mu)
            + (beta - 1) * np.log(x)
            - beta * np.log(alpha)
            + (mu - 1) * np.log1p(-np.exp(-((x / alpha) ** beta)))
            - ((x / alpha) ** beta)
        )

    def log_ff(self, x, alpha, beta, mu):
        return mu * np.log1p(-np.exp(-((x / alpha) ** beta)))

    def log_sf(self, x, alpha, beta, mu):
        # log of the cancellation-free sf form; the naive log1p(-(...)^mu)
        # returns -inf once the inner power rounds to 1 (#257).
        return np.log(
            -np.expm1(mu * np.log1p(-np.exp(-((x / alpha) ** beta))))
        )

    def mean(self, alpha, beta, mu):
        def func(x):
            return x * self.df(x, alpha, beta, mu)

        return integrate.quad(func, 0, np.inf)[0]

    def entropy(self, alpha, beta, mu):
        r"""

        Calculates the entropy of the ExpoWeibull distribution.

        The entropy of the ExpoWeibull distribution has no closed form
        and is therefore computed by numerical integration of:

        .. math::
            S = -\int_{0}^{\infty} f(x) \ln f(x) dx

        Parameters
        ----------

        alpha : numpy array or scalar
            scale parameter for the ExpoWeibull distribution
        beta : numpy array or scalar
            shape parameter for the ExpoWeibull distribution
        mu : numpy array or scalar
            shape parameter for the ExpoWeibull distribution

        Returns
        -------

        entropy : scalar
            The entropy of the ExpoWeibull distribution

        Examples
        --------
        >>> from surpyval import ExpoWeibull
        >>> ExpoWeibull.entropy(3, 1.5, 0.8)
        1.8227536487527594
        """

        def func(x):
            f = self.df(x, alpha, beta, mu)
            return xlogy(f, f)

        return -integrate.quad(func, 0, np.inf)[0]

    def mpp_x_transform(self, x, gamma=0):
        return np.log(x - gamma)

    def mpp_y_transform(self, y, *params):
        mu = params[-1]
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = np.log(-np.log1p(-y[~mask] ** (1.0 / mu)))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        i = len(params)
        mu = params[i - 1]
        return (1 - np.exp(-np.exp(y))) ** mu

    def unpack_rr(self, params, rr):
        if rr == "y":
            beta = params[0]
            alpha = np.exp(params[1] / -beta)
        elif rr == "x":
            beta = 1.0 / params[0]
            alpha = np.exp(params[1] / (beta * params[0]))
        return alpha, beta, 1.0


ExpoWeibull: ExpoWeibull_ = ExpoWeibull_("ExpoWeibull")
