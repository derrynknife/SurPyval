from scipy import integrate
from scipy.stats import uniform

from surpyval import np
from surpyval import parametric as para
from surpyval.parametric.parametric_fitter import ParametricFitter


class ExpoWeibull_(ParametricFitter):
    def __init__(self, name):
        self.name = name
        self.k = 3
        self.bounds = (
            (0, None),
            (0, None),
            (0, None),
        )
        self.support = (0, np.inf)
        self.plot_x_scale = "log"
        self.y_ticks = [
            0.0001,
            0.0002,
            0.0003,
            0.001,
            0.002,
            0.003,
            0.005,
            0.01,
            0.02,
            0.03,
            0.05,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.95,
            0.99,
            0.999,
            0.9999,
        ]
        self.param_names = ["alpha", "beta", "mu"]
        self.param_map = {"alpha": 0, "beta": 1, "mu": 2}

    def _parameter_initialiser(self, x, c=None, n=None, offset=False):
        log_x = np.log(x)
        log_x[np.isnan(log_x)] = 0
        gumb = para.Gumbel.fit(log_x, c, n, how="MLE")
        if not gumb.res.success:
            gumb = para.Gumbel.fit(log_x, c, n, how="MPP")
        mu, sigma = gumb.params
        alpha, beta = np.exp(mu), 1.0 / sigma
        if np.isinf(alpha) | np.isnan(alpha):
            alpha = np.median(x)
        if np.isinf(beta) | np.isnan(beta):
            beta = 1.0
        if offset:
            gamma = np.min(x) - (np.max(x) - np.min(x)) / 10.0
            return gamma, alpha, beta, 1.0
        else:
            return alpha, beta, 1.0

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
        return 1 - np.power(1 - np.exp(-((x / alpha) ** beta)), mu)

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

        sf : scalar or numpy array
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

        sf : scalar or numpy array
            The value(s) of the reliability function at x.

        Examples
        --------
        >>> import numpy as np
        >>> from surpyval import ExpoWeibull
        >>> x = np.array([1, 2, 3, 4, 5])
        >>> ExpoWeibull.sf(x, 1, 3, 4, 1.2)
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
        return alpha * (-np.log(1 - p ** (1.0 / mu))) ** (1 / beta)

    def mean(self, alpha, beta, mu):
        def func(x):
            return x * self.df(x, alpha, beta, mu)

        top = 2 * self.qf(0.999, alpha, beta, mu)
        return integrate.quadrature(func, 0, top)[0]

    def random(self, size, alpha, beta, mu):
        U = uniform.rvs(size=size)
        return self.qf(U, alpha, beta, mu)

    def mpp_x_transform(self, x, gamma=0):
        return np.log(x - gamma)

    def mpp_y_transform(self, y, *params):
        mu = params[-1]
        mask = (y == 0) | (y == 1)
        out = np.zeros_like(y)
        out[~mask] = np.log(-np.log(1 - y[~mask] ** (1.0 / mu)))
        out[mask] = np.nan
        return out

    def mpp_inv_y_transform(self, y, *params):
        i = len(params)
        mu = params[i - 1]
        return (1 - np.exp(-np.exp(y))) ** mu

    def unpack_rr(self, params, rr):
        # UPDATE ME
        if rr == "y":
            beta = params[0]
            alpha = np.exp(params[1] / -beta)
        elif rr == "x":
            beta = 1.0 / params[0]
            alpha = np.exp(params[1] / (beta * params[0]))
        return alpha, beta, 1.0


ExpoWeibull: ParametricFitter = ExpoWeibull_("ExpoWeibull")
